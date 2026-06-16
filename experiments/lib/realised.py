"""Paired realised-spend backtest — the objective-aligned arbiter (issue #255).

WFCV per-row log-loss is a *screen* for decision-timing / trough / cycle-phase
features; their value lands in realised buyer outcome, which a calibration average
washes out (see docs/CONVENTIONS.md § Choosing the gate metric, memory
``feedback-wfcv-logloss-screen-not-verdict``). This module makes the realised
backtest a one-call, in-process, walk-forward capability — the same rigor as
``experiments/lib/folds.py`` gives the log-loss screen — so the verdict is no
longer a clunky branch + 4-CLI single-window dance (the #254 settling exposed
that gap).

What it does, per arm (baseline vs candidate):
  - walk-forward over the SAME folds as the WFCV screen (``walk_forward_folds``);
  - per fold, train a raw LightGBM + isotonic calibrator + pick τ via OOF on the
    fold's train (mirrors the production #236 OOF τ selection; isotonic-only —
    the AC3 lock — to hold a paired run near WFCV wall-clock rather than 2×);
  - replay the fold's val window through the real ``aggregate_backtest`` economics
    (never forks the buy/wait simulation);
  - score each arm at its OWN τ and at a HELD common τ (clean attribution — the
    #254 lesson was that the τ move dominated the apparent feature win);
  - pool spend + litres across windows for an honest aggregate CPL, plus per-window
    rows so a caller can group by regime (the #259 gate-1 use).

In-process candidate injection (no production branch): an ``ArmSpec`` carries its
own feature frame (canonical cycle columns hold THAT arm's values) and a
``detector_factory`` (the live cycle-feature source for the backtest). Both must
be consistent — the model is trained on the arm's frame and scored live through
its detector. The baseline arm uses the production ``CycleDetector`` and the
unmodified feature frame.

Arms must share the same DataFrame index (candidate = ``baseline.copy()`` with the
relevant cycle columns overwritten) so a fold's train rows select identically
across arms.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fuel_signal import db as _db
from fuel_signal import evaluate as _ev
from fuel_signal.backtest import (
    AlwaysBuyStrategy,
    ModelStrategy,
    TankParams,
    _evaluation_dates,
    load_history,
)
from fuel_signal.backtest_phase2 import aggregate_backtest
from fuel_signal.calibrate import _CalibratedPipeline, pool_oof_predictions
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.cycle import CycleDetector
from fuel_signal.score_phase2 import pick_tau, threshold_sweep
from fuel_signal.train_lgbm import build_pipeline


@dataclass
class ArmSpec:
    """One backtest arm.

    name: arm label (e.g. "baseline", "regime").
    df: feature frame whose canonical cycle columns hold THIS arm's values; the
        model is trained on it. Must share its index with the other arms.
    detector_factory: builds the live cycle detector from avg_series for the
        realised replay (default = production CycleDetector).
    """

    name: str
    df: pd.DataFrame
    detector_factory: Callable[[list[tuple[str, float]]], CycleDetector] = CycleDetector


@dataclass
class _FoldPlan:
    fold: int
    train_index: pd.Index
    val_start: str
    val_end: str


@dataclass
class RealisedResult:
    per_window: pd.DataFrame      # one row per (fold, arm): own/held τ, cpl, saving
    aggregate: pd.DataFrame       # one row per arm: pooled cpl + saving at own & held τ
    deltas: pd.DataFrame          # candidate − baseline, per window + aggregate
    meta: dict = field(default_factory=dict)


def _train_calibrate_select_tau(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    inner_fold_params: dict,
) -> tuple[Any, float]:
    """Fit a 100%-trained raw LGBM + isotonic calibrator; pick τ via OOF sweep.

    Mirrors the production calibrate / score_phase2 OOF path (isotonic only):
      - raw base fit on the full fold-train (no 80% handicap);
      - isotonic calibrator fit on walk-forward OOF predictions over fold-train;
      - τ = argmax(expected_cents_per_row) on calibrated OOF, isotonic adjustment 0.0.
    Returns (calibrated_pipeline, tau).
    """
    y = train_df["label"].to_numpy(dtype=int)

    raw_pipe = build_pipeline(seed)
    raw_pipe.fit(train_df[feature_columns], y)

    p_oof, y_oof = pool_oof_predictions(
        build_pipeline(seed), train_df, feature_columns, inner_fold_params
    )
    if p_oof.size == 0:
        raise ValueError(
            "realised: no OOF folds over fold-train — pass a smaller inner_fold_params "
            "(train_min_days/val_days/step_days) for this train window."
        )
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_oof, y_oof)
    cal_pipe = _CalibratedPipeline(raw_pipe, iso, "isotonic", feature_columns)

    p_oof_iso = np.clip(iso.predict(p_oof), 0.0, 1.0)
    sweep = threshold_sweep(y_oof, p_oof_iso)
    tau = pick_tau(sweep, calibration_method="isotonic", tau_adjustment=0.0)
    return cal_pipe, tau


def _plan_folds(
    df: pd.DataFrame,
    outer_fold_params: dict,
    fold_subset: Iterable[int] | None,
) -> list[_FoldPlan]:
    plans: list[_FoldPlan] = []
    keep = set(fold_subset) if fold_subset is not None else None
    for i, (train_df, val_df) in enumerate(
        _ev.walk_forward_folds(df, **outer_fold_params), start=1
    ):
        if val_df.empty:
            continue
        if keep is not None and i not in keep:
            continue
        vd = pd.to_datetime(val_df["price_date"])
        plans.append(
            _FoldPlan(
                fold=i,
                train_index=train_df.index,
                val_start=vd.min().strftime("%Y-%m-%d"),
                val_end=vd.max().strftime("%Y-%m-%d"),
            )
        )
    return plans


def _saving_pct(always_cpl: float, model_cpl: float) -> float:
    if not (always_cpl > 0) or math.isnan(model_cpl):
        return float("nan")
    return (always_cpl - model_cpl) / always_cpl * 100


def run_paired_realised_backtest(
    arms: list[ArmSpec],
    feature_columns: list[str],
    *,
    station_codes: list[int] | None = None,
    seed: int = 42,
    held_tau: float | None = None,
    outer_fold_params: dict | None = None,
    inner_fold_params: dict | None = None,
    fold_subset: Iterable[int] | None = None,
    db_path: Any = None,
    tank: TankParams | None = None,
    verbose: bool = True,
) -> RealisedResult:
    """Run the paired walk-forward realised-spend backtest.

    arms: 1+ ArmSpec (arms[0] is the baseline; deltas are arm − arms[0]). A single
        arm is the degenerate gate-1 use (per-regime realised regret for one model).
    feature_columns: canonical training columns, shared across arms.
    held_tau: common τ for clean attribution. None → each non-baseline arm is also
        scored at the baseline arm's OWN per-fold τ (isolates the feature from the
        operating-point move).
    outer_fold_params / inner_fold_params: walk_forward_folds kwargs for the outer
        windows and the inner OOF (calibration + τ). Defaults = production folds.
    fold_subset: 1-indexed fold numbers to run (iteration / smoke speed-up).
    """
    if not arms:
        raise ValueError("run_paired_realised_backtest() needs at least one arm.")
    names = [a.name for a in arms]
    if len(names) != len(set(names)):
        # name keys histories AND groups every result frame — a collision would
        # silently overwrite an arm's history and corrupt the deltas.
        raise ValueError(f"ArmSpec names must be unique; got {names}.")
    ref_index = arms[0].df.index
    for a in arms[1:]:
        if not a.df.index.equals(ref_index):
            raise ValueError(
                f"arm {a.name!r} index differs from baseline {arms[0].name!r}; arms must "
                "share an index (candidate = baseline.copy() with cycle cols overwritten)."
            )

    station_codes = station_codes or list(PREFERRED_STATIONS)
    outer_fold_params = outer_fold_params or {}
    inner_fold_params = inner_fold_params or {}
    tank = tank or TankParams()

    t0 = time.perf_counter()
    plans = _plan_folds(arms[0].df, outer_fold_params, fold_subset)
    if not plans:
        raise ValueError("run_paired_realised_backtest(): no folds planned.")

    # Union of eval dates across all windows for the PIT lga_days_since features.
    eval_dates: list[str] = sorted(
        {
            d
            for p in plans
            for d in _evaluation_dates(p.val_start, p.val_end, tank.evaluation_interval_days)
        }
    )

    # One PriceHistory per arm (price path is shared; only the detector differs).
    conn = _db.open_db(db_path) if db_path is not None else _db.open_db()
    try:
        histories = {
            a.name: load_history(
                conn, station_codes, eval_dates=eval_dates, detector_factory=a.detector_factory
            )
            for a in arms
        }
    finally:
        conn.close()
    if verbose:
        print(f"[realised] loaded {len(arms)} arm histories  ({time.perf_counter()-t0:.1f}s)", flush=True)

    rows: list[dict] = []
    for p in plans:
        ft0 = time.perf_counter()
        # Always-buy CPL is detector-independent — compute once per window.
        always_cpl = aggregate_backtest(
            histories[arms[0].name], AlwaysBuyStrategy(), station_codes, p.val_start, p.val_end, tank
        )["cpl"]

        # Train each arm and record its own τ first (baseline τ defines the held τ).
        fitted: dict[str, tuple[Any, float]] = {}
        for a in arms:
            train_df = a.df.loc[p.train_index]
            fitted[a.name] = _train_calibrate_select_tau(
                train_df, feature_columns, seed, inner_fold_params
            )
        held = held_tau if held_tau is not None else fitted[arms[0].name][1]

        for a in arms:
            cal_pipe, own_tau = fitted[a.name]
            cpl_own = aggregate_backtest(
                histories[a.name],
                ModelStrategy(pipeline=cal_pipe, feature_columns=feature_columns, threshold=own_tau),
                station_codes, p.val_start, p.val_end, tank,
            )
            cpl_held = aggregate_backtest(
                histories[a.name],
                ModelStrategy(pipeline=cal_pipe, feature_columns=feature_columns, threshold=held),
                station_codes, p.val_start, p.val_end, tank,
            )
            rows.append({
                "fold": p.fold, "arm": a.name,
                "val_start": p.val_start, "val_end": p.val_end,
                "always_cpl": always_cpl,
                "own_tau": own_tau, "held_tau": held,
                "cpl_own": cpl_own["cpl"], "saving_own_pct": _saving_pct(always_cpl, cpl_own["cpl"]),
                "cpl_held": cpl_held["cpl"], "saving_held_pct": _saving_pct(always_cpl, cpl_held["cpl"]),
                "spend_own": cpl_own["total_spend_cents"], "litres_own": cpl_own["total_litres"],
                "spend_held": cpl_held["total_spend_cents"], "litres_held": cpl_held["total_litres"],
            })
        if verbose:
            print(
                f"[realised] fold {p.fold:>2} {p.val_start}→{p.val_end} "
                f"always={always_cpl:.2f}  ({time.perf_counter()-ft0:.1f}s)",
                flush=True,
            )

    per_window = pd.DataFrame(rows)

    # Aggregate: pool spend + litres across windows for an honest CPL per arm.
    agg_rows: list[dict] = []
    for a in arms:
        sub = per_window[per_window["arm"] == a.name]
        cpl_own = sub["spend_own"].sum() / sub["litres_own"].sum() if sub["litres_own"].sum() > 0 else float("nan")
        cpl_held = sub["spend_held"].sum() / sub["litres_held"].sum() if sub["litres_held"].sum() > 0 else float("nan")
        agg_rows.append({
            "arm": a.name,
            "n_windows": len(sub),
            "own_tau_median": float(sub["own_tau"].median()),
            # held_tau can vary per window when held_tau=None (it's the baseline's
            # per-fold own τ), so summarise rather than expose one window's value.
            "held_tau_median": float(sub["held_tau"].median()),
            "cpl_own": cpl_own, "cpl_held": cpl_held,
        })
    aggregate = pd.DataFrame(agg_rows)

    # Deltas vs baseline (arm − arms[0]), per window and aggregate, at HELD τ
    # (clean attribution) and at each arm's OWN τ.
    base = arms[0].name
    delta_rows: list[dict] = []
    bw = per_window[per_window["arm"] == base].set_index("fold")
    for a in arms[1:]:
        aw = per_window[per_window["arm"] == a.name].set_index("fold")
        for fold in aw.index:
            delta_rows.append({
                "fold": fold, "arm": a.name,
                "delta_cpl_held": aw.loc[fold, "cpl_held"] - bw.loc[fold, "cpl_held"],
                "delta_cpl_own": aw.loc[fold, "cpl_own"] - bw.loc[fold, "cpl_own"],
                "tau_diverges": not math.isclose(aw.loc[fold, "own_tau"], bw.loc[fold, "own_tau"]),
            })
    deltas = pd.DataFrame(delta_rows)

    meta = {
        "issue": 255,
        "arms": [a.name for a in arms],
        "baseline_arm": base,
        "feature_columns": list(feature_columns),
        "station_codes": station_codes,
        "seed": seed,
        "held_tau": held_tau,
        "outer_fold_params": outer_fold_params,
        "inner_fold_params": inner_fold_params,
        "fold_subset": list(fold_subset) if fold_subset is not None else None,
        "n_windows": len(plans),
        "calibration": "isotonic",
        "total_wall_seconds": time.perf_counter() - t0,
    }
    if verbose:
        print(f"[realised] done  ({meta['total_wall_seconds']:.1f}s)", flush=True)
    return RealisedResult(per_window=per_window, aggregate=aggregate, deltas=deltas, meta=meta)
