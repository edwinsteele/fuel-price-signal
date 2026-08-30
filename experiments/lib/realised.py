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

Baseline (arms[0]) caching across paired calls (fps-e2l, parent design fps-3jj):
when ``held_tau`` is ``None`` (the default — baseline's own per-fold τ IS the held
τ in that mode), every call captures its per-fold baseline fit + economics into
``RealisedResult.baseline_cache``. A LATER call against the SAME baseline arm,
feature_columns, station_codes, seed, tank, and fold geometry can pass that cache
back in via ``baseline_cache=`` to skip refitting and re-replaying arms[0] entirely —
correct, not an approximation, because within one frozen batch the baseline arm's
data, columns, and folds are bit-identical across every candidate run against it
(see fps-3jj.4's runner.py, the caller that actually persists this across nights).
A mismatched cache raises ``BaselineCacheMismatch`` (a ``ValueError``) rather than
silently reusing a stale fit.
"""
from __future__ import annotations

import dataclasses
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
    require_tank_stamp,
    tank_params_fields,
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
    # Per-arm training/scoring columns. None → the call-level ``feature_columns``.
    # Lets a candidate arm carry an ADDED column (e.g. baseline 54 vs vel7 55); the
    # arm's ``df`` must contain every column listed here.
    feature_columns: list[str] | None = None
    # In-process source for an added feature with no production PriceHistory yet —
    # passed straight to this arm's ModelStrategy (see ModelStrategy docstring).
    # ``(as_of, station_code, station_price) -> {col: value}``. None → baseline path.
    extra_feature_provider: (
        Callable[[str, int, float], dict[str, float | None]] | None
    ) = None


@dataclass
class _FoldPlan:
    fold: int
    train_index: pd.Index
    val_start: str
    val_end: str


class BaselineCacheMismatch(ValueError):
    """A supplied ``baseline_cache`` doesn't match this call's configuration.

    A ``ValueError`` subclass deliberately: experiments/pipeline/runner.py's
    existing ``except ValueError`` branch already classifies
    ``run_paired_realised_backtest``'s config-shaped errors as
    ``aborted_pipeline`` (retryable, not the candidate's fault) — this reuses
    that classification rather than needing a new one. runner.py itself
    pre-checks the cache before passing it in and falls back to a full refit on
    mismatch (see its ``_load_baseline_cache``), so this is normally only seen
    by a caller that skips that pre-check.
    """


@dataclass
class BaselineCache:
    """Captured per-fold baseline (``arms[0]``) fit + economics (fps-e2l).

    Every ``run_paired_realised_backtest`` call attaches one to
    ``RealisedResult.baseline_cache`` — but ONLY when ``held_tau is None``: in
    that mode the baseline's own per-fold τ IS the held τ, so nothing about the
    other arms' scoring depends on anything not captured here. Pass a prior
    call's cache back in as ``baseline_cache=`` on a LATER call (same baseline
    arm, feature_columns, station_codes, seed, tank, and fold geometry) to skip
    refitting and re-replaying ``arms[0]`` entirely. This is exact, not an
    approximation: within one frozen batch the baseline arm's data, columns,
    and fold plan are bit-identical across every candidate run against it, so
    reusing its fit changes nothing about the result.

    fingerprint: the resolved config that must match on reuse. A mismatch
    raises ``BaselineCacheMismatch`` rather than silently reusing a stale fit.
    per_fold: fold number -> captured record (val_start/val_end, always-buy
    economics, the fitted calibrated pipeline + own τ, own-τ economics, and —
    when the capturing call used ``collect_fills=True`` — the always-buy and
    baseline fill-ledger rows for that fold).
    """

    fingerprint: dict
    per_fold: dict[int, dict] = field(default_factory=dict)


@dataclass
class RealisedResult:
    per_window: pd.DataFrame      # one row per (fold, arm): own/held τ, cpl, saving
    aggregate: pd.DataFrame       # one row per arm: pooled cpl + saving at own & held τ
    deltas: pd.DataFrame          # candidate − baseline, per window + aggregate
    meta: dict = field(default_factory=dict)
    # Per-fill ledger (fold, arm, own_tau, date, station_code, price, litres,
    # spend_cents, emergency) at each arm's OWN τ — populated only when
    # run_paired_realised_backtest(..., collect_fills=True). Empty otherwise.
    # The reusable realised analog of rowpreds.parquet: a caller groups these by
    # regime / season / LGA (the #259 per-regime gate). None when not collected.
    fills: pd.DataFrame | None = None
    # This call's baseline (arms[0]) fit + economics, reusable by a later call —
    # see BaselineCache. None when held_tau was not None (caching unsupported
    # in that mode; see BaselineCache's docstring).
    baseline_cache: BaselineCache | None = None


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


def _fill_row(fold: int, arm: str, own_tau: float, fr: Any) -> dict:
    """One ledger row from a FillRecord (single schema for both collection sites)."""
    return {
        "fold": fold, "arm": arm, "own_tau": own_tau,
        "date": fr.date, "station_code": fr.station_code,
        "price": fr.price, "litres": fr.litres,
        "spend_cents": fr.spend_cents, "emergency": fr.emergency,
    }


def _arm_cols(arm: ArmSpec, shared: list[str]) -> list[str]:
    """Resolve an arm's training/scoring columns. Explicit None (not falsy) check so
    an intentional empty list isn't silently swapped for the shared list — it's a
    misconfiguration the caller should hear about, not a fallback trigger."""
    cols = arm.feature_columns if arm.feature_columns is not None else shared
    if not cols:
        raise ValueError(f"arm {arm.name!r} resolved to empty feature_columns.")
    return cols


def _saving_pct(always_cpl: float, model_cpl: float) -> float:
    if not (always_cpl > 0) or math.isnan(model_cpl):
        return float("nan")
    return (always_cpl - model_cpl) / always_cpl * 100


def _baseline_cache_fingerprint(
    baseline: ArmSpec,
    feature_columns: list[str],
    station_codes: list[int],
    seed: int,
    outer_fold_params: dict,
    inner_fold_params: dict,
    collect_fills: bool,
    tank: TankParams,
) -> dict:
    """The config a baseline_cache must match to be safely reusable.

    Everything that feeds arms[0]'s fit, its τ selection, its economics, or the
    fold grid itself — including ``tank``: tank_size_litres/daily_consumption_
    litres/evaluation_interval_days/floor_fraction feed directly into
    aggregate_backtest's spend/litres/cpl for BOTH the always-buy replay and
    the model replay, and evaluation_interval_days also drives the PIT
    lga_days_since eval_dates grid, so a cache captured under one TankParams
    is not valid economics for another (fps-e2l review finding). Doesn't
    include the baseline arm's actual DataFrame contents (too expensive to
    hash on every call) — callers are trusted to only pass a cache back in for
    the SAME frozen batch it was captured against (runner.py does this by
    keying the cache file to batch_dir); the per-fold val_start/val_end
    cross-check in run_paired_realised_backtest catches a date-grid drift even
    so.

    Deliberately excludes ``fold_subset``: fold NUMBERING and each fold's
    val_start/val_end come from enumerating walk_forward_folds(df,
    **outer_fold_params) BEFORE fold_subset filters which are kept (see
    _plan_folds), so a fold's identity and economics don't depend on which
    subset of folds any particular call asked for — only outer_fold_params
    does. Including fold_subset here would defeat the whole point of
    BaselineCache.per_fold's per-fold ``.get()`` lookup: a call with
    fold_subset={1} and a later call with fold_subset={1, 2} share fold 1's
    exact same baseline fit, and this fingerprint must say so, not force a
    full mismatch/refit over a difference that's irrelevant to the baseline's
    actual computation. Partial coverage (this run needs a fold the cache
    doesn't have) degrades per-fold to a fresh compute for just that fold, not
    a fingerprint-level rejection — see the cache_hit_folds bookkeeping in
    run_paired_realised_backtest.

    outer_fold_params/inner_fold_params are copied (``dict(...)``), not stored
    by reference: they're the caller's own dict objects, and a cache's
    fingerprint must be a frozen snapshot — if the caller mutated the SAME
    dict in place before a later call (e.g. reusing one dict variable across
    two run_paired_realised_backtest calls), an aliased reference would drift
    right along with it, so a fingerprint comparison against the (also-just-
    mutated) current call's params would wrongly compare equal and reuse a
    cache captured under different fold geometry.
    """
    return {
        "baseline_arm": baseline.name,
        "baseline_feature_columns": list(_arm_cols(baseline, feature_columns)),
        "station_codes": sorted(station_codes),
        "seed": seed,
        "outer_fold_params": dict(outer_fold_params),
        "inner_fold_params": dict(inner_fold_params),
        "collect_fills": collect_fills,
        "tank": dataclasses.asdict(tank),
    }


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
    collect_fills: bool = False,
    baseline_cache: BaselineCache | None = None,
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
    collect_fills: when True, attach a per-fill ledger DataFrame to
        ``result.fills`` (each arm's fills at its OWN τ, tagged fold/arm). The
        #259 gate-1 use groups it by cycle regime. Off by default (the extra
        per-fill bookkeeping is wasted for a pure CPL run).
    baseline_cache: a prior call's ``result.baseline_cache`` (see BaselineCache)
        to skip refitting/re-replaying arms[0]. Raises BaselineCacheMismatch
        (a ValueError) if it doesn't match this call's config, or if held_tau
        is not None (caching is only valid when baseline's own τ IS the held
        τ). None (default) — every fold is computed fresh, as before fps-e2l.
    """
    if not arms:
        raise ValueError("run_paired_realised_backtest() needs at least one arm.")
    names = [a.name for a in arms]
    if len(names) != len(set(names)):
        # name keys histories AND groups every result frame — a collision would
        # silently overwrite an arm's history and corrupt the deltas.
        raise ValueError(f"ArmSpec names must be unique; got {names}.")
    if collect_fills and "always_buy" in names:
        # "always_buy" is reserved for the baseline ledger in the fills frame.
        raise ValueError("'always_buy' is a reserved arm name when collect_fills=True.")
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

    fingerprint = _baseline_cache_fingerprint(
        arms[0], feature_columns, station_codes, seed,
        outer_fold_params, inner_fold_params, collect_fills, tank,
    )
    if baseline_cache is not None:
        if held_tau is not None:
            raise BaselineCacheMismatch(
                "baseline_cache is only valid when held_tau is None — baseline's own "
                "τ IS the held τ in that mode. This call passed an explicit held_tau "
                "override, so a cached baseline fit can't be reused without recomputing "
                "its held-τ economics."
            )
        if baseline_cache.fingerprint != fingerprint:
            raise BaselineCacheMismatch(
                f"baseline_cache fingerprint mismatch — cache was captured under "
                f"{baseline_cache.fingerprint!r}, this call resolved {fingerprint!r}. "
                "Refusing to reuse a cache captured under different config."
            )

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

    # A cache entry covers a fold; skip building the baseline's PriceHistory
    # entirely when every fold this run needs is already covered (the rest of
    # the ~12min/night the parent design describes — fps-e2l).
    baseline_fully_cached = baseline_cache is not None and all(
        p.fold in baseline_cache.per_fold for p in plans
    )

    # One PriceHistory per arm (price path is shared; only the detector differs).
    conn = _db.open_db(db_path) if db_path is not None else _db.open_db()
    try:
        histories = {
            a.name: load_history(
                conn, station_codes, eval_dates=eval_dates, detector_factory=a.detector_factory
            )
            for a in arms
            if not (a is arms[0] and baseline_fully_cached)
        }
    finally:
        conn.close()
    if verbose:
        print(
            f"[realised] loaded {len(histories)}/{len(arms)} arm histories "
            f"({'baseline cached' if baseline_fully_cached else 'no baseline cache hit'})  "
            f"({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )

    rows: list[dict] = []
    fill_rows: list[dict] = []
    baseline_per_fold: dict[int, dict] = {}
    cache_hit_folds: list[int] = []
    for p in plans:
        ft0 = time.perf_counter()
        cached_fold = baseline_cache.per_fold.get(p.fold) if baseline_cache is not None else None
        if cached_fold is not None and (
            cached_fold["val_start"] != p.val_start or cached_fold["val_end"] != p.val_end
        ):
            # Fingerprint matched but the actual fold grid didn't — a defensive
            # belt-and-braces check (see _baseline_cache_fingerprint's docstring
            # on why the arm's DataFrame contents aren't hashed there).
            raise BaselineCacheMismatch(
                f"baseline_cache fold {p.fold} val window "
                f"{cached_fold['val_start']}..{cached_fold['val_end']} != this run's "
                f"{p.val_start}..{p.val_end} despite a matching fingerprint — refusing "
                "to reuse (the frozen batch's data may have changed underneath it)."
            )

        if cached_fold is not None:
            always_cpl = cached_fold["always_cpl"]
            always_spend = cached_fold["always_spend"]
            always_litres = cached_fold["always_litres"]
            if collect_fills:
                fill_rows.extend(cached_fold["always_fills"])
            cache_hit_folds.append(p.fold)
        else:
            # Always-buy is detector-independent — compute once per window. Keep its
            # spend/litres so the aggregate can pool a single always-buy CPL (and thus
            # an honest pooled saving%) rather than averaging per-window CPLs.
            always = aggregate_backtest(
                histories[arms[0].name], AlwaysBuyStrategy(), station_codes, p.val_start, p.val_end, tank,
                collect_fills=collect_fills,
            )
            always_cpl = always["cpl"]
            always_spend = always["total_spend_cents"]
            always_litres = always["total_litres"]
            always_fill_rows: list[dict] = []
            if collect_fills:
                # The always-buy fill ledger is the regime-matched denominator for a
                # stratified saving% — tag it arm="always_buy" (reserved name).
                for fr in always["fills"]:
                    row = _fill_row(p.fold, "always_buy", float("nan"), fr)
                    fill_rows.append(row)
                    always_fill_rows.append(row)

        # Train each arm and record its own τ first (baseline τ defines the held τ).
        # Each arm trains/scores on its own feature_columns (None → the shared list),
        # so a candidate arm can carry an ADDED column the baseline lacks. The
        # baseline is skipped here (fit already captured in cached_fold) when cached.
        fitted: dict[str, tuple[Any, float]] = {}
        for a in arms:
            if a is arms[0] and cached_fold is not None:
                fitted[a.name] = (cached_fold["cal_pipe"], cached_fold["own_tau"])
                continue
            cols = _arm_cols(a, feature_columns)
            train_df = a.df.loc[p.train_index]
            fitted[a.name] = _train_calibrate_select_tau(
                train_df, cols, seed, inner_fold_params
            )
        held = held_tau if held_tau is not None else fitted[arms[0].name][1]

        for a in arms:
            if a is arms[0] and cached_fold is not None:
                # own τ IS held (held_tau is None whenever a cache is in play —
                # enforced above), so cpl_own doubles as cpl_held, same as the
                # math.isclose(own_tau, held) shortcut below for the fresh path.
                rows.append({
                    "fold": p.fold, "arm": a.name,
                    "val_start": p.val_start, "val_end": p.val_end,
                    "always_cpl": always_cpl,
                    "always_spend": cached_fold["always_spend"], "always_litres": cached_fold["always_litres"],
                    "own_tau": cached_fold["own_tau"], "held_tau": held,
                    "cpl_own": cached_fold["cpl_own"], "saving_own_pct": _saving_pct(always_cpl, cached_fold["cpl_own"]),
                    "cpl_held": cached_fold["cpl_own"], "saving_held_pct": _saving_pct(always_cpl, cached_fold["cpl_own"]),
                    "spend_own": cached_fold["spend_own"], "litres_own": cached_fold["litres_own"],
                    "spend_held": cached_fold["spend_own"], "litres_held": cached_fold["litres_own"],
                })
                if collect_fills:
                    fill_rows.extend(cached_fold["own_fills"])
                baseline_per_fold[p.fold] = cached_fold
                continue

            cal_pipe, own_tau = fitted[a.name]
            cols = _arm_cols(a, feature_columns)
            cpl_own = aggregate_backtest(
                histories[a.name],
                ModelStrategy(
                    pipeline=cal_pipe, feature_columns=cols, threshold=own_tau,
                    extra_feature_provider=a.extra_feature_provider,
                ),
                station_codes, p.val_start, p.val_end, tank,
                collect_fills=collect_fills,
            )
            own_fill_rows: list[dict] = []
            if collect_fills:
                for fr in cpl_own["fills"]:
                    row = _fill_row(p.fold, a.name, own_tau, fr)
                    fill_rows.append(row)
                    own_fill_rows.append(row)
            # The held-τ replay is identical when own τ == held (always true for the
            # baseline in the held_tau=None path) — reuse it, don't pay it twice.
            if math.isclose(own_tau, held):
                cpl_held = cpl_own
            else:
                cpl_held = aggregate_backtest(
                    histories[a.name],
                    ModelStrategy(
                        pipeline=cal_pipe, feature_columns=cols, threshold=held,
                        extra_feature_provider=a.extra_feature_provider,
                    ),
                    station_codes, p.val_start, p.val_end, tank,
                )
            rows.append({
                "fold": p.fold, "arm": a.name,
                "val_start": p.val_start, "val_end": p.val_end,
                "always_cpl": always_cpl,
                "always_spend": always_spend, "always_litres": always_litres,
                "own_tau": own_tau, "held_tau": held,
                "cpl_own": cpl_own["cpl"], "saving_own_pct": _saving_pct(always_cpl, cpl_own["cpl"]),
                "cpl_held": cpl_held["cpl"], "saving_held_pct": _saving_pct(always_cpl, cpl_held["cpl"]),
                "spend_own": cpl_own["total_spend_cents"], "litres_own": cpl_own["total_litres"],
                "spend_held": cpl_held["total_spend_cents"], "litres_held": cpl_held["total_litres"],
            })

            if a is arms[0] and held_tau is None:
                # Capture this fold's fresh baseline computation for export — a
                # LATER call can reuse it even though THIS call didn't (fps-e2l).
                baseline_per_fold[p.fold] = {
                    "val_start": p.val_start, "val_end": p.val_end,
                    "always_cpl": always_cpl,
                    "always_spend": always_spend, "always_litres": always_litres,
                    "cal_pipe": cal_pipe, "own_tau": own_tau,
                    "cpl_own": cpl_own["cpl"],
                    "spend_own": cpl_own["total_spend_cents"], "litres_own": cpl_own["total_litres"],
                    "always_fills": always_fill_rows,
                    "own_fills": own_fill_rows,
                }
        if verbose:
            hit = " [baseline cache hit]" if cached_fold is not None else ""
            print(
                f"[realised] fold {p.fold:>2} {p.val_start}→{p.val_end} "
                f"always={always_cpl:.2f}{hit}  ({time.perf_counter()-ft0:.1f}s)",
                flush=True,
            )

    per_window = pd.DataFrame(rows)

    # Aggregate: pool spend + litres across windows for an honest CPL per arm, and
    # a pooled always-buy CPL so saving% is computed on pooled totals (not a mean of
    # per-window saving%). always_cpl is detector-independent, so it's shared.
    def _pooled(spend_col: str, litres_col: str, frame: pd.DataFrame) -> float:
        litres = frame[litres_col].sum()
        return frame[spend_col].sum() / litres if litres > 0 else float("nan")

    pooled_always = _pooled("always_spend", "always_litres", per_window.drop_duplicates("fold"))
    agg_rows: list[dict] = []
    for a in arms:
        sub = per_window[per_window["arm"] == a.name]
        cpl_own = _pooled("spend_own", "litres_own", sub)
        cpl_held = _pooled("spend_held", "litres_held", sub)
        agg_rows.append({
            "arm": a.name,
            "n_windows": len(sub),
            "own_tau_median": float(sub["own_tau"].median()),
            # held_tau can vary per window when held_tau=None (it's the baseline's
            # per-fold own τ), so summarise rather than expose one window's value.
            "held_tau_median": float(sub["held_tau"].median()),
            "always_cpl": pooled_always,
            "cpl_own": cpl_own, "saving_own_pct": _saving_pct(pooled_always, cpl_own),
            "cpl_held": cpl_held, "saving_held_pct": _saving_pct(pooled_always, cpl_held),
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
        # Record the RESOLVED columns each arm actually trained/scored on (None →
        # the shared list), not the raw override — a reader reconstructing an arm
        # shouldn't have to re-apply the resolution rule, and None would misstate
        # what the baseline arm used.
        "arm_feature_columns": {a.name: list(_arm_cols(a, feature_columns)) for a in arms},
        "station_codes": station_codes,
        "seed": seed,
        # The cadence every cpl_own/cpl_held value in per_window/aggregate/deltas was
        # produced at (fps-15c) — `tank` is already resolved (never None) above, so
        # this can never actually raise; the call still goes through the shared guard
        # so every downstream reader can rely on the SAME stamping path, not a
        # bespoke format_tank_params(tank) call that could drift from it.
        "tank_params": require_tank_stamp(tank, what="run_paired_realised_backtest"),
        # Exact numeric fields behind the stamp above (fps-o0h) — persisted alongside it so a
        # downstream reader (experiments.lib.flips's cascade_window_days/regret_horizon_days)
        # can derive tank-scale windows without round-tripping the display-rounded stamp,
        # which can land on a rounding tie the exact fields never can.
        "tank_params_fields": tank_params_fields(tank),
        "held_tau": held_tau,
        "outer_fold_params": outer_fold_params,
        "inner_fold_params": inner_fold_params,
        "fold_subset": list(fold_subset) if fold_subset is not None else None,
        "n_windows": len(plans),
        "calibration": "isotonic",
        "collect_fills": collect_fills,
        "total_wall_seconds": time.perf_counter() - t0,
        # Cache diagnostics (fps-e2l) — lets a caller (or a results.json reader)
        # confirm the caching path actually fired, rather than inferring it from
        # wall-clock time alone.
        "baseline_cache_used": baseline_cache is not None,
        "baseline_cache_hit_folds": cache_hit_folds,
    }
    fills = pd.DataFrame(fill_rows) if collect_fills else None
    baseline_cache_out = (
        BaselineCache(fingerprint=fingerprint, per_fold=baseline_per_fold)
        if held_tau is None
        else None
    )
    if verbose:
        print(f"[realised] done  ({meta['total_wall_seconds']:.1f}s)", flush=True)
    return RealisedResult(
        per_window=per_window, aggregate=aggregate, deltas=deltas, meta=meta, fills=fills,
        baseline_cache=baseline_cache_out,
    )
