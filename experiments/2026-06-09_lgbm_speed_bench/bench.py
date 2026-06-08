"""LightGBM fit-speed benchmark — issue #220.

Compares three fit configurations against the 54-feat baseline to decide
whether either becomes a convention for paired_wfcv.py-style scripts.

  C0   LGBMClassifier (current baseline)
  C1   LGBMClassifier + force_col_wise=True
  C2   lgb.train with Dataset built once per fold, reused across seeds

14 folds × 5 seeds × 3 configs = 210 LightGBM fits.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-09_lgbm_speed_bench/bench.py \\
    2>&1 | tee experiments/2026-06-09_lgbm_speed_bench/run.log
"""
from __future__ import annotations

import json
import pathlib
import time

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    load_features,
)

OUT = pathlib.Path(__file__).parent
SEEDS = (42, 43, 44, 45, 46)

# 54-feat RAC_full baseline (locked in PR #225 / commit e925bd6)
FEATURE_COLS = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS

# LightGBM params that match LGBMClassifier(random_state=seed, verbose=-1,
# subsample=0.8, subsample_freq=1) — confirmed by inspecting booster_.params.
_LGBM_PARAMS_BASE: dict = {
    "objective": "binary",
    "verbosity": -1,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "bagging_fraction": 0.8,  # subsample
    "bagging_freq": 1,  # subsample_freq
    "min_child_samples": 20,
    "feature_fraction": 1.0,  # colsample_bytree
    "lambda_l1": 0.0,  # reg_alpha
    "lambda_l2": 0.0,  # reg_lambda
}
_NUM_BOOST_ROUND = 100  # n_estimators default

_EQUIV_TOL = 1e-6  # max acceptable |ll(Cx) - ll(C0)| on seed=42


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------

def fit_c0(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    **_,
) -> tuple[float, float]:
    """C0: standard LGBMClassifier."""
    t0 = time.perf_counter()
    m = LGBMClassifier(random_state=seed, verbose=-1, subsample=0.8, subsample_freq=1)
    m.fit(train_df[FEATURE_COLS], train_df["label"].to_numpy(dtype=int))
    p = m.predict_proba(val_df[FEATURE_COLS])[:, 1]
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, time.perf_counter() - t0


def fit_c1(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    **_,
) -> tuple[float, float]:
    """C1: force_col_wise=True."""
    t0 = time.perf_counter()
    m = LGBMClassifier(
        random_state=seed, verbose=-1, subsample=0.8, subsample_freq=1,
        force_col_wise=True,
    )
    m.fit(train_df[FEATURE_COLS], train_df["label"].to_numpy(dtype=int))
    p = m.predict_proba(val_df[FEATURE_COLS])[:, 1]
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, time.perf_counter() - t0


def fit_c2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    train_ds: lgb.Dataset,
    val_ds: lgb.Dataset,
) -> tuple[float, float]:
    """C2: lgb.train with Dataset built once per fold, reused across seeds."""
    t0 = time.perf_counter()
    params = {**_LGBM_PARAMS_BASE, "seed": seed}
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=_NUM_BOOST_ROUND,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[lgb.log_evaluation(-1)],
    )
    p = booster.predict(val_df[FEATURE_COLS].to_numpy())
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading features ...", flush=True)
    t0 = time.perf_counter()
    df = load_features()
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}", flush=True)

    assert len(FEATURE_COLS) == 54, f"expected 54 features, got {len(FEATURE_COLS)}"
    print(f"Feature set: {len(FEATURE_COLS)} cols (54-feat RAC_full baseline)", flush=True)
    print(f"Seeds: {SEEDS}  configs: C0, C1, C2", flush=True)

    folds = list(_ev.walk_forward_folds(df, train_min_days=1825,
                                        val_days=90, step_days=90))
    print(f"Walk-forward folds: {len(folds)}\n", flush=True)

    hdr = (f"{'fold':>4}  {'val_start':>10}  {'val_end':>10}  {'val_rows':>8}  "
           f"{'config':<6}  {'seed':>4}  {'ll':>8}  {'fit_s':>6}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    rows: list[dict] = []
    # ll_c0_seed42[fold] used for equivalence check
    ll_c0_seed42: dict[int, float] = {}

    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        vd = pd.to_datetime(val_df["price_date"])
        val_start, val_end = vd.min(), vd.max()

        # Build C2 datasets once per fold — reused across all seeds.
        # free_raw_data=False keeps the internal bins cached between lgb.train calls.
        train_ds = lgb.Dataset(
            train_df[FEATURE_COLS], label=train_df["label"].to_numpy(dtype=int),
            free_raw_data=False,
        )
        val_ds = lgb.Dataset(
            val_df[FEATURE_COLS], label=val_df["label"].to_numpy(dtype=int),
            reference=train_ds, free_raw_data=False,
        )
        # Construct train_ds once explicitly so the cost is visible at fold level,
        # not buried in the first seed's fit_s.
        train_ds.construct()
        val_ds.construct()

        configs = [("C0", fit_c0), ("C1", fit_c1), ("C2", fit_c2)]
        for cfg_name, fit_fn in configs:
            for seed in SEEDS:
                ll, fit_s = fit_fn(
                    train_df, val_df, seed, train_ds=train_ds, val_ds=val_ds
                )
                if cfg_name == "C0" and seed == 42:
                    ll_c0_seed42[i] = ll
                rows.append({
                    "fold": i,
                    "val_start": val_start.strftime("%Y-%m-%d"),
                    "val_end": val_end.strftime("%Y-%m-%d"),
                    "val_rows": len(val_df),
                    "config": cfg_name,
                    "seed": seed,
                    "ll": ll,
                    "fit_s": fit_s,
                })
                print(
                    f"{i:>4}  {val_start.strftime('%Y-%m-%d'):>10}  "
                    f"{val_end.strftime('%Y-%m-%d'):>10}  {len(val_df):>8,}  "
                    f"{cfg_name:<6}  {seed:>4}  {ll:>8.5f}  {fit_s:>5.2f}s",
                    flush=True,
                )

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "runs.csv", index=False)
    print(f"\nPer-(fold,config,seed) results: {OUT / 'runs.csv'}", flush=True)

    # -----------------------------------------------------------------------
    # Equivalence check (seed=42 only, per-fold)
    # -----------------------------------------------------------------------
    print("\n=== Equivalence check (seed=42, tolerance=1e-6) ===", flush=True)
    equiv_rows: list[dict] = []
    for cfg_name in ("C1", "C2"):
        cfg_s42 = df_rows[(df_rows["config"] == cfg_name) & (df_rows["seed"] == 42)]
        for _, row in cfg_s42.iterrows():
            fold = int(row["fold"])
            delta = abs(row["ll"] - ll_c0_seed42[fold])
            equiv_rows.append({
                "config": cfg_name, "fold": fold,
                "ll_c0": ll_c0_seed42[fold], "ll_cx": row["ll"],
                "delta": delta, "pass": delta <= _EQUIV_TOL,
            })
            status = "PASS" if delta <= _EQUIV_TOL else "FAIL"
            print(f"  {cfg_name} fold {fold:>2}: delta={delta:.2e}  [{status}]", flush=True)

    equiv_df = pd.DataFrame(equiv_rows)
    c1_pass = bool(equiv_df[equiv_df["config"] == "C1"]["pass"].all())
    c2_pass = bool(equiv_df[equiv_df["config"] == "C2"]["pass"].all())
    print(f"\n  C1 equivalence: {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(f"  C2 equivalence: {'PASS' if c2_pass else 'FAIL'}", flush=True)
    if not c2_pass:
        c2_max_delta = float(equiv_df[equiv_df["config"] == "C2"]["delta"].max())
        print(f"  C2 max delta: {c2_max_delta:.2e} — check param mapping in _LGBM_PARAMS_BASE",
              flush=True)

    # -----------------------------------------------------------------------
    # Speed summary
    # -----------------------------------------------------------------------
    print("\n=== Speed summary (mean fit_s per config, all seeds + folds) ===", flush=True)
    speed = (
        df_rows.groupby("config")["fit_s"]
        .agg(mean_s="mean", std_s="std", n="count")
        .reset_index()
    )
    c0_mean = float(speed.loc[speed["config"] == "C0", "mean_s"].iloc[0])
    speed["speedup_pct"] = (c0_mean - speed["mean_s"]) / c0_mean * 100

    config_stats: dict[str, dict] = {}
    for _, r in speed.iterrows():
        cfg = r["config"]
        print(
            f"  {cfg}: mean={r['mean_s']:.3f}s  std={r['std_s']:.3f}s  "
            f"n={int(r['n'])}  speedup={r['speedup_pct']:+.1f}%",
            flush=True,
        )
        config_stats[cfg] = {
            "mean_fit_s": round(float(r["mean_s"]), 4),
            "std_fit_s": round(float(r["std_s"]), 4),
            "n": int(r["n"]),
            "speedup_pct": round(float(r["speedup_pct"]), 2),
        }

    # -----------------------------------------------------------------------
    # Preliminary decisions (fill in analysis.md after reviewing)
    # -----------------------------------------------------------------------
    def _decide(cfg: str, equiv_ok: bool) -> str:
        if not equiv_ok:
            return "SKIP (equivalence failed)"
        sp = config_stats[cfg]["speedup_pct"]
        if sp >= 10.0:
            return "ADOPT (candidate)"
        if sp >= 5.0:
            return "NEEDS-MORE"
        return "SKIP (< 5% speedup)"

    decisions = {
        "C1": _decide("C1", c1_pass),
        "C2": _decide("C2", c2_pass),
    }
    print("\n=== Preliminary decisions ===", flush=True)
    for cfg, verdict in decisions.items():
        print(f"  {cfg}: {verdict}", flush=True)
    print("  (Override in analysis.md after reviewing per-fold variance.)", flush=True)

    # -----------------------------------------------------------------------
    # meta.json
    # -----------------------------------------------------------------------
    meta = {
        "issue": 220,
        "feature_cols": len(FEATURE_COLS),
        "n_folds": len(folds),
        "seeds": list(SEEDS),
        "total_fits": len(rows),
        "total_wall_s": round(time.perf_counter() - overall_t0, 1),
        "equiv_tol": _EQUIV_TOL,
        "c1_equiv_pass": c1_pass,
        "c2_equiv_pass": c2_pass,
        "config_stats": config_stats,
        "decisions": decisions,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmeta.json written: {OUT / 'meta.json'}", flush=True)
    print(f"Total wall: {meta['total_wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
