"""Interaction-column probe for A in the shallow-elongated corner — issue #231.

The #214 oracle diagnostic (see ``experiments/2026-06-09_shallow_elongated/
analysis.md``) showed the ingredients to carve out the failure corner are all
present in the feature set: A's level (``network_px_std``) separates elongated
from normal, and ``cycle_descent_slope_so_far`` separates shallow from steep.
But R_raw (the raw axes added separately) failed because expressing "A misreads
when (elongated AND shallow)" needs THREE nested tree splits, and the corner is
~8% of train data — the tree never found the conditional. R_composite (one
binary flag) was also flat.

This probe hands the tree the **product directly** as a column so it only needs
one split. The test discriminates two hypotheses: failure is about *combination
representation* (interaction column should help) vs *underlying signal
availability* (interaction column multiplies noise and stays flat, per the
oracle diagnostic's pessimistic read). See ``feedback_tree_interaction_limits``.

All candidate columns are computed **in-script** from columns already present
in features.csv — they only land in ``fuel_signal/features.py`` via a follow-up
PR if this experiment graduates them (mirrors a_c_ablation → #216, #214).

Building blocks (same as #214, PIT-safe):

- ``elongation_ratio`` = cycle_days_since_peak / station_baseline_cml, where
  station_baseline_cml = network-wide rolling median of cycle_mean_length over
  the 730d window ending at (date - 1), closed='left' (non-adaptive frozen
  baseline). Computed via ``experiments.lib.features.rolling.rolling_baseline``.
- ``cycle_descent_slope_so_far`` =
  (station_price_cents - cycle_last_max_cents) / cycle_days_since_peak; null at
  the peak.
- ``is_extended_shallow_descent`` = (elongation_ratio > 1.0) AND
  (cycle_descent_slope_so_far > -0.9). The corner mask.

Interaction columns under test (A = ``network_px_std``, already in baseline):

- ``A_x_shallow_elong`` = A × is_extended_shallow_descent
  (A's value inside the corner, 0 outside — one split isolates corner-A).
- ``A_x_other``        = A × (1 - is_extended_shallow_descent)
  (complement — lets the tree split normal-regime A independently of corner A).
- ``A_x_smooth``       = A × elongation_ratio × max(0, slope + 0.9)
  (continuous version; weight grows with elongation and shallowness, 0 for
  steep descent — avoids the hard threshold).

Run grid (4 runs):

  R0    54-feat baseline (A+C already locked via #212 / RAC_full)
  R1    + A_x_shallow_elong
  R2    + A_x_shallow_elong + A_x_other
  R3    + A_x_smooth

4 runs × 14 folds × 5 seeds = 280 LightGBM fits.

Methodology (mirrors #214):
- Report MEAN and MEDIAN seed-aggregations. Median is the headline.
- Seed-variance gate: per (cohort, fold, run), flag ratio > 5× cohort median
  (per ``feedback_check_seed_variance_before_trusting_mean``).
- Save per-row predictions for the gate-4 row-level diagnostic on the
  ext_descent_shallow bucket.
- Walltime per fit + per phase. load_features() helper. PYTHONPATH=. prefix.
  LightGBM fit + predict with DataFrames.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-11_interaction_column/paired_wfcv.py \\
    2>&1 | tee experiments/2026-06-11_interaction_column/run.log
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd

from experiments.lib.aggregate import aggregate_with_deltas
from experiments.lib.cohorts import hard_quantile_mask
from experiments.lib.constants import SEEDS, SHOCK_FOLDS
from experiments.lib.features.rolling import rolling_baseline
from experiments.lib.fit import fit_score, per_row_log_loss
from experiments.lib.folds import iter_folds_with_baseline_fit
from experiments.lib.gates import seed_variance_gate
from experiments.lib.io import write_meta
from experiments.lib.timing import time_block
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    load_features,
)

OUT = pathlib.Path(__file__).parent

CML_BASELINE_WINDOW_DAYS = 730
EXT_DESCENT_ELONGATION_THRESHOLD = 1.0
SHALLOW_SLOPE_THRESHOLD = -0.9

A_COL = "network_px_std"  # the "A" signal, locked into the baseline via #212

ELONGATION = "elongation_ratio"
SLOPE = "cycle_descent_slope_so_far"
COMPOSITE = "is_extended_shallow_descent"

A_X_CORNER = "A_x_shallow_elong"
A_X_OTHER = "A_x_other"
A_X_SMOOTH = "A_x_smooth"

RUNS: dict[str, list[str]] = {
    "R0": [],
    "R1": [A_X_CORNER],
    "R2": [A_X_CORNER, A_X_OTHER],
    "R3": [A_X_SMOOTH],
}


def add_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the corner mask and the three interaction columns.

    Reads ``cycle_mean_length``, ``cycle_days_since_peak``,
    ``station_price_cents``, ``cycle_last_max_cents`` and ``network_px_std`` —
    all present in features.csv.
    """
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])

    # station_baseline_cml: network-wide rolling median of cycle_mean_length.
    # cycle_mean_length is identical across stations on a given date (Sydney-avg
    # series under the hood), so the per-date first value matches the median
    # across stations and is cheaper. closed='left' → today's value does NOT
    # enter today's median (PIT-strict).
    cml_by_date = (
        df.dropna(subset=["cycle_mean_length"])
        .drop_duplicates("price_date")
        .set_index("price_date")["cycle_mean_length"]
    )
    baseline = rolling_baseline(
        cml_by_date, CML_BASELINE_WINDOW_DAYS, closed="left", min_periods=1,
    ).rename("station_baseline_cml")
    df = df.join(baseline, on="price_date")

    safe = df["station_baseline_cml"] > 0
    df[ELONGATION] = np.where(
        safe, df["cycle_days_since_peak"] / df["station_baseline_cml"], np.nan,
    )

    nonzero = df["cycle_days_since_peak"] > 0
    df[SLOPE] = np.where(
        nonzero,
        (df["station_price_cents"] - df["cycle_last_max_cents"]) / df["cycle_days_since_peak"],
        np.nan,
    )

    ext_descent = df[ELONGATION] > EXT_DESCENT_ELONGATION_THRESHOLD
    shallow = df[SLOPE] > SHALLOW_SLOPE_THRESHOLD
    df[COMPOSITE] = (ext_descent & shallow).astype(np.int8)

    # Interaction columns. A × {corner, complement}: the binary mask zeroes A
    # outside / inside the corner so the tree splits corner-A from the rest in
    # one node. NaN A propagates (LightGBM treats as missing) — only ~early
    # rows where the network dispersion baseline is undefined.
    a = df[A_COL]
    df[A_X_CORNER] = a * df[COMPOSITE]
    df[A_X_OTHER] = a * (1 - df[COMPOSITE])
    # Smooth: weight grows with elongation and shallowness, 0 for steep descent
    # (slope <= -0.9). slope NaN at the peak → product NaN (missing).
    shallow_weight = np.maximum(0.0, df[SLOPE] + 0.9)
    df[A_X_SMOOTH] = a * df[ELONGATION] * shallow_weight

    return df


def main() -> None:
    overall_t0 = time.perf_counter()

    print("Loading features ...", flush=True)
    with time_block("load_features"):
        df = load_features()
    print(f"  rows={len(df):,}", flush=True)

    print("Computing candidate features in-script ...", flush=True)
    with time_block("add_candidate_columns"):
        df = add_candidate_columns(df)
    print(
        f"  null rates: {ELONGATION}={df[ELONGATION].isna().mean()*100:.2f}%  "
        f"{SLOPE}={df[SLOPE].isna().mean()*100:.2f}%  "
        f"{COMPOSITE}_positive={df[COMPOSITE].mean()*100:.2f}%  "
        f"{A_X_CORNER}_nonzero={(df[A_X_CORNER].fillna(0) != 0).mean()*100:.2f}%  "
        f"{A_X_SMOOTH}_nonzero={(df[A_X_SMOOTH].fillna(0) != 0).mean()*100:.2f}%",
        flush=True,
    )

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    assert len(baseline_cols) == 54, f"expected 54, got {len(baseline_cols)}"
    assert A_COL in baseline_cols, f"{A_COL} must be in the baseline"
    print(f"\nBaseline features: {len(baseline_cols)}", flush=True)
    print(f"Run grid: {list(RUNS.keys())}", flush=True)
    print(f"Seeds: {SEEDS} (n={len(SEEDS)})", flush=True)

    print(
        f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  "
        f"{'val_rows':>8}  {'run':<10}  {'seed':>4}  "
        f"{'ll_all':>7}  {'ll_h25':>7}  {'fit_s':>6}",
        flush=True,
    )
    print("-" * 110, flush=True)

    rows: list[dict] = []
    pred_blocks: list[pd.DataFrame] = []

    for fold_idx, regime, train_df, val_df, ll0, p0, t0, prl0 in iter_folds_with_baseline_fit(
        df, baseline_cols
    ):
        vd = pd.to_datetime(val_df["price_date"])
        val_start = vd.min()
        val_end = vd.max()
        y = val_df["label"].to_numpy(dtype=int)
        hard25_mask = hard_quantile_mask(prl0, 0.75)

        elong = val_df[ELONGATION].to_numpy(dtype=float)
        slope = val_df[SLOPE].to_numpy(dtype=float)
        is_ext_descent = np.where(
            np.isfinite(elong), elong > EXT_DESCENT_ELONGATION_THRESHOLD, False,
        )
        is_ext_descent_shallow = is_ext_descent & np.where(
            np.isfinite(slope), slope > SHALLOW_SLOPE_THRESHOLD, False,
        )

        ident = pd.DataFrame({
            "fold": np.int8(fold_idx),
            "station_code": val_df["station_code"].to_numpy(),
            "price_date": vd.to_numpy(),
            "label": y.astype(np.int8),
            "is_hard25": hard25_mask.astype(np.int8),
            "is_ext_descent": is_ext_descent.astype(np.int8),
            "is_ext_descent_shallow": is_ext_descent_shallow.astype(np.int8),
        })

        for run_name, extra in RUNS.items():
            cols = baseline_cols + extra
            for seed in SEEDS:
                if run_name == "R0" and seed == SEEDS[0]:
                    ll, p, t = ll0, p0, t0
                else:
                    ll, p, t = fit_score(train_df, val_df, cols, seed)
                prl = per_row_log_loss(y, p)
                ll_hard25 = float(prl[hard25_mask].mean()) if hard25_mask.any() else float("nan")

                rows.append({
                    "fold": fold_idx, "regime": regime,
                    "val_start": val_start.strftime("%Y-%m-%d"),
                    "val_end": val_end.strftime("%Y-%m-%d"),
                    "val_rows": len(val_df),
                    "run": run_name, "n_features": len(cols),
                    "seed": seed,
                    "ll_all": ll, "ll_hard25": ll_hard25,
                    "fit_s": t,
                })

                block = ident.copy()
                block["run"] = run_name
                block["seed"] = np.int8(seed)
                block["proba"] = p.astype(np.float32)
                pred_blocks.append(block)

                print(
                    f"{fold_idx:>4}  {regime:>6}  "
                    f"{val_start.strftime('%Y-%m-%d'):>10}  "
                    f"{val_end.strftime('%Y-%m-%d'):>10}  "
                    f"{len(val_df):>8,}  {run_name:<10}  {seed:>4}  "
                    f"{ll:>7.4f}  {ll_hard25:>7.4f}  {t:>5.1f}s",
                    flush=True,
                )

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "runs.csv", index=False)
    print(f"\nPer-(fold,run,seed) results: {OUT / 'runs.csv'}", flush=True)

    pred_df = pd.concat(pred_blocks, ignore_index=True)
    out_parquet = OUT / "rowpreds.parquet"
    pred_df.to_parquet(out_parquet, index=False, compression="zstd")
    print(f"Per-row predictions: {out_parquet}  ({len(pred_df):,} rows)", flush=True)

    cohort_ll = {"all": "ll_all", "hard25": "ll_hard25"}
    seed_var_summary, seed_var_flags = seed_variance_gate(df_rows, cohort_ll)
    fold_run = aggregate_with_deltas(df_rows, cohort_ll)
    fold_run.to_csv(OUT / "fold_run.csv", index=False)

    summary: list[dict] = []
    print("\n=== Aggregate per run (MEDIAN seed-agg headline; MEAN shown alongside) ===", flush=True)
    print(
        f"    {'run':<10}  {'Δh25 (med)':>12}  {'Δh25 (mean)':>12}  "
        f"{'Δall med':>10}  {'helps_h25':>10}  {'fold7_Δh25':>11}  {'worst_Δh25':>11}",
        flush=True,
    )
    for run_name in RUNS:
        sub = fold_run[fold_run["run"] == run_name]
        n_folds = len(sub)
        d_h25_med = sub["delta_ll_hard25_median"].to_numpy()
        d_h25_mean = sub["delta_ll_hard25_mean"].to_numpy()
        d_all_med = sub["delta_ll_all_median"].to_numpy()
        fold7 = sub.loc[sub["fold"] == 7, "delta_ll_hard25_median"]
        fold7_h25 = float(fold7.iloc[0]) if len(fold7) else float("nan")
        worst_h25 = float(d_h25_med.max()) if n_folds else float("nan")

        if run_name == "R0":
            print(
                f"    {run_name:<10}  baseline reference "
                f"(median ll_h25 across folds = "
                f"{float(np.nanmedian(sub['ll_hard25_median'])):.4f})",
                flush=True,
            )
        else:
            print(
                f"    {run_name:<10}  "
                f"{float(d_h25_med.mean()):>+12.4f}  "
                f"{float(d_h25_mean.mean()):>+12.4f}  "
                f"{float(d_all_med.mean()):>+10.4f}  "
                f"{(d_h25_med < 0).sum():>4}/{n_folds:<5}  "
                f"{fold7_h25:>+11.4f}  {worst_h25:>+11.4f}",
                flush=True,
            )

        summary.append({
            "run": run_name,
            "n_folds": n_folds,
            "delta_all_median_mean": float(np.nanmean(d_all_med)) if n_folds else None,
            "delta_hard25_median_mean": float(np.nanmean(d_h25_med)) if n_folds else None,
            "delta_hard25_mean_mean": float(np.nanmean(d_h25_mean)) if n_folds else None,
            "helps_hard25_n": int((d_h25_med < 0).sum()),
            "helps_hard25_n_folds": n_folds,
            "fold7_delta_hard25_median": fold7_h25,
            "worst_fold_delta_hard25_median": worst_h25 if n_folds else None,
        })

    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "n_baseline_features": len(baseline_cols),
        "candidate_columns_computed_in_script": [A_X_CORNER, A_X_OTHER, A_X_SMOOTH],
        "definitions": {
            A_X_CORNER: f"{A_COL} × {COMPOSITE} (A inside the corner, 0 outside)",
            A_X_OTHER: f"{A_COL} × (1 - {COMPOSITE}) (A outside the corner, 0 inside)",
            A_X_SMOOTH: (
                f"{A_COL} × {ELONGATION} × max(0, {SLOPE} + 0.9) "
                "(continuous corner weight; 0 for steep descent)"
            ),
            ELONGATION: (
                "cycle_days_since_peak / station_baseline_cml, where "
                f"station_baseline_cml = rolling median over {CML_BASELINE_WINDOW_DAYS}d "
                "ending d-1 (closed='left'). Non-adaptive (frozen baseline)."
            ),
            SLOPE: (
                "(station_price_cents - cycle_last_max_cents) / cycle_days_since_peak; "
                "null at the peak."
            ),
            COMPOSITE: (
                f"({ELONGATION} > {EXT_DESCENT_ELONGATION_THRESHOLD}) AND "
                f"({SLOPE} > {SHALLOW_SLOPE_THRESHOLD})"
            ),
        },
        "bucket_definitions": {
            "ext_descent": f"{ELONGATION} > {EXT_DESCENT_ELONGATION_THRESHOLD}",
            "ext_descent_shallow":
                f"ext_descent AND {SLOPE} > {SHALLOW_SLOPE_THRESHOLD}",
        },
        "run_grid": dict(RUNS),
        "cohort_definitions": {
            "all": "full val set",
            "hard25": "top quartile baseline per-row log-loss per fold",
        },
        "aggregation_convention": (
            "Headline = median across 5 seeds per (fold, run); summary then "
            "averages those medians across 14 folds. Mean shown alongside."
        ),
        "seed_variance_gate": {
            "rule": "ratio = seed_std / median(seed_std across cohort cells); flag > 5×",
            "per_cohort": seed_var_summary,
            "flagged_cells": seed_var_flags,
        },
        "summary": summary,
        "total_wall_seconds": time.perf_counter() - overall_t0,
    }

    write_meta(OUT, meta)
    print(f"[total wall] {time.perf_counter() - overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
