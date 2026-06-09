"""Shallow-elongated regime constraint for A — issue #214.

Tests whether the two-axis (elongation × shallowness) features close the
fold-7-style regression that the step5 row-level analysis traced to A's
``network_px_std`` signal misreading coordination in extended-descent rows.

All candidate features are computed in-script from columns already present
in features.csv — they only land in ``fuel_signal/features.py`` via a
follow-up PR if this experiment graduates them (mirrors the a_c_ablation
→ #216 pattern).

Candidate features (all PIT-safe, derived from existing FEATURE_COLUMNS):

- ``elongation_ratio`` = cycle_days_since_peak / station_baseline_cml,
  where station_baseline_cml = network-wide median of cycle_mean_length over
  the 730d window ending at (date - 1). Non-adaptive — the frozen baseline
  is the failure-mode fix for step5d_leakage's adaptive recalibration.
  (cycle_mean_length is derived from Sydney-wide average in CycleDetector,
  so it's identical across stations on a given date; "per-station" in
  #214's framing → network-wide here.)
- ``cycle_descent_slope_so_far`` =
  (station_price_cents - cycle_last_max_cents) / cycle_days_since_peak,
  null at the peak.
- ``is_extended_shallow_descent`` = (elongation_ratio > 1.0) AND
  (cycle_descent_slope_so_far > -0.9). Fallback test only.

Run grid (3 runs):

  R0           54-feat baseline (A+C already locked via #212 / RAC_full)
  R_raw        + elongation_ratio + cycle_descent_slope_so_far
  R_composite  + is_extended_shallow_descent

3 runs × 14 folds × 5 seeds = 210 LightGBM fits.

Per #214 methodology:
- Report MEAN and MEDIAN seed-aggregations. Median is the headline.
- Seed-variance gate: per (cohort, fold, run), flag ratio > 5× cohort median
  (per ``feedback_check_seed_variance_before_trusting_mean``).
- Save per-row predictions for downstream row-level diagnostic on the
  ext_descent_shallow bucket (gate #4 in #214).
- Walltime per fit + per phase. load_features() helper. PYTHONPATH=. prefix.
  LightGBM fit + predict with DataFrames.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-09_shallow_elongated/paired_wfcv.py \\
    2>&1 | tee experiments/2026-06-09_shallow_elongated/run.log
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
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
SHOCK_FOLDS = frozenset({1, 4, 9, 13})

CML_BASELINE_WINDOW_DAYS = 730
EXT_DESCENT_ELONGATION_THRESHOLD = 1.0
SHALLOW_SLOPE_THRESHOLD = -0.9

ELONGATION = "elongation_ratio"
SLOPE = "cycle_descent_slope_so_far"
COMPOSITE = "is_extended_shallow_descent"

RUNS: dict[str, list[str]] = {
    "R0": [],
    "R_raw": [ELONGATION, SLOPE],
    "R_composite": [COMPOSITE],
}


def add_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the three candidate columns from FEATURE_COLUMNS inputs.

    Reads ``cycle_mean_length``, ``cycle_days_since_peak``,
    ``station_price_cents`` and ``cycle_last_max_cents`` — all present in
    features.csv.
    """
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])

    # --- station_baseline_cml: network-wide rolling median of cycle_mean_length ---
    # cycle_mean_length is identical across stations on a given date (Sydney avg
    # series under the hood), so taking the per-date first value matches
    # taking the median across stations and is cheaper.
    cml_by_date = (
        df.dropna(subset=["cycle_mean_length"])
        .drop_duplicates("price_date")
        .set_index("price_date")["cycle_mean_length"]
        .sort_index()
    )
    # Reindex to a contiguous daily series so rolling(window=730, by days) is
    # well-defined even across data gaps.
    full_idx = pd.date_range(cml_by_date.index.min(), cml_by_date.index.max(), freq="D")
    cml_daily = cml_by_date.reindex(full_idx)
    # closed='left' → today's value does NOT enter today's median (PIT-strict).
    # min_periods=1 → emit a baseline as soon as any prior value is available.
    baseline = (
        cml_daily.rolling(f"{CML_BASELINE_WINDOW_DAYS}D", closed="left", min_periods=1)
        .median()
        .rename("station_baseline_cml")
    )
    df = df.join(baseline, on="price_date")

    # --- elongation_ratio ---
    safe = df["station_baseline_cml"] > 0
    df[ELONGATION] = np.where(
        safe, df["cycle_days_since_peak"] / df["station_baseline_cml"], np.nan,
    )

    # --- cycle_descent_slope_so_far ---
    nonzero = df["cycle_days_since_peak"] > 0
    df[SLOPE] = np.where(
        nonzero,
        (df["station_price_cents"] - df["cycle_last_max_cents"]) / df["cycle_days_since_peak"],
        np.nan,
    )

    # --- is_extended_shallow_descent (composite) ---
    ext_descent = df[ELONGATION] > EXT_DESCENT_ELONGATION_THRESHOLD
    shallow = df[SLOPE] > SHALLOW_SLOPE_THRESHOLD
    df[COMPOSITE] = (ext_descent & shallow).astype(np.int8)

    return df


def fit_score(
    train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str], seed: int,
) -> tuple[float, np.ndarray, float]:
    t0 = time.perf_counter()
    model = LGBMClassifier(random_state=seed, verbose=-1, subsample=0.8, subsample_freq=1)
    model.fit(train_df[cols], train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols])[:, 1]
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, p, time.perf_counter() - t0


def per_row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading features ...", flush=True)
    t0 = time.perf_counter()
    df = load_features()
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}", flush=True)

    print("Computing candidate features in-script ...", flush=True)
    t0 = time.perf_counter()
    df = add_candidate_columns(df)
    print(f"  [add_candidate_columns] {time.perf_counter() - t0:.1f}s", flush=True)
    print(
        f"  null rates: {ELONGATION}={df[ELONGATION].isna().mean()*100:.2f}%  "
        f"{SLOPE}={df[SLOPE].isna().mean()*100:.2f}%  "
        f"{COMPOSITE}_positive={df[COMPOSITE].mean()*100:.2f}%",
        flush=True,
    )

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    assert len(baseline_cols) == 54, f"expected 54, got {len(baseline_cols)}"
    print(f"\nBaseline features: {len(baseline_cols)}", flush=True)
    print(f"Run grid: {list(RUNS.keys())}", flush=True)
    print(f"Seeds: {SEEDS} (n={len(SEEDS)})", flush=True)

    folds = list(_ev.walk_forward_folds(df, train_min_days=1825, val_days=90, step_days=90))
    print(f"Walk-forward folds: {len(folds)}\n", flush=True)

    print(
        f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  "
        f"{'val_rows':>8}  {'run':<12}  {'seed':>4}  "
        f"{'ll_all':>7}  {'ll_h25':>7}  {'fit_s':>6}",
        flush=True,
    )
    print("-" * 110, flush=True)

    rows: list[dict] = []
    pred_blocks: list[pd.DataFrame] = []

    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        vd = pd.to_datetime(val_df["price_date"])
        val_start = vd.min()
        val_end = vd.max()
        regime = "shock" if i in SHOCK_FOLDS else "normal"
        y = val_df["label"].to_numpy(dtype=int)

        # Hard-cohort mask: derived from baseline + seed42 per-row log-loss.
        ll0, p0, t0 = fit_score(train_df, val_df, baseline_cols, SEEDS[0])
        prl0 = per_row_log_loss(y, p0)
        hard25_thresh = float(np.quantile(prl0, 0.75))
        hard25_mask = prl0 >= hard25_thresh

        # Row buckets for the row-level diagnostic (gate #4 in #214).
        elong = val_df[ELONGATION].to_numpy(dtype=float)
        slope = val_df[SLOPE].to_numpy(dtype=float)
        is_ext_descent = np.where(
            np.isfinite(elong), elong > EXT_DESCENT_ELONGATION_THRESHOLD, False,
        )
        is_ext_descent_shallow = is_ext_descent & np.where(
            np.isfinite(slope), slope > SHALLOW_SLOPE_THRESHOLD, False,
        )

        ident = pd.DataFrame({
            "fold": np.int8(i),
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
                    "fold": i, "regime": regime,
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
                    f"{i:>4}  {regime:>6}  "
                    f"{val_start.strftime('%Y-%m-%d'):>10}  "
                    f"{val_end.strftime('%Y-%m-%d'):>10}  "
                    f"{len(val_df):>8,}  {run_name:<12}  {seed:>4}  "
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

    # --- Seed-variance gate ---
    cohort_ll = {"all": "ll_all", "hard25": "ll_hard25"}
    seed_var_flags: list[dict] = []
    seed_var_summary: dict[str, dict] = {}
    for cohort, col in cohort_ll.items():
        agg = (
            df_rows.groupby(["fold", "run"], as_index=False)
            .agg(seed_std=(col, lambda s: float(np.nanstd(s, ddof=1))))
        )
        cohort_med = float(np.nanmedian(agg["seed_std"])) if len(agg) else float("nan")
        if not np.isfinite(cohort_med) or cohort_med <= 0:
            raise ValueError(
                f"Seed-variance gate: cohort {cohort!r} median seed_std is "
                f"{cohort_med!r} (n_cells={len(agg)}). Investigate before trusting aggregates."
            )
        agg["seed_std_ratio"] = agg["seed_std"] / cohort_med
        flagged = agg[agg["seed_std_ratio"] > 5.0]
        seed_var_summary[cohort] = {
            "cohort_median_seed_std": cohort_med,
            "n_cells": int(len(agg)),
            "n_flagged_gt_5x": int(len(flagged)),
        }
        for _, r in flagged.iterrows():
            seed_var_flags.append({
                "cohort": cohort,
                "fold": int(r["fold"]), "run": r["run"],
                "seed_std": float(r["seed_std"]),
                "ratio_vs_cohort_median": float(r["seed_std_ratio"]),
            })

    if seed_var_flags:
        print("\n!! SEED-VARIANCE FLAGS (seed_std > 5× cohort median) !!", flush=True)
        for f in seed_var_flags:
            print(
                f"   [{f['cohort']:<6}] fold={f['fold']:>2}  run={f['run']:<12}  "
                f"seed_std={f['seed_std']:.4f}  ratio={f['ratio_vs_cohort_median']:.1f}×",
                flush=True,
            )
    else:
        print("\nSeed-variance gate: no flagged cells (all seed_std ≤ 5× cohort median).", flush=True)

    # --- Aggregations: mean AND median across seeds per (fold, run) ---
    agg_kwargs: dict[str, tuple[str, object]] = {}
    for col in cohort_ll.values():
        agg_kwargs[f"{col}_mean"] = (col, "mean")
        agg_kwargs[f"{col}_median"] = (col, "median")
        agg_kwargs[f"{col}_seedstd"] = (col, lambda s: float(np.nanstd(s, ddof=1)))
    fold_run = df_rows.groupby(["fold", "regime", "run"], as_index=False).agg(**agg_kwargs)

    base_rename = {}
    for c in cohort_ll.values():
        base_rename[f"{c}_mean"] = f"{c}_mean_base"
        base_rename[f"{c}_median"] = f"{c}_median_base"
    base = fold_run[fold_run["run"] == "R0"][
        ["fold"]
        + [f"{c}_mean" for c in cohort_ll.values()]
        + [f"{c}_median" for c in cohort_ll.values()]
    ].rename(columns=base_rename)
    fold_run = fold_run.merge(base, on="fold")
    for c in cohort_ll.values():
        fold_run[f"delta_{c}_mean"] = fold_run[f"{c}_mean"] - fold_run[f"{c}_mean_base"]
        fold_run[f"delta_{c}_median"] = fold_run[f"{c}_median"] - fold_run[f"{c}_median_base"]

    fold_run.to_csv(OUT / "fold_run.csv", index=False)

    # --- Summary across folds (headline = median-of-seeds, mean-across-folds) ---
    summary: list[dict] = []
    print("\n=== Aggregate per run (MEDIAN seed-agg headline; MEAN shown alongside) ===", flush=True)
    print(
        f"    {'run':<12}  {'Δh25 (med)':>12}  {'Δh25 (mean)':>12}  "
        f"{'Δall med':>10}  {'helps_h25':>10}  {'fold7_Δh25':>11}",
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

        if run_name == "R0":
            print(
                f"    {run_name:<12}  baseline reference "
                f"(median ll_h25 across folds = "
                f"{float(np.nanmedian(sub['ll_hard25_median'])):.4f})",
                flush=True,
            )
        else:
            print(
                f"    {run_name:<12}  "
                f"{float(d_h25_med.mean()):>+12.4f}  "
                f"{float(d_h25_mean.mean()):>+12.4f}  "
                f"{float(d_all_med.mean()):>+10.4f}  "
                f"{(d_h25_med < 0).sum():>4}/{n_folds:<5}  "
                f"{fold7_h25:>+11.4f}",
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
            "worst_fold_delta_hard25_median":
                float(d_h25_med.max()) if n_folds else None,
        })

    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "n_baseline_features": len(baseline_cols),
        "candidate_columns_computed_in_script": [ELONGATION, SLOPE, COMPOSITE],
        "definitions": {
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

    def _to_jsonable(o):
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(x) for x in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    (OUT / "meta.json").write_text(json.dumps(_to_jsonable(meta), indent=2, default=str))
    print(f"\nMeta: {OUT / 'meta.json'}", flush=True)
    print(f"[total wall] {time.perf_counter() - overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
