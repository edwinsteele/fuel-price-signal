"""Within-family A ablation (+ C as complement) — issue #212.

Decides which subset of Family A graduates to ``fuel_signal/features.py``,
and whether C should graduate alongside.

Run grid (5 runs):

  R0         baseline (50-feat Phase 4)
  RA_level   + network_px_std
  RA_delta   + network_px_std_delta_3d
  RA_both    + network_px_std + network_px_std_delta_3d
  RAC_full   + network_px_std + network_px_std_delta_3d
             + lga_phase_std + lga_phase_std_delta_3d

5 runs x 14 folds x 5 seeds = 350 LightGBM fits.

Per #212 methodology:
- Report MEAN and MEDIAN seed-aggregations. Median is the headline.
- Seed-variance gate: for each (cohort, fold, run) cell compute
  ratio = seed_std / median(seed_std across cohort). Flag ratio > 5x in
  stdout AND meta.json (per ``feedback_check_seed_variance_before_trusting_mean``).
- Elongation-conditional diagnostic (informational, not gating): per-fold
  delta vs frozen-baseline elongation exposure for each non-baseline run.
- Walltime per fit. ``load_features()`` helper. ``PYTHONPATH=.`` prefix.
  LightGBM fit + predict with DataFrames.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-07_a_c_ablation/paired_wfcv.py \\
    2>&1 | tee experiments/2026-06-07_a_c_ablation/run.log
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, load_features

OUT = pathlib.Path(__file__).parent
SEEDS = (42, 43, 44, 45, 46)
SHOCK_FOLDS = frozenset({1, 4, 9, 13})

# Cohort thresholds (provisional pending #207); matches step2_paired_wfcv.py.
COMP_BAND_CENTS = 5.0
DELTA_LAG_DAYS = 3
BASELINE_WINDOW_DAYS = 730  # frozen elongation baseline window (per step4)

# Brand `days_since_trough_entry_*` cols live in features.csv but are not
# part of the 50-feat baseline; excluded from the LGA set used for Signal C.
_BRANDS = {"7_eleven", "ampol_foodary", "bp", "budget", "eg_ampol",
           "independent", "metro_fuel", "reddy_express", "shell", "speedway"}

A_LEVEL = "network_px_std"
A_DELTA = "network_px_std_delta_3d"
C_LEVEL = "lga_phase_std"
C_DELTA = "lga_phase_std_delta_3d"

NEW_COLS = [A_LEVEL, A_DELTA, C_LEVEL, C_DELTA]

RUNS: dict[str, list[str]] = {
    "R0":        [],
    "RA_level":  [A_LEVEL],
    "RA_delta":  [A_DELTA],
    "RA_both":   [A_LEVEL, A_DELTA],
    "RAC_full":  [A_LEVEL, A_DELTA, C_LEVEL, C_DELTA],
}


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute A and C feature columns + the `_px_5d_change` diagnostic for
    the ``lated`` cohort mask. PIT-safe per same construction as step2."""
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])

    # --- Signal A: cross-station dispersion within competitive cohort ---
    comp = df[df["stickiness_score"].abs() <= COMP_BAND_CENTS]
    a_by_date = comp.groupby("price_date")["station_price_cents"].std().rename(A_LEVEL)
    df = df.join(a_by_date, on="price_date")

    # --- Signal C: std of days_since_trough across 35 LGAs ---
    lga_cols = [c for c in df.columns
                if c.startswith("days_since_trough_entry_")
                and c.removeprefix("days_since_trough_entry_") not in _BRANDS]
    if len(lga_cols) != 35:
        raise ValueError(
            f"Expected 35 LGA columns for Signal C; found {len(lga_cols)}. "
            "If the LGA set changed upstream, review the C signal definition."
        )
    per_date = df.drop_duplicates("price_date").set_index("price_date")[lga_cols]
    c_by_date = per_date.std(axis=1).rename(C_LEVEL)
    df = df.join(c_by_date, on="price_date")

    # --- Calendar-aware deltas ---
    for level_col, delta_col in [(A_LEVEL, A_DELTA), (C_LEVEL, C_DELTA)]:
        per_date_level = (
            df.drop_duplicates("price_date").set_index("price_date")[level_col].sort_index()
        )
        full_idx = pd.date_range(per_date_level.index.min(),
                                 per_date_level.index.max(), freq="D")
        s = per_date_level.reindex(full_idx)
        delta = (s - s.shift(DELTA_LAG_DAYS)).rename(delta_col)
        df = df.join(delta, on="price_date")

    # --- Diagnostic for `lated` cohort mask (NOT a feature) ---
    lookup = df[["station_code", "price_date", "station_price_cents"]].rename(
        columns={"price_date": "_lookup_date", "station_price_cents": "_px_5d_ago"}
    )
    df["_lookup_date"] = df["price_date"] - pd.Timedelta(days=5)
    df = df.merge(lookup, on=["station_code", "_lookup_date"],
                  how="left", validate="m:1")
    df["_px_5d_change"] = df["station_price_cents"] - df["_px_5d_ago"]
    df = df.drop(columns=["_lookup_date", "_px_5d_ago"])

    return df


def fit_score(train_df: pd.DataFrame, val_df: pd.DataFrame,
              cols: list[str], seed: int) -> tuple[float, np.ndarray, float]:
    t0 = time.perf_counter()
    model = LGBMClassifier(random_state=seed, verbose=-1,
                           subsample=0.8, subsample_freq=1)
    model.fit(train_df[cols], train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols])[:, 1]
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, p, time.perf_counter() - t0


def per_row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def elongation_score(features: pd.DataFrame, val_start: pd.Timestamp,
                     val_end: pd.Timestamp) -> float:
    """Median per-row (cycle_days_since_peak / frozen baseline cycle length).

    Frozen baseline: median ``cycle_mean_length`` per station over the
    730d window ending at (val_start - 1). Defeats the adaptive recalibration
    problem of attempt 1 in ``project_late_descent_elongation_regime``.
    """
    base_lo = val_start - pd.Timedelta(days=BASELINE_WINDOW_DAYS)
    base_hi = val_start - pd.Timedelta(days=1)
    base_mask = (features["price_date"] >= base_lo) & (features["price_date"] <= base_hi)
    base = (
        features.loc[base_mask, ["station_code", "cycle_mean_length"]]
        .dropna()
        .groupby("station_code", as_index=False)["cycle_mean_length"]
        .median()
        .rename(columns={"cycle_mean_length": "cml_base"})
    )
    val_mask = (features["price_date"] >= val_start) & (features["price_date"] <= val_end)
    val = features.loc[val_mask, ["station_code", "cycle_days_since_peak"]].dropna()
    val = val.merge(base, on="station_code", how="inner")
    val = val[val["cml_base"] > 0]
    if val.empty:
        return float("nan")
    return float((val["cycle_days_since_peak"] / val["cml_base"]).median())


def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading features ...", flush=True)
    t0 = time.perf_counter()
    df = load_features()
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}", flush=True)

    print("Computing A+C features ...", flush=True)
    t0 = time.perf_counter()
    df = compute_features(df)
    print(f"  [compute_features] {time.perf_counter() - t0:.1f}s", flush=True)
    for c in NEW_COLS:
        null_pct = df[c].isna().mean() * 100
        print(f"    {c}: nulls = {null_pct:.2f}%", flush=True)

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    assert len(baseline_cols) == 50, f"expected 50, got {len(baseline_cols)}"
    print(f"\nBaseline features: {len(baseline_cols)}", flush=True)
    print(f"Run grid: {list(RUNS.keys())}", flush=True)
    print(f"Seeds: {SEEDS} (n={len(SEEDS)})", flush=True)

    folds = list(_ev.walk_forward_folds(df, train_min_days=1825,
                                       val_days=90, step_days=90))
    print(f"Walk-forward folds: {len(folds)}\n", flush=True)

    print(f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  "
          f"{'val_rows':>8}  {'run':<10}  {'seed':>4}  "
          f"{'ll_all':>7}  {'ll_h25':>7}  {'ll_h10':>7}  {'ll_lated':>8}  "
          f"{'fit_s':>6}", flush=True)
    print("-" * 128, flush=True)

    rows: list[dict] = []
    elong_rows: list[dict] = []

    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        vd = pd.to_datetime(val_df["price_date"])
        val_start = vd.min()
        val_end = vd.max()
        regime = "shock" if i in SHOCK_FOLDS else "normal"
        y = val_df["label"].to_numpy(dtype=int)

        # Fold-stable hard-cohort masks: derived from baseline+seed42 per-row
        # log-loss. All runs scored on the same mask per fold (apples-to-apples).
        ll0, p0, t0 = fit_score(train_df, val_df, baseline_cols, SEEDS[0])
        prl0 = per_row_log_loss(y, p0)
        hard25_thresh = np.quantile(prl0, 0.75)
        hard10_thresh = np.quantile(prl0, 0.90)
        hard25_mask = prl0 >= hard25_thresh
        hard10_mask = prl0 >= hard10_thresh
        pct = val_df["cycle_pct_through"].to_numpy(dtype=float)
        d5 = val_df["_px_5d_change"].to_numpy(dtype=float)
        lated_mask = (pct >= 0.9) & np.isfinite(d5) & (d5 <= -2.0)
        n_lated = int(lated_mask.sum())

        # Per-fold elongation diagnostic (computed once, used in summary).
        elong = elongation_score(df, val_start, val_end)
        elong_rows.append({
            "fold": i, "regime": regime,
            "val_start": val_start.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "elongation": elong,
        })

        for run_name, extra in RUNS.items():
            cols = baseline_cols + extra
            for seed in SEEDS:
                if run_name == "R0" and seed == SEEDS[0]:
                    ll, p, t = ll0, p0, t0
                else:
                    ll, p, t = fit_score(train_df, val_df, cols, seed)
                prl = per_row_log_loss(y, p)
                ll_hard25 = float(prl[hard25_mask].mean())
                ll_hard10 = float(prl[hard10_mask].mean())
                ll_lated = float(prl[lated_mask].mean()) if n_lated > 0 else float("nan")
                rows.append({
                    "fold": i, "regime": regime,
                    "val_start": val_start.strftime("%Y-%m-%d"),
                    "val_end": val_end.strftime("%Y-%m-%d"),
                    "val_rows": len(val_df), "n_lated": n_lated,
                    "run": run_name, "n_features": len(cols),
                    "seed": seed,
                    "ll_all": ll, "ll_hard25": ll_hard25,
                    "ll_hard10": ll_hard10, "ll_lated": ll_lated,
                    "fit_s": t,
                })
                print(f"{i:>4}  {regime:>6}  "
                      f"{val_start.strftime('%Y-%m-%d'):>10}  "
                      f"{val_end.strftime('%Y-%m-%d'):>10}  "
                      f"{len(val_df):>8,}  {run_name:<10}  {seed:>4}  "
                      f"{ll:>7.4f}  {ll_hard25:>7.4f}  {ll_hard10:>7.4f}  "
                      f"{ll_lated:>8.4f}  {t:>5.1f}s", flush=True)

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "runs.csv", index=False)
    print(f"\nPer-(fold,run,seed) results: {OUT / 'runs.csv'}", flush=True)

    elong_df = pd.DataFrame(elong_rows)
    elong_df.to_csv(OUT / "elongation_per_fold.csv", index=False)

    # --- Seed-variance gate ---
    # Per (cohort, fold, run): seed_std and ratio = seed_std / median across
    # all cells of the cohort. Flag ratio > 5x.
    cohort_ll = {"all": "ll_all", "hard25": "ll_hard25",
                 "hard10": "ll_hard10", "lated": "ll_lated"}
    seed_var_flags: list[dict] = []
    seed_var_summary: dict[str, dict] = {}
    for cohort, col in cohort_ll.items():
        agg = (
            df_rows.groupby(["fold", "run"], as_index=False)
            .agg(seed_std=(col, lambda s: float(np.nanstd(s, ddof=1))))
        )
        cohort_med = float(np.nanmedian(agg["seed_std"])) if len(agg) else float("nan")
        # Guard the degenerate denominator: a NaN/0 cohort median would
        # silently broadcast NaN across the whole ratio column and mask any
        # real outlier (the gate would then report n_flagged_gt_5x = 0).
        # Raise instead so the operator sees the problem.
        if not np.isfinite(cohort_med) or cohort_med <= 0:
            raise ValueError(
                f"Seed-variance gate: cohort {cohort!r} median seed_std is "
                f"{cohort_med!r} (n_cells={len(agg)}). The gate cannot run on "
                "a zero or NaN denominator; investigate the seeds tuple or "
                "the cohort definition before quoting any aggregate."
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
        print("\n!! SEED-VARIANCE FLAGS (seed_std > 5x cohort median) !!", flush=True)
        print("   Drill into these cells per-seed BEFORE quoting their aggregates.", flush=True)
        for f in seed_var_flags:
            print(f"   [{f['cohort']:<6}] fold={f['fold']:>2}  run={f['run']:<10}  "
                  f"seed_std={f['seed_std']:.4f}  ratio={f['ratio_vs_cohort_median']:.1f}x",
                  flush=True)
    else:
        print("\nSeed-variance gate: no flagged cells (all seed_std ≤ 5x cohort median).",
              flush=True)

    # --- Aggregations: mean AND median across seeds per (fold, run) ---
    agg_kwargs: dict[str, tuple[str, object]] = {}
    for col in cohort_ll.values():
        agg_kwargs[f"{col}_mean"] = (col, "mean")
        agg_kwargs[f"{col}_median"] = (col, "median")
        agg_kwargs[f"{col}_seedstd"] = (col, lambda s: float(np.nanstd(s, ddof=1)))
    fold_run = df_rows.groupby(
        ["fold", "regime", "run"], as_index=False
    ).agg(**agg_kwargs)

    # Delta vs baseline per fold, computed for BOTH mean and median.
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
    print("\n=== Aggregate per run (MEDIAN seed-agg headline; MEAN shown alongside) ===",
          flush=True)
    print("    Cohorts: all=full val; h25=top-quartile baseline ll; "
          "h10=top-decile; lated=pct≥0.9 ∧ 5dΔ≤-2c", flush=True)
    print(f"    {'run':<10}  {'Δh25 (med)':>12}  {'Δh25 (mean)':>12}  "
          f"{'Δh10 med':>10}  {'Δlated med':>11}  {'helps_h25':>10}  "
          f"{'norm_med':>9}  {'shock_med':>9}", flush=True)
    for run_name in RUNS:
        sub = fold_run[fold_run["run"] == run_name]
        n_folds = len(sub)
        d_h25_med = sub["delta_ll_hard25_median"].to_numpy()
        d_h25_mean = sub["delta_ll_hard25_mean"].to_numpy()
        d_h10_med = sub["delta_ll_hard10_median"].to_numpy()
        d_lat_med = sub["delta_ll_lated_median"].to_numpy()
        d_all_med = sub["delta_ll_all_median"].to_numpy()
        norm_h25 = sub.loc[sub["regime"] == "normal", "delta_ll_hard25_median"].to_numpy()
        shock_h25 = sub.loc[sub["regime"] == "shock", "delta_ll_hard25_median"].to_numpy()
        if run_name == "R0":
            print(f"    {run_name:<10}  baseline reference (median ll_h25 across folds = "
                  f"{float(np.nanmedian(sub['ll_hard25_median'])):.4f})", flush=True)
        else:
            n_lat_valid = int(np.isfinite(d_lat_med).sum())
            lat_med = float(np.nanmean(d_lat_med)) if n_lat_valid > 0 else float("nan")
            print(f"    {run_name:<10}  "
                  f"{float(d_h25_med.mean()):>+12.4f}  "
                  f"{float(d_h25_mean.mean()):>+12.4f}  "
                  f"{float(d_h10_med.mean()):>+10.4f}  "
                  f"{lat_med:>+11.4f}  "
                  f"{(d_h25_med < 0).sum():>4}/{n_folds:<5}  "
                  f"{float(norm_h25.mean()) if len(norm_h25) else float('nan'):>+9.4f}  "
                  f"{float(shock_h25.mean()) if len(shock_h25) else float('nan'):>+9.4f}",
                  flush=True)
        summary.append({
            "run": run_name,
            "n_folds": n_folds,
            "delta_all_median_mean": float(np.nanmean(d_all_med)) if n_folds else None,
            "delta_hard25_median_mean": float(np.nanmean(d_h25_med)) if n_folds else None,
            "delta_hard25_mean_mean": float(np.nanmean(d_h25_mean)) if n_folds else None,
            "delta_hard10_median_mean": float(np.nanmean(d_h10_med)) if n_folds else None,
            "delta_lated_median_mean":
                float(np.nanmean(d_lat_med))
                if int(np.isfinite(d_lat_med).sum()) > 0 else None,
            "delta_lated_n_folds_valid": int(np.isfinite(d_lat_med).sum()),
            "helps_hard25_n": int((d_h25_med < 0).sum()),
            "helps_hard25_n_folds": n_folds,
            "delta_hard25_normal_median_mean":
                float(np.nanmean(norm_h25)) if len(norm_h25) else None,
            "delta_hard25_shock_median_mean":
                float(np.nanmean(shock_h25)) if len(shock_h25) else None,
        })

    # --- Elongation-conditional diagnostic ---
    # Per #212: informational, not gating. Pearson r per non-baseline run
    # between per-fold elongation and per-fold delta_hard25_median.
    print("\n=== Elongation-conditional diagnostic (informational) ===", flush=True)
    print("    Pearson r(elongation, delta_hard25_median) across folds.", flush=True)
    print("    Negative r = more elongation correlates with helping more "
          "(opposite of original hypothesis).", flush=True)
    elong_corr: dict[str, float | None] = {}
    elong_df_merged = elong_df.merge(
        fold_run[fold_run["run"] != "R0"][["fold", "run", "delta_ll_hard25_median"]],
        on="fold",
    )
    for run_name in [r for r in RUNS if r != "R0"]:
        sub = elong_df_merged[elong_df_merged["run"] == run_name][
            ["elongation", "delta_ll_hard25_median"]
        ].dropna()
        if len(sub) < 3:
            elong_corr[run_name] = None
            print(f"    {run_name:<10}  n={len(sub)} (too few)", flush=True)
            continue
        r = float(sub.corr().iloc[0, 1])
        elong_corr[run_name] = r
        print(f"    {run_name:<10}  r={r:+.3f}  n={len(sub)}", flush=True)

    # --- Meta ---
    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "comp_band_cents": COMP_BAND_CENTS,
        "delta_lag_days": DELTA_LAG_DAYS,
        "elongation_baseline_window_days": BASELINE_WINDOW_DAYS,
        "n_baseline_features": len(baseline_cols),
        "new_feature_columns": NEW_COLS,
        "run_grid": dict(RUNS),
        "cohort_definitions": {
            "all": "full val set",
            "hard25": "top quartile baseline per-row log-loss per fold",
            "hard10": "top decile baseline per-row log-loss per fold",
            "lated": "cycle_pct_through ≥ 0.9 AND _px_5d_change ≤ -2.0c",
        },
        "aggregation_convention": (
            "Headline = median across 5 seeds per (fold, run); "
            "summary then averages those medians across 14 folds. "
            "Mean shown alongside for #212's mean+median requirement."
        ),
        "seed_variance_gate": {
            "rule": "ratio = seed_std / median(seed_std across cohort cells); flag > 5x",
            "per_cohort": seed_var_summary,
            "flagged_cells": seed_var_flags,
        },
        "elongation_corr_per_run": elong_corr,
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
