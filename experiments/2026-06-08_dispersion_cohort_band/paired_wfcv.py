"""Dispersion-cohort band ablation — issue #219.

Compares the current ±5c competitive cohort (set by #212, landed in PR #217)
against the canonical ±10c Competitive band (boundary defined by
``classify.PREMIUM_BAND_CENTS``). Decides whether to widen
``COMP_BAND_CENTS`` in ``fuel_signal/features.py`` to match the canonical
classification.

The two thresholds operate on the same column: ``stickiness_score`` in
features.csv is ``median_premium_decicents / 10`` (cents), so
``|stickiness_score| <= 10`` selects exactly ``sc.class = 'Competitive'``.

Run grid (3 runs):

  R0    baseline (50-feat Phase 4)
  R1    + network_px_std (±5c cohort) + network_px_std_delta_3d (±5c)
  R2    + network_px_std (±10c cohort) + network_px_std_delta_3d (±10c)

3 runs x 14 folds x 5 seeds = 210 LightGBM fits.

Headline: MEDIAN seed-aggregation of delta_ll_hard25 across folds (per
``feedback_check_seed_variance_before_trusting_mean``). Mean shown alongside.

Decision rule (issue #219):
  - Switch to ±10c if  delta_h25(R2) <= delta_h25(R1) + 0.005  (parsimony).
  - Keep ±5c          if  delta_h25(R2) >  delta_h25(R1) + 0.005.
  - Keep ±5c + document if results within noise.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-08_dispersion_cohort_band/paired_wfcv.py \\
    2>&1 | tee experiments/2026-06-08_dispersion_cohort_band/run.log
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

# Cohort thresholds under test.
TIGHT_BAND_CENTS = 5.0   # current production (set by #212, landed via PR #217)
CANONICAL_BAND_CENTS = 10.0  # PREMIUM_BAND_CENTS — sc.class = 'Competitive'
DELTA_LAG_DAYS = 3

# Per-run column suffixes — kept distinct so they coexist in the same DataFrame.
A_LEVEL_R1 = "network_px_std_5c"
A_DELTA_R1 = "network_px_std_delta_3d_5c"
A_LEVEL_R2 = "network_px_std_10c"
A_DELTA_R2 = "network_px_std_delta_3d_10c"

NEW_COLS = [A_LEVEL_R1, A_DELTA_R1, A_LEVEL_R2, A_DELTA_R2]

RUNS: dict[str, list[str]] = {
    "R0": [],
    "R1_5c": [A_LEVEL_R1, A_DELTA_R1],
    "R2_10c": [A_LEVEL_R2, A_DELTA_R2],
}


def _add_band_pair(df: pd.DataFrame, band_cents: float,
                   level_col: str, delta_col: str) -> pd.DataFrame:
    """Attach (level, delta) network-dispersion columns for one cohort band.

    PIT-safe: stickiness_score in features.csv is the per-row (per-station,
    per-date) snapshot value, so the cohort at date D contains only rows
    whose station_class as-of D fell within ±band_cents.
    """
    comp = df[df["stickiness_score"].abs() <= band_cents]
    level_by_date = comp.groupby("price_date")["station_price_cents"].std().rename(level_col)
    df = df.join(level_by_date, on="price_date")

    per_date_level = (
        df.drop_duplicates("price_date")
        .set_index("price_date")[level_col]
        .sort_index()
    )
    full_idx = pd.date_range(per_date_level.index.min(),
                             per_date_level.index.max(), freq="D")
    s = per_date_level.reindex(full_idx)
    delta = (s - s.shift(DELTA_LAG_DAYS)).rename(delta_col)
    df = df.join(delta, on="price_date")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute both band variants of the A pair + the `_px_5d_change`
    diagnostic for the ``lated`` cohort mask."""
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])

    df = _add_band_pair(df, TIGHT_BAND_CENTS, A_LEVEL_R1, A_DELTA_R1)
    df = _add_band_pair(df, CANONICAL_BAND_CENTS, A_LEVEL_R2, A_DELTA_R2)

    # Cohort-size sanity check: ±10c should be a strict superset of ±5c.
    n_5c = int((df["stickiness_score"].abs() <= TIGHT_BAND_CENTS).sum())
    n_10c = int((df["stickiness_score"].abs() <= CANONICAL_BAND_CENTS).sum())
    print(f"    cohort rows: ±5c={n_5c:,}  ±10c={n_10c:,}  "
          f"(ratio {n_10c / max(n_5c, 1):.2f}x)", flush=True)
    if n_10c < n_5c:
        raise ValueError(
            f"Cohort sanity check failed: ±10c={n_10c} < ±5c={n_5c}. "
            "Expected ±10c to be a strict superset of ±5c."
        )

    # Diagnostic for `lated` cohort mask (NOT a feature).
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


def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading features ...", flush=True)
    t0 = time.perf_counter()
    df = load_features()
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}", flush=True)

    print("Computing band-variant features ...", flush=True)
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
          f"{'val_rows':>8}  {'run':<8}  {'seed':>4}  "
          f"{'ll_all':>7}  {'ll_h25':>7}  {'ll_h10':>7}  {'ll_lated':>8}  "
          f"{'fit_s':>6}", flush=True)
    print("-" * 126, flush=True)

    rows: list[dict] = []

    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        vd = pd.to_datetime(val_df["price_date"])
        val_start = vd.min()
        val_end = vd.max()
        regime = "shock" if i in SHOCK_FOLDS else "normal"
        y = val_df["label"].to_numpy(dtype=int)

        # Fold-stable hard-cohort masks: derived from baseline+seed42 per-row
        # log-loss. All runs scored on the same mask per fold.
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
                      f"{len(val_df):>8,}  {run_name:<8}  {seed:>4}  "
                      f"{ll:>7.4f}  {ll_hard25:>7.4f}  {ll_hard10:>7.4f}  "
                      f"{ll_lated:>8.4f}  {t:>5.1f}s", flush=True)

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "runs.csv", index=False)
    print(f"\nPer-(fold,run,seed) results: {OUT / 'runs.csv'}", flush=True)

    # --- Seed-variance gate ---
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
        print("   Drill into these cells per-seed BEFORE quoting their aggregates.",
              flush=True)
        for f in seed_var_flags:
            print(f"   [{f['cohort']:<6}] fold={f['fold']:>2}  run={f['run']:<8}  "
                  f"seed_std={f['seed_std']:.4f}  ratio={f['ratio_vs_cohort_median']:.1f}x",
                  flush=True)
    else:
        print("\nSeed-variance gate: no flagged cells (all seed_std <= 5x cohort median).",
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
          "h10=top-decile; lated=pct>=0.9 AND 5d-change<=-2c", flush=True)
    print(f"    {'run':<8}  {'Δh25 (med)':>12}  {'Δh25 (mean)':>12}  "
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
            print(f"    {run_name:<8}  baseline reference (median ll_h25 across folds = "
                  f"{float(np.nanmedian(sub['ll_hard25_median'])):.4f})", flush=True)
        else:
            n_lat_valid = int(np.isfinite(d_lat_med).sum())
            lat_med = float(np.nanmean(d_lat_med)) if n_lat_valid > 0 else float("nan")
            print(f"    {run_name:<8}  "
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

    # --- Decision rule ---
    # Δh25 is negative-when-helpful, so "R2 within ~0.005 of R1's lift, or
    # better" is the parsimony-favoured switch condition.
    r1 = next(s for s in summary if s["run"] == "R1_5c")
    r2 = next(s for s in summary if s["run"] == "R2_10c")
    d1 = r1["delta_hard25_median_mean"]
    d2 = r2["delta_hard25_median_mean"]
    margin = d2 - d1  # negative => R2 helps more than R1; positive => R2 worse
    if margin <= 0.005:
        verdict = "SWITCH_TO_10C"
        reasoning = (
            f"R2 (±10c) Δh25={d2:+.4f} <= R1 (±5c) Δh25={d1:+.4f} + 0.005 "
            f"(margin={margin:+.4f}). Parsimony favours canonical cohort."
        )
    else:
        verdict = "KEEP_5C"
        reasoning = (
            f"R2 (±10c) Δh25={d2:+.4f} > R1 (±5c) Δh25={d1:+.4f} + 0.005 "
            f"(margin={margin:+.4f}). Canonical cohort underperforms; keep ±5c."
        )
    print(f"\n=== Decision: {verdict} ===", flush=True)
    print(f"    {reasoning}", flush=True)

    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "tight_band_cents": TIGHT_BAND_CENTS,
        "canonical_band_cents": CANONICAL_BAND_CENTS,
        "delta_lag_days": DELTA_LAG_DAYS,
        "n_baseline_features": len(baseline_cols),
        "new_feature_columns": NEW_COLS,
        "run_grid": dict(RUNS),
        "cohort_definitions": {
            "all": "full val set",
            "hard25": "top quartile baseline per-row log-loss per fold",
            "hard10": "top decile baseline per-row log-loss per fold",
            "lated": "cycle_pct_through >= 0.9 AND _px_5d_change <= -2.0c",
        },
        "aggregation_convention": (
            "Headline = median across 5 seeds per (fold, run); summary "
            "then averages those medians across 14 folds. Mean alongside."
        ),
        "seed_variance_gate": {
            "rule": "ratio = seed_std / median(seed_std across cohort cells); flag > 5x",
            "per_cohort": seed_var_summary,
            "flagged_cells": seed_var_flags,
        },
        "summary": summary,
        "decision": {"verdict": verdict, "reasoning": reasoning,
                     "margin_r2_minus_r1": margin},
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
