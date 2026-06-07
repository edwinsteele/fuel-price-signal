"""Step 2: paired walk-forward CV for the late-descent triplet (A + B + C),
with full attribution grid (drop-one + standalone).

8 runs:
  R0 baseline   — 50-feat Phase 4 baseline
  R1 ABC        — baseline + all 6 triplet features
  R2 drop_A     — baseline + B + C
  R3 drop_B     — baseline + A + C
  R4 drop_C     — baseline + A + B
  R5 A_only     — baseline + A
  R6 B_only     — baseline + B
  R7 C_only     — baseline + C

Per fold per run: 5 seeds, paired (same fold split).

Reports:
- Overall log-loss per (run, fold, seed); aggregates per run.
- Hard-cohort log-loss (top quartile baseline per-row log-loss per fold) —
  primary empirical-labelling cut per #206.
- Shock-fold (predecessor taxonomy: {1, 4, 9, 13}) supplementary cut.

Usage: PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step2_paired_wfcv.py
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, load_features

OUT = pathlib.Path(__file__).parent
SEEDS = (42, 43, 44, 45, 46)
SHOCK_FOLDS = frozenset({1, 4, 9, 13})

# Cohort thresholds for A/B (see README; provisional pending #207).
COMP_BAND_CENTS = 5.0   # competitive: |stickiness_score| <= 5
DISC_THRESH = -5.0      # discount: stickiness_score < -5
DELTA_LAG_DAYS = 3      # for the *_delta_3d features

# Brand `days_since_trough_entry_*` cols are in features.csv but not in baseline.
_BRANDS = {"7_eleven", "ampol_foodary", "bp", "budget", "eg_ampol",
           "independent", "metro_fuel", "reddy_express", "shell", "speedway"}

# --- New feature columns added by this experiment ---
A_LEVEL = "network_px_std"
A_DELTA = "network_px_std_delta_3d"
B_LEVEL = "network_disc_gap"
B_DELTA = "network_disc_gap_delta_3d"
C_LEVEL = "lga_phase_std"
C_DELTA = "lga_phase_std_delta_3d"

A_COLS = [A_LEVEL, A_DELTA]
B_COLS = [B_LEVEL, B_DELTA]
C_COLS = [C_LEVEL, C_DELTA]
NEW_COLS = A_COLS + B_COLS + C_COLS

# --- Run grid: name → list of triplet columns added to baseline ---
RUNS: dict[str, list[str]] = {
    "R0_baseline":   [],
    "R1_ABC":        A_COLS + B_COLS + C_COLS,
    "R2_drop_A":              B_COLS + C_COLS,
    "R3_drop_B":     A_COLS          + C_COLS,
    "R4_drop_C":     A_COLS + B_COLS,
    "R5_A_only":     A_COLS,
    "R6_B_only":     B_COLS,
    "R7_C_only":     C_COLS,
}


def compute_triplet_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 6 new features. PIT-safe: each per-date stat uses only
    same-date rows; deltas use prior-date stats. Returns df with new columns
    joined."""
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])

    # --- Signal A: cross-station dispersion within competitive cohort ---
    comp = df[df["stickiness_score"].abs() <= COMP_BAND_CENTS]
    a_by_date = comp.groupby("price_date")["station_price_cents"].std().rename(A_LEVEL)
    df = df.join(a_by_date, on="price_date")

    # --- Signal B: comp_median − discount_median per date ---
    comp_med = df[df["stickiness_score"].abs() <= COMP_BAND_CENTS] \
        .groupby("price_date")["station_price_cents"].median()
    disc_med = df[df["stickiness_score"] < DISC_THRESH] \
        .groupby("price_date")["station_price_cents"].median()
    b_by_date = (comp_med - disc_med).rename(B_LEVEL)
    df = df.join(b_by_date, on="price_date")

    # --- Signal C: std of days_since_trough across 35 LGAs ---
    lga_cols = [c for c in df.columns
                if c.startswith("days_since_trough_entry_")
                and c.replace("days_since_trough_entry_", "") not in _BRANDS]
    if len(lga_cols) != 35:
        # If LGA set changes upstream (e.g. via #207) the C signal definition
        # needs reviewing — bail early with a clear message rather than mid-run.
        raise ValueError(
            f"Expected 35 LGA columns for Signal C; found {len(lga_cols)}. "
            f"Cols: {sorted(lga_cols)}. "
            "If the LGA set has legitimately changed, review the C signal "
            "definition (lga_phase_std semantics may shift) before re-running."
        )
    per_date = df.drop_duplicates("price_date").set_index("price_date")[lga_cols]
    c_by_date = per_date.std(axis=1).rename(C_LEVEL)
    df = df.join(c_by_date, on="price_date")

    # --- Deltas (level minus value DELTA_LAG_DAYS days prior) ---
    # Per-date series → shift by DELTA_LAG_DAYS calendar days, not row positions.
    for level_col, delta_col in [(A_LEVEL, A_DELTA), (B_LEVEL, B_DELTA), (C_LEVEL, C_DELTA)]:
        per_date_level = df.drop_duplicates("price_date").set_index("price_date")[level_col].sort_index()
        # Calendar-aware shift: reindex by date, then subtract value at date − lag.
        # Since the series is dated daily (with possible gaps), use shift on a
        # complete date range to enforce calendar lag.
        full_idx = pd.date_range(per_date_level.index.min(), per_date_level.index.max(), freq="D")
        s = per_date_level.reindex(full_idx)
        delta = (s - s.shift(DELTA_LAG_DAYS)).rename(delta_col)
        df = df.join(delta, on="price_date")

    # --- Diagnostic column (NOT a feature): per-station 5d backward price change ---
    # Used only for the "true late descent" cohort mask in main().
    # Calendar-aware: for each row, look up the same station's price exactly 5
    # calendar days earlier. NaN when no observation at that exact lookup date
    # (e.g. across a fill-pipeline gap > max_gap_days). Row-positional .diff(5)
    # would silently span >5 calendar days across unfilled gaps and corrupt
    # the `lated` cohort definition.
    lookup = df[["station_code", "price_date", "station_price_cents"]].rename(
        columns={"price_date": "_lookup_date", "station_price_cents": "_px_5d_ago"}
    )
    df["_lookup_date"] = df["price_date"] - pd.Timedelta(days=5)
    # validate="m:1": fail loudly if (station_code, price_date) is ever
    # non-unique in features.csv, which would silently row-explode the merge.
    df = df.merge(lookup, on=["station_code", "_lookup_date"],
                  how="left", validate="m:1")
    df["_px_5d_change"] = df["station_price_cents"] - df["_px_5d_ago"]
    df = df.drop(columns=["_lookup_date", "_px_5d_ago"])

    return df


def fit_score(train_df: pd.DataFrame, val_df: pd.DataFrame,
              cols: list[str], seed: int) -> tuple[float, np.ndarray, float]:
    """Fit LGBM and return (log_loss_overall, per_row_predictions, fit_seconds)."""
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
    print("Loading features ...")
    t0 = time.perf_counter()
    df = load_features()
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}")

    print("Computing triplet features ...")
    t0 = time.perf_counter()
    df = compute_triplet_features(df)
    print(f"  [compute_triplet] {time.perf_counter() - t0:.1f}s")
    # Report null counts (debug — large null counts inflate variance)
    for c in NEW_COLS:
        null_pct = df[c].isna().mean() * 100
        print(f"    {c}: nulls = {null_pct:.2f}%")

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    assert len(baseline_cols) == 50, f"expected 50, got {len(baseline_cols)}"
    print(f"\nBaseline features: {len(baseline_cols)}")
    print(f"Run grid: {list(RUNS.keys())}")
    print(f"Seeds: {SEEDS} (n={len(SEEDS)})")

    folds = list(_ev.walk_forward_folds(df, train_min_days=1825, val_days=90, step_days=90))
    print(f"Walk-forward folds: {len(folds)}\n")

    print(f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  {'val_rows':>8}  {'run':<12}  "
          f"{'seed':>4}  {'ll_all':>7}  {'ll_h25':>7}  {'ll_h10':>7}  {'ll_lated':>8}  {'fit_s':>6}")
    print("-" * 130)

    rows = []
    shap_rows = []  # per-fold mean |SHAP| for R1's 6 new features (seed 42 only)
    shap_corr_rows = []  # per-fold cross-feature SHAP corr matrix for the 6 new
    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        vd = pd.to_datetime(val_df["price_date"])
        val_start = vd.min().strftime("%Y-%m-%d")
        val_end   = vd.max().strftime("%Y-%m-%d")
        regime = "shock" if i in SHOCK_FOLDS else "normal"
        y = val_df["label"].to_numpy(dtype=int)

        # First pass: baseline with seed 0 to derive the hard-cohort masks.
        # Masks are fold-stable (one seed's predictions); per-run/seed we
        # report log-loss on those fixed masks.
        ll0, p0, t0 = fit_score(train_df, val_df, baseline_cols, SEEDS[0])
        prl0 = per_row_log_loss(y, p0)
        hard25_thresh = np.quantile(prl0, 0.75)
        hard10_thresh = np.quantile(prl0, 0.90)
        hard25_mask = prl0 >= hard25_thresh
        hard10_mask = prl0 >= hard10_thresh
        # "True late descent" cohort: pct ≥ 0.9 AND prices falling 5d back ≤ -2c.
        pct = val_df["cycle_pct_through"].to_numpy(dtype=float)
        d5 = val_df["_px_5d_change"].to_numpy(dtype=float)
        lated_mask = (pct >= 0.9) & np.isfinite(d5) & (d5 <= -2.0)
        n_lated = int(lated_mask.sum())

        # Run the full grid (including R0 again with seed 0 → reused).
        for run_name, extra in RUNS.items():
            cols = baseline_cols + extra
            for seed in SEEDS:
                # Skip the (R0, SEEDS[0]) refit we already did
                if run_name == "R0_baseline" and seed == SEEDS[0]:
                    ll, p, t = ll0, p0, t0
                else:
                    ll, p, t = fit_score(train_df, val_df, cols, seed)
                prl = per_row_log_loss(y, p)
                ll_hard25 = float(prl[hard25_mask].mean())
                ll_hard10 = float(prl[hard10_mask].mean())
                ll_lated = float(prl[lated_mask].mean()) if n_lated > 0 else float("nan")
                rows.append({
                    "fold": i, "regime": regime,
                    "val_start": val_start, "val_end": val_end,
                    "val_rows": len(val_df), "n_lated": n_lated,
                    "run": run_name, "n_features": len(cols),
                    "seed": seed,
                    "ll_all": ll, "ll_hard25": ll_hard25,
                    "ll_hard10": ll_hard10, "ll_lated": ll_lated,
                    "fit_s": t,
                })
                print(f"{i:>4}  {regime:>6}  {val_start:>10}  {val_end:>10}  "
                      f"{len(val_df):>8,}  {run_name:<12}  {seed:>4}  "
                      f"{ll:>7.4f}  {ll_hard25:>7.4f}  {ll_hard10:>7.4f}  "
                      f"{ll_lated:>8.4f}  {t:>5.1f}s")

        # --- SHAP on R1 (full triplet), seed 42 only, per fold ---
        # Refit a clean R1 model for SHAP (cheap given one extra fit per fold).
        t_shap0 = time.perf_counter()
        r1_cols = baseline_cols + RUNS["R1_ABC"]
        r1_model = LGBMClassifier(random_state=SEEDS[0], verbose=-1,
                                  subsample=0.8, subsample_freq=1)
        r1_model.fit(train_df[r1_cols], train_df["label"].to_numpy(dtype=int))
        # Sample val rows for SHAP speed (50k cap).
        shap_n = min(50_000, len(val_df))
        shap_sample = val_df.sample(shap_n, random_state=SEEDS[0])[r1_cols]
        explainer = shap.TreeExplainer(r1_model)
        sv = explainer.shap_values(shap_sample)
        # lgbm binary: sv may be a list [class0, class1] or a single array.
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        # Per-feature mean |SHAP| across the sample
        mean_abs = np.abs(sv).mean(axis=0)
        for j, c in enumerate(r1_cols):
            if c in NEW_COLS:
                shap_rows.append({
                    "fold": i, "feature": c,
                    "mean_abs_shap": float(mean_abs[j]),
                })
        # Cross-feature SHAP correlations for the 6 new features (redundancy check).
        # Guard against zero-variance SHAP columns (feature never used by R1 in
        # this fold) which would otherwise produce a NaN row in corrcoef.
        new_idx = [r1_cols.index(c) for c in NEW_COLS]
        sv_new = sv[:, new_idx]
        sv_std = sv_new.std(axis=0)
        for a_i, a_c in enumerate(NEW_COLS):
            for b_i, b_c in enumerate(NEW_COLS):
                if a_i < b_i:
                    if sv_std[a_i] == 0 or sv_std[b_i] == 0:
                        # Skip pairs where either feature was never split on;
                        # correlation is undefined.
                        shap_corr_rows.append({
                            "fold": i, "feat_a": a_c, "feat_b": b_c,
                            "shap_corr": None,
                            "note": "zero-variance SHAP column",
                        })
                        continue
                    r = float(np.corrcoef(sv_new[:, a_i], sv_new[:, b_i])[0, 1])
                    shap_corr_rows.append({
                        "fold": i, "feat_a": a_c, "feat_b": b_c,
                        "shap_corr": r, "note": "",
                    })
        print(f"          ↑ SHAP for R1 done in {time.perf_counter() - t_shap0:.1f}s "
              f"(n_sample={shap_n:,})")

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "step2_runs.csv", index=False)
    print(f"\nPer-(fold,run,seed) results: {OUT / 'step2_runs.csv'}")

    # SHAP outputs
    if shap_rows:
        pd.DataFrame(shap_rows).to_csv(OUT / "step2_shap_mean_abs.csv", index=False)
        pd.DataFrame(shap_corr_rows).to_csv(OUT / "step2_shap_corr.csv", index=False)
        print(f"SHAP outputs: step2_shap_mean_abs.csv, step2_shap_corr.csv")

    # --- Aggregate: mean across seeds per (fold, run), then deltas vs baseline ---
    seed_mean = df_rows.groupby(["fold", "regime", "run"], as_index=False).agg(
        ll_all=("ll_all", "mean"),
        ll_hard25=("ll_hard25", "mean"),
        ll_hard10=("ll_hard10", "mean"),
        ll_lated=("ll_lated", "mean"),
        ll_all_std=("ll_all", "std"),
        ll_hard25_std=("ll_hard25", "std"),
    )
    base_per_fold = seed_mean[seed_mean["run"] == "R0_baseline"][
        ["fold", "ll_all", "ll_hard25", "ll_hard10", "ll_lated"]
    ].rename(columns={"ll_all": "ll_all_base",
                      "ll_hard25": "ll_hard25_base",
                      "ll_hard10": "ll_hard10_base",
                      "ll_lated": "ll_lated_base"})
    seed_mean = seed_mean.merge(base_per_fold, on="fold")
    seed_mean["delta_all"]    = seed_mean["ll_all"]    - seed_mean["ll_all_base"]
    seed_mean["delta_hard25"] = seed_mean["ll_hard25"] - seed_mean["ll_hard25_base"]
    seed_mean["delta_hard10"] = seed_mean["ll_hard10"] - seed_mean["ll_hard10_base"]
    seed_mean["delta_lated"]  = seed_mean["ll_lated"]  - seed_mean["ll_lated_base"]
    seed_mean.to_csv(OUT / "step2_fold_run.csv", index=False)

    print("\n=== Aggregate per run (mean Δ vs baseline across folds, after seed-averaging) ===")
    print("    Cohorts: all=full val; h25=top-quartile baseline ll; h10=top-decile baseline ll; lated=pct≥0.9 ∧ 5dΔ≤-2c")
    summary = []
    for run_name in RUNS:
        sub = seed_mean[seed_mean["run"] == run_name]
        if run_name == "R0_baseline":
            print(f"  {run_name:<12}  ll_all={sub['ll_all'].mean():.4f}  "
                  f"ll_h25={sub['ll_hard25'].mean():.4f}  "
                  f"ll_h10={sub['ll_hard10'].mean():.4f}  "
                  f"ll_lated={sub['ll_lated'].mean():.4f}  n_folds={len(sub)}")
            summary.append({
                "run": run_name,
                "n_folds": len(sub),
                "ll_all_mean": float(sub["ll_all"].mean()),
                "ll_hard25_mean": float(sub["ll_hard25"].mean()),
                "ll_hard10_mean": float(sub["ll_hard10"].mean()),
                "ll_lated_mean": float(sub["ll_lated"].mean()),
            })
            continue
        # Deltas
        d_all = sub["delta_all"].to_numpy()
        d_h25 = sub["delta_hard25"].to_numpy()
        d_h10 = sub["delta_hard10"].to_numpy()
        d_lat = sub["delta_lated"].to_numpy()  # may contain NaN for folds with 0 lated rows
        # Per-regime (hard25 only, for compact print)
        d_h25_norm = sub.loc[sub["regime"] == "normal", "delta_hard25"].to_numpy()
        d_h25_shock = sub.loc[sub["regime"] == "shock", "delta_hard25"].to_numpy()
        # Seed adequacy: compare mean Δ_h25 magnitude to seed std (typical)
        seed_std_typ = float(sub["ll_hard25_std"].mean())
        # Use nan-safe stats for d_lat — folds with zero lated rows produce NaN.
        n_lat_valid = int(np.isfinite(d_lat).sum())
        lat_mean = float(np.nanmean(d_lat)) if n_lat_valid > 0 else None
        print(f"  {run_name:<12}  "
              f"Δall={d_all.mean():+.4f}  "
              f"Δh25={d_h25.mean():+.4f}±{d_h25.std(ddof=1):.4f}  "
              f"Δh10={d_h10.mean():+.4f}  "
              f"Δlated={'NA' if lat_mean is None else f'{lat_mean:+.4f}'}({n_lat_valid}/{len(d_lat)} folds)  "
              f"helps_h25={(d_h25 < 0).sum()}/{len(d_h25)}  "
              f"norm_h25={d_h25_norm.mean():+.4f}  shock_h25={d_h25_shock.mean():+.4f}  "
              f"seed_σ_h25={seed_std_typ:.4f}")
        summary.append({
            "run": run_name,
            "n_folds": len(sub),
            "delta_all_mean":  float(d_all.mean()),
            "delta_all_std":   float(d_all.std(ddof=1)) if len(d_all) > 1 else 0.0,
            "delta_hard25_mean": float(d_h25.mean()),
            "delta_hard25_std":  float(d_h25.std(ddof=1)) if len(d_h25) > 1 else 0.0,
            "delta_hard10_mean": float(d_h10.mean()),
            "delta_lated_mean":  lat_mean,
            "delta_lated_n_folds_valid": n_lat_valid,
            "delta_hard25_helps_n": int((d_h25 < 0).sum()),
            "delta_hard25_n":      int(len(d_h25)),
            "delta_hard25_normal_mean": float(d_h25_norm.mean()) if len(d_h25_norm) else None,
            "delta_hard25_shock_mean":  float(d_h25_shock.mean()) if len(d_h25_shock) else None,
            "mean_seed_std_hard25":     seed_std_typ,
        })

    # --- Attribution per family per cohort ---
    # Single sign convention: `improvement` = log-loss reduction when this
    # family is present. Always positive = good. Eliminates the standalone vs
    # marginal sign mismatch flagged in code review.
    def get_mean(run: str, col: str) -> float:
        sub = seed_mean[seed_mean["run"] == run]
        if len(sub) == 0:
            raise KeyError(f"Run {run!r} missing from seed_mean — cannot compute attribution.")
        arr = sub[col].to_numpy()
        # Guard the all-NaN case (lated cohort can be empty across all folds)
        # so np.nanmean doesn't emit "Mean of empty slice" noise.
        if not np.isfinite(arr).any():
            return float("nan")
        return float(np.nanmean(arr))

    def attribution_for(cohort: str, ll_col: str) -> dict:
        ll_R0 = get_mean("R0_baseline", ll_col)
        ll_R1 = get_mean("R1_ABC", ll_col)
        return {
            "_convention": "positive = log-loss REDUCTION when family is present",
            "A": {
                "standalone_improvement": ll_R0 - get_mean("R5_A_only", ll_col),
                "marginal_improvement":   get_mean("R2_drop_A", ll_col) - ll_R1,
            },
            "B": {
                "standalone_improvement": ll_R0 - get_mean("R6_B_only", ll_col),
                "marginal_improvement":   get_mean("R3_drop_B", ll_col) - ll_R1,
            },
            "C": {
                "standalone_improvement": ll_R0 - get_mean("R7_C_only", ll_col),
                "marginal_improvement":   get_mean("R4_drop_C", ll_col) - ll_R1,
            },
        }

    attribution_all = attribution_for("all", "ll_all")
    attribution_h25 = attribution_for("hard25", "ll_hard25")
    attribution_h10 = attribution_for("hard10", "ll_hard10")
    attribution_lat = attribution_for("lated", "ll_lated")

    print("\n=== Attribution: standalone (R5–R7 vs R0) vs marginal-given-others (drop-one) ===")
    print("    Convention: positive = log-loss REDUCTION when family is present (i.e. family helps).")
    print(f"    {'Cohort':<8}  {'Family':<2}  {'standalone +':>14}  {'marginal +':>14}")
    for cohort_label, attr in [("all", attribution_all), ("hard25", attribution_h25),
                               ("hard10", attribution_h10), ("lated", attribution_lat)]:
        for fam in ("A", "B", "C"):
            s = attr[fam]["standalone_improvement"]
            m = attr[fam]["marginal_improvement"]
            print(f"    {cohort_label:<8}  {fam:<2}  {s:+14.4f}  {m:+14.4f}")

    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "comp_band_cents": COMP_BAND_CENTS,
        "discount_threshold": DISC_THRESH,
        "delta_lag_days": DELTA_LAG_DAYS,
        "n_baseline_features": len(baseline_cols),
        "new_feature_columns": NEW_COLS,
        "run_grid": {k: v for k, v in RUNS.items()},
        "cohort_definitions": {
            "all": "full val set",
            "hard25": "top quartile baseline per-row log-loss per fold",
            "hard10": "top decile baseline per-row log-loss per fold",
            "lated": "cycle_pct_through ≥ 0.9 AND _px_5d_change ≤ -2.0c (true late descent)",
        },
        "summary": summary,
        "attribution_all": attribution_all,
        "attribution_hard25": attribution_h25,
        "attribution_hard10": attribution_h10,
        "attribution_lated": attribution_lat,
        "total_wall_seconds": time.perf_counter() - overall_t0,
    }
    # Recursive NaN → None so the JSON is strict-valid for any consumer.
    def _to_jsonable(o):
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(x) for x in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    (OUT / "step2_meta.json").write_text(json.dumps(_to_jsonable(meta), indent=2, default=str))
    print(f"\nMeta: {OUT / 'step2_meta.json'}")
    print(f"[total wall] {time.perf_counter() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
