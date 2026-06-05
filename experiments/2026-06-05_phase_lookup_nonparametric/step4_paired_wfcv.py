"""Step 4: paired walk-forward CV for the non-parametric phase-lookup feature.

Three configs (mirrors predecessor experiments/2026-06-04_cycle_pct_through_interaction
step4 protocol), same single seed per fold (paired CV; fold-to-fold variance dominates
seed variance for this question):
- baseline: 50 Phase 4 features.
- additive: + station_minus_expected_phase_lookup (51).
- ablationA: + station_minus_expected_phase_lookup
              − station_minus_last_min_cents
              − station_minus_last_max_cents          (49).

Difference vs predecessor step 4: the engineered feature is built from a
non-parametric lookup table E[norm_price | cycle_pct_through] **refit per fold
on that fold's train data only** (leakage-safe). Replaces the closed-form
linear-interp formula `last_min + pct × (last_max − last_min)` that was
shown to be shape-inverted (see README and project_cycle_pct_through_semantics).

Bin design: equal-width, 30 bins of width 0.05 over [0.0, 1.5]. pct clipped at 1.5.

Empty-bin handling: NaN propagation (LightGBM learns default direction).
Zero-amplitude rows: excluded at fit; NaN at apply.

Per-fold report follows feedback_regime_segmented_evaluation: shock-fold
taxonomy pre-committed in the README. Aggregates reported per normal/shock
split as well as overall.
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
SEED = 42
REGRESSION_THRESHOLD = 0.005

NEW_COL = "station_minus_expected_phase_lookup"
DROP_FOR_ABLATION_A = (
    "station_minus_last_min_cents",
    "station_minus_last_max_cents",
)

N_BINS = 30
PCT_CLIP = 1.5
BIN_EDGES = np.linspace(0.0, PCT_CLIP, N_BINS + 1)

# Pre-committed shock-fold taxonomy (README § Pre-committed shock-fold taxonomy).
SHOCK_FOLDS = frozenset({1, 4, 9, 13})


def fit_lookup(train_df: pd.DataFrame) -> np.ndarray:
    """Return length-N_BINS array of bin means of price-position.

    Price-position = (station_price − last_min) / (last_max − last_min).
    Zero-amplitude rows excluded. Empty bins → NaN.
    """
    pct = np.clip(train_df["cycle_pct_through"].to_numpy(dtype=float), 0.0, PCT_CLIP)
    last_min = train_df["cycle_last_min_cents"].to_numpy(dtype=float)
    last_max = train_df["cycle_last_max_cents"].to_numpy(dtype=float)
    amp = last_max - last_min
    station = train_df["station_price_cents"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (station - last_min) / amp

    valid = (amp != 0.0) & np.isfinite(norm)
    bin_idx = np.clip(np.digitize(pct, BIN_EDGES) - 1, 0, N_BINS - 1)

    sums = np.bincount(bin_idx[valid], weights=norm[valid], minlength=N_BINS)
    counts = np.bincount(bin_idx[valid], minlength=N_BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return means


def apply_lookup(df: pd.DataFrame, lookup: np.ndarray) -> np.ndarray:
    """Return the feature column: station_price − expected_price (cents).

    NaN where bin is empty in the fitted lookup OR row amplitude is zero.
    """
    pct = np.clip(df["cycle_pct_through"].to_numpy(dtype=float), 0.0, PCT_CLIP)
    last_min = df["cycle_last_min_cents"].to_numpy(dtype=float)
    last_max = df["cycle_last_max_cents"].to_numpy(dtype=float)
    amp = last_max - last_min
    station = df["station_price_cents"].to_numpy(dtype=float)

    bin_idx = np.clip(np.digitize(pct, BIN_EDGES) - 1, 0, N_BINS - 1)
    expected_norm = lookup[bin_idx]  # NaN if bin was empty in train
    expected = last_min + expected_norm * amp
    feature = station - expected
    feature[amp == 0.0] = np.nan
    return feature


def fit_score(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    t0 = time.perf_counter()
    model = LGBMClassifier(random_state=SEED, verbose=-1, subsample=0.8, subsample_freq=1)
    # Pass DataFrames (not numpy) so LightGBM stores real feature names at fit
    # and matches them at predict — avoids sklearn's "no valid feature names" warning.
    model.fit(train_df[cols], train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols])[:, 1]
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, time.perf_counter() - t0


def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading features (load_features) …")
    t0 = time.perf_counter()
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}")

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    additive_cols = baseline_cols + [NEW_COL]
    ablation_cols = [c for c in additive_cols if c not in DROP_FOR_ABLATION_A]
    assert len(baseline_cols) == 50, f"expected 50 baseline features, got {len(baseline_cols)}"
    assert len(additive_cols) == 51
    assert len(ablation_cols) == 49

    folds = list(_ev.walk_forward_folds(df, train_min_days=1825, val_days=90, step_days=90))
    print(f"Walk-forward folds: {len(folds)}  seed={SEED}")
    print(f"Configs: baseline={len(baseline_cols)}  additive={len(additive_cols)}  ablationA={len(ablation_cols)}")
    print(f"Lookup: {N_BINS} equal-width bins over [0.0, {PCT_CLIP}]  shock_folds={sorted(SHOCK_FOLDS)}\n")

    print(f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  {'val_rows':>8}  {'BUY%':>5}  "
          f"{'ll_base':>7}  {'ll_add':>7}  {'ll_abl':>7}  "
          f"{'Δ_add':>8}  {'Δ_abl':>8}  {'Δ_abl-add':>10}  {'fit_s':>7}  {'lkp_s':>6}")
    print("-" * 140)

    rows = []
    fold_lookups: list[dict] = []  # one entry per fold for diagnostic CSV
    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue

        # Per-fold lookup fit (leakage-safe: train only).
        t_lkp0 = time.perf_counter()
        lookup = fit_lookup(train_df)
        # Apply to BOTH train and val so the engineered feature is available to fit_score.
        train_df = train_df.assign(**{NEW_COL: apply_lookup(train_df, lookup)})
        val_df = val_df.assign(**{NEW_COL: apply_lookup(val_df, lookup)})
        t_lkp = time.perf_counter() - t_lkp0

        # Record the lookup curve (debug).
        for bin_i, m in enumerate(lookup):
            fold_lookups.append({
                "fold": i,
                "bin_idx": bin_i,
                "bin_left": float(BIN_EDGES[bin_i]),
                "bin_right": float(BIN_EDGES[bin_i + 1]),
                "mean_norm_price": float(m) if np.isfinite(m) else None,
            })

        ll_b, t_b = fit_score(train_df, val_df, baseline_cols)
        ll_a, t_a = fit_score(train_df, val_df, additive_cols)
        ll_x, t_x = fit_score(train_df, val_df, ablation_cols)
        d_a = ll_a - ll_b
        d_x = ll_x - ll_b
        d_xa = ll_x - ll_a
        vd = pd.to_datetime(val_df["price_date"])
        regime = "shock" if i in SHOCK_FOLDS else "normal"
        row = {
            "fold": i,
            "regime": regime,
            "val_start": vd.min().strftime("%Y-%m-%d"),
            "val_end":   vd.max().strftime("%Y-%m-%d"),
            "val_rows":  len(val_df),
            "buy_rate":  float(val_df["label"].mean()),
            "ll_baseline":  ll_b,
            "ll_additive":  ll_a,
            "ll_ablationA": ll_x,
            "delta_additive":  d_a,
            "delta_ablationA": d_x,
            "delta_abl_vs_add": d_xa,
            "fit_s_baseline":  t_b,
            "fit_s_additive":  t_a,
            "fit_s_ablationA": t_x,
            "lookup_fit_apply_s": t_lkp,
            "lookup_n_empty_bins": int(np.isnan(lookup).sum()),
        }
        rows.append(row)
        print(f"{row['fold']:>4}  {regime:>6}  {row['val_start']:>10}  {row['val_end']:>10}  "
              f"{row['val_rows']:>8,}  {row['buy_rate']*100:>4.1f}%  "
              f"{ll_b:>7.4f}  {ll_a:>7.4f}  {ll_x:>7.4f}  "
              f"{d_a:>+8.4f}  {d_x:>+8.4f}  {d_xa:>+10.4f}  "
              f"{(t_b + t_a + t_x):>6.1f}s  {t_lkp:>5.2f}s")

    if not rows:
        print("No folds produced.")
        return

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "step4_folds.csv", index=False)

    df_lookups = pd.DataFrame(fold_lookups)
    df_lookups.to_csv(OUT / "step4_fold_lookups.csv", index=False)

    print("-" * 140)

    def aggregate(name: str, col: str, mask: np.ndarray | None = None) -> dict:
        sub = df_rows if mask is None else df_rows[mask]
        v = sub[col].to_numpy()
        if v.size == 0:
            print(f"{name:<30s} (no folds)")
            return {"name": name, "n_folds": 0}
        m, med, sd = float(v.mean()), float(np.median(v)), float(v.std(ddof=1)) if v.size > 1 else 0.0
        n_help = int((v < 0).sum())
        n_hurt = int((v > 0).sum())
        print(f"{name:<30s} mean={m:+.4f}  median={med:+.4f}  std={sd:.4f}  "
              f"min={v.min():+.4f}  max={v.max():+.4f}  "
              f"helps={n_help}/{len(v)}  hurts={n_hurt}/{len(v)}")
        return {"name": name, "mean": m, "median": med, "std": sd,
                "min": float(v.min()), "max": float(v.max()),
                "n_help": n_help, "n_hurt": n_hurt, "n_folds": len(v)}

    is_shock = df_rows["regime"].eq("shock").to_numpy()
    is_normal = ~is_shock

    print("\nAggregate deltas across ALL folds (single seed, paired):")
    agg_a_all = aggregate("Δ additive − baseline   [all]", "delta_additive")
    agg_x_all = aggregate("Δ ablationA − baseline  [all]", "delta_ablationA")
    agg_xa_all = aggregate("Δ ablationA − additive [all]", "delta_abl_vs_add")

    print("\nAggregate deltas across NORMAL folds:")
    agg_a_n = aggregate("Δ additive − baseline   [normal]", "delta_additive", is_normal)
    agg_x_n = aggregate("Δ ablationA − baseline  [normal]", "delta_ablationA", is_normal)

    print("\nAggregate deltas across SHOCK folds:")
    agg_a_s = aggregate("Δ additive − baseline   [shock]", "delta_additive", is_shock)
    agg_x_s = aggregate("Δ ablationA − baseline  [shock]", "delta_ablationA", is_shock)

    print(f"\nNamed regressions (Δ > +{REGRESSION_THRESHOLD:.3f}):")
    for cfg_label, col in [("additive", "delta_additive"),
                           ("ablationA", "delta_ablationA")]:
        regs = df_rows[df_rows[col] > REGRESSION_THRESHOLD]
        if regs.empty:
            print(f"  {cfg_label}: none")
        else:
            for _, r in regs.iterrows():
                print(f"  {cfg_label}: fold {int(r.fold)} ({r.regime}, {r.val_start}→{r.val_end})  "
                      f"Δ={r[col]:+.4f}  ll_baseline={r.ll_baseline:.4f}")

    meta = {
        "seed": SEED,
        "n_folds": len(rows),
        "baseline_n_features": len(baseline_cols),
        "additive_n_features": len(additive_cols),
        "ablationA_n_features": len(ablation_cols),
        "engineered_col": NEW_COL,
        "dropped_for_ablation_A": list(DROP_FOR_ABLATION_A),
        "regression_threshold": REGRESSION_THRESHOLD,
        "lookup": {
            "n_bins": N_BINS,
            "pct_clip": PCT_CLIP,
            "bin_edges": [float(x) for x in BIN_EDGES],
        },
        "shock_folds": sorted(SHOCK_FOLDS),
        "agg_additive_vs_baseline_all": agg_a_all,
        "agg_ablationA_vs_baseline_all": agg_x_all,
        "agg_ablationA_vs_additive_all": agg_xa_all,
        "agg_additive_vs_baseline_normal": agg_a_n,
        "agg_ablationA_vs_baseline_normal": agg_x_n,
        "agg_additive_vs_baseline_shock": agg_a_s,
        "agg_ablationA_vs_baseline_shock": agg_x_s,
    }
    (OUT / "step4_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[total wall] {time.perf_counter() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
