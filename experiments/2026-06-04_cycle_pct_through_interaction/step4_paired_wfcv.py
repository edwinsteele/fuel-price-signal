"""Step 4: paired walk-forward CV for the phase-residual feature.

Three configs, same single seed per fold (cv_compare_phase4 precedent —
fold-to-fold variance dominates seed variance for this question):
- baseline: 50 Phase 4 features.
- additive: + station_minus_expected_phase_price (51).
- ablationA: + station_minus_expected_phase_price
              − station_minus_last_min_cents
              − station_minus_last_max_cents          (49).

The gate per CONVENTIONS.md (paired walk-forward CV for feature-set
changes) and [[feedback_regime_segmented_evaluation]]: report per-fold
deltas, median Δ, and any folds where the engineered configs materially
regress vs baseline.
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
REGRESSION_THRESHOLD = 0.005  # logloss regression considered "material" per fold

NEW_COL = "station_minus_expected_phase_price"
DROP_FOR_ABLATION_A = (
    "station_minus_last_min_cents",
    "station_minus_last_max_cents",
)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    expected = df["cycle_last_min_cents"] + df["cycle_pct_through"] * (
        df["cycle_last_max_cents"] - df["cycle_last_min_cents"]
    )
    out = df.copy()
    out[NEW_COL] = df["station_price_cents"] - expected
    return out


def fit_score(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    t0 = time.perf_counter()
    model = LGBMClassifier(random_state=SEED, verbose=-1, subsample=0.8, subsample_freq=1)
    model.fit(train_df[cols].to_numpy(dtype=float), train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols].to_numpy(dtype=float))[:, 1]
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, time.perf_counter() - t0


def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading features (load_features) …")
    t0 = time.perf_counter()
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}")

    df = add_engineered(df)

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    additive_cols = baseline_cols + [NEW_COL]
    ablation_cols = [c for c in additive_cols if c not in DROP_FOR_ABLATION_A]
    assert len(baseline_cols) == 50
    assert len(additive_cols) == 51
    assert len(ablation_cols) == 49

    folds = list(_ev.walk_forward_folds(df, train_min_days=1825, val_days=90, step_days=90))
    print(f"Walk-forward folds: {len(folds)}  seed={SEED}")
    print(f"Configs: baseline={len(baseline_cols)}  additive={len(additive_cols)}  ablationA={len(ablation_cols)}\n")

    print(f"{'fold':>4}  {'val_start':>10}  {'val_end':>10}  {'val_rows':>8}  {'BUY%':>5}  "
          f"{'ll_base':>7}  {'ll_add':>7}  {'ll_abl':>7}  "
          f"{'Δ_add':>8}  {'Δ_abl':>8}  {'Δ_abl-add':>10}")
    print("-" * 120)

    rows = []
    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        ll_b, t_b = fit_score(train_df, val_df, baseline_cols)
        ll_a, t_a = fit_score(train_df, val_df, additive_cols)
        ll_x, t_x = fit_score(train_df, val_df, ablation_cols)
        d_a = ll_a - ll_b
        d_x = ll_x - ll_b
        d_xa = ll_x - ll_a
        vd = pd.to_datetime(val_df["price_date"])
        row = {
            "fold": i,
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
        }
        rows.append(row)
        print(f"{row['fold']:>4}  {row['val_start']:>10}  {row['val_end']:>10}  "
              f"{row['val_rows']:>8,}  {row['buy_rate']*100:>4.1f}%  "
              f"{ll_b:>7.4f}  {ll_a:>7.4f}  {ll_x:>7.4f}  "
              f"{d_a:>+8.4f}  {d_x:>+8.4f}  {d_xa:>+10.4f}")

    if not rows:
        print("No folds produced.")
        return

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "step4_folds.csv", index=False)
    print("-" * 120)

    def aggregate(name: str, col: str) -> dict:
        v = df_rows[col].to_numpy()
        m, med, sd = float(v.mean()), float(np.median(v)), float(v.std(ddof=1))
        n_help = int((v < 0).sum())
        n_hurt = int((v > 0).sum())
        print(f"{name:<22s} mean={m:+.4f}  median={med:+.4f}  std={sd:.4f}  "
              f"min={v.min():+.4f}  max={v.max():+.4f}  helps={n_help}/{len(v)}  hurts={n_hurt}/{len(v)}")
        return {"name": name, "mean": m, "median": med, "std": sd,
                "min": float(v.min()), "max": float(v.max()),
                "n_help": n_help, "n_hurt": n_hurt, "n_folds": len(v)}

    print("\nAggregate deltas across folds (single seed, paired):")
    agg_a = aggregate("Δ additive − baseline", "delta_additive")
    agg_x = aggregate("Δ ablationA − baseline", "delta_ablationA")
    agg_xa = aggregate("Δ ablationA − additive", "delta_abl_vs_add")

    # Named regressions: folds where engineered configs regress materially.
    print(f"\nNamed regressions (Δ > +{REGRESSION_THRESHOLD:.3f}):")
    for cfg_label, col in [("additive", "delta_additive"),
                           ("ablationA", "delta_ablationA")]:
        regs = df_rows[df_rows[col] > REGRESSION_THRESHOLD]
        if regs.empty:
            print(f"  {cfg_label}: none")
        else:
            for _, r in regs.iterrows():
                print(f"  {cfg_label}: fold {int(r.fold)} ({r.val_start}→{r.val_end})  "
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
        "agg_additive_vs_baseline": agg_a,
        "agg_ablationA_vs_baseline": agg_x,
        "agg_ablationA_vs_additive": agg_xa,
    }
    (OUT / "step4_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[total wall] {time.perf_counter() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
