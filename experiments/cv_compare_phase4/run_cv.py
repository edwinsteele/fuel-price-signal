"""Paired walk-forward CV: 15-feat (Phase 3c) vs 50-feat (Phase 4) LGBM.

For each fold yielded by evaluate.walk_forward_folds(), trains LGBM at seed=42
on both feature sets and prints per-fold val logloss + delta. Goal: confirm
the Phase 4 train/val improvement (−0.0361 raw val logloss vs Phase 3c) is
robust across fold windows — and surface any regime where the 35 LGA
features regress (fold 5 / Ukraine spike is the known tail risk per
project_stickiness_regime_lag).

Single seed per fold — fold-to-fold variance dominates seed variance for this
question (same rationale as cv_compare_15feat).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS

SEED = 42

FEATS_15 = FEATURE_COLUMNS
FEATS_50 = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS


def _fit_score(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]) -> float:
    model = LGBMClassifier(random_state=SEED, verbose=-1)
    model.fit(train_df[cols].to_numpy(dtype=float), train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols].to_numpy(dtype=float))[:, 1]
    return float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))


def main() -> None:
    df = pd.read_csv("data/features.csv")
    folds = list(_ev.walk_forward_folds(df, train_min_days=1825, val_days=90, step_days=90))
    print(f"Walk-forward folds generated: {len(folds)}")
    print(f"Features 15 (Phase 3c): {len(FEATS_15)} cols")
    print(f"Features 50 (Phase 4):  {len(FEATS_50)} cols (+{len(LGA_FEATURE_COLUMNS)} LGA trough)\n")

    print(f"{'fold':>4}  {'val_start':>10}  {'val_end':>10}  "
          f"{'val_rows':>8}  {'BUY%':>5}  "
          f"{'ll_15':>7}  {'ll_50':>7}  {'Δ (50-15)':>9}")
    print("-" * 88)

    rows = []
    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        ll15 = _fit_score(train_df, val_df, FEATS_15)
        ll50 = _fit_score(train_df, val_df, FEATS_50)
        delta = ll50 - ll15
        vd = pd.to_datetime(val_df["price_date"])
        row = {
            "fold": i,
            "val_start": vd.min().strftime("%Y-%m-%d"),
            "val_end":   vd.max().strftime("%Y-%m-%d"),
            "val_rows":  len(val_df),
            "buy_rate":  float(val_df["label"].mean()),
            "ll_15":     ll15,
            "ll_50":     ll50,
            "delta":     delta,
        }
        rows.append(row)
        print(f"{row['fold']:>4}  {row['val_start']:>10}  {row['val_end']:>10}  "
              f"{row['val_rows']:>8,}  {row['buy_rate']*100:>4.1f}%  "
              f"{ll15:>7.4f}  {ll50:>7.4f}  {delta:>+9.4f}")

    if not rows:
        print("No folds produced.")
        return

    deltas = np.array([r["delta"] for r in rows])
    n_helps = int((deltas < 0).sum())
    n_hurts = int((deltas > 0).sum())
    print("-" * 88)
    print(f"folds: {len(rows)}    15-feat mean: {np.mean([r['ll_15'] for r in rows]):.4f}    "
          f"50-feat mean: {np.mean([r['ll_50'] for r in rows]):.4f}")
    print(f"Δ (50 − 15): mean {deltas.mean():+.4f}  median {np.median(deltas):+.4f}  "
          f"std {deltas.std(ddof=1):+.4f}  min {deltas.min():+.4f}  max {deltas.max():+.4f}")
    print(f"Folds where 50-feat is BETTER (Δ<0): {n_helps}/{len(rows)}  "
          f"WORSE (Δ>0): {n_hurts}/{len(rows)}")

    out = pathlib.Path(__file__).parent / "results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
