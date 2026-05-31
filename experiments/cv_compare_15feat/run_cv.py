"""[Superseded by fuel_signal.cv_report — see README Walk-forward CV report section]

Paired walk-forward CV: 14-feat vs 15-feat LGBM.

For each fold yielded by evaluate.walk_forward_folds(), trains LGBM at seed=42
on both feature sets (drop stickiness_score for the 14-feat run) and prints
per-fold val logloss + delta. Goal: see whether the 15-feat val improvement
on the canonical Phase-3 val window is robust across fold windows, or
window-specific.

Single seed per fold — fold-to-fold variance dominates seed variance for this
question; multi-seed would multiply compute without proportionate insight.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS

SEED = 42
DROP_FEATURE = "stickiness_score"

FEATS_15 = FEATURE_COLUMNS
FEATS_14 = [c for c in FEATURE_COLUMNS if c != DROP_FEATURE]


def _fit_score(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]) -> float:
    model = LGBMClassifier(random_state=SEED, verbose=-1)
    model.fit(train_df[cols].to_numpy(dtype=float), train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols].to_numpy(dtype=float))[:, 1]
    return float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))


def main() -> None:
    df = pd.read_csv("data/features.csv")
    folds = list(_ev.walk_forward_folds(df, train_min_days=1825, val_days=90, step_days=90))
    print(f"Walk-forward folds generated: {len(folds)}")
    print(f"Features 14: {FEATS_14}")
    print(f"Features 15: includes {DROP_FEATURE}\n")

    print(f"{'fold':>4}  {'val_start':>10}  {'val_end':>10}  "
          f"{'val_rows':>8}  {'BUY%':>5}  "
          f"{'ll_14':>7}  {'ll_15':>7}  {'Δ (15-14)':>9}")
    print("-" * 88)

    rows = []
    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        ll14 = _fit_score(train_df, val_df, FEATS_14)
        ll15 = _fit_score(train_df, val_df, FEATS_15)
        delta = ll15 - ll14
        vd = pd.to_datetime(val_df["price_date"])
        row = {
            "fold": i,
            "val_start": vd.min().strftime("%Y-%m-%d"),
            "val_end":   vd.max().strftime("%Y-%m-%d"),
            "val_rows":  len(val_df),
            "buy_rate":  float(val_df["label"].mean()),
            "ll_14":     ll14,
            "ll_15":     ll15,
            "delta":     delta,
        }
        rows.append(row)
        print(f"{row['fold']:>4}  {row['val_start']:>10}  {row['val_end']:>10}  "
              f"{row['val_rows']:>8,}  {row['buy_rate']*100:>4.1f}%  "
              f"{ll14:>7.4f}  {ll15:>7.4f}  {delta:>+9.4f}")

    if not rows:
        print("No folds produced.")
        return

    deltas = np.array([r["delta"] for r in rows])
    n_helps = int((deltas < 0).sum())
    n_hurts = int((deltas > 0).sum())
    print("-" * 88)
    print(f"folds: {len(rows)}    14-feat mean: {np.mean([r['ll_14'] for r in rows]):.4f}    "
          f"15-feat mean: {np.mean([r['ll_15'] for r in rows]):.4f}")
    print(f"Δ (15 − 14): mean {deltas.mean():+.4f}  std {deltas.std(ddof=1):+.4f}  "
          f"min {deltas.min():+.4f}  max {deltas.max():+.4f}")
    print(f"Folds where 15-feat is BETTER (Δ<0): {n_helps}/{len(rows)}  "
          f"WORSE (Δ>0): {n_hurts}/{len(rows)}")

    out = pathlib.Path(__file__).parent / "results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
