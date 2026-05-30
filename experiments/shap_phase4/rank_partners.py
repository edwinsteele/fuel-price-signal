"""Cheap partner ranking for stickiness_score.

Reuses experiments/shap_phase4/shap_values.npy (already saved) and rebuilds
X_val from the same train/val split. Runs SHAP's approximate_interactions
heuristic by hand so we can report the *scores*, not just the rank.

Heuristic (matches shap.utils.approximate_interactions):
  1. sort rows by main feature
  2. bin into chunks of inc = clip(N/10, 1, 50) rows
  3. per bin, |corr(partner_value, main_shap_value)|
  4. sum across bins → score per partner

Prints scores so you can eyeball how concentrated the interaction signal is.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS

OUT = pathlib.Path(__file__).parent
FEATURES_CSV = pathlib.Path("data/features.csv")
ALL_FEATURES = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
TARGET = "stickiness_score"


def approx_interaction_scores(main_idx: int, sv: np.ndarray, X: np.ndarray) -> np.ndarray:
    x_main = X[:, main_idx]
    order = np.argsort(x_main)
    shap_main = sv[order, main_idx]
    inc = int(np.clip(len(x_main) / 10.0, 1, 50))
    scores = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        if j == main_idx:
            continue
        partner = X[order, j].astype(np.float64)
        s = 0.0
        for k in range(0, len(x_main), inc):
            a = shap_main[k : k + inc]
            b = partner[k : k + inc]
            mask = ~np.isnan(b)
            if mask.sum() < 3:
                continue
            a = a[mask]
            b = b[mask]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            s += abs(np.corrcoef(a, b)[0, 1])
        scores[j] = s
    return scores


def main() -> None:
    print("Loading features + reproducing val split…")
    df = pd.read_csv(FEATURES_CSV, parse_dates=["price_date"])
    _train, val, _test = _ev.split(df)
    X_val = val[ALL_FEATURES].to_numpy(dtype=float)
    print(f"  val rows: {X_val.shape[0]:,}  features: {X_val.shape[1]}")

    sv = np.load(OUT / "shap_values.npy")
    assert sv.shape == X_val.shape, f"shape mismatch {sv.shape} vs {X_val.shape}"

    ti = ALL_FEATURES.index(TARGET)
    print(f"\nRanking partners for '{TARGET}' (idx {ti})…")
    scores = approx_interaction_scores(ti, sv, X_val)

    df_out = (
        pd.DataFrame({"partner": ALL_FEATURES, "score": scores})
        .drop(index=ti)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    df_out.to_csv(OUT / f"partner_scores_{TARGET}.csv", index=False)

    top = df_out.head(15)
    top_score = top["score"].iloc[0]
    print("\nTop 15 candidate partners (score; share of #1):")
    for i, row in top.iterrows():
        share = row["score"] / top_score if top_score > 0 else 0
        bar = "█" * int(round(share * 30))
        print(f"  {i + 1:>2}. {row['partner']:<45} {row['score']:>7.3f}  {share:>5.1%}  {bar}")

    total = df_out["score"].sum()
    top1_share = df_out["score"].iloc[0] / total if total > 0 else 0
    top4_share = df_out["score"].head(4).sum() / total if total > 0 else 0
    print(f"\n  top-1 share of total interaction signal: {top1_share:.1%}")
    print(f"  top-4 share of total interaction signal: {top4_share:.1%}")


if __name__ == "__main__":
    main()
