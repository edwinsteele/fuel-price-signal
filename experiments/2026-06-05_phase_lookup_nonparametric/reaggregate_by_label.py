"""Re-aggregate step4 results under alternative shock/normal labellings.

Reads step4_folds.csv (the per-fold deltas from the paired CV — no model re-run
needed) and computes split aggregates under three labellings:

1. PRE-COMMITTED: the original {1, 4, 9, 13} macro-event labels.
2. TOP-QUARTILE BASELINE LOGLOSS: the 4 hardest folds by ll_baseline.
3. TOP-THIRD BASELINE LOGLOSS: the 5 hardest folds by ll_baseline.

Output is printed side-by-side so we can see whether the gate verdict survives
relabelling.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
PRE_COMMITTED_SHOCKS = {1, 4, 9, 13}


def fmt(v: np.ndarray, label: str) -> str:
    if v.size == 0:
        return f"  {label:<10s}  (no folds)"
    return (
        f"  {label:<10s}  n={v.size:>2d}  mean={v.mean():+.4f}  "
        f"median={float(np.median(v)):+.4f}  "
        f"min={v.min():+.4f}  max={v.max():+.4f}  "
        f"helps={int((v < 0).sum())}/{v.size}  hurts={int((v > 0).sum())}/{v.size}"
    )


def report(df: pd.DataFrame, shock_set: set[int], scheme_name: str) -> None:
    print(f"\n========== {scheme_name} ==========")
    print(f"shock folds: {sorted(shock_set)}")
    is_shock = df["fold"].isin(shock_set).to_numpy()
    is_normal = ~is_shock

    for col, label in [("delta_additive", "Δ additive  − baseline"),
                       ("delta_ablationA", "Δ ablationA − baseline")]:
        print(f"\n{label}:")
        print(fmt(df[col].to_numpy(), "ALL"))
        print(fmt(df[col].to_numpy()[is_normal], "NORMAL"))
        print(fmt(df[col].to_numpy()[is_shock], "SHOCK"))


def main() -> None:
    df = pd.read_csv(HERE / "step4_folds.csv")
    print(f"Loaded {len(df)} folds from step4_folds.csv\n")

    sorted_by_diff = df.sort_values("ll_baseline", ascending=False)
    print("Folds ranked by baseline log-loss (hardest first):")
    print(sorted_by_diff[["fold", "regime", "val_start", "val_end", "ll_baseline",
                          "delta_additive", "delta_ablationA"]].to_string(index=False))

    top_quartile = set(sorted_by_diff.head(4)["fold"].astype(int).tolist())
    top_third = set(sorted_by_diff.head(5)["fold"].astype(int).tolist())

    report(df, PRE_COMMITTED_SHOCKS, "Scheme 1: PRE-COMMITTED (macro events)")
    report(df, top_quartile, f"Scheme 2: TOP-QUARTILE ll_baseline  shocks={sorted(top_quartile)}")
    report(df, top_third,    f"Scheme 3: TOP-THIRD ll_baseline    shocks={sorted(top_third)}")


if __name__ == "__main__":
    main()
