"""Test the fold 9 leakage hypothesis.

Claim: my is_elongated flag (cycle_days_since_peak / baseline_cml > 1.3) uses
an adaptive baseline. On fold 9 ("two back-to-back elongated cycles"), the
baseline drifts upward during the elongation, so late rows of an elongated
cycle get re-classified as normal_descent — leaking the failure mode into
the wrong bucket.

Verification: among fold 9's normal_descent rows, bin by elongation_ratio
(0 to 1.3). If A's positive delta is concentrated in the high-ratio bins,
the leakage is real and those rows are extended-descent in disguise.

Controls:
- Fold 7 normal_descent: same binning. Should show similar pattern if
  leakage is a population-wide artefact.
- Fold 8 normal_descent: A helps strongly here. Should NOT show ratio
  dependence — clean baseline.

Output: step5d_leakage.png, console summary.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent


def bin_mean_delta(sub: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    sub = sub.dropna(subset=["elongation_ratio", "delta_ll"]).copy()
    sub["bin"] = pd.cut(sub["elongation_ratio"], edges,
                        include_lowest=True, right=False)
    g = (sub.groupby("bin", observed=True)
         .agg(n=("delta_ll", "size"),
              mean_delta=("delta_ll", "mean"),
              median_delta=("delta_ll", "median"))
         .reset_index())
    return g


def main() -> None:
    rd = pd.read_parquet(OUT / "step5_rowdelta.parquet")
    rd = rd[(rd["cycle_days_since_peak"] > 0) & rd["baseline_cml"].notna()].copy()
    rd["elongation_ratio"] = rd["cycle_days_since_peak"] / rd["baseline_cml"]

    edges = np.array([0.0, 0.4, 0.7, 1.0, 1.3])  # final edge = is_elongated threshold

    folds_to_test = {
        "fold 9 (suspect leakage)": 9,
        "fold 7 (also a regressor)": 7,
        "fold 8 (clean — A helps)": 8,
        "fold 3 (A helps strongly)": 3,
    }

    print("Per-fold: normal_descent rows binned by elongation_ratio (0 to 1.3)\n")
    results = {}
    for label, f in folds_to_test.items():
        sub = rd[(rd["fold"] == f) & (rd["bucket"] == "normal_descent")]
        binned = bin_mean_delta(sub, edges)
        print(f"=== {label} ===")
        print(binned.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        # Sanity: what's the mean delta on fold's ext_descent rows for comparison?
        ext = rd[(rd["fold"] == f) & (rd["bucket"] == "ext_descent")]
        if len(ext):
            print(f"  reference: ext_descent mean delta = {ext['delta_ll'].mean():+.4f}  "
                  f"(n={len(ext)})")
        print()
        results[label] = binned

    # Plot: per-fold mean delta_ll vs ratio bin (lines)
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = {"fold 9 (suspect leakage)": "#c0392b",
               "fold 7 (also a regressor)": "#e67e22",
               "fold 8 (clean — A helps)": "#2980b9",
               "fold 3 (A helps strongly)": "#27ae60"}
    for label, b in results.items():
        # Use midpoint of each bin as x
        mid = b["bin"].apply(lambda iv: (iv.left + iv.right) / 2)
        ax.plot(mid, b["mean_delta"], "o-", label=label,
                color=colours[label], linewidth=2, markersize=8)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(1.3, color="grey", linestyle=":", linewidth=0.7,
               label="is_elongated threshold (1.3)")
    ax.set_xlabel("Elongation ratio (cycle_days_since_peak / baseline_cml)")
    ax.set_ylabel("Mean delta_ll on normal_descent rows (positive = A hurts)")
    ax.set_title("Fold 9 leakage test — does A's normal_descent harm "
                 "concentrate at the high-ratio end?")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_png = OUT / "step5d_leakage.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
