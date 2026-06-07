"""Test the shallowness axis: does steep vs shallow descent flip A's effect
on elongated rows?

Hypothesis (after step5d): elongation ratio is the right axis — A's effect
bends sharply at high elongation in every fold — but the SIGN per fold is
determined by descent gradient. Steep elongated descent: A helps more.
Shallow elongated descent: A hurts more.

Operational test: among rows with elongation_ratio > 1.0 (relaxed from 1.3
per the step5d leakage finding), split by per-row cycle_descent_slope and
ask: do shallow rows show A hurting and steep rows show A helping?

Threshold for shallow vs steep: global median of cycle_descent_slope across
all rows with elongation_ratio > 1.0. Single global threshold so the
shallow/steep labels are consistent across folds.

Output: per-fold mean delta_ll on (elongated_shallow, elongated_steep) +
plot. Sign-flip per fold = constraint design has 2-axis traction.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

OUT = pathlib.Path(__file__).parent
ELONG_TAU = 1.0  # relaxed from 1.3 per step5d leakage finding


def main() -> None:
    rd = pd.read_parquet(OUT / "step5_rowdelta.parquet")
    feats = load_features()
    feats["price_date"] = pd.to_datetime(feats["price_date"])
    rd["price_date"] = pd.to_datetime(rd["price_date"])

    # Join cycle_last_max_cents (not carried in rd) for descent-slope calc.
    covars = feats[["station_code", "price_date",
                    "station_price_cents", "cycle_last_max_cents"]]
    rd = rd.merge(covars, on=["station_code", "price_date"],
                  how="left", validate="m:1")

    valid = (rd["cycle_days_since_peak"] > 0) & rd["cycle_last_max_cents"].notna() \
        & rd["baseline_cml"].notna()
    rd = rd[valid].copy()
    rd["elongation_ratio"] = rd["cycle_days_since_peak"] / rd["baseline_cml"]
    rd["cycle_descent_slope"] = (
        (rd["station_price_cents"] - rd["cycle_last_max_cents"])
        / rd["cycle_days_since_peak"]
    )
    rd["is_descent"] = rd["px_change_5d"] < 0
    rd = rd[rd["is_descent"]].copy()  # focus on descent rows only

    elong = rd[rd["elongation_ratio"] > ELONG_TAU].copy()
    shall_thresh = float(elong["cycle_descent_slope"].median())
    print(f"Population: descent rows with elongation_ratio > {ELONG_TAU}: "
          f"n={len(elong):,}")
    print(f"Global shallowness threshold (median cycle_descent_slope on this "
          f"population): {shall_thresh:+.4f} cents/day")
    print("  'shallow' = above this (less negative slope); 'steep' = below.")

    elong["regime2"] = np.where(
        elong["cycle_descent_slope"] > shall_thresh,
        "elong_shallow", "elong_steep",
    )

    print("\n=== Per fold — mean delta_ll on elongated descent rows, "
          "split by shallowness ===")
    pf = (
        elong.groupby(["fold", "regime2"])
        .agg(n=("delta_ll", "size"), mean_delta=("delta_ll", "mean"),
             median_delta=("delta_ll", "median"))
        .reset_index()
    )
    pivot_mean = pf.pivot(index="fold", columns="regime2", values="mean_delta")
    pivot_n = pf.pivot(index="fold", columns="regime2", values="n").fillna(0).astype(int)
    pivot_med = pf.pivot(index="fold", columns="regime2", values="median_delta")

    print("\nmean_delta:")
    print(pivot_mean.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nmedian_delta:")
    print(pivot_med.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nn rows:")
    print(pivot_n.to_string())

    # Sign-flip score: positive on shallow (A hurts) AND negative on steep
    # (A helps) = clean flip. Compute the "diff" = shallow - steep.
    pivot_mean["shallow_minus_steep"] = (
        pivot_mean.get("elong_shallow") - pivot_mean.get("elong_steep")
    )
    print("\nshallow_minus_steep (positive = A worse on shallow than steep, "
          "as the hypothesis predicts):")
    print(pivot_mean[["shallow_minus_steep"]].to_string(
        float_format=lambda x: f"{x:+.4f}"))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: per-fold scatter (steep on x, shallow on y).
    ax = axes[0]
    p = pivot_mean.reset_index()
    ax.scatter(p["elong_steep"], p["elong_shallow"], s=130, edgecolor="black")
    for _, r in p.iterrows():
        ax.annotate(f"f{int(r['fold'])}",
                    (r["elong_steep"], r["elong_shallow"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    lim = float(np.nanmax(np.abs(p[["elong_steep", "elong_shallow"]].to_numpy())))
    ax.plot([-lim, lim], [-lim, lim], ls=":", color="grey", lw=0.7,
            label="y = x")
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="grey", lw=0.4)
    # Shade hypothesis quadrant: hurt on shallow + help on steep
    ax.fill_between([-lim, 0], 0, lim, color="#c0392b", alpha=0.08,
                    label="hypothesis quadrant\n(steep helps, shallow hurts)")
    ax.set_xlabel("Mean delta_ll on ELONG_STEEP descent rows")
    ax.set_ylabel("Mean delta_ll on ELONG_SHALLOW descent rows")
    ax.set_title("Per-fold — does shallowness flip A's effect on elongated rows?")
    ax.legend(loc="lower right", fontsize=9)

    # Panel 2: bar chart of shallow_minus_steep per fold.
    ax = axes[1]
    diffs = pivot_mean["shallow_minus_steep"].dropna().sort_index()
    colours = ["#c0392b" if v > 0 else "#2980b9" for v in diffs]
    ax.bar(diffs.index.astype(str), diffs.values, color=colours,
           edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Fold")
    ax.set_ylabel("mean delta_ll (shallow - steep)")
    ax.set_title("Per-fold gap: A's harm on shallow elongated minus harm on steep elongated\n"
                 "(positive = hypothesis confirmed for this fold)")
    for f, v in zip(diffs.index.astype(str), diffs.values):
        ax.annotate(f"{v:+.3f}", (f, v),
                    textcoords="offset points",
                    xytext=(0, 4 if v >= 0 else -12),
                    ha="center", fontsize=8)

    fig.tight_layout()
    out_png = OUT / "step5e_shallowness.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
