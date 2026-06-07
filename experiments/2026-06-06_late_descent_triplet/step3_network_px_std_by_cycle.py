"""Distribution of network_px_std across cycle_pct_through bins.

Uses the same competitive-cohort definition as step1 (stickiness_score < p75,
station-level rows) so the trough dip is correctly resolved.

Top panel: fine-bin (0.05) median trajectory — matches step1_dispersion.png right panel.
Bottom panel: box plots at 7 representative positions spanning the cycle.

Usage: PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step3_network_px_std_by_cycle.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from fuel_signal.features import load_features

OUT = pathlib.Path(__file__).parent

# Match step2_paired_wfcv.py competitive-cohort definition exactly.
COMP_BAND_CENTS = 5.0   # |stickiness_score| <= 5


def compute_px_std(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date std of station_price_cents over competitive stations (|stickiness| ≤ 5c)."""
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])
    comp = df[df["stickiness_score"].abs() <= COMP_BAND_CENTS]
    px_std = comp.groupby("price_date")["station_price_cents"].std().rename("px_std")
    return df.join(px_std, on="price_date"), COMP_BAND_CENTS


# 7 representative positions: centre of each box, chosen to span the cycle
# and highlight key phases. Labels describe empirical meaning.
BOX_CENTERS = [0.05, 0.20, 0.40, 0.55, 0.70, 0.90, 1.10, 1.35]
BOX_HALF_WIDTH = 0.075  # ±0.075 around centre → 0.15-wide window
BOX_LABELS = [
    "0.0–0.1\npost-peak",
    "0.15–0.25\ndescent",
    "0.35–0.45",
    "0.5–0.6",
    "0.65–0.75",
    "0.85–0.95",
    "1.05–1.15\nelongated",
    "1.28–1.43\ndeep elongated",
]
FINE_BIN_WIDTH = 0.05


def main() -> None:
    print("Loading features …")
    df = load_features()
    print(f"  rows={len(df):,}")

    print("Computing px_std (step2 definition: |stickiness| ≤ 5c) …")
    df, sticky_thresh = compute_px_std(df)
    print(f"  competitive cohort: |stickiness_score| ≤ {sticky_thresh:.1f}c")

    working = df[["price_date", "cycle_pct_through", "px_std"]].dropna()
    print(f"  rows with px_std: {len(working):,}")

    # --- Fine-bin trajectory (station-level rows, matches step1 right panel) ---
    fine_bins = np.arange(0.0, 1.55, FINE_BIN_WIDTH)
    fine_centers = fine_bins[:-1] + FINE_BIN_WIDTH / 2
    working["fine_bin"] = pd.cut(working["cycle_pct_through"], bins=fine_bins, right=False)
    fine_medians = working.groupby("fine_bin", observed=True)["px_std"].median()
    fine_n = working.groupby("fine_bin", observed=True)["px_std"].count()

    # --- Box groups at representative positions ---
    groups = []
    for ctr in BOX_CENTERS:
        lo, hi = ctr - BOX_HALF_WIDTH, ctr + BOX_HALF_WIDTH
        mask = (working["cycle_pct_through"] >= lo) & (working["cycle_pct_through"] < hi)
        groups.append(working.loc[mask, "px_std"].dropna().values)

    medians = [np.median(g) if len(g) > 0 else np.nan for g in groups]
    ns = [len(g) for g in groups]

    # Print summary
    print("\nBox-plot groups:")
    for lbl, med, n in zip(BOX_LABELS, medians, ns):
        print(f"  {lbl!r:30s}  median={med:.1f}c  n={n:,}")

    # --- Plot ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 10),
                                          gridspec_kw={"height_ratios": [1, 1.6]})

    # Top: fine-bin median trajectory
    ax_top.plot(fine_centers[:len(fine_medians)], fine_medians.values,
                color="steelblue", linewidth=2, marker="o", markersize=3, label="median px_std per 0.05 bin")
    ax_top.axvline(0.5, color="grey", linestyle=":", linewidth=1.5, label="empirical trough (pct≈0.5)")
    ax_top.axvspan(0.30, 0.50, color="orange", alpha=0.12, label="normal late descent")
    ax_top.axvspan(0.90, 1.50, color="purple", alpha=0.08, label="extended / elongated")
    ax_top.set_xlim(0, 1.5)
    ax_top.set_ylabel("median px_std (cents)", fontsize=10)
    ax_top.set_title(
        "px_std trajectory across cycle (station-level rows, 0.05 bins)\n"
        f"Competitive cohort: |stickiness_score| ≤ {sticky_thresh:.1f}c  —  matches step2_paired_wfcv.py",
        fontsize=10,
    )
    ax_top.legend(fontsize=8)
    ax_top.grid(axis="y", alpha=0.3, linestyle="--")

    # Bottom: box plots at representative positions
    positions = list(range(len(BOX_CENTERS)))
    bp = ax_bot.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="crimson", linewidth=2.5),
        boxprops=dict(facecolor="steelblue", alpha=0.40),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=2, alpha=0.25, color="grey"),
        showfliers=True,
    )

    for x, (med, n) in enumerate(zip(medians, ns)):
        if not np.isnan(med):
            ax_bot.text(x, med + 0.2, f"{med:.1f}c", ha="center", va="bottom",
                        fontsize=9, color="crimson", fontweight="bold")
        ax_bot.text(x, -1.5, f"n={n:,}", ha="center", va="top", fontsize=7, color="dimgrey",
                    transform=ax_bot.get_xaxis_transform())

    ax_bot.set_xticks(positions)
    ax_bot.set_xticklabels(BOX_LABELS, fontsize=8.5)
    ax_bot.set_ylabel("px_std (cents)", fontsize=10)
    ax_bot.set_title(
        "Distribution at 7 representative cycle positions  •  median in red",
        fontsize=10,
    )
    ax_bot.grid(axis="y", alpha=0.3, linestyle="--")
    ax_bot.set_ylim(bottom=0)

    # Shade regimes on bottom panel (positions 2 and 5 are the two zones of interest)
    ax_bot.axvspan(1.5, 2.5, color="orange", alpha=0.10, label="normal late descent")
    ax_bot.axvspan(4.5, 6.5, color="purple", alpha=0.08, label="extended / elongated")
    ax_bot.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT / "step3_network_px_std_by_cycle.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
