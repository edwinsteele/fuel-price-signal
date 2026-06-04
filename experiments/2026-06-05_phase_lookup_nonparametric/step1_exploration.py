"""Step 1 exploration: cycle_pct_through distribution and empirical phase shape.

Reproducible analysis behind the parked-session findings in README.md.
Outputs:
- phase_shape_empirical.csv — per-bin mean/std/n of norm_price over equal-width
  bins of cycle_pct_through in [0, 2.0]. Committed.
- phase_shape_vs_linear.png — empirical curve vs step-4 linear-interp diagonal.
  Gitignored (per experiments/**/*.png).
- cycle_pct_through_dist.png — raw pct distribution. Gitignored.

Run from repo root:
    uv run python experiments/2026-06-05_phase_lookup_nonparametric/step1_exploration.py
"""

from __future__ import annotations

import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

OUT = pathlib.Path(__file__).parent


def main() -> None:
    t0 = time.perf_counter()
    print("loading features …")
    df = load_features()
    print(f"  [load] {time.perf_counter() - t0:.1f}s  rows={len(df):,}")

    # --- raw pct_through distribution -----------------------------------
    x = df["cycle_pct_through"].dropna().to_numpy()
    print(f"\ncycle_pct_through summary  (n={len(x):,})")
    print(f"  mean   {x.mean():.3f}")
    print(f"  median {np.median(x):.3f}")
    print(f"  pct 0  {(x == 0.0).sum():,}  ({(x == 0.0).mean() * 100:.1f}%)")
    print(f"  pct >1 {(x > 1).sum():,}  ({(x > 1).mean() * 100:.1f}%)")
    print(f"  p90/99/max  {np.quantile(x, 0.9):.3f} / {np.quantile(x, 0.99):.3f} / {x.max():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(x, bins=120, edgecolor="none")
    axes[0].axvline(1.0, color="red", ls="--", alpha=0.6, label="pct = 1.0")
    axes[0].set_title(f"cycle_pct_through, full range (n={len(x):,})")
    axes[0].set_xlabel("cycle_pct_through")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[1].hist(x[x <= 1.5], bins=60, edgecolor="none")
    axes[1].axvline(1.0, color="red", ls="--", alpha=0.6)
    axes[1].set_title("zoom [0, 1.5]")
    axes[1].set_xlabel("cycle_pct_through")
    plt.tight_layout()
    plt.savefig(OUT / "cycle_pct_through_dist.png", dpi=110)

    # --- empirical phase shape ------------------------------------------
    amp = df["cycle_last_max_cents"] - df["cycle_last_min_cents"]
    mask = amp > 0
    d = df.loc[mask, ["cycle_pct_through", "station_price_cents",
                      "cycle_last_min_cents", "cycle_last_max_cents"]].copy()
    d["norm_price"] = (d["station_price_cents"] - d["cycle_last_min_cents"]) / (
        d["cycle_last_max_cents"] - d["cycle_last_min_cents"])

    edges = np.linspace(0, 2.0, 41)
    centres = 0.5 * (edges[:-1] + edges[1:])
    d["bin"] = pd.cut(d["cycle_pct_through"].clip(upper=2.0), bins=edges, include_lowest=True)
    agg = d.groupby("bin", observed=True)["norm_price"].agg(["mean", "std", "count"]).reset_index()
    agg["bin_centre"] = centres[: len(agg)]
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    out_csv = agg[["bin_centre", "mean", "std", "se", "count"]].copy()
    out_csv.columns = ["pct_bin_centre", "mean_norm_price", "std_norm_price",
                       "se_norm_price", "n_rows"]
    out_csv.to_csv(OUT / "phase_shape_empirical.csv", index=False)
    print(f"\nsaved phase_shape_empirical.csv  ({len(out_csv)} bins)")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.errorbar(agg["bin_centre"], agg["mean"], yerr=2 * agg["se"],
                fmt="o-", color="steelblue",
                label="Empirical E[norm_price | pct]  (±2 SE)", linewidth=2, markersize=5)
    xs = np.linspace(0, 2.0, 200)
    ax.plot(xs, xs, "--", color="crimson",
            label="Step-4 linear-interp formula:  expected_norm = pct", linewidth=2)
    ax.axvline(1.0, color="grey", ls=":", alpha=0.6)
    ax.axhline(0.0, color="grey", ls=":", alpha=0.4)
    ax.axhline(1.0, color="grey", ls=":", alpha=0.4)
    ax.set_xlabel("cycle_pct_through  (days_since_last_peak / mean_cycle_length)")
    ax.set_ylabel("Normalised station price\n0 = last_min  ·  1 = last_max")
    ax.set_title("What the price actually does over a cycle vs what step-4 assumed")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "phase_shape_vs_linear.png", dpi=120)

    print(f"\n[total] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
