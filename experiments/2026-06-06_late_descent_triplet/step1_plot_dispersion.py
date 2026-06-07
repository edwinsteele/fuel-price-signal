"""Plot density of px_std for three regimes: early descent, normal late
descent, extended descent.

Usage: PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step1_plot_dispersion.py
"""
from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = pathlib.Path("experiments/2026-06-06_late_descent_triplet")

# Regime definitions (cycle_pct_through; peak-anchored, trough ≈ 0.5)
REGIMES = {
    "early descent (pct 0.05–0.30)":   (0.05, 0.30, "tab:blue"),
    "normal late descent (pct 0.30–0.50)": (0.30, 0.50, "tab:green"),
    "extended descent (pct ≥ 0.90)":   (0.90, np.inf, "tab:red"),
}

t0 = time.perf_counter()
print("Loading per-row signal sample...")
df = pd.read_csv(OUT_DIR / "step1_signals_sample.csv")
print(f"  {len(df):,} rows loaded in {time.perf_counter()-t0:.1f}s")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: KDE-like normalized histograms (overlay) ---
ax = axes[0]
bins = np.linspace(0, 30, 80)
for label, (lo, hi, color) in REGIMES.items():
    mask = (df.cycle_pct_through >= lo) & (df.cycle_pct_through < hi)
    vals = df.loc[mask, "px_std"].dropna().values
    if len(vals) < 100:
        continue
    ax.hist(vals, bins=bins, density=True, alpha=0.45, color=color,
            label=f"{label}  (n={len(vals):,}, μ={vals.mean():.1f}c, σ={vals.std():.1f}c)")
ax.set_xlabel("network px_std (cents) — cross-station std of competitive prices")
ax.set_ylabel("density")
ax.set_title("Cross-station dispersion by phase regime")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# --- Right: median px_std across cycle_pct_through bins (trajectory view) ---
ax = axes[1]
# Bin pct from 0 to 1.3 in 0.025 buckets
df["pct_bin"] = pd.cut(df.cycle_pct_through, bins=np.arange(0.0, 1.5, 0.025))
agg = df.groupby("pct_bin", observed=True)["px_std"].agg(["median", "count", "std"])
agg = agg[agg["count"] > 200]
centers = [(iv.left + iv.right) / 2 for iv in agg.index]
ax.plot(centers, agg["median"].values, color="tab:purple", lw=2,
        label="median px_std per pct bin")
ax.fill_between(
    centers,
    agg["median"].values - agg["std"].values / np.sqrt(agg["count"].values),
    agg["median"].values + agg["std"].values / np.sqrt(agg["count"].values),
    alpha=0.2, color="tab:purple",
)
# Mark the three regimes
for label, (lo, hi, color) in REGIMES.items():
    upper = min(hi, 1.45)
    ax.axvspan(lo, upper, alpha=0.10, color=color)
ax.axvline(0.5, color="k", ls=":", alpha=0.4, label="empirical trough (pct≈0.5)")
ax.set_xlabel("cycle_pct_through (peak-anchored)")
ax.set_ylabel("median px_std (cents)")
ax.set_title("Dispersion trajectory across the cycle")
ax.set_xlim(0, 1.4)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "step1_dispersion.png"
plt.savefig(plot_path, dpi=120)
print(f"\nPlot saved to {plot_path}")

# Numerical summary by regime (and early descent in particular)
print("\nPer-regime summary of px_std:")
for label, (lo, hi, _) in REGIMES.items():
    mask = (df.cycle_pct_through >= lo) & (df.cycle_pct_through < hi)
    vals = df.loc[mask, "px_std"].dropna()
    print(f"  {label:<42}: n={len(vals):>7,}  mean={vals.mean():.2f}c  "
          f"median={vals.median():.2f}c  std={vals.std():.2f}c  p90={vals.quantile(0.9):.2f}c")

print(f"\nTotal elapsed: {time.perf_counter()-t0:.1f}s")
