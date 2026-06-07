"""Plot lga_phase_std (LGA-leader phase divergence) across three regimes,
mirroring step1_plot_dispersion.py for Signal A.

Usage: PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step1_plot_lga_divergence.py
"""
from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = pathlib.Path("experiments/2026-06-06_late_descent_triplet")

REGIMES = {
    "early descent (pct 0.05–0.30)":      (0.05, 0.30, "tab:blue"),
    "normal late descent (pct 0.30–0.50)": (0.30, 0.50, "tab:green"),
    "extended descent (pct ≥ 0.90)":      (0.90, np.inf, "tab:red"),
}

t0 = time.perf_counter()
print("Loading per-row signal sample...")
df = pd.read_csv(OUT_DIR / "step1_signals_sample.csv")
print(f"  {len(df):,} rows loaded in {time.perf_counter()-t0:.1f}s")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: density histograms by regime ---
ax = axes[0]
bins = np.linspace(0, 25, 80)
for label, (lo, hi, color) in REGIMES.items():
    mask = (df.cycle_pct_through >= lo) & (df.cycle_pct_through < hi)
    vals = df.loc[mask, "lga_phase_std"].dropna().values
    if len(vals) < 100:
        continue
    ax.hist(vals, bins=bins, density=True, alpha=0.45, color=color,
            label=f"{label}  (n={len(vals):,}, μ={vals.mean():.1f}d, σ={vals.std():.1f}d)")
ax.set_xlabel("lga_phase_std (days) — std of days_since_trough across 35 LGAs")
ax.set_ylabel("density")
ax.set_title("LGA-leader phase divergence by regime")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# --- Right: median trajectory across cycle_pct_through ---
ax = axes[1]
df["pct_bin"] = pd.cut(df.cycle_pct_through, bins=np.arange(0.0, 1.5, 0.025))
agg = df.groupby("pct_bin", observed=True)["lga_phase_std"].agg(["median", "count", "std"])
agg = agg[agg["count"] > 200]
centers = [(iv.left + iv.right) / 2 for iv in agg.index]
ax.plot(centers, agg["median"].values, color="tab:purple", lw=2,
        label="median lga_phase_std per pct bin")
ax.fill_between(
    centers,
    agg["median"].values - agg["std"].values / np.sqrt(agg["count"].values),
    agg["median"].values + agg["std"].values / np.sqrt(agg["count"].values),
    alpha=0.2, color="tab:purple",
)
for label, (lo, hi, color) in REGIMES.items():
    upper = min(hi, 1.45)
    ax.axvspan(lo, upper, alpha=0.10, color=color)
ax.axvline(0.5, color="k", ls=":", alpha=0.4, label="empirical trough (pct≈0.5)")
ax.set_xlabel("cycle_pct_through (peak-anchored)")
ax.set_ylabel("median lga_phase_std (days)")
ax.set_title("LGA-leader divergence trajectory across the cycle")
ax.set_xlim(0, 1.4)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "step1_lga_divergence.png"
plt.savefig(plot_path, dpi=120)
print(f"\nPlot saved to {plot_path}")

# Per-regime numerical summary
print("\nPer-regime summary of lga_phase_std:")
for label, (lo, hi, _) in REGIMES.items():
    mask = (df.cycle_pct_through >= lo) & (df.cycle_pct_through < hi)
    vals = df.loc[mask, "lga_phase_std"].dropna()
    print(f"  {label:<42}: n={len(vals):>7,}  mean={vals.mean():.2f}d  "
          f"median={vals.median():.2f}d  std={vals.std():.2f}d  p90={vals.quantile(0.9):.2f}d")

print(f"\nTotal elapsed: {time.perf_counter()-t0:.1f}s")
