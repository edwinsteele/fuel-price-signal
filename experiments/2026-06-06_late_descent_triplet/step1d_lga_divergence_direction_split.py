"""Within extended descent (pct > 0.9), does lga_phase_std live in the
descending sub-population or the ascending one? Tests whether Signal C is
"descent-coordination-breakdown" or just "ascent leader-lag in disguise".

Usage: PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step1d_lga_divergence_direction_split.py
"""
from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

OUT_DIR = pathlib.Path("experiments/2026-06-06_late_descent_triplet")
EXCLUDE_BRANDS = {"7_eleven", "ampol_foodary", "bp", "budget", "eg_ampol",
                  "independent", "metro_fuel", "reddy_express", "shell", "speedway"}

t0 = time.perf_counter()
print("Loading features (station_price + LGA columns)...")
df = load_features()
df["price_date"] = pd.to_datetime(df["price_date"])
print(f"  {len(df):,} rows  ({time.perf_counter()-t0:.1f}s)")

# Recompute lga_phase_std per date (same logic as Step 1)
lga_cols = [c for c in df.columns
            if c.startswith("days_since_trough_entry_")
            and c.replace("days_since_trough_entry_", "") not in EXCLUDE_BRANDS]
print(f"  LGA columns: {len(lga_cols)}")

t = time.perf_counter()
per_date = df.drop_duplicates("price_date").set_index("price_date")[lga_cols]
per_date_spread = per_date.std(axis=1).rename("lga_phase_std")
df = df.join(per_date_spread, on="price_date")
print(f"  lga_phase_std computed ({time.perf_counter()-t:.1f}s)")

# Per-station 5-day backward price change
t = time.perf_counter()
df = df.sort_values(["station_code", "price_date"])
df["px_5d_change"] = df.groupby("station_code")["station_price_cents"].diff(5)
print(f"  5d delta computed ({time.perf_counter()-t:.1f}s)")

# Extended descent only
ext = df[df.cycle_pct_through >= 0.9].copy()
print(f"\nExtended descent rows: {len(ext):,}")
print(f"  with non-null 5d change: {ext['px_5d_change'].notna().sum():,}")

# Direction buckets
def bucket(d):
    if pd.isna(d): return "no5d"
    if d < -2: return "down (5d Δ < -2c)"
    if d > 2:  return "up   (5d Δ > +2c)"
    return "flat (|5d Δ| ≤ 2c)"
ext["dir"] = ext["px_5d_change"].apply(bucket)

# Per-bucket summary
print("\n=== lga_phase_std within extended descent, split by 5d price direction ===")
for dir_label in ["down (5d Δ < -2c)", "flat (|5d Δ| ≤ 2c)", "up   (5d Δ > +2c)"]:
    vals = ext.loc[ext["dir"] == dir_label, "lga_phase_std"].dropna()
    n = len(vals)
    if n < 100:
        print(f"  {dir_label:<26}: n={n:>8,}  (insufficient)")
        continue
    print(f"  {dir_label:<26}: n={n:>8,}  mean={vals.mean():6.2f}d  median={vals.median():6.2f}d  "
          f"std={vals.std():5.2f}d  p90={vals.quantile(0.9):6.2f}d")

# Direct comparison: normal late-descent (pct 0.30-0.50) as the anchor
norm = df[(df.cycle_pct_through >= 0.30) & (df.cycle_pct_through < 0.50)].copy()
norm["dir"] = norm["px_5d_change"].apply(bucket)
print("\n=== Same split for NORMAL late-descent (pct 0.30–0.50) — anchor for comparison ===")
for dir_label in ["down (5d Δ < -2c)", "flat (|5d Δ| ≤ 2c)", "up   (5d Δ > +2c)"]:
    vals = norm.loc[norm["dir"] == dir_label, "lga_phase_std"].dropna()
    n = len(vals)
    if n < 100:
        print(f"  {dir_label:<26}: n={n:>8,}  (insufficient)")
        continue
    print(f"  {dir_label:<26}: n={n:>8,}  mean={vals.mean():6.2f}d  median={vals.median():6.2f}d  "
          f"std={vals.std():5.2f}d  p90={vals.quantile(0.9):6.2f}d")

# Verdict
ext_down = ext[ext["dir"] == "down (5d Δ < -2c)"]["lga_phase_std"].dropna()
ext_up = ext[ext["dir"] == "up   (5d Δ > +2c)"]["lga_phase_std"].dropna()
norm_down = norm[norm["dir"] == "down (5d Δ < -2c)"]["lga_phase_std"].dropna()
norm_up = norm[norm["dir"] == "up   (5d Δ > +2c)"]["lga_phase_std"].dropna()

print("\n=== Interpretation ===")
print(f"Extended-descent / DOWN vs Normal-late-descent / DOWN: "
      f"means {ext_down.mean():.2f}d vs {norm_down.mean():.2f}d  "
      f"(diff = {ext_down.mean() - norm_down.mean():+.2f}d)")
print(f"Extended-descent / UP   vs Normal-late-descent / UP  : "
      f"means {ext_up.mean():.2f}d vs {norm_up.mean():.2f}d  "
      f"(diff = {ext_up.mean() - norm_up.mean():+.2f}d)")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
groups = ["down", "flat", "up"]
labels_map = {"down": "down (5d Δ < -2c)", "flat": "flat (|5d Δ| ≤ 2c)", "up": "up   (5d Δ > +2c)"}
x = np.arange(len(groups))
width = 0.35
norm_means = [norm[norm["dir"] == labels_map[g]]["lga_phase_std"].mean() for g in groups]
ext_means = [ext[ext["dir"] == labels_map[g]]["lga_phase_std"].mean() for g in groups]
ax.bar(x - width/2, norm_means, width, label="normal late-descent (pct 0.30–0.50)", color="tab:green")
ax.bar(x + width/2, ext_means, width, label="extended (pct ≥ 0.90)", color="tab:red")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_xlabel("station-level 5-day backward price change direction")
ax.set_ylabel("mean lga_phase_std (days)")
ax.set_title("LGA-leader divergence split by recent price direction\n(does C live in descending or ascending sub-population?)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plot_path = OUT_DIR / "step1d_lga_divergence_by_direction.png"
plt.savefig(plot_path, dpi=120)
print(f"\nPlot saved to {plot_path}")
print(f"\nTotal elapsed: {time.perf_counter()-t0:.1f}s")
