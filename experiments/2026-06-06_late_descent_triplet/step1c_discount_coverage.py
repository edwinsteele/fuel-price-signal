"""Symmetric coverage check for the Discount cohort + competitive-vs-discount
divergence probe across phase regimes (alt to Signal B).

Discount stations: median premium < -10c per classify.PREMIUM_BAND_CENTS.
i.e. structurally cheap relative to their LGA cluster reference.

Usage: PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step1c_discount_coverage.py
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd

from fuel_signal.features import load_features
from fuel_signal.postcode_council import primary_council

OUT_DIR = pathlib.Path("experiments/2026-06-06_late_descent_triplet")

# Phase proxies (must match Step 1)
PHASE_EARLY = (0.05, 0.30)
PHASE_NORMAL_LATE = (0.30, 0.50)
PHASE_EXTENDED = (0.90, np.inf)


def smd(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    pooled = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")


t0 = time.perf_counter()
print("Loading features...")
df = load_features()
df["price_date"] = pd.to_datetime(df["price_date"])
print(f"  {len(df):,} rows, {df.station_code.nunique()} stations  ({time.perf_counter()-t0:.1f}s)")

# Latest stickiness_score per station (negative = discount, positive = sticky)
latest = (
    df.dropna(subset=["stickiness_score"])
    .sort_values("price_date")
    .groupby("station_code")
    .tail(1)[["station_code", "stickiness_score"]]
    .set_index("station_code")
)

# --- 1. Discount cohort count by threshold ---
print("\n=== Discount station count by threshold (premium < -Tc) ===")
print(f"{'Threshold':>12}  {'Stations':>9}  {'% fleet':>8}")
for T in [-1, -2, -5, -7, -10, -12, -15]:
    n = int((latest["stickiness_score"] < T).sum())
    print(f"   <{T:>4.0f}c  {n:>9,}  {100*n/len(latest):>7.1f}%")

# --- 2. Per-LGA distribution at canonical -10c ---
print("\nLoading station → postcode map from latest snapshot...")
snap_files = sorted(pathlib.Path("data/snapshots").rglob("*.csv"))
snap = pd.read_csv(snap_files[-1], usecols=["station_code", "postcode"]).drop_duplicates("station_code")
snap["postcode"] = snap["postcode"].astype(str).str.zfill(4)
snap["lga"] = snap["postcode"].map(primary_council)
station_lga = snap.set_index("station_code")["lga"]
latest["lga"] = station_lga
mapped = latest.dropna(subset=["lga"])

T = -10
per_lga = mapped.assign(is_disc=lambda x: x["stickiness_score"] < T).groupby("lga").agg(
    total=("stickiness_score", "size"),
    n_discount=("is_disc", "sum"),
)
per_lga = per_lga.sort_values("n_discount", ascending=False)
print(f"\n=== Discount stations (<{T}c) per LGA — canonical threshold ===")
print(per_lga.to_string())
print(f"\nLGAs with ZERO discount stations at <{T}c: "
      f"{(per_lga['n_discount']==0).sum()} of {len(per_lga)}")
print(f"LGAs with 3+ discount stations: "
      f"{(per_lga['n_discount']>=3).sum()} of {len(per_lga)}")
print(f"LGAs with 5+ discount stations: "
      f"{(per_lga['n_discount']>=5).sum()} of {len(per_lga)}")

# --- 3. Coverage table at relaxed thresholds ---
print(f"\n=== Per-LGA coverage at relaxed discount thresholds ===")
print(f"{'LGA':<25}  {'<-10c':>5}  {'<-7c':>5}  {'<-5c':>5}  {'<-2c':>5}  {'total':>6}")
for lga, _ in per_lga.iterrows():
    sub = mapped[mapped["lga"] == lga]
    print(f"{str(lga):<25}  "
          f"{int((sub['stickiness_score']<-10).sum()):>5}  "
          f"{int((sub['stickiness_score']<-7).sum()):>5}  "
          f"{int((sub['stickiness_score']<-5).sum()):>5}  "
          f"{int((sub['stickiness_score']<-2).sum()):>5}  "
          f"{len(sub):>6}")

# --- 4. Divergence probe: comp_p50 vs discount cohort per date, by phase regime ---
# We need station-level price + classification flag for each row. The feature
# row already carries stickiness_score, so we can flag at row level.
print("\n=== Divergence probe: median(competitive) − median(discount) per date ===")

# Use threshold variants so we can pick whichever cohort is viable
def divergence_gap(thresh_disc, thresh_sticky, label):
    df["_is_disc"] = df["stickiness_score"] < thresh_disc
    df["_is_comp"] = (df["stickiness_score"] >= thresh_disc) & (df["stickiness_score"] <= thresh_sticky)
    # Per-date medians
    disc_med = df[df["_is_disc"]].groupby("price_date")["station_price_cents"].median()
    comp_med = df[df["_is_comp"]].groupby("price_date")["station_price_cents"].median()
    gap_by_date = (comp_med - disc_med).rename("gap")
    # join back
    df["gap"] = df["price_date"].map(gap_by_date)
    early = (df.cycle_pct_through >= PHASE_EARLY[0]) & (df.cycle_pct_through < PHASE_EARLY[1])
    norm = (df.cycle_pct_through >= PHASE_NORMAL_LATE[0]) & (df.cycle_pct_through < PHASE_NORMAL_LATE[1])
    extd = df.cycle_pct_through >= PHASE_EXTENDED[0]
    e_vals = df.loc[early, "gap"].values
    n_vals = df.loc[norm, "gap"].values
    x_vals = df.loc[extd, "gap"].values
    print(f"\n  --- {label} (discount<{thresh_disc}c, comp ∈ [{thresh_disc},{thresh_sticky}]c) ---")
    print(f"    n discount stations: {df.loc[df['_is_disc']].station_code.nunique()}")
    print(f"    n competitive stations: {df.loc[df['_is_comp']].station_code.nunique()}")
    print(f"    gap (comp_med − disc_med), by phase:")
    print(f"      early   (pct 0.05–0.30): mean={np.nanmean(e_vals):+.2f}c  std={np.nanstd(e_vals):.2f}c  n={(~np.isnan(e_vals)).sum():,}")
    print(f"      normal  (pct 0.30–0.50): mean={np.nanmean(n_vals):+.2f}c  std={np.nanstd(n_vals):.2f}c  n={(~np.isnan(n_vals)).sum():,}")
    print(f"      extended(pct ≥ 0.90)   : mean={np.nanmean(x_vals):+.2f}c  std={np.nanstd(x_vals):.2f}c  n={(~np.isnan(x_vals)).sum():,}")
    print(f"    SMD (extended − normal): {smd(x_vals, n_vals):+.3f}")
    print(f"    SMD (early − normal)   : {smd(e_vals, n_vals):+.3f}")
    return gap_by_date

# Try multiple thresholds
divergence_gap(-10, 10, "canonical (-10 / +10)")
divergence_gap(-5,  5,  "relaxed (-5 / +5)")
divergence_gap(-2,  2,  "very relaxed (-2 / +2)")

print(f"\nTotal elapsed: {time.perf_counter()-t0:.1f}s")
