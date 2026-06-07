"""Sticky-station coverage diagnostic for Signal B (sticky-floor gap).

Two questions:
  1. How many distinct stations qualify as "sticky" at threshold T?
     Sensitivity table over T ∈ {1, 2, 5, 7, 10, 15} cents.
  2. At the canonical threshold (10c per classify.PREMIUM_BAND_CENTS),
     are they distributed across LGAs?

Usage: PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step1b_sticky_coverage.py
"""
from __future__ import annotations

import pathlib
import time

import pandas as pd

from fuel_signal.features import load_features
from fuel_signal.postcode_council import primary_council

OUT_DIR = pathlib.Path("experiments/2026-06-06_late_descent_triplet")

# --- 1. Per-station stickiness_score ---
t0 = time.perf_counter()
print("Loading features...")
df = load_features()
df["price_date"] = pd.to_datetime(df["price_date"])
print(f"  {len(df):,} rows, {df.station_code.nunique()} stations  ({time.perf_counter()-t0:.1f}s)")

# Most recent stickiness_score per station (PIT-safe: latest observation)
latest = (
    df.dropna(subset=["stickiness_score"])
    .sort_values("price_date")
    .groupby("station_code")
    .tail(1)[["station_code", "stickiness_score"]]
    .set_index("station_code")
)
print(f"  stations with any stickiness_score: {len(latest):,}")

# --- 2. Threshold sensitivity ---
print("\n=== Sticky station count by threshold ===")
print(f"{'Threshold':>10}  {'Stations':>9}  {'% of fleet':>11}")
total = len(latest)
for T in [1, 2, 5, 7, 10, 12, 15]:
    n = int((latest["stickiness_score"] > T).sum())
    print(f"   >{T:>3.0f}c  {n:>9,}  {100*n/total:>10.1f}%")

# --- 3. Map station_code → postcode → LGA via most recent snapshot ---
print("\nLoading most recent snapshot for station → postcode map...")
snap_files = sorted(pathlib.Path("data/snapshots").rglob("*.csv"))
latest_snap = snap_files[-1]
print(f"  using {latest_snap}")
snap = pd.read_csv(latest_snap, usecols=["station_code", "postcode"]).drop_duplicates("station_code")
snap["postcode"] = snap["postcode"].astype(str).str.zfill(4)
snap["lga"] = snap["postcode"].map(primary_council)
station_lga = snap.set_index("station_code")["lga"]
print(f"  stations in snapshot: {len(snap):,}, with mapped LGA: {snap['lga'].notna().sum():,}")

# Merge LGA onto latest stickiness scores
latest["lga"] = station_lga
mapped = latest.dropna(subset=["lga"])
print(f"  stations with sticky-score AND LGA: {len(mapped):,}")

# --- 4. Per-LGA distribution at canonical threshold (10c) ---
T = 10
sticky = mapped[mapped["stickiness_score"] > T]
print(f"\n=== Sticky stations (>{T}c) per LGA — canonical threshold ===")
per_lga = mapped.assign(is_sticky=lambda x: x["stickiness_score"] > T).groupby("lga").agg(
    total_stations=("stickiness_score", "size"),
    n_sticky=("is_sticky", "sum"),
)
per_lga["sticky_pct"] = (100 * per_lga["n_sticky"] / per_lga["total_stations"]).round(1)
per_lga = per_lga.sort_values("n_sticky", ascending=False)
print(per_lga.to_string())

print(f"\nLGAs with ZERO sticky stations at >{T}c: "
      f"{(per_lga['n_sticky'] == 0).sum()} of {len(per_lga)}")
print(f"LGAs with FEWER THAN 3 sticky stations at >{T}c: "
      f"{(per_lga['n_sticky'] < 3).sum()} of {len(per_lga)}")
print(f"LGAs with 5+ sticky stations at >{T}c: "
      f"{(per_lga['n_sticky'] >= 5).sum()} of {len(per_lga)}")

# --- 5. Per-LGA at relaxed thresholds (would more sticky stations be available?) ---
print(f"\n=== Per-LGA coverage at relaxed thresholds ===")
print(f"{'LGA':<25}  {'>10c':>5}  {'>7c':>5}  {'>5c':>5}  {'>2c':>5}  {'total':>6}")
for lga, row in per_lga.iterrows():
    lga_stations = mapped[mapped["lga"] == lga]
    n10 = int((lga_stations["stickiness_score"] > 10).sum())
    n7  = int((lga_stations["stickiness_score"] >  7).sum())
    n5  = int((lga_stations["stickiness_score"] >  5).sum())
    n2  = int((lga_stations["stickiness_score"] >  2).sum())
    total = len(lga_stations)
    print(f"{str(lga):<25}  {n10:>5}  {n7:>5}  {n5:>5}  {n2:>5}  {total:>6}")

print(f"\nTotal elapsed: {time.perf_counter()-t0:.1f}s")
