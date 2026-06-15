"""Cheap sanity check of the regime denominator (no LightGBM). #254.

Confirms: (1) the recompute path runs end-to-end through cycle.py, (2) the
regime denominator tracks the COVID step (~28d pre -> ~40d post) where the
baseline expanding mean lags, (3) PIT monotonic non-retraction of the regime
estimate, (4) fold-7-era inflation is removed. Run before the heavy WFCV.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cycle_regime import BREAK_DATE, RegimeCycleDetector  # noqa: E402

from fuel_signal import db as _db  # noqa: E402
from fuel_signal.cycle import CycleDetector  # noqa: E402

conn = _db.open_db()
series = _db.average_price_series(conn)
conn.close()
print(f"avg series: {len(series)} days  {series[0][0]} .. {series[-1][0]}")

base = CycleDetector(series)
regime = RegimeCycleDetector(series)

checkpoints = [
    "2018-06-30", "2019-12-31", "2020-04-15", "2020-09-30",
    "2021-11-05", "2022-06-01", "2023-06-01", "2024-06-01", "2026-05-30",
]
print(f"\n{'date':>12}  {'base_mean':>9}  {'regime':>7}  {'base_pct':>8}  {'reg_pct':>7}")
for d in checkpoints:
    b = base.detect(d)
    r = regime.detect(d)
    if b is None or r is None:
        print(f"{d:>12}  (None)")
        continue
    print(f"{d:>12}  {b.mean_cycle_length:>9.2f}  {r.mean_cycle_length:>7.2f}  "
          f"{b.pct_through_cycle:>8.3f}  {r.pct_through_cycle:>7.3f}")

# PIT non-retraction: regime estimate must not jump down on added future data
# (it is an expanding median, monotone in information once post-break warm-up
# converges). Scan monthly post-break and report any large downward steps.
dates = pd.date_range("2020-04-01", series[-1][0], freq="MS")
prev = None
drops = []
traj = []
for ts in dates:
    r = regime.detect(ts.strftime("%Y-%m-%d"))
    if r is None:
        continue
    traj.append((ts.date(), r.mean_cycle_length))
    if prev is not None and r.mean_cycle_length < prev - 3.0:
        drops.append((ts.date(), prev, r.mean_cycle_length))
    prev = r.mean_cycle_length
print(f"\npost-break regime trajectory: {len(traj)} monthly points, "
      f"range {min(v for _, v in traj):.1f}..{max(v for _, v in traj):.1f}d")
print(f"large month-on-month downward steps (>3d): {len(drops)}")
for dt, a, b_ in drops:
    print(f"   {dt}: {a:.1f} -> {b_:.1f}")

# fold-7-era inflation check (the falsifiable framing): how much does the
# baseline under-report cycle length, and therefore inflate pct_through, in
# fold 7's window vs the regime denominator?
f7_lo, f7_hi = pd.Timestamp("2023-05-01"), pd.Timestamp("2023-08-01")
sample = pd.date_range(f7_lo, f7_hi, freq="7D")
bm = np.median([base.detect(d.strftime("%Y-%m-%d")).mean_cycle_length for d in sample])
rm = np.median([regime.detect(d.strftime("%Y-%m-%d")).mean_cycle_length for d in sample])
print(f"\nfold-7 era ({f7_lo.date()}..{f7_hi.date()}): "
      f"base median {bm:.1f}d  regime median {rm:.1f}d  "
      f"=> baseline inflates pct_through by ~{rm / bm:.2f}x")

print(f"\nBREAK_DATE = {BREAK_DATE.date()}")
print("OK" if rm > bm else "WARN: regime not above baseline in fold-7 era")
