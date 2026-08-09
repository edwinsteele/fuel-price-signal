"""Reference prototypes for fps-53j (lga_leadership) and fps-2us (classify),
with a parity check against the current per-snapshot queries.

Read-only. The two functions below are the shapes to lift into fuel_signal/;
each is followed by a check that its windows are identical to what
_load_lga_sums / daily_prices_in_window return today.
"""

import logging
import sqlite3
import sys
import time
from collections import defaultdict, deque
from datetime import date, timedelta

logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402

import fuel_signal.classify as C  # noqa: E402
from fuel_signal import lga_leadership as L  # noqa: E402
from fuel_signal.dates import date_to_int  # noqa: E402
from fuel_signal.db import _date_from_int, daily_prices_in_window, fuel_type_id  # noqa: E402

DB = sys.argv[1] if len(sys.argv) > 1 else "fuel_signal.db"
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
FID = fuel_type_id(conn, "E10")


# --------------------------------------------------------------------------
# fps-2us — classify: one ordered scan feeding a sliding deque of day buckets.
#
# Resident set is O(window_days), not O(history): the naive full-range load is
# 540 MB / 881 MB peak (see profile_backfill.py) and OOMs Viking.
# Yields the deque itself — flattening to a row list per snapshot rebuilds
# ~31k tuples each time and gives back most of the win.
# --------------------------------------------------------------------------
def day_bucket_stream(conn, start_date, end_date, window_days=C.WINDOW_DAYS, fuel_code="E10"):
    """Yield (snapshot_date, deque of (date_int, date_str, [(station, lga, cents)]))."""
    fid = fuel_type_id(conn, fuel_code)
    scan_lo = date_to_int(
        (date.fromisoformat(start_date) - timedelta(days=window_days)).isoformat())
    scan_hi = date_to_int(
        (date.fromisoformat(end_date) - timedelta(days=1)).isoformat())
    cur = conn.execute(
        "SELECT dp.price_date, dp.station_code, s.council, dp.price_decicents"
        " FROM daily_prices dp JOIN stations s USING(station_code)"
        " WHERE dp.fuel_type_id = ? AND s.council IS NOT NULL"
        "   AND dp.price_date >= ? AND dp.price_date <= ?"
        " ORDER BY dp.price_date",
        (fid, scan_lo, scan_hi),
    )

    buf: deque = deque()
    pending = cur.fetchone()
    d, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    while d <= end:
        hi = date_to_int((d - timedelta(days=1)).isoformat())
        lo = date_to_int((d - timedelta(days=window_days)).isoformat())
        while pending is not None and pending[0] <= hi:
            day_int = pending[0]
            day_str = _date_from_int(day_int)
            day = []
            while pending is not None and pending[0] == day_int:
                day.append((pending[1], pending[2], pending[3] / 10))
                pending = cur.fetchone()
            buf.append((day_int, day_str, day))
        while buf and buf[0][0] < lo:
            buf.popleft()
        yield d.isoformat(), buf
        d += timedelta(days=1)


# --------------------------------------------------------------------------
# fps-53j — lga_leadership: one full-range load, sliced per snapshot.
#
# Safe to hold entire history in memory (30 MB) because the SQL has already
# aggregated to (date, LGA). Parity is structural: the HAVING groups on
# (price_date, council) and every group lives inside one date, so the window
# filter can never change a group's membership.
# --------------------------------------------------------------------------
def lga_sums_slicer(conn, fid):
    """Return slice(start_date, end_date) -> same dict shape as _load_lga_sums."""
    by_date: dict[int, dict[str, tuple[float, int]]] = defaultdict(dict)
    for (date_int, lga), value in L._load_lga_sums(conn, fid).items():
        by_date[date_int][lga] = value
    dates = np.array(sorted(by_date), dtype=int)

    def slice_window(start_date, end_date):
        lo = int(np.searchsorted(dates, date_to_int(start_date), side="left"))
        hi = int(np.searchsorted(dates, date_to_int(end_date), side="right"))
        return {
            (int(di), lga): v
            for di in dates[lo:hi]
            for lga, v in by_date[int(di)].items()
        }

    return slice_window


# --------------------------------------------------------------------------
# Parity + speed
# --------------------------------------------------------------------------
CL_START, CL_END = "2024-06-01", "2024-07-15"  # 45 consecutive daily snapshots

print("== fps-2us classify: window parity ==")
t = time.perf_counter()
old = {}
d = date.fromisoformat(CL_START)
while d <= date.fromisoformat(CL_END):
    s = (d - timedelta(days=C.WINDOW_DAYS)).isoformat()
    e = (d - timedelta(days=1)).isoformat()
    old[d.isoformat()] = sorted(daily_prices_in_window(conn, s, e))
    d += timedelta(days=1)
t_old = time.perf_counter() - t

t = time.perf_counter()
new = {
    snap: sorted((st, lga, ds, p) for _, ds, day in buf for st, lga, p in day)
    for snap, buf in day_bucket_stream(conn, CL_START, CL_END)
}
t_new = time.perf_counter() - t
print(f"  identical: {old == new}  ({len(old)} windows,"
      f" {len(next(iter(old.values()))):,} rows/window)")
print(f"  fetch {t_old:.2f} s -> {t_new:.2f} s  ({t_old/t_new:.1f}x)")

print("\n== fps-2us classify: read+compute through _classify_lga ==")


def run_classify(source):
    total = 0
    for _snap, buf in source:
        by_lga = defaultdict(lambda: defaultdict(dict))
        for _di, ds, day in buf:
            for st, lga, p in day:
                by_lga[lga][st][ds] = p
        for sp in by_lga.values():
            total += len(C._classify_lga({c: dict(x) for c, x in sp.items()})[0])
    return total


def current_source(start_date, end_date):
    d = date.fromisoformat(start_date)
    while d <= date.fromisoformat(end_date):
        s = (d - timedelta(days=C.WINDOW_DAYS)).isoformat()
        e = (d - timedelta(days=1)).isoformat()
        rows = daily_prices_in_window(conn, s, e)
        by_day: dict[str, list] = defaultdict(list)
        for st, lga, ds, p in rows:
            by_day[ds].append((st, lga, p))
        yield d.isoformat(), [(0, ds, v) for ds, v in by_day.items()]
        d += timedelta(days=1)


t = time.perf_counter()
n_old = run_classify(current_source(CL_START, CL_END))
t_old = time.perf_counter() - t
t = time.perf_counter()
n_new = run_classify(day_bucket_stream(conn, CL_START, CL_END))
t_new = time.perf_counter() - t
print(f"  classes {n_old:,} == {n_new:,}: {n_old == n_new}")
print(f"  {len(old)} snaps: {t_old:.2f} s -> {t_new:.2f} s  ({t_old/t_new:.1f}x)")
print(f"  projected x3597 (scan amortised): {t_old/45*3597:.0f} s"
      f" -> ~{2.5 + (t_new - 0.35)/45*3597:.0f} s")

print("\n== fps-53j leadership: window parity ==")
SNAPS = ["2024-01-04", "2024-03-07", "2025-06-05", "2022-02-03"]
t = time.perf_counter()
slicer = lga_sums_slicer(conn, FID)
t_build = time.perf_counter() - t

ok = True
t_old = t_new = 0.0
for snap in SNAPS:
    sd = date.fromisoformat(snap)
    s = (sd - timedelta(days=L.LEADERSHIP_WINDOW_DAYS)).isoformat()
    e = (sd - timedelta(days=1)).isoformat()
    t = time.perf_counter()
    a = L._load_lga_sums(conn, FID, start_date=s, end_date=e)
    t_old += time.perf_counter() - t
    t = time.perf_counter()
    b = slicer(s, e)
    t_new += time.perf_counter() - t
    ok &= a == b
n = len(SNAPS)
print(f"  identical: {ok}  ({n} sampled snapshots)")
print(f"  {t_old/n*1000:.0f} ms/snap -> {t_new/n*1000:.1f} ms/snap"
      f"  (+{t_build:.1f} s one-off build)")
print(f"  projected x523: {t_old/n*523:.0f} s -> {t_build + t_new/n*523:.1f} s"
      f"  ({(t_old/n*523)/(t_build + t_new/n*523):.0f}x)")
