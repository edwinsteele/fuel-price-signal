"""Read-only component profile of classify_range / score_leadership_range.

Splits each backfill step into per-snapshot SQL vs python compute, measures the
memory cost of the naive "load the whole range once" alternative, and checks
whether ORDER BY price_date forces a sort.
"""

import sqlite3
import sys
import time
import tracemalloc
from collections import defaultdict
from datetime import date, timedelta

from fuel_signal import classify as C
from fuel_signal import lga_leadership as L
from fuel_signal.dates import date_to_int
from fuel_signal.db import daily_prices_in_window, fuel_type_id

DB = sys.argv[1] if len(sys.argv) > 1 else "fuel_signal.db"
HISTORY_START, HISTORY_END = "2016-08-01", "2026-06-06"
N_CLASSIFY_SNAPS = 3597   # distinct daily_prices dates
N_LEADERSHIP_SNAPS = 523  # weekly snapshots over the same span

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
FID = fuel_type_id(conn, "E10")


def best_of(fn, n=3):
    t = 1e9
    for _ in range(n):
        s = time.perf_counter()
        r = fn()
        t = min(t, time.perf_counter() - s)
    return t, r


def window(snapshot, days):
    d = date.fromisoformat(snapshot)
    return (d - timedelta(days=days)).isoformat(), (d - timedelta(days=1)).isoformat()


print("== DB shape ==")
for q in (
    "SELECT COUNT(*) FROM daily_prices",
    "SELECT COUNT(DISTINCT price_date) FROM daily_prices",
    "SELECT COUNT(*) FROM stations WHERE council IS NOT NULL",
    "SELECT COUNT(DISTINCT council) FROM stations WHERE council IS NOT NULL",
):
    print(f"  {q[:60]:60s} {conn.execute(q).fetchone()[0]:,}")

print("\n== classify.py: per-snapshot cost (45d window) ==")
start, end = window("2024-06-01", C.WINDOW_DAYS)
t_sql, rows = best_of(lambda: daily_prices_in_window(conn, start, end))
print(f"  daily_prices_in_window  {len(rows):,} rows  {t_sql*1000:6.1f} ms"
      f"  -> x{N_CLASSIFY_SNAPS} = {t_sql*N_CLASSIFY_SNAPS:.0f} s")


def classify_compute(rows):
    by_lga = defaultdict(lambda: defaultdict(dict))
    for station_code, lga, d, price in rows:
        by_lga[lga][station_code][d] = price
    return sum(len(C._classify_lga({c: dict(x) for c, x in sp.items()})[0])
               for sp in by_lga.values())


t_cmp, n = best_of(lambda: classify_compute(rows))
print(f"  python compute          {n:,} classes  {t_cmp*1000:6.1f} ms"
      f"  -> x{N_CLASSIFY_SNAPS} = {t_cmp*N_CLASSIFY_SNAPS:.0f} s")
print(f"  SQL share of read+compute: {t_sql/(t_sql+t_cmp)*100:.0f}%")

print("\n== lga_leadership.py: per-snapshot cost (730d window) ==")
start, end = window("2024-06-01", L.LEADERSHIP_WINDOW_DAYS)
t_sql_l, sums = best_of(
    lambda: L._load_lga_sums(conn, FID, start_date=start, end_date=end), n=2)
print(f"  _load_lga_sums          {len(sums):,} groups  {t_sql_l*1000:6.1f} ms"
      f"  -> x{N_LEADERSHIP_SNAPS} = {t_sql_l*N_LEADERSHIP_SNAPS:.0f} s")


def leadership_compute(sums):
    import numpy as np

    from fuel_signal.dates import int_to_date
    sydney_sum, sydney_n = {}, {}
    for (di, _lga), (s, k) in sums.items():
        sydney_sum[di] = sydney_sum.get(di, 0.0) + s
        sydney_n[di] = sydney_n.get(di, 0) + k
    by_lga = {}
    for (di, lga), v in sums.items():
        by_lga.setdefault(lga, {})[di] = v
    for lga, lga_data in by_lga.items():
        lga_dates = sorted(lga_data)
        lga_prices = np.array([lga_data[d][0] / lga_data[d][1] for d in lga_dates])
        anchor = {}
        for d in sorted(sydney_sum):
            ls, ln = lga_data.get(d, (0.0, 0))
            a_s, a_n = sydney_sum[d] - ls, sydney_n[d] - ln
            if a_n >= L.MIN_STATION_FLOOR:
                anchor[d] = a_s / a_n
        ad = sorted(anchor)
        ap = np.array([anchor[d] for d in ad])
        li = L.detect_trough_events(lga_prices)
        ai = L.detect_trough_events(ap)
        L._match_events([int_to_date(lga_dates[i]) for i in li],
                        [int_to_date(ad[i]) for i in ai])
    return len(by_lga)


t_cmp_l, n_lga = best_of(lambda: leadership_compute(sums), n=2)
print(f"  python compute          {n_lga} LGAs  {t_cmp_l*1000:6.1f} ms"
      f"  -> x{N_LEADERSHIP_SNAPS} = {t_cmp_l*N_LEADERSHIP_SNAPS:.0f} s")
print(f"  SQL share of read+compute: {t_sql_l/(t_sql_l+t_cmp_l)*100:.0f}%")

print("\n== naive 'load the full range once' cost + memory ==")
tracemalloc.start()
t = time.perf_counter()
allrows = daily_prices_in_window(conn, HISTORY_START, HISTORY_END)
dt = time.perf_counter() - t
cur, peak = tracemalloc.get_traced_memory()
print(f"  classify   full load  {len(allrows):,} rows  {dt:.1f} s"
      f"  resident {cur/1e6:.0f} MB  peak {peak/1e6:.0f} MB")
del allrows
tracemalloc.stop()

tracemalloc.start()
t = time.perf_counter()
allsums = L._load_lga_sums(conn, FID)
dt = time.perf_counter() - t
cur, peak = tracemalloc.get_traced_memory()
print(f"  leadership full load  {len(allsums):,} groups  {dt:.1f} s"
      f"  resident {cur/1e6:.0f} MB  peak {peak/1e6:.0f} MB")
tracemalloc.stop()

print("\n== is ORDER BY price_date free? ==")
sql = ("SELECT dp.station_code, s.council, dp.price_date, dp.price_decicents"
       " FROM daily_prices dp JOIN stations s USING(station_code)"
       " WHERE dp.fuel_type_id = ? AND s.council IS NOT NULL"
       "   AND dp.price_date >= ? AND dp.price_date <= ?")
lo, hi = date_to_int(HISTORY_START), date_to_int(HISTORY_END)
for suffix in ("", " ORDER BY dp.price_date"):
    plan = [r[-1] for r in conn.execute("EXPLAIN QUERY PLAN " + sql + suffix, (FID, lo, hi))]
    t = time.perf_counter()
    n = sum(1 for _ in conn.execute(sql + suffix, (FID, lo, hi)))
    dt = time.perf_counter() - t
    print(f"  {suffix or '(unordered)':28s} {n:,} rows  {dt:.2f} s")
    for p in plan:
        print(f"      {p}")
