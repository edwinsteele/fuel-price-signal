"""Write-path breakdown for classify_range: read+compute vs upsert vs commit.

MUTATES the DB it is given — pass a throwaway copy, never fuel_signal.db.
Runs three disjoint 30-snapshot ranges: baseline, commits suppressed, and
writes+commits suppressed. Differences give the per-snapshot write and fsync cost.
"""

import logging
import pathlib
import sys
import time

logging.disable(logging.CRITICAL)

import fuel_signal.classify as C  # noqa: E402
from fuel_signal.db import open_db  # noqa: E402

DB = sys.argv[1]
N = 30
N_CLASSIFY_SNAPS = 3597
assert "fuel_signal.db" not in DB, "pass a throwaway copy — this script writes"


class NoCommit:
    """Connection proxy that swallows commit() (sqlite3.Connection.commit is read-only)."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def commit(self):
        pass


conn = open_db(pathlib.Path(DB))
print("journal_mode:", conn.execute("PRAGMA journal_mode").fetchone()[0],
      " synchronous:", conn.execute("PRAGMA synchronous").fetchone()[0])

t = time.perf_counter()
C.classify_range(conn, "2024-07-01", "2024-07-30")
base = time.perf_counter() - t

nc = NoCommit(conn)
t = time.perf_counter()
C.classify_range(nc, "2024-08-01", "2024-08-30")
nocommit = time.perf_counter() - t
conn.commit()

for name in ("upsert_station_class_rows", "upsert_classification_summary_rows",
             "delete_station_class_for_date", "delete_classification_summary_for_date"):
    setattr(C, name, lambda *a, **k: None)
t = time.perf_counter()
C.classify_range(nc, "2024-09-01", "2024-09-30")
nowrite = time.perf_counter() - t

print(f"\nclassify_range, {N} snapshots each:")
print(f"  A baseline             {base:5.2f} s  ({base/N*1000:5.1f} ms/snap)")
print(f"  B commits suppressed   {nocommit:5.2f} s  -> commit {(base-nocommit)/N*1000:4.1f} ms/snap"
      f"  (x{N_CLASSIFY_SNAPS} = {(base-nocommit)/N*N_CLASSIFY_SNAPS:.0f} s)")
print(f"  C writes+commits off   {nowrite:5.2f} s  -> write  {(nocommit-nowrite)/N*1000:4.1f} ms/snap"
      f"  (x{N_CLASSIFY_SNAPS} = {(nocommit-nowrite)/N*N_CLASSIFY_SNAPS:.0f} s)")
print(f"  read+compute only      {nowrite/N*1000:5.1f} ms/snap"
      f"  (x{N_CLASSIFY_SNAPS} = {nowrite/N*N_CLASSIFY_SNAPS:.0f} s)")
print(f"  full backfill projection: {base/N*N_CLASSIFY_SNAPS:.0f} s")
