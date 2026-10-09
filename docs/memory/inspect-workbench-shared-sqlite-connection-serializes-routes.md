---
name: inspect-workbench-shared-sqlite-connection-serializes-routes
description: Every fuel_signal.inspect route shares one sqlite3 connection (check_same_thread=False) across waitress threads; SQLite serializes steps on it, so a slow page's aggregate query blocks a concurrent /api/v1 read. Writes must never use it.
metadata:
  type: project
---

`fuel_signal/inspect.py` `main()` opens **one** connection,
`sqlite3.connect(..., check_same_thread=False)` (around line 1159). Every route
closes over it: the heavy workbench pages (`/lead-lag` → `_compute_lead_lag(conn, …)`,
`/classification-health` → `_classification_health_data(conn)`) and the cheap
`/api/v1` reads (`_load_signal_cache` → one-row `SELECT` from `signal_cache`).

Python's `sqlite3.threadsafety == 3` (serialized), so SQLite holds that
connection's mutex for each `sqlite3_step`. An aggregate or sort query does all
its work in its first step, so a cold-disk page query on viking (minutes, right
after a fill) holds the mutex the whole time. **A concurrent `/api/v1` read queues
behind it.** That matters because the app's background fetch has a 10 s request
timeout.

Consequences already designed around (2026-10-10, contract rev 4 / fps-app#142):
- the nightly push is sent **after** the daily script's warm-up loop
  (setup-scripts#17), never before it;
- `POST /api/v1/devices` (#435) must open its own per-request connection. A
  write through the shared connection from a request thread would also race
  other threads' transactions, which today's read-only routes never trigger.

Open fix: #439 moves the `/api/v1` GETs onto per-request connections. WAL allows
concurrent readers on separate connections. Until it lands, don't assume an
`/api/v1` latency measurement is independent of whatever else the workbench is
serving. Related: [[inspect-workbench-cycle-state-frozen-at-startup]].
