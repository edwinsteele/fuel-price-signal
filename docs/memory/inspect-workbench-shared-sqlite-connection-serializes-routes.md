---
name: inspect-workbench-shared-sqlite-connection-serializes-routes
description: Workbench pages still share inspect.py's connection, so long SQLite steps serialize those routes; /api/v1 cache GETs and device writes use short-lived request connections instead.
metadata:
  type: project
---

`fuel_signal/inspect.py` `main()` opens one connection with
`sqlite3.connect(..., check_same_thread=False)` for the workbench routes. Python's
`sqlite3.threadsafety == 3` (serialized), so SQLite holds that connection's mutex
for each `sqlite3_step`. Aggregate or sort queries do their work in the first
step, so a cold-disk page query on viking (minutes, right after a fill) can hold
the mutex for the whole computation and queue other workbench routes using that
same connection.

The `/api/v1` cache reads no longer use this shared connection (#439). Each GET
opens `db.open_request_connection` for its cache read and closes it afterward;
`POST /api/v1/devices` follows the same per-request pattern. Separate WAL
connections let these API requests proceed while a workbench query holds the
workbench connection. Keep the shared connection for workbench pages, and use
per-request connections for any future API route that writes or needs isolation.

Related: [[inspect-workbench-cycle-state-frozen-at-startup]].
