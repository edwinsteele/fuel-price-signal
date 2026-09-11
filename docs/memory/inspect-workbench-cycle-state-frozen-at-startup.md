---
name: inspect-workbench-cycle-state-frozen-at-startup
description: fuel_signal.inspect's cd/cycle_state/peak_data/summary/boundaries are built once in main() and never refreshed; freshness depends on a service restart, not per-request queries.
metadata:
  type: project
---

`fuel_signal/inspect.py`'s `main()` builds `CycleDetector` (`cd`), `cycle_state`, `peak_data`,
`summary`, `boundaries`, and `today` once at process startup and passes them into `_create_app`,
which closes over them for the life of the process (#419). Every route that reads these — chiefly
`/` (`index()`), but `summary`/`today` also reach `/lead-lag`, `/classification-health`, and
`/features` — serves whatever was true when the workbench last started, not the live DB.

This is deliberate, not a bug to "fix" by rebuilding per request: #419 measured the rebuild cost
(`CycleDetector` construction is ~1-10ms even over the full daily series; the dominant cost is the
`average_price_series()` `GROUP BY` query) and decided in favour of restarting the workbench
service after each daily data load instead, keeping routes simple and cheap. **That restart lives
in the deployment repo (`setup-scripts`'s `ansible/roles/fuel_signal`, specifically the
`fuelsignal-daily-update.sh` template), not here** — as of 2026-09-12 that script does not restart
`fuelsignal-workbench.service`, and its own comment incorrectly claims no restart is needed because
routes "query SQLite fresh per request", which is only true of the routes that query through `conn`
directly. Flagged to the owner in #419; fix pending in that other repo.

The one on-page staleness indicator is the "Cycle State — as of {{ today }}" heading in
`workbench.html` — `today` is the last date that was in `daily_prices` when the process started, so
comparing it to the real current date reveals staleness. Don't confuse it with the page's other "as
of {{ now }}" line near the top, which is the per-request render time and is always current — that
one says nothing about data freshness.

**How to apply:** if you see the workbench showing an implausibly old cycle phase, check whether
`fuelsignal-workbench.service` has been restarted since the last `daily_prices` write, not whether
the code is querying stale data — it's process age, not a query bug. If you're the one fixing the
deployment side, see the `setup-scripts` repo directly; do not carry that fix in this repo.
