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
`fuelsignal-daily-update.sh` template), not here.** It already does this: `systemctl restart
fuelsignal-workbench.service` was added there in `setup-scripts` commit `c389325` (2026-09-06),
which also corrected that template's comment (previously claimed no restart was needed — see
`setup-scripts` issue #15, filed against a stale local checkout that predated `c389325` and closed
same-day as a duplicate of already-shipped work). **Do not re-file this against setup-scripts** —
check `git log`/`grep restart` in a freshly-fetched `setup-scripts` checkout first; this repo has no
way to detect drift in that one.

The one on-page staleness indicator is the "Cycle State — as of {{ today }}" heading in
`workbench.html` — `today` is the last date that was in `daily_prices` when the process started, so
comparing it to the real current date reveals staleness. Don't confuse it with the page's other "as
of {{ now }}" line near the top, which is the per-request render time and is always current — that
one says nothing about data freshness.

**How to apply:** if you see the workbench showing an implausibly old cycle phase, first check
whether `fuelsignal-workbench.service` actually restarted after the last `daily_prices` write
(journal/systemd, not this memory) — the deployment-side fix has been in place since 2026-09-06,
so a stale phase now points at something else (the restart failing, a slow warm-up — see
`setup-scripts` commit `c6696ff` same day — or a genuinely new regression), not at this repo's code
or at the restart being unwired.
