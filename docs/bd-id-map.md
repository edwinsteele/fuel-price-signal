# bd → GitHub issue map

Written by the Beads → GitHub cutover, 2026-09-07 (`migration/migrate_to_github.py`, PLAN_beads_cutover.md phase 3).

GitHub issue and PR numbers share one sequence, so these numbers start at 365 — well clear of the pre-Beads issues (#1–#273) that some of these reference.

## Migrated (24)

The 23 issues that were live in bd, plus `fps-3jj` — a *closed* parent with four open children, migrated as a live tracking issue because GitHub sub-issues require one (PLAN 3.1).

| bd id | GitHub | title |
|---|---|---|
| `fps-1785999730419-8-28589a4e` | [#365](https://github.com/edwinsteele/fuel-price-signal/issues/365) | Revisit FP penalty using model-conditional damage distribution (View 3) |
| `fps-1785999730624-10-3a7a4d8d` | [#366](https://github.com/edwinsteele/fuel-price-signal/issues/366) | Tune label percentile hyperparameter (25th vs 33rd vs other) |
| `fps-3fn` | [#367](https://github.com/edwinsteele/fuel-price-signal/issues/367) | Pull TGP data from api.aip.com.au/public/tgpTables (1-2 day lag) alongside weekly xlsx |
| `fps-3jj` | [#368](https://github.com/edwinsteele/fuel-price-signal/issues/368) | Design: AI-sourced feature engineering pipeline (nightly local routine) _(parent exception)_ |
| `fps-3jj.10` | [#369](https://github.com/edwinsteele/fuel-price-signal/issues/369) | Pipeline: human-sourced leads for the second AI batch (held back) |
| `fps-3jj.18` | [#370](https://github.com/edwinsteele/fuel-price-signal/issues/370) | batch1 — first AI-sourced candidate batch (5 candidates, generated 2026-08-23) |
| `fps-3jj.18.2` | [#371](https://github.com/edwinsteele/fuel-price-signal/issues/371) | batch1: act on the outcome — graduate, iterate or record as mapped ground |
| `fps-3jj.19` | [#372](https://github.com/edwinsteele/fuel-price-signal/issues/372) | batch2: freeze the batch and run the generator (10-15 candidates) |
| `fps-3jj.22` | [#373](https://github.com/edwinsteele/fuel-price-signal/issues/373) | Wide (high-arity) candidates pay an effective-draw penalty — shrink it (placebo sources outside the lock) if one is ever proposed |
| `fps-490` | [#374](https://github.com/edwinsteele/fuel-price-signal/issues/374) | Measure each locked feature block against the noise floor in realised CPL |
| `fps-55e` | [#375](https://github.com/edwinsteele/fuel-price-signal/issues/375) | write_meta hardcodes meta.json, so two scripts in one experiment dir clobber each other |
| `fps-77s` | [#376](https://github.com/edwinsteele/fuel-price-signal/issues/376) | StationPriceSource can't distinguish mid-series gap from out-of-range |
| `fps-9rw` | [#377](https://github.com/edwinsteele/fuel-price-signal/issues/377) | Apply ptp-exactness fix to std==0 degeneracy checks outside dossier_tables.py |
| `fps-evn` | [#378](https://github.com/edwinsteele/fuel-price-signal/issues/378) | Decision-timing accuracy: a zone diagnostic that never allocates cost to a period |
| `fps-ghr` | [#379](https://github.com/edwinsteele/fuel-price-signal/issues/379) | fill.py's 28-day gap cap cannot distinguish a station closure from a source-data outage |
| `fps-gzz` | [#380](https://github.com/edwinsteele/fuel-price-signal/issues/380) | decision_flips is blind to volume-only changes (same date, different litres) |
| `fps-hc7` | [#381](https://github.com/edwinsteele/fuel-price-signal/issues/381) | Thread --n-stations through launch.py's build_runner_cmd |
| `fps-ie0` | [#382](https://github.com/edwinsteele/fuel-price-signal/issues/382) | Wire check_freeze_cadence into the candidate runner |
| `fps-r13` | [#383](https://github.com/edwinsteele/fuel-price-signal/issues/383) | Act on the locked-block ablation: set the screen bar, and test whether removing a block helps |
| `fps-sk0` | [#384](https://github.com/edwinsteele/fuel-price-signal/issues/384) | Worker Routine: bd dolt push blocked by egress proxy 403 on Dolt git-ref namespace _(closed as moot)_ |
| `fps-v31` | [#385](https://github.com/edwinsteele/fuel-price-signal/issues/385) | Proactive push/email notification for the signal |
| `fps-wac` | [#386](https://github.com/edwinsteele/fuel-price-signal/issues/386) | Pin realised.py <-> dossier_tables.py's tau_diverges column contract end to end |
| `fps-wst` | [#387](https://github.com/edwinsteele/fuel-price-signal/issues/387) | universe.py gates on daily_prices coverage, which is NOT feature-row availability |
| `fps-x0f` | [#388](https://github.com/edwinsteele/fuel-price-signal/issues/388) | Is late descent a real target, and does the TGP gap help there? Reconcile the phase axis first |

## Not migrated (137 closed)

Closed bd issues were deliberately not recreated on GitHub. Their full content — descriptions, comments, dependency edges — is archived at `docs/bd-archive/` (`issues.json`, `comments.json`, `edges.json`).

**Citations survive by lookup, not by rewriting.** 1,439 references to 127 closed bd IDs appear across 119 tracked files, 60% of them in `experiments/**` lab-book entries. Those are historical records — a README that cited `fps-zci` in August is a record of what was known then, and editing it would falsify the record. Look an id up in the archive instead:

```bash
jq -r '.[] | select(.id=="fps-zci") | .title, .description' \
  docs/bd-archive/issues.json
```

## Dependency edges

Nine edges were recreated natively (`blockedBy` / `parent`). Eight more pointed at closed issues and were not: a closed `blocks` target is a *discharged gate*, not a live constraint, and `relates-to` has no GitHub equivalent. Each appears as a provenance line in its issue body instead.

