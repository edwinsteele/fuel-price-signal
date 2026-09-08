---
name: noise-floor-fixture-needs-n-placebo-columns
description: A multi-column noise_floor.json fixture without n_placebo_columns silently trips the arity refusal branch.
metadata:
  type: project
---

A noise_floor.json test fixture for a candidate with 2+ columns MUST set n_placebo_columns explicitly, or dossier_tables._noise_band() defaults it to 1 (the pre-fps-3jj.14 assumption: a floor with no n_placebo_columns key predates arity-awareness and WAS arity 1) — a 2-column candidate against an implicit arity-1 floor silently triggers the ARITY refusal branch (available: false, reason_code floor_arity_below_run) instead of whatever the test actually meant to exercise. Bit tests/test_dossier_tables.py's new decision_flips tests (fps-gez, 2026-08-24) until n_placebo_columns was added to every noise_floor.json fixture using columns=[...] with 2+ entries.
