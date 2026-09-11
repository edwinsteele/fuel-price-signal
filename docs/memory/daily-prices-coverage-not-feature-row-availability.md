---
name: daily-prices-coverage-not-feature-row-availability
description: High daily_prices coverage in a window does not imply the ML feature pipeline emits any row there — the two disagree badly on exactly the folds that matter.
metadata:
  type: project
---

`experiments/lib/universe.py`'s `min_coverage` gate counts distinct `daily_prices` dates —
the REPLAY axis (`aggregate_backtest` / `run_backtest` walk this table). It is NOT a proxy
for whether `fuel_signal.labels.assemble_training_rows` / `fuel_signal.features
.assemble_feature_rows` actually produce a row for that station-date, and the two axes
disagree badly on exactly the folds a coverage gate exists to protect. Cause: `fill.py`
forward-fills a `daily_prices` gap of up to `MAX_GAP_FILL_DAYS = 28` days, so a filled
gap that short is invisible to anything reading `daily_prices` — including the labels
calendar-gap mask. What the mask DOES see is a gap that remains MISSING from
`daily_prices` after filling (longer than `MAX_GAP_FILL_DAYS`, or past the trail-fill
horizon); around one of those it strips `lookback_days` (90) days before, and
`horizon_days` (7) days after — far wider than the hole itself. Measured on batch1's
frozen artifacts (2026-09-05): fold 1, both Blue Mountains stations sit at 0.96
`daily_prices` coverage (a station-window that DOES still carry an unfilled gap) with
ZERO feature rows; station 414 is at 1.00 coverage in fold 8 with only 8 feature rows
(#387). A coverage gate's 1-in-20 tolerance at the 0.90 default can, by itself, blank
out a whole 90+7-day stretch of feature rows around the missing day.

Fixed in #387/PR #413 by adding `describe_universe`'s `worst_window_label_fraction` — the
`worst_window_coverage` analogue for this axis, using `assemble_training_rows` (not the
full `assemble_feature_rows` network-feature pass, to stay a cheap leaf-module computation
— it is an upper bound on the true feature-row count, never an exact match). REPORTED
only, never gated, same policy as `observed_fraction_*`. If you are sizing a broad universe
or reading a WFCV/homogeneity result, check `worst_window_label_fraction` alongside
`worst_window_coverage` — a station can pass the coverage gate cleanly and still contribute
nothing to the thing being measured.
