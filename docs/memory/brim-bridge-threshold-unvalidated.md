---
name: brim-bridge-threshold-unvalidated
description: signal.py's FALLING_CENTS_PER_DAY = -0.5 is reasoned, not measured, and lands on the median of the drift distribution — the least stable place to cut.
metadata:
  type: project
---

`fuel_signal/signal.py`'s `FALLING_CENTS_PER_DAY = -0.5` decides brim vs bridge:
below it the CLI says cheaper fuel is coming so buy the minimum, at or above it
says today is as good as it gets so fill up. It is the only free parameter in the
new decision layer (#410) and **it was reasoned, not measured** — ~3.5 c/L over a
week "felt like" the point where waiting earns its inconvenience.

**Measured after the fact, it is in the worst available position.** Over 3287 days
of the Sydney average the 7-day drift distribution is:

| p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---:|---:|----:|----:|----:|----:|----:|----:|----:|
| -2.67 | -2.03 | -1.64 | -1.12 | **-0.47** | +1.24 | +2.49 | +2.96 | +3.68 |

The cut sits within 0.03 c/day of the **median** and splits days 49/51. Cutting at
the mode of a distribution is the least stable choice available: the verdict flips
on noise, and sensitivity is steep there — -0.75 bridges on 39.9% of days, -0.50 on
49.0%, -0.25 on 55.6%. A coin-flip split is also a hint the parameter is carrying
no information.

**Two ways out, in increasing order of rigour.** Derive it from the reference
distribution — p25 = -1.12 expresses "genuinely falling" (~30% of days) rather than
"a shade below average" (~50%) — see [[feedback_derive_thresholds_from_reference_distribution]]
in the private notes for why that beats an eyeballed constant. Better, score
brim-vs-bridge policies against realised CPL with the tank engine in
`backtest.py`, the bar every other lock parameter in this project had to clear;
`TankParams` already models the partial-fill behaviour the choice is about.

**Do not quietly re-tune it by eye.** That swaps one unmeasured number for another
and makes the next reader think it was validated. Related:
[[cadence-not-a-free-knob]], [[tau-selector-is-cadence-blind]] — both are cases in
this repo where a parameter that looked like a knob turned out to need a declared
measurement.
