---
name: pandas-assert-equal-default-tolerance-hides-leaks
description: assert_frame_equal's default rtol silently passes real leaks at large magnitudes; pass check_exact=True in leak tests.
metadata:
  type: project
---

pandas assert_frame_equal/assert_series_equal default to tolerance-based comparison (rtol=1e-5), which silently PASSES a real leak in a differential PIT/leak-detection test when values are large-magnitude — e.g. a whole-series mean of YYYYMMDD-scale price_date differed by ~1.5 out of ~20260812, well inside the default relative tolerance despite being a genuine, detectable leak. Any equality check whose JOB is to catch a difference (leak tests, point-in-time recomputation checks, idempotency checks) must pass check_exact=True explicitly — the default is for numeric-tolerance comparisons like float arithmetic results, not equality-as-a-safety-property. Found and fixed in experiments/lib/pit_test.py's differential_pit_test (fps-3jj.2, PR #298) via the runner's own new tests, not a pre-existing test gap.
