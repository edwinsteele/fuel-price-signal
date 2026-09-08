---
name: default-flip-breaks-contrast-tests
description: Flipping a default silently disarms every test that used the new value as its contrast arm.
metadata:
  type: project
---

When you flip a DEFAULT, grep for tests that used the NEW value as their contrast — they silently stop asserting anything.

Found flipping TankParams.evaluation_interval_days 7 -> 1 (fps-oqz, PR #323). Six fps-15c tests proved "default and non-default are mechanically distinguishable" by comparing TankParams() against TankParams(evaluation_interval_days=1). After the flip both sides construct the SAME tank. Exactly one of the six failed loudly (an inequality assert); the other five kept passing while comparing a value to itself — a green suite that had quietly lost its guard.

The fix is to move the contrast arm to the OLD value (7d here), not to delete the test: coverage of an explicitly-passed non-default cadence is still worth having, and a test that just follows the default tests nothing about the default.

Same shape applies to any default: a sentinel, a feature flag, a model name, a path. The tell is a test whose two arms are "bare constructor" vs "constructor with X" where X is the value you are moving TO.

Related: [[baseline-fingerprint-before-comparing-runs]] (same family — two things that look comparable but aren't).
