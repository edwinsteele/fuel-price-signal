---
name: api-contract-golden-tests-prove-serialization-not-evaluation
description: test_api_v1's worked-example tests assert the serializer emits docs/api-contract.md's examples exactly, but they feed it hand-set inputs — the rule-signal description strings and prices in the examples are NOT checked against signal.py.
metadata:
  type: project
---

Since #433, `tests/test_api_v1.py` extracts the ```` ```json stations-response ````
and ```` ```json recommendation-response ```` fences from `docs/api-contract.md`
at test time (`tests/contract_examples.py`) and asserts
`build_*_response(...) == contract_example(...)`. An edit to an example that
the serializer can't produce now fails here.

**What it does not prove:** the inputs. `_CONTRACT_EVALUATIONS` (the four
rule-signal descriptions), `_CONTRACT_PRICES`, `_CONTRACT_PROBS` and the cycle
values are literals in the test, fed straight into a `SignalPayload`. So the
example's `rules.signals[].description` strings and prices are checked for
*shape and serialization*, not for being what `evaluate_all_signals` /
`build_signals` would actually emit. The contract claims every example value
is "derived from the server's code paths"; for those inputs that is still an
audit (done by hand in fps-app `7679c25`), not a test.

**How to apply:** if you change a description format in `signal.py`, update
`_CONTRACT_EVALUATIONS` *and* the contract example together — the golden test
will stay green if you change neither, and fail only if they disagree with
each other. A green run is not evidence the strings match `signal.py`.

Related: [[api-contract-byte-identical-copy-wording-trap]] (the file is vendored
byte-for-byte by `fuel-price-signal-app`, whose tests decode the same fences).
