---
name: Polish
about: Small contained features, test additions, minor refactors — requires owner review before merge
title: ""
labels: polish
assignees: ""
---

## What
<!-- One sentence: what needs to change -->

## Where (files likely affected)
<!-- List the files or modules. Be specific — the worker acts on this cold. -->
-

## Test plan
<!-- What test should the worker add or update? -->
- [ ]

## Behaviour contract

<!--
Delete this section if behaviour doesn't vary by state — a one-row table is
ceremony, and a section that's usually ceremony trains everyone to skim it.

Keep it if there's more than one state, and settle every cell before filing.
An unresolved "decide whether" / "probably" / "whatever the design lands on"
that reaches a worker gets answered in the place with the least information and
no authority to answer it, and then arrives at review looking like a defect.
If a cell isn't decided yet, this is `design`, not `polish`.

    | State                     | Required result | Required side effects  |
    | before series             | unscored        | counted out-of-range   |
    | within series, no price   | <decided>       | ...                    |
    | after series              | unscored        | counted out-of-range   |
-->

| State | Required result | Required side effects |
| ----- | --------------- | --------------------- |
|       |                 |                       |

## Path that must work

<!--
Delete this section only if the change is genuinely single-module.

"Where (files likely affected)" doesn't tell a worker where the seams are.
Name the path from the real entry point to the final consumer:

    CLI main → run_candidate → realised.deltas → results.json → build_facts → dossier

Then add, where they apply:
- the failure-state postcondition — e.g. "on refusal, no stale results.json
  may remain"
- "do not hand-construct between X and Y" for any seam the test must cross for
  real. A test that builds its input at an intermediate layer proves the code
  works on input the production path can never hand it.
-->

## Adversarial witness

<!--
The smallest input that defeats the tempting-but-wrong implementation, as a
literal value. A fixture that can't tell a correct implementation from the
obvious wrong one certifies nothing: "a saved fixture" once passed while the
scraper took the first table on the page and the generic Sydney link — a
fixture with diesel first and both Sydney links would have made that
implementation impossible.

For stateful or parser work, also name **which dimensions share mutable state**,
since those are the ones that have to compose. Don't enumerate the full
Cartesian product; five independent dimensions is 32 mostly-uninteresting cells
and will get skimmed.
-->

## Production resource envelope

<!--
Delete this section unless the change touches data volume.

Real measurements are not enough on their own — state the population the code
must survive and the memory shape that is acceptable, as an acceptance
criterion. Without it a query gets written against the sample and materialises
hundreds of thousands of rows at once in production.

e.g. "~180k station-days across ~1,400 stations; process one station at a time,
never materialise the full result set."
-->

## Is this actually design? Checklist
<!-- If any box is checked, relabel to `design` before filing -->
- [ ] Touches cycle detection, peak finding, or smoothing parameters
- [ ] Touches signal thresholds or signal class logic
- [ ] Changes DB schema
- [ ] Involves ML model training or feature engineering
- [ ] Requires architectural judgment (new module, new aggregation layer, etc.)
- [ ] Leaves a behavioural or data-semantics decision for the worker to make
      ("decide whether", "probably", "whatever the design lands on")

## Acceptance criteria
- [ ]

## Notes for the worker
<!--
Prior art in the codebase, related issues, naming conventions, anything a cold
agent would need. Keep provenance — "why I noticed this" — to a sentence or
two; those tokens are better spent on the sections above.
-->
