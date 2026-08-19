# <batch name> — retrospective

- **Date:** YYYY-MM-DD
- **Batch:** experiments/batches/<batch>
- **SHA:** <git_sha from retrospective_facts.json>
- **Candidates filed:** <outcome_tally.total_candidates_filed>

## What this is

The payoff artifact for aim (b) (gaining experience with an AI-sourced pipeline, not
turnaround time) — not a verdict on any individual candidate (those live in each
candidate's own `README.md`). Everything below traces to `retrospective_facts.json`,
written by `experiments/pipeline/retrospective.py`; this file is prose plus judgement
on top of it, same Facts/Judgement split as a candidate dossier.

## Leaderboard

Ranked by noise-band percentile (higher = better) where the batch has a noise floor,
else by raw `delta_cpl_held` (a COST — more negative is better).

| candidate | status | delta_cpl_held | noise-band percentile | clears batch threshold? |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

**Multiple-comparisons note:** with N candidates graded against the same noise band,
picking the best of N is a different question from grading one candidate alone — the
`family_wise_percentile_threshold` (Bonferroni-corrected) is the bar a candidate needs
to clear to be read as surprising at the BATCH level, not the raw 95th percentile.
State the threshold used and which candidates (if any) clear it.

## Outcome tally

| status | count |
|---|---|
| ... | ... |

State plainly whether any candidates are `never_run`, `retryable_incomplete`, or
`pending_dossier` — all three are missing data, not negative results, and the
retrospective is incomplete while any remain. Don't write up final conclusions until
the tally is clean (all three at zero).

## Confidence calibration

Cumulative across every batch so far (see `retrospective_facts.json`'s
`confidence_calibration.scope`), not just this one. If `insufficient_data` is true,
say so plainly and report the raw pairs without claiming a calibration read — five (or
even a few dozen) candidates is not enough on its own (see `docs/routines/generator.md`).

## Judgement

- Did the batch's headline candidate(s) hold up against noise?
- Did `CONFIDENCE_EFFECT`/`CONFIDENCE_ZONE` predict what happened, to the extent the
  calibration data supports saying anything yet?
- One-at-a-time additive screening never validates a *combination* — if several
  candidates in this batch each look promising alone, say so, but don't recommend
  "graduate all of them" without a combined run.
- `not_tested` — adjacent ground this retrospective doesn't settle.

## Recommendation

Not a binary verdict — what should the next generator session or human do with this
batch's result?
