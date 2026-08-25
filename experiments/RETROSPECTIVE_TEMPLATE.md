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

Ranked by noise-band z (`noise_band_z` — lower/more-negative is better, `delta_cpl_held`
is a COST) where the batch has a noise floor with a usable z, else by raw `delta_cpl_held`
ascending.

| candidate | status | cadence (`tank_params`) | delta_cpl_held | noise-band z | noise-band percentile (colour) | clears batch threshold? |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

**Cadence:** every `delta_cpl_held` above is in the units of the cadence its own run used
(`fps-fii` measured realised CPL at 189.67/187.85/187.82 c/L across 7/2/1-day evaluation
intervals on an otherwise identical run — larger than the deltas this table ranks). Rows
carry `tank_params` from their own dossier and `noise_floor.tank_params` stamps the band
(`fps-aam`); transcribe them, and if they are not all the same string, say so and do not
rank across the difference — that is a mixed-cadence leaderboard, not a result.

**Multiple-comparisons note:** with N candidates graded against the same noise band,
picking the best of N is a different question from grading one candidate alone — the
`family_wise_z_threshold` (Bonferroni-corrected, t-distributed, in band-standard-deviation
space, `fps-awz`) is the bar a candidate's `noise_band_z` needs to clear to be read as
surprising at the BATCH level. `noise_band_percentile`/`family_wise_percentile_threshold`
are still reported per candidate as descriptive colour — read well, but not the gate; with
~20 draws the percentile only has ~21 distinct values, too coarse to drive a decision.
State the z-threshold used and which candidates (if any) clear it.

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
