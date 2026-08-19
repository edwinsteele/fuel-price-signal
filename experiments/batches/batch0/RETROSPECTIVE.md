# batch0 — retrospective

- **Date:** 2026-08-19
- **Batch:** experiments/batches/batch0
- **SHA:** 25a86ee6a4f489bc0e8c920f84927357dbbb5aac
- **Candidates filed:** 1

## What this is

The payoff artifact for aim (b) (gaining experience with an AI-sourced pipeline, not
turnaround time) — not a verdict on the one candidate filed here (that lives in
`experiments/candidates/batch0/tgp_delta_7d/README.md`). Everything below traces to
`retrospective_facts.json`, written by `experiments/pipeline/retrospective.py`
(bd `fps-3jj.8`).

batch0 is the smallest possible batch: one candidate (`tgp_delta_7d`, deliberately —
it's the known-graduate sanity check for the pipeline itself, per `docs/routines/
generator.md`'s batch-sizing rule). The mechanism below is exercised end to end and
correct, but a one-candidate batch cannot exercise what a leaderboard or a
multiple-comparisons correction is actually FOR — that's batch 2 (5 candidates) and
beyond.

## Leaderboard

Ranked by noise-band percentile (batch0 has a noise floor — `fps-3jj.9`, `n_draws=5`).

| candidate | status | delta_cpl_held | noise-band percentile | clears batch threshold? |
|---|---|---|---|---|
| tgp_delta_7d | rejected | +0.00585 | 40.0 | no |

**Multiple-comparisons note:** `family_wise_percentile_threshold` at n=1 dossiered
candidate is 95.0 — the Bonferroni correction reduces to the plain single-draw bar
when there's only one candidate to grade, exactly as designed
(`family_wise_percentile_threshold(1) == 95.0`). `tgp_delta_7d` at the 40th percentile
doesn't come close, whether graded individually or at the batch-corrected bar — this
is not a case where the correction changes the verdict.

## Outcome tally

| status | count |
|---|---|
| rejected | 1 |

`never_run` / `retryable_incomplete` / `pending_dossier` are all 0 — the tally is
clean, batch0 is genuinely finished.

## Confidence calibration

Cumulative across every batch dossiered so far — currently just this one candidate
(`n_dossiered_with_resolved_effect` = 1, `n_usable_for_calibration` = 1, both far below
`min_calibration_n` = 10). `insufficient_data: true`. No means computed, correctly —
one data point calibrates nothing, and the field says so rather than reporting a
number that would look like a finding.

The one pair on record: `tgp_delta_7d` was filed with `CONFIDENCE_EFFECT = 0.5` (no
real prior stated either way — appropriate, since this candidate's role was pipeline
calibration, not a genuine AI-sourced hypothesis) and `effect_resolved: false`
(the realised delta didn't move the arbiter). Nothing to calibrate against yet; this
pair just starts the running record batch 2 will build on.

## Judgement

- **Did the standout candidate hold up against noise at the batch-corrected bar?**
  N/A in the interesting sense — `tgp_delta_7d` was never expected to "win" a
  leaderboard; it was the pipeline's own calibration check (see its README's own
  framing: "batch0 exists to calibrate the pipeline, not to decide anything about
  `tgp_delta_7d`"). It did its job: the pipeline reproduced an independent reference
  exactly, and separately, the noise-band grading confirms +0.00585 c/L is
  indistinguishable from pure fit noise (40th percentile, z=+0.14).
- **Did CONFIDENCE_EFFECT predict what happened?** Not a meaningful question with one
  data point and a deliberately uninformative 0.5 prior on a calibration-only
  candidate. The real read starts at batch 2.
- **One-at-a-time additive screening never validates a combination** — moot here,
  one candidate.
- **not_tested:** everything the leaderboard/multiple-comparisons machinery is
  actually for. This retrospective proves the code path is correct (see the module's
  own test suite plus this real run), not that it's useful yet — that's an honest
  gap, not a hedge.

## Recommendation

**Nothing to decide about `tgp_delta_7d` here** — see its own dossier and
`docs/STATUS.md` for the standing `#271` hold, unchanged by this retrospective.

**For the pipeline itself:** batch0 is done, cleanly, with every artifact this
routine is supposed to produce (leaderboard, noise-band grading, outcome tally,
confidence-calibration record) present and correct. The generator is unblocked to
file batch2's 5 candidates once `#271`'s ordering constraint is satisfied (already
the case — batch0 reached a verdict, which was the actual gate; see
`project_feature_pipeline` memory). Batch2 is where this retrospective mechanism
does real work: a real leaderboard, a Bonferroni correction that can actually bite,
and the second data point for confidence calibration.
