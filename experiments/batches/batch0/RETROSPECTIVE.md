# batch0 — retrospective

- **Date:** 2026-08-19
- **Batch:** experiments/batches/batch0
- **SHA:** 25a86ee6a4f489bc0e8c920f84927357dbbb5aac
- **Cadence:** `50/3.571/7d/10%` — batch0 is a **7-day batch**, frozen and graded
  entirely before the 2026-08-22 re-lock to daily (`fps-oqz`). Every CPL and
  noise-band figure below is conditioned on it; none has been restated. Its
  `freeze.json` / `noise_floor.json` stamps still read 7d, so a new candidate run
  (which now defaults to 1d) is refused against this floor rather than silently
  graded across the boundary.
- **Candidates filed:** 1

## Grandfathered — batch0's numbers are not comparable to batch1+ (2026-09-05, `fps-sjk`)

**Decision: grandfather, do not recompute.** `noise_floor.json` here carries no
`null_method`: it is a pre-`fps-awz` **seed-swap** null (5 R0-vs-R0 paired-seed
draws, `fps-3jj.9`), not the **placebo-column** null every batch1+ floor uses.
`dossier_tables._noise_band()` refuses it outright, so `batch0/tgp_delta_7d`
cannot be graded against any noise floor under current tooling. **That refusal is
correct behaviour, not a defect** — nothing should be filed against it.

Four independent reasons, any one of which settles it:

1. **Recompute is not available from the shipped CLI at any price.** The obvious
   command — `noise_floor batch0 --force` — cannot run. `check_freeze_cadence`
   (`fps-oqz`) refuses it: this batch's `freeze.json` declares `50/3.571/7d/10%`
   while the CLI resolves `TankParams()` = `50/3.571/1d/10%` and exposes **no
   cadence override** (`compute_noise_floor` takes `tank=`; `main` never passes
   it). Already asserted against the committed artifact by
   `tests/test_noise_floor.py::test_batch0s_real_freeze_manifest_refuses_the_current_default_cadence`,
   and already written down in `docs/CONVENTIONS.md` § cadence. So option 1's
   price is not "the placebo draws" (batch1's 10-draw floor cost 8,368 s wall;
   `DEFAULT_N_DRAWS` is now 20) — it is **new code first**, then that compute, on
   a batch nothing builds on.
2. **The sign forecloses the only outcome worth paying for.** The candidate's
   realised `delta_cpl_held` is **+0.00585 c/L** — the *unfavourable* direction
   (higher CPL is higher cost). No floor of any width turns a positive delta into
   a graduation. A recomputed placebo band (narrower than a seed-swap band, by
   `fps-awz`'s own argument) could only leave it `inconclusive` or push it to
   `rejected`. There is no upside branch.
3. **Nothing downstream is riding on the answer.** The ledger's
   `evidence: experiments/candidates/batch0/tgp_delta_7d` entry is
   `outcome: inconclusive`, and was already `inconclusive` when `fps-sjk` was
   filed (checked against `experiments/ledger.yaml` at commit `3cbf496`,
   2026-09-02) — the bead's premise that it "currently lists that run as
   graduated" was wrong. The only `graduated` label `tgp_delta_7d` ever carried
   was June's, which its own ledger entry already records as SUPERSEDED and
   `inconclusive`. Re-grading would be cosmetic in the strict sense: it cannot
   move the recorded verdict anywhere the record does not already sit.
4. **The reading is retired on a second, independent axis anyway.** batch0 is a
   7-day batch and the canonical cadence has been 1d since 2026-08-22
   (`fps-oqz`). A correctly-shaped ruler for a measurement that `fps-cde` has
   already ruled out of the record does not make that measurement count.

**Read every noise-band figure in this batch accordingly.** The 40th percentile,
`z = +0.14`, and the band (mean −0.0135 c/L, std 0.142 c/L) are a historical
record of what the **seed-swap** null said about a **7d** run. They are not a
current-method grading and must not be compared against a batch1+ percentile.

**If `tgp_delta_7d` is ever wanted on equal footing**, `fps-cde` is the only
route: measure it at 1d as a batch2 candidate. That produces a current-method
floor and a current-cadence run in one pass. batch0 stays untouched — it cannot
be re-frozen (`batch_freeze` allows one freeze per batch) regardless.

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
| tgp_delta_7d | graded | +0.00585 | 40.0 | no |

**Multiple-comparisons note:** `family_wise_percentile_threshold` at n=1 dossiered
candidate is 95.0 — the Bonferroni correction reduces to the plain single-draw bar
when there's only one candidate to grade, exactly as designed
(`family_wise_percentile_threshold(1) == 95.0`). `tgp_delta_7d` at the 40th percentile
doesn't come close, whether graded individually or at the batch-corrected bar — this
is not a case where the correction changes the verdict.

## Outcome tally

| status | count |
|---|---|
| graded | 1 |

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
