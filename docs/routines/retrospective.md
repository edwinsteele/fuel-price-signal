# Retrospective routine — prompt source

Canonical, tracked home for the prompt used by the **retrospective session** (`fps-3jj.8`)
in the AI-sourced feature engineering pipeline (`fps-3jj`). Unlike `docs/routines/dossier.md`
and `docs/routines/worker.md`, this is **not a nightly scheduled task** — its cadence is an
event ("this batch's candidates have all reached a terminal dossier"), the same framing
`docs/routines/generator.md` uses for its own invocation. Invoke it interactively once a
batch is done, not on a timer.

## When to invoke

Run `experiments.pipeline.dossier_tables --scan` first if there's any doubt a batch is
fully dossiered. A batch is ready for its retrospective when every candidate module filed
against it (`experiments/candidates/<batch>/*.py`) has reached a terminal dossier — check
this by running Step 0 below and confirming `outcome_tally.never_run == 0`,
`outcome_tally.retryable_incomplete == 0`, and `outcome_tally.pending_dossier == 0`. If any
of those are nonzero, the batch isn't finished: some candidates never ran, are still
cycling through the launch routine's retry budget (or exhausted it and sit `blocked` in
bd — this module reads disk only and can't tell those two apart, see "Known gaps" below),
or finished but haven't been dossiered yet. Don't write up a retrospective around a hole in
the data; wait, or go find out why a candidate is stuck.

## What this session does, in order

### Step 0 — deterministic pass (no judgement, no arithmetic)

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.retrospective <batch-name>
```

This does one thing, mechanically (`experiments/pipeline/retrospective.py`, fps-3jj.8):
reads every candidate's own `facts.json` (already written by the dossier routine — nothing
here recomputes a noise-band percentile or a headline delta) plus the batch's
`noise_floor.json`, and writes `<batch_dir>/retrospective_facts.json`. Same code/Claude
split as the dossier routine: this pass forms no verdict, only facts.

Refuses to overwrite an existing `retrospective_facts.json` unless `--force` is passed —
same overwrite guard as `noise_floor.py`, since a retrospective already written may already
be referenced elsewhere.

### Step 1 — read `retrospective_facts.json` in full

Four sections, matching the parent bead's acceptance criteria:

1. **`leaderboard`** — every DOSSIERED candidate, ranked by `noise_band_percentile`
   (higher = better) when the batch has a noise floor, else by raw `delta_cpl_held`
   ascending (it's a COST — more negative is better). Candidates that never reached a
   dossier are NOT in the leaderboard; they're in `outcome_tally` instead (see below) —
   don't silently read their absence as "these did nothing," they simply have no result.

2. **Noise-band comparison, with the multiple-comparisons correction already applied.**
   `family_wise_z_threshold` (fps-awz) is a Bonferroni correction, computed over the number
   of DOSSIERED candidates in this batch, on a t-distributed critical value in
   band-standard-deviation space rather than percentile space — with N candidates graded
   against the same noise band, picking the best of N is a different, easier-to-satisfy-by-
   chance question than grading one candidate alone (order statistics) — the bead body calls
   this out explicitly ("the ranking step is exactly where a noise delta gets promoted to a
   finding"). Each leaderboard row's `clears_family_wise_threshold` uses this corrected bar
   against `noise_band_z`, not `noise_band_percentile`. **`family_wise_percentile_threshold`
   and `noise_band_percentile` are still in the payload, but are descriptive colour only** —
   with the noise floor's draw count in the tens, not thousands, the empirical-rank
   percentile only has `n_draws + 1` distinct values, which at the old 5-draw default meant
   "gate on percentile" had silently collapsed to "beat every single draw" regardless of the
   nominal threshold (fps-awz). Report the percentile as colour if it reads well, but the
   verdict is `clears_family_wise_threshold`, not "candidate X sat at the 92nd percentile."

3. **`outcome_tally`** — `total_candidates_filed` is the universe (every `.py` module
   under `experiments/candidates/<batch>/`), not just the ones that finished.
   `dossiered_by_status` breaks down terminal outcomes (`rejected` / `disqualified` /
   `aborted_candidate`) SEPARATELY — folding `aborted_*` into `rejected` would bias any
   calibration read below, since an aborted candidate never got a fair hearing at all.
   `never_run` / `retryable_incomplete` / `pending_dossier` are missing data, not
   findings — see "When to invoke" above; ideally all three are zero by the time you're
   writing this up.

4. **`confidence_calibration`** — **scoped to EVERY batch dossiered so far, not just this
   one** (`confidence_calibration.scope` states this; see the module's own docstring for
   why — `docs/routines/generator.md`: "Five candidates cannot calibrate anything on
   their own... reads calibration across batches, not within one"). If
   `insufficient_data` is true (fewer than `min_calibration_n` resolved-effect candidates
   pooled across every batch), report the raw `pairs` list without claiming a
   calibration verdict — don't compute or assert a correlation the data can't support.
   Once sufficient, `mean_confidence_effect_when_resolved_true` vs
   `..._when_resolved_false` is the first, roughest calibration signal ("did the ones
   the generator rated 0.6 beat the ones it rated 0.2?") — read it as a first signal in
   prose, not the last word.

### Step 2 — write `RETROSPECTIVE.md`

In the run directory (`<batch_dir>/RETROSPECTIVE.md`), using
`experiments/RETROSPECTIVE_TEMPLATE.md`, with the same **Facts / Judgement split** as a
candidate dossier:

- **Facts**: transcribed from `retrospective_facts.json` — no new arithmetic. Leaderboard
  table, the multiple-comparisons threshold and which candidates (if any) clear it,
  outcome tally table, confidence-calibration pairs (and means, if not
  `insufficient_data`).
- **Judgement**: did the batch's standout candidate(s) hold up against noise at the
  batch-corrected bar? Did the generator's confidence priors predict what happened, to
  the extent the data supports saying anything? **One-at-a-time additive screening never
  validates a *combination*** (bead body's own caution) — if several candidates in this
  batch each look promising alone, say so, but recommend a combined run before
  "graduate all of them," not instead of it. `not_tested` — adjacent ground this
  retrospective doesn't settle.
- **Recommendation** — not a binary verdict. What should the next generator session (or a
  human) do with this batch's result?

Commit `RETROSPECTIVE.md` + `retrospective_facts.json` together, straight to `main`
(`experiments/**` is PR-exempt) — same convention as a candidate dossier.

## Known gaps to flag, not silently work around

- **Retryable-vs-blocked ambiguity.** `retrospective.py` reads disk only (`results.json`'s
  status), so a candidate in `RETRYABLE_STATUSES` reads as `retryable_incomplete` whether
  it's still cycling through the launch routine's retry budget or has exhausted it and sits
  permanently `blocked` in bd (`fps-rtd`'s mechanism). Telling these apart needs
  `bd show <id>` on the candidate's bead. Don't assume either state without checking.
- **Small-N leaderboards are still correct, just not very informative.** Batch 0 has one
  candidate; `family_wise_percentile_threshold(1)` reduces to the plain 95th percentile, and
  `family_wise_z_threshold(1, n_draws)` reduces to the plain one-tailed critical value at the
  band's own draw count. The mechanism is exercised end-to-end from batch 0, but the real
  payoff (a leaderboard that actually differentiates candidates, and the start of a real
  confidence-calibration read) arrives with `batch1` (5 candidates) and beyond.
- **`family_wise_z_threshold` needs `n_draws >= 2`.** With fewer draws the band's std can't
  be estimated at all (no degrees of freedom for a t-critical value) — the payload's
  top-level `family_wise_z_threshold` (a scalar) is `None` and every row's
  `clears_family_wise_threshold` is `False`, the same
  condition under which `dossier_tables._noise_band()` already returns
  `candidate_z_vs_band: None`. Not reachable at the fps-awz default of ~20 draws; only
  matters if a batch's `noise_floor.json` was hand-computed with a very small `--n-draws`.

## The shim

Unlike the dossier/launch/worker routines, this one is not registered as a scheduled task
at all (see "When to invoke" above) — there is no shim to register. Invoke it directly by
following this file when a batch's outcome tally is clean.
