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
   Each row carries the `tank_params` its own dossier ran at, and `noise_floor.tank_params`
   stamps the band (`fps-aam`, extending `fps-15c` to this artifact) — a `delta_cpl_held`
   is only readable in its own cadence's units. Transcribe the stamp with the number. A
   batch declares ONE cadence for every candidate run against it (`freeze.json`), so
   `build_leaderboard` REFUSES outright if the rows disagree rather than ranking across
   them — if you hit that error, the fix is upstream (re-run the odd candidates or
   re-freeze the batch), not in the write-up.

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
   `dossiered_by_status` breaks down terminal outcomes (`graded` / `disqualified` /
   `aborted_candidate`) SEPARATELY — folding `aborted_*` into `graded` would bias any
   calibration read below, since an aborted candidate never got a fair hearing at all.
   `never_run` / `retryable_incomplete` / `pending_dossier` are missing data, not
   findings — see "When to invoke" above; ideally all three are zero by the time you're
   writing this up.

   `aborted_candidate` itself further splits on `abort_reason` when one was recorded
   (fps-3jj.13): a key like `"aborted_candidate:leak_by_declaration"` tallies candidates
   whose `INPUTS`/`COLUMNS` named a label column (`future_min_cents`/`label`) — the
   generator proposing an oracle feature, aim (b)'s "how often it produces leaky ...
   candidates" — separately from plain `"aborted_candidate"`, which is an ordinary bug
   in the candidate's own arithmetic. Report both counts and the leak rate
   (`leak-tagged / total_candidates_filed`) explicitly rather than quoting one combined
   `aborted_candidate` number.

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
  table (including each row's cadence stamp), the multiple-comparisons threshold and which
  candidates (if any) clear it, outcome tally table, confidence-calibration pairs (and
  means, if not `insufficient_data`). No CPL-shaped number reaches the prose without the
  `tank_params` it was produced at — same rule one artifact upstream (`fps-15c`).
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
(`experiments/**` is PR-exempt) — same convention as a candidate dossier — then
**`git push`**. Finish by confirming `git status --short` and
`git log --oneline origin/main..main` are both empty (docs/CONVENTIONS.md § the
exemption): the batch retrospective is the artifact other work cites, so an unpushed
one is invisible to everyone but this machine.

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
- **Batch 0's committed `retrospective_facts.json` predates two reworks and is NOT
  regenerated.** It carries no `tank_params` anywhere (`fps-aam`) and its `noise_floor`
  block reads `available: true` against a floor today's `dossier_tables._noise_band` would
  refuse (no `null_method`, i.e. the pre-`fps-awz` seed-swap null). It is a dated record of
  what was computed on 2026-08-19 — `computed_at`/`git_sha` say so — and `RETROSPECTIVE.md`'s
  prose was written from it, so re-running with `--force` would silently rewrite a published
  finding rather than just add a stamp. Read it as history; don't diff it against this
  schema.
- **batch1 ranks against its COMMITTED ruler, and that ruler's `effective_n_draws: 10.0` is
  wrong (`fps-3jj.24`).** Decided 2026-08-27; do not re-open it in the write-up. batch1's
  `noise_floor.json` carries the `fps-3jj.20` seed collision — draw 10 reuses draw 1's seeds
  97/101/103, landing at placebo correlations 0.965 / 0.778 / 0.765 against it. **`fps-3jj.25`'s
  reuse correction does not catch this and cannot**: `placebo.effective_n_draws` prices SHARED
  SOURCE COLUMNS, and the two draws share none (30 picks, 30 distinct columns), so
  `_resolve_effective_n_draws` returns the nominal 10.0 exactly. Against
  `docs/CONVENTIONS.md`'s own measured table the machinery charges the cheap channel (same
  column, different seed — median |ρ| 0.038) at `icc = placebo.TEXTURE_ICC_BOUND` (0.391 when
  this was written; 0.274 since `fps-3jj.23` measured it by column) and the expensive one
  (different column, same seed — max |ρ| 0.971) at zero. Closed by construction for every FUTURE floor
  (`placebo.block_seed` never restarts), so this is a defect of two committed artifacts, not of
  the code — do not "fix" `effective_n_draws` to chase it.

  Priced by hand at the collided pair's observed 0.836 correlation, the bank is worth **8.57**
  effective draws, and the bar is that much too EASY:

  | gate | committed | collision-corrected | shift |
  |---|---|---|---|
  | single candidate (`n_candidates=1`) | −0.1670 | −0.1727 | −0.0057 c/L |
  | batch gate (`n_candidates=5`) | −0.2706 | −0.2850 | −0.0144 c/L |

  **Rank on the committed numbers anyway** — they are what all five dossiers were graded
  against, and a retrospective citing a bar no dossier used would put the batch verdict and the
  per-candidate write-ups in different units. A recompute was considered and REFUSED: a fresh
  10-draw band estimates its std to ±23.6% (`1/sqrt(2(n-1))`), which is ±0.0697 c/L on the
  `n_candidates=5` bar — **4.8× the 0.0144 c/L bias it would remove** — so the difference it
  reported would be dominated by resampling luck, and `docs/CONVENTIONS.md` already says a
  post-2026-08-26 recompute is not comparable draw-for-draw anyway. The closed form above is a
  strictly better instrument than the ~2.3h run.

  **What the write-up must do instead**, because the correction is monotone-HARDENING and so can
  only ever demote a candidate, never promote one: no candidate is wrongly failed by the
  committed ruler, only possibly wrongly PASSED. So check each cleared row against the flip
  window and say so explicitly:

  | gate | flip window on \|`noise_band_z`\| | on `delta_cpl_held` |
  |---|---|---|
  | `n_candidates=1` | [1.9226, 1.9797) | (−0.1727, −0.1670] |
  | `n_candidates=5` | [2.9591, 3.1030) | (−0.2850, −0.2706] |

  A row inside its window clears ONLY by the collision's optimism and must be reported as
  not-established rather than as clearing. A row outside is unaffected, and the write-up says
  one line — "the collision did not change any verdict" — and moves on. As of 2026-08-27 the
  four dossiered candidates sit at |z| = 0.523 / 0.926 / 0.987 / **1.805**; the best
  (`network_move_breadth`) misses the single-candidate bar under both rulers, so no verdict
  moves. `tgp_cycle_displacement`, the one candidate still unrun, gets the same check when it
  lands. Note that every one of those `facts.json` files emits `effective_n_draws: 10.0` — that
  field is the thing this entry says is wrong, so do not quote it as evidence the band is
  clean.

- **The retired `noise_floor_k1.json` carries the same defect, and one documented number rests
  on it (`fps-3jj.24`).** 20 draws at arity 1, two collided pairs (draws 1/19 on seed 97, draws
  2/20 on seed 101), and again `n_eff` reads the nominal 20.0. It is the k=1 baseline behind
  `docs/CONVENTIONS.md`'s "two thirds of batch1's k=1 → k=3 bar move is draw thinness, not
  arity" attribution. Both floors are biased the same direction, so it largely self-cancels:
  correcting both moves the k1→k3 gap from −0.0536 to −0.0637 c/L at `n_candidates=5`. **The
  two-thirds claim survives** — cite it without a caveat; this note exists so nobody re-derives
  the concern from scratch.

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
