# Dossier routine — prompt source

Canonical, tracked home for the prompt used by the **dossier session** (`fps-3jj.6`) in the
AI-sourced feature engineering pipeline (`fps-3jj`). Like `docs/routines/worker.md` and unlike
`docs/routines/generator.md`, this **is** a scheduled task: nightly, same night as the launch
routine (`fps-3jj.5`), Claude active only briefly. The scheduler's stored prompt is a three-line
shim — see "The shim" below.

## What this session does, in order

### Step 0 — deterministic pass (no judgement, no arithmetic)

Run the whole queue through the code-only step first:

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables --scan experiments/candidates
```

This does one thing, mechanically (`experiments/pipeline/dossier_tables.py`, fps-3jj.6): **facts.json
+ PNGs for every completed, undossiered run.** For every directory with a `results.json` and no
`README.md` yet, it writes `facts.json` (the only source of numbers you are allowed to use below)
and the applicable plots (always: `per_fold_delta_bars.png`, `seed_mean_vs_median.png`,
`realised_cpl_by_fold.png`, `tau_sweep.png`, `candidate_over_time.png`; conditional:
`axis_breakdown.png`, `cycle_phase_breakdown.png`, `external_series_overlay.png`). One bad/partial
run doesn't stop the rest of the scan — it's logged and skipped; check the command's own output
for any `failed, skipping` lines and flag them rather than silently ignoring.

**Stale-claim recovery is NOT this routine's job.** It runs entirely inside the launch routine
(`fps-3jj.5`, merged) as part of every nightly `launch` invocation — `recover_stale_claims()`
checks `bd list --status in_progress --label experiment` for issues whose `run.log` ends in a
traceback and releases them. An earlier version of `dossier_tables.py` had its own,
weaker, parallel stale-claim mechanism keyed on a `claim.json` file nothing ever wrote — removed
once launch.py's real implementation was confirmed to already cover this correctly. Don't
reintroduce it here.

**CONFIRMED BROKEN — run-directory convention (fps-icv, not yet fixed).** This routine's queue
model (`find_pending_runs`: a directory with `results.json` and no `README.md`) assumes one
subdirectory per candidate. That does **not** hold against the merged launch.py/runner.py: both
default a candidate's `out_dir` to `candidate_path.parent`, which is the whole **batch** directory
shared by every candidate module filed against it. Consequence: candidate 2+ in any batch
overwrites candidate 1's `results.json`/`rowpreds.parquet`/`fills.parquet` in place, and once this
routine has written a `README.md` into that shared directory for candidate 1, the directory is
permanently excluded from the queue — candidate 2+ are **silently never dossiered**. Batch 1 (a
single candidate) does not exercise this; batch 2 (5 candidates) will. Check `bd show fps-icv`
before trusting `--scan`'s output on any batch with more than one candidate, and don't paper over
missing dossiers by writing them by hand — that's this bug resurfacing, not a one-off.

If `--scan` reports nothing to process, **exit quietly** — this mirrors the launch routine's "if
the run is still going" case, just one step later in the pipeline.

### Step 1 — for each run with a fresh facts.json and no README.md yet

Everything in this step is judgement, and everything you write must trace back to a `facts.json`
field or be visibly marked as your own reasoning. **Do not compute or restate a number that isn't
already in facts.json** — if a number you want isn't there, that's a gap in `dossier_tables.py`
worth a follow-up bead (`bd create`), not something to compute in-session.

1. **Read `facts.json` in full**, including `plots` (the list of PNG filenames actually written for
   this run — reference only those, don't assume one exists).

2. **Grade the run against its own `PREDICTED_SIGNATURE`** (`facts["grading"]["predicted_signature"]`).
   This is the one place this session forms a verdict, not a transcription — decided KISS
   (fps-3jj.6): no structured `PREDICTIONS` companion, just your read of whether `headline` +
   `breakdowns` match what the candidate predicted. Write your verdict directly back into
   `facts.json`'s `grading` block (`verdict`: one of `"matched"` / `"partially_matched"` /
   `"contradicted"` / `"inconclusive"`; `explanation`: one or two sentences; `pending`: `false`).
   This is the only field in `facts.json` this session ever edits — everything else stays exactly
   as `dossier_tables.py` wrote it, so the file stays traceable to its deterministic source plus
   this one recorded judgement.

3. **Write `README.md`** in the run directory, in the house style (`experiments/TEMPLATE.md`), with
   an explicit **Facts / Judgement split**:

   - **Facts** (traceable to `facts.json`, verbatim or lightly tabulated — no new numbers):
     provenance (candidate, batch, snapshot date, git SHA, seeds, bead), the headline realised-CPL
     delta and its `effect_resolved` verdict, the WFCV log-loss delta labelled as descriptive
     colour, the per-fold / per-regime / per-axis breakdown tables (mark suppressed cells plainly —
     "n=18, below the min-cell-n guard, not a finding"; if `per_axis` is present, also carry
     `breakdowns["per_axis_coverage_note"]` verbatim — the axis lookup is built from
     `rowpreds.parquet`'s WFCV-validation-window rows, not the full frame, so it can under-cover
     relative to `headline.zone` and the note says by how much), the `seed_flags` list (if empty,
     say so — "no cells exceeded 5× cohort median seed_std" is itself a fact worth stating),
     validation (NaN rate, PIT test result, INPUTS check result), and the noise-floor delta
     (`facts["noise_band"]`) — if `available: false`, say plainly "noise-floor band: not available
     yet (fps-3jj.9)" rather than omitting the line; if available, report
     `candidate_percentile_better_than_noise` as-is (already oriented so higher = better —
     don't re-derive or re-sign it).
   - **Judgement** (your reasoning, visibly separated — a `## Judgement` heading is enough):
     - The `PREDICTED_SIGNATURE` grading verdict + explanation you just wrote.
     - **`not_tested`** — adjacent ground this run does *not* rule out. This is the highest-value
       sentence in the whole dossier (see the ledger's own docstring) — a rejected candidate whose
       series still has room ("the raw gap failed, but its velocity might not") is exactly what
       lets the next generator session revisit it legitimately. Actually think about this; don't
       write "none" as a default.
     - An overall recommendation — **not a binary verdict**. Something like "worth a second look
       once the batch has more shock-fold coverage" or "dead end on this series, but see
       not_tested" is the right register. The dossier exists to support a human (or a follow-up
       session) interrogating it further, not to close the question.
   - Embed the plots that exist for this run (`![](per_fold_delta_bars.png)` etc. — only ones
     listed in `facts["plots"]`).

4. **Prepend one entry to `experiments/ledger.yaml`** (newest-first, matching its existing
   convention) using the same fields as every prior entry:
   - `candidate`: `facts["candidate"]["name"]`
   - `inputs`: `facts["candidate"]["inputs"]`
   - `claim`: `facts["candidate"]["hypothesis"]`
   - `falsified`: empty string only if the grading verdict is `"matched"` **and**
     `facts["headline"]["realised"]["effect_resolved"]` is true (i.e. the claim held up and the
     arbiter moved the right way) — otherwise, one or two sentences on what the run actually
     showed, in the same style as existing entries (see `tgp_delta_7d` vs
     `station_minus_tgp_cents` in the file for the contrast).
   - `not_tested`: the same judgement you wrote into the README — reuse it verbatim, don't redo it.
   - `outcome`: the pipeline's own status code — `rejected` / `disqualified` /
     `aborted_candidate` / `aborted_environment` (`facts["provenance"]["status"]`). **Never write
     `graduated` here.** Per the parent design's decision boundary, this pipeline never touches
     `fuel_signal/features.py`; graduation is a separate, human-initiated PR, and a human updates
     this same ledger entry's `outcome` to `graduated` by hand when that PR lands (see how the
     `tgp_delta_7d` entry already reads `graduated` even though every pipeline run of it would have
     closed `rejected`).
   - `evidence`: the run directory's path, relative to the repo root.

5. **Prepend one row to `experiments/INDEX.md`'s table** in the same pass (its own convention: the
   two files must never drift, because neither is derived from the other) — Date / Name /
   Hypothesis / Result / Status, `Status` = `open` while the pipeline still has follow-ups queued
   for this candidate's series, `done` otherwise, `abandoned` for `disqualified` /
   `aborted_candidate` (never `graduated` — same human-edit-later rule as the ledger).

6. **Commit `README.md` + the run's updated `facts.json` (`grading` filled in) + the PNGs +
   `experiments/ledger.yaml` + `experiments/INDEX.md` together, straight to `main`** —
   `experiments/**` is PR-exempt (`CLAUDE.md`), and `ledger.yaml`/`INDEX.md` are prose/data files
   under the same exemption since they only ever get written by this routine or hand-backfills of
   experiment content. One commit per run if several are queued in one session (keeps `git blame`
   readable per candidate).

### Step 2 — summary

Print one line per run processed (candidate name, outcome, one-clause verdict) and one line per
run the deterministic scan logged as `failed, skipping`. No PR, no further action — the next thing
that touches this candidate is either a human reading the README, or the retrospective bead
(`fps-3jj.8`) once the whole batch is in.

## Known gaps to flag, not silently work around

- **Run-directory model (fps-icv, CONFIRMED BROKEN, not a hypothetical).** See Step 0 above — every
  candidate after the first in a multi-candidate batch is silently never dossiered until this is
  fixed in `launch.py`/`runner.py`. Do not hand-write a dossier to compensate; surface it instead.
- **`noise_floor.json` contract.** `facts["noise_band"]["available"]` will read `false` for every
  run until `fps-3jj.9` (P3, not yet built) writes `<batch_dir>/noise_floor.json =
  {"deltas_cpl_held": [...]}` at batch-setup time. Report this plainly in the README (see Step 1
  above) rather than inventing a noise estimate.

## The shim

This is the exact text that should be the Routine's stored prompt (three lines, same pattern as
`docs/routines/worker.md`'s shim — no embedded steps, no reminders about what this file says):

```
You are the fuel-price-signal dossier routine.
Your working directory already has a fresh checkout of `edwinsteele/fuel-price-signal`.
Follow docs/routines/dossier.md exactly.
```

**Registering the actual scheduled task is an owner action**, same as every other routine in this
project — not done as part of `fps-3jj.6`. Register it as a separate Claude Code Routine from both
the chore/polish worker and the launch routine (three independent schedules against the same repo),
timed to run after the launch routine's candidate for the night has had time to finish (hours, not
minutes — see the parent design's "Run (hours, detached)" timing).
