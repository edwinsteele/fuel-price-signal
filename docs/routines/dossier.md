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

**Retryable aborts are skipped entirely, before anything is written (fps-g31).** `find_pending_runs`
excludes any run whose `results.json` carries a `RETRYABLE_STATUSES` status (`aborted_pipeline` /
`aborted_environment`). Those runs never got a fair hearing, their bd claim goes back on the queue,
and the re-run **reuses the same directory** — so a `README.md` written for the aborted attempt
would permanently exclude the successful re-run from this queue (`find_pending_runs` requires *no*
`README.md`). No `facts.json`, no `README.md`, no ledger row, no INDEX row: wait for the re-run.
`aborted_candidate` and `disqualified` are NOT retryable — those are real verdicts and get written
up normally.

**Stale-claim recovery is NOT this routine's job.** It runs entirely inside the launch routine
(`fps-3jj.5`, merged) as part of every nightly `launch` invocation — `recover_stale_claims()`
checks `bd list --status in_progress --label experiment` for issues whose `run.log` ends in a
traceback and releases them. An earlier version of `dossier_tables.py` had its own,
weaker, parallel stale-claim mechanism keyed on a `claim.json` file nothing ever wrote — removed
once launch.py's real implementation was confirmed to already cover this correctly. Don't
reintroduce it here.

**FIXED — run-directory convention (fps-icv).** This routine's queue model (`find_pending_runs`: a
directory with `results.json` and no `README.md`) assumes one subdirectory per candidate. That now
holds: launch.py's `launch_detached` and runner.py's `run_candidate` both default a candidate's
`out_dir` to `runner.default_out_dir(candidate_path)` (candidate_path with its `.py` suffix
stripped), not `candidate_path.parent` (the whole **batch** directory shared by every candidate
module filed against it). Before this fix, candidate 2+ in any batch would overwrite candidate 1's
`results.json`/`rowpreds.parquet`/`fills.parquet` in place, and once this routine had written a
`README.md` into that shared directory for candidate 1, the directory would be permanently
excluded from the queue — candidate 2+ silently never dossiered. `find_pending_runs`'s
`root.rglob(RESULTS_FILENAME)` already recurses into per-candidate subdirectories, so `batch1`
(5 candidates) is expected to dossier correctly.

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
     BOTH `breakdowns["per_axis_identification_note"]` and `breakdowns["per_axis_coverage_note"]`
     verbatim — the first says the per-axis **CPL** cells are a path-coupled cost allocated to a
     sub-period and so are **not identified** (report them as colour, never as a finding; the
     `per_fold` and `per_regime` cells beside them are fine, because a fold is an independent
     simulation rather than a slice through one — see `experiments/2026-08-21_path_coupling_audit/`
     and `docs/CONVENTIONS.md` § *Bucketed results*), the second says the axis lookup is built from
     `rowpreds.parquet`'s WFCV-validation-window rows, not the full frame, so it can under-cover
     relative to `headline.zone` and by how much. Never let a per-axis delta reach the Judgement
     section as evidence — if a candidate's whole claim rests on one, say plainly that the claim is
     untestable as posed and that the zone should be re-expressed as a set of folds), the
     `seed_flags` list (if empty,
     say so — "no cells exceeded 5× cohort median seed_std" is itself a fact worth stating),
     validation (NaN rate, PIT test result, INPUTS check result), and the noise-floor delta
     (`facts["noise_band"]`) — if `available: false`, say plainly "noise-floor band: not
     available" and quote `facts["noise_band"]["reason"]` verbatim rather than omitting the
     line or paraphrasing it (as of `fps-cf8` there is more than one refusal reason — no floor
     yet, a partial `--fold-subset` floor, or a `baseline_fingerprint` mismatch against a
     stale/re-locked floor — and the reason string already says which); if available, report
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
   - `outcome`: mechanical, from `facts["provenance"]["status"]` and, for a completed run,
     `facts["noise_band"]` — **not a judgement call** (`fps-3jj.17`, decided 2026-08-23):
     - `status == "disqualified"` → `outcome: disqualified`.
     - `status == "aborted_candidate"` → `outcome: aborted_candidate`.
     - `status == "rejected"` — the pipeline's own name for "ran to completion and reached the
       scoring stages" (`runner.py`'s outcome taxonomy comment; it says nothing yet about the
       claim's merit) — apply the **rejected-vs-inconclusive split**:
       - `facts["noise_band"]["available"]` is `false` → `outcome: rejected`. No noise floor
         means no ruler to grade "below the instrument's resolution" against, so there is nothing
         to call inconclusive — same as before this rule existed. Note the unavailability in the
         README as usual (Step 1 above); don't invent a floor to unblock this decision.
       - Otherwise read `z = facts["noise_band"]["candidate_z_vs_band"]` and
         `t = facts["noise_band"]["single_candidate_z_threshold"]` — both already computed by
         `dossier_tables._noise_band` (`t` is `family_wise_z_threshold(n_candidates=1, n_draws=...)`,
         i.e. the *single*-candidate detection bar, docs/CONVENTIONS.md's "-0.15 c/L judged
         singly" for `batch1`; deliberately **not** the batch-level `clears_family_wise_threshold`
         gate, which needs the whole batch dossiered and exists to answer a different question —
         "is any candidate surprising after correcting for picking the best of N", the retrospective
         routine's job, not this one's):
         - `z >= t` → `outcome: rejected`. `delta_cpl_held` is clearly the WRONG SIGN by more than
           this batch's own noise band — measurably worse than a placebo column, not merely
           unresolved.
         - `-t < z < t` → `outcome: inconclusive`. The effect sits inside the noise band — this run
           could not distinguish the candidate from noise. Per the ledger header, this is **not**
           `rejected`: the ground may still be open (more data, a different arm of the same
           mechanism, etc. is a legitimate follow-up, not a re-proposal of dead ground).
         - `z <= -t` → `outcome: rejected`, same as every run before this rule existed. The
           candidate clears single-candidate noise in the GOOD direction, but this dossier step
           still never decides graduation (see below) — a human who reads a strongly negative `z`
           and wants to pursue it upgrades this same entry's `outcome` to `graduated` by hand.
     The retryable statuses are filtered out at Step 0 and never reach a ledger entry. **Never
     write `graduated` here.** Per the parent design's decision boundary, this pipeline never
     touches `fuel_signal/features.py`; graduation is a separate, human-initiated PR, and a human
     updates this same ledger entry's `outcome` to `graduated` by hand when that PR lands (see how
     the `tgp_delta_7d` entry already reads `graduated` even though every pipeline run of it would
     have closed `rejected`).
   - `evidence`: the run directory's path, relative to the repo root.

5. **Prepend one row to `experiments/INDEX.md`'s table** in the same pass (its own convention: the
   two files must never drift, because neither is derived from the other) — Date / Name /
   Hypothesis / Result / Status, `Status` = `open` while the pipeline still has follow-ups queued
   for this candidate's series, `done` otherwise, `abandoned` for `disqualified` /
   `aborted_candidate` (never `graduated` — same human-edit-later rule as the ledger). Retryable
   statuses can't reach this step at all — see Step 0.

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
- **`noise_floor.json` per-batch step.** `batch_freeze.py` computes this as its own final step
  (`fps-cf8`), so a batch frozen normally always has one. `facts["noise_band"]["available"]`
  still reads `false` in three cases: an older batch frozen before `fps-cf8` or with
  `--skip-noise-floor` (never computed), a floor computed with `--fold-subset` (partial fold
  coverage), or a floor whose `baseline_fingerprint` no longer matches this run's (a re-lock —
  graduation or `LOCKED_FEATURE_COLUMNS` change — invalidates an existing batch's floor; see
  `docs/CONVENTIONS.md`). If you see `available: false`, that means this batch is missing a
  *matching* floor, not that the capability is unbuilt. Report it plainly in the README (see
  Step 1 above), quoting `facts["noise_band"]["reason"]`, rather than inventing a noise
  estimate.

## The shim

This is the exact text that should be the Routine's stored prompt (three lines, same pattern as
`docs/routines/worker.md`'s shim — no embedded steps, no reminders about what this file says):

```
You are the fuel-price-signal dossier routine.
Your working directory is /Users/esteele/Code/fuel-price-signal (the
fuel-price-signal repo, primary worktree — this is a persistent local checkout, not
a fresh clone).
Follow docs/routines/dossier.md exactly.
```

**This is a LOCAL routine.** An earlier revision of this shim said "your working directory already
has a fresh checkout", copied from the cloud worker. That is wrong here and not merely untidy: a
fresh clone has neither the run artifacts (`results.json`, `rowpreds.parquet`, `fills.parquet` are
produced on this machine) nor the batch snapshot (`features.parquet` and the cloned DB are
gitignored), so a fresh-checkout dossier routine would find an empty work queue every night and
exit quietly, forever. It must name the primary worktree, exactly as `docs/routines/launch.md`'s
shim does.

**Registered** (2026-08-22, `fps-3jj.11` prep) as the local scheduled task
`fuel-price-signal-dossier`, cron `0 5 * * *` local — 5:00 AM AEST daily. Separate from both the
chore/polish worker and the launch routine: three independent schedules against the same repo.

**Why 5 AM and not "later the same night".** Two constraints pull against each other. The run must
have finished (launch fires at 9 PM, a candidate takes tens of minutes at 1-day cadence, so ~10 PM
is already enough), but this routine uses Claude *actively* to write the dossier, and
`docs/routines/launch.md` puts the 10 PM–4 AM AEST window off limits for exactly that reason —
it is why launch itself sits at 9 PM. Same-night dossiering lands squarely inside it. 5 AM clears
the window, leaves hours of headroom over even a badly overrunning candidate, and has the dossier
waiting when the owner wakes up. If the run is still going, this routine exits quietly and picks
the candidate up the next morning, so a late finish costs a day rather than a write-up.
