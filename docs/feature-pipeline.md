# AI-sourced feature engineering pipeline

Overview of the machinery under `experiments/pipeline/` — written once it settled (bd
`fps-3jj.8`, all of `.1`–`.7` and `.9` shipped). For design rationale and the full decision
history, read `bd show fps-3jj` and its children (`fps-3jj.1` … `.11`); this doc is the map,
not the territory.

## Why this exists

Two aims, both stated in the parent bead:

- **(a)** Find features that clear the realised-CPL arbiter without a human hand-designing
  and hand-running every candidate one at a time.
- **(b)** Gain experience running an AI-sourced pipeline end to end — the batch retrospective
  (`fps-3jj.8`) is the artifact that makes (b) legible: did the generator's stated
  confidence predict what actually happened?

**Local only.** GHA is out (2000 min/month budget); the remote worker routine gets a fresh
checkout with no data and structurally cannot run this; Viking OOMs on `features.py`. Every
piece below runs on the owner's Mac.

## The pipeline, in order

```text
generator ──files──▶ candidate modules ──claimed by──▶ launch (nightly) ──runs──▶ runner
                                                                                     │
                                                                                     ▼
                                                                          results.json / rowpreds / fills
                                                                                     │
                                                                                     ▼
                                                                     dossier (nightly) ──writes──▶ facts.json + README.md
                                                                                     │
                                                              (every candidate in the batch dossiered)
                                                                                     ▼
                                                                    retrospective (event-triggered) ──writes──▶
                                                                            retrospective_facts.json + RETROSPECTIVE.md
```

A batch is set up once (`batch_freeze.py`, one-off per batch, not scheduled — its final step
computes the noise floor itself, `fps-cf8`), then candidates are filed against it and cycle
through launch → dossier nightly until the batch is done, at which point the retrospective
closes it out.

| Stage | Module | Cadence | Bead |
|---|---|---|---|
| Batch freeze (incl. noise floor) | `experiments/pipeline/batch_freeze.py` | once per batch, at setup | `fps-3jj.1`, `fps-cf8` |
| Noise floor (recompute only) | `experiments/pipeline/noise_floor.py` | after a re-lock, or after `--skip-noise-floor` | `fps-3jj.9` |
| Generator | (a Claude session following `docs/routines/generator.md`) | event: queue drained / new batch | `fps-3jj.7` |
| Launch | `experiments/pipeline/launch.py` | nightly | `fps-3jj.5` |
| Runner | `experiments/pipeline/runner.py` | invoked BY launch, or by hand | `fps-3jj.4` |
| Validation | `experiments/pipeline/validate.py` | invoked BY runner | `fps-3jj.2` |
| Dossier | `experiments/pipeline/dossier_tables.py` | nightly, same night as launch | `fps-3jj.6` |
| Retrospective | `experiments/pipeline/retrospective.py` | event: batch fully dossiered | `fps-3jj.8` |
| Ledger | `experiments/ledger.yaml` + `experiments/INDEX.md` | written BY the dossier/retrospective routines | `fps-3jj.3` |

**Two things are nightly scheduled tasks (Claude Code Routines); the rest are event-triggered
interactive sessions.** Launch and dossier are the nightly pair — see
`docs/automation.md` for the mechanics of running Claude as a Routine, including the known
`bd dolt push` auth gap that currently has the worker routine disabled (unrelated to this
pipeline's own routines, but the same platform). The generator and retrospective are invoked
by a human (or an interactive Claude session) at batch boundaries, not on a timer —
`docs/routines/generator.md` and `docs/routines/retrospective.md` both say so explicitly.

## Directory layout

```text
experiments/
  batches/<batch>/            frozen snapshot: features.parquet, fuel_signal.db,
                               baseline_columns.json, freeze.json, noise_floor.json,
                               fit_stability.json (opt-in diagnostic, not auto-computed),
                               r0_cache.joblib, retrospective_facts.json
  candidates/<batch>/
    <name>.py                 the candidate module itself (generator output)
    <name>/                   run directory (runner.default_out_dir — .py suffix stripped)
      results.json            runner's raw output
      rowpreds.parquet        row-level predictions (WFCV-validation-window rows)
      fills.parquet           per-fill economics for the realised backtest
      facts.json              dossier's deterministic facts (numbers only, no prose)
      README.md               dossier's Facts/Judgement write-up
      *.png                   dossier's plots
  ledger.yaml                 structured claim record, one entry per claim tested
  INDEX.md                    prose lab book, one row per experiment dir
  RETROSPECTIVE_TEMPLATE.md   template for a batch's RETROSPECTIVE.md
```

## The code/Claude split

Every stage that writes a verdict follows the same division, decided at `fps-3jj.6`:
**code writes facts, a Claude session writes judgement, and every prose claim traces back to
a specific field in the facts file.** `dossier_tables.py` writes `facts.json`, never a verdict;
the dossier routine reads it and writes `README.md`. `retrospective.py` writes
`retrospective_facts.json`, never a verdict; the retrospective routine reads it and writes
`RETROSPECTIVE.md`. This is deliberate, not incidental: a number computed in-session by
Claude can't be checked short of a re-run, while a number in a facts file is inspectable and
the judgement session can crash and resume without losing the facts.

## Key contracts

- **The baseline is declared, never discovered** (`fps-zci`, `docs/CONVENTIONS.md`). R0's 54
  columns and their ORDER are `fuel_signal.features.LOCKED_FEATURE_COLUMNS`, re-exported as
  `experiments.lib.constants.BASELINE_COLUMNS`. Never re-derive this by inspecting a features
  frame — two real bugs (`fps-sa1` wrong set, `fps-zci` wrong order) came from doing exactly
  that, and both were invisible in the run artifacts that recorded them until fingerprinted.
- **Fingerprint everything.** `baseline_fingerprint()` / `LOCKED_FEATURE_FINGERPRINT` stamp a
  run's baseline identity into its meta so two runs' comparability is a mechanical `==` check,
  not an eyeball one (`bd recall baseline-fingerprint-before-comparing-runs`).
- **Outcome-status taxonomy** (`experiments/pipeline/runner.py`): `rejected` /
  `disqualified` / `aborted_candidate` are TERMINAL — the candidate got a fair hearing.
  `aborted_pipeline` / `aborted_environment` are RETRYABLE — it didn't, and the claim goes
  back on the queue (bounded by `fps-rtd`'s retry budget, one retry, then `blocked`). Every
  downstream consumer (dossier, retrospective) treats these differently: folding a retryable
  abort into `rejected` would misreport a candidate that was never actually tested.
- **`graduated` is a human-only edit.** No pipeline stage ever writes `outcome: graduated`
  into `ledger.yaml` — graduation means landing code in `fuel_signal/features.py`, a separate,
  deliberate PR. The pipeline's own terminal verdict for a feature that clears the arbiter is
  still just `rejected`/`disqualified` from ITS run; a human updates the ledger entry by hand
  once the graduation PR lands.
- **Noise floor is now a placebo-column null (`fps-awz`, reopening the decision this bullet
  used to record).** `noise_floor.py` fits the SAME frozen baseline at a SINGLE fixed seed
  (matching the realised arbiter's own `realised_seed` default) ~20 times, adding one PLACEBO
  column each draw — `experiments/pipeline/placebo.py` builds it as a block permutation of a
  real baseline column (contiguous date-blocks reordered) along whichever axis that column
  actually varies on (market-wide/per-date, or per-station), so it resembles a real feature
  in distribution and autocorrelation while its date-alignment to the target is destroyed. This measures the SAME
  operation a real candidate's `effect_delta_cpl_held` measures (hold seed/data/folds fixed,
  add one column) — the earlier seed-swap null held columns fixed and varied the seed
  instead, a different, unpaired perturbation with no fixed ratio to what candidates are
  judged on. Seed-swap is retained, not deleted, as a separate opt-in diagnostic
  (`noise_floor.py --fit-stability`, writing `fit_stability.json`, NOT part of the automatic
  `batch_freeze` path) — it measures real fit instability, just not this. **The earlier
  rejection of a per-NIGHT placebo still stands**: a fresh draw compared against a fresh
  candidate every night is a different, noisier design than a fixed bank of ~20 draws
  computed once at batch setup, the shape this replacement actually takes. A naive
  within-date shuffle is also still unsafe for a market-wide series (45 of the 54 locked
  columns are per-date, shared by every station on that date) — the construction reorders
  along the axis a column actually varies on rather than shuffling across stations for
  exactly this reason (see `placebo.py`'s module docstring).
- **CLOSED (`fps-d7m`): the construction is a block permutation, not a circular time-shift.**
  The shift could not decorrelate a slowly-varying column from itself — on batch0 it excluded
  ALL FOUR cycle-magnitude columns and every price-level column, leaving a bank dominated by
  counters/phase/differences (12 of 20 draws were `days_since_trough_entry_<lga>`), so the
  band characterized "adding a fast-varying column" rather than "adding an arbitrary column"
  while a real candidate feature can be level-like. Reordering contiguous date-blocks breaks
  the "adjacent in time implies similar value" property that caused it, and it covers the
  fast-varying columns too, so it REPLACED the shift outright rather than being added
  alongside it — no "is this column level-like?" gating heuristic to maintain. Verified the
  same way the original was: a full screen of all 54 batch0 columns, where 49 of 49 usable
  ones (5 are all-NaN) now pass at every seed — 245 checks, zero failures, median
  self-correlation 0.046 — so no *correlation*-driven substitution fires. Two substitutions
  still happen on every batch0 run for the unrelated all-NaN skip: two of the 20 primaries
  land on all-NaN LGA columns and are replaced from the fallback tail. **A narrower limitation remains and is
  named in `placebo.py`'s docstring**: a column whose information is mostly cross-sectional
  rather than temporal (`stickiness_score` at 0.47, `station_minus_sydney_avg_cents` at 0.33)
  cannot be decorrelated by ANY time-axis reorder, so those enter the bank as its weakest
  draws. `noise_floor.json` stamps every draw's `self_correlation` so this is visible per
  floor rather than only in prose.
- **The gate reads distance from the band, not empirical rank (`fps-awz`).**
  `dossier_tables.py`'s `family_wise_z_threshold` (moved there from `retrospective.py`,
  `fps-3jj.17`, so `dossier_tables._noise_band()` can attach a per-run threshold to
  `facts.json`; `retrospective.py` imports it) is a Bonferroni-corrected, t-distributed
  critical value in band-standard-deviation space; `clears_family_wise_threshold` compares
  each candidate's `candidate_z_vs_band` against it. The empirical percentile
  (`candidate_percentile_better_than_noise`) is still reported as descriptive colour but no
  longer drives the gate — with a draw count in the tens, the empirical-rank statistic only
  has `n_draws + 1` distinct values, so "gate on percentile" had silently collapsed to "beat
  every single draw" at the old 5-draw default.
- **The noise floor carries its own baseline identity, and a mismatch is refused, not
  trusted** (`fps-cf8`). `noise_floor.json` stamps `baseline_fingerprint` the same way every
  candidate run does; `dossier_tables._noise_band()` refuses (`available: False`) rather than
  grade whenever the floor's fingerprint doesn't match the run's — the same failure class as
  `fps-sa1`/`fps-zci`, just one level up: the floor is a run whose number is compared against
  EVERY candidate's. A re-lock (a graduation, or any `LOCKED_FEATURE_COLUMNS` change)
  invalidates an existing batch's floor; recompute with `noise_floor.py <batch> --force`
  (`docs/CONVENTIONS.md`).
- **Multiple comparisons in the retrospective.** Grading N candidates against the SAME noise
  band and asking "did the best one beat noise?" is a different, easier-to-satisfy-by-chance
  question than grading one candidate alone. `retrospective.py` calls `family_wise_z_threshold`
  (imported from `dossier_tables.py`, `fps-3jj.17`) with the whole batch's `n_candidates` here —
  applying a Bonferroni correction (in band-standard-deviation space, `fps-awz`) so a batch's
  leaderboard doesn't silently promote a noise delta to a finding just because it was the
  best of several; `family_wise_percentile_threshold` is still computed too, purely as the
  descriptive-colour percentile equivalent.

## Running it by hand

Every stage is a `click` CLI, invocable directly (not only via its routine):

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.batch_freeze <batch>
# noise_floor.py is invoked automatically as batch_freeze's final step (fps-cf8) — only
# run it directly to recompute (--force) after a re-lock, or after --skip-noise-floor:
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> --force
PYTHONPATH=. uv run python -m experiments.pipeline.runner --batch-dir experiments/batches/<batch> --candidate experiments/candidates/<batch>/<name>.py
PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables --scan experiments/candidates
PYTHONPATH=. uv run python -m experiments.pipeline.retrospective <batch>
```

`batch_freeze.py` (now including the noise floor) and `runner.py`'s realised stage are both
heavy (single-arm or paired full walk-forward fits, ~10–25 min) — see `bd recall` /
`feedback_user_runs_pipeline` if invoking interactively: hand the command to the user rather
than shelling out to a long-running process from within a session.

## Further reading

- `docs/routines/generator.md` — candidate authoring format, diversity/redundancy gates,
  the two `CONFIDENCE` fields, batch sizing.
- `docs/routines/launch.md` — nightly claim/run/retry mechanics.
- `docs/routines/dossier.md` — deterministic facts pass + the judgement write-up.
- `docs/routines/retrospective.md` — batch-level rollup, invocation timing, known gaps.
- `docs/routines/worker.md` — the separate chore/polish worker Routine (not part of this
  pipeline, but the same Claude Code Routines mechanism — see `docs/automation.md`).
- `bd show fps-3jj` — parent design bead, full rationale for every decision above.
