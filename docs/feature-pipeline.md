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
- **Outcome-status taxonomy** (`experiments/pipeline/runner.py`): `graded` /
  `disqualified` / `aborted_candidate` are TERMINAL — the candidate got a fair hearing.
  `aborted_pipeline` / `aborted_environment` are RETRYABLE — it didn't, and the claim goes
  back on the queue (bounded by `fps-rtd`'s retry budget, one retry, then `blocked`). Every
  downstream consumer (dossier, retrospective) treats these differently: folding a retryable
  abort into `graded` would misreport a candidate that was never actually tested.
- **`graduated` is a human-only edit.** No pipeline stage ever writes `outcome: graduated`
  into `ledger.yaml` — graduation means landing code in `fuel_signal/features.py`, a separate,
  deliberate PR. The pipeline's own terminal status for a feature that clears the arbiter is
  still just `graded`/`disqualified` from ITS run; a human updates the ledger entry by hand
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
- **Arity is not capped; reuse is PRICED (`fps-3jj.25`, `fps-3jj.21`, `fps-3jj.14`).** A
  candidate is one mechanism and may be any number of columns — the field's own precedent
  (per-LGA candidates) needs wide groups, and a hard `MAX_RULER_ARITY` cap was tried and
  reverted: it let the ruler dictate what the generator was allowed to propose, which cost a
  wide candidate its verdict entirely rather than just a harder bar. Instead,
  `placebo.effective_n_draws` prices column reuse: two draws sharing `s` of `arity` source
  columns are charged `TEXTURE_ICC_BOUND × s / arity` correlation, not the full ICC, and
  `family_wise_z_threshold` (`scipy.stats.t.ppf`, fractional `df` allowed) uses that effective
  draw count in place of the nominal one. A wider candidate reuses more source columns per
  draw, so its band is worth fewer effective draws, so its bar widens automatically — harder
  as width grows, never refused. Both draw pools (source columns, block seeds) are unbounded;
  pick `n_draws` for the certainty you want and pay the compute.
- **`TEXTURE_ICC_BOUND = 0.274` is MEASURED, not assumed (`fps-3jj.23`, 2026-08-27) — a 95%
  upper bound, not a point estimate.** One-way ANOVA of 32 pinned-source draws (8 source
  columns × 4 block seeds, `noise_floor.py --same-source-column`) grouped by source column:
  point estimate ~0 (`F(7,24)=0.735, p=0.65`), one-sided 95% upper bound 0.274. Read a point
  estimate of ~0 as "could not see it," never "it is not there" — the constant deliberately
  carries the pessimistic (upper-bound) end, and a cheaper design (one column × 10 seeds) was
  tried first and rejected as worthless (its best possible bound was looser than the value it
  would have replaced). Full derivation: `experiments/2026-08-27_texture_icc/`.
- **A shared block SEED — not a shared source column — is what destroys a placebo bank's
  independence (`fps-3jj.20`, fixed by `fps-3jj.21`).** Two placebos built from the same block
  seed get the identical date-block rearrangement, so a near-duplicate pair of source columns
  comes out of the shuffle still correlated with each other (measured up to |ρ|=0.97).
  `placebo.block_seed`'s counter never restarts, closing this by construction. A floor
  computed before 2026-08-26 may carry the old failure mode; floors after that date are clean.
- **Recomputing a floor after a re-lock — the two paths differ.** A *column*-lock (a
  graduation changing `LOCKED_FEATURE_COLUMNS`) recomputes in place:
  `noise_floor.py <batch> --force`. A *cadence* re-lock does NOT — `check_freeze_cadence`
  refuses `--force` outright when the batch's `freeze.json` cadence disagrees with the live
  `TankParams()` default, because data frozen at the old cadence can't retroactively become a
  new-cadence batch. Freeze a NEW batch at the new cadence instead.
- **Promoting a wider-arity floor to be the grading ruler is a rename, not a compute** —
  `_noise_band()` reads one hardcoded filename, `noise_floor.json`; nothing reads an
  arity-suffixed side-file. Compute beside the current ruler at the SAME `n_draws`, then:
  ```
  mv <batch>/noise_floor.json    <batch>/noise_floor_k1.json
  mv <batch>/noise_floor_k3.json <batch>/noise_floor.json
  ```
  Never `--force` over `noise_floor.json` directly — it's mechanically safe for grading
  (one-sided: a wider ruler only raises the bar) but destroys the k=1 baseline and the ruler
  already-written dossiers were graded against. Both are tracked in git and recoverable, but
  recovery isn't a procedure — rename, keep both.
- **A fresh batch freeze always produces an arity-1 floor**, because at freeze time no
  candidates exist yet to know the real modal arity. Budget one extra floor run per batch
  (~2h at k=3) to promote to the batch's actual grading ruler once the first candidates are
  filed — this is the normal shape, not a surprise (`fps-3jj.19`).
- **The bar is a detection threshold, not a target, and it is BATCH-SPECIFIC — never reuse a
  number from one batch as a constant for another.** `batch1`'s own honest single-candidate
  bar (10 draws, arity 3, 1d cadence): −0.17 c/L judged alone, −0.27 c/L at the batch level (5
  candidates, Bonferroni-corrected). Recompute and cite each batch's own floor.

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

### Grading on a wider station population (`--n-stations`)

Both the floor and the runs it grades default to the five commute stations
(`PREFERRED_STATIONS`). `fps-nas` decided to **grade on a broad 410-station Sydney sample
and report the five**; `--n-stations N` is how each side is pointed at that population.
Both take the flag, both draw through `experiments.lib.universe.draw_batch_universe` at the
same `UNIVERSE_SEED`, and both stamp `station_population` (a digest of the codes actually
replayed) into their artifact:

```bash
# the ruler — arity must be >= the widest candidate in the batch, and --out-name keeps it
# BESIDE the existing floor rather than replacing the one every dossier was graded against
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> \
    --n-stations 410 --n-draws 40 --arity 3 --out-name noise_floor_n410_k3.json

# a candidate at the same width (launch.py does NOT pass this — wide runs are by hand).
# NOTE the candidate path: point it at a WIDE-ONLY copy of the module, never at the
# five-station one in-place — see the out-dir bullet below.
PYTHONPATH=. uv run python -m experiments.pipeline.runner \
    --batch-dir experiments/batches/<batch> \
    --candidate experiments/candidates/<batch>_n410/<name>.py --n-stations 410
```

Five things that bite, in the order people hit them:

- **A run and a floor must agree on population, or the grade is refused.**
  `dossier_tables._bank_admissibility` compares the two `station_population` stamps. This is
  the intended behaviour (a five-station run graded against a 410-station band is
  meaningless), but it means widening is a *pair* of jobs — the floor alone grades nothing.
- **Doing both halves is still not enough: the dossier cannot make a wide bank the headline
  grade.** `_noise_band` reads one file by fixed name — `NOISE_FLOOR_FILENAME =
  "noise_floor.json"`, the canonical five-station bank — and there is no population-aware
  bank selection. So a 410 run is refused against the canonical bank on
  `station_population`, and the matching 410 bank is reachable only through
  `_comparable_noise_banks`, which is corroboration by contract: *"The canonical bank alone
  drives `candidate_z_vs_band` and therefore the ledger outcome … a sibling that disagrees
  is a fact to report, never a grade to substitute."* `fps-nas`'s criterion 5 (grade broad,
  report five) is therefore **not implementable through the dossier path as the code
  stands** — grade a wide run by reading the sibling's z into a hand-written experiment
  write-up, or change bank selection deliberately and by PR. Do not let a `--scan` quietly
  record five "refused: station_population" dossiers and call that the answer.
- **A floor must also be at least as wide in ARITY as the candidate.** An arity-1 floor
  refuses every arity-2/3 candidate. batch1's candidates are arity 2–3, so a grading floor
  there needs `--arity 3`. Arity costs essentially no extra wall clock — the fit loop is
  per draw, not per column (`noise_floor.py:409`), and the two batch1 410-station banks
  measured 42,572s at arity 1 against 44,316s at arity 3 for the same 40 draws x 14 folds
  (+4%). It costs a little resolution through source-column reuse, which
  `placebo.effective_n_draws` prices into the bar: 40 draws at arity 3 draw 120 column
  slots from a ~49-column pool, giving 28.08 effective draws and a `z_gate` of 1.7332
  against 1.7058 at arity 1.
- **`r0_cache.joblib` is one file per batch dir**, fingerprinted on `station_codes`. A
  mismatch falls back to a full refit — never a correctness risk — but alternating widths in
  one batch dir re-fits R0 on each flip (~17 min/run at n=410: 975s cached vs 2024s not).
  Group runs by width, or give a wide sweep its own batch dir.
- **A wide re-run will DESTROY the narrow run's `results.json` unless you give it its own
  candidate path.** `runner`'s output directory is `default_out_dir(candidate_path)` — the
  candidate path with `.py` stripped — and `run_candidate` unlinks any existing
  `results.json` there before it starts (`runner.py:393`). So re-running an
  already-dossiered candidate at a new width in place deletes the old result and leaves it
  desynchronised from the `facts.json`, `README.md` and PNGs beside it, which together are
  the batch's dossier record. **A separate batch dir does NOT fix this** — `out_dir` is
  derived from the candidate path, not the batch dir, so that only isolates `r0_cache`.
  Give the wide runs a parallel candidate directory of **symlinks** to the originals, so
  each gets its own out-dir while there is still one definition of each candidate. Nothing
  couples the batch to the candidate's parent directory name: the batch comes from
  `--batch-dir` and is stamped as `meta.batch_dir`. Worked example:
  `experiments/candidates/batch1_n410/`.

Measured properties of the wide floor, and why the bar moves, are in
`experiments/2026-09-06_noise_floor_n410/README.md`. **Read its Phase 2 section before
quoting any figure from it**: the arity-1 findings (the band narrows 1.44x with width; the
410 null carries a +0.056 c/L positive mean) do not reproduce at arity 3, which is the only
arity that can legally grade a batch1 candidate. At arity 3 the width narrowing is 1.065x
[0.679, 1.995] and the band mean is +0.0298 (p = 0.10). Widening the population is worth
+0.0342 c/L of bar at that arity, and the largest term in it is the draw count, not the
width.

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
