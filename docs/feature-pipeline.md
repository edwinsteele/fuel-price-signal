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

A batch is set up once (`batch_freeze.py` + `noise_floor.py`, both one-off per batch, not
scheduled), then candidates are filed against it and cycle through launch → dossier nightly
until the batch is done, at which point the retrospective closes it out.

| Stage | Module | Cadence | Bead |
|---|---|---|---|
| Batch freeze | `experiments/pipeline/batch_freeze.py` | once per batch, at setup | `fps-3jj.1` |
| Noise floor | `experiments/pipeline/noise_floor.py` | once per batch, right after freeze | `fps-3jj.9` |
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
- **Noise floor, not a placebo.** `noise_floor.py` fits the SAME frozen baseline at
  paired seeds (production `SEEDS` 42–46 vs a disjoint 47–51) and takes the realised-CPL
  delta — pure fit noise by construction, no definition of an "uninformative column" needed
  (a per-night placebo/permuted-column null was considered and rejected: for a market-wide
  series shared by every station, a within-date permutation is a no-op).
- **Multiple comparisons in the retrospective.** Grading N candidates against the SAME noise
  band and asking "did the best one beat noise?" is a different, easier-to-satisfy-by-chance
  question than grading one candidate alone. `retrospective.py`'s
  `family_wise_percentile_threshold` applies a Bonferroni correction so a batch's leaderboard
  doesn't silently promote a noise delta to a finding just because it was the best of several.

## Running it by hand

Every stage is a `click` CLI, invocable directly (not only via its routine):

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.batch_freeze <batch>
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch>
PYTHONPATH=. uv run python -m experiments.pipeline.runner --batch-dir experiments/batches/<batch> --candidate experiments/candidates/<batch>/<name>.py
PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables --scan experiments/candidates
PYTHONPATH=. uv run python -m experiments.pipeline.retrospective <batch>
```

`noise_floor.py` and `runner.py`'s realised stage are both heavy (single-arm or paired full
walk-forward fits, ~10–25 min) — see `bd recall` / `feedback_user_runs_pipeline` if invoking
interactively: hand the command to the user rather than shelling out to a long-running
process from within a session.

## Further reading

- `docs/routines/generator.md` — candidate authoring format, diversity/redundancy gates,
  the two `CONFIDENCE` fields, batch sizing.
- `docs/routines/launch.md` — nightly claim/run/retry mechanics.
- `docs/routines/dossier.md` — deterministic facts pass + the judgement write-up.
- `docs/routines/retrospective.md` — batch-level rollup, invocation timing, known gaps.
- `docs/routines/worker.md` — the separate chore/polish worker Routine (not part of this
  pipeline, but the same Claude Code Routines mechanism — see `docs/automation.md`).
- `bd show fps-3jj` — parent design bead, full rationale for every decision above.
