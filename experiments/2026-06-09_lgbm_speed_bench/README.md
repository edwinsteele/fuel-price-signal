# LightGBM fit-speed benchmark — issue #220

- **Date:** 2026-06-09
- **Branch:** main (committed directly — `experiments/**` exemption)
- **SHA:** 4caeaa2
- **Status:** done

## Hypothesis

Two config changes could cut per-fit time 10–20% at zero quality cost:

1. `force_col_wise=True` (C1) — switches LightGBM's internal histogram-building
   strategy from row-wise to col-wise; claimed faster on wide tabular data.
2. `lgb.train` + Dataset reuse (C2) — builds `lgb.Dataset` (feature binning)
   once per fold and reuses it across seeds, saving N−1 binning passes per fold.

Both were proposed in the 2026-06-07 session as quick wins for the A/C ablation
but never tested.  With 54-feat fits at ~3–4s each and 200–350 fits per
experiment, a 15% speedup saves ~1.5 min per run × N future experiments.

## Run grid

| Config | Description                                           |
|--------|-------------------------------------------------------|
| C0     | `LGBMClassifier(random_state=seed, verbose=-1, subsample=0.8, subsample_freq=1)` (current baseline) |
| C1     | C0 + `force_col_wise=True`                            |
| C2     | `lgb.train` with `lgb.Dataset` built once per fold, reused across seeds |

14 folds × 5 seeds × 3 configs = **210 LightGBM fits**.

Feature set: 54-feat RAC_full baseline (`FEATURE_COLUMNS + LGA_FEATURE_COLUMNS +
NETWORK_FEATURE_COLUMNS`, locked in PR #225).

## How to run

```bash
PYTHONPATH=. uv run python experiments/2026-06-09_lgbm_speed_bench/bench.py \
  2>&1 | tee experiments/2026-06-09_lgbm_speed_bench/run.log
```

## Outputs

- `runs.csv` — one row per `(fold, config, seed)`: ll, fit_s.
- `meta.json` — config, equivalence pass/fail, speed summary, preliminary decisions.
- `run.log` — captured stdout (gitignored).

## Equivalence check

`bench.py` reports per-fold `|ll(Cx) − ll(C0)|` on seed=42.  Tolerance = 1e-6.
**Any config that fails equivalence is automatically skipped** regardless of
speedup.

C2 uses a manually mapped param dict (`_LGBM_PARAMS_BASE`) to replicate
`LGBMClassifier` defaults.  If C2 fails equivalence, check the param mapping
first — add a side-by-side comparison of `model.get_params()` vs `_LGBM_PARAMS_BASE`.

## Decision criteria

The script prints a preliminary verdict; override in `analysis.md`:

| Condition | Decision |
|-----------|----------|
| equivalence fail | SKIP |
| speedup ≥ 10% + equiv pass | ADOPT (candidate) |
| speedup 5–10% + equiv pass | NEEDS-MORE |
| speedup < 5% + equiv pass | SKIP |

## Convention update (if adopted)

If C1 adopted: add `force_col_wise=True` to the `LGBMClassifier(...)` call in
all future `paired_wfcv.py`-style scripts and document in
`docs/CONVENTIONS.md` § Experiment scripts.

If C2 adopted: document the Dataset-reuse pattern in `docs/CONVENTIONS.md`
and consider a `fuel_signal/` helper so future scripts don't re-implement it.

## Next steps once results land

1. Verify equivalence per config.
2. Read speed summary — apply decision table above.
3. Write `analysis.md` with verdict + reasoning.
4. If adopting either config: update `docs/CONVENTIONS.md` and create a
   feedback memory entry (see issue #220 body).
5. Update `experiments/INDEX.md` and close #220.
