# Within-family A ablation (+ C as complement) — issue #212

Decides which subset of Family A graduates to `fuel_signal/features.py`, and
whether C should graduate alongside.

Supersedes the broader 8-run grid in `experiments/2026-06-06_late_descent_triplet/`
under the corrected step2 verdict (B dropped, A standalone is the workhorse,
C is an additive complement with no synergy).

## Run grid

| Run        | Columns added to 50-feat baseline                                            | Question answered                       |
|------------|------------------------------------------------------------------------------|-----------------------------------------|
| R0         | (baseline)                                                                   | reference                               |
| RA_level   | `network_px_std`                                                             | does the level alone suffice?           |
| RA_delta   | `network_px_std_delta_3d`                                                    | does the Δ alone suffice?               |
| RA_both    | `network_px_std`, `network_px_std_delta_3d`                                  | does adding both beat either alone?     |
| RAC_full   | `network_px_std`, `network_px_std_delta_3d`, `lga_phase_std`, `lga_phase_std_delta_3d` | does C add a useful increment on A_both? |

5 runs × 14 folds × 5 seeds = **350 LightGBM fits**. Expected wall ≈ 15–20 min on
the dev machine (extrapolating from step2's 560-fit run).

## Decision rule

Pick the smallest A subset whose **median** hard25 lift is within ~0.005 of
RA_both (parsimony — favour fewer columns when the lift is comparable). Then:

- If RAC_full's median hard25 lift exceeds the best A subset by ≥ +0.010, C
  also graduates.
- Otherwise A graduates alone (subset per parsimony rule).

`median` not `mean` is the headline — per `feedback_check_seed_variance_before_trusting_mean`,
the mean-aggregated step2 verdict was poisoned by the s44 fold-2 outlier.
Both are reported.

## Methodology requirements (#212)

- **Mean AND median seed-aggregations** reported. Median = headline.
- **Seed-variance gate.** For each `(cohort, fold, run)` cell, compute
  `ratio = seed_std / median(seed_std across cohort cells)`. Flag any cell
  with `ratio > 5×` in stdout and in `meta.json`. Any flagged cell must be
  drilled into (per-seed listing) BEFORE quoting its aggregate.
- **Elongation-conditional diagnostic** (informational, NOT a graduation gate).
  Per-fold delta_hard25_median vs frozen-baseline elongation exposure,
  reported as Pearson r per non-baseline run. Same cycle-level definition as
  `experiments/2026-06-06_late_descent_triplet/step4_elongation_gradient_2d.py`:
  median per-row `cycle_days_since_peak / cml_baseline` where `cml_baseline` is
  the per-station 730d-trailing median of `cycle_mean_length`. The step2
  investigation showed this signal didn't cleanly generalise; #212 keeps it as
  a diagnostic for "does this feature help on the regime it was designed for".
- **Walltime instrumentation** per fit (per `feedback_instrument_walltime`).
- `from fuel_signal.features import load_features` (per `feedback_load_features_helper`).
- `PYTHONPATH=.` invocation prefix (per `feedback_experiment_scripts_pythonpath`).
- LightGBM fit + predict with DataFrames (per `feedback_lgbm_dataframe_consistency`).

## Run

```bash
PYTHONPATH=. uv run python experiments/2026-06-07_a_c_ablation/paired_wfcv.py \
  2>&1 | tee experiments/2026-06-07_a_c_ablation/run.log
```

## Outputs

- `runs.csv` — one row per `(fold, run, seed)`: ll per cohort, fit seconds.
- `fold_run.csv` — one row per `(fold, run)`: mean + median per cohort, seed_std,
  delta vs R0 baseline (both mean and median).
- `elongation_per_fold.csv` — per-fold frozen-baseline elongation score.
- `meta.json` — config, summary, seed-variance flags, per-run elongation r.
- `run.log` — captured stdout (tee'd).

## Cohort definitions

- `all` — full val set.
- `hard25` — top quartile of baseline (R0, seed 42) per-row log-loss, per fold.
  Primary empirical-labelling cut per #206.
- `hard10` — top decile, same construction.
- `lated` — "true late descent": `cycle_pct_through ≥ 0.9` AND 5d backward
  price change ≤ −2c. Cohort the original late-descent triplet was designed for.

## Acceptance criteria (mirrored from #212)

- [ ] Lab-book entry committed (README + script + result CSVs + meta JSON + log).
- [ ] Median-aggregated attribution tables per cohort `{all, hard25, hard10, lated}`
      for the 4 non-baseline runs.
- [ ] Seed-variance diagnostic reported. Any flagged cell drilled into per-seed
      and explained before its aggregate is quoted.
- [ ] Per-fold delta vs elongation-exposure diagnostic reported for the winning
      A subset and RAC_full.
- [ ] Decision recorded with reasoning: which subset graduates
      (RA_level / RA_delta / RA_both / RAC_full), or none.
- [ ] If a subset graduates: follow-up issue to land the column(s) in
      `fuel_signal/features.py` and retrain the 50→N-feat baseline.
- [ ] If no subset graduates: close the intra-series late-descent track per
      `project_late_descent_investigation` (and resolve #215 accordingly).

## Next steps once results land

1. Drill into any seed-variance flag (per-seed listing for the offending cell).
2. Read the median-headline summary. Apply the decision rule above.
3. Write `analysis.md` summarising the verdict, the elongation read, and the
   contingency decision for #215.
4. If A graduates: file the implementation issue (land the columns in
   `fuel_signal/features.py`, retrain the 50→N-feat baseline).
5. Hand off to #214 (shallow-elongated constraint) per the planning chain.
