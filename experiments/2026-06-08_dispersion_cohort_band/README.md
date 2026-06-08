# Dispersion-cohort band ablation — issue #219

Decides whether `network_px_std` / `network_px_std_delta_3d` should use the
current ±5c cohort (`COMP_BAND_CENTS = 5.0`, set by the #212 ablation, landed
in PR #217) or the canonical Competitive band (±10c, defined by
`classify.PREMIUM_BAND_CENTS = 10.0` and persisted in
`station_class.class = 'Competitive'`).

The two thresholds are equivalent in code: `stickiness_score` in
`features.csv` is `median_premium_decicents / 10` in cents, so
`|stickiness_score| <= 10` selects exactly the rows that `sc.class =
'Competitive'` selects. ±5c is therefore a strict subset of the canonical
cohort, roughly half the size.

## Run grid

| Run  | Columns added to 50-feat baseline                                 | Cohort                          |
|------|-------------------------------------------------------------------|---------------------------------|
| R0   | (baseline)                                                        | n/a                             |
| R1   | `network_px_std`, `network_px_std_delta_3d`                       | `|stickiness_score| <= 5c`      |
| R2   | `network_px_std`, `network_px_std_delta_3d`                       | `|stickiness_score| <= 10c`     |

3 runs × 14 folds × 5 seeds = **210 LightGBM fits**. Expected wall ≈ 10–12 min
on the dev machine (extrapolating from the 350-fit ablation in
`experiments/2026-06-07_a_c_ablation/` which took ~15–20 min).

C (`lga_phase_std`, `+Δ`) is held out of this experiment — its cohort
(`LGA_FEATURE_COUNCILS`) is unaffected by the dispersion-band question.

## Decision rule (from issue #219)

Headline metric: **median-aggregated Δh25** (per `feedback_check_seed_variance_before_trusting_mean`).

- **Switch to canonical (±10c)** if `Δh25(R2) ≤ Δh25(R1) + 0.005` (note Δh25
  is negative-when-helpful, so this is "R2 within ~0.005 of R1's lift, or
  better"). Parsimony — prefer the cohort that reuses the existing
  classification semantics.
- **Keep ±5c** if `Δh25(R2) > Δh25(R1) + 0.005`.
- **Keep ±5c + document** if results are within noise — confirms the
  threshold is feature-specific and not equivalent to the canonical band.

## Methodology requirements

Mirrored from `experiments/2026-06-07_a_c_ablation/`:

- **Mean AND median seed-aggregations** reported. Median = headline.
- **Seed-variance gate.** Per `(cohort, fold, run)` cell, compute
  `ratio = seed_std / median(seed_std across cohort cells)`. Flag any cell
  with `ratio > 5×` in stdout and in `meta.json`. Any flagged cell must be
  drilled into per-seed BEFORE quoting its aggregate.
- **Walltime instrumentation** per fit (per `feedback_instrument_walltime`).
- `from fuel_signal.features import load_features` (per
  `feedback_load_features_helper`).
- `PYTHONPATH=.` invocation prefix (per
  `feedback_experiment_scripts_pythonpath`).
- LightGBM fit + predict with DataFrames (per
  `feedback_lgbm_dataframe_consistency`).

Note: no elongation-conditional diagnostic in this experiment — the band
question is orthogonal to elongation regime. #214 covers that axis.

## Run

```bash
PYTHONPATH=. uv run python experiments/2026-06-08_dispersion_cohort_band/paired_wfcv.py \
  2>&1 | tee experiments/2026-06-08_dispersion_cohort_band/run.log
```

## Outputs

- `runs.csv` — one row per `(fold, run, seed)`: ll per cohort, fit seconds.
- `fold_run.csv` — one row per `(fold, run)`: mean + median per cohort,
  seed_std, delta vs R0 baseline (both mean and median).
- `meta.json` — config, summary, seed-variance flags, decision.
- `run.log` — captured stdout (tee'd).

## Cohort definitions

- `all` — full val set.
- `hard25` — top quartile of baseline (R0, seed 42) per-row log-loss, per fold.
  Primary empirical-labelling cut per #206.
- `hard10` — top decile, same construction.
- `lated` — `cycle_pct_through ≥ 0.9` AND 5d backward price change ≤ −2c.

## Acceptance criteria (mirrored from #219)

- [ ] Lab-book entry committed (README + script + result CSVs + meta JSON + log).
- [ ] Median-aggregated attribution tables per cohort `{all, hard25, hard10, lated}`
      for R1 and R2.
- [ ] Seed-variance diagnostic reported. Any flagged cell drilled into per-seed
      and explained before its aggregate is quoted.
- [ ] Decision recorded (switch / keep / document-and-keep) with reasoning.
- [ ] If switching to ±10c: follow-up PR removes `COMP_BAND_CENTS` from
      `fuel_signal/features.py`, updates `_network_px_std_per_date` to filter
      via `sc.class = 'Competitive'`, and re-trains the 54-feat baseline.

## Next steps once results land

1. Drill into any seed-variance flag (per-seed listing for the offending cell).
2. Read the median-headline summary. Apply the decision rule above.
3. Write `analysis.md` summarising the verdict.
4. If switching: file the implementation issue (update
   `fuel_signal/features.py` to use `sc.class = 'Competitive'`, drop
   `COMP_BAND_CENTS`, retrain).
5. Either way: update `experiments/INDEX.md` and resolve #219.
