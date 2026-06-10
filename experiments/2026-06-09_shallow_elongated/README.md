# Shallow-elongated regime constraint for A — issue #214

Tests whether the two-axis (elongation × shallowness) features close the
fold-7-style regression that the step5 row-level analysis traced to A's
`network_px_std` signal misreading coordination in extended-descent rows.

All three candidate features are computed **in-script** in `paired_wfcv.py`
from columns already present in features.csv (`cycle_mean_length`,
`cycle_days_since_peak`, `station_price_cents`, `cycle_last_max_cents`).
They only land in `fuel_signal/features.py` via a follow-up PR if this
experiment graduates them — mirrors the a_c_ablation → #216 pattern.

## Candidate features

- **`elongation_ratio`** = `cycle_days_since_peak / station_baseline_cml`,
  where `station_baseline_cml` is the network-wide rolling median of
  `cycle_mean_length` over the 730d window ending at `(date - 1)` (`closed='left'`).
  Non-adaptive — the frozen baseline is the failure-mode fix for
  `step5d_leakage`'s adaptive recalibration. `cycle_mean_length` is identical
  across stations on a given date (Sydney-avg series under the hood), so
  "per-station median" in #214's framing → network-wide here.
- **`cycle_descent_slope_so_far`** =
  `(station_price_cents - cycle_last_max_cents) / cycle_days_since_peak`, null
  at the peak. Negative during descent; less negative = shallower.
- **`is_extended_shallow_descent`** =
  `(elongation_ratio > 1.0) AND (cycle_descent_slope_so_far > -0.9)`.

## Run grid

Baseline is the locked 54-feat baseline from #216 (`FEATURE_COLUMNS +
LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS` — A and C already in).

| Run         | Columns added to 54-feat baseline                                    |
|-------------|----------------------------------------------------------------------|
| R0          | (baseline — A+C already locked in via #212/RAC_full)                 |
| R_raw       | + `elongation_ratio` + `cycle_descent_slope_so_far`                  |
| R_composite | + `is_extended_shallow_descent`                                      |

3 runs × 14 folds × 5 seeds = **210 LightGBM fits**. Expected wall ≈ 10–12 min
(extrapolated from a_c_ablation's 350-fit run at ~17 min).

## Decision rule (per #214)

Primary gates, median across 5 seeds:

1. **Fold-7 hard25 Δll** for the winning run vs R0: reduction of ≥ +0.04
   (closing >half of the +0.084 regression observed in step5).
2. **No fold's hard25 Δll** vs R0 may regress by more than +0.01.
3. **Net population Δll** vs R0 must be ≥ 0 (across all rows).
4. **Per-(fold, bucket) row-level**: the winning run must reduce mean Δll
   on `ext_descent_shallow` rows specifically — verifying the mechanism.

If R_raw passes → graduate the two raw features.
If only R_composite passes → graduate the composite with the threshold
caveat noted in `analysis.md`.
If neither → document the negative result in `project_late_descent_investigation`
and raise priority of #215's external-data branch.

## Per-row predictions

Per #214 gate 4, the script saves a parquet with predicted probabilities for
every (fold, run, seed, row) so the row-level test on the
`ext_descent_shallow` bucket can be done downstream (mirrors `step5_rowpreds.parquet`).

Bucket definitions (matching step5d/step5e):

- `ext_descent` = row's `elongation_ratio > 1.0` (using the frozen 730d
  baseline, not the live `cycle_mean_length`).
- `ext_descent_shallow` = `ext_descent` AND `cycle_descent_slope_so_far > -0.9`.

The bucket masks are computed at row save time and stored alongside the
predictions; row-level diagnostic is `analysis.py` work post-run.

## Methodology requirements (#214)

- Mean AND median seed-aggregations reported; median is the headline.
- Seed-variance gate: per (cohort, fold, run), flag any cell whose
  `seed_std / median(seed_std across cohort) > 5×`. Drill in before quoting.
- Walltime per fit + per phase. `load_features()`. `PYTHONPATH=.` prefix.
  LightGBM fit + predict with DataFrames.

## Run

```bash
PYTHONPATH=. uv run python experiments/2026-06-09_shallow_elongated/paired_wfcv.py \
  2>&1 | tee experiments/2026-06-09_shallow_elongated/run.log
```

## Outputs

- `runs.csv` — one row per `(fold, run, seed)`: ll per cohort, fit seconds.
- `fold_run.csv` — one row per `(fold, run)`: mean + median per cohort,
  seed_std, delta vs R0 (both mean and median).
- `rowpreds.parquet` — `(fold, run, seed, station_code, price_date, label,
  proba, is_hard25, is_ext_descent, is_ext_descent_shallow)`.
- `meta.json` — config, summary, seed-variance flags.
- `run.log` — captured stdout (tee'd).

## Acceptance criteria (mirrored from #214)

- [ ] Lab-book entry committed (README + script + result CSVs + meta + log + rowpreds).
- [ ] Decision recorded against the four gates in `analysis.md`.
- [ ] If a feature graduates → follow-up issue to land it in `fuel_signal/features.py`
      (with PIT-safe + frozen-baseline unit tests) and retrain the 54→N-feat baseline.
- [ ] If rejected → document in `project_late_descent_investigation`, raise
      priority of #215's external-data branch.

## Out of scope

- **Fold 9** will not be rescued here (step5e: shallow +0.094, steep +0.046 —
  both buckets bad). The non-adaptive `cycle_days_since_peak_vs_95p_history`
  side-quest gets its own issue if R_raw or R_composite passes; tracked as
  out-of-scope in #214.

## Related

- Source: `experiments/2026-06-06_late_descent_triplet/step5_analysis.md`
  (row-level evidence) + `step5d_leakage.png` + `step5e_shallowness.png`.
- Memory: `project_late_descent_triplet_outcome`,
  `project_late_descent_elongation_regime`,
  `feedback_check_seed_variance_before_trusting_mean`,
  `feedback_seed_discipline`.
- Planning: #215 (resolves on this outcome).
- Supersedes the broader scoping in #206 (closed 2026-06-09).
