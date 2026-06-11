# Interaction-column probe for A in the shallow-elongated corner — issue #231

Tests whether handing the tree an explicit **interaction column** — rather than
the raw axes separately — closes the fold-7-style regression that `elongation_ratio`
+ `cycle_descent_slope_so_far` (R_raw, #214) failed to close.

## Why

The #214 oracle diagnostic (`experiments/2026-06-09_shallow_elongated/analysis.md`)
established:

1. A's level (`network_px_std`) **does** separate elongated vs normal from
   phase 0.2 onwards.
2. A's level **does not** separate shallow vs steep within elongated until
   phase > 1.4 — the wrong axis for the failure regime.
3. R_raw failed because expressing "A misreads when (elongated AND shallow)"
   needs **three nested tree splits**, and the corner is ~8% of train data —
   the tree never found the conditional. R_composite (one binary flag) was also
   flat, ruling out "model doesn't know the regime exists".

So the open question is whether the failure is about **combination
representation** (the tree can't assemble the interaction from raw axes) or
**underlying signal availability** (A simply doesn't carry distinctive
information in the corner — the oracle diagnostic's pessimistic read). Handing
the tree the product directly tests exactly this: if it's representation, the
interaction column helps with one split; if it's availability, the column
multiplies noise and stays flat. See memory `feedback_tree_interaction_limits`.

## Candidate columns (computed in-script)

All computed **in-script** in `paired_wfcv.py` from columns already in
features.csv. They land in `fuel_signal/features.py` only via a follow-up PR if
this experiment graduates them (mirrors a_c_ablation → #216, #214).

Building blocks (same construction as #214, PIT-safe):

- **`elongation_ratio`** = `cycle_days_since_peak / station_baseline_cml`, where
  `station_baseline_cml` = network-wide rolling median of `cycle_mean_length`
  over the 730d window ending `(date - 1)` (`closed='left'`, non-adaptive frozen
  baseline). Computed via `experiments.lib.features.rolling.rolling_baseline`.
- **`cycle_descent_slope_so_far`** =
  `(station_price_cents - cycle_last_max_cents) / cycle_days_since_peak`, null at
  the peak. Less negative = shallower.
- **`is_extended_shallow_descent`** =
  `(elongation_ratio > 1.0) AND (cycle_descent_slope_so_far > -0.9)` — the corner.

Interaction columns under test (`A` = `network_px_std`, already in the baseline):

| Column | Formula | What it gives the tree |
|---|---|---|
| `A_x_shallow_elong` | `A × is_extended_shallow_descent` | A's value inside the corner, 0 outside — one split isolates corner-A |
| `A_x_other` | `A × (1 - is_extended_shallow_descent)` | complement — split normal-regime A independently of corner-A |
| `A_x_smooth` | `A × elongation_ratio × max(0, slope + 0.9)` | continuous corner weight; 0 for steep descent, avoids the hard threshold |

## Run grid

Baseline is the locked 54-feat baseline (`FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
+ NETWORK_FEATURE_COLUMNS` — A and C already in via #212/RAC_full).

| Run | Columns added to 54-feat baseline |
|-----|-----------------------------------|
| R0  | (baseline)                        |
| R1  | + `A_x_shallow_elong`             |
| R2  | + `A_x_shallow_elong` + `A_x_other` |
| R3  | + `A_x_smooth`                    |

4 runs × 14 folds × 5 seeds = **280 LightGBM fits**. Expected wall ≈ 13–16 min
(extrapolated from #214's 210-fit run at ~10–12 min).

## Decision rule (same four gates as #214)

Median across 5 seeds, headline:

Sign convention (per `docs/CONVENTIONS.md`): `Δll = run − R0`, negative is better.

1. **Fold-7 hard25 Δll** for the winning run vs R0: `Δll ≤ −0.04` (an improvement of at least 0.04).
2. **No fold's hard25 Δll** vs R0 may regress by more than +0.01 (`Δll ≤ +0.01` on every fold).
3. **Net population Δll** vs R0 must be `≤ 0` (neutral-or-better across all rows).
4. **Per-(fold, bucket) row-level**: the winning run must reduce mean Δll on
   `ext_descent_shallow` rows specifically — verifying the mechanism (parquet
   saved for downstream `analysis.py`).

If a candidate passes all four → file a follow-up issue to land the winning
column(s) in `fuel_signal/features.py` (with PIT-safe + frozen-baseline unit
tests) and retrain the 54 → N-feat baseline.

If all fail → the failure is about underlying signal availability, not
representation; confidence in intra-series late-descent improvement drops.
Update `project_late_descent_investigation` and re-weight #215's external-data
track higher.

## Methodology requirements (#214)

- Mean AND median seed-aggregations; median is the headline.
- Seed-variance gate: per (cohort, fold, run), flag any cell whose
  `seed_std / median(seed_std across cohort) > 5×` before quoting aggregates.
- Walltime per fit + per phase. `load_features()`. `PYTHONPATH=.` prefix.
  LightGBM fit + predict with DataFrames.

## Run

```bash
PYTHONPATH=. uv run python experiments/2026-06-11_interaction_column/paired_wfcv.py \
  2>&1 | tee experiments/2026-06-11_interaction_column/run.log
```

## Outputs

- `runs.csv` — one row per `(fold, run, seed)`: ll per cohort, fit seconds.
- `fold_run.csv` — one row per `(fold, run)`: mean + median per cohort,
  seed_std, delta vs R0.
- `rowpreds.parquet` — `(fold, run, seed, station_code, price_date, label,
  proba, is_hard25, is_ext_descent, is_ext_descent_shallow)`. Gitignored.
- `meta.json` — config, summary, seed-variance flags.
- `run.log` — captured stdout (gitignored).

## Acceptance criteria

- [ ] Lab-book entry committed (README + script). Result CSVs + meta + analysis
      committed after the user runs the harness.
- [ ] Decision recorded against the four gates in `analysis.md`.
- [ ] If a column graduates → follow-up issue to land it in `fuel_signal/features.py`.
- [ ] If rejected → update `project_late_descent_investigation`; re-weight
      #215's external-data branch.

## Related

- Source: `experiments/2026-06-09_shallow_elongated/analysis.md`
  § "Or: try a model-level intervention — RECOMMENDED next probe".
- Memory: `feedback_tree_interaction_limits`, `feedback_oracle_diagnostic_pattern`,
  `project_a_dual_effect_mechanism`, `project_late_descent_investigation`.
- Predecessor: #214 (rejected, PR #228). Planning: #215 (closed — update after
  this resolves).
