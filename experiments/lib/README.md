# experiments/lib — paired walk-forward CV scaffolding

Shared helpers for `paired_wfcv.py` scripts. All imports require `PYTHONPATH=.`.

## In-script / lib seam

**In-script (you own per experiment):**
- `add_candidate_columns()` — compute new columns from features.csv columns
- `RUNS` dict — run grid (R0 = baseline, R1+ add candidate cols)
- `GATE` / `GateSpec` thresholds — numeric pass/fail criteria for this experiment
- Cohort and bucket boolean masks — anything beyond `hard_quantile_mask`
- `meta["definitions"]` — human-readable column/bucket descriptions

**Lib (always import; never inline):**
- Fold iteration, fitting, per-row loss, cohort mask, row-pred collection, seed-variance gate, aggregation, gate evaluation, meta I/O, timing, shared constants

**Promotion rule:** if an `add_candidate_columns` block is copied into 2+ experiments unchanged, extract the primitive into `experiments/lib/features/` and import it.

**Canonical skeleton:** `experiments/TEMPLATE_paired_wfcv.py` — copy, rename, fill in the TODOs.

## constants.py
`SEEDS`, `SHOCK_FOLDS`, `LGBM_DEFAULTS` — shared constants. Import; never redefine per-script. Constants drift (LGBM params, seed tuple) between scripts is exactly what this lib prevents.

## fit.py
`fit_score(train_df, val_df, cols, seed)` — trains one LightGBM model with `LGBM_DEFAULTS` and returns `(log_loss, probas, wall_seconds)`. `per_row_log_loss(y, p)` — element-wise binary cross-entropy.

## folds.py
`iter_folds_with_baseline_fit(df, baseline_cols, seed=SEEDS[0])` — yields `(fold_idx, regime, train_df, val_df, baseline_ll, baseline_p, baseline_t, baseline_prl)`. Encapsulates the "fit baseline once per fold, reuse for R0+seed0" pattern. The per-fold loop body and all experiment-specific cohort masks stay in the calling script.

## cohorts.py
`hard_quantile_mask(prl, q)` — returns a boolean mask of rows in the hardest `(1-q)` fraction by per-row log-loss. `q=0.75` → hard25; `q=0.90` → hard10.

## gates.py

**Sign convention (single-sourced here): `Δ = run − R0`; negative is better. A gate passes when `value <= threshold`.**

`GateSpec` — frozen dataclass: `cohort_col`, `pop_col`, `target_fold`, `target_max`, `worst_fold_max`, `net_pop_max`. Thresholds stay in the calling script; the helper owns the direction/comparison.

`evaluate_gates(fold_run, spec, run) -> list[dict]` — evaluates the three standard gates for one run. Returns `[{name, threshold, value, passed}, ...]`. Feed directly into `meta.json` and print a verdict table.

`seed_variance_gate(df_rows, cohort_ll_map)` — flags `(fold, run)` cells where `seed_std > 5× cohort median`. Returns `(summary_dict, flags_list)` and prints flagged cells. Raises `ValueError` if any cohort median is NaN or ≤ 0 (a zero denominator would silently suppress real outliers).

## aggregate.py
`aggregate_with_deltas(df_rows, cohort_ll_map, baseline_run="R0")` — groups by `(fold, regime, run)`, computes mean/median/`{col}_seedstd` per cohort column, and appends `delta_*_mean` / `delta_*_median` columns vs the baseline run. Ready to write directly to `fold_run.csv`.

## io.py
`to_jsonable(o)` — recursively converts non-finite floats to `None`. `write_meta(out_dir, meta)` — serialises `meta` with `to_jsonable`, writes `meta.json`, and prints the path.

## timing.py
`time_block(label)` — context manager that prints `  [label] N.Ns` on exit.

## rowpreds.py
`RowPredCollector(ident_base)` — accumulates per-`(run, seed)` row-level prediction blocks across all folds and writes a single parquet. Call `collector.ident_base = ident` at the start of each fold to update the base DataFrame, then `collector.add(run, seed, proba)` inside the run×seed loop, and `collector.to_parquet(path)` after all folds. Owns the dtype decisions: `seed` → `int16` (never overflows for any plausible seed value), `proba` → `float32`.

## realised.py — the objective-aligned arbiter (#255)

`run_paired_realised_backtest(arms, feature_columns, ...)` — the realised-spend
counterpart to the `folds.py` log-loss screen. Walk-forward over the **same** WFCV
folds; per fold trains a raw LightGBM + isotonic calibrator and picks τ via OOF on
the fold's train (mirrors production #236); replays each fold's val window through
the real `aggregate_backtest` economics; scores each arm at its **own** τ and a
**held** common τ (clean attribution — the #254 τ-move-vs-feature decomposition);
pools spend + litres across windows for an honest aggregate CPL. Returns a
`RealisedResult` (`per_window`, `aggregate`, `deltas`, `meta`) — programmatic, no
production-artifact or `results.csv` writes.

- `ArmSpec(name, df, detector_factory=CycleDetector)` — one arm. `df`'s canonical
  cycle columns hold THAT arm's values (trained on); `detector_factory` is the live
  cycle source for the replay. Arms share an index (`candidate = baseline.copy()`
  with cycle cols overwritten). In-process injection — **no production branch**.
- A single arm is the degenerate gate-1 use (per-regime realised regret, #259):
  group the per-window CPL by the caller's regime labels.
- Isotonic-only calibration (the AC3 lock) keeps a paired run near WFCV wall-clock.
  Tune cost with `fold_subset` / `inner_fold_params` / `station_codes`.

Relies on two `fuel_signal/backtest.py` injection seams (added in #255):
`PriceHistory(..., detector_factory=...)` and `ModelStrategy(pipeline=..., feature_columns=...)`.

## zones.py

Helpers shared across realised-fill ledger experiments that tag fills by cycle zone.

`CYCLE_REGIME_BANDS` — canonical three-band cut on `cycle_pct_through` (= days_since_peak / mean_cycle_length): `normal` [0, 0.6), `late_descent` [0.6, 1.0), `overdue` [1.0, ∞). Single-sourced here; never redefine per-script.

`assign_regime(pct)` — maps a `cycle_pct_through` float to its regime name. Returns `"unmatched"` for NaN (fill date had no prior feature row via the as-of join in `cleanup_checks.py`); falls back to `"normal"` for any value outside the band table.

`pooled_cpl(fills)` — pooled cost-per-litre from a fill ledger: `spend_cents.sum() / litres.sum()`. Returns NaN when the ledger has zero total litres.

## features/ (sub-package)

Primitives for the inside of `compute_features()` / `add_candidate_columns()`. See `features/README.md` for full docs.

| Helper | Module |
|---|---|
| `cohort_std_by_date(df, mask)` | `features/dispersion` |
| `cohort_agg_diff_by_date(df, mask_a, mask_b)` | `features/dispersion` |
| `calendar_aware_delta(per_date_series, lag_days)` | `features/deltas` |
| `rolling_baseline(per_date_series, window_days)` | `features/rolling` |
| `px_change_lag_diagnostic(df, lag_days)` | `features/diagnostics` |
