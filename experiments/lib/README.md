# experiments/lib — paired walk-forward CV scaffolding

Shared helpers for `paired_wfcv.py` scripts. All imports require `PYTHONPATH=.`.

## constants.py
`SEEDS`, `SHOCK_FOLDS`, `LGBM_DEFAULTS` — shared constants. Import; never redefine per-script. Constants drift (LGBM params, seed tuple) between scripts is exactly what this lib prevents.

## fit.py
`fit_score(train_df, val_df, cols, seed)` — trains one LightGBM model with `LGBM_DEFAULTS` and returns `(log_loss, probas, wall_seconds)`. `per_row_log_loss(y, p)` — element-wise binary cross-entropy.

## folds.py
`iter_folds_with_baseline_fit(df, baseline_cols, seed=SEEDS[0])` — yields `(fold_idx, regime, train_df, val_df, baseline_ll, baseline_p, baseline_t, baseline_prl)`. Encapsulates the "fit baseline once per fold, reuse for R0+seed0" pattern. The per-fold loop body and all experiment-specific cohort masks stay in the calling script.

## cohorts.py
`hard_quantile_mask(prl, q)` — returns a boolean mask of rows in the hardest `(1-q)` fraction by per-row log-loss. `q=0.75` → hard25; `q=0.90` → hard10.

## gates.py
`seed_variance_gate(df_rows, cohort_ll_map)` — flags `(fold, run)` cells where `seed_std > 5× cohort median`. Returns `(summary_dict, flags_list)` and prints flagged cells. Raises `ValueError` if any cohort median is NaN or ≤ 0 (a zero denominator would silently suppress real outliers).

## aggregate.py
`aggregate_with_deltas(df_rows, cohort_ll_map, baseline_run="R0")` — groups by `(fold, regime, run)`, computes mean/median/seedstd per cohort column, and appends `delta_*_mean` / `delta_*_median` columns vs the baseline run. Ready to write directly to `fold_run.csv`.

## io.py
`to_jsonable(o)` — recursively converts non-finite floats to `None`. `write_meta(out_dir, meta)` — serialises `meta` with `to_jsonable`, writes `meta.json`, and prints the path.

## timing.py
`time_block(label)` — context manager that prints `  [label] N.Ns` on exit.
