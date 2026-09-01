# ML pipeline reference

CLI reference for training, evaluating, calibrating, and diagnosing the ML signal
model — feature/label generation, model training, SHAP/redundancy analysis,
backtesting, and the AI-sourced feature pipeline. This is development reference
material, not day-to-day usage — see [README.md](../README.md) for setup, building
the database, and getting the signal.

For the facts this doc's commands produce, don't restate them here:

- [docs/STATUS.md](STATUS.md) — canonical current model state (feature count, on-disk artifact, calibration, τ, active phase)
- [docs/ML_SIGNAL.md](ML_SIGNAL.md) — design decisions behind the model
- [docs/feature-pipeline.md](feature-pipeline.md) — the AI-sourced candidate-feature pipeline's machinery (batch freeze → launch → dossier → retrospective)

## Generating ML training labels

Assemble a training table with one row per (station, date) that has a computable label:

```bash
# Default: 7-day horizon, 3c threshold, output to data/labels.csv
uv run python -m fuel_signal.labels

# Custom horizon and threshold
uv run python -m fuel_signal.labels --horizon 14 --threshold 5.0

# Custom output path
uv run python -m fuel_signal.labels --output /tmp/labels.csv
```

Each row contains `station_code`, `price_date`, `today_price_cents`, `future_min_cents`, and `label`. `label=1` (BUY) when **both** conditions hold: no cheaper price arrives within `--horizon` days (by more than `--threshold` cents), **and** today's price is at or below the `--percentile`th percentile of the past `--lookback` days. Rows without sufficient forward or lookback history are excluded.

### Diagnosing label distributions

```bash
# Produce two diagnostic plots in data/
uv run python -m fuel_signal.label_viz

# Custom input / output location
uv run python -m fuel_signal.label_viz --input /tmp/labels.csv --output /tmp/plots/
```

Writes `positive_rate_by_date.png` (fraction of stations labeled BUY per day — should oscillate with the ~45d price cycle) and `positive_rate_by_station.png` (histogram of per-station BUY rates — healthy distribution clusters near the marginal rate with no stations stuck at 0% or 100%).

### Inspecting individual label decisions

```bash
# Show per-day label decomposition for one station (21 days from --date)
uv run python -m fuel_signal.label_inspect --station 585 --date 2024-01-15

# Adjust window length or label parameters
uv run python -m fuel_signal.label_inspect --station 414 --date 2023-06-01 --days 30
uv run python -m fuel_signal.label_inspect --station 585 --date 2024-08-01 --horizon 14 --threshold 5.0
```

Prints a table showing `today_price`, `future_min`, the rolling `P33` threshold, and the two condition flags (`Cheap?`, `NoDrop?`) alongside the final label for each day. Useful for understanding why a specific date was or wasn't labeled BUY.

## Assembling ML feature rows

Join cycle features onto the labels table to produce a model-ready training set:

```bash
# Default: 7-day horizon, 3c threshold, output to data/features.csv
uv run python -m fuel_signal.features

# Custom horizon and threshold
uv run python -m fuel_signal.features --horizon 14 --threshold 5.0

# Custom output path
uv run python -m fuel_signal.features --output /tmp/features.csv
```

Requires `classify` to have run first — the LGA/brand mean joins fail silently to NULL otherwise.

Output includes all label columns (`station_code`, `price_date`, `today_price_cents`, `future_min_cents`, `label`) plus 15 feature columns:

- **Cycle features:** `cycle_pct_through`, `cycle_days_since_peak`, `cycle_mean_length`, `cycle_last_min_cents`, `cycle_last_max_cents`, `cycle_peak_count`
- **Station-vs-aggregate features:** `station_price_cents`, `station_minus_last_min_cents`, `station_minus_last_max_cents`, `station_minus_sydney_avg_cents`
- **LGA/brand mean features (Phase 3):** `lga_mean_cents`, `station_minus_lga_mean_cents`, `brand_mean_cents`, `station_minus_brand_mean_cents`
- **Station identity features:** `stickiness_score` — 45-day median of `station_price − LGA-Competitive-cluster median` (cents), sourced from `station_class.median_premium_decicents`. Provides a dedicated channel for the persistent-price-identity signal. Sticky stations receive the largest scores. NaN when no `station_class` row exists for that (station, date) pair.

The `stickiness_score` and the four LGA/brand mean columns can be NaN when `station_class` data is absent for that (station, date) pair. Rows are kept rather than dropped — downstream training scripts must handle the NaN (e.g. with imputation or a NaN-tolerant model like LightGBM).

Rows with insufficient history for cycle detection are excluded.

Production training runs on the locked 54-feature contract, not this raw 15-column
output — see [AGENTS.md § Canonical feature set](../AGENTS.md#canonical-feature-set-54-feat-baseline-locked-issue-216).

## Evaluation harness

`fuel_signal/evaluate.py` defines the canonical train/val/test split for the ML model and provides scoring utilities. The split is fixed — never adjust it after results are in. See [docs/STATUS.md § Canonical train/val/test split](STATUS.md#canonical-trainvaltest-split-fixed--do-not-adjust-after-results-are-in) for the dates.

7-day buffers between splits prevent label leakage (labels look 7 days forward). Val is sized for hyperparameter tuning; test is touched only for final evaluation.

Experiment results are appended to `experiments/results.csv` via `log_experiment()`. The baseline row (constant predictor at the train marginal rate ≈ 0.26) is the floor every model must beat:

```python
from fuel_signal.evaluate import split, baseline_prior, log_loss, brier, log_experiment
import numpy as np

train, val, test = split(df)  # df has price_date and label columns
p = baseline_prior(train)
pred = np.full(len(test), p)
print(log_loss(test["label"].values, pred))   # ≈ 0.573
print(brier(test["label"].values, pred))      # ≈ 0.192

log_experiment("my_model", features=["cycle_pct_through"], holdout_logloss=0.52, brier=0.18)
```

## Training the logistic regression baseline

The first real ML model — a vanilla logistic regression on the cycle features. Train on the canonical train split, score on val. Test is reserved for the locked final-model evaluation, so this command does **not** write to `experiments/results.csv`.

```bash
# Default: reads data/features.csv, writes data/models/logreg.joblib
# and experiments/reliability_logreg_val.png
uv run python -m fuel_signal.train_logreg

# Custom paths
uv run python -m fuel_signal.train_logreg \
    --features-csv /tmp/features.csv \
    --model-out /tmp/logreg.joblib \
    --reliability-out /tmp/reliability.png
```

Pipeline: `StandardScaler` → `LogisticRegression(max_iter=1000)`. Output prints train/val sizes and class balance, val log-loss / Brier, and the delta versus the constant-predictor baseline. The reliability plot uses 10 quantile bins with a `y=x` reference line; points below the diagonal indicate over-confidence, above indicate under-confidence.

## Walk-forward cross-validation report

Paired comparison across all pre-test folds. Re-trains both models on each 90-day fold and reports per-fold logloss delta (model − baseline). Use this before locking a Phase upgrade to confirm that the val improvement holds across fold windows and is not an artifact of the canonical val window.

Three modes:

**Two-artifact mode** — compare two saved joblib models:

```bash
uv run python -m fuel_signal.cv_report \
  --model data/models/lgbm.joblib \
  --baseline data/models/lgbm_phase3c.joblib \
  --features data/features.csv \
  --seed 42 \
  --output experiments/cv_phase4/results.csv
```

**Drop-feature mode** — test dropping one or more features from a single model without pre-building a baseline artifact:

```bash
uv run python -m fuel_signal.cv_report \
  --model data/models/lgbm.joblib \
  --drop-feature station_minus_last_max_cents \
  --features data/features.csv \
  --seed 42 \
  --output experiments/<dir>/results.csv
```

**Single-window mode** — cheap logreg walk-forward sanity check; no model artifact required:

```bash
uv run python -m fuel_signal.cv_report \
  --single-window \
  --features data/features.csv
```

`--drop-feature` is repeatable (`--drop-feature fa1 --drop-feature fa2`). `--baseline` and `--drop-feature` are mutually exclusive. Use `--single-window` for a quick logreg baseline check without a saved model. Providing neither `--baseline`, `--drop-feature`, nor `--single-window` raises an error.

Output: one line per fold (`val start→end`, `n_val`, `baseline=`, `model=`, `Δ=`) followed by a summary line (`folds`, `wins/n`, `median Δ`, `mean Δ`). Folds where `Δ > +0.05` are listed as named regressions. The `--output` CSV has columns: `fold_idx, train_start, train_end, val_start, val_end, n_val, baseline_logloss, model_logloss, delta`.

Paired walk-forward CV evidence is required before changing the production feature
set — see [docs/CONVENTIONS.md § Changing the production feature set](CONVENTIONS.md#changing-the-production-feature-set).

## SHAP analysis

`fuel_signal/shap_report.py` runs TreeExplainer on a fitted joblib model and emits three artifacts:

| Artifact | Contents |
|---|---|
| `shap_values.npy` | `(n_rows, n_features)` raw SHAP values — reused by ad-hoc notebooks and the `/features` inspect view |
| `summary.csv` | Per-feature: `mean_abs_shap`, `rank`, `sign_of_r` (sign of Pearson r between feature and SHAP), `nan_fraction` |
| `dependence/<feature>.png` | Scatter of feature value vs. SHAP value, one PNG per feature |

```bash
# Val split (default) — standard per-phase diagnostic
uv run python -m fuel_signal.shap_report \
    --model data/models/lgbm.joblib \
    --features data/features.csv \
    --output experiments/shap_phase4/

# Test split
uv run python -m fuel_signal.shap_report \
    --model data/models/lgbm.joblib \
    --features data/features.csv \
    --split test \
    --output experiments/shap_test/
```

Prints a ranked table (top 25 features) with `mean|SHAP|`, sign of correlation, and NaN fraction. NaN-bearing LGA features (stations below the 3-station floor) are handled without error; their SHAP contributions are computed from non-NaN rows.

## Feature redundancy + decomposition (SHAP)

`fuel_signal/feature_redundancy.py` answers two questions in one pass over a fitted model:

1. **Which features are redundant?** Hierarchical clustering on row-wise correlation of SHAP-value columns. Features in the same cluster contribute the same signal to the model, even when their raw values are not linearly correlated.
2. **Which features are carrying multiple signals?** SHAP interaction matrix → per-feature normalised entropy of partner mass. A diffuse interaction distribution flags a feature as a candidate to decompose into separate engineered features.

```bash
uv run python -m fuel_signal.feature_redundancy \
    --model data/models/lgbm.joblib \
    --features data/features.csv \
    --split val \
    --output experiments/redundancy_phase4/ \
    --cluster-threshold 0.5 \
    --interaction-sample 3000
# Fast SHAP-only pass (CV skipped; paired_cv_* columns emitted as NaN/empty):
uv run python -m fuel_signal.feature_redundancy ... --skip-paired-cv
```

Artifacts: `shap_corr.csv`, `clusters.csv`, `dendrogram.png`, `interaction_matrix.csv`, `decomposition_candidates.csv` (ranked by `entropy_norm` desc), plus `feature_columns.json` and `params.json`. When paired CV is enabled (default), `cv_clusters/` and `cv_decomp/` subdirectories hold per-fold CSVs.

`--interaction-sample` caps the row count used for `shap_interaction_values`, which is O(rows × trees × depth²) — much slower than plain SHAP.

**Regime-stability columns** (added to both `clusters.csv` and `decomposition_candidates.csv`):

| column | description |
|---|---|
| `paired_cv_median_delta` | Median Δ logloss across folds (dropped vs full-feature baseline). Negative = drop wins. |
| `paired_cv_worst_fold_delta` | Worst (most adverse) single fold Δ. A large positive value flags regime sensitivity. |
| `paired_cv_fold_wins` | `n_folds_won/n_folds` — how often the drop beats baseline. |
| `paired_cv_csv` | Relative path to the per-fold CSV under `--output`. |

**CV strategy:** One CV run per unique cluster (dropping the cluster representative — highest `mean_abs_shap` member) for `clusters.csv`; one CV run per feature for `decomposition_candidates.csv`. All available walk-forward folds are used (default: 14 folds × 90-day windows over the pre-test span). With ~50 features this takes 10–30 minutes depending on model size. Use `--skip-paired-cv` for a fast SHAP-only screening pass.

## Training the LightGBM baseline (Phase 3a.1)

Vanilla LightGBM on the **same 10 features** as Phase 2 — no new features, no tuning, `random_state=42`. No `StandardScaler` (trees are scale-invariant). This is the apples-to-apples model-class comparison. Does **not** write to `experiments/results.csv`.

```bash
# Default: every feature in the CSV — FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
# + any brand trough columns discovered in the header. Phase 4b when brand
# cols are present, Phase 4 otherwise. Reads data/features.csv, writes
# data/models/lgbm.joblib.
uv run python -m fuel_signal.train_lgbm

# Opt-out: Phase 4 (ignore brand trough columns even when present)
uv run python -m fuel_signal.train_lgbm --no-brand-features

# Opt-out: Phase 3c schema (15 features only)
uv run python -m fuel_signal.train_lgbm --no-lga-features

# Custom paths
uv run python -m fuel_signal.train_lgbm \
    --features-csv /tmp/features.csv \
    --model-out /tmp/lgbm.joblib \
    --reliability-out /tmp/reliability_lgbm.png

# Ablation: drop one or more features + sweep seeds (one-liner).
# --drop-feature is repeatable; errors out if the column is not in the
# resolved feature set. --seed is honoured end-to-end (build_pipeline +
# train_and_evaluate), so identical seeds reproduce val logloss exactly.
for seed in 0 1 2 3 42; do
  uv run python -m fuel_signal.train_lgbm \
      --drop-feature station_minus_last_max_cents \
      --seed $seed \
      --model-out /tmp/lgbm_drop_minus_max_seed$seed.joblib
done
```

**Phase 3a.1 val result** (2026-05-14, real DB): val logloss 0.3926 (baseline 0.6428, Δ −0.2501) vs logreg val logloss 0.4112. LGBM captures non-linearities logreg cannot.

The locked production baseline is `train_lgbm --no-brand-features` on the 54-feat
contract — never `train_lgbm`'s own default. See
[docs/STATUS.md § Model artifact paths](STATUS.md#model-artifact-paths-important).

## Feature diagnostics (LightGBM)

Prints feature importance, FN/FP mean-delta analysis, and an error-group summary against the canonical val split. Use this to understand which features drive misclassifications.

```bash
# Default: reads data/models/lgbm_calibrated.joblib and data/features.csv
uv run python -m fuel_signal.feature_diagnostics

# Custom model artifact or threshold
uv run python -m fuel_signal.feature_diagnostics --model-path data/models/lgbm_calibrated.joblib
uv run python -m fuel_signal.feature_diagnostics --threshold 0.35
```

Outputs three sections: (1) gain % and split count per feature sorted by gain; (2) FN−TP and FP−TN mean delta per feature sorted by |FN−TP|, showing where the model mis-ranks BUY vs WAIT rows; (3) TP/FP/TN/FN counts and predicted-BUY rate.

## LOO ablation (feature contribution check)

Measures whether dropping one or more features hurts, helps, or has no effect on val logloss. Fits LightGBM at multiple seeds with and without the dropped column(s), then reports mean ± std and a one-line verdict.

```bash
# Ablate a single feature (5-seed protocol)
uv run python -m fuel_signal.loo_ablation \
    --features-csv data/features.csv \
    --drop station_minus_lga_mean_cents \
    --seeds 1,7,42,99,2024

# Ablate a group of features at once
uv run python -m fuel_signal.loo_ablation \
    --features-csv data/features.csv \
    --drop lga_mean_cents \
    --drop station_minus_lga_mean_cents \
    --seeds 1,7,42,99,2024
```

`--drop` is repeatable. Omitting it entirely is an error (`nothing to ablate`). Each named column must appear in `FEATURE_COLUMNS` or the command exits with a clear error.

**Verdict thresholds** (relative to baseline std across seeds):

| Condition | Verdict |
|---|---|
| \|Δ\| < baseline_std | `within noise / redundant` |
| Δ > 0, outside band | `feature contributes (starved)` |
| Δ < 0, outside band | `feature harmful (unexpected)` |

Δ = LOO mean − baseline mean; positive means removing the feature(s) raised logloss (feature was useful).

## Calibrating the model

Check calibration quality and produce a calibrated model artifact. Works with any fitted model (logreg or LightGBM).

```bash
# Default: reads data/features.csv + data/models/lgbm.joblib,
# writes data/models/lgbm_calibrated.joblib
uv run python -m fuel_signal.calibrate

# Skip writing to experiments/results.csv (e.g. during pipeline rebuild)
uv run python -m fuel_signal.calibrate --skip-results-csv

# Custom model artifact (e.g. logreg)
uv run python -m fuel_signal.calibrate \
    --model-in data/models/logreg.joblib \
    --model-out data/models/logreg_calibrated.joblib \
    --model-name logreg
```

Reports class balance (BUY rate) for all splits and prints a 10-bin reliability table on val. If miscalibrated (max |gap| > 0.05), compares sigmoid (Platt) vs isotonic calibration wrappers and saves the better one. Calibration uses `sklearn.base.clone` of the input model, so it works generically for any sklearn-compatible estimator. Appends a result row to `experiments/results.csv`.

## Cost model diagnostics

Three commands ground the TP reward and FP/FN penalties used in `score_phase2.py` in empirical data.

### TP benefit

Measures how much cheaper label=1 days are compared to the subsequent `--horizon` days at the same station:

```bash
uv run python -m fuel_signal.tp_benefit
uv run python -m fuel_signal.tp_benefit --horizon 14 --plot data/tp_benefit_14d.png
```

### FP cost

Shows the actual damage of a false-positive BUY on label=0 days. The label=0 population is bimodal: cluster A (only the percentile gate failed — small damage) vs cluster B (a cheaper price was coming — larger damage):

```bash
uv run python -m fuel_signal.fp_cost
uv run python -m fuel_signal.fp_cost --features-csv data/features.csv --plot data/fp.png --threshold 3.0
```

### FN cost

Measures the cost of a false-negative WAIT on label=1 days — the price `--delay` days after a missed BUY opportunity:

```bash
uv run python -m fuel_signal.fn_cost
uv run python -m fuel_signal.fn_cost --delay 14 --plot data/fn_cost_14d.png
```

## Phase 2 final evaluation (lock the model)

Threshold sweep on val → pick τ → **score test once** → append to `experiments/results.csv`. Run this command once to lock Phase 2. Do not re-run to tune τ after seeing test results.

Also used for Phase 3+ models via `--model-path` and `--model-name`.

```bash
# Default: loads data/models/lgbm_calibrated.joblib, runs realised-spend backtest
# against ./fuel_signal.db, writes lgbm_cycle_features to results.csv
uv run python -m fuel_signal.score_phase2

# Custom artifact or name
uv run python -m fuel_signal.score_phase2 \
    --model-path data/models/lgbm_calibrated.joblib \
    --model-name lgbm_cycle_features

# Point the backtest at a non-default DB
uv run python -m fuel_signal.score_phase2 --db /path/to/fuel_signal.db

# Skip the backtest (e.g. quick CSV-only sniff-test)
uv run python -m fuel_signal.score_phase2 --no-backtest
```

**What it does:**

1. Loads the model artifact at `--model-path` (default: `data/models/lgbm_calibrated.joblib`).
2. Selects τ. **With `--model-path` (current default), threshold selection runs on walk-forward CV out-of-fold (OOF) predictions over train** (#236). OOF predictions sit at the training base rate (~0.24), so val's elevated BUY rate (~0.32) no longer biases the choice and **no τ adjustment is applied**. τ ∈ [0.05, 0.95] (step 0.05); τ = argmax(expected cents/row).
3. *(Legacy path, preserved for backward compat.)* Without the OOF path, τ is swept on val directly and a model-aware adjustment is applied for val's elevated BUY rate: **+0.05 for raw artifacts, 0.00 for isotonic-calibrated**. The fixed +0.05 was a workaround for the val BUY-rate bias and can cross an isotonic plateau — hence 0.00 for isotonic. Override either path with `--tau-adjustment <float>`.
4. Runs the realised-spend backtest at chosen τ over the test window using `--db` (default: `./fuel_signal.db`), populating `realised_spend_cpl` and `realised_savings_vs_always_buy_pct`. Skipped silently when the DB file or `--model-path` are absent (e.g. CI); use `--no-backtest` to skip explicitly.
5. Scores test at chosen τ. Appends one row to `experiments/results.csv` using `--model-name`.

**Cost model:** TP → +6.37c; FP → −5.80c; FN → −11.14c.

**Calibration warning:** If `--model-path` points at a raw (uncalibrated) artifact and `--tau-adjustment` is not passed explicitly, the CLI prints a `WARNING:` line surfacing the implicit `+0.05` default — the artifact filename (`lgbm_calibrated.joblib`) does not distinguish raw from isotonic, so this warning is the only visible signal that a raw model was loaded.

Locked τ and current-model results: see [docs/STATUS.md § ML Phase results](STATUS.md#ml-phase-results) — don't restate them here.

## Phase 2 τ re-validation on realised spend (Issue #64)

Sweeps τ ∈ [0.30, 0.70] on the test window via the backtest engine. Use `--no-patch` to dry-run; without it, patches `experiments/results.csv` with realised-spend columns for the Phase 2 and always-buy baseline rows.

```bash
# Dry-run: print sweep table only, do not patch results.csv
uv run python -m fuel_signal.backtest_phase2 \
    --model-path data/models/logreg_calibrated.joblib --no-patch

# Run and patch results.csv (default)
uv run python -m fuel_signal.backtest_phase2 \
    --model-path data/models/logreg_calibrated.joblib
```

**Phase 2 realised-spend result** (2026-05-10, preferred stations, test window 2025-07-01 → 2025-12-31, **7-day decision cadence** — the canonical cadence was re-locked to 1 day on 2026-08-22, so this table is not comparable to a figure produced after that date; see [CONVENTIONS.md](CONVENTIONS.md#the-decision-cadence-is-a-lock-parameter-declared--not-a-default)):

| Strategy | CPL (c/L) | vs always-buy |
|----------|-----------|---------------|
| Always-buy baseline | 191.78 | — |
| Logreg τ=0.40 (Phase 2) | 190.35 | +0.74% |
| Logreg τ=0.30 (spend-optimal) | 189.35 | +1.27% |

Spend-optimal τ=0.30 beats τ=0.40 by 1.01 c/L (≈0.5%). Phase 3 must beat 190.35 c/L.

## Phase 3a.1 LightGBM baseline (Issue #73)

Apples-to-apples LightGBM vs logreg — same 10 features, vanilla defaults, `random_state=42`.

```bash
# Full sequence: train → calibrate → score → backtest (all use Phase 4 defaults)
uv run python -m fuel_signal.train_lgbm
uv run python -m fuel_signal.calibrate --skip-results-csv
uv run python -m fuel_signal.score_phase2 --model-name lgbm_cycle_features
uv run python -m fuel_signal.backtest_phase2 \
    --model-path data/models/lgbm_calibrated.joblib --no-patch
```

**Phase 3a.1 result** (2026-05-14, real DB):

Calibration: LGBM is heavily over-confident out of the box (max |gap| = 0.38). Isotonic calibration chosen: val logloss 0.3926 (raw) → 0.3613 (isotonic), vs sigmoid 0.3904.

| Model | Val logloss | Test logloss | Test brier | vs baseline |
|---|---|---|---|---|
| Marginal-rate baseline | — | 0.5579 | 0.1855 | — |
| Logreg (Phase 2, τ=0.40) | 0.4112 | 0.4029 | 0.1346 | −0.155 logloss, −0.051 brier |
| LightGBM (Phase 3a.1, τ=0.65) | 0.3613 | 0.3444 | 0.1110 | −0.214 logloss, −0.074 brier |

LGBM val logloss beats raw-logreg val logloss (0.3926 < 0.4112) ✓

Realised-spend backtest (τ sweep, preferred stations, test window 2025-07-01 → 2025-12-31, **7-day decision cadence** — see the note on the Phase 2 table above):

| τ | CPL (c/L) | vs always-buy |
|---|---|---|
| 0.30 | 185.02 | +3.53% |
| 0.35 | 186.60 | +2.70% |
| 0.40 | 186.60 | +2.70% |
| 0.45 | 187.34 | +2.31% |
| 0.50 | 188.83 | +1.54% |
| 0.55 | 189.37 | +1.26% |
| 0.60 | 189.57 | +1.15% |
| **0.65** | **189.57** | **+1.15%** ← chosen |
| 0.70 | 189.69 | +1.09% |

LGBM τ=0.65: **189.57 c/L (+1.15% vs always-buy)**, beating Phase 2 logreg τ=0.40 (190.35, +0.74%) by **0.78 c/L**. Spend-optimal τ=0.30 (185.02 c/L); gap to chosen τ is 4.55 c/L — the val-based τ selection is conservative relative to the realised-spend optimum.

## Backtesting purchasing strategies

Replay a purchasing strategy over historical prices and compare realised spend against an always-buy baseline:

```bash
# All preferred stations, rule-based signal, 2023–2024
uv run python -m fuel_signal.backtest --preferred --strategy rule_based \
    --start 2023-01-01 --end 2024-12-31

# Single station, model strategy (requires fitted model)
uv run python -m fuel_signal.backtest \
    --station 414 --strategy model \
    --model-path data/models/logreg.joblib --threshold 0.40 \
    --start 2023-01-01 --end 2024-12-31

# Compare all strategies side-by-side (threshold defaults to 0.40)
uv run python -m fuel_signal.backtest --preferred --strategy all \
    --model-path data/models/logreg.joblib \
    --start 2023-01-01 --end 2024-12-31

# Custom tank size and consumption (default: 50L tank, 50L/14d, decide daily)
uv run python -m fuel_signal.backtest --preferred --strategy rule_based \
    --start 2023-01-01 --end 2024-12-31 \
    --tank-size 60 --daily-use 4.0 --eval-interval 7
```

Output is a table per station showing cents-per-litre (CPL), savings vs always-buy, fill events, and total litres for each strategy. Available strategies: `always_buy` (baseline, always included), `rule_based` (four-signal heuristic), `model` (logistic regression at `--threshold`), `all` (all three side-by-side).

`--tank-size`/`--daily-use`/`--eval-interval` are rejected with a `UsageError` if the combination could run the tank dry before the next evaluation — the CLI validates this up front with `validate_never_dry()` rather than silently producing a wrong CPL.

## AI-sourced feature pipeline (experiment batches)

`experiments/pipeline/` runs candidate feature columns through a two-arm (R0 vs candidate)
realised-CPL backtest, mostly unattended. Full design: bd `fps-3jj`; the machinery overview
is [docs/feature-pipeline.md](feature-pipeline.md); prompts for the two scheduled
pieces live in `docs/routines/{launch,dossier}.md`; candidate-filing rules in
`docs/routines/generator.md`.

**One-time per batch — freeze the data (owner-run, not scheduled):**

```bash
PYTHONPATH=. uv run python experiments/pipeline/batch_freeze.py <batch-name>
```

Hard-gates a full `make update` (pull, db, fill, classify, lga-leadership) and pins the result —
`data/features.parquet`, the live SQLite DB, and the resolved baseline feature-column list — into
`experiments/batches/<batch-name>/`, so every candidate in the batch runs against identical day-0
data. Aborts loudly on refresh failure rather than freeze stale data. Run this before filing any
candidate against `<batch-name>`.

**Per candidate — write the module and file the bd issue:**

1. Copy `experiments/candidates/TEMPLATE.py` to `experiments/candidates/<batch-name>/<NAME>.py` and
   fill in `NAME`, `HYPOTHESIS`, `PREDICTED_SIGNATURE`, `CONFIDENCE_EFFECT`, `CONFIDENCE_ZONE`,
   `TARGET`, `MECHANISM_FAMILY`, `PRIOR_ART`, `COLUMNS`, `INPUTS`, `add_columns` (and optional
   `add_axis`). Commit straight to `main` — `experiments/**` is exempt from the PR rule.
2. File the candidate bead:
   ```bash
   bd create --title "<NAME> candidate" --labels experiment --description "$(cat <<'EOF'
   HYPOTHESIS: ...
   TARGET: ...
   PREDICTED_SIGNATURE: ...
   CONFIDENCE_EFFECT: ...
   CONFIDENCE_ZONE: ...
   MECHANISM_FAMILY: ...
   PRIOR_ART: ...

   Batch: experiments/batches/<batch-name>
   Module: experiments/candidates/<batch-name>/<NAME>.py
   EOF
   )"
   bd dolt push
   ```
   The last two lines are machine-parsed by `launch.py` — line-anchored, exact text, no extra
   whitespace. The bead needs the `experiment` label so the chore/polish worker can't see it and
   the launch routine can.

**Running it:** the `fuel-price-signal-launch` scheduled task (nightly, ~9:00 PM local) claims the
oldest ready `experiment` bead, validates its candidate module (differential PIT test, restricted-
frame `INPUTS` check, NaN-rate assert), and launches the hours-long runner detached — one candidate
per night. To run immediately instead of waiting for the schedule:

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.launch
```

**Dossier — not yet a scheduled task.** Once a night's run finishes, a Claude session following
`docs/routines/dossier.md` turns it into a write-up: first the deterministic pass —

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables --scan experiments/candidates
```

— which writes `facts.json` + plots for every completed, undossiered run (no judgement calls), then
the session reads `facts.json`, grades the run against its own `PREDICTED_SIGNATURE`, and writes
`experiments/<batch>/<NAME>/README.md`, an `experiments/INDEX.md` row, and an `experiments/ledger.yaml`
entry by hand, per the routine doc. Only `fuel-price-signal-launch` is registered as a scheduled
task so far — this step needs to be invoked manually (or the scheduled task set up) until then.
