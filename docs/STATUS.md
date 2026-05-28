# Build status

Project-level state for agents picking up cold. Update this file when a phase completes or a module ships.

## All modules: built and tested

| Module | Status | Notes |
|--------|--------|-------|
| `config.py` | Done | API credentials, `PREFERRED_STATIONS`, `SYDNEY_METRO_POSTCODES` |
| `history.py` | Done | Bulk CSV downloader + transformer |
| `db.py` | Done | SQLite schema; upsert/load helpers; all read helpers |
| `fill.py` | Done | Forward-fill per-station gaps into `daily_prices` |
| `live.py` | Done | FuelCheck OAuth2; all-NSW all-fuel-type snapshots |
| `series.py` | Done | `resolve()`, `resolve_members()`, `enumerate_groups()`, `SeriesError` |
| `cycle.py` | Done | `CycleDetector`; `detect(as_of_date)` → `CycleState`; 21 unit tests |
| `signal.py` | Done | Four-signal port; `combine_signals`; 38 unit + integration tests |
| `compare.py` | Done | Station/LGA/brand/sydney series comparison |
| `inspect.py` | Done | Flask workbench; line/scatter/heatmap charts; URL-driven state |
| `stations.py` | Done | Station lookup CLI |
| `labels.py` | Done | Two-condition BUY label; `lookback_days`, `percentile_pct` params |
| `label_viz.py` | Done | Diagnostic plots for label distributions |
| `label_inspect.py` | Done | Per-station per-day label decomposition table |
| `features.py` | Done | Cycle features + station-vs-aggregate features → model-ready CSV |
| `evaluate.py` | Done | Canonical train/val/test split; `split()`, `log_loss()`, `brier()`, `reliability_table()`, `walk_forward_folds()` |
| `train_logreg.py` | Done | `StandardScaler → LogisticRegression` pipeline; saves `data/models/logreg.joblib` |
| `calibrate.py` | Done | Isotonic calibration wins; saves `data/models/logreg_calibrated.joblib` |
| `score_phase2.py` | Done | Threshold sweep on val → τ=0.40 → test scored once → results.csv |
| `tp_benefit.py` | Done | Empirical TP benefit distribution |
| `fp_cost.py` | Done | Empirical FP cost distribution (bimodal) |
| `fn_cost.py` | Done | Empirical FN cost distribution |
| `backtest.py` | Done | `AlwaysBuyStrategy`, `RuleBasedSignalStrategy`, `ModelStrategy`; CPL table per station |
| `.github/workflows/daily-snapshot.yml` | Done | Daily cron + workflow_dispatch; confirmed working |

## Canonical train/val/test split (fixed — do not adjust after results are in)

| Split | Start | End |
|-------|-------|-----|
| Train | 2016-08-01 | 2025-03-17 |
| Val | 2025-03-25 | 2025-06-23 |
| Test | 2025-07-01 | 2025-12-31 |

7-day buffers between splits prevent label leakage.

## ML Phase results

### Phase 2 locked (2026-05-09)

Logistic regression on cycle features only. τ=0.40 chosen via val sweep + 0.05 adjustment for val/test BUY-rate gap.

| Model | Test logloss | Test brier | vs baseline |
|-------|-------------|------------|-------------|
| Marginal-rate baseline | 0.5821 | 0.1966 | — |
| Logreg cycle features (τ=0.40) | 0.4029 | 0.1346 | −0.179 logloss, −0.062 brier |

At τ=0.40: P=0.618, R=0.581, F1=0.599, BUY%=25% on test.

**Realised spend (preferred stations, test window 2025-07-01 → 2025-12-31):**

| Strategy | CPL (c/L) | vs always-buy |
|----------|-----------|---------------|
| Always-buy baseline | 191.78 | — |
| Logreg τ=0.40 (Phase 2 locked) | 190.35 | +0.74% |

### Phase 3 progression — locks not narrated here (see `experiments/results.csv` + commit history)

Phase 3a (10-feat LGBM cycle features) → Phase 3b (14-feat, +LGA/brand aggregates) → Phase 3c (15-feat, +stickiness_score). Each lock's row sits in `experiments/results.csv`; design notes are in the `project_*` memory files.

### Phase 4 locked (2026-05-25)

50-feature LightGBM: 15 base (Phase 3c set) + 35 `days_since_trough_entry_<lga>` event-based leadership features. Raw (uncalibrated) selected over isotonic — the 35 LGA features absorbed enough of the calibration slack that the 80%-train handicap of `compare_calibrations` dominated; isotonic regressed +0.006 vs raw. τ=0.60 (val argmax) with model-aware +0.05 adjustment since raw is not isotonic-calibrated.

| Model | Test logloss | Test brier | Test F1 | vs Phase 3c logloss |
|-------|-------------|------------|---------|---------------------|
| Phase 3c (15-feat isotonic, τ=0.60) | 0.3395 | 0.1115 | 0.749 | — |
| **Phase 4 (50-feat raw, τ=0.60+0.05)** | **0.3012** | **0.0973** | **0.769** | **−0.0383 (−11.3%)** |

At τ=0.65 test: P=0.702, R=0.851, F1=0.769, BUY%=29.9%.

#### Re-locked 2026-05-29 (post-#144 boundary-postcode fix)

The #144 from-scratch DB rebuild corrected postcode→LGA boundaries (which feed every `days_since_trough_entry_<lga>` feature) but its retrain silently dropped the LGA features (`--include-lga-features` defaults off), leaving a 15-feat model on disk. Retrained 50-feat on the corrected data and re-locked. The boundary fix *improved* the model:

| Model | Test logloss | Test brier | Test F1 |
|-------|-------------|------------|---------|
| Phase 4 pre-fix (buggy LGA boundaries, single seed) | 0.3012 | 0.0973 | 0.769 |
| **Phase 4 post-fix (corrected boundaries)** | **0.2854** | **0.0914** | **0.767** |

Calibrate again chose raw over isotonic. Seed-banked (raw uncalibrated test logloss, seeds {1,7,42,99,2024}): **mean 0.2919, std 0.0053 (3σ=0.0158)** — vs `lgbm_council_fix` raw mean 0.3205, a leadership lift of −0.0286, beyond the 3σ band of both. See `experiments/results.csv` row `phase4_event_leadership_postfix` and the throwaway `experiments/seed_bank_phase4/run.py` (superseded once #145 lands a real `--seeds` flag).

**Not re-run post-fix:** the three validation gates below were computed pre-fix. The from-scratch rebuild also left `lga_leadership` empty (0 rows), so `inspect.py`'s board and the SHAP cross-reference can't run until it's repopulated. Re-validation is a follow-up, not done in the re-lock.

**Validation (all three gates passed — pre-#144-fix numbers):**

- `experiments/trough_weakness/` — target cohort: `lead −7..−4` 0.522 → 0.4794 (Δ −0.043), `lead −3..−1` 0.630 → 0.6096 (Δ −0.020). Acceptance ≥0.02 in either, both hit.
- `experiments/cv_compare_phase4/` — paired walk-forward CV vs Phase 3c: 13 of 14 folds improve; median Δ −0.029, mean Δ −0.081. **Fold 5 (2022-10 → 2023-01, Ukraine spike)** — the known stickiness regime-lag tail risk — inverted from Phase 3c's +0.353 regression to a −0.486 improvement (50-feat 0.295 vs 15-feat 0.781). Event-based trough features don't carry the 45-day rolling lag that hurt Phase 3c.
- `experiments/shap_phase4/` — 6 zero-station-floor LGAs have SHAP exactly 0 ✓. 8 of 35 LGA features make the overall top-25 (woollahra #10, randwick #11, blue_mountains #13). LGA mean|SHAP| is materially higher in trough-adjacent cohorts than mid-cycle (woollahra 0.18–0.19 in lead/trough vs 0.11 mid-cycle) — exactly the event-locked behaviour the design predicted.

**Open follow-ups from validation:**

- **#136 (design) — RESOLVED 2026-05-28.** SHAP vs `trough_lead_consistency` disagreement investigated (`experiments/shap_phase4/nan_analysis.py`, `dependence_grid.py`, `lga_shap_plots.py`, `lga_dependence_interaction.py`). Findings: (1) NaN-routing artefact ruled out — val window is 0% NaN on the LGA features (the 5.59% woollahra figure is full-CSV cold-start only). (2) Three empirical roles by sign of Pearson r(feature, SHAP): **leaders** (woollahra r=−0.42, randwick r=−0.91 → recent trough toward BUY), **inverted-phase** (blue_mountains r=+0.79, parramatta r=+0.67 → recent trough toward WAIT; LGA leads the rise), **inert** (the four highest-consistency LGAs — sutherland/northern_beaches/penrith/ku_ring_gai — all |SHAP|<0.013, signal already in stickiness_score + station_minus_last_min). **Decision:** model unchanged (per-LGA features, no weighting — validated); `trough_lead_consistency` demoted to a descriptive stat (information value ≠ price-space consistency); future `inspect.py /leadership` view ranks by mean|SHAP| + sign, not consistency.
- Camden missing-data chore — outer-metro Sydney LGA with real petrol stations but 0 stations in our DB; trace upstream feed.
- v2 peak features — `lead −8+` and `lead −7..−4` cohorts remain miscalibrated by +0.12 and +0.22 (over-confident BUY). Symmetric `days_since_peak_entry_<lga>` design worth scoping next; expect the same leader/inverted/inert split — design symmetrically, do not curate by consistency.

## Pending work

### Phase 5 (macro model)
- Separate longer-horizon model (~30/60/90 days)
- Upstream commodity features dominate at this horizon
- Upstream features (TGP first, then MOPS/crude/FX)

## Model artifact paths (IMPORTANT)

Models are written to fixed canonical paths. **Each Phase lock overwrites the previous Phase's artifact** — there is no per-phase suffix on the filename. Phase identification lives in `experiments/results.csv` (`name` column) and in commit history, not the filename.

| Path | Writer | Currently (as of Phase 4 re-lock, 2026-05-29) |
|------|--------|--------------------------------------------|
| `data/models/lgbm.joblib` | `train_lgbm.py` | Phase 4 50-feat raw (post-#144 boundary fix) |
| `data/models/lgbm_calibrated.joblib` | `calibrate.py` | Phase 4 50-feat (raw chosen over isotonic), post-#144 |
| `data/models/logreg.joblib` | `train_logreg.py` | Phase 2 10-feat raw |
| `data/models/logreg_calibrated.joblib` | `calibrate.py` | Phase 2 10-feat (raw chosen over calibration) |

To verify what's currently on disk before scoring:

```python
import joblib
m = joblib.load("data/models/lgbm_calibrated.joblib")
print(len(m["feature_columns"]), m["feature_columns"][-1])
# 50 + ending in 'days_since_trough_entry_woollahra' → Phase 4
# 15 + ending in 'stickiness_score' → Phase 3c
# 14 + ending in 'station_minus_brand_mean_cents' → Phase 3b
# 10 + ending in 'station_minus_sydney_avg_cents' → Phase 3a
```

Reproducing an older lock requires `git checkout <commit>` followed by re-running train + calibrate. The joblib binaries are gitignored. Phase 3c reproducible from pre-Phase-4-lock commits; Phase 3b from pre-#127 (`0ed9795`).

Worker-run logs may reference different filenames (e.g. `lgbm_stickiness_calibrated.joblib`); those are worktree-local aliases and were never persisted on `main`.

## Key architectural notes

- `station_code` primary key comes only from FuelCheck API live snapshot — `stations` table cannot be populated from historical CSVs alone.
- `stations.latitude/longitude` always NULL (API returns them but snapshot CSV doesn't include them).
- Storage format: `price_date INTEGER YYYYMMDD` (e.g. 20240101), `price_decicents INTEGER` (e.g. 1619 = 161.9c). Conversion is transparent at db.py boundary.
- `daily_average_e10` queries raw prices (not gap-filled); `sydney_average_series` queries `daily_prices` (gap-filled). Use `sydney_average_series` for cycle detection.
- Stations only upserted when they have ≥1 matching price row — prevents EV chargers and non-petrol venues from causing duplicate normalised-address collisions.
