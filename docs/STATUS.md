# Build status

Project-level state for agents picking up cold. Update this file when a phase completes or a module ships.

> **Canonical source for current model state** (feature count, on-disk artifact, calibration, τ, active phase). Other docs link here — don't restate these facts elsewhere. See [CONVENTIONS.md § One source of truth](CONVENTIONS.md#one-source-of-truth-for-current-model-state).

## All modules: built and tested

| Module | Status | Notes |
|--------|--------|-------|
| `config.py` | Done | API credentials, `PREFERRED_STATIONS`, `SYDNEY_METRO_POSTCODES` |
| `history.py` | Done | Bulk CSV downloader + transformer |
| `db.py` | Done | SQLite schema; upsert/load helpers; all read helpers |
| `fill.py` | Done | Forward-fill per-station gaps into `daily_prices` |
| `live.py` | Done | FuelCheck OAuth2; all-NSW all-fuel-type snapshots |
| `tgp.py` | Partial | AIP Sydney TGP downloader → `data/tgp/tgp_sydney.csv`; daily `tgp-fetch.yml` action. `tgp_delta_7d` is computed into every features.csv row (chips 1–3 of #271 done) but deliberately held OUT of `FEATURE_COLUMNS`/the trained model. The AI-sourced pipeline's batch0 run reached a clean result on 2026-08-19 after two contract defects were fixed (`fps-sa1`/PR #309, wrong column set; `fps-zci`/PR #310, wrong column order): pooled realised delta_cpl_held **+0.0059 c/L — inert** (batch0 is a 7-day-cadence batch; every CPL, delta and noise-band figure in this row is a 7d number, and the flip quantum below is a 7d quantity — see the cadence note under Current production model), neither the June graduation (−0.039) nor the rejection that run's earlier attempts reported. All of these sit within a couple of decision flips of zero (one flip ≈ 0.05 c/L pooled), so the realised arbiter cannot resolve this effect — June's −0.039 was itself smaller than one flip, and does not reproduce. The noise floor that was the unblocking step has since been **run and applied** (`fps-3jj.9`; `experiments/pipeline/noise_floor.py`): +0.0059 c/L sits at the **40th percentile** of pure fit noise (band mean −0.0135, std 0.142 c/L over 5 draws), z=+0.14 — indistinguishable from noise. On that reading **`#271` is CLOSED as superseded** (2026-08-20): the `FEATURE_COLUMNS` bump and `decide()` wiring will not happen, and **no TGP feature is in the lock** — `tgp_delta_7d` is registered in `features.NON_MODEL_COLUMNS` as `evaluated-inconclusive` (measured below the arbiter's resolution, so neither graduated nor dead ground). The track stays open under a different question: successor bd `fps-x0f` asks whether *any* TGP expression earns a place, the live claim being the **gap** (r²=0.63 vs depth-remaining, never tested on late descent) rather than the velocity. Since `fps-cf8` the floor is computed by `batch_freeze` itself and is fingerprint-gated, so a re-lock invalidates it. Full detail: `experiments/candidates/batch0/tgp_delta_7d/README.md`, bd `fps-nor`. |
| `series.py` | Done | `resolve()`, `resolve_members()`, `enumerate_groups()`, `SeriesError` |
| `cycle.py` | Done | `CycleDetector`; `detect(as_of_date)` → `CycleState`; sticky `find_peaks` confirmation (#250, no boundary whipsaw); 26 unit tests |
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
| `backtest_phase2.py` | Done | Phase 2 τ re-validation on realised spend |
| `train_lgbm.py` | Done | LightGBM trainer; `--no-brand-features` for the locked 54-feat baseline |
| `classify.py` | Done | Competitive/Discount/Sticky station classifier → `station_class` table |
| `score_phase2.py` | Done | Threshold sweep on val → score test → append `results.csv`; multi-seed support |
| `cv_report.py` | Done | Paired walk-forward CV for feature add/drop/swap decisions |
| `lga_leadership.py` | Done | Phase 4 LGA event-based leadership features |
| `brand_leadership.py` | Done | Brand trough features (computed; not in locked model — Phase 4b walked away) |
| `feature_redundancy.py` | Done | SHAP-redundancy cluster analysis |
| `feature_diagnostics.py` | Done | Feature-level diagnostic utilities |
| `shap_report.py` | Done | SHAP importance + per-prediction explanation |
| `loo_ablation.py` | Done | Leave-one-out feature ablation |
| `postcode_council.py` | Done | Postcode → LGA mapping; `SYDNEY_METRO_COUNCILS` |
| `.github/workflows/daily-snapshot.yml` | Done | Daily cron + workflow_dispatch; confirmed working |

## Canonical train/val/test split (fixed — do not adjust after results are in)

| Split | Start | End |
|-------|-------|-----|
| Train | 2016-08-01 | 2025-03-17 |
| Val | 2025-03-25 | 2025-06-23 |
| Test | 2025-07-01 | 2025-12-31 |

7-day buffers between splits prevent label leakage.

## ML Phase results

### Current production model (54-feat baseline)

On-disk artifact: **54-feat LightGBM, isotonic-calibrated, τ=0.25.** Last feature column `lga_phase_std_delta_3d`.

**Decision cadence: 1 day** (`TankParams.evaluation_interval_days`, stamp
`50/3.571/1d/10%`) — re-locked from 7 days on 2026-08-22, bd `fps-oqz`. Cadence is a
declared lock parameter, not a knob: see
[CONVENTIONS.md § The decision cadence is a lock parameter](CONVENTIONS.md#the-decision-cadence-is-a-lock-parameter-declared--not-a-default)
for the rationale and the full before/after. Two consequences for reading this file:

- **The re-lock is not a retrain and not a τ change.** The on-disk artifact and τ=0.25
  are unchanged — the fit takes no `TankParams`. Nor is it a production code change:
  `evaluation_interval_days` is consumed only by the backtest engines, and the live
  daily signal (`fuel_signal/signal.py`) is rule-based and already emits daily.
- **Every realised-CPL figure in the historical tables below was measured at 7 days**
  and has not been restated. They remain correct as lock records; they are not
  comparable to a number produced after 2026-08-22 without re-running at 1d. On the
  cadence-ceiling folds the same model reads 189.67 at 7d and 187.82 at 1d, and
  headroom vs the perfect-foresight oracle goes 1.54 → 2.97 c/L.

- **#216** (2026-06-09) — graduated the RAC_full network group (4 cols: `network_px_std`, `network_px_std_delta_3d`, `lga_phase_std`, `lga_phase_std_delta_3d`), retraining the 50-feat Phase 4 baseline to 54. Δh25 −0.045 over LGA-only. See [AGENTS.md § Canonical feature set](../AGENTS.md#canonical-feature-set-54-feat-baseline-locked-issue-216).
- **#236** (2026-06-13, commit 740b601) — calibration + threshold selection moved to OOF CV over train with an 80/20 eval split; isotonic chosen over raw; τ=0.25. Realised backtest 3.04% (185.94 c/L) vs always-buy 191.78 at the time (prior raw/τ=0.55 was 1.98%).
- **#250 cycle-fix rebuild** (2026-06-15, commit 7bee0e8) — re-cut the same 54-feat pipeline on the merged #250 whipsaw fix (sticky causal `find_peaks`), changing two cycle inputs (`cycle_days_since_peak`, `cycle_pct_through`). Isotonic re-chosen on OOF (0.3036 < raw 0.3051), **τ held at 0.25**. Realised backtest moved forward to **3.37% (185.32 c/L)** vs always-buy 191.78; test logloss flat (0.2629→0.2626). This rebuild is what's on disk now. Single window/seed; a multi-fold paired CV of realised CPL was scoped but not run.

The phase-by-phase tables below are the historical lock record; the three items above are the current state.

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

#### Phase 4 50-feat state (post-#144 boundary fix + post-#154 Central Coast removal — superseded by the 54-feat #216 lock above)

Model at this lock: 50-feat raw LGBM, banked as `phase4_5seed_lock_post_cc_removal` (commit 9afdd2f).

| Metric | Value |
|---|--:|
| Test logloss (seed 42) | 0.2676 |
| Test brier | 0.0855 |
| Test F1 | 0.768 |
| 5-seed test logloss mean / std | 0.2666 / 0.0024 |

Calibrate chose raw over isotonic.

**Validation gates re-run 2026-05-30 (issue #156):**

- `experiments/trough_weakness/` — overall test logloss 0.2676 (matches the seed-42 bank). Lead-cohort losses: `lead −7..−4` **0.3904** (pre-fix 0.4794), `lead −3..−1` **0.5749** (pre-fix 0.6096). Both substantially lower than the pre-fix Phase 4 figures at which the ≥0.02 acceptance bar already passed. `lead −7..−4` remains the worst-calibrated cohort (mean p 0.333 vs rate 0.175, +0.157 over-prediction) — the residual structure #157 is filed against.
- `experiments/cv_compare_phase4/` — paired walk-forward CV vs Phase 3c, 14 folds, seed=42: **11/14 folds improve** (was 13/14 pre-fix); median Δ **−0.032**, mean Δ **−0.042**, std 0.064. **The "fold 5 Ukraine-spike rescue" story does not survive the fix** — post-fix Phase 3c handles fold 5 normally (ll_15 = 0.263) and Phase 4 regresses mildly (Δ +0.036). The pre-fix Phase 3c collapse (ll_15 = 0.781) was a buggy postcode→LGA boundary artifact feeding `lga_mean_cents` and `station_minus_lga_mean_cents`, not the stickiness-regime-lag phenomenon previously claimed. Other minor regressions: fold 6 (+0.019), fold 11 (+0.062).
- `experiments/shap_phase4/` — 8 of 35 LGA features in overall top 25 (woollahra #10, blue_mountains #12, randwick #13, burwood #14, cumberland #15, parramatta #17, mosman #18, hawkesbury #20). Cohort taxonomy intact: woollahra mean|SHAP| 0.18 in lead/trough cohorts vs 0.11 mid-cycle. 5 of 6 historically-zero LGAs still zero; **Camden now non-zero** (mean|SHAP|=0.029) — boundary fix gave it 14 stations, retiring #138.

**Phase 4b evaluated and walked away (2026-06-02).** 60-feat schema (Phase 4 + 10 `days_since_trough_entry_<brand>`) hit 5-seed bank Δ −0.0062 (borderline) but lost 9/14 folds in paired walk-forward CV with a non-shock regression at fold 11 (+0.0297). Brand-trough feature code (PRs #183, #184) stays merged; columns continue to be computed in `features.csv` but are not used at the model level. Phase 4 (`phase4_5seed_lock_post_cc_removal`) remains operational baseline. Ledger: `phase4b_cv_negative` row in `experiments/results.csv`; per-fold artifacts at `experiments/cv_compare_phase4b/results.csv` (PR #187, close-not-merge). Compute-features per-row brand-col gap (#185) closed as moot.

**Open follow-ups:**

- **#157** (open, under consideration) — peak-side v2 features for the `lead −7..−4` over-prediction (+0.157), the dominant residual miscalibration. Design taxonomy: `docs/PLAN_phase4_event_leadership.md` § LGA feature roles in SHAP.

The late-descent / extended-shallow-descent investigation that drove the 50→54-feat work is closed: #212 (RAC_full graduated → #216 lock), #219→#221 (canonical cohort), #214 rejected, #215 and #231 (interaction-column probe) closed. See those issues for the external-data decision. **Reopened 2026-08-19 as bd `fps-x0f`** — that chain's log-loss-based rejections stand, but the *economic* closure argument does not: #262's "regime axis FLAT" was withdrawn 2026-08-20 and the 2026-06-18 gate-1 per-regime saving% 2026-08-21, both as path-coupled costs allocated to a sub-period (`experiments/2026-08-21_path_coupling_audit/`, docs/CONVENTIONS.md § *Bucketed results*). The one surviving reading on that axis is gate-1's per-row proxy regret, which says late descent is the **worst** zone.

## Pending work

### Phase 5 (macro model)
- Separate longer-horizon model (~30/60/90 days)
- Upstream commodity features dominate at this horizon
- Upstream features (TGP first, then MOPS/crude/FX)

## Model artifact paths (IMPORTANT)

Models are written to fixed canonical paths. **Each Phase lock overwrites the previous Phase's artifact** — there is no per-phase suffix on the filename. Phase identification lives in `experiments/results.csv` (`name` column) and in commit history, not the filename.

| Path | Writer | Currently (as of #250 cycle-fix rebuild, 2026-06-15) |
|------|--------|--------------------------------------------|
| `data/models/lgbm.joblib` | `train_lgbm.py` | 54-feat baseline (#216 feature set, re-cut on the #250 cycle fix) |
| `data/models/lgbm_calibrated.joblib` | `calibrate.py` | 54-feat, isotonic-calibrated, τ=0.25 (#250 rebuild; calibration design from #236) |
| `data/models/logreg.joblib` | `train_logreg.py` | Phase 2 10-feat raw |
| `data/models/logreg_calibrated.joblib` | `calibrate.py` | Phase 2 10-feat (raw chosen over calibration) |

To verify what's currently on disk before scoring:

```python
import joblib
m = joblib.load("data/models/lgbm_calibrated.joblib")
print(len(m["feature_columns"]), m["feature_columns"][-1])
# 54 + ending in 'lga_phase_std_delta_3d' → 54-feat baseline (#216 RAC_full, current)
# ~60 + ending in a 'days_since_trough_entry_<brand_slug>' (e.g. 'speedway') → Phase 4b
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
