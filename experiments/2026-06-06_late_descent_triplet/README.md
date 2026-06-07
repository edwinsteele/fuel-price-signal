# Late-descent triplet: dispersion / sticky-floor / leader divergence

**Issue:** #206 (late-descent / trough-proximity feature investigation)

**Date opened:** 2026-06-06

## Hypothesis

The model's residual error budget is concentrated in the late-descent /
trough-proximity zone, and the failure mode is **regime ambiguity**: the model
cannot distinguish

- **Normal late descent** — 5 days from a sharp coordinated trough — from
- **Extended descent / gradual decline** — the longer descending arm of an
  elongated cycle, which looks locally identical but where the trough is days
  or weeks further out.

Predecessor evidence (2026-06-05 phase-lookup retest) showed that single-scalar
phase encodings can't separate these regimes because they collapse
"how-far-through" with "how-noisy-the-tail-is". The siblings (`days_since_peak`,
`cycle_mean_length`, etc.) carry information that the scalar discards.

**Question:** is there *any* information in the dataset that separates these
two regimes — i.e. can we tell, from prices alone, that the network is *not*
coordinating for an imminent trough? If the answer is no, external data
(wholesale, AUD/USD, macro) is justified as the next step.

## Candidate information sources

Three independent mechanisms, each motivated by a different aspect of "how
networks coordinate around troughs":

1. **Cross-station dispersion compression.** Into a normal trough, competitive
   stations *converge* on a floor; cross-sectional std/IQR collapses sharply
   in the last 2–3 days. In extended descent, dispersion stays elevated
   because nobody's racing to a coordinated bottom.

2. **Sticky-floor reference gap.** Sticky stations barely move and encode the
   slow cost baseline. The gap `competitive_min − sticky_median` should narrow
   into a normal trough and *stay flat or widen* in gradual decline (because
   the floor itself is sliding down with the stickies).

3. **Leader-phase divergence.** The `lga_leadership` architecture currently
   encodes consensus. The complement — *disagreement* between LGAs in
   `days_since_trough_entry_<lga>` — distinguishes coordinated late descent
   (small spread) from regime drift (large spread, leaders out of phase).

Discarded by user from the original brainstorm:

- **DoW × phase interaction** (Tuesday) — discarded by user observation; no
  Tuesday effect observed locally.
- **Cycle-shape persistence (#4)** — included as side probe in Step 1; only
  graduates to Step 2 if Step 1 shows meaningful autocorrelation.
- **Drop-arrival topology (#5)** — reframed by user: "are big drops
  distinguishable from regular downward trends?" → bimodality test on the
  drop-size distribution, included as side probe in Step 1.

## Design

### Step 1 — Information probe (pre-design evidence)

`step1_information_probe.py` (no model training; runs on `data/features.csv`
alone).

For each candidate signal A, B, C, compute the signal across the full dataset
and report its distribution conditional on a **phase proxy**:

- **Normal late descent**: `cycle_pct_through ∈ [0.30, 0.50]` (descending arm
  leading into the empirical trough at pct≈0.5; per memory
  `project_cycle_pct_through_semantics` the empirical shape is
  peak→trough→peak, non-monotonic, with the trough near pct=0.5).
- **Extended descent**: `cycle_pct_through > 0.90` (cycle has run past its
  expected length; elongation regime).

Decision rule per signal: clear separation if the two distributions differ by
≥ 0.3σ in the relevant direction. Standardised mean difference reported.

Side probes:

- **#4 cycle-length autocorrelation** (per-station lag-1 and lag-2; and
  conditional on recent-3-cycle median).
- **#5 drop-size separability** (histogram + percentiles of daily down-moves;
  conditional on days-to-next-trough).
- **DoW sanity** for Tuesday troughs.

Step 1 acceptance: at least one of A / B / C shows ≥ 0.3σ separation →
graduate to Step 2. Otherwise: write up the negative result and close the
late-descent track in favour of external data sources.

### Step 1b/1c — Coverage diagnostics for Signal B

`step1b_sticky_coverage.py` revealed Signal B's original "sticky-floor" framing
is unworkable at the canonical 10c premium threshold: only 2 sticky stations
across all of Sydney, 30/32 LGAs have zero. Triggered issue #207 to review the
threshold + redefine sticky as temporal-stability rather than premium.

`step1c_discount_coverage.py` pivoted Signal B to **competitive-vs-discount
gap**: per-date row-level classification (`stickiness_score < -5c` for
discount; `|score| ≤ 5c` for competitive). 345 distinct stations have
populated the discount cohort dynamically across the timeline (vs only 6 at
canonical -10c snapshot). Divergence signal:
`SMD(extended−normal) = +0.40` (threshold-robust across −10/−5/−2).
Pivoted-B is the version carried into Step 2.

### Step 1d — Direction validation for Signal C

`step1d_lga_divergence_direction_split.py` answered an in-session pushback:
"the high lga_phase_std in extended descent could just be ascent leader-lag
in rows where the next peak hasn't been detected yet." Split extended-descent
rows by station-level 5d backward price change.

Result: when prices are **rising**, extended/normal look identical (~6d each)
— that part of C's signal IS ascent leader-lag in disguise. But when prices
are **falling**, extended descent has `lga_phase_std ≈ 10.4d` vs normal
`≈ 4.7d` (2.2× spread). C measures genuine descent-coordination-breakdown,
not just ascent leader-lag — confirmed on rows where prices are actually
falling. Feature design (level + Δ-3d) lets the model compose with existing
price-direction features rather than pre-baking the direction split.

### Step 2 — Feature implementation + paired WFCV with attribution grid

Implement A + pivoted-B + C as features (additive on top of 50-feat Phase 4
baseline). Six new columns:

- A: `network_px_std`, `network_px_std_delta_3d`
- B: `network_disc_gap`, `network_disc_gap_delta_3d`
- C: `lga_phase_std`, `lga_phase_std_delta_3d`

Thresholds (provisional, pending #207):
- A competitive cohort: `stickiness_score < 5c` (excludes top ~25 high-premium
  stations; canonical 10c would exclude only 2, making the cohort trivially
  network-wide).
- B discount cohort: row-level `stickiness_score < -5c`; competitive
  `|stickiness_score| ≤ 5c`.

**8-run attribution grid** (`step2_paired_wfcv.py`):

| Run | Features added to baseline | Question answered |
|---|---|---|
| R0 baseline | none (50) | reference |
| R1 ABC | A + B + C (6) | full-triplet lift |
| R2 drop A | B + C (4) | marginal contribution of A given B,C |
| R3 drop B | A + C (4) | marginal contribution of B given A,C |
| R4 drop C | A + B (4) | marginal contribution of C given A,B |
| R5 A only | A (2) | standalone contribution of A |
| R6 B only | B (2) | standalone contribution of B |
| R7 C only | C (2) | standalone contribution of C |

Comparison structure: gap between (standalone) and (marginal-given-others)
per family = redundancy with the other families. Walk-forward CV folds and
shock-fold taxonomy mirror the predecessor experiment. **5 seeds** per fold
per run per `feedback_seed_discipline` since expected lifts are in the
0.005–0.02 range (close to seed noise) — single-seed CV is too noisy to
separate small marginal contributions cleanly.

Empirical labelling per #206: top-quartile baseline per-row log-loss per fold
defines the "hard" cohort. Δ reported overall AND on hard rows. Shock-fold
taxonomy from predecessor (folds {1, 4, 9, 13}) kept as supplementary cut.

Acceptance bar (per family, for Step 3 inclusion):
- standalone (R5/R6/R7 vs R0) shows ≥ 0.005 logloss reduction on hard rows; AND
- marginal-given-others (R1 vs R2/R3/R4) shows ≥ 0.005 reduction; AND
- no fold blows up by more than the lift gives back.

Total compute: 8 runs × ~14 folds × 5 seeds ≈ 560 LightGBM fits. Wall-time
instrumented per fit per `feedback_instrument_walltime`.

## Methodology constraints (from #206 and memory)

- `from fuel_signal.features import load_features`; no raw `pd.read_csv` of
  `features.csv` (per `feedback_load_features_helper`).
- `PYTHONPATH=.` prefix for invocation (per
  `feedback_experiment_scripts_pythonpath`).
- LightGBM fit + predict both with DataFrames (per
  `feedback_lgbm_dataframe_consistency`).
- Wall-time instrumentation per stage (per `feedback_instrument_walltime`).
- Paired WFCV per `feedback_seed_discipline` and
  `feedback_regime_segmented_evaluation`.

## Outcome

**Step 1:** at least one of A/B/C cleared the 0.3σ separation bar; Signal B
required pivoting to discount-cohort framing (Step 1c) and Signal C required
direction validation (Step 1d). Triplet graduated to Step 2.

**Step 2 (2026-06-06):** triplet delivers a modest but real lift. Full
analysis in `step2_analysis.md`.

⚠ **Correction note:** initial verdict (logged in commit history) was
"triplet does not deliver" based on mean-across-seeds aggregation. One
of 560 cells (fold 2, R1_ABC, seed 44) produced a LightGBM fit failure
(ll_all = 4.73 vs ~0.24 for the other 4 seeds; features clean, attributable
to an unlucky bagging RNG draw). That single cell dominated the mean.
Median-across-seeds aggregation reflects the true headline.

Median-aggregated headline:

- R1_ABC: hard25 -0.043, hard10 -0.097, lated -0.016, all -0.006 (all
  helping or neutral).
- **A** (`network_px_std`, +Δ): standalone hard25 +0.030, marginal +0.049;
  hard10 +0.068 / +0.069; lated +0.017 / +0.033. Strong, pulling real
  weight in combo. **Graduate** subject to within-family ablation.
- **B** (`network_disc_gap`, +Δ): standalone -0.005, marginal -0.002 on
  hard25. Flat, not pulling weight, but NOT actively harmful (the
  catastrophic numbers in the initial analysis were s44-poisoned).
  **Drop for parsimony.**
- **C** (`lga_phase_std`, +Δ): standalone +0.014, marginal +0.010 on
  hard25; standalone +0.007, marginal +0.018 on lated. Modestly positive
  in combo. SHAP-corr with A's Δ ≥ 0.5 in 6/14 folds is real, but
  predictive redundancy is not established. **Include in the #212
  ablation alongside A; don't drop pre-emptively.**

**Step 3 decision:**

1. Drop B.
2. Run within-family ablation (issue #212) — extend to test
   A-only / A+Δ / A+C / A+C+Δ subsets, not just A's level vs Δ.
3. If a useful subset graduates, land in `fuel_signal/features.py` and
   retrain the 50→51 or 52-feat baseline.
4. Late-descent intra-series track is NOT exhausted; external-data
   move per `project_late_descent_investigation` (issue #211) is
   contingent on #212 outcome, not a foregone conclusion.

Methodological lesson logged in memory:
`feedback_check_seed_variance_before_trusting_mean` — scan per-cell
seed_std before reading mean-aggregated multi-seed CV headlines.
