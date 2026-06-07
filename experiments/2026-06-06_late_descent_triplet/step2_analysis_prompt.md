# Fresh-session prompt: analyse Step 2 results

Copy everything below the line into a fresh Claude Code session at the
repository root. Don't include this header.

---

I just finished running an experiment. I'd like you to analyse the results and
recommend what happens next. The full experimental design and mechanism story
is in the experiment README; read that first.

## Files to read

In order:

1. `experiments/2026-06-06_late_descent_triplet/README.md` — full design,
   hypothesis, mechanism notes for each signal family (A / B / C), acceptance
   bar. Read **all of it** before opening the result files.
2. `experiments/2026-06-06_late_descent_triplet/step2_meta.json` — top-level
   numerical summary: per-run aggregates, per-cohort attribution tables,
   wall-time totals.
3. `experiments/2026-06-06_late_descent_triplet/step2_fold_run.csv` — per
   (fold, run) seed-averaged results with deltas-vs-baseline already computed.
   Use this for per-fold blowup checks.
4. `experiments/2026-06-06_late_descent_triplet/step2_runs.csv` — raw per
   (fold, run, seed) rows. Use only if you need to look at seed variance
   for a specific cell.
5. `experiments/2026-06-06_late_descent_triplet/step2_shap_mean_abs.csv` —
   per-fold mean |SHAP| for the 6 new features, R1 model (seed 42 only).
   Tells you which new features the model actually splits on.
6. `experiments/2026-06-06_late_descent_triplet/step2_shap_corr.csv` —
   per-fold pairwise SHAP-value correlations between the 6 new features.
   High correlation between two features = they're encoding the same
   information (SHAP-level redundancy).
7. `experiments/2026-06-06_late_descent_triplet/step2.log` — full stdout
   trace if you need it.

Also look up these memories before deciding anything: `feedback_seed_discipline`,
`project_shap_redundancy_regime_caveat`, `feedback_regime_segmented_evaluation`,
`project_late_descent_investigation`.

## Quick context (mechanism per family)

- **A (network px_std, network px_std_delta_3d)** — cross-station price
  dispersion within the competitive cohort and its 3-day derivative.
  Hypothesis: in a normal cycle the network compresses into a coordinated
  trough; in extended descent it doesn't.
- **B (network_disc_gap, network_disc_gap_delta_3d)** — per-date gap between
  the competitive cohort median and the discount cohort median (and its
  derivative). Hypothesis: the discount cohort encodes the slow-moving cost
  baseline; the gap widens when the floor itself is sliding down (i.e. in
  gradual decline regimes).
- **C (lga_phase_std, lga_phase_std_delta_3d)** — std across 35 LGAs of
  `days_since_trough_entry`, and its derivative. Hypothesis: in a normal
  cycle the leader-follower sequence is intact and LGAs trough within a few
  days of each other; in extended descent that sequence breaks and the
  spread blows out. (Pre-experiment validation in `step1d` confirmed this
  signal lives in genuinely-still-descending rows, not just ascent leader-lag.)

## How to read the run grid

- **R0_baseline** = 50-feat Phase 4 baseline. The reference everything else
  is compared against.
- **R1_ABC** = baseline + all 6 new features. Headline lift.
- **R2_drop_A / R3_drop_B / R4_drop_C** = R1 minus one family.
- **R5_A_only / R6_B_only / R7_C_only** = baseline + one family.

Attribution in `step2_meta.json` uses a single sign convention:
**positive = log-loss reduction when family is present** (family helps). The
attribution table reports per family per cohort:

- `standalone_improvement` = `ll(R0_baseline) − ll(R_X_only)`. Positive = X
  helps when added to baseline alone.
- `marginal_improvement` = `ll(R_drop_X) − ll(R1_ABC)`. Positive = X is doing
  work in R1 — dropping it makes the model worse.

The gap between standalone and marginal-given-others tells you redundancy.
If `standalone(A) = +0.010` but `marginal(A) = +0.001`, A's information is
mostly already accessible to the model when B and C are in — A is redundant.
If both numbers agree, A is orthogonal to the others.

## Cohorts (most important: hard25 and lated)

- **all** — full val set per fold.
- **hard25** — top-quartile baseline per-row log-loss per fold ("hard
  cohort"). This is the primary acceptance cohort per issue #206 — the
  rows the baseline finds hardest, where late-descent ambiguity concentrates.
- **hard10** — top-decile. Even more concentrated.
- **lated** — rows with `cycle_pct_through ≥ 0.9 AND _px_5d_change ≤ -2c`.
  This is the **most targeted slice** — "true late descent" rows where C's
  signal was validated in step1d. A family that helps `lated` specifically
  is doing exactly what we hoped.

## Acceptance bar (per family, for Step 3 inclusion)

A family stays iff **all** of:

1. `standalone_improvement` on hard25 ≥ +0.005 (logloss reduction).
2. `marginal_improvement` on hard25 ≥ +0.005 (when dropped from R1, things
   get worse).
3. No single fold blows up by more than the average lift gives back. Per
   memory `feedback_regime_segmented_evaluation`, look at per-fold deltas in
   `step2_fold_run.csv`; a family that helps on average but regresses by
   +0.05 on one fold is a different verdict from one that helps uniformly.
4. Seed adequacy check: mean improvement / mean seed_std_hard25 ≥ ~2.
   (If the lift is comparable to seed noise we can't trust it.)

Soft criterion: a family that fails hard25 but helps `lated` materially is
worth flagging. The `lated` cohort is small but is the precise regime we
were trying to fix.

## What I want you to produce

1. **A markdown report** (you can write inline in chat or under
   `experiments/2026-06-06_late_descent_triplet/step2_analysis.md`,
   whichever you prefer) covering:

   - Headline: did the triplet (R1) materially lift the model? On which
     cohorts and by how much?
   - Per-family verdict (A, B, C): standalone Δ, marginal Δ, redundancy
     gap, fold blowup check, seed adequacy check, recommended verdict
     (KEEP / DROP / WITHIN-FAMILY-FOLLOW-UP).
   - SHAP read: which of the 6 new columns were used? Any SHAP-corr pairs
     above 0.5 between the 6 new features (redundancy candidates)? Apply
     the `project_shap_redundancy_regime_caveat` discipline — flag, don't
     drop.
   - Regime split: how did the triplet do on shock folds {1, 4, 9, 13}
     vs normal? Per `feedback_regime_segmented_evaluation`, a feature that
     blows up on one regime needs to be called out even if averages are fine.
   - Recommendation for Step 3: which features graduate to
     `fuel_signal/features.py`? Any within-family ablations needed (level
     vs delta per family)? Any threshold-sensitivity follow-ups?

2. **Update the experiment README** under "Outcome" with the result + the
   Step 3 decision. Keep it tight (≤ 1 screen).

3. **File follow-up GH issues** if the analysis surfaces them. Specifically
   consider:
   - If a family graduates: an issue to land the feature columns in
     `fuel_signal/features.py` + retrain the 50→52/54/56-feat baseline.
   - If a family is ambiguous: an issue to run the within-family ablations
     (level only vs delta only).
   - If the entire triplet flops: an issue to close the late-descent track
     and motivate external-data work per `project_late_descent_investigation`.

4. **Update memory** for any persistent learnings — e.g. if A is the clear
   winner, the "cross-station dispersion is a high-marginal signal for
   late-descent" finding is worth a project memory entry.

## Important constraints

- Do NOT modify `experiments/results.csv` — that's main-only and the user
  fills it in, per memory `feedback_results_csv_main_only`.
- Do NOT spawn subagents for analysis — this is straightforward enough to
  do inline.
- Do NOT re-run the experiment.
- If recommending feature graduation, the actual feature implementation
  goes in a separate PR (per repo conventions); file the issue, don't write
  the code yet.

Start by reading the README and step2_meta.json in parallel, then walk
through the analysis. One screen of markdown summary per family is plenty.
