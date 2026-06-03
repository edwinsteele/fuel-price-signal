# redundancy_phase4b

- **Date:** 2026-06-03 (lab book entry; runs from late May 2026)
- **Branch:** feat/feature-redundancy-shap → main (merged via PR #189-ish, commits 189449d / 4b57c18)
- **SHA:** 4b57c18 (most recent fix; earlier runs used 189449d)
- **Status:** open

> ⚠ **How to use this output.** The candidate list in `clusters.csv` /
> `decomposition_candidates.csv` is a list of **hypotheses to test**, not a
> list of features to drop. SHAP correlation here was measured on a single
> training run + val window; redundancy on one window does not imply
> redundancy across regimes.
>
> Before proposing any drop motivated by this list, run a paired walk-forward
> CV (`fuel_signal.cv_report`) and reject if any fold regresses by > +0.05
> logloss. See `experiments/2026-06-03_drop_redundant_pair/` for a worked
> example where a high-SHAP-correlation pair looked great on one window
> (Δ −0.018) but failed walk-forward CV (fold 9: **+0.103**, abandoned).
>
> Tracked: design issue [#195](../../../../issues/195).

## Hypothesis
The 50-feature Phase 4b set has redundant SHAP-correlated features. At the **default cluster threshold (0.5)**, surface the loosest grouping to motivate either removing dupes or decomposing combined features into independent components.

## Setup
- Model: `data/models/lgbm.joblib`
- Features: `data/features.csv` (50 cols, 59,811 val rows)
- Split: `val`
- Cluster threshold: **0.5** (loose)
- Interaction sample: 3000 rows
- Seed: 0

Run via `fuel_signal.shap_redundancy` (see commit 189449d). Full config in `params.json`.

## Results
- `clusters.csv` — feature clusters at threshold 0.5
- `decomposition_candidates.csv` — features flagged for decomposition
- `interaction_matrix.csv` — pairwise SHAP interaction strengths
- `shap_corr.csv` — pairwise SHAP correlation
- `dendrogram.png` — clustering tree (gitignored)

Compare to `redundancy_phase4b_tight/` (0.3) and `redundancy_phase4b_p2/` (0.2) for sensitivity.

## Conclusion
TBD — fill in after threshold-sweep comparison.

## Followups
- Possible decomposition issues for the top candidates in `decomposition_candidates.csv`.
- Cross-reference against [[project_lga_feature_mechanisms]] to classify each redundancy by mechanism.
