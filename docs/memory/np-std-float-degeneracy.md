---
name: np-std-float-degeneracy
description: np.std of identical floats isn't 0.0 — pair band-degeneracy checks with np.ptp or a 1e18 z-score reads as a confident reject.
metadata:
  type: project
---

np.std of identical floats is NOT 0.0 — guard band degeneracy with np.ptp, not `std > 0`. np.std([0.01]*20, ddof=1) == 1.78e-18 because 0.01 has no exact binary representation (0.02 -> 3.6e-18; 187.8 -> 2.9e-14; only exact 0.0 gives 0.0). A `band_std > 0` test therefore does NOT catch 'every draw landed on the same value' — the case noise_floor.py's docstring names — and a z of (delta-mean)/1.78e-18 is order 1e18. In dossier_tables._noise_band that read as a confident REJECT under the mechanical -t<z<t split and wrote 'rejected' into ledger.yaml: a wrong grade, not a refusal. Pair the check with float(np.ptp(deltas)) > 0, which is exact and zero iff the draws really are identical. Fixed on the sibling-bank path in PR #345, then on the canonical path in PR #346 (fps-tnz) — both now call a single shared _band_std_usable(deltas, band_std) helper (extracted after PR #346 review) so the two checks cannot drift apart again the way they did between #345 and #346. The same pattern recurred in fuel_signal/shap_report.py, rank_partners.py, lga_shap_plots.py, dependence_grid.py and lga_dependence_interaction.py; all five now call `fuel_signal.shap_report.is_degenerate(arr)` (fps-9rw / #377, PRs #406 and eee9939). There are now THREE copies of this predicate — `_band_std_usable` (dossier_tables.py), `_finite_positive_spread` (experiments/lib/flips.py) and `is_degenerate` (shap_report.py, the negated form). Keep them in step: #406 shipped `is_degenerate` missing the `np.isfinite` half the other two have, so a ±inf element read as *usable* and fed NaN into approx_interaction_scores' running sum.

**The artefact is LENGTH-dependent, and not monotonically so — never assume it shows up at the n your test happens to use.** Measured on numpy 2.4.4, `np.std(np.full(n, 0.01))`:

| n | 2 | 3 | 4 | 5 | 10 | 16 | 20 | 25 | 32 | 50 | 64 | 100 | 128 |
|---|---|---|---|---|----|----|----|----|----|----|----|-----|-----|
| std | 0.0 | 0.0 | 0.0 | 0.0 | 1.7e-18 | 0.0 | 1.7e-18 | 0.0 | 0.0 | 0.0 | 0.0 | 1.7e-18 | 0.0 |

Only n = 10, 20, 100 are nonzero in that range (pairwise summation cancels exactly at most lengths). So a regression test written at n=50 asserts nothing — it exercises the exact-0.0 path the bare `std == 0` check already caught, and would still pass against the unfixed code. `tests/test_shap_report.py` pins n=20 for exactly this reason and asserts `np.std(arr) != 0.0` first, so the test fails loudly if a numpy upgrade moves the boundary rather than silently going vacuous. See [[test-discrimination-claims-need-mutation]].
