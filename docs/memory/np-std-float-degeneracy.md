---
name: np-std-float-degeneracy
description: np.std of identical floats isn't 0.0 — pair band-degeneracy checks with np.ptp or a 1e18 z-score reads as a confident reject.
metadata:
  type: project
---

np.std of identical floats is NOT 0.0 — guard band degeneracy with np.ptp, not `std > 0`. np.std([0.01]*20, ddof=1) == 1.78e-18 because 0.01 has no exact binary representation (0.02 -> 3.6e-18; 187.8 -> 2.9e-14; only exact 0.0 gives 0.0). A `band_std > 0` test therefore does NOT catch 'every draw landed on the same value' — the case noise_floor.py's docstring names — and a z of (delta-mean)/1.78e-18 is order 1e18. In dossier_tables._noise_band that read as a confident REJECT under the mechanical -t<z<t split and wrote 'rejected' into ledger.yaml: a wrong grade, not a refusal. Pair the check with float(np.ptp(deltas)) > 0, which is exact and zero iff the draws really are identical. Fixed on the sibling-bank path in PR #345, then on the canonical path in PR #346 (fps-tnz) — both now call a single shared _band_std_usable(deltas, band_std) helper (extracted after PR #346 review) so the two checks cannot drift apart again the way they did between #345 and #346. The same std==0-without-ptp-pairing pattern recurs uncorrected elsewhere in the repo (fuel_signal/shap_report.py, rank_partners.py, lga_shap_plots.py, dependence_grid.py) — tracked separately as fps-9rw.
