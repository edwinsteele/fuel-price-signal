# LightGBM fit-speed benchmark — analysis

**Verdict: C1 SKIP · C2 SKIP**

No convention changes. C0 (`LGBMClassifier` defaults) remains the standard.

---

## C1 — `force_col_wise=True`

| | C0 | C1 |
|---|---|---|
| Equivalence | — | **PASS** (all folds/seeds within 1e-6) |
| Mean fit_s | 3.33 | 5.65 |
| Speedup | — | **−69%** |

C1 passes equivalence — identical log-loss — but is 1.5–2× *slower* than the
baseline, getting worse at larger folds (fold 14, seed 46: 10.1s vs 4.6s for C0).

**Why**: LightGBM's auto-heuristic already chose row-wise histogram building for
this workload and was right.  The col-wise crossover typically requires ~100+
features; at 54 features and 50–60k training rows, col-wise overhead exceeds its
benefit.  Overriding the heuristic was net harmful.

**Decision: SKIP.** Do not add `force_col_wise=True` to any script.

---

## C2 — `lgb.train` + `lgb.Dataset` reuse

| | C0 | C2 |
|---|---|---|
| Equivalence | — | **FAIL** |
| Mean fit_s | 3.33 | 3.03 |
| Speedup | — | +9% |

C2 fails equivalence.  The `_LGBM_PARAMS_BASE` dict does not exactly replicate
`LGBMClassifier` internal defaults — most folds show small systematic ll
differences above the 1e-6 tolerance.  More critically, fold 4 seed 46 diverges
badly (C0: 0.423, C2: 0.819 — a near-degenerate model).  The missing parameter
is not obvious from inspection, which is itself the warning: manually mapping
sklearn-API defaults to the booster API is fragile and requires a line-by-line
diff against `booster_.params` output.

Even if equivalence were fixed, 9% speedup (~0.3s/fit, ~1 min per 210-fit run)
is marginal for the complexity it introduces: loss of the sklearn API, per-fold
Dataset bookkeeping, and the param-mapping maintenance burden.

**Decision: SKIP.** The Dataset-reuse pattern is not safe to adopt without a
verified param mapping.  If revisited, the correct starting point is
`clf.get_params()` output → `booster_.params` output diff, not a hand-rolled
dict.

---

## Summary

The 54-feat baseline already runs at ~3.3s/fit on this hardware.  A 14-fold
× 5-seed sweep takes ~4 min wall time; there is no compelling fit-speed problem
to solve until feature count or fold count grows substantially.
