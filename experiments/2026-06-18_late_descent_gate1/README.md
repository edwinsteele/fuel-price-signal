# Late-descent / trough-proximity — economic Gate 1 (#259)

- **Date:** 2026-06-18
- **Branch:** main (proxy pre-read); realised gate rides a `fuel_signal/backtest.py` PR
- **Status:** open — **Gate-1 pre-read FIRED** (directional); realised gate in progress

## Question

Is realised buyer economics **materially worse in the late-descent / elongated
zone than in normal rows**? If not, the whole regime/late-descent thread stays
dormant (the #254 dormancy wake-up, gate 1). Must be measured **economically,
not in log-loss** — WFCV per-row log-loss is a non-rejecting screen for
decision-timing features (`feedback-wfcv-logloss-screen-not-verdict`,
CONVENTIONS.md § Choosing the gate metric).

This is the successor to the closed #237/#253 corner threads; their "retest on
the corrected ruler" premise died with #254 (the regime denominator is
economically inert — decision-twins, fold 7 the lowest disagreement at 1.3%).

## Regime tag

`cycle_pct_through` (production feature = `days_since_peak / mean_cycle_length`,
PIT-safe), three bands:

| band | cut | meaning |
|---|---|---|
| `normal` | pct < 0.6 | early/mid descent — prices high, still falling → WAIT |
| `late_descent` | 0.6 ≤ pct < 1.0 | descent into the trough |
| `overdue` | pct ≥ 1.0 | elongated / at-or-past a deep trough |

Validated by base rate: BUY-worthy rows (`label=1` = cheap now **and** no >3c
drop coming within 7d) cluster at pct ≥ 0.6 (base rate 0.33–0.35) and are scarce
below (0.09). So the bands isolate the right zones.

## Layer 1 — proxy pre-read (cheap, directional only)

`proxy_regret_by_regime.py`: reads the already-computed #254 `rowpreds.parquet`
(R0 = honest live post-#250 baseline arm) — **zero new fits** — joins the regime
tag, and per (regime, seed) computes:

- `peak_cents` = max over τ of `expected_cents_per_row` (value the model extracts);
- `oracle_cents` = `base_rate × 6.37` (perfect-classifier ceiling, TP_REWARD);
- `regret` = `oracle_cents − peak_cents` (value left on the table — normalises out
  the differing base rates so the bands are comparable).

```bash
PYTHONPATH=. uv run python experiments/2026-06-18_late_descent_gate1/proxy_regret_by_regime.py
```

### Result (mean over 5 seeds)

| regime | n | base_rate | peak_cents | regret | efficiency (peak/oracle) | τ |
|---|---|---|---|---|---|---|
| normal | 342,572 | 0.092 | −0.46 | **1.05** | −79% | 0.20 |
| late_descent | 248,707 | 0.331 | 0.39 | **1.72** | **19%** | 0.25 |
| overdue | 207,332 | 0.353 | 0.93 | 1.32 | **42%** | 0.05 |

Seed std on regret is 0.018–0.036 — <2% of the levels and far below the ~0.67
c/row cross-regime gap, so it passes the variance sanity gate and the means are
trustworthy.

### Reading (regret vs efficiency reconcile)

Use `normal` and `overdue` as controls:

- **`normal` is not a problem** — peak is *negative* (buying in early descent is
  dominated by waiting); the model correctly almost-abstains (lowest buy_rate).
  Its 1.05 regret is essentially irreducible.
- **`overdue` is the easy win** — cycle so stretched the price is unambiguously
  cheap; τ collapses to the 0.05 floor ("just buy"); model captures 42%.
- **`late_descent` is the real weak spot** — it has *the same opportunity* as
  overdue (oracle 2.11 vs 2.25) but the model captures only **19% vs 42%**. Equal
  money available, less than half captured → a genuine **skill gap**, not "more
  troughs to miss." The model is good when the trough is obvious (overdue) and
  fumbles when the bottom is ambiguous (still falling, ±1–2 day jitter).

**Verdict:** Gate-1 pre-read **fires**, concentrated in `late_descent`.

**Caveat (why this is a pre-read, not the gate):** proxy `expected_cents_per_row`
is a TP/FP/FN classification score, not realised CPL through the buy/wait
simulation (two-exams). And the oracle ceiling assumes *perfect* trough-calling,
which the jitter makes impossible — so 1.72 is not "1.72 recoverable." A fired
proxy is permission to build the realised gate, not the gate passing.

## Layer 2 — realised gate (DONE — fires, with a reframe)

`realised_by_regime.py`: production baseline (54 feat) through the #255 harness
with `collect_fills=True`, 14-fold walk-forward, seed 42, isotonic. Every fill
tagged by the cycle regime at its fill date; model CPL vs a regime-matched
always-buy. 523s.

> **Harness gotcha:** the inner-OOF calibration runs inside each *outer* fold's
> train. Fold 1's train is ~1825d (the outer `train_min_days`), so the inner
> default (also 1825d) yields 0 folds → `ValueError`. Pass
> `inner_fold_params={"train_min_days": 1095, "val_days": 90, "step_days": 90}`
> (3y inner min-train, production 90d val/step). Only moves τ uniformly — the
> per-regime comparison is a post-hoc tag on the same fitted models, so unaffected.

### Result — saving% vs regime-matched always-buy

| regime | model_cpl | always_cpl | **saving%** | emergency_frac |
|---|---|---|---|---|
| normal | 195.80 | 197.05 | **0.64%** | 0.88 |
| late_descent | 184.54 | 188.56 | **2.13%** | 0.49 |
| overdue | 188.59 | 196.58 | **4.06%** | 0.46 |

Pooled: 1.87% at τ=0.25 (harness independently picked τ=0.25 = the production lock).

- **Confirms the proxy on real spend.** Monotonic gradient `0.64 → 2.13 → 4.06`
  (6× spread); overdue ≈ 2× late_descent — same "captures far more once the trough
  is obvious" pattern as the proxy (42% vs 19%). The regime axis is economically
  **live**.
- **Reframes the hypothesis.** #254/#259 expected the *elongation* zone to be
  economically *worse*. The opposite: **overdue/elongated is the model's BEST zone**
  (crushes always-buy 196.6→188.6 — a stretched cycle is an unambiguous, easy-to-call
  deep trough). The soft spot is **late descent** (the ambiguous descent *into* the
  trough), and even it beats normal. So the target sharpens to "lift late-descent
  capture toward overdue's," not "rescue an elongation pit."

### Verdict: **Gate 1 FIRES** → proceed to Gate 2.

### Interpretation caveat + pending cleanup (do before final writeup)

- **saving% ≠ recoverable headroom.** It's measured vs *always-buy* (weak baseline),
  so it's value-*delivered*, not catchable-value-*missed*. late_descent's modest
  number is partly because always-buy is itself cheap there (188.6, near-trough) —
  little room to beat, not necessarily poor skill. Whether the gap is **recoverable**
  is exactly Gate 2's oracle job (the realised arbiter has no oracle ceiling, by
  design).
- **Emergency-fill confound.** `normal` is 88% forced (tank-floor) fills → an
  abstention zone, not skill. Even late/overdue are ~half forced. A **chosen-only
  (non-emergency) saving%** would isolate decision skill — `result.fills` has the
  `emergency` flag; cheap post-hoc, **PENDING**.
- **9.5% dropped fills (158/1659)** lost the pct-join (dropped, not mis-bucketed).
  Verify the drops aren't **regime-correlated** (would weaken a band's numbers).
  Cheap post-hoc on the ledger, **PENDING**.

## Gate 2 (next — Gate 1 fired)

Train-only oracle existence check: does an oracle regime feature cut the
`late_descent` excess loss **beyond what the model's existing `cycle_mean_length`
drift clock already delivers** (the sharpened #254 bar)? Reuses the #237
`phase_oracle_cycles.py` template.
