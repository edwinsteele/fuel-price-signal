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

Seed std on regret 0.018–0.036 — <2% of the levels and far below the ~0.67 c/row
cross-regime gap. Passes the variance sanity gate; means are trustworthy.

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

## Layer 2 — realised gate (in progress)

Harden Layer 1 on **realised CPL** via a reusable per-fill ledger on
`run_backtest` (the realised analog of `rowpreds.parquet`): emit per-fill
`(date, station, price, litres, spend)`, bucket by regime-at-fill-date, compute
regime-stratified realised CPL + a regime-matched always-buy baseline, walk-forward
via the #255 harness. The engine stays agnostic about *why* you slice; the
experiment does the stratification. (Reusable for Phase-4 leadership, shock-fold,
season/LGA slicing.)

## Gate 2 (only if Layer 2 confirms)

Train-only oracle existence check: does an oracle regime feature cut the
`late_descent` excess loss **beyond what the model's existing `cycle_mean_length`
drift clock already delivers** (the sharpened #254 bar)? Reuses the #237
`phase_oracle_cycles.py` template.
