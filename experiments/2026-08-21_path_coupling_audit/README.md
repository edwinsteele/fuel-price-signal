# Path-coupling audit — what else in this repo slices a path-coupled total by time period?

- **Date:** 2026-08-21
- **Branch:** `claude/fps-grp-be6124`
- **SHA:** 8e19f8b
- **Status:** done
- **Beads:** `fps-grp` (this audit); simplifies `fps-x0f`; successor to
  `fps-1785999730023-4-264564ac` (`experiments/2026-08-20_headroom_attribution/`)

## Hypothesis

`2026-08-20_headroom_attribution` showed that per-zone headroom is **not
identified**: allocating a path-coupled total cost across sub-periods has no
unique answer, and the free bookkeeping conventions move each zone further than
the zones differ from each other. That defect is a property of the *quantity*,
not of the #262 script. Expected: other results in this repo compute the same
shape of quantity and inherit the same defect, silently.

**No new runs.** Every verdict below is either structural (read from the code
that produces the number) or arithmetic on numbers already committed to the
repo. Nothing was re-run, re-fit or deleted.

## The test applied

A quantity is **at risk** if BOTH hold:

1. it is a cost/economics aggregate over a strategy whose value is
   **path-dependent** — a buy now changes what is possible later, true of every
   tank-based backtest here; AND
2. it is reported **sliced by a time-varying label** (cycle regime, quarter,
   month, volatility band, sub-window zone).

A quantity is **safe** if it is natively stamped at a moment (per-row log-loss,
per-decision accuracy, prices, predictions, feature values), or if it is a
**whole-window total** — the granularity `run_oracle_backtest` optimises, where
`oracle <= model` holds by construction.

**The pivotal structural fact, verified rather than assumed:** a *fold* is not a
sub-period. `experiments/lib/realised.py` calls
`aggregate_backtest(..., p.val_start, p.val_end, tank)` once per fold, and
`fuel_signal/backtest_phase2.py:aggregate_backtest` calls `run_backtest` once
per station with a fresh tank. So each (fold, station) is an **independent
simulation with its own tank**, and per-fold / per-station / per-fold-group
economics are sums of complete windows — no allocation is performed, so nothing
is unidentified. That is why fold-cut economics survive and row-label-cut
economics do not, and the distinction is mechanical, not a judgement call.

## Results — the enumeration

`sliced by` says what the number is cut on. `verdict` applies the test above.

### At risk

| # | quantity | where | sliced by | why at risk |
|---|---|---|---|---|
| A1 | realised saving% by cycle regime — `normal 0.64% / late_descent 2.13% / overdue 4.06%` | `2026-06-18_late_descent_gate1/realised_by_regime.{py,csv}`, README Layer 2 | row-level `cycle_pct_through` band at each fill's own date | purchase-event attribution of a tank-path CPL. **Demonstrated unidentified below**, not merely suspected. |
| A2 | chosen-only realised saving% — `normal 11.45% / late_descent 6.45% / overdue 11.03%` and the flat chosen-only CPL `174.5 / 176.4 / 174.9` | `2026-06-18_late_descent_gate1/cleanup_checks.py`, `cleanup_chosen_only.csv`, README Layer 3 | same regime band | same construction. It is the *second convention*, not the fix — see below. |
| A3 | every per-zone headroom row of #262 | `2026-06-19_headroom_map/` (`headroom_map.py`, `headroom_periods.py`, README, CSVs) | regime / quarter / volatility band / month | **already withdrawn 2026-08-20**; listed for completeness. The README itself carried no annotation until this audit. |
| A4 | `breakdowns.per_axis[].delta_cpl_own` — candidate−baseline CPL per `add_axis` label | `experiments/pipeline/dossier_tables.py:_breakdowns` | a candidate's own per-row `add_axis` label, joined per (station_code, date) | **live and prospective.** Not yet triggered — batch0 declared no `add_axis`, so `per_axis` is `null` in the only dossier on disk — but any batch-2 candidate that declares one gets an unidentified number presented next to identified ones. |
| A5 | `CONFIDENCE_ZONE` grading when `TARGET["axis"]` is a candidate `add_axis` label | `experiments/pipeline/runner.py:_grade_zone` | same | same. Resolves a *scored prior* off it, so the defect propagates into the cross-batch calibration record. |
| A6 | the #262 per-zone findings quoted verbatim as standing guidance to the feature generator | `docs/routines/generator.md` §"Reading order", item 4 | — | not a measurement, but the highest-impact leak: it instructs every future candidate to aim at `12–16c 7.09` and to treat "regime axis is FLAT" as settled. |

### Safe

| # | quantity | where | why safe |
|---|---|---|---|
| S1 | Layer-1 proxy regret by regime — `normal 1.05 / late_descent 1.72 / overdue 1.32` | `2026-06-18_late_descent_gate1/proxy_regret_by_regime.py` | **natively stamped at a moment.** `expected_cents_per_row` is a TP/FP/FN classification score: each row's contribution depends only on that row's prediction and label. No tank, no ledger, no path. Passes clause 1 of the test, so clause 2 never bites. (Its own separate caveats — proxy-not-arbiter, τ re-picked inside each bucket, defective phase axis — are unchanged by this audit and are `fps-x0f`'s business.) |
| S2 | `realised_spend_cpl` + `realised_savings_vs_always_buy_pct` | `experiments/results.csv` (6 rows), written by `fuel_signal/evaluate.py:log_experiment` via `fuel_signal/score_phase2.py:run_realised_spend_backtest` | every one is a **single whole window** — all 6 are `test 2025-07-01 → 2025-12-31`, one CPL per row, no period cut. The schema has no zone column, so a sliced figure cannot be stored here. |
| S3 | `breakdowns.per_fold[].delta_cpl_own` | `dossier_tables.py:_breakdowns` | fold = independent simulation with its own tank (see above). A sum of complete windows. |
| S4 | `breakdowns.per_regime[].delta_cpl_own` (`shock` vs `normal`) | `dossier_tables.py:_breakdowns`, `runner.py:_grade_zone` when `axis == "regime"` | cut on `SHOCK_FOLDS`, i.e. a **grouping of whole folds** — `fills["fold"].isin(folds)`. Never splits a window. This is the safe way to ask a regime question, and it is already the pipeline's default. |
| S5 | `CONFIDENCE_ZONE` grading when `TARGET` names `folds` or `axis: "regime"` | `runner.py:_grade_zone` | same as S3/S4. |
| S6 | per-fold headroom and the 12/14-fold robustness read | `2026-08-20_cadence_ceiling/` | per-fold = per-window. The cadence experiment's own headline (`1.54 → 2.97 c/L`) is window-level pooled. |
| S7 | per-fold Δcpl, incl. "gains concentrate in expensive folds 8/9/10" | `2026-06-20_leading_indicators/` | per-fold. **Note:** this claim is retracted, but on reproducibility/noise-floor grounds (batch0, 2026-08-19) — *not* by this audit. Two different failures; don't merge them. |
| S8 | realised CPL by arm, 2025-H2 | `2026-06-16_regime_cycle_length/` | single window per arm; the comparison is arm-vs-arm at window level. |
| S9 | drop/retention mix by regime (`cleanup_drop_mix.csv`) | `2026-06-18_late_descent_gate1/cleanup_checks.py` | row counts, not costs. A coverage fact. It stays valid and remains a real caveat on A1/A2's overdue cell. |
| S10 | per-date feature aggregates, SHAP, WFCV log-loss, reliability tables, `cycle_phase_breakdown.png` | throughout `experiments/`, `fuel_signal/evaluate.py` | natively stamped at a moment. |

Also checked and clear: `fuel_signal/score_phase2.py` (single-window CPL only),
`fuel_signal/evaluate.py` (one CPL per experiment row; `reliability_table` is
per-row), `experiments/lib/zones.py:pooled_cpl` (a correct primitive — the
defect is in what callers group before calling it, never in the function).

## A1/A2 decided: the gate1 per-regime saving% is NOT identified

The bead flagged that this one is measured against **always-buy rather than an
oracle**, so it cannot produce impossible negatives and the defect would be
silent. It is — but the repo already contains the demonstration, in the
experiment's own Layer 3.

**Emergency-fill inclusion is a free bookkeeping convention.** Both readings are
defensible and both are used in this very experiment: include forced fills
("what the strategy actually cost you", Layer 2) or exclude them ("the quality of
the decisions the model actually made", Layer 3). Nothing physical decides it.

And it is the path-coupling carrier in its purest form: an emergency fill
stamped in `normal` is the settlement of a **wait chosen in a different
regime**. The model's emergency fraction is `0.87 / 0.49 / 0.46` by regime while
always-buy's is `0.00` everywhere, so the convention lands on the buckets
unequally by construction.

Applying `docs/CONVENTIONS.md` § *Bucketed results* to the two committed
conventions, in c/L (arithmetic on `realised_by_regime.csv` +
`cleanup_chosen_only.csv`, no re-run):

| bucket | all fills | chosen only | **spread ACROSS conventions** |
|---|---|---|---|
| normal | 1.25 | 22.56 | **21.30** |
| late_descent | 4.01 | 12.16 | **8.15** |
| overdue | 7.99 | 21.67 | **13.69** |
| **spread BETWEEN buckets** | **6.73** | **10.40** | |

**The convention spread exceeds the between-bucket spread on every bucket**
(8.15–21.30 vs 6.73–10.40). And the ordering does not merely blur, it
**reverses**:

```
all fills   best -> worst:  overdue, late_descent, normal
chosen only best -> worst:  normal,  overdue,      late_descent
```

Best and worst swap places. That is one free convention. The two #262 found —
cost basis (`fifo`/`average`) and interval label (`start`/`mid`/`end`) — are
additional and untested here, and this construction is **worse coupled than
#262's** on the two axes that mattered there:

- **Denominators are wholly unpaired.** #262's arms at least filled under the
  same tank; here the always-buy arm fills on every evaluation date, so its
  per-regime litres bear no relation to the model's. `2026-08-20_headroom_attribution`
  measured ~30% fill-count differences under purchase-event attribution against
  0.0–1.7% litre differences under consumption-interval allocation.
- **The published record cannot even be checked.** `realised_by_regime.csv`
  stores `model_fills` and `model_litres` but no `always_litres` / `always_fills`,
  so the denominator's size is not recoverable from the committed artifacts.
  (`realised_fills.parquet` is gitignored.)

**Verdict: A1 and A2 are withdrawn as economics.** Both directions go: the
Layer-2 monotonic gradient AND the Layer-3 flat-chosen-only reading. Layer 3 is
not the corrected version of Layer 2 — it is a second draw from the same
unidentified quantity, which is why it disagrees.

**What survives.** The *diagnosis* in Layer 3 survives and is strengthened: the
Layer-2 gradient is driven by emergency dilution and a regime-varying always-buy
denominator, not by model skill. What does not survive is the positive claim
built on top of it — "the model pays ~175 c/L wherever it chooses to buy, in
every regime". That was a measurement, and it is not identified either. The
defensible position is weaker and different: **the realised regime axis has
never been measured with an identified instrument**, in either direction.

Also surviving, unaffected: S1 (Layer-1 proxy regret), S9 (the overdue drop
censor), and the *structure* of the argument — always-buy is a weak yardstick
whose regime-varying level contaminates any saving% cut on regime.

## Consequence for `fps-x0f`: the contradiction dissolves

`fps-x0f` exists to reconcile three mutually-inconsistent late-descent findings.
After this audit:

| # | finding | reads as | status |
|---|---|---|---|
| 1 | Gate-1 Layer-1 proxy regret (vs oracle ceiling, per-row) | late descent is the **worst** zone | **stands** (S1) |
| 2 | Gate-1 Layer-2 realised saving% (vs always-buy) | late descent is **better than normal** | **withdrawn** (A1) |
| 3 | #262 per-regime headroom (vs oracle ceiling, realised) | late descent is the **least recoverable** | **withdrawn** 2026-08-20 (A3) |

Two of the three are gone, and **they are the two that disagreed with each
other**. There is no longer a contradiction to reconcile — there is one
un-contradicted proxy reading, which says late descent is the worst zone, and
**no identified realised measurement of the regime axis at all**.

This does not resurrect the late-descent thread. It does change what `fps-x0f`
step 1 is for: not "why do three findings disagree" but "there is one surviving
reading, it is a proxy, it is cut on an axis with a known unfixed defect
(`cycle_mean_length`), and nothing has ever measured this axis in realised
economics without allocating a path-coupled cost." Note also that the thread was
*rested* on finding 3, which is withdrawn — so the closure argument is gone
independently of whether the conclusion was right.

## Conclusion

Six at-risk quantities, ten safe. The two that matter are **A1/A2** (decided
here: withdrawn, with the demonstration done in arithmetic on committed numbers)
and **A4/A5/A6** — the live AI-feature pipeline, where the defect is not yet in
any published number but is wired into `add_axis` / `CONFIDENCE_ZONE` and into
the generator's standing instructions.

The single durable rule, already in `docs/CONVENTIONS.md`
§ *Bucketed results — check the convention spread before believing an ordering*:
**a bucket comparison is readable only when the gap between buckets exceeds the
gap between free bookkeeping conventions**, and that gap is a *bias* term — more
stations, folds or seeds will never shrink it.

The mechanical corollary this audit adds: **cut economics on folds, not on row
labels.** A fold is a whole simulation; a row label is a slice through one.

## Followups

- `fps-x0f` — commented with the table above; its step 1 is re-scoped, not blocked.
- `docs/routines/generator.md` — the withdrawn #262 zone numbers replaced with
  what actually stands, plus an `add_axis` warning.
- `experiments/pipeline/` — `per_axis` now ships an identification caveat with
  the number, and `_grade_zone` marks a row-label-axis grade as unidentified.
