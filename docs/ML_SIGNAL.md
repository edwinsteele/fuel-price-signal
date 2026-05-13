# ML signal: design decisions

This document captures the decisions behind the ML model so agents don't inadvertently undo them. Read before touching `labels.py`, `features.py`, `evaluate.py`, or any model training/scoring module.

## Target formulation

Binary classification: **P(min price over next H days < today − X cents)**.

The decision rule falls out of the probability — "BUY when P ≥ threshold". Trains on log loss; evaluated with reliability plots, Brier score, and realised-spend backtest.

## Label design (load-bearing — do not change without owner sign-off)

**label=1 means BUY.** BUY=1 is the natural polarity — boolean true means "do the thing."

**Two conditions, both must hold for label=1:**
1. `future_min >= today_price - threshold` — no significantly cheaper price arriving within `H` days (no drop predicted)
2. `today_price <= Nth percentile of past lookback_days` — price is cheap in absolute terms (catches plateaus where no drop is predicted but price is still high)

**Why two conditions:** condition 1 alone labels high-price plateaus as BUY (no drop coming, but price is still expensive). Condition 2 filters those out.

**Parameter defaults:** horizon=7d, threshold=3c, lookback=90d (≈2 cycle lengths), percentile=33.

**What this is targeting:** avoiding buying at the top of the cycle. NOT finding the exact cycle minimum.

The price cycle is ~45 days; most users refuel every 1–4 weeks and cannot hold out for a full cycle. Targeting the trough is impractical. This design decision is explicit and deliberate — do not silently reframe toward trough-finding.

## Label polarity note

A common mistake: treating label=0 as the "positive" class (waiting). Don't. BUY=1 is positive. Precision/recall/F1 reported in results are for label=1.

## Pooled model — one model for all stations

Per-station models were considered and rejected on data volume grounds. One model, all stations contribute training rows. Station enters as a categorical feature alongside brand/suburb.

## Cycle detector: kept as feature source

`CycleDetector` is not replaced by ML — it's a feature source. Its outputs (`cycle_pct_through`, `days_since_peak`, plateau flag, last-cycle min/max, deviation from cycle mean) feed into the model. The user-facing "Day 41/46" narrative stays as the interpretability anchor.

Old rule "intra-series → cycle detector, cross-series → ML" is superseded. Both flows feed one pooled model now.

## Sequencing

| Phase | Work | Status |
|-------|------|--------|
| 1 | Feature pipeline + PIT validation | Done |
| 2 | Logistic regression baseline (cycle features only, 7d H, 3c X) | Done — locked 2026-05-09 |
| 3 | LightGBM + station/cross-station lead features | Next |
| 4 | Upstream features (TGP first, then MOPS/crude/FX) | Deferred |
| 5 | Macro model (separate, longer horizon ~30–90d) | Deferred |

## Phase 2 results (locked baseline)

τ=0.40 on test: logloss 0.4029 (vs baseline 0.5821), brier 0.1346 (vs 0.1966). Realised CPL 190.35 c/L vs always-buy 191.78. Phase 3 must beat 190.35 c/L. See [docs/STATUS.md](STATUS.md) for full table.

## Non-goals

- Deep learning
- Per-station models
- Predicting absolute price levels
- Replacing the cycle detector
- Finding the exact cycle trough (see label design above)
- Seasonal decomposition (data is cyclic, not seasonal)

## Val BUY rate anomaly (not a bug)

Val BUY rate (35.8%) is elevated vs train (25.8%) and test (26.6%). This is a window-specific anomaly — the val period happened to contain more BUY-eligible days. The +0.05 τ adjustment in `score_phase2.py` corrects for this when picking the threshold.
