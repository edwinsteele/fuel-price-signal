# Data semantics

Reference for how this repo's price data is shaped and classified. Architecture and module
layout live in [AGENTS.md](../AGENTS.md); process rules in [CONVENTIONS.md](CONVENTIONS.md);
current model state in [STATUS.md](STATUS.md).

## Station classification (Competitive / Discount / Sticky)

LGA- and Brand-level mean features used by the ML model must reflect **current pricing that buyers can actually act on**. Stations fall into three behavioural classes; aggregation policy depends on which class they're in. The classifier is built (`classify.py` → `station_class` table); [issue #108](https://github.com/edwinsteele/fuel-price-signal/issues/108) (closed) holds the original design discussion.

**The three classes:**

| Class | Description | Examples |
|---|---|---|
| **Competitive** | Price tracks the cycle; sits within ±10c of the LGA competitive cluster | Metro Tuggerah, Pearl Energy Wyong North, most BP/Caltex/Ampol metro stations |
| **Sticky** | Set-and-forget pricing; sits persistently above the competitive cluster | Shell Reddy Express Woy Woy, EG Ampol Umina, Ampol Foodary motorway sites, BP Berowra |
| **Discount** | Sits persistently below the competitive cluster; real, accessible cheap prices | Costco, Powerfuel, Speedway, Budget Petroleum |

**Aggregation policy:** blended Competitive + Discount means **exclude Sticky only**. Discount stations stay in because their low prices are real and accessible to a buyer; Sticky stations leave because their stale peak prices don't reflect what buyers are currently being offered.

**Why blended (not Competitive-only):** the level shift introduced by including Discount stations (LGAs with discounters look cheaper) reflects real prices buyers can access. Competitive-only would be a cleaner cycle-position signal but discards level information that matters for a purchasing decision.

**Brand aggregates use the same classification.** Brand mean is computed Sydney-wide across stations of brand B where `class != Sticky` (using the same per-LGA-derived classification). One classifier, one `station_class` table; LGA and Brand aggregates are just different slicings. The principle "ML features for pricing decisions must exclude stations that aren't informative about current pricing" applies regardless of slicing dimension — a Sticky Shell in Woy Woy is stale whether you aggregate it by LGA or by brand.

**Out of scope here:** Members-only stations (e.g. Costco) have prices that aren't accessible to a non-member. A separate accessibility filter may be warranted before they enter any "available to buyer" feature — deferred.

**The classifier (1D on premium):**

| Setting | Value |
|---|---|
| Classification axis | Median price-vs-cluster premium |
| Window | 45 days (NSW mean cycle length) |
| Band | ±10c (Sticky if median premium > +10c; Discount if < −10c; else Competitive) |
| Frequency role | Bootstrap seeding of the initial cluster only. Not in the classifier itself, and not a recency filter at aggregation time (see below). |

The classifier is deliberately 1D on premium, not 2D on (frequency, premium). Frequency was a noisy proxy for the property premium measures directly — Sticky stations update less because they're set-and-forget at high prices. A high-frequency station with persistently high premium (e.g. BP Berowra) is still Sticky.

The 45-day window is the empirical mean NSW cycle length. The classifier does **not** try to model cycle-length variation (cycles run 35–70 days) — cycle modelling is out of scope for the current ML model.

**PIT discipline:** The classification window for a training row at date D ends at D−1. Past price classifying past behaviour is not target leakage; the prediction target (future price) is not in the classification window.

**Median is computed in Python**, not SQL — SQLite has no native MEDIAN. Classification is a batch step (daily re-computation across ~800 stations), so the per-call cost doesn't matter.

**Materialisation:** classifications are pre-computed and stored in a `station_class` table:

```sql
CREATE TABLE station_class (
    station_code             INTEGER NOT NULL REFERENCES stations(station_code),
    snapshot_date            INTEGER NOT NULL,   -- YYYYMMDD; classification valid as of this date
    class                    TEXT    NOT NULL,   -- 'Competitive' | 'Sticky' | 'Discount'
    median_premium_decicents INTEGER NOT NULL,   -- median (station_price − cluster_mean) over 45d
    PRIMARY KEY (station_code, snapshot_date)
);
```

**Daily cadence.** Each day's classification uses the 45-day window ending at `snapshot_date − 1`. Daily (rather than monthly) snapshots avoid step-changes in LGA aggregates when borderline stations flip class — the rolling window smooths drift, so daily materialisation just propagates that smoothness into the feature. Storage cost is trivial (~290k rows/year × 5 years ≈ 1.5M rows). Compute cost is sub-second per day.

**No active-reporter recency filter.** An earlier design floated a 14d "last raw observation" filter at aggregation time as a guard against stale forward-filled prices. It was dropped (2026-05-19) for KISS reasons: empirical measurement showed it would exclude 5–10% of non-Sticky stations during peak plateaus where the forward-fill is *correct* (price genuinely unchanged), in exchange for limited protection against downcycle staleness that the 28d forward-fill cap and the classifier already partially handle. If model artefacts traceable to ramp-day staleness appear later, revisit — likely with a phase-aware filter rather than a static threshold.

**Aggregation floor:** if fewer than 3 non-Sticky stations are available for a given LGA/brand/date, emit NULL rather than fall back. A silently-thin aggregate is worse than a gap. The floor protects against *thin samples* (high-variance aggregates from few stations); it does **not** protect against staleness — staleness protection lives entirely in the 28d forward-fill cap and the classifier.

**Cold-start handling:** a station gets a classification entry as soon as it has at least one raw observation in the 45-day window. No minimum-observation threshold, no `is_classified` boolean, no default class. Stations with zero observations in the window have no entry and are excluded from aggregates (consistent with their absence from `daily_prices`). Because the pooled ML model uses numeric features only — no station/brand/suburb categorical (the locked baseline is 54 features; see § Canonical feature set) — there is no OOV problem at inference for a brand-new station: its numeric features can be computed from a single observation and the model produces a prediction without special handling. The LGA/brand mean features may be NULL (NaN) if fewer than 3 non-Sticky stations are available; the model handles NaN natively.

## Known source data limitations

**2022-03-12 to 2022-03-21: NSW source data collapses ~85-98%, unrecoverable.** `prices` row counts (raw event log, all fuel grades, all NSW stations) drop from a ~3,500-1,100/day baseline to double- and triple-digit counts (e.g. 2022-03-20: 49 rows), with the surrounding week each side (2022-03-10 to 2022-04-01) also thinner than normal. P95/P98 grades hit **zero events statewide** 2022-03-11 to 2022-03-29 while E10/U91 continued at reduced volume, so this is a genuine source-side reporting collapse, not a real 19-day market freeze — the network E10 median kept moving through the window (183.9 → 176.9 → 185.8 → 198.9 c/L). The 2022-03-30/31 spike (515/549 stations repricing E10 in two days) is the federal fuel excise cut (~22 c/L, effective 2022-03-30), not stations reopening after a genuine pause.

Checked 2026-08-25, no alternative source exists: data.nsw.gov.au's one March-2022 resource (`d707be7a-dbb9-47a7-ae57-599428731fac`) has never been revised since 2022-04-04; rows-per-MB of the published XLSX is consistent with neighbouring months (Feb 19.6k/MB, Mar 19.0k/MB, Apr 19.8k/MB), so this project's extraction is complete — the *source file itself* is short, not our parsing of it. data.gov.au only mirrors the same resource. The NSW real-time Fuel API is snapshot-only and cannot be queried retrospectively. `data/snapshots/` starts 2026-08. **The gap is permanent; plan around it, don't try to fill it.**

Within a single station's `fuel_signal.db` history this mostly self-heals: `fill.py`'s `MAX_GAP_FILL_DAYS = 28` forward-fills across the ~10-19 day hole without triggering exclusion. It becomes a much bigger problem in combination with `labels.py`'s exclusion window for the minority of stations that also have a gap on the *other* side of March 2022 — tracked separately as `fps-ghr` (turns a 2-3 week source hole into a 125+ day exclusion for 81 stations). This entry is the underlying data fact; `fps-ghr` is the code defect it interacts with.

**Evaluation treatment (decided 2026-08-27, `fps-tpy`): flag, don't exclude.** The canonical train/val/test split (`docs/STATUS.md` § Canonical split) never scores this window directly — Val and Test both start in 2025. It only touches the pre-test `walk_forward_folds()` used for feature-change CV (`docs/CONVENTIONS.md` § Changing the production feature set): with the project's standard `train_min_days=1825, val_days=90, step_days=90`, exactly one fold's **val window** is `2022-02-03 to 2022-05-03` (its train window for any later fold that reaches this far also absorbs the hole via forward-filled rows, same as any other station gap). Only ~10-19 of that fold's 90 val days are degraded, `fill.py`'s cap keeps `val_df` non-empty (so `cv_report.py`'s `if val_df.empty: continue` skip does not fire), and the same window already coincides with the Ukraine-invasion price shock (2022-02-24) that regime-segmented CV runs treat as elevated-variance. Given the effect is small, partial, and confined to one fold of many, excluding it would be special-casing evaluation logic for a bounded, already-measured effect — proportionate response is to treat a March-2022-spanning fold the same way `docs/CONVENTIONS.md`'s existing override clause treats any other "known price-shock period": eligible for the fold-regression override, not a reason to rebuild the CV harness. Don't re-litigate this; if a specific CV run shows a March-2022 fold behaving anomalously, cite this note rather than re-deriving the cause.

**A structurally different, non-alarming gap also exists in 2017.** Several months in 2017 (03-27–03-31, 05-19–05-31, 06-18–06-30, 07-19–07-31, 09-05–09-30, 10-14–10-31 — 5 to 26 days each) show **exactly zero** `prices` rows for the tail of the month, resuming cleanly on the 1st of the next month. Unlike March 2022 this is not mid-month and not partial — it lines up precisely with month boundaries in both directions, which is the signature of an early-history bulk-CSV resource that simply didn't cover the full calendar month, not a reporting collapse. The longest (Sept 2017, 26 days) sits just under `fill.py`'s 28-day cap, so it forward-fills without tripping the `fps-ghr` exclusion chain. No other window in 2016-2026 shows either signature — the remaining single- and double-day dips found by the same scan (holidays, weekends) sit at normal-baseline magnitude and don't warrant documentation here.

Composition-drift measurement (the panel-size question this event also raised, originally filed as `fps-ghr`): **answered and closed**, see `fps-tpy` in the archive (`jq -r '.[] | select(.id=="fps-tpy") | .description' docs/bd-archive/issues.json`) — chain-linked index test over 2021-11-05..2025-04-17 (1,260 dates × 714 stations) found total drift +0.461 c/L over 3.4 years, daily |drift change| median exactly 0, concentrated in ~8 days, largest single-day move 0.567 c/L on 2022-06-29 (a different event, not this one). Small and bounded; feeds `station_minus_sydney_avg_cents` and the LGA/brand-mean derivatives on those specific days but is not a first-order threat. Do not re-investigate.

