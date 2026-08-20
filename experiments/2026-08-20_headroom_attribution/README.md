# Per-zone headroom attribution — consumption-interval allocation + bootstrap CIs

- **Date:** 2026-08-20
- **Branch:** main
- **SHA:** dfcba29
- **Status:** done — the estimator improvement works; the quantity it estimates turns out not to be identified
- **Beads:** `fps-1785999730023-4-264564ac` (this issue); unblocks/redirects `fps-x0f` step 2

## Hypothesis

The #262 headroom map produces **impossible negative** per-zone headroom cells
(`>=16c` −0.53, 2025-01 −3.6, 2023-10 −2.9). A perfect-foresight oracle cannot
lose, so those are attribution artifacts. The bd issue diagnosed two causes —
unpaired denominators (the arms fill on different dates and in different counts)
and path coupling (the oracle buys cheaply just *before* an expensive stretch and
coasts through it, so the saving lands in the neighbouring zone) — and proposed
consumption-interval attribution plus station-bootstrap CIs as the fix.

Expected: negatives disappear, cells get error bars, and the per-zone numbers
become trustworthy enough to drive feature decisions.

## How to invoke these scripts

```bash
PYTHONPATH=. uv run python experiments/2026-08-20_headroom_attribution/attribution.py
PYTHONPATH=. uv run python experiments/2026-08-20_headroom_attribution/plot_regime.py
```

## Setup

Reuses `experiments/2026-06-19_headroom_map/{model,oracle}_fills.parquet` — no
re-fit, no re-simulation. The tank model is deterministic (50 L, 50/14 L/day,
7-day evaluation, so exactly 25 L per interval and the level only ever takes
{0, 25, 50}), all 1454 fills sit on a clean 7-day lattice, and `emergency` pins
the post-fill level. That makes the interval-by-interval burn schedule exactly
reconstructible from the fill ledger alone.

**Two independent free conventions**, neither with physical content:

| convention | choices | what it decides |
|---|---|---|
| cost basis | `fifo` / `average` | which litre in the tank you burn first |
| interval label | `start` / `mid` / `end` | which date stamps a burn spanning [d, d+7) |

Both leave each unit's **total** cost identical. They only move cost between
periods — which is the entire quantity under measurement. 6 combinations.

Bootstrap resamples whole **units** (folds primary, 14; stations and
station-folds also reported) so each unit's two arms stay together and the
pairing survives the resample. 4000 draws, percentile CIs.

## Results

### The estimator improvement works

- **Denominators are paired.** Litres per zone now differ between arms by
  0.0–1.7% (`normal` exactly 0.0%), against ~30% fill-count differences under
  purchase-event attribution.
- **Reconstruction reconciles exactly** on all 140 station-fold-arms, both cost
  bases: every litre bought is burned, still in the tank, or prior inventory.
- **Zone tagging validated against the source:** the recomputed purchase-event
  estimator reproduces #262's published 7.09 and −0.53 to the decimal.

### The quantity is not identified — the actual finding

Per-zone headroom under all six conventions (c/L):

```
-- regime --                                -- volatility --
cost         average         fifo           cost      average         fifo
label        end   mid start end  mid start label     end  mid start  end  mid start
late_descent 0.62 1.97 1.49  1.38 3.24 2.65 12-16c   2.58 2.04 3.97  2.25 0.53 3.17
normal       3.34 2.56 2.10  1.63 0.81 1.21 8-12c    2.58 2.70 1.69  1.83 1.37 0.94
overdue     -0.02 -0.49 1.21  2.46 1.49 1.55 <8c     1.05 1.06 1.07  0.96 1.04 1.13
                                            >=16c   -0.11 0.58 0.89  2.74 5.02 3.46
```

| axis | per-zone spread ACROSS conventions | spread BETWEEN zones | identified? |
|---|---|---|---|
| regime | 2.53 – 2.95 | 0.90 – 3.36 | **no** |
| volatility | 0.16 – **5.13** | 1.78 – 4.49 | **no** |

**A zone comparison is only readable when the gap between zones exceeds the gap
between conventions.** On both axes it does not. The "worst zone" changes three
times across six estimators on regime; on volatility `>=16c` ranges −0.11 to
5.02 — a 5.13 c/L swing from choices with no physical content.

**Impossible negatives are not robustly fixed either.** They vanish at
`start` labelling (which is what made the first pass look like a clean win) but
return at `average/end` (overdue −0.02), `average/mid` (overdue −0.49) and
`average/end` volatility (`>=16c` −0.11): 3 negative cells across the 42-cell
grid.

### Contrasts: nothing separates

Ordering claims are contrasts, so they were tested as contrasts (bootstrapped
difference, which is tighter than comparing two overlapping CIs because the
folds are shared). Under `fifo/start`:

- **regime:** no pair separates. Best is late_descent > normal, p=0.90,
  CI [−0.77, +3.49].
- **volatility:** no pair involving 12-16c separates. 12-16c vs `>=16c` is
  p=0.42 — a coin flip on which is larger. (One pair does separate, 8-12c vs
  `>=16c`, in this one convention out of six.)

### What survives

- **Window-level headroom.** #262's overall 1.66 c/L needs no allocation across
  sub-periods and is unaffected. The model really is ~⅔ of the way from
  always-buy to the oracle floor.
- **`<8c` volatility.** 0.96–1.13 across all six conventions — a 0.16 spread,
  the only stable cell in the grid. Mechanically sensible: in calm periods prices
  barely move, so where you book the cost hardly matters. It is the exception
  that demonstrates the rule.

## Conclusion

**The bias fix succeeded and the target quantity is still unusable.** Per-zone
headroom is not identified: the free conventions move it further than any effect
it could reveal, on every axis tried. This is not a variance problem — more
stations, more folds and more seeds shrink error bars and do nothing to
convention spread. The bd issue's own caveat was right and is the headline
rather than a footnote: **the only fully-clean statement is window-level.**

Granularities at which `oracle <= model` holds:

| granularity | bound holds? |
|---|---|
| whole window, per station (what the oracle optimises) | yes, by construction |
| pooled across stations/folds, window-level | yes, empirically |
| any sub-window zone (regime / volatility / quarter / month) | **no** — 3/42 grid cells negative |

On the bd issue's third acceptance criterion — does the 12–16c ~7 c/L peak
survive with a CI clear of 0? **Partly, and not as a peak.** Its own CI clears 0
under both cost bases at `start` labelling, so there is headroom in that band.
But the point estimate roughly halves (7.09 → 0.53–3.97 depending on
convention), no contrast against any other band separates, and under `fifo` the
largest band is `>=16c`, not 12-16c. "There is headroom around 12–16c
volatility" is supportable; "12–16c is the hot zone" is not.

## Followups

- `fps-x0f` step 2 ("is late descent a real target?") **cannot** be answered by
  slicing headroom into cycle zones, at any sample size. Recorded on that issue.
  The replacement measure under consideration is decision-timing accuracy — did
  the model choose a worse week than the oracle? — which never allocates a cost
  to a period and so sidesteps the identification problem entirely.
- Any future work reading #262's per-zone rows (regime FLAT, volatility hump,
  monthly episode clustering, the ±0.5 per-band noise floor derived from the
  negatives) should treat them as withdrawn. Its window-level rows stand.
