# What does the placebo bank's distinct-column rule actually buy? (bd `fps-3jj.21`)

- **Date:** 2026-08-26
- **Branch:** `claude/fps-3jj-21-6de4ec`
- **Status:** done — landed in `experiments/pipeline/placebo.py`

## The finding

**The constraint that binds is not the constraint that matters.**

`placebo.screen_draw_groups` required `n_draws × arity` DISTINCT source columns and the same
number of entries from a fixed 30-long seed pool, capping a bank at
`min(⌊54/k⌋, ⌊30/k⌋)` draws at arity k. Since `family_wise_z_threshold` carries
`df = n_draws − 1`, the bar did not degrade gracefully as arity rose — it exploded, and
almost none of the explosion was the arity effect. A 35-column per-LGA candidate — this
project's own motivating precedent for group candidates — could not have a band computed at
all.

Measured on batch1's frozen frame, placebo-pair |ρ| by what the pair shares:

| pair shares | n | median | p95 | max |
|---|---|---|---|---|
| self (what the screen already accepts at τ=0.6) | 52 | 0.048 | 0.134 | 0.183 |
| **same source column, different seed** | 78 | **0.038** | 0.123 | 0.218 |
| different source column, different seed | 156 | 0.015 | 0.115 | 0.175 |
| **different source column, SAME seed** | 156 | 0.097 | 0.782 | **0.971** |

Reusing a source column is cheap. **Reusing a seed is not** — an order of magnitude worse in
the tail, and it is the one thing the construction did not guard.

### Why

A block seed selects a *rearrangement* of the date blocks, not a slice. Correlation between
two time series lives in their **alignment**, not their values, and a different rearrangement
destroys alignment regardless of which column it started from — two placebos from the same
column are "prices from randomly chosen months" against "prices from a *different* random set
of months", same distribution, no co-movement.

The same seed does the opposite. Both columns get the *identical* rearrangement, so their
alignment **to each other** survives untouched, and a pair of near-duplicate sources comes out
as correlated as it went in.

That is live in batch1's committed grading ruler, and it is deterministic rather than unlucky:

```
seed 97:  draw 1 cycle_pct_through      |  draw 10 cycle_days_since_peak   (sources ρ=0.971)
seed 101: draw 1 cycle_mean_length      |  draw 10 cycle_last_min_cents    (sources ρ=0.778)
seed 103: draw 1 cycle_last_max_cents   |  draw 10 station_price_cents     (sources ρ=0.765)
```

Draw 10 shares **all three** of draw 1's seeds — it is not one unlucky pair, it is an entire
duplicated draw. The chain: `select_draws` spreads picks by position, three land on all-NaN
`days_since_trough_entry_*` columns, the screen skips them as unverifiable, substitutes come
from `candidate_pool`'s fallback tail — **and the tail restarted the seed cycle at index 0**,
which is exactly where draw 1's seeds came from. Every step deterministic; recomputing the
floor reproduces it.

### Whole-bank comparison

| bank | pairs | median | p95 | max cross-draw \|ρ\| |
|---|---|---|---|---|
| A: current 10×3, all columns distinct | 405 | 0.015 | 0.111 | **0.965** |
| B: proposed 20×3, columns reused | 1710 | 0.012 | 0.106 | **0.350** |

**Double the draws, strictly more independent on every statistic.** B's worst pair is
`station_minus_lga_mean_cents` vs `stickiness_score` — the between-station correlation floor
already named as a limitation in `placebo.py`'s module docstring — not a reuse artifact. A's
worst pair *is* the seed collision.

And the premise behind "distinct columns" was already false: the 49 usable lock columns have
**298 pairs at |ρ| ≥ 0.7** and only 5 texture families (30 of them are LGA trough counters).
Bank A shares a texture family across 51% of its draw pairs; bank B, 38%.

## The channel correlation cannot see, and how far it was pinned

A placebo reaches its delta two ways: **alignment** (does this shuffle line up with the target
somewhere the fit can exploit — what correlation measures) and **texture** (how much does
adding a column of this *shape* perturb the fit at all — distribution, cardinality, NaN rate,
autocorrelation). Two placebos from the same source share texture **exactly**. If texture
drives the band's variance, reuse correlates the deltas, the sample std shrinks, and the bar
gets too **easy** — `fps-3jj.14`'s bias, by a different door.

Tested on the committed floors, no fits: one-way ANOVA of batch1's 20 arity-1 deltas grouped
by texture family. **F(3,16) = 0.411, p = 0.75**, ICC point estimate ≈ 0; the 11 near-identical
LGA-counter draws do not cluster (std 1.25× the whole bank's, if anything wider).

**But the design can only resolve ICC ≥ 0.359.** So this is "could not see it", not "it is not
there" — see `feedback_correction_no_finer_than_estimator`. The honest output is the bound,
**ICC ≤ 0.391**. Two caveats, both cutting the same way: family is a *coarser* grouping than
column, so this probably **understates** the same-column ICC; and it reuses a bank built for a
different purpose.

Carrying the pessimistic bound through a 20-draw bank's reuse exposure — the table that sets
the cap:

| arity | picks from 49 columns | draw pairs sharing a column | effective draws @ ICC 0.391 |
|---|---|---|---|
| 1–2 | 20–40 | 0% | 20.0 |
| 3 | 60 | 6% | 14.0 |
| **4** | 80 | 16% | **9.0** ← the cap |
| 5 | 100 | 28% | 6.5 |
| 6 | 120 | 49% | 4.3 |
| 10 | 200 | ~100% | 2.4 |
| 35 | 700 | 100% | 2.4 |

Today's live 10-draw ruler is worth ~**8.3** effective draws once its 0.965 pair is counted.
Arity 4 is the last value whose worst case still beats that.

## The decision

**Option 3 from the bead (reuse source columns across draws), plus removing the seed pool
outright.** Three parts:

1. **`placebo.block_seed(i)`** — an unbounded, never-restarting seed supply whose first 30
   values are the historical pool verbatim, so any bank of ≤30 picks stays byte-identical.
   Closes the 0.971 route by construction rather than making it less likely. Resolves
   `fps-3jj.20`.
2. **`placebo.candidate_pool` lays repeated laps** over the column list instead of stopping at
   one, with reuse spread as evenly as possible. `n_draws` and `arity` are now independent —
   pick the draw count for the certainty wanted and pay the compute. It also returns
   `n_picks + n_columns` entries instead of exactly `n_columns`, so a bank that consumes the
   column list still has substitution headroom (the bead's "hard-fail after hours of compute,
   or keep a column that FAILED the screen" hazard).
3. **`placebo.MAX_RULER_ARITY = 4`** — a hard refusal above it, from the table above.

**Independence cost, stated as the bead asks:** reusing a source column costs a factor of ~2.5
on the *median* placebo-pair correlation (0.038 vs 0.015) and **nothing on the tail** (p90 flat
at 0.07–0.11 across every source-correlation band, same-source included). The median gap is
largely composition — "different source" includes 32 pairs of genuinely unrelated columns
sitting at 0.003, a bin same-source cannot occupy. Both figures sit far under the 0.6 the
screen already tolerates on a placebo's correlation to its own source. In the delta channel the
cost is bounded at ICC ≤ 0.391 and is not resolvable below 0.359.

### What was NOT decided, and must not be read as decided

**High-arity candidates are not solved.** The change makes a 35-column candidate *computable*;
it does not make the resulting band mean anything, and the pessimistic case says it does not.
This is a pigeonhole limit, not a sampler defect — **two 35-column draws taken from 49 columns
must overlap in at least 21 of them**. No sampling rule fixes that; it needs placebo sources
from outside the lock, bd `fps-3jj.22`.

Note that the cap lands close to the bead's **option 4**, the arity cap the owner rejected on
`fps-3jj.14`. The rejection was of a cap arrived at *by the pool running out*. This one is
measured, is the last value that beats the incumbent ruler, and coincides exactly with the
"2–4 columns" `docs/routines/generator.md` already invites — so the ruler and the invitation
now agree instead of the ruler quietly being narrower. That was the bead's own stated condition
for accepting a cap.

## What is still a bound, not a number

`ICC ≤ 0.391` is the load-bearing input to `MAX_RULER_ARITY` and it comes from an underpowered
ANOVA. bd **`fps-3jj.23`** measures it directly: a noise floor where every draw uses the *same*
source column with a different seed varies **alignment alone**, so the difference between its
variance and the existing multi-column floor's **is** the texture component. ~2.3h. ICC near 0
lets the cap rise (and may make `fps-3jj.22` unnecessary); ICC near the bound means the cap is
right or should come down.

## Verification against real data

Shipped code, batch1 frozen frame, banks the old contract could not build:

```
arity 1: 20 draws x 1   seeds distinct 20/20   columns used 20 (max reuse 1)
arity 3: 20 draws x 3   seeds distinct 60/60   columns used 49 (max reuse 2)
arity 4: 20 draws x 4   seeds distinct 80/80   columns used 49 (max reuse 2)

20x3 shipped bank, worst cross-draw placebo correlation: 0.350
  draw 3 station_minus_lga_mean_cents  <->  draw 4 stickiness_score
  (batch1's committed 10x3 ruler: 0.965 — draw 1 vs draw 10, shared seed 97)
```

Arity 1 at 20 draws uses 20 columns with zero reuse — unchanged from today.

## Consequence for existing floors

**A floor recomputed after this lands is not comparable to one computed before it, wherever
substitution fired** — the seeds assigned to substituted draws changed. Committed floors on
disk are untouched (nothing recomputes them) and the no-substitution case is unaffected, since
the first 30 seeds are the historical pool verbatim. Treat a recompute like a re-lock:
recompute, don't mix. batch1's committed ruler still carries its 0.965 pair; it was not
recomputed here, because doing so would move the ruler under dossiers already written against
it.

## How to reproduce

```bash
PYTHONPATH=. uv run python experiments/2026-08-26_placebo_draw_independence/measure_independence.py
PYTHONPATH=. uv run python experiments/2026-08-26_placebo_draw_independence/texture_channel.py
```

Both read the batch1 frozen frame from the primary worktree read-only and need no fits;
`measure_independence.py` is ~60s, `texture_channel.py` is instant. `measure_independence.py`
writes `source_column_correlations.csv`, `reuse_channel.csv` and `bank_cross_draw.csv` into
this dir; they are regenerated on every run and deliberately not committed — every number this
write-up rests on is quoted above. (`texture_channel.py` reads `reuse_channel.csv`'s companion
`source_column_correlations.csv`, so run `measure_independence.py` first.)

`measure_independence.py` prototypes the proposed construction as `deck_picks`/`deck_draws`
rather than importing it, so the measurement that justified the change stays runnable against
the pre-change code.

## Followups

- bd `fps-3jj.23` — measure the texture ICC directly (replaces the ≤ 0.391 bound)
- bd `fps-3jj.22` — placebo sources from outside the lock (the only route above arity 4)
- bd `fps-3jj.20` — closed, fixed here
