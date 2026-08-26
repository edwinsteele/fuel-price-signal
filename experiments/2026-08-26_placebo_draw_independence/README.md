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
| A: the **committed** 10×3 ruler, all columns distinct | 405 | 0.015 | 0.111 | **0.965** |
| B: proposed 20×3, columns reused | 1710 | 0.012 | 0.106 | **0.350** |

Bank A is read from `batch1/noise_floor.json` rather than re-derived by calling
`screen_draw_groups`. That call now returns the *post*-fix construction, so re-deriving would
silently stop measuring the thing this write-up argues against and quietly turn the headline
0.965 into ~0.26. The committed ruler is also the honest object of comparison: it is the bank
batch1's dossiers were actually graded against.

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
| 1–2 | 20–40 | 0% | 20.00 |
| **3** | 60 | 5.8% | **13.98** ← the cap |
| 4 | 80 | 16.3% | 9.04 |
| 5 | 100 | 27.9% | 6.51 |
| 6 | 120 | 48.9% | 4.31 |
| 10 | 200 | ~100% | 2.4 |
| 35 | 700 | 100% | 2.4 |

Which arity that makes the cap depends on what it is compared against — see the next section,
because the first answer here was 4 and it was wrong.

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
3. **No arity cap.** Reuse is PRICED, not forbidden — see the correction below.

### SUPERSEDED 2026-08-26 by `fps-3jj.25`: the cap was the wrong instrument

This write-up originally shipped `placebo.MAX_RULER_ARITY`, first at 4 and then at 3 after
review. **Both are gone.** The owner's standing position — stated on `fps-3jj.14` and again
here — is that constraining candidate shape to suit the ruler distorts the thing being
measured. A cap did exactly that: it made the instrument dictate what the generator was allowed
to propose, and cost a wide candidate its verdict entirely.

**And the cap rested on a crude model.** The exposure arithmetic charged two draws the FULL
texture ICC if they shared even one column. But two draws sharing 1 of 10 columns share about a
tenth of their texture, not all of it. Charging the shared FRACTION instead:

| candidate columns | n_eff (binary charge) | n_eff (fraction charge) | bar (1 candidate) |
|---|---|---|---|
| 1–2 | 20.00 | 20.00 | −0.152 |
| 3 | 13.99 | 17.49 | −0.154 |
| 4 | 9.04 | 15.35 | −0.156 |
| 10 | 2.40 | 9.04 | −0.171 |
| 35 | 2.40 | 3.23 | −0.287 |

The binary charge put a 20-draw arity-10 bank at 2.4 effective draws, which is what made a cap
look necessary. Under the faithful charge the bar **degrades gently at every width** — which is
exactly what the parent bead said the old design failed to do ("the bar does not degrade
gracefully as arity rises — it EXPLODES"). The cap was solving a problem the modelling invented.

`placebo.effective_n_draws` now prices reuse from the bank that actually got built, and
`family_wise_z_threshold` takes the effective count in place of the nominal one. A wider
candidate reuses more, so its band is worth fewer draws, so its bar is automatically wider. No
refusal, no `run_arity_above_cap`, no nightly re-pick loop, and nothing telling the generator
what shape to propose.

**The motivating case is solved.** A 35-column per-LGA candidate — which this bead opened by
noting could not have a band computed *at all* — builds a 20-draw bank on batch1's real frame
and grades at −0.287 c/L. Verified with the shipped code, not modelled.

**Independence cost, stated as the bead asks:** reusing a source column costs a factor of ~2.5
on the *median* placebo-pair correlation (0.038 vs 0.015) and **nothing on the tail** (p90 flat
at 0.07–0.11 across every source-correlation band, same-source included). The median gap is
largely composition — "different source" includes 32 pairs of genuinely unrelated columns
sitting at 0.003, a bin same-source cannot occupy. Both figures sit far under the 0.6 the
screen already tolerates on a placebo's correlation to its own source. In the delta channel the
cost is bounded at ICC ≤ 0.391 and is not resolvable below 0.359.

### What was NOT decided, and must not be read as decided

**Nothing here measures the texture ICC.** It is bounded, not known, and the bound is what every
bar above is priced against — see the next section. That is the single weakest load-bearing
number in the change.

**The physical limit is real and stays.** Two 35-column draws taken from 49 columns must overlap
in at least 21 of them, by pigeonhole, so a wide bank genuinely is worth fewer effective draws
and genuinely does get a harder bar. That is now priced rather than refused. Drawing placebo
sources from outside the lock (bd `fps-3jj.22`) would reduce the overlap and tighten the bar —
an improvement now, where it used to be a precondition.

## What is still a bound, not a number

`ICC ≤ 0.391` (`placebo.TEXTURE_ICC_BOUND`) is the load-bearing input to every bar, and it comes
from an underpowered ANOVA. bd **`fps-3jj.23`** measures it directly, and is in flight —
`experiments/2026-08-27_texture_icc/` holds the instrument, the power analysis and the result.

**Two corrections to this section, from that work.** First, "load-bearing input to EVERY bar" is
true but overstated: at arity 1–2 the constant is irrelevant by construction, and at arity 3–4 —
every candidate this batch ran — its *whole range* moves the bar by ≤ 0.011 c/L. It is
load-bearing for wide groups only. Second, the construction sketched in the next sentence (one
same-source column, variance compared against the multi-column floor) **cannot discharge the
bead at the ~2.3h budgeted here**: it is an unpaired variance ratio on `F(9,19)`, resolvable only
above 0.661, and its best possible outcome is a 0.587 upper bound — looser than the 0.391 it
would replace (both at the one-sided 5% tail, the tail `texture_channel.py` used for 0.391). The design that works groups several pinned columns and runs the ANOVA **by
column**, at 32 draws (~6.1h).

**It is not established as conservative, despite being an upper bound.** The bound is on the ICC
by texture *family*; the quantity actually used is the ICC by *column*, and family is coarser —
two different `days_since_trough_entry_<lga>` columns are same-family but different-column, while
two placebos from the *same* column are more alike than that. The column ICC is plausibly
**larger**, so 0.391 is not demonstrably an upper bound on it, and if the true value exceeds it
then bars on wide candidates are too **easy** — the failure this mechanism exists to prevent.
Weak evidence against a very large one: it should have produced *some* family clustering, and the
ANOVA found none. Weak is why `fps-3jj.23` exists.

At the point estimate (ICC ≈ 0) every width sits at 20.00 effective draws and every bar collapses
to arity 1's −0.152 — at which point `fps-3jj.22` buys nothing and should be closed. `fps-3jj.23`
is recorded in bd as **blocking** `fps-3jj.22` for exactly that reason.

## Verification against real data

Shipped code, batch1 frozen frame, banks the old contract could not build:

```
arity 1: 20 draws x 1   seeds distinct 20/20   columns used 20 (max reuse 1)
arity 2: 20 draws x 2   seeds distinct 40/40   columns used 40 (max reuse 1)
arity 3: 20 draws x 3   seeds distinct 60/60   columns used 49 (max reuse 2)

arity  draws   n_eff   bar(1)          <- fps-3jj.25, same frame, shipped code
    1     20   20.00   -0.152
    2     20   20.00   -0.152
    3     20   17.49   -0.154
   10     20    9.04   -0.171
   35     20    3.23   -0.287          <- the per-LGA candidate this bead opened on

20x3 shipped bank, worst cross-draw placebo correlation: 0.350
  draw 3 station_minus_lga_mean_cents  <->  draw 4 stickiness_score
  (batch1's committed 10x3 ruler: 0.965 — draw 1 vs draw 10, shared seed 97)
```

Every arity reaches the full 20 draws, which is the defect fps-3jj.21 was filed on: under the
old contract arity 3 could reach only 10, and arity 18+ could produce no band at all. `max reuse 2` at arity 3 matches
`divmod(60, 49)` exactly, so the exposure model in the table above describes what the shipped
sampler actually produces, not an idealisation of it.

Arity 1 at 20 draws uses 20 columns with zero reuse — unchanged from today.

## Consequence for existing floors

**A floor recomputed after this lands is not comparable to one computed before it, wherever
substitution fired** — the seeds assigned to substituted draws changed. Committed floors on
disk are untouched (nothing recomputes them) and the no-substitution case is unaffected, since
the first 30 seeds are the historical pool verbatim. Treat a recompute like a re-lock:
recompute, don't mix. batch1's committed ruler still carries its 0.965 pair; it was not
recomputed here, because doing so would move the ruler under dossiers already written against
it.

**Outcome (`fps-3jj.24`, decided 2026-08-27): it stays uncomputed, and the retrospective ranks
against the committed ruler with the collision priced in prose rather than in the band.** Three
things settled it.

*First, `fps-3jj.25`'s reuse correction does not catch this collision and cannot.*
`placebo.effective_n_draws` prices shared SOURCE COLUMNS; draw 1 and draw 10 share none —
`cycle_pct_through / cycle_mean_length / cycle_last_max_cents` against `cycle_days_since_peak /
cycle_last_min_cents / station_price_cents`, 30 picks and 30 distinct columns across the bank —
so `dossier_tables._resolve_effective_n_draws` reconstructs the nominal **10.0 exactly**. This
write-up's own table is why: the machinery charges *same column, different seed* (median |ρ|
0.038) at `icc = 0.391` and charges *different column, same seed* (max |ρ| 0.971) **nothing**.
It prices the cheap channel and ignores the expensive one. That inversion is harmless for every
future floor — `block_seed`'s never-restarting counter closes the seed channel by construction —
so it is a defect of two committed artifacts, not of the code, and `effective_n_draws` should
**not** be changed to chase it.

*Second, the bias is exactly computable with no compute, and it only ever points one way.*
Pricing the collided pair at its observed 0.836 correlation through this project's own formula
puts the bank at **8.57** effective draws:

| gate | committed | corrected | shift |
|---|---|---|---|
| single candidate (`n_candidates=1`) | −0.1670 | −0.1727 | −0.0057 c/L |
| batch gate (`n_candidates=5`) | −0.2706 | −0.2850 | −0.0144 c/L |

The correction is monotone-HARDENING, so it can only demote a candidate, never promote one — the
committed ruler wrongly fails nobody, and can only wrongly PASS a candidate sitting in
|`z`| ∈ [1.9226, 1.9797) at `n_candidates=1` or [2.9591, 3.1030) at `n_candidates=5`. That is a
free per-row check at write-up time, and `docs/routines/retrospective.md` now carries it as a
standing instruction. None of the four candidates dossiered so far is inside it (|z| = 0.523 /
0.926 / 0.987 / 1.805), and each of their `facts.json` files emits a reassuring-looking
`effective_n_draws: 10.0` that is exactly the number this section says to distrust.

*Third, the recompute was refused because it is a WORSE instrument than the arithmetic above,
not merely a more expensive one.* A fresh 10-draw band estimates its own std to ±23.6%
(`1/sqrt(2(n-1))`) — ±0.0697 c/L on the `n_candidates=5` bar, **4.8× the 0.0144 c/L bias it
would remove**. Combined with the non-comparability rule stated just above, a recomputed floor
could not answer "did the collision matter": a ranking change would be indistinguishable from
resampling luck, and an unchanged ranking would prove nothing the closed form does not already
give for free.

*Incidental finding.* The retired `noise_floor_k1.json` carries the same defect — 20 draws at
arity 1, two collided pairs (draws 1/19 on seed 97, draws 2/20 on seed 101), `n_eff` again
reading the nominal 20.0. It is the k=1 baseline behind `docs/CONVENTIONS.md`'s "two thirds of
the k=1 → k=3 bar move is draw thinness, not arity" attribution. Both floors are biased the same
direction, so correcting both moves the gap only from −0.0536 to −0.0637 c/L at
`n_candidates=5`: **the two-thirds claim survives.**

## How to reproduce

```bash
PYTHONPATH=. uv run python experiments/2026-08-26_placebo_draw_independence/measure_independence.py
PYTHONPATH=. uv run python experiments/2026-08-26_placebo_draw_independence/texture_channel.py
```

Run from the **repo root** (paths are repo-relative, like every sibling experiment). The frozen
frame `measure_independence.py` needs is gitignored, so it exists only in the clone that ran the
freeze — from a worktree, point `FUEL_BATCH_DIR` at that clone's batch dir. `texture_channel.py`
needs nothing untracked.

Both read the batch1 frozen frame from the primary worktree read-only and need no fits;
`measure_independence.py` is ~60s, `texture_channel.py` is instant. `measure_independence.py`
writes `source_column_correlations.csv`, `reuse_channel.csv` and `bank_cross_draw.csv` into
this dir; they are regenerated on every run and deliberately not committed — every number this
write-up rests on is quoted above. (`texture_channel.py` reads `reuse_channel.csv`'s companion
`source_column_correlations.csv`, so run `measure_independence.py` first.)

`measure_independence.py` prototypes the proposed construction as `deck_picks`/`deck_draws`
rather than importing it, so the measurement that justified the change stays runnable against
the code it was arguing about. `placebo` ships that shape as `candidate_sequence`; the
prototype is deliberately not refactored to call it.

## Followups

- bd `fps-3jj.23` — measure the texture ICC directly (replaces the ≤ 0.391 bound)
- bd `fps-3jj.22` — placebo sources from outside the lock (the only route to high arity)
- bd `fps-3jj.20` — closed, fixed here
