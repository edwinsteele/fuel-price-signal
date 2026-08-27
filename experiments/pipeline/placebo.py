"""Placebo-column construction for the noise floor (fps-awz, reworked fps-d7m).

`experiments/pipeline/noise_floor.py` needs a null that measures the SAME operation a real
candidate's `effect_delta_cpl_held` measures: hold seed, data, and folds fixed; add the
candidate's columns. This module builds those columns — each one resembling a real feature
in distribution and autocorrelation while having its alignment to the target destroyed,
without ever degrading to a within-date permutation.

**The null's ARITY is a parameter** (fps-3jj.14). It was fixed at one column until batch1
made the mismatch load-bearing: `docs/routines/generator.md` explicitly invites 2-4 column
candidates (every feature win in this project's history arrived as a group), and three of
batch1's five candidates are 3-column groups, so grading them against a 1-column band
compares a k-column arm to a ruler that does not know k. `screen_draw_groups` and
`make_placebo_frame` below build a k-column draw; `noise_floor.compute_noise_floor` takes
`arity` and stamps `n_placebo_columns` into the payload, and `dossier_tables._noise_band`
refuses a floor whose arity is below the run's. Arity 1 is unchanged in every particular,
by construction rather than by inspection — see `placebo_column_names`.

Why a within-date permutation is unsafe here: of the 54 locked baseline columns
(`fuel_signal.features.LOCKED_FEATURE_COLUMNS`), 45 are market-wide — every station shares
the exact same value on a given date. That's all 35 `days_since_trough_entry_<lga>` columns,
all 4 `NETWORK_FEATURE_COLUMNS`, AND 6 of `FEATURE_COLUMNS` (`cycle_pct_through`,
`cycle_days_since_peak`, `cycle_mean_length`, `cycle_last_min_cents`, `cycle_last_max_cents`,
`cycle_peak_count` — all read straight off the market-wide cycle-detector state,
`fuel_signal/features.py`'s `_build_feature_dict`, keyed by date alone, never by station).
Shuffling a market-wide column's values ACROSS STATIONS on the same date is a no-op: every
station already has the same value, so the "shuffled" frame is identical to the original.
fps-3jj's own earlier rejection of a placebo null was of exactly this failure mode. The fix
is not a smarter shuffle but a different axis entirely: reorder the column along TIME,
applied to whichever axis the column actually varies on (`axis_is_market_wide` decides
which, per column, from the data itself rather than a hand-maintained group list that could
drift out of sync with `features.py`).

## Why BLOCK PERMUTATION, not a circular time-shift (fps-d7m)

The original construction was a circular shift (`np.roll`) by a large prime offset. It was
replaced wholesale — not supplemented — because a full screen of batch0's 54 locked columns
showed the shift systematically excluding an entire CATEGORY of feature rather than a few
unlucky draws.

A circular shift cannot decorrelate a slowly-varying (level-like) column from itself: the
column's value on date `d` and on date `d + offset` are both drawn from the same slow drift,
so a big offset still lands on a similar value. Measured on real batch0 data (3572 dates,
718 stations), sweeping ALL 30 primes in the old shift pool, minimum |self-correlation|:

    cycle_last_min_cents  0.653      station_price_cents  0.493
    cycle_peak_count      0.642      lga_mean_cents       0.529
    cycle_mean_length     0.532      brand_mean_cents     0.537
    cycle_last_max_cents  0.548      stickiness_score     0.554

Every one of those minima occurred at the LARGEST prime in the pool (251) — the correlation
falls monotonically as the offset grows, because a longer offset simply reaches further
along the drift. That is the tell: whether a level-like column survived screening depended
on the offset's size relative to the batch's own calendar span, not on any property that
makes a placebo a good null. Two of the eight (`cycle_last_min_cents`, `cycle_peak_count`)
failed at all 30 offsets outright; the other six could only pass at the pool's top end, and
`screen_draws` assigns offsets by POSITION rather than searching for a passing one per
column, so in practice all eight dropped out of the bank. The 20 draws that did survive were
dominated by counters/phase/differences (12 of 20 were `days_since_trough_entry_<lga>`
columns) — so the band characterized "adding a fast-varying column", not "adding an
arbitrary column", while a real candidate feature can be level-like.

Block permutation breaks the property that actually causes this: "adjacent in time implies
similar value". Cut the date axis into contiguous blocks of `PLACEBO_BLOCK_DAYS` and reorder
whole blocks. A level-like column keeps its own levels and its own local texture WITHIN each
block, but the level sitting on a given date is now the level from a randomly chosen other
period, so the long-range drift no longer lines up. Same eight columns, same full batch0
screen, this module's shipped construction, median |self-correlation| over 5 seeds:

    cycle_last_min_cents  0.128      station_price_cents  0.050
    cycle_peak_count      0.033      lga_mean_cents       0.037
    cycle_mean_length     0.066      brand_mean_cents     0.040
    cycle_last_max_cents  0.054      stickiness_score     0.465

— all eight comfortably inside `MAX_SELF_CORRELATION`. Across the whole batch, 49 of 49
usable columns (54 locked, less 5 all-NaN) pass at EVERY seed tried — 245 checks, zero
failures, median 0.046 and worst case 0.470 — so `screen_draws` never substitutes for a
CORRELATION failure on this batch. It does still substitute twice, for a different and
expected reason: `select_draws` spreads its 20 primaries across the column list by position,
which lands on two of the five all-NaN `days_since_trough_entry_<lga>` columns
(`lane_cove` seed 157, `waverley` seed 181). Those correlate to NaN, are skipped as
unverifiable, and are replaced from the fallback tail by `cycle_days_since_peak` and
`cycle_mean_length`. That happens deterministically on every batch0 run. The bank that
results is no longer dominated by one feature kind — 11 trough-counters, 3 cycle-magnitude,
1 price-level, 5 other, against 12-of-20 trough-counters before — though note the two
substitutes are part of how it gets there. Because
block permutation covers BOTH cases, there is one construction here, not two gated by a "is
this column level-like?" heuristic that would need maintaining as `features.py` grows.

The cost is fidelity: a circular shift introduces exactly one discontinuity (the wrap
point), block permutation introduces one per block boundary. Measured, it is small — lag-1
autocorrelation of the market-wide date series, original vs shifted vs block-permuted at 61
days: `cycle_last_min_cents` 0.998 / 0.998 / 0.981, `network_px_std` 0.964 / 0.964 / 0.951,
`cycle_pct_through` 0.893 / 0.893 / 0.884. At 3572 dates a 61-day block leaves 57 seams in
3571 transitions (1.6%). Shorter blocks decorrelate no better and preserve less
(`cycle_last_min_cents` lag-1 drops to 0.961 at 29 days), longer blocks preserve slightly
more but leave fewer blocks to permute — 61 is chosen to sit above the project's own cycle
length (`cycle_mean_length` on batch0: mean 30.5, max 35.3 days) so a block holds at least
one whole cycle, while still leaving ~58 blocks on a batch0-sized date range.

## What this construction still cannot decorrelate (NAMED LIMITATION)

A station-varying column whose variation is mostly BETWEEN stations rather than within one
station over time is untouched by ANY time-axis reordering, block permutation included: each
station keeps its own values, so the cross-sectional structure — which is where that column's
information lives — survives intact. On batch0 this is visible as an irreducible correlation
floor: `stickiness_score` sits at 0.47 and `station_minus_sydney_avg_cents` at 0.33 under
block permutation at every seed and every block length tried, and they are the top two of the
whole batch — the median column sits at 0.046. Both still pass `MAX_SELF_CORRELATION`, so
they DO enter the bank, but they enter as the weakest draws in it — closer to "add a redundant
near-copy of a column the model already has" than to "add a column with no alignment to the
target". Closing this would need
a construction that reorders along the STATION axis as well, which reintroduces exactly the
within-date-permutation hazard the first section rules out; it is recorded here rather than
fixed. `noise_floor.json` stamps every draw's `self_correlation`, so a human reading a floor
can see which draws were weak ones.

## Screening mechanics

"Reordered" is necessary but not sufficient — a particular permutation can land close to the
identity by chance, and the between-station floor above is not fixable by any draw. So
`screen_draws` is the actual safeguard: it checks `self_correlation` for every candidate
BEFORE any backtest fit and substitutes the next candidate on a violation, rather than
committing to a fixed draw list up front and aborting the whole computation when one entry
turns out to be unusable.

The screen tests |correlation|, not signed correlation: a placebo perfectly ANTI-aligned with
its source carries exactly as much information as one perfectly aligned with it, so a signed
test would let an inverted near-copy through. No such draw showed up in the batch0 screen —
this closes a latent gap in the check, it is not a fix for an observed failure.

The self-check itself uses `pandas.Series.corr` (pairwise-complete: rows where either side
is NaN are dropped before correlating), not `numpy.corrcoef` (whole-result NaN the moment
ANY element is NaN). Verified against real batch0 data: `lga_mean_cents` and
`brand_mean_cents` carry PARTIAL NaN (too few classified stations on some dates —
`fuel_signal/features.py`'s own documented behaviour), which `corrcoef` would report as an
unverifiable NaN — masking a real violation as "can't tell" rather than "fails" — and a
column with an incidental NaN entry elsewhere would get the same false pass/refuse ambiguity
for a reason that has nothing to do with its actual, checkable correlation. Five of batch0's
35 `days_since_trough_entry_<lga>` columns are all-NaN (no station ever classified into that
LGA); those correlate to NaN and are skipped as unverifiable.

## Why the two axis branches, and not one date-map

A tempting simplification is to permute the DATE AXIS once and look every row's value up at
(station, permuted_date), which would handle market-wide and station-varying columns with
one code path and would preserve each date's whole cross-section coherently. Rejected on
measurement: batch0's panel is only 81.1% dense (718 stations x 3572 dates, stations entering
and leaving over ten years), so a station present on date `d` is often absent on the date
that maps to it — 13.4% of rows would get a NaN placebo value at a 61-day block, and a
13%-NaN column is a materially different perturbation from the one a real candidate adds.
The per-station branch below reorders each station's own series instead, which cannot miss.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
import pandas as pd

PLACEBO_COLUMN_NAME = "__placebo__"

#: Arity of the placebo arm: how many placebo columns one draw adds. 1 is the historical
#: (and still the default) shape — see `placebo_column_names` and `screen_draw_groups` for
#: why it became a parameter (fps-3jj.14).
DEFAULT_PLACEBO_ARITY = 1

# Contiguous date-block length for the permutation, in date POSITIONS (not calendar days —
# the frame's own sorted unique dates, so a gap in the calendar doesn't shorten a block).
# Chosen above the project's measured cycle length (batch0 `cycle_mean_length`: mean 30.5,
# max 35.3 days) so one block holds at least a whole cycle, and low enough to leave ~58
# blocks on a batch0-sized range. See the module docstring for the fidelity measurements
# behind it.
PLACEBO_BLOCK_DAYS = 61

# A permutation of K blocks only decorrelates as well as K allows: the correlation between a
# series and its block-permuted self is a random quantity whose spread grows as K shrinks, so
# too few blocks makes even a derangement unreliable. `_effective_block_days` shortens the
# block rather than let a short series fall below this.
#
# Set from measurement, not intuition. 500 seeds against a deliberately adversarial level-like
# series (monotone drift plus a 31-day cycle — drift-dominated, the hardest case for any
# reordering), |self-correlation| by block count:
#
#     K     median    p95     p99     max
#      8     0.238   0.642   0.753   0.792
#     12     0.190   0.472   0.652   0.751
#     24     0.145   0.392   0.472   0.555
#     59     0.078   0.244   0.315   0.332      <- batch0's market-wide branch
#
# 24 is the smallest of these whose WORST case over 500 seeds still clears
# `noise_floor.MAX_SELF_CORRELATION` (0.6). At 12 the p99 is already 0.652: roughly one draw
# in a hundred would fail the screen and be substituted away — and it would be a level-like
# column being substituted away, which is precisely the systematic exclusion fps-d7m exists
# to remove, just arriving by a different route. The cost is paid only by short series (this
# binds below 24 * 61 = 1464 positions, so never on batch0's 3572-date market-wide branch,
# and on roughly the thinnest tenth of its stations): batch0's thinnest station, 389 dates,
# drops to a 16-position block. That is less texture preserved, but a station with a short
# history has less texture to preserve, and unreliable decorrelation is the worse defect.
MIN_BLOCKS = 24

# The FIRST 30 per-draw permutation seeds. Retained verbatim from the original construction's
# large-prime pool so every bank of 30 picks or fewer is byte-identical to what this module
# produced before `block_seed` existed; `block_seed` continues the supply past the end of it.
#
# The values only need to be distinct and deterministic — they seed `np.random.default_rng`
# and are no longer shift offsets, which is why the extension below can be plain consecutive
# integers rather than more primes.
PLACEBO_BLOCK_SEED_POOL: tuple[int, ...] = (
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
    149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
)

# Where `block_seed` picks up once `PLACEBO_BLOCK_SEED_POOL` is exhausted. Chosen only to sit
# clear of the pool's own values (all <= 251) so a seed printed in a `noise_floor.json` says
# unambiguously which half of the supply it came from.
PLACEBO_SEED_EXTENSION_BASE = 1_000_000

# Texture's share of a placebo draw's delta variance — the correlation two draws inherit purely
# from being built on the SAME source column (they share that column's values, NaN pattern and
# autocorrelation exactly; only the dates move). This is what makes a bank of reused columns
# worth fewer draws than it has, and `effective_n_draws` below turns it into a number.
#
# **MEASURED, by column, on 2026-08-27 (bd `fps-3jj.23`).** 32 pinned-source draws on batch1 —
# 8 source columns x 4 block seeds, `noise_floor --same-source-column` — grouped BY SOURCE
# COLUMN in a one-way ANOVA, which is the by-column quantity this constant stands for rather
# than a proxy for it: F(7,24) = 0.735, p = 0.65, point estimate -0.071 (i.e. 0), 95% upper
# bound at the one-sided 5% tail **0.274**. Working:
# `experiments/2026-08-27_texture_icc/`.
#
# The value shipped is the UPPER BOUND, not the point estimate. The design resolves down to
# 0.262, so a point estimate of ~0 reads "could not see it", never "it is not there" — an ICC
# of, say, 0.05 is entirely consistent with what was measured, and so is one of 0.25. Use the
# pessimistic end; that is what makes the bar safe rather than merely small.
#
# **What this replaced, and why the replacement is tighter in both directions.** Until now this
# was 0.391: a 95% upper bound on the ICC by texture FAMILY, standing in for the ICC by COLUMN.
# Family is coarser — two different `days_since_trough_entry_<lga>` columns are same-family but
# different-column, while two placebos from the SAME column are more alike than that — so 0.391
# was never established as an upper bound on the quantity actually used, and the direction of
# its error was unknown. The by-column measurement closes that gap: it is the right quantity,
# and it is smaller. Independently, `fps-8o0` was checked against the old artifact at the same
# time: batch1's `network` texture family is a SINGLETON, and `texture_channel.py` dropped it
# from the sums of squares while keeping n=20 and k=5 in the degrees of freedom and in `k_bar`.
# Recomputed correctly the by-family bound is **0.226**, not 0.391 — so the old constant was
# also overstated on its own terms. Two estimators of two different quantities, both under
# 0.391, agreeing on direction.
#
# **What is still assumed.** The 8 pinned columns were chosen by `select_draws`' own even spread
# over the 49 usable columns — sampled the way the bank samples, not curated — so the estimate
# is for the population the bank actually draws from. But their WITHIN-column (alignment)
# variance ran 2.2x the committed 20-draw multi-column floor's TOTAL variance, which under an
# equal-alignment-variance model is impossible. The difference is not resolvable (95% CI on the
# ratio [0.90, 5.18], p = 0.08, and the design cannot call a ratio significant below 0.574), so
# there is no established evidence the pinned set is unrepresentative — but if that lean is
# real, the ANOVA's ICC is biased LOW and 0.274 is optimistic. That is the residual, and it is
# the reason this stays a bound rather than becoming a point estimate.
#
# **It sets every bar, but it only MOVES the wide ones.** Swept across its entire possible
# range on batch1's live ruler with `effective_n_draws` itself (20-draw bank,
# `experiments/2026-08-27_texture_icc/power.py`), the single-candidate bar in c/L:
#
#   arity     ICC 0    0.274      1.0    most it can move
#       1-2  -0.152   -0.152   -0.152    0.000   <- no two draws share a column AT ALL
#       3    -0.152   -0.153   -0.157    0.005
#       4    -0.152   -0.155   -0.163    0.011
#      10    -0.152   -0.165   -0.211    0.059
#      20    -0.152   -0.185   -0.431    0.280
#      35    -0.152   -0.226  no band    0.383
#
# So at every width batch1 has actually run (1 and 3 columns) this constant cannot move a
# verdict — its whole range is worth 0.005 c/L there, an order of magnitude under the realised
# arbiter's decision quantum, and the 0.391 -> 0.274 move itself is worth 0.001. It is
# load-bearing for the WIDE groups docs/routines/generator.md invites, and nowhere else. Say
# that, rather than the bare "it sets every bar".
#
# **Lowering it LOOSENS wide bars, so do not lower it again without a measurement.** Less
# correlation means more effective draws means a narrower band means an easier bar: arity 35
# went from -0.287 to -0.226 c/L with this change. That is the direction that admits false
# positives, which is why the number moved only on a direct measurement of the right quantity
# and why the bound, not the estimate, is what ships. To tighten it further, take more draws:
# the bound is limited by the design's resolution, and 12 columns x 5 seeds (60 draws, ~11.4h)
# would land near 0.23 even if the point estimate came out at 0. `power.py` prices it.
TEXTURE_ICC_BOUND = 0.274


def effective_n_draws(
    draws: list[list[tuple[str, int, float]]], icc: float = TEXTURE_ICC_BOUND
) -> float:
    """How many INDEPENDENT draws a bank is worth, given that draws may share source columns.

    A band's standard deviation is only as good as the independence of the draws behind it, and
    `candidate_sequence` reuses source columns once a bank needs more picks than there are
    columns (fps-3jj.21). Two draws sharing a column share that column's TEXTURE exactly, so
    their deltas are correlated and the bank carries less information than its draw count
    suggests. This converts that into the number the threshold actually needs.

    **This is what replaced the arity CAP (fps-3jj.25).** The cap refused to build a band above
    a fixed arity, which meant the instrument dictated what shape of candidate the generator was
    allowed to propose — a constraint the owner has rejected twice, and rightly: constraining
    candidate shape to suit the ruler distorts the thing being measured. A correction is
    strictly better than a refusal here, because the degradation is gradual and can simply be
    priced in. `family_wise_z_threshold` takes this in place of the nominal draw count, so a
    wider candidate automatically gets a wider, harder bar instead of being turned away.

    Two draws sharing `s` of `arity` columns are charged `icc * s / arity`, not the full `icc`.
    The fraction matters and the binary version is badly wrong: charging any overlap the full
    ICC put a 20-draw arity-10 bank at 3.2 effective draws, which is what made a cap look
    necessary. Scaled by share it is 10.81, and the bar degrades gently enough that no cap is
    needed at all — arity 35 lands at 4.31 effective draws and a -0.226 c/L single-candidate bar
    against arity 1's -0.152, where the pre-fps-3jj.21 construction could not produce a band
    above arity 17 at any price. (Every figure in this paragraph is quoted at the live
    `TEXTURE_ICC_BOUND`. At the 0.391 shipped before `fps-3jj.23` measured it they read 2.4 /
    9.04 / 3.23 / -0.287 — the argument is unchanged, the numbers all moved.)

    Computed from the ACTUAL bank rather than from a model of it: `screen_draw_groups` may
    substitute a column that fails the self-correlation screen, so which draws really overlap is
    a property of the bank that got built, not of `n_draws` and `arity`.

    Returns the nominal count exactly when no two draws share a column, so a bank small enough
    to need no reuse (any arity where `n_draws * arity <= n_columns`) is unaffected — arity 1
    and 2 at 20 draws included.
    """
    if not draws:
        raise ValueError("cannot compute effective draws for an empty bank")
    n = len(draws)
    if n < 2:
        return float(n)
    columns = [{col for col, _seed, _corr in draw} for draw in draws]
    pair_rhos = [
        icc * len(a & b) / max(len(a), 1)
        for a, b in itertools.combinations(columns, 2)
    ]
    rho_bar = sum(pair_rhos) / len(pair_rhos)
    return n / (1.0 + (n - 1) * rho_bar)


def block_seed(index: int) -> int:
    """The block seed at flat position `index` of a bank, for any `index >= 0`.

    Replaces indexing `PLACEBO_BLOCK_SEED_POOL` directly, which could only supply 30 seeds and
    WRAPPED when it ran out (fps-3jj.20). The wrap was not a mild degradation: a block seed
    selects a rearrangement of the date blocks, so two placebos built with the same seed get
    the IDENTICAL rearrangement, which destroys each column's alignment to the target while
    leaving the two columns' alignment TO EACH OTHER untouched. Applied to two source columns
    that were already near-duplicates, the pair comes out of the shuffle as correlated as it
    went in. That is live in batch1's committed grading ruler right now — `candidate_pool`'s
    fallback tail restarted the seed cycle at index 0, so draw 10 (built entirely of
    substitutes) reuses draw 1's seeds 97/101/103 and lands at placebo correlations
    0.965/0.778/0.765 against it. Ten draws, one of which is a near-copy of another.
    Deterministic, not unlucky: recomputing that floor reproduces it exactly.

    An unbounded, never-restarting supply closes that by construction rather than making it
    less likely — no two picks in a bank can share a rearrangement.

    Consecutive integers are a legitimate extension: `np.random.default_rng` routes an int
    through `SeedSequence`, which mixes it thoroughly, so adjacent seeds give well-separated
    streams. NumPy documents this explicitly; the pool's primes were only ever primes because
    the superseded circular-shift construction used the value as a SHIFT OFFSET, where being
    coprime to the series length mattered. It is a seed now.
    """
    if index < 0:
        raise ValueError(f"block seed index must be >= 0, got {index}")
    if index < len(PLACEBO_BLOCK_SEED_POOL):
        return PLACEBO_BLOCK_SEED_POOL[index]
    return PLACEBO_SEED_EXTENSION_BASE + index


def axis_is_market_wide(df: pd.DataFrame, column: str) -> bool:
    """True if `column` holds one value per `price_date`, shared by every row on that date.

    Detected from the data, not a hand-maintained list of "which columns are market-wide" —
    that list would silently drift the moment `fuel_signal/features.py` adds a group, exactly
    the class of defect `docs/CONVENTIONS.md`'s "declared, never discovered" rule warns about
    for the baseline itself. Here the direction is reversed on purpose: the SET of locked
    columns must be declared (`fuel_signal.features.LOCKED_FEATURE_COLUMNS`), but which AXIS
    each one varies on is a structural property of the data that can be checked directly,
    cheaply, and correctly every time — nothing is lost by discovering it.
    """
    return bool(df.groupby("price_date")[column].nunique(dropna=False).le(1).all())


def _effective_block_days(block_days: int, n_positions: int) -> int:
    """`block_days` reduced to a length that still yields at least `MIN_BLOCKS` blocks over
    `n_positions` ordered positions.

    Short series are real here, not hypothetical: batch0's thinnest station has 389 dates
    against a 3572-date range, so a fixed 61-position block would leave it 7 blocks. Shrinking
    the block is the right trade for those — a station with a short history has less texture
    to preserve in the first place, and an unreliable permutation would be the worse defect.
    Directly analogous to the reduction the previous circular-shift construction applied
    per-station for the same reason.

    Raises if the series cannot support `MIN_BLOCKS` blocks even at a block of 1 — that means
    the batch's own date range is too small for a reorder to mean anything, not a fixable
    input.
    """
    if n_positions < MIN_BLOCKS:
        raise ValueError(
            f"cannot block-permute over {n_positions} position(s) — need at least "
            f"{MIN_BLOCKS} so there is one block per position at minimum."
        )
    return max(1, min(block_days, n_positions // MIN_BLOCKS))


def _block_derangement(n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of `range(n_blocks)` with NO fixed point.

    A fixed block leaves that whole stretch of dates holding its own original values at its
    own original dates — for `PLACEBO_BLOCK_DAYS` positions the placebo would be the source
    column exactly. A uniform random permutation fixes one block on average, so this is the
    common case, not a rare one. Rejection-sampling a derangement is cheap (the fraction of
    permutations that are derangements tends to 1/e ≈ 0.37, so this succeeds in ~3 draws);
    the rotation fallback exists only so the function is total for tiny `n_blocks`, where a
    rotation IS a derangement.
    """
    if n_blocks < 2:
        raise ValueError(f"need at least 2 blocks to permute, got {n_blocks}")
    identity = np.arange(n_blocks)
    for _ in range(100):
        order = rng.permutation(n_blocks)
        if not np.any(order == identity):
            return order
    return np.roll(identity, 1)


def _permute_blocks(values: np.ndarray, block_days: int, rng: np.random.Generator) -> np.ndarray:
    """`values` (in time order) cut into contiguous blocks and reassembled in deranged order.

    The value multiset is preserved exactly by construction — same values, same distribution,
    same within-block autocorrelation — while which date each block's values land on is
    reassigned. A trailing partial block (when the series length isn't a multiple of
    `block_days`) participates in the permutation like any other, so no tail is left pinned
    in place; the reassembled array is the same length as the input because concatenating a
    permutation of the same blocks cannot change the total.
    """
    effective = _effective_block_days(block_days, len(values))
    starts = range(0, len(values), effective)
    blocks = [values[s:s + effective] for s in starts]
    order = _block_derangement(len(blocks), rng)
    return np.concatenate([blocks[j] for j in order])


def make_placebo_series(df: pd.DataFrame, source_column: str, block_seed: int) -> pd.Series:
    """A placebo column: `source_column`'s own real values with contiguous date-blocks
    reordered, along whichever axis it actually varies on, reindexed to `df.index`.

    Market-wide column (`axis_is_market_wide`): permute the blocks of the deduplicated,
    date-sorted series of per-date values, then broadcast each value back onto every row
    sharing that date. Every row on one date still shares one value afterwards — the
    market-wide invariant holds throughout, so this can never collapse into the
    within-date-permutation no-op.

    Station-varying column: permute EACH STATION's own date-sorted series independently
    (grouped by `station_code`), then reassemble in the original row order. Preserves each
    station's own distribution and local autocorrelation; destroys the date's alignment to
    that date's true label without mixing one station's history into another's. Each station
    draws from the same seeded generator, so a station's permutation is deterministic given
    `block_seed` and the group order, but is not shared across stations — see the module
    docstring for why the shared-date-map alternative was rejected.
    """
    rng = np.random.default_rng(block_seed)
    if axis_is_market_wide(df, source_column):
        dates = np.sort(df["price_date"].unique())
        values = df.drop_duplicates("price_date").set_index("price_date").loc[dates, source_column]
        permuted = _permute_blocks(values.to_numpy(), PLACEBO_BLOCK_DAYS, rng)
        return df["price_date"].map(pd.Series(permuted, index=dates)).rename(PLACEBO_COLUMN_NAME)

    out = pd.Series(index=df.index, dtype=float, name=PLACEBO_COLUMN_NAME)
    for _station, group in df.groupby("station_code", sort=False):
        ordered = group.sort_values("price_date")
        out.loc[ordered.index] = _permute_blocks(
            ordered[source_column].to_numpy(), PLACEBO_BLOCK_DAYS, rng
        )
    return out


def select_draws(baseline_columns: list[str], n_draws: int) -> list[tuple[str, int]]:
    """`n_draws` deterministic (source_column, block_seed) pairs, spread evenly across
    `baseline_columns` so the bank samples both market-wide and station-varying axes rather
    than clustering in whichever group happens to sort first, and each paired with a distinct
    seed from `block_seed` at its own position. Those are `PLACEBO_BLOCK_SEED_POOL`'s values
    for the first 30 positions and the unbounded extension past that — so a request larger
    than the historical pool is served rather than refused (fps-3jj.21).

    One EVEN SPREAD is by definition at most one pick per column, so this still raises above
    `len(baseline_columns)`. That is no longer a ceiling on a bank, though — `candidate_pool`
    lays further laps beyond this one, and the seed supply (`block_seed`) is unbounded, so a
    caller wanting more picks than there are columns asks `candidate_pool`, not this function.
    """
    n_cols = len(baseline_columns)
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    if n_draws > n_cols:
        raise ValueError(
            f"n_draws ({n_draws}) exceeds the baseline column count ({n_cols}) — one even "
            "spread holds at most one pick per column. Use candidate_pool() for a longer "
            "sequence; it lays additional laps over the same columns."
        )
    indices = [round(i * n_cols / n_draws) % n_cols for i in range(n_draws)]
    # round() can collide on small n_cols/n_draws ratios; nudge forward to the next unused
    # index rather than silently drawing the same column twice.
    seen: set[int] = set()
    deduped: list[int] = []
    for idx in indices:
        while idx in seen:
            idx = (idx + 1) % n_cols
        seen.add(idx)
        deduped.append(idx)
    return [
        (baseline_columns[idx], block_seed(i))
        for i, idx in enumerate(deduped)
    ]


def candidate_sequence(
    baseline_columns: list[str], n_primary: int
) -> Iterator[tuple[str, int]]:
    """The candidate ordering `screen_draws` and `screen_draw_groups` consume, UNBOUNDED.

    **Lap 0 is exactly what `candidate_pool` used to return in full**: `select_draws`'s even
    spread over `n_primary` picks (the primaries) followed by every remaining baseline column
    in natural order (the fallback tail, used when a primary fails the `self_correlation`
    screen). Those two pieces together are a permutation of the whole column list — one
    complete lap.

    Past the end of lap 0 it lays further laps, each the column list rotated one position
    further so a lap boundary does not reproduce the previous lap's adjacency. Columns
    therefore REPEAT down a long sequence, as uniformly as possible: over `m` picks from `n`
    columns every column is used `floor(m/n)` or `ceil(m/n)` times. Every pick, in every lap,
    gets its own seed off a flat counter — `block_seed(i)` at position `i` — which never
    restarts.

    Two things changed here and both are deliberate (fps-3jj.21):

    - **Repeats are allowed at all.** Requiring a distinct source column per draw is what
      capped a bank at `floor(n_columns / arity)` draws and made the bar explode with arity,
      and it buys much less than it looks like it does: two placebos built from the same
      column under DIFFERENT seeds are about as uncorrelated as two built from different
      columns (measured on batch1: median 0.038 against 0.015, both far under the 0.6 the
      screen already tolerates on a placebo's correlation to its own source). The residual
      cost is real, bounded, and PRICED IN rather than capped: `effective_n_draws` turns
      the resulting overlap into the draw count the threshold actually uses.
    - **Seeds come off a flat counter that never restarts**, where the fallback tail used to
      restart the cycle at index 0. That restart is what put batch1's substituted draw 10 on
      draw 1's own seeds — see `block_seed` for why a shared seed is the one thing that
      genuinely destroys a bank's independence. Fixing it also resolves bd `fps-3jj.20`.

    **Unbounded rather than "n_picks plus a lap of headroom"** (review finding, PR #335). A
    fixed headroom is not a substitution guarantee: a batch where a large minority of columns
    fail the screen can exhaust it while `arity` perfectly good columns are still passing, and
    the caller then sees "this batch cannot support this many draws" when it plainly can. That
    is the same class of defect as the empty fallback tail fps-3jj.21 exists to remove.
    Laps are also not merely repetition here — a column that failed the screen at its lap-0
    seed gets a DIFFERENT rearrangement next lap and may pass, which is the per-column retry
    the old position-assigned construction never had. Callers are responsible for their own
    stopping rule; `_assemble_draws` uses "a full lap with nothing passing".
    """
    n_cols = len(baseline_columns)
    if n_cols == 0:
        raise ValueError("baseline_columns is empty — nothing to draw placebos from.")
    if n_primary < 1:
        raise ValueError(f"n_primary must be >= 1, got {n_primary}")

    primary = [c for c, _ in select_draws(baseline_columns, min(n_primary, n_cols))]
    primary_columns = set(primary)
    lap_zero = primary + [c for c in baseline_columns if c not in primary_columns]

    index = 0
    for column in lap_zero:
        yield (column, block_seed(index))
        index += 1
    lap = 1
    while True:
        for i in range(n_cols):
            yield (baseline_columns[(i + lap) % n_cols], block_seed(index))
            index += 1
        lap += 1


def candidate_pool(baseline_columns: list[str], n_picks: int) -> list[tuple[str, int]]:
    """The first `n_picks + len(baseline_columns)` entries of `candidate_sequence` as a list —
    the picks themselves plus one lap of substitution headroom.

    Retained as the readable, finite view of the sequence (and what the tests assert the lap
    structure against). The screening path consumes `candidate_sequence` directly, so it is
    NOT limited to this much headroom — see that function.
    """
    return list(itertools.islice(candidate_sequence(baseline_columns, n_picks), n_picks + len(baseline_columns)))


def screen_draws(
    df: pd.DataFrame, baseline_columns: list[str], n_draws: int, *, max_self_correlation: float,
) -> list[tuple[str, int, float]]:
    """`n_draws` (source_column, block_seed, self_correlation) triples, each verified —
    BEFORE any backtest fit — to pass `abs(self_correlation) <= max_self_correlation`.

    Screening the whole candidate list (`candidate_pool`) up front and substituting on a
    violation, rather than committing to `select_draws`'s fixed list, keeps a failure cheap
    (a correlation check, not a wasted fit) and means the computation does not abort just
    because one of the first `n_draws` candidates happens to be unusable. Under the previous
    circular-shift construction this substitution was load-bearing on every batch0 run —
    eight columns failed deterministically for CORRELATION (see the module docstring). Block
    permutation passes all 49 usable batch0 columns, so no correlation-driven substitution
    fires any more. The tail is still exercised on every batch0 run by the OTHER skip
    condition: two of the 20 primaries land on all-NaN columns, correlate to NaN, and are
    substituted. It is also kept because a column that a permutation cannot decorrelate still
    exists in principle (the between-station floor named in the module docstring), and a batch
    whose panel or date range differs from batch0's has not been screened.

    Uses `pd.Series.corr` (pairwise-complete — see the module docstring for why this isn't
    `np.corrcoef`), so a column with partial NaN is graded on its real, non-NaN correlation,
    not masked as an unverifiable NaN. An all-NaN column correlates to NaN and is skipped.

    Raises ValueError when a FULL LAP passes with nothing clearing the screen — one
    presentation of every baseline column, each at a fresh seed. The candidate sequence is
    unbounded (fps-3jj.21), so "the pool ran out" is no longer the failure mode; this one is a
    genuine "this batch's data cannot support these draws at this threshold", rare but real,
    and worth surfacing clearly rather than silently returning fewer draws than asked for. See
    `_assemble_draws` for why the count is of CONSECUTIVE failures rather than total ones.
    """
    groups = _assemble_draws(
        df, baseline_columns, n_draws, arity=1, max_self_correlation=max_self_correlation
    )
    return [triple for group in groups for triple in group]


def _assemble_draws(
    df: pd.DataFrame,
    baseline_columns: list[str],
    n_draws: int,
    *,
    arity: int,
    max_self_correlation: float,
) -> list[list[tuple[str, int, float]]]:
    """`n_draws` groups of `arity` screened triples, consumed off `candidate_sequence`.

    Shared by `screen_draws` (arity 1) and `screen_draw_groups`, so the two cannot drift —
    the arity-1 identity between them is structural rather than something the tests have to
    keep true.

    Candidates are screened LAZILY: a placebo series is not free to build (at batch1's size
    each is a permutation over ~2.1M rows), and both callers stop early.

    **Stopping rule: a full lap with nothing passing.** `candidate_sequence` is unbounded, so
    something has to decide when a batch genuinely cannot supply the draws asked for. A whole
    lap is exactly one presentation of every column, each at a fresh seed; if not one of them
    clears the screen, the next lap is offering the same columns again and the batch's data is
    the problem, not the sequence's length. Counting CONSECUTIVE failures rather than total
    ones is what keeps a batch with a few permanently-unusable columns (batch1 has five
    all-NaN `days_since_trough_entry_<lga>` columns, skipped as unverifiable on every lap)
    from being refused a bank it can perfectly well supply.
    """
    sequence = candidate_sequence(baseline_columns, n_draws * arity)
    groups: list[list[tuple[str, int, float]]] = []
    current: list[tuple[str, int, float]] = []
    used: set[str] = set()
    tried = 0
    consecutive_failures = 0

    for source_column, seed in sequence:
        if consecutive_failures >= len(baseline_columns):
            break
        tried += 1
        if source_column in used:
            # Would repeat a column already in the group being assembled. Not a screen
            # failure — it says nothing about the batch's data — so it must not count toward
            # the stopping rule.
            continue
        placebo_series = make_placebo_series(df, source_column, seed)
        corr = placebo_series.corr(df[source_column])
        if pd.isna(corr) or abs(corr) > max_self_correlation:
            consecutive_failures += 1
            continue
        consecutive_failures = 0
        current.append((source_column, seed, float(corr)))
        used.add(source_column)
        if len(current) == arity:
            groups.append(current)
            current, used = [], set()
            if len(groups) == n_draws:
                return groups

    raise ValueError(
        f"only assembled {len(groups)}/{n_draws} placebo draws at arity {arity} passing "
        f"abs(self_correlation) <= {max_self_correlation} after trying {tried} candidates "
        f"and {consecutive_failures} consecutive failures (a full pass over all "
        f"{len(baseline_columns)} baseline columns with none passing) — this batch's data "
        "does not support this many draws at this arity and threshold."
    )


#: How many CONSECUTIVE fresh block seeds a pinned source column may fail the self-correlation
#: screen at before `screen_pinned_column_draws` gives up on it. The multi-column path's
#: stopping rule is "one full presentation of every column, each at a fresh seed"
#: (`_assemble_draws`); a pinned bank has only one column, so the analogue is one full
#: presentation of the historical seed supply. A column that cannot be decorrelated at 30
#: different rearrangements is the NAMED LIMITATION in this module's docstring (a mostly
#: cross-sectional column keeps an irreducible correlation no time-axis reorder removes), not
#: bad luck — and there is nothing else to try, because substituting a different column is the
#: one thing a pinned bank must never do.
PINNED_COLUMN_SEED_ATTEMPTS = len(PLACEBO_BLOCK_SEED_POOL)


def screen_pinned_column_draws(
    df: pd.DataFrame,
    source_columns: list[str],
    n_draws: int,
    *,
    max_self_correlation: float,
) -> list[list[tuple[str, int, float]]]:
    """`n_draws` arity-1 draws built from `source_columns` ONLY, cycling them round-robin so
    every column gets `n_draws // k` or `n_draws // k + 1` draws, each at its own fresh seed.

    This is the ICC MEASUREMENT bank (bd `fps-3jj.23`), not a grading ruler, and the two must
    not be confused — `noise_floor.compute_noise_floor` stamps it with a different
    `null_method` so `dossier_tables._noise_band` refuses to grade anything against it. Its
    whole point is the thing that disqualifies it as a ruler: draws deliberately REPEAT a
    source column, so their deltas share texture exactly and the band is narrower than a real
    one by however much texture contributes.

    Why that measures the ICC. A placebo's delta moves through two channels — ALIGNMENT (did
    this particular rearrangement happen to line up with the target?) which varies with the
    block seed, and TEXTURE (how much does a column OF THIS SHAPE perturb the fit at all?)
    which is fixed by the source column. Two draws from the same column share texture
    identically, so the correlation they inherit from sharing a column IS the quantity
    `effective_n_draws` prices. Grouping these draws BY COLUMN and running a one-way ANOVA
    gives it directly: between-column variance is texture, within-column is alignment.

    **Substitution is refused, not silent.** `_assemble_draws` answers a screen failure by
    moving to the next candidate column, which is right for a ruler (one unusable column must
    not abort a calibration) and catastrophic here — a substituted column would quietly turn a
    same-column pair into a different-column pair and bias the measured ICC toward zero, in
    the direction that says "no problem". A pinned column that fails is retried at fresh seeds
    (a different rearrangement may well pass) up to `PINNED_COLUMN_SEED_ATTEMPTS`, and then the
    whole call raises.

    Arity is 1 by construction and cannot be raised: a draw needs `arity` DISTINCT source
    columns (`screen_draw_groups`), so "arity k, all from one column" does not exist. That is
    not a limitation of this function — the ICC is a property of one column's texture, and one
    column is what a draw here is made of.
    """
    if not source_columns:
        raise ValueError("source_columns is empty — a pinned bank needs at least one column.")
    duplicates = {c for c in source_columns if source_columns.count(c) > 1}
    if duplicates:
        raise ValueError(
            f"source_columns repeats {sorted(duplicates)} — the round-robin already gives every "
            "column multiple draws; listing one twice would silently double its weight in the "
            "by-column ANOVA."
        )
    missing = [c for c in source_columns if c not in df.columns]
    if missing:
        raise ValueError(f"source_columns not present in the batch frame: {missing}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")

    groups: list[list[tuple[str, int, float]]] = []
    # One flat seed counter across the WHOLE bank, exactly as `candidate_sequence` does, so no
    # two draws anywhere in it can share a rearrangement — the defect `block_seed` exists to
    # close. Round-robin rather than column-by-column so a bank truncated early (or one whose
    # n_draws is not a multiple of k) stays as close to balanced as it can be, which is what
    # the ANOVA's power depends on.
    index = 0
    for i in range(n_draws):
        source_column = source_columns[i % len(source_columns)]
        for _attempt in range(PINNED_COLUMN_SEED_ATTEMPTS):
            seed = block_seed(index)
            index += 1
            corr = make_placebo_series(df, source_column, seed).corr(df[source_column])
            if not pd.isna(corr) and abs(corr) <= max_self_correlation:
                groups.append([(source_column, seed, float(corr))])
                break
        else:
            raise ValueError(
                f"pinned source column {source_column!r} failed abs(self_correlation) <= "
                f"{max_self_correlation} at all {PINNED_COLUMN_SEED_ATTEMPTS} consecutive seeds "
                f"tried for draw {i + 1}/{n_draws} (an all-NaN column correlates to NaN and "
                "fails the same way). A pinned bank must NOT substitute another column — that "
                "would turn a same-column pair into a different-column pair and bias the "
                "measured ICC toward zero. Pick a different source column and re-run."
            )
    return groups


def placebo_column_names(arity: int) -> list[str]:
    """The `arity` column names one placebo draw adds to the candidate arm.

    Arity 1 returns the historical single name (`PLACEBO_COLUMN_NAME`) rather than
    `__placebo_1__`. A column's NAME cannot change a LightGBM fit — splits are chosen by
    feature index, and the placebo is appended last either way — so renaming it would be a
    no-op BY DERIVATION. That is exactly the kind of claim this repo has been bitten by
    asserting instead of executing (see the run-dry clamp's "no-op by derivation ... but
    that has not been EXECUTED" in experiments/ledger.yaml), and batch1's committed
    20-draw floor is the ruler its whole verdict rests on. Keeping the arity-1 name
    byte-identical costs one branch and removes the question.
    """
    if arity < 1:
        raise ValueError(f"arity must be >= 1, got {arity}")
    if arity == 1:
        return [PLACEBO_COLUMN_NAME]
    return [f"__placebo_{i}__" for i in range(1, arity + 1)]


def screen_draw_groups(
    df: pd.DataFrame,
    baseline_columns: list[str],
    n_draws: int,
    *,
    arity: int = DEFAULT_PLACEBO_ARITY,
    max_self_correlation: float,
) -> list[list[tuple[str, int, float]]]:
    """`n_draws` groups of `arity` screened (source_column, block_seed, self_correlation)
    triples — one group per draw, columns distinct WITHIN a draw.

    **Arity 1 is exactly `screen_draws`**, one group per triple: same call, same screening
    order, same substitutions. That identity is deliberate and is what makes an existing
    arity-1 floor reproducible after this function landed.

    For arity k it consumes screened candidates in `candidate_pool` order, assembling groups
    of k and skipping a pick that would repeat a column already in the group being built.
    Two consequences worth stating rather than discovering:

    - **A group's members are ADJACENT in `candidate_pool`'s order.** `select_draws` spreads
      its picks evenly across the column list by position, so a group of 3 tends to be
      three columns from the same feature GROUP (three `cycle_*` columns, or three
      `days_since_trough_entry_<lga>` columns). That is a feature, not an accident: a real
      multi-column candidate is one MECHANISM, and its members are usually related
      (docs/routines/generator.md, "A candidate is one MECHANISM"). A null built from
      three unrelated columns would be a weaker analogue of the thing being measured.
    - **`n_draws` no longer trades off against `arity`** (fps-3jj.21). It used to: both pools
      bound `n_draws * arity`, capping a bank at `min(floor(n_cols / k), floor(30 / k))`
      draws, which meant a wider candidate was graded against a THINNER band — and since
      `family_wise_z_threshold` puts `df = n_draws - 1` into its t-critical, the bar exploded
      with arity for a reason that had nothing to do with arity. Two thirds of the k=1 -> k=3
      bar move measured in `experiments/2026-08-23_placebo_arity/` was draw count, not the
      arity effect it was attributed to. Both pools are unbounded now, so the draw count is a
      free parameter again — pick it for the certainty wanted and pay the compute. Arity is
      not capped either: reuse is priced by `effective_n_draws`, so a wider candidate gets a
      wider, harder bar rather than being turned away (fps-3jj.25).

    What is guaranteed is DISTINCT SOURCE COLUMNS WITHIN A DRAW, and a distinct block seed
    for every column in the whole bank. Distinctness ACROSS draws is deliberately not
    guaranteed — see `candidate_sequence` for the measurement behind that, and
    `effective_n_draws` for how the resulting overlap is priced into the bar.

    **No arity cap** (fps-3jj.25). An earlier revision refused above a `MAX_RULER_ARITY`
    constant, which made the instrument dictate what shape of candidate the generator was
    allowed to propose — a constraint the owner has rejected twice, and rightly: constraining
    candidate shape to suit the ruler distorts the thing being measured. Reuse degrades a bank
    GRADUALLY, so it can be priced rather than forbidden, and pricing it is strictly more
    informative than a refusal. The only arity limit left is physical: a draw needs `arity`
    DISTINCT source columns, so it cannot exceed the baseline column count.
    """
    if arity < 1:
        raise ValueError(f"arity must be >= 1, got {arity}")
    if arity > len(baseline_columns):
        raise ValueError(
            f"arity ({arity}) exceeds the baseline column count ({len(baseline_columns)}) — "
            "one draw needs that many DISTINCT source columns."
        )

    return _assemble_draws(
        df, baseline_columns, n_draws, arity=arity,
        max_self_correlation=max_self_correlation,
    )


def make_placebo_frame(
    df: pd.DataFrame, draw: list[tuple[str, int, float]]
) -> pd.DataFrame:
    """One draw's placebo columns as a frame indexed like `df`, named by `placebo_column_names`.

    Each member is built by `make_placebo_series` from its OWN source column and its OWN
    block seed, so the members are not permuted in lockstep: two columns sharing a seed
    would share a block ORDER, which would make the pair as correlated as their sources
    are, for a reason that has nothing to do with how a real candidate's columns relate.
    """
    names = placebo_column_names(len(draw))
    return pd.DataFrame(
        {
            name: make_placebo_series(df, source_column, block_seed).rename(name)
            for name, (source_column, block_seed, _corr) in zip(names, draw, strict=True)
        },
        index=df.index,
    )
