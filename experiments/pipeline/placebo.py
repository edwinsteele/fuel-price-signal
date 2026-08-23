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

# Per-draw permutation seeds. Kept as the original construction's large-prime pool: the
# values only need to be distinct and deterministic (they seed `np.random.default_rng`, they
# are no longer shift offsets), and reusing the same pool keeps `select_draws`'s "one integer
# per draw" shape unchanged.
PLACEBO_BLOCK_SEED_POOL: tuple[int, ...] = (
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
    149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
)


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
    seed from `PLACEBO_BLOCK_SEED_POOL`.

    Raises ValueError if `n_draws` exceeds either pool — both are sized generously above the
    ~20-draw default (fps-awz), so this only fires on a deliberately large override.
    """
    n_cols = len(baseline_columns)
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    if n_draws > n_cols:
        raise ValueError(f"n_draws ({n_draws}) exceeds the baseline column count ({n_cols}).")
    if n_draws > len(PLACEBO_BLOCK_SEED_POOL):
        raise ValueError(
            f"n_draws ({n_draws}) exceeds the block-seed pool size ({len(PLACEBO_BLOCK_SEED_POOL)})."
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
        (baseline_columns[idx], PLACEBO_BLOCK_SEED_POOL[i])
        for i, idx in enumerate(deduped)
    ]


def candidate_pool(baseline_columns: list[str], n_draws: int) -> list[tuple[str, int]]:
    """An ordered candidate sequence for `screen_draws`: `select_draws`'s own `n_draws`
    (evenly spread, distinct seeds) FIRST, then every remaining baseline column as a
    fallback tail (natural order, seeds cycling through `PLACEBO_BLOCK_SEED_POOL`) — used
    when a primary candidate fails the `self_correlation` screen and a substitute is
    needed. Deliberately longer than `n_draws`; `screen_draws` stops consuming it once it
    has `n_draws` PASSING candidates, so the fallback tail is only touched on a real
    failure — the common case (nothing fails) sees exactly `select_draws`'s own output and
    behaves identically to before this existed.
    """
    primary = select_draws(baseline_columns, n_draws)
    primary_columns = {c for c, _ in primary}
    remaining = [c for c in baseline_columns if c not in primary_columns]
    fallback = [
        (c, PLACEBO_BLOCK_SEED_POOL[i % len(PLACEBO_BLOCK_SEED_POOL)])
        for i, c in enumerate(remaining)
    ]
    return primary + fallback


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

    Raises ValueError if the full candidate pool is exhausted before finding `n_draws` valid
    candidates — a genuine "this batch cannot support this many draws at this threshold"
    case, rare but real, and worth surfacing clearly rather than silently returning fewer
    draws than asked for.
    """
    candidates = candidate_pool(baseline_columns, n_draws)
    screened: list[tuple[str, int, float]] = []
    for source_column, block_seed in candidates:
        if len(screened) >= n_draws:
            break
        placebo_series = make_placebo_series(df, source_column, block_seed)
        corr = placebo_series.corr(df[source_column])
        if pd.isna(corr) or abs(corr) > max_self_correlation:
            continue
        screened.append((source_column, block_seed, float(corr)))
    if len(screened) < n_draws:
        raise ValueError(
            f"only found {len(screened)}/{n_draws} placebo draws passing "
            f"abs(self_correlation) <= {max_self_correlation} after trying all "
            f"{len(candidates)} candidates — this batch's data may not support this many "
            "draws at this threshold."
        )
    return screened


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
    triples — one group per draw, every column distinct across the whole bank.

    **Arity 1 is exactly `screen_draws`**, one group per triple: same call, same screening
    order, same substitutions. That identity is deliberate and is what makes an existing
    arity-1 floor reproducible after this function landed.

    For arity k it screens `n_draws * k` candidates in one pass and chunks them
    consecutively. Two consequences worth stating rather than discovering:

    - **Each group's members are ADJACENT in `baseline_columns`.** `select_draws` spreads
      its picks evenly across the column list by position, so a group of 3 tends to be
      three columns from the same feature GROUP (three `cycle_*` columns, or three
      `days_since_trough_entry_<lga>` columns). That is a feature, not an accident: a real
      multi-column candidate is one MECHANISM, and its members are usually related
      (docs/routines/generator.md, "A candidate is one MECHANISM"). A null built from
      three unrelated columns would be a weaker analogue of the thing being measured.
    - **The pools bind `n_draws * arity`, not `n_draws`.** `select_draws` raises above
      either the baseline column count or `PLACEBO_BLOCK_SEED_POOL`'s size (30), so the
      maximum draw count at arity k is `floor(30 / k)` — 10 at arity 3, which is exactly
      what fps-3jj.14 asks for and leaves no headroom above it. Raised as a plain
      ValueError from `select_draws` with the product in the message.

    What is guaranteed across the bank is DISTINCT SOURCE COLUMNS, not distinct seeds.
    `candidate_pool`'s fallback tail cycles the seed pool from index 0, so once a primary
    is substituted away a later draw can reuse an earlier draw's seed on a different
    column — visible in batch1's own committed arity-1 floor, where seeds 97 and 101 each
    appear twice after two all-NaN LGA primaries were substituted. Pre-existing, unchanged
    by arity, and worth knowing when reading a floor: two draws whose sources are strongly
    correlated AND which share a block order are less independent than the draw count
    suggests.
    """
    if arity < 1:
        raise ValueError(f"arity must be >= 1, got {arity}")
    flat = screen_draws(
        df, baseline_columns, n_draws * arity, max_self_correlation=max_self_correlation
    )
    return [flat[i * arity:(i + 1) * arity] for i in range(n_draws)]


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
