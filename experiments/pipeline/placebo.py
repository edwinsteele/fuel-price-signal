"""Placebo-column construction for the noise floor (fps-awz).

`experiments/pipeline/noise_floor.py` needs a null that measures the SAME operation a real
candidate's `effect_delta_cpl_held` measures: hold seed, data, and folds fixed; add ONE
column. This module builds that column — one that resembles a real feature in distribution
and autocorrelation while having its alignment to the target destroyed, without ever
degrading to a within-date permutation.

Why a within-date permutation is unsafe here: of the 54 locked baseline columns
(`fuel_signal.features.LOCKED_FEATURE_COLUMNS`), the 35 `days_since_trough_entry_<lga>`
columns and the 4 `NETWORK_FEATURE_COLUMNS` are market-wide — every station shares the exact
same value on a given date (`fuel_signal/features.py`'s row-construction loop keys them by
date alone, never by station). Shuffling a market-wide column's values ACROSS STATIONS on
the same date is a no-op: every station already has the same value, so the "shuffled" frame
is identical to the original. fps-3jj's own earlier rejection of a placebo null was of
exactly this failure mode. The fix here is not a smarter shuffle but a different axis
entirely: a CIRCULAR TIME-SHIFT, applied along whichever axis the column actually varies on
(`axis_is_market_wide` decides which, per column, from the data itself rather than a
hand-maintained group list that could drift out of sync with `features.py`).

A circular shift (`np.roll`) preserves the exact value multiset by construction — same
values, same per-station or market-wide distribution, same autocorrelation structure up to
the shift itself — while reassigning which date each value lands on, destroying its
alignment with that date's true label. `shift_days` is drawn from a fixed pool of large
primes: taking `shift_days % n_unique_dates` for the actual roll means a prime can never
silently degenerate to shift 0 against a small date range the way a round number
(e.g. a multiple of 7, which could re-align a weekly pattern) could.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLACEBO_COLUMN_NAME = "__placebo__"

# Large primes, comfortably past any cycle length this project has measured
# (cycle_mean_length sits in the tens of days) and never a multiple of a small
# calendar period (7, 14, 30) by construction. `_effective_shift` reduces one of
# these mod the batch's own unique-date count, so the exact numbers only need to be
# "large and mutually distinct", not tuned to any one batch's date range.
PLACEBO_SHIFT_DAYS_POOL: tuple[int, ...] = (
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


def _effective_shift(shift_days: int, n_unique: int) -> int:
    """`shift_days` reduced into a valid, non-degenerate roll offset for a series of
    `n_unique` distinct positions. Raises if every choice degenerates (n_unique < 2) — that
    means the batch's own date range is too small for a shift to mean anything, not a
    fixable input.
    """
    if n_unique < 2:
        raise ValueError(
            f"cannot build a placebo shift over {n_unique} distinct date(s) — need at least 2."
        )
    effective = shift_days % n_unique
    if effective == 0:
        # A prime pool member can only land on 0 mod n_unique if it EQUALS n_unique (primes
        # are never multiples of anything smaller) — bump by one date position rather than
        # silently shipping a no-op shift.
        effective = 1
    return effective


def make_placebo_series(df: pd.DataFrame, source_column: str, shift_days: int) -> pd.Series:
    """A placebo column: `source_column`'s own real values, circularly time-shifted along
    whichever axis it actually varies on, reindexed to `df.index`.

    Market-wide column (`axis_is_market_wide`): shift the deduplicated, date-sorted series of
    per-date values, then broadcast each shifted value back onto every row sharing that date.
    Every row on one date still shares one value after the shift — the market-wide invariant
    holds throughout, so this can never collapse into the within-date-permutation no-op.

    Station-varying column: shift EACH STATION's own date-sorted series independently
    (grouped by `station_code`), then reassemble in the original row order. Preserves each
    station's own distribution/autocorrelation; destroys the date's alignment to that date's
    true label without mixing one station's history into another's.
    """
    if axis_is_market_wide(df, source_column):
        dates = np.sort(df["price_date"].unique())
        shift = _effective_shift(shift_days, len(dates))
        values = df.drop_duplicates("price_date").set_index("price_date").loc[dates, source_column]
        shifted = pd.Series(np.roll(values.to_numpy(), shift), index=dates)
        return df["price_date"].map(shifted).rename(PLACEBO_COLUMN_NAME)

    out = pd.Series(index=df.index, dtype=float, name=PLACEBO_COLUMN_NAME)
    for _station, group in df.groupby("station_code", sort=False):
        ordered = group.sort_values("price_date")
        shift = _effective_shift(shift_days, len(ordered))
        rolled = np.roll(ordered[source_column].to_numpy(), shift)
        out.loc[ordered.index] = rolled
    return out


def select_draws(baseline_columns: list[str], n_draws: int) -> list[tuple[str, int]]:
    """`n_draws` deterministic (source_column, shift_days) pairs, spread evenly across
    `baseline_columns` so the bank samples both market-wide and station-varying axes rather
    than clustering in whichever group happens to sort first, and each paired with a distinct
    large-prime shift from `PLACEBO_SHIFT_DAYS_POOL`.

    Raises ValueError if `n_draws` exceeds either pool — both are sized generously above the
    ~20-draw default (fps-awz), so this only fires on a deliberately large override.
    """
    n_cols = len(baseline_columns)
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    if n_draws > n_cols:
        raise ValueError(f"n_draws ({n_draws}) exceeds the baseline column count ({n_cols}).")
    if n_draws > len(PLACEBO_SHIFT_DAYS_POOL):
        raise ValueError(
            f"n_draws ({n_draws}) exceeds the shift-day pool size ({len(PLACEBO_SHIFT_DAYS_POOL)})."
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
        (baseline_columns[idx], PLACEBO_SHIFT_DAYS_POOL[i])
        for i, idx in enumerate(deduped)
    ]
