"""Tests for experiments/pipeline/placebo.py — the placebo-column construction for the
reworked noise floor (fps-awz, block permutation per fps-d7m)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.pipeline.placebo import (
    MAX_RULER_ARITY,
    MIN_BLOCKS,
    PLACEBO_BLOCK_DAYS,
    PLACEBO_BLOCK_SEED_POOL,
    PLACEBO_COLUMN_NAME,
    _block_derangement,
    _effective_block_days,
    axis_is_market_wide,
    block_seed,
    candidate_pool,
    candidate_sequence,
    make_placebo_frame,
    make_placebo_series,
    placebo_column_names,
    screen_draw_groups,
    screen_draws,
    select_draws,
)

# MIN_BLOCKS dates minimum — _effective_block_days refuses anything shorter.
DATES = [20260801 + i for i in range(60)]
STATIONS = [100, 200, 300]


def _fixture_df() -> pd.DataFrame:
    """3 stations x 60 dates, one market-wide column (constant per date across every
    station) and one station-varying column (distinct per (station, date))."""
    rows = []
    for date_idx, date in enumerate(DATES):
        for station in STATIONS:
            rows.append(
                {
                    "price_date": date,
                    "station_code": station,
                    "mw_col": float(date_idx),  # depends only on date
                    "station_col": float(station + date_idx * 10),  # depends on both
                }
            )
    return pd.DataFrame(rows)


def test_axis_is_market_wide_true_for_a_per_date_only_column():
    df = _fixture_df()
    assert axis_is_market_wide(df, "mw_col") is True


def test_axis_is_market_wide_false_for_a_station_varying_column():
    df = _fixture_df()
    assert axis_is_market_wide(df, "station_col") is False


def test_effective_block_days_always_leaves_at_least_min_blocks():
    for n in range(MIN_BLOCKS, 4000, 37):
        block = _effective_block_days(PLACEBO_BLOCK_DAYS, n)
        assert block >= 1
        n_blocks = -(-n // block)  # ceil
        assert n_blocks >= MIN_BLOCKS, f"n={n} block={block} gave only {n_blocks} blocks"


def test_effective_block_days_leaves_a_long_series_at_the_declared_block_length():
    assert _effective_block_days(PLACEBO_BLOCK_DAYS, 3572) == PLACEBO_BLOCK_DAYS


def test_effective_block_days_shortens_the_block_for_a_short_series():
    """batch0's thinnest station has 389 dates against a 3572-date range — a fixed 61-position
    block would leave it 7 blocks, below MIN_BLOCKS."""
    assert _effective_block_days(PLACEBO_BLOCK_DAYS, 389) < PLACEBO_BLOCK_DAYS


def test_effective_block_days_raises_below_min_blocks_positions():
    with pytest.raises(ValueError, match="cannot block-permute"):
        _effective_block_days(PLACEBO_BLOCK_DAYS, MIN_BLOCKS - 1)


def test_block_derangement_never_fixes_a_block():
    """A fixed block leaves that whole stretch of dates holding its own original values at
    its own original dates. A uniform permutation fixes one block on average, so this is the
    common case rather than a rare one."""
    for n_blocks in range(2, 40):
        for seed in (1, 7, 97, 251):
            order = _block_derangement(n_blocks, np.random.default_rng(seed))
            assert sorted(order) == list(range(n_blocks))  # still a permutation
            assert not np.any(order == np.arange(n_blocks))


def test_block_derangement_raises_below_two_blocks():
    with pytest.raises(ValueError, match="at least 2 blocks"):
        _block_derangement(1, np.random.default_rng(0))


def test_make_placebo_series_on_market_wide_column_stays_constant_per_date():
    df = _fixture_df()
    placebo = make_placebo_series(df, "mw_col", block_seed=97)
    check = df.assign(**{PLACEBO_COLUMN_NAME: placebo})
    # The market-wide invariant must survive the reorder — this is the whole point of routing
    # a market-wide column through the date-only branch rather than a within-date shuffle.
    assert axis_is_market_wide(check, PLACEBO_COLUMN_NAME) is True


def test_make_placebo_series_on_market_wide_column_preserves_the_value_multiset():
    df = _fixture_df()
    placebo = make_placebo_series(df, "mw_col", block_seed=97)
    original_per_date = sorted(df.drop_duplicates("price_date")["mw_col"])
    permuted_per_date = sorted(placebo.loc[df.drop_duplicates("price_date").index])
    assert original_per_date == permuted_per_date


def test_make_placebo_series_on_market_wide_column_actually_moves_values():
    df = _fixture_df()
    placebo = make_placebo_series(df, "mw_col", block_seed=97)
    # Not identical to the source column — the permutation must have actually moved dates.
    assert not placebo.equals(df["mw_col"].rename(PLACEBO_COLUMN_NAME))


def test_make_placebo_series_on_station_varying_column_preserves_each_stations_own_values():
    df = _fixture_df()
    placebo = make_placebo_series(df, "station_col", block_seed=101)
    check = df.assign(**{PLACEBO_COLUMN_NAME: placebo})
    for station, group in check.groupby("station_code"):
        original = sorted(group["station_col"])
        permuted = sorted(group[PLACEBO_COLUMN_NAME])
        assert original == permuted, f"station {station} lost/gained a value under the permutation"


def test_make_placebo_series_on_station_varying_column_actually_moves_values():
    df = _fixture_df()
    placebo = make_placebo_series(df, "station_col", block_seed=101)
    assert not placebo.equals(df["station_col"].rename(PLACEBO_COLUMN_NAME))


def test_make_placebo_series_returns_aligned_index():
    df = _fixture_df()
    placebo = make_placebo_series(df, "station_col", block_seed=101)
    assert placebo.index.equals(df.index)
    placebo_mw = make_placebo_series(df, "mw_col", block_seed=101)
    assert placebo_mw.index.equals(df.index)


def test_select_draws_returns_n_draws_distinct_columns():
    baseline_columns = [f"col{i}" for i in range(54)]
    draws = select_draws(baseline_columns, 20)
    assert len(draws) == 20
    columns = [c for c, _ in draws]
    assert len(set(columns)) == 20
    assert all(c in baseline_columns for c in columns)


def test_select_draws_is_deterministic():
    baseline_columns = [f"col{i}" for i in range(54)]
    assert select_draws(baseline_columns, 20) == select_draws(baseline_columns, 20)


def test_select_draws_spreads_across_the_full_column_list():
    baseline_columns = [f"col{i}" for i in range(54)]
    draws = select_draws(baseline_columns, 20)
    indices = sorted(baseline_columns.index(c) for c, _ in draws)
    # Not clustered at the front — the last drawn index should be well past the halfway
    # point of a 54-column list when spreading 20 draws evenly.
    assert indices[-1] >= 27


def test_select_draws_uses_distinct_block_seeds():
    baseline_columns = [f"col{i}" for i in range(54)]
    draws = select_draws(baseline_columns, 20)
    seeds = [s for _, s in draws]
    assert len(set(seeds)) == len(seeds)
    assert all(s in PLACEBO_BLOCK_SEED_POOL for s in seeds)


# ── candidate_pool ──────────────────────────────────────────────────────────

def test_candidate_pool_starts_with_select_draws_own_output():
    """The common case (nothing fails screening) must behave identically to plain
    select_draws — candidate_pool only adds a FALLBACK tail beyond it."""
    baseline_columns = [f"col{i}" for i in range(10)]
    pool = candidate_pool(baseline_columns, 6)
    assert pool[:6] == select_draws(baseline_columns, 6)


def test_candidate_pool_lap_zero_is_a_permutation_of_every_column():
    """Lap 0 — select_draws's primaries then every remaining column as the fallback tail —
    is what this function used to return in full, and it must still cover the column list
    exactly once before any repetition starts."""
    baseline_columns = [f"col{i}" for i in range(10)]
    pool = candidate_pool(baseline_columns, 6)
    lap_zero = [c for c, _ in pool[: len(baseline_columns)]]
    primary_columns = {c for c, _ in pool[:6]}
    fallback_columns = lap_zero[6:]
    assert set(fallback_columns) == set(baseline_columns) - primary_columns
    assert sorted(lap_zero) == sorted(baseline_columns)


def test_candidate_pool_returns_the_picks_asked_for_plus_a_lap_of_headroom():
    """The old contract returned exactly len(baseline_columns) entries, so a bank that
    consumed the column list had NO substitution headroom left — fps-3jj.21's failure mode,
    where a screen failure either hard-fails the run after hours of compute or forces keeping
    a column that FAILED the screen (a placebo still correlated with its source, which
    NARROWS the band and makes the bar easier)."""
    baseline_columns = [f"col{i}" for i in range(10)]
    for n_picks in (3, 10, 25):
        pool = candidate_pool(baseline_columns, n_picks)
        assert len(pool) == n_picks + len(baseline_columns)


def test_candidate_pool_extends_past_the_column_list_by_repeating_it_uniformly():
    """The change that removes the draw ceiling: more picks than columns is no longer an
    error, it lays further laps. Reuse is spread as evenly as possible — over m picks from n
    columns every column is used floor(m/n) or ceil(m/n) times — so no column carries more of
    the bank than it has to."""
    import collections

    baseline_columns = [f"col{i}" for i in range(10)]
    n_picks = 25
    picks = [c for c, _ in candidate_pool(baseline_columns, n_picks)][:n_picks]
    counts = collections.Counter(picks)
    assert set(counts) == set(baseline_columns)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_candidate_pool_never_repeats_a_block_seed():
    """fps-3jj.20, and the single most load-bearing assertion in this file.

    The fallback tail used to restart the seed cycle at index 0, so a substituted draw picked
    up an EARLIER draw's seed. A block seed selects a rearrangement of the date blocks, so two
    placebos sharing a seed get the identical rearrangement — which destroys each column's
    alignment to the target while leaving the two columns' alignment TO EACH OTHER intact.
    Applied to two source columns that were already near-duplicates the pair survives the
    shuffle: batch1's committed grading ruler has draw 10 reusing draw 1's seeds 97/101/103
    and landing at placebo correlations 0.965/0.778/0.765 against it — one of ten draws is a
    near-copy of another, deterministically, on every recompute.

    A flat counter that never restarts closes that by construction rather than making it less
    likely, which is why this is asserted over the WHOLE returned sequence, headroom included.
    """
    baseline_columns = [f"col{i}" for i in range(len(PLACEBO_BLOCK_SEED_POOL) + 10)]
    for n_picks in (5, 40, 120):
        seeds = [seed for _c, seed in candidate_pool(baseline_columns, n_picks)]
        assert len(set(seeds)) == len(seeds)


def test_block_seed_extends_the_historical_pool_without_disturbing_it():
    """Every bank of 30 picks or fewer must be byte-identical to what this module produced
    before block_seed existed — committed floors stay reproducible, and fps-3jj.14's
    deliberate arity-1 byte-identity survives for any bank that fits inside the old pool."""
    assert [block_seed(i) for i in range(len(PLACEBO_BLOCK_SEED_POOL))] == list(
        PLACEBO_BLOCK_SEED_POOL
    )
    extension = [block_seed(i) for i in range(len(PLACEBO_BLOCK_SEED_POOL), 200)]
    assert len(set(extension)) == len(extension)
    assert not set(extension) & set(PLACEBO_BLOCK_SEED_POOL)
    with pytest.raises(ValueError, match="must be >= 0"):
        block_seed(-1)


def test_block_seed_extension_gives_well_separated_permutations():
    """Consecutive integers are only a legitimate seed extension because default_rng routes
    them through SeedSequence, which mixes thoroughly. Asserted rather than derived: adjacent
    extension seeds must not produce near-identical placebos."""
    df = _level_like_df()
    base = len(PLACEBO_BLOCK_SEED_POOL)
    a = make_placebo_series(df, "level_col", block_seed(base))
    b = make_placebo_series(df, "level_col", block_seed(base + 1))
    assert abs(a.corr(b)) < 0.4


# ── screen_draws ────────────────────────────────────────────────────────────

def _diverse_fixture_df(n_dates: int, n_cols: int) -> pd.DataFrame:
    """A longer, market-wide-only fixture whose columns are deliberately diverse (not the
    slow/short-series case) so a block permutation decorrelates all of them — used where a
    test needs several REAL candidates that pass screening, not a forced failure."""
    dates = list(range(20260101, 20260101 + n_dates))
    df = pd.DataFrame({"price_date": dates, "station_code": [111] * n_dates})
    for i in range(n_cols):
        df[f"col{i}"] = [float((d * (i + 3) + i) % 23) for d in range(n_dates)]
    return df


def test_screen_draws_happy_path_returns_exactly_n_draws_with_correlations():
    df = _diverse_fixture_df(n_dates=200, n_cols=10)
    baseline_columns = [f"col{i}" for i in range(10)]

    screened = screen_draws(df, baseline_columns, n_draws=5, max_self_correlation=0.9)

    assert len(screened) == 5
    for source_column, seed, corr in screened:
        assert source_column in baseline_columns
        assert seed in PLACEBO_BLOCK_SEED_POOL
        assert pd.notna(corr) and corr <= 0.9


def test_screen_draws_substitutes_a_failing_primary_candidate(monkeypatch):
    """A candidate that fails the correlation screen must be replaced by the next
    candidate in candidate_pool, not abort the whole computation — the fps-awz review
    finding: a slowly-varying column failed at EVERY circular-shift offset, deterministically, so an
    earlier abort-on-first-failure design could not produce a floor on real batch0 data
    at all."""
    import experiments.pipeline.placebo as placebo_module

    df = _diverse_fixture_df(n_dates=200, n_cols=6)
    baseline_columns = [f"col{i}" for i in range(6)]
    real_make_placebo_series = placebo_module.make_placebo_series

    def _fake(frame, source_column, block_seed):
        if source_column == "col2":
            # A no-op reorder — guaranteed correlation 1.0, guaranteed to fail any
            # reasonable threshold.
            return frame[source_column].rename(placebo_module.PLACEBO_COLUMN_NAME)
        return real_make_placebo_series(frame, source_column, block_seed)

    monkeypatch.setattr(placebo_module, "make_placebo_series", _fake)

    screened = screen_draws(df, baseline_columns, n_draws=5, max_self_correlation=0.9)

    columns = {c for c, _, _ in screened}
    assert len(screened) == 5
    assert "col2" not in columns
    # n_draws == n_cols - 1 and exactly one column fails, so the result must be every
    # OTHER column — this holds regardless of primary/fallback index arithmetic.
    assert columns == set(baseline_columns) - {"col2"}


def test_screen_draws_skips_an_all_nan_column():
    df = _fixture_df()
    df["all_nan_col"] = float("nan")
    baseline_columns = ["mw_col", "station_col", "all_nan_col"]

    screened = screen_draws(df, baseline_columns, n_draws=2, max_self_correlation=0.99)

    assert {c for c, _, _ in screened} == {"mw_col", "station_col"}


def test_screen_draws_uses_pairwise_complete_correlation_not_np_corrcoef():
    """Review finding: np.corrcoef returns NaN the instant ANY element is NaN, which would
    mask a column's real, checkable correlation as unverifiable the moment it has even one
    missing value (e.g. lga_mean_cents/brand_mean_cents on real batch0 data, which carry
    partial NaN on dates with too few classified stations — fuel_signal/features.py's own
    documented behaviour). pd.Series.corr is pairwise-complete and must be used instead."""
    n = 200
    df = pd.DataFrame({
        "price_date": list(range(20260101, 20260101 + n)),
        "station_code": [111] * n,
        "ramp_col": [float(i) for i in range(n)],
    })
    df.loc[df.index[::10], "ramp_col"] = float("nan")  # a few, not all, NaN

    placebo = make_placebo_series(df, "ramp_col", block_seed=101)
    np_corr = np.corrcoef(placebo.to_numpy(dtype=float), df["ramp_col"].to_numpy(dtype=float))[0, 1]
    assert np.isnan(np_corr)  # confirms this fixture actually exercises the discrepancy

    screened = screen_draws(df, ["ramp_col"], n_draws=1, max_self_correlation=0.99)
    assert len(screened) == 1
    assert pd.notna(screened[0][2])


def test_screen_draws_raises_when_the_whole_pool_cannot_meet_n_draws(monkeypatch):
    import experiments.pipeline.placebo as placebo_module

    df = _diverse_fixture_df(n_dates=200, n_cols=2)
    baseline_columns = ["col0", "col1"]

    def _always_no_op(frame, source_column, block_seed):
        return frame[source_column].rename(placebo_module.PLACEBO_COLUMN_NAME)

    monkeypatch.setattr(placebo_module, "make_placebo_series", _always_no_op)

    with pytest.raises(ValueError, match=r"only assembled 0/2"):
        screen_draws(df, baseline_columns, n_draws=2, max_self_correlation=0.9)


def test_select_draws_raises_when_n_draws_exceeds_column_count():
    baseline_columns = [f"col{i}" for i in range(5)]
    with pytest.raises(ValueError, match="exceeds the baseline column count"):
        select_draws(baseline_columns, 10)


def test_select_draws_has_no_seed_ceiling():
    """The seed pool was a fossil of the superseded circular-shift construction, where the
    number was a SHIFT OFFSET and had to be coprime to the series length. As RNG seeds the
    values only need to be distinct and deterministic, so a fixed list of 30 was capping bank
    size for no reason — and its wraparound was actively harmful (fps-3jj.20)."""
    baseline_columns = [f"col{i}" for i in range(100)]
    draws = select_draws(baseline_columns, len(PLACEBO_BLOCK_SEED_POOL) + 20)
    seeds = [s for _c, s in draws]
    assert len(set(seeds)) == len(seeds)


def test_select_draws_raises_on_non_positive_n_draws():
    baseline_columns = [f"col{i}" for i in range(5)]
    with pytest.raises(ValueError, match="n_draws must be >= 1"):
        select_draws(baseline_columns, 0)


# ── fps-d7m: level-like columns must survive screening ──────────────────────

def _level_like_df(n_dates: int = 3572) -> pd.DataFrame:
    """A market-wide column that is LEVEL-LIKE: a slow drift plus a slow cycle, the shape
    that defeated the previous circular-shift construction. Stands in for batch0's
    cycle_last_min_cents / lga_mean_cents / station_price_cents, all of which sat at 0.49-0.80
    self-correlation at every offset in the old prime pool."""
    dates = list(range(20260101, 20260101 + n_dates))
    drift = np.linspace(120.0, 190.0, n_dates)
    cycle = 8.0 * np.sin(np.arange(n_dates) * 2 * np.pi / 31.0)
    return pd.DataFrame(
        {"price_date": dates, "station_code": [111] * n_dates, "level_col": drift + cycle}
    )


def test_block_permutation_beats_a_circular_shift_on_a_level_like_column():
    """The whole point of fps-d7m, asserted as the comparison it actually is. A circular
    shift leaves a slowly-drifting column correlated with itself at EVERY offset, because the
    value at date d and at date d + offset are both drawn from the same drift; reordering
    contiguous blocks breaks the 'adjacent in time implies similar value' property that
    causes it. Measured on real batch0 data the gap is 0.49-0.80 (shift, best over all 30
    offsets in the old pool) against 0.02-0.09 (block permutation)."""
    df = _level_like_df()
    source = df["level_col"]

    shift_corrs = [
        abs(pd.Series(np.roll(source.to_numpy(), off)).corr(source))
        for off in PLACEBO_BLOCK_SEED_POOL  # the old construction's own offsets
    ]
    block_corrs = [
        abs(make_placebo_series(df, "level_col", block_seed=seed).corr(source))
        for seed in PLACEBO_BLOCK_SEED_POOL
    ]

    # A typical offset leaves the column heavily correlated with itself — and note the
    # shift's BEST case is not asserted to fail: on real batch0 data six of the eight
    # excluded columns could squeak under 0.6 at the single largest offset in the pool, with
    # the minimum always landing there. That is the tell, not a reprieve: whether a level-like
    # column survived depended on the offset's size relative to the batch's calendar span
    # rather than on anything that makes a placebo a good null, and screen_draws assigns
    # offsets by POSITION rather than searching for a passing one per column.
    assert np.median(shift_corrs) > 0.6
    # Block permutation, by contrast, clears the screen at EVERY seed with room to spare.
    assert max(block_corrs) < 0.4
    assert max(block_corrs) < min(shift_corrs)


def test_a_level_like_column_survives_screening():
    """The failure fps-d7m was filed on: this column could not enter the bank at all under
    the old construction, at any threshold, so the noise band characterized 'adding a
    fast-varying column' rather than 'adding an arbitrary column'."""
    df = _level_like_df()
    screened = screen_draws(df, ["level_col"], n_draws=1, max_self_correlation=0.6)
    assert len(screened) == 1
    assert screened[0][0] == "level_col"


def test_block_permutation_preserves_a_level_like_columns_value_multiset():
    """Decorrelating must not change WHAT the column is — same values, same distribution,
    only their placement in time changes."""
    df = _level_like_df()
    placebo = make_placebo_series(df, "level_col", block_seed=97)
    assert sorted(placebo.to_numpy()) == pytest.approx(sorted(df["level_col"].to_numpy()))


def test_screen_rejects_strong_anti_correlation():
    """A placebo perfectly ANTI-aligned with its source carries exactly as much information
    as one perfectly aligned with it, so the screen tests |corr|, not signed corr."""
    import experiments.pipeline.placebo as placebo_module

    df = _level_like_df()

    def _negated(frame, source_column, block_seed):
        return (-frame[source_column]).rename(placebo_module.PLACEBO_COLUMN_NAME)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(placebo_module, "make_placebo_series", _negated)
        with pytest.raises(ValueError, match=r"only assembled 0/1"):
            screen_draws(df, ["level_col"], n_draws=1, max_self_correlation=0.6)


def test_make_placebo_series_is_deterministic_given_a_block_seed():
    df = _level_like_df()
    a = make_placebo_series(df, "level_col", block_seed=97)
    b = make_placebo_series(df, "level_col", block_seed=97)
    pd.testing.assert_series_equal(a, b)


def test_different_block_seeds_give_different_permutations():
    df = _level_like_df()
    a = make_placebo_series(df, "level_col", block_seed=97)
    b = make_placebo_series(df, "level_col", block_seed=251)
    assert not a.equals(b)


def test_trailing_partial_block_is_not_left_pinned_in_place():
    """A series whose length isn't a multiple of the block length ends in a short block. It
    must participate in the permutation like any other — a pinned tail would leave those
    dates holding their own original values."""
    n = 401
    df = _level_like_df(n)
    block = _effective_block_days(PLACEBO_BLOCK_DAYS, n)
    assert n % block != 0, "fixture must exercise a trailing partial block"
    placebo = make_placebo_series(df, "level_col", block_seed=97)
    assert len(placebo) == n
    tail = slice(n - (n % block), n)
    assert not np.array_equal(
        placebo.to_numpy()[tail], df["level_col"].to_numpy()[tail]
    )


def test_each_station_gets_its_own_permutation():
    """`make_placebo_series`'s docstring states per-station independence as a design property,
    and nothing else in this file would notice if it were lost: every station-varying
    assertion here is per-station (multiset preserved, values moved), so hoisting the
    derangement out of the groupby loop to share ONE block order across all stations would
    keep them all green.

    That refactor would matter. Sharing one order makes every station's placebo move in
    lockstep, which preserves the cross-sectional structure of each date — exactly the axis
    the module's own NAMED LIMITATION already flags as the one a time-axis reorder cannot
    scramble. The null would quietly weaken along its known weak axis."""
    df = _fixture_df()
    placebo = make_placebo_series(df, "station_col", block_seed=97)
    check = df.assign(**{PLACEBO_COLUMN_NAME: placebo})

    # station_col == station_code + date_idx * 10, so (value - station) / 10 recovers which
    # date_idx each slot now holds — i.e. the permutation itself, comparably across stations.
    orders = {}
    for station, group in check.groupby("station_code"):
        ordered = group.sort_values("price_date")
        orders[station] = tuple((ordered[PLACEBO_COLUMN_NAME] - station) // 10)

    assert len(orders) > 1, "fixture must have several stations for this to mean anything"
    for order in orders.values():
        assert sorted(order) == sorted(next(iter(orders.values()))), "not a permutation"
    assert len(set(orders.values())) > 1, (
        "every station received the SAME block order — the permutation is shared across "
        "stations rather than drawn per station"
    )


# ── arity (fps-3jj.14) ────────────────────────────────────────────────────────

def test_placebo_column_names_keeps_the_historical_name_at_arity_1():
    """Not cosmetic: batch1's committed 20-draw floor is the ruler its verdict rests on,
    and 'a column rename cannot change a LightGBM fit' is a derivation, not a measurement.
    Keeping arity 1 byte-identical removes the question."""
    assert placebo_column_names(1) == [PLACEBO_COLUMN_NAME]
    assert placebo_column_names(3) == ["__placebo_1__", "__placebo_2__", "__placebo_3__"]
    with pytest.raises(ValueError, match="arity must be >= 1"):
        placebo_column_names(0)


def test_screen_draw_groups_at_arity_1_is_exactly_screen_draws():
    """The identity that makes an existing arity-1 floor reproducible after this landed."""
    df = _diverse_fixture_df(n_dates=200, n_cols=10)
    baseline_columns = [f"col{i}" for i in range(10)]

    flat = screen_draws(df, baseline_columns, n_draws=5, max_self_correlation=0.9)
    grouped = screen_draw_groups(
        df, baseline_columns, n_draws=5, arity=1, max_self_correlation=0.9
    )

    assert grouped == [[triple] for triple in flat]


def test_screen_draw_groups_keeps_columns_distinct_within_a_draw():
    df = _diverse_fixture_df(n_dates=200, n_cols=20)
    baseline_columns = [f"col{i}" for i in range(20)]

    groups = screen_draw_groups(
        df, baseline_columns, n_draws=4, arity=3, max_self_correlation=0.9
    )

    assert len(groups) == 4
    assert all(len(g) == 3 for g in groups)
    for group in groups:
        sources = [col for col, _seed, _corr in group]
        assert len(set(sources)) == len(sources), "a draw repeated one of its own columns"
    # Every column in the whole bank gets its own block seed, always — that one IS bank-wide
    # (fps-3jj.20). Source-column distinctness is deliberately NOT: see the next test.
    seeds = [seed for g in groups for _col, seed, _corr in g]
    assert len(set(seeds)) == len(seeds)


def test_screen_draw_groups_reuses_source_columns_across_draws_when_it_must():
    """The change that removes the draw ceiling (fps-3jj.21). Requiring a distinct source
    column per draw capped a bank at floor(n_columns / arity) draws, and because
    family_wise_z_threshold carries df = n_draws - 1, a thinner band means a much harder bar
    — for a reason unrelated to arity. Reuse costs little: two placebos from the same column
    under DIFFERENT seeds are about as uncorrelated as two from different columns (batch1
    median 0.038 vs 0.015, both far under the 0.6 the screen already tolerates)."""
    import collections

    df = _diverse_fixture_df(n_dates=200, n_cols=10)
    baseline_columns = [f"col{i}" for i in range(10)]

    groups = screen_draw_groups(
        df, baseline_columns, n_draws=8, arity=3, max_self_correlation=0.9
    )

    assert len(groups) == 8  # 24 picks from 10 columns — impossible under the old contract
    sources = [col for g in groups for col, _seed, _corr in g]
    counts = collections.Counter(sources)
    assert max(counts.values()) > 1, "fixture must actually force reuse"
    assert max(counts.values()) - min(counts.values()) <= 1, "reuse must be spread evenly"


def test_screen_draw_groups_draw_count_no_longer_trades_off_against_arity():
    """The defect fps-3jj.21 was filed on, stated as the thing a caller sees: the SAME draw
    count must be reachable at every gradeable arity. It was not — the pools bound
    n_draws * arity, so a batch got 30 draws at arity 1 and 7 at arity 4, and the bar exploded
    on the collapsing draw count rather than on arity."""
    df = _diverse_fixture_df(n_dates=200, n_cols=12)
    baseline_columns = [f"col{i}" for i in range(12)]

    for arity in range(1, MAX_RULER_ARITY + 1):
        groups = screen_draw_groups(
            df, baseline_columns, n_draws=20, arity=arity, max_self_correlation=0.9
        )
        assert len(groups) == 20
        assert all(len(g) == arity for g in groups)


def test_screen_draw_groups_refuses_above_the_measured_arity_cap():
    """Above MAX_RULER_ARITY a band is still COMPUTABLE and that is precisely the hazard: a
    draw needs `arity` distinct columns, so wide draws are forced to overlap, their deltas
    stop being independent samples of the null, the sample std understates the true spread and
    the bar comes out too EASY — the bias direction fps-3jj.14 exists to remove. Refusing
    beats returning a number that is not a ruler."""
    df = _diverse_fixture_df(n_dates=200, n_cols=20)
    baseline_columns = [f"col{i}" for i in range(20)]

    with pytest.raises(ValueError, match=r"exceeds MAX_RULER_ARITY"):
        screen_draw_groups(
            df, baseline_columns, n_draws=5, arity=MAX_RULER_ARITY + 1,
            max_self_correlation=0.9,
        )


def test_assembly_survives_more_failures_than_one_lap_of_headroom(monkeypatch):
    """Review finding (PR #335): `candidate_pool` returns `n_picks + n_columns` entries, so a
    screener bounded by that list gives up after enough failures even when `arity` perfectly
    good columns are still passing — "this batch cannot support this many draws" when it
    plainly can. Same class of defect as the empty fallback tail fps-3jj.21 exists to remove.
    The screening path therefore consumes the UNBOUNDED `candidate_sequence`, not the list.

    Here 8 of 10 columns fail permanently and 20 draws are requested — far more failures than
    one lap of headroom absorbs.
    """
    import experiments.pipeline.placebo as placebo_module

    df = _diverse_fixture_df(n_dates=200, n_cols=10)
    baseline_columns = [f"col{i}" for i in range(10)]
    passing = {"col0", "col1"}
    real = placebo_module.make_placebo_series

    def _fake(frame, source_column, seed):
        if source_column not in passing:
            return frame[source_column].rename(placebo_module.PLACEBO_COLUMN_NAME)
        return real(frame, source_column, seed)

    monkeypatch.setattr(placebo_module, "make_placebo_series", _fake)

    screened = screen_draws(df, baseline_columns, n_draws=20, max_self_correlation=0.9)

    assert len(screened) == 20
    assert {c for c, _s, _r in screened} == passing
    # Still one seed per pick, even across all that substitution.
    seeds = [s for _c, s, _r in screened]
    assert len(set(seeds)) == len(seeds)


def test_assembly_stops_after_a_full_lap_with_nothing_passing(monkeypatch):
    """The other half of an unbounded sequence: it has to terminate. A whole lap is one
    presentation of every column, each at a fresh seed — if not one clears the screen, the
    next lap offers the same columns again and the DATA is the problem. Counting CONSECUTIVE
    failures rather than total ones is what stops a batch with a few permanently-unusable
    columns (batch1 has five all-NaN LGA columns, skipped on every lap) from being refused a
    bank it can supply — which the previous test covers."""
    import experiments.pipeline.placebo as placebo_module

    df = _diverse_fixture_df(n_dates=200, n_cols=4)
    baseline_columns = [f"col{i}" for i in range(4)]

    def _always_no_op(frame, source_column, seed):
        return frame[source_column].rename(placebo_module.PLACEBO_COLUMN_NAME)

    monkeypatch.setattr(placebo_module, "make_placebo_series", _always_no_op)

    with pytest.raises(ValueError, match=r"a full pass over all 4 baseline columns"):
        screen_draws(df, baseline_columns, n_draws=2, max_self_correlation=0.9)


def test_candidate_sequence_is_unbounded():
    """`candidate_pool` is the finite, readable VIEW of the sequence; the screener consumes
    the sequence itself, so it must not terminate on its own."""
    import itertools

    baseline_columns = [f"col{i}" for i in range(4)]
    picks = list(itertools.islice(candidate_sequence(baseline_columns, 2), 500))
    assert len(picks) == 500
    seeds = [s for _c, s in picks]
    assert len(set(seeds)) == 500, "a seed repeated deep into the sequence"


def test_screen_draw_groups_refuses_an_arity_wider_than_the_column_list():
    """A separate refusal from the MAX_RULER_ARITY cap, and it has to stay reachable: a draw
    needs `arity` DISTINCT columns, so a batch with fewer columns than that cannot build one
    at any draw count. Exercised at exactly the cap so the cap check cannot mask it."""
    n_cols = MAX_RULER_ARITY - 1
    df = _diverse_fixture_df(n_dates=200, n_cols=n_cols)
    baseline_columns = [f"col{i}" for i in range(n_cols)]
    with pytest.raises(ValueError, match=r"exceeds the baseline column count"):
        screen_draw_groups(
            df, baseline_columns, n_draws=2, arity=MAX_RULER_ARITY,
            max_self_correlation=0.9,
        )


def test_make_placebo_frame_builds_one_column_per_member_named_by_arity():
    df = _diverse_fixture_df(n_dates=200, n_cols=10)
    baseline_columns = [f"col{i}" for i in range(10)]
    draw = screen_draw_groups(
        df, baseline_columns, n_draws=1, arity=3, max_self_correlation=0.9
    )[0]

    frame = make_placebo_frame(df, draw)

    assert list(frame.columns) == ["__placebo_1__", "__placebo_2__", "__placebo_3__"]
    assert frame.index.equals(df.index)
    for name, (source_column, seed, _corr) in zip(frame.columns, draw, strict=True):
        expected = make_placebo_series(df, source_column, seed)
        pd.testing.assert_series_equal(
            frame[name], expected.rename(name), check_names=True
        )


def test_make_placebo_frame_members_do_not_share_a_block_order():
    """Two members permuted with the SAME seed share a block ORDER, which makes the pair as
    correlated as their source columns are — for a reason that has nothing to do with how a
    real candidate's columns relate. Distinct seeds are what stops that."""
    df = _diverse_fixture_df(n_dates=400, n_cols=10)
    same_source = [("col0", 97, 0.0), ("col0", 97, 0.0)]
    shared = make_placebo_frame(df, same_source)
    assert shared["__placebo_1__"].equals(shared["__placebo_2__"])

    distinct_seeds = [("col0", 97, 0.0), ("col0", 101, 0.0)]
    varied = make_placebo_frame(df, distinct_seeds)
    assert not varied["__placebo_1__"].equals(varied["__placebo_2__"])
