"""Tests for fuel_signal.features — feature pipeline and PIT validation.

Synthetic series strategy
--------------------------
A sawtooth price series with known cycle parameters is used throughout.
Each cycle: 3-day sharp rise to peak, then 43-day linear decline to trough.
Total cycle length = 46 days, amplitude = 25c, base = 150c.

Three full cycles gives peaks at approximately days 2, 48, 94 —
enough for CycleDetector to confirm >= 2 peaks.

Station setup
-------------
LGA/brand mean features require ≥3 non-Sticky stations.  All tests that
exercise the success path set up STATION_A, STATION_B, STATION_C
(all postcode 2777 = Blue Mountains, brand = Shell) with the same price
series, then insert Competitive classification rows via
_insert_competitive() so the aggregation floor is satisfied.

The LGA mean and brand mean both equal the common station price when all
three stations have identical prices, giving zero delta features.
"""

from __future__ import annotations

import datetime

import pytest

from fuel_signal.db import (
    create_schema,
    open_db,
    upsert_daily_prices,
    upsert_station_classes,
    upsert_stations,
)
from fuel_signal.features import (
    AGGREGATE_MIN_STATIONS,
    FEATURE_COLUMNS,
    MIN_TRAINING_ROWS_PER_STATION,
    assemble_feature_rows,
    compute_features,
)

# ---------------------------------------------------------------------------
# Synthetic series helpers
# ---------------------------------------------------------------------------

_CYCLE_LENGTH = 46
_RISE_DAYS = 3
_BASE_PRICE = 150.0
_AMPLITUDE = 25.0


def _sawtooth_series(
    n_cycles: float = 3.0,
    start: str = "2020-01-01",
) -> list[tuple[str, float]]:
    """Return [(date_str, price_cents)] with n_cycles full sawtooth cycles."""
    total_days = int(n_cycles * _CYCLE_LENGTH)
    start_date = datetime.date.fromisoformat(start)
    result = []
    for day in range(total_days):
        pos = day % _CYCLE_LENGTH
        if pos < _RISE_DAYS:
            price = _BASE_PRICE + _AMPLITUDE * (pos / _RISE_DAYS)
        else:
            price = _BASE_PRICE + _AMPLITUDE * (
                1.0 - (pos - _RISE_DAYS) / (_CYCLE_LENGTH - _RISE_DAYS)
            )
        result.append(((start_date + datetime.timedelta(days=day)).isoformat(), round(price, 1)))
    return result


STATION_A = 1001
STATION_B = 1002
STATION_C = 1003  # third station needed to satisfy LGA/brand floor (≥3)

_START = "2020-01-01"
_3_CYCLES = _sawtooth_series(3.0, start=_START)  # 138 days
_5_CYCLES = _sawtooth_series(5.0, start=_START)  # 230 days
_16_CYCLES = _sawtooth_series(16.0, start=_START)  # 736 days, > 365 label rows


def _date_at_day(day: int, start: str = _START) -> str:
    return (datetime.date.fromisoformat(start) + datetime.timedelta(days=day)).isoformat()


# Day 120 is well into cycle 3, past all three confirmed peaks (2, 48, 94).
_DATE_D = _date_at_day(120)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


def _add_station(conn, station_code: int) -> None:
    upsert_stations(conn, [{
        "station_code": station_code,
        "name": f"Station {station_code}",
        "address": f"{station_code} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": "2777",
        "brand": "Shell",
    }])


def _add_prices(conn, station_code: int, series: list[tuple[str, float]]) -> None:
    upsert_daily_prices(conn, [(station_code, "E10", d, p) for d, p in series])
    conn.commit()


STATION_D = 1004  # fourth station for tests needing 3 companions


def _set_price(conn, station_code: int, date_d: str, price_cents: float) -> None:
    """Insert or replace a price in daily_prices (idempotent)."""
    fid = conn.execute("SELECT id FROM fuel_types WHERE code='E10'").fetchone()[0]
    conn.execute(
        "INSERT OR REPLACE INTO daily_prices (station_code, fuel_type_id, price_date, price_decicents)"
        " VALUES (?, ?, ?, ?)",
        (station_code, fid, int(date_d.replace("-", "")), round(price_cents * 10)),
    )
    conn.commit()


def _insert_competitive(conn, station_codes: list[int], dates: list[str]) -> None:
    """Directly insert Competitive (premium=0) classification for stations × dates.

    Bypasses the classifier algorithm; use when the test is exercising feature
    computation (not classifier correctness).  All stations must be pre-inserted
    in the stations table before this is called.
    """
    rows = [
        (sc, int(d.replace("-", "")), "Competitive", 0)
        for sc in station_codes
        for d in dates
    ]
    upsert_station_classes(conn, rows)
    conn.commit()


def _setup_triple_with_classification(
    conn,
    series: list[tuple[str, float]],
    date_d: str,
) -> None:
    """Add STATION_A, B, C with the given price series; insert Competitive for date_d.

    Provides the minimum 3 non-Sticky stations required for LGA/brand aggregates.
    """
    for sc in (STATION_A, STATION_B, STATION_C):
        _add_station(conn, sc)
        _add_prices(conn, sc, series)
    _insert_competitive(conn, [STATION_A, STATION_B, STATION_C], [date_d])


def _setup_triple_full_classify(
    conn,
    series: list[tuple[str, float]],
) -> None:
    """Add STATION_A, B, C with the given price series; insert Competitive for ALL dates.

    Needed by assemble_feature_rows tests which span every date in the series.
    """
    for sc in (STATION_A, STATION_B, STATION_C):
        _add_station(conn, sc)
        _add_prices(conn, sc, series)
    dates = [d for d, _ in series]
    _insert_competitive(conn, [STATION_A, STATION_B, STATION_C], dates)


# ---------------------------------------------------------------------------
# compute_features — None-return cases
# ---------------------------------------------------------------------------

def test_no_station_price_returns_none(conn):
    """Returns None when station has no price on date_d (check happens before LGA lookup)."""
    _add_station(conn, STATION_A)
    # Prices exist but not on _DATE_D (series ends at day 119)
    _add_prices(conn, STATION_A, _3_CYCLES[: 120])  # days 0..119
    assert compute_features(conn, STATION_A, _DATE_D) is None


def test_insufficient_cycle_data_returns_none(conn):
    """Returns None when CycleDetector.detect returns None (< 2 peaks)."""
    _add_station(conn, STATION_A)
    # Only ~30 days of data — not enough for two peaks (first peak is around day 2,
    # but a second peak requires another 46 days minimum).
    short_series = _sawtooth_series(0.7, start=_START)  # ~32 days, only 1 peak
    _add_prices(conn, STATION_A, short_series)
    date_d = _date_at_day(len(short_series) - 1)
    assert compute_features(conn, STATION_A, date_d) is None


def test_returns_none_when_no_lga_mean(conn):
    """Returns None when station_class not populated (< 3 non-Sticky for the LGA)."""
    # Set up only ONE station — LGA mean will be NULL (floor = 3 stations).
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    # No station_class rows → _lga_mean_on_date returns None → compute_features returns None.
    assert compute_features(conn, STATION_A, _DATE_D) is None


# ---------------------------------------------------------------------------
# compute_features — success path and feature contract
# ---------------------------------------------------------------------------

def test_feature_keys_match_documented_columns(conn):
    """Feature dict keys match FEATURE_COLUMNS exactly (catches accidental renames)."""
    _setup_triple_with_classification(conn, _3_CYCLES, _DATE_D)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert set(features.keys()) == set(FEATURE_COLUMNS)


def test_station_price_matches_daily_prices(conn):
    """station_price_cents reflects the actual price on date_d."""
    _setup_triple_with_classification(conn, _3_CYCLES, _DATE_D)
    expected_price = dict(_3_CYCLES)[_DATE_D]
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["station_price_cents"] - expected_price) < 0.05


def test_derived_features_are_consistent(conn):
    """station_minus_last_min/max are consistent with the other features."""
    _setup_triple_with_classification(conn, _3_CYCLES, _DATE_D)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    price = features["station_price_cents"]
    assert abs(features["station_minus_last_min_cents"] - (price - features["cycle_last_min_cents"])) < 1e-9
    assert abs(features["station_minus_last_max_cents"] - (price - features["cycle_last_max_cents"])) < 1e-9


def test_single_station_sydney_avg_delta_is_zero(conn):
    """With three stations at equal prices, station_minus_sydney_avg_cents must be 0."""
    _setup_triple_with_classification(conn, _3_CYCLES, _DATE_D)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["station_minus_sydney_avg_cents"]) < 1e-9


def test_lga_mean_equals_station_price_when_all_stations_identical(conn):
    """When all stations have equal prices, lga_mean_cents == station_price_cents."""
    _setup_triple_with_classification(conn, _3_CYCLES, _DATE_D)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["lga_mean_cents"] - features["station_price_cents"]) < 1e-6
    assert abs(features["station_minus_lga_mean_cents"]) < 1e-6


def test_brand_mean_equals_station_price_when_all_stations_identical(conn):
    """When all stations have equal prices and same brand, brand_mean_cents == station_price_cents."""
    _setup_triple_with_classification(conn, _3_CYCLES, _DATE_D)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["brand_mean_cents"] - features["station_price_cents"]) < 1e-6
    assert abs(features["station_minus_brand_mean_cents"]) < 1e-6


def test_lga_mean_excludes_sticky_station(conn):
    """LGA mean is computed excluding Sticky stations; Sticky station classification is present."""
    # STATION_A, STATION_B, STATION_C at 170c; STATION_D Sticky at 185c.
    _add_station(conn, STATION_D)
    for sc in (STATION_A, STATION_B, STATION_C):
        _add_station(conn, sc)
        _add_prices(conn, sc, _3_CYCLES)
    _add_prices(conn, STATION_D, _3_CYCLES)

    COMP_PRICE = 170.0
    STICKY_PRICE = 185.0
    date_d = _DATE_D
    # Override prices on test date (INSERT OR REPLACE — _3_CYCLES already has that date)
    for sc in (STATION_A, STATION_B, STATION_C):
        _set_price(conn, sc, date_d, COMP_PRICE)
    _set_price(conn, STATION_D, date_d, STICKY_PRICE)

    date_int = int(date_d.replace("-", ""))
    upsert_station_classes(conn, [
        (STATION_A, date_int, "Competitive", 0),
        (STATION_B, date_int, "Competitive", 0),
        (STATION_C, date_int, "Competitive", 0),
        (STATION_D, date_int, "Sticky", 150),
    ])
    conn.commit()

    features = compute_features(conn, STATION_A, date_d)
    assert features is not None
    # LGA mean should be avg(A, B, C) = 170c, NOT avg(A, B, C, D) = 173.75c
    assert abs(features["lga_mean_cents"] - COMP_PRICE) < 0.5
    assert abs(features["lga_mean_cents"] - (3 * COMP_PRICE + STICKY_PRICE) / 4) > 0.5


def test_lga_mean_null_when_below_floor(conn):
    """Returns None when only 2 non-Sticky stations exist (below AGGREGATE_MIN_STATIONS=3)."""
    assert AGGREGATE_MIN_STATIONS == 3
    for sc in (STATION_A, STATION_B):
        _add_station(conn, sc)
        _add_prices(conn, sc, _3_CYCLES)
    _insert_competitive(conn, [STATION_A, STATION_B], [_DATE_D])
    # Only 2 stations → LGA mean is NULL → None returned
    assert compute_features(conn, STATION_A, _DATE_D) is None


# ---------------------------------------------------------------------------
# PIT (point-in-time) safety validation
# ---------------------------------------------------------------------------

def test_pit_safety(tmp_path):
    """Features computed with future data present must equal features without it.

    Procedure:
      1. Build a full DB with 5 cycles of prices (230 days) + classification for _DATE_D.
      2. Compute f1 at DATE_D — full DB; future data exists past day 120.
      3. Build a truncated DB with data only up to DATE_D (days 0..120 inclusive)
         + same classification for _DATE_D.
      4. Compute f2 at DATE_D from the truncated DB.
      5. Assert f1 == f2 within float tolerance.

    Any difference would mean a feature has a forward-looking dependency.
    """
    three_codes = [STATION_A, STATION_B, STATION_C]

    # --- Full DB (5 cycles = 230 days) ---
    conn_full = open_db(tmp_path / "full.db")
    create_schema(conn_full)
    for sc in three_codes:
        _add_station(conn_full, sc)
        _add_prices(conn_full, sc, _5_CYCLES)
    _insert_competitive(conn_full, three_codes, [_DATE_D])

    f1 = compute_features(conn_full, STATION_A, _DATE_D)
    conn_full.close()

    # --- Truncated DB (days 0..120 inclusive) ---
    conn_trunc = open_db(tmp_path / "trunc.db")
    create_schema(conn_trunc)
    for sc in three_codes:
        _add_station(conn_trunc, sc)
        truncated = [row for row in _5_CYCLES if row[0] <= _DATE_D]
        _add_prices(conn_trunc, sc, truncated)
    _insert_competitive(conn_trunc, three_codes, [_DATE_D])

    f2 = compute_features(conn_trunc, STATION_A, _DATE_D)
    conn_trunc.close()

    assert f1 is not None, "Full-DB features should not be None"
    assert f2 is not None, "Truncated-DB features should not be None"

    for key in FEATURE_COLUMNS:
        assert abs(f1[key] - f2[key]) < 1e-6, (
            f"PIT violation: feature '{key}' differs between full and truncated DB "
            f"(full={f1[key]}, truncated={f2[key]})"
        )


# ---------------------------------------------------------------------------
# Pre-built CycleDetector passthrough
# ---------------------------------------------------------------------------

def test_prebuilt_cycle_detector_matches_standalone(conn):
    """Passing a pre-built CycleDetector gives the same result as building inline."""
    from fuel_signal.cycle import CycleDetector
    from fuel_signal.db import average_price_series

    _setup_triple_with_classification(conn, _3_CYCLES, _DATE_D)

    f_standalone = compute_features(conn, STATION_A, _DATE_D)
    cd = CycleDetector(average_price_series(conn))
    f_batched = compute_features(conn, STATION_A, _DATE_D, cycle_detector=cd)

    assert f_standalone is not None
    assert f_batched is not None
    for key in FEATURE_COLUMNS:
        assert abs(f_standalone[key] - f_batched[key]) < 1e-9


# ---------------------------------------------------------------------------
# assemble_feature_rows
# ---------------------------------------------------------------------------

def test_assembler_drops_none_rows(conn):
    """Rows where compute_features returns None are excluded from the output.

    STATION_B (short series, no cycle state) rows are dropped.
    STATION_C and STATION_D are companions for the LGA/brand floor (≥3 required)
    and are not in station_codes.
    """
    for sc in (STATION_C, STATION_D, STATION_A):
        _add_station(conn, sc)
        _add_prices(conn, sc, _3_CYCLES)

    # STATION_B: 30 days only — too short for cycle detection; labels will be
    # produced but compute_features returns None for all STATION_B rows.
    _add_station(conn, STATION_B)
    short = _sawtooth_series(0.7, start=_START)
    _add_prices(conn, STATION_B, short)

    # Insert Competitive for A, C, D across the full 3-cycle date range (3 stations → floor met).
    dates = [d for d, _ in _3_CYCLES]
    _insert_competitive(conn, [STATION_A, STATION_C, STATION_D], dates)
    _insert_competitive(conn, [STATION_B], [d for d, _ in short])

    df = assemble_feature_rows(conn, station_codes=[STATION_A, STATION_B], min_rows_per_station=0)
    # All rows in output must be for STATION_A only (B has no cycle state)
    assert len(df) > 0
    assert set(df["station_code"].unique()) == {STATION_A}


def test_assembler_columns(conn):
    """Output DataFrame has label columns followed by all FEATURE_COLUMNS."""
    _setup_triple_full_classify(conn, _3_CYCLES)

    df = assemble_feature_rows(conn, station_codes=[STATION_A], min_rows_per_station=0)
    assert len(df) > 0
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"
    # Label columns present
    for col in ("station_code", "price_date", "today_price_cents", "future_min_cents", "label"):
        assert col in df.columns


def test_assembler_empty_station_list(conn):
    """Empty station_codes returns an empty DataFrame with correct columns."""
    df = assemble_feature_rows(conn, station_codes=[])
    assert len(df) == 0
    assert set(FEATURE_COLUMNS).issubset(set(df.columns))


def test_assembler_excludes_stations_below_min_rows(conn):
    """Stations with fewer label rows than min_rows_per_station are excluded.

    STATION_A gets 3 cycles (138 days) → ~40 label rows with default params.
    STATION_B gets 5 cycles (230 days) → ~133 label rows.
    STATION_C is the third LGA station (full 5-cycle series, same price).
    Setting min_rows_per_station=100 should admit only STATION_B.
    """
    _add_station(conn, STATION_A)
    _add_station(conn, STATION_B)
    _add_station(conn, STATION_C)
    _add_prices(conn, STATION_A, _3_CYCLES)    # ~40 label rows → below threshold
    _add_prices(conn, STATION_B, _5_CYCLES)    # ~133 label rows → above threshold
    _add_prices(conn, STATION_C, _5_CYCLES)    # companion for LGA floor

    # Classify all three for all dates in the larger series.
    all_dates = [d for d, _ in _5_CYCLES]
    _insert_competitive(conn, [STATION_A, STATION_B, STATION_C], all_dates)

    df = assemble_feature_rows(
        conn,
        station_codes=[STATION_A, STATION_B],
        min_rows_per_station=100,
    )
    assert len(df) > 0
    assert set(df["station_code"].unique()) == {STATION_B}


def test_assembler_default_min_rows_enforces_365(conn):
    """Default min_rows_per_station applies MIN_TRAINING_ROWS_PER_STATION (365).

    STATION_A gets 3 cycles (138 days) → label-row count well below 365.
    STATION_B gets 16 cycles (736 days) → label-row count well above 365.
    STATION_C and STATION_D are companions with 16-cycle series so the LGA
    floor (≥3 non-Sticky stations) is met on ALL of B's dates, not just
    the first 138 days when A is present.
    Calling without min_rows_per_station must rely on the default and admit
    only STATION_B, pinning the constant + default-path behaviour.
    """
    assert MIN_TRAINING_ROWS_PER_STATION == 365

    _add_station(conn, STATION_A)
    _add_station(conn, STATION_B)
    _add_station(conn, STATION_C)
    _add_station(conn, STATION_D)
    _add_prices(conn, STATION_A, _3_CYCLES)
    _add_prices(conn, STATION_B, _16_CYCLES)
    _add_prices(conn, STATION_C, _16_CYCLES)  # companion #1
    _add_prices(conn, STATION_D, _16_CYCLES)  # companion #2: B+C+D = 3 on all 16-cycle dates

    all_dates = [d for d, _ in _16_CYCLES]
    _insert_competitive(conn, [STATION_A, STATION_B, STATION_C, STATION_D], all_dates)

    df = assemble_feature_rows(conn, station_codes=[STATION_A, STATION_B])
    assert len(df) > 0
    assert set(df["station_code"].unique()) == {STATION_B}
    assert (df["station_code"] == STATION_B).sum() >= MIN_TRAINING_ROWS_PER_STATION


def test_assembler_rejects_negative_min_rows(conn):
    """assemble_feature_rows raises ValueError for negative min_rows_per_station."""
    with pytest.raises(ValueError, match="min_rows_per_station must be >= 0"):
        assemble_feature_rows(conn, min_rows_per_station=-1)


def test_assembler_sticky_station_excluded_from_lga_mean(conn):
    """A Sticky station contributes to its own label rows but NOT to the LGA mean.

    Setup: STATION_A (Competitive) at 170c, STATION_B (Competitive) at 170c,
           STATION_C (Sticky) at 185c. LGA mean should be avg(A, B) = 170c,
           not avg(A, B, C) = 175c.
    """
    STATION_D = 1004  # second Competitive (needed alongside A, B for floor when C is Sticky)
    upsert_stations(conn, [{
        "station_code": STATION_D,
        "name": "Station D",
        "address": f"{STATION_D} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": "2777",
        "brand": "Shell",
    }])

    # Use flat prices so the assertion is exact.
    COMP_PRICE = 170.0
    STICKY_PRICE = 185.0

    # Add enough cycle history for CycleDetector to work.
    for sc in (STATION_A, STATION_B, STATION_C, STATION_D):
        _add_station(conn, sc)
        _add_prices(conn, sc, _3_CYCLES)

    # Override the single date we test on with known flat prices (INSERT OR REPLACE).
    date_d = _DATE_D
    date_int = int(date_d.replace("-", ""))
    for sc in (STATION_A, STATION_B, STATION_D):
        _set_price(conn, sc, date_d, COMP_PRICE)
    _set_price(conn, STATION_C, date_d, STICKY_PRICE)

    # Classify: A, B, D → Competitive; C → Sticky.
    upsert_station_classes(conn, [
        (STATION_A, date_int, "Competitive", 0),
        (STATION_B, date_int, "Competitive", 0),
        (STATION_D, date_int, "Competitive", 0),
        (STATION_C, date_int, "Sticky", 150),
    ])
    conn.commit()

    features = compute_features(conn, STATION_A, date_d)
    assert features is not None
    # LGA mean = avg(A, B, D) = 170c (C excluded)
    assert abs(features["lga_mean_cents"] - COMP_PRICE) < 0.5
    # C's price (185c) should not influence LGA mean
    assert abs(features["lga_mean_cents"] - STICKY_PRICE) > 1.0


def test_assembler_discount_station_included_in_lga_mean(conn):
    """A Discount station (class = Discount) IS included in the LGA mean (blended policy)."""
    STATION_D = 1004
    upsert_stations(conn, [{
        "station_code": STATION_D,
        "name": "Station D",
        "address": f"{STATION_D} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": "2777",
        "brand": "Shell",
    }])

    COMP_PRICE = 170.0
    DISC_PRICE = 155.0

    for sc in (STATION_A, STATION_B, STATION_C, STATION_D):
        _add_station(conn, sc)
        _add_prices(conn, sc, _3_CYCLES)

    date_d = _DATE_D
    date_int = int(date_d.replace("-", ""))
    for sc in (STATION_A, STATION_B, STATION_C):
        _set_price(conn, sc, date_d, COMP_PRICE)
    _set_price(conn, STATION_D, date_d, DISC_PRICE)

    # STATION_D is Discount (included in blended mean)
    upsert_station_classes(conn, [
        (STATION_A, date_int, "Competitive", 0),
        (STATION_B, date_int, "Competitive", 0),
        (STATION_C, date_int, "Competitive", 0),
        (STATION_D, date_int, "Discount", -150),
    ])
    conn.commit()

    features = compute_features(conn, STATION_A, date_d)
    assert features is not None
    # LGA mean = avg(A, B, C, D) = (170 + 170 + 170 + 155) / 4 = 166.25
    expected = (3 * COMP_PRICE + DISC_PRICE) / 4
    assert abs(features["lga_mean_cents"] - expected) < 0.5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_csv(conn, tmp_path):
    """CLI happy path: exits 0 and writes a CSV with expected columns."""
    from click.testing import CliRunner

    from fuel_signal.features import main

    _setup_triple_full_classify(conn, _3_CYCLES)

    out_csv = tmp_path / "features.csv"
    result = CliRunner().invoke(main, [
        "--db", str(tmp_path / "test.db"),
        "--output", str(out_csv),
        "--min-rows", "0",
    ])
    assert result.exit_code == 0, result.output
    assert out_csv.exists()

    import pandas as pd
    df = pd.read_csv(out_csv)
    assert len(df) > 0
    for col in FEATURE_COLUMNS:
        assert col in df.columns
