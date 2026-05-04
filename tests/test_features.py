"""Tests for fuel_signal.features — feature pipeline and PIT validation.

Synthetic series strategy
--------------------------
A sawtooth price series with known cycle parameters is used throughout.
Each cycle: 3-day sharp rise to peak, then 43-day linear decline to trough.
Total cycle length = 46 days, amplitude = 25c, base = 150c.

Three full cycles gives peaks at approximately days 2, 48, 94 —
enough for CycleDetector to confirm >= 2 peaks.
"""

from __future__ import annotations

import datetime

import pytest

from fuel_signal.db import create_schema, open_db, upsert_daily_prices, upsert_stations
from fuel_signal.features import FEATURE_COLUMNS, assemble_feature_rows, compute_features

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

_START = "2020-01-01"
_3_CYCLES = _sawtooth_series(3.0, start=_START)  # 138 days
_5_CYCLES = _sawtooth_series(5.0, start=_START)  # 230 days


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


# ---------------------------------------------------------------------------
# compute_features — None-return cases
# ---------------------------------------------------------------------------

def test_no_station_price_returns_none(conn):
    """Returns None when station has no price on date_d."""
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


# ---------------------------------------------------------------------------
# compute_features — success path and feature contract
# ---------------------------------------------------------------------------

def test_feature_keys_match_documented_columns(conn):
    """Feature dict keys match FEATURE_COLUMNS exactly (catches accidental renames)."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert set(features.keys()) == set(FEATURE_COLUMNS)


def test_station_price_matches_daily_prices(conn):
    """station_price_cents reflects the actual price on date_d."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    expected_price = dict(_3_CYCLES)[_DATE_D]
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["station_price_cents"] - expected_price) < 0.05


def test_derived_features_are_consistent(conn):
    """station_minus_last_min/max are consistent with the other features."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    price = features["station_price_cents"]
    assert abs(features["station_minus_last_min_cents"] - (price - features["cycle_last_min_cents"])) < 1e-9
    assert abs(features["station_minus_last_max_cents"] - (price - features["cycle_last_max_cents"])) < 1e-9


def test_single_station_sydney_avg_delta_is_zero(conn):
    """With one station, station_minus_sydney_avg_cents must be 0."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["station_minus_sydney_avg_cents"]) < 1e-9


# ---------------------------------------------------------------------------
# PIT (point-in-time) safety validation
# ---------------------------------------------------------------------------

def test_pit_safety(tmp_path):
    """Features computed with future data present must equal features without it.

    Procedure:
      1. Build a full DB with 5 cycles of prices (230 days).
      2. Compute f1 at DATE_D (day 120) — full DB; future data exists past day 120.
      3. Build a truncated DB with data only up to DATE_D (days 0..120 inclusive).
      4. Compute f2 at DATE_D from the truncated DB.
      5. Assert f1 == f2 within float tolerance.

    Any difference would mean a feature has a forward-looking dependency.
    """
    # --- Full DB (5 cycles = 230 days) ---
    conn_full = open_db(tmp_path / "full.db")
    create_schema(conn_full)
    _add_station(conn_full, STATION_A)
    _add_prices(conn_full, STATION_A, _5_CYCLES)

    f1 = compute_features(conn_full, STATION_A, _DATE_D)
    conn_full.close()

    # --- Truncated DB (days 0..120 inclusive) ---
    conn_trunc = open_db(tmp_path / "trunc.db")
    create_schema(conn_trunc)
    _add_station(conn_trunc, STATION_A)
    truncated = [row for row in _5_CYCLES if row[0] <= _DATE_D]
    _add_prices(conn_trunc, STATION_A, truncated)

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

    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)

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
    """Rows where compute_features returns None are excluded from the output."""
    _add_station(conn, STATION_A)
    _add_station(conn, STATION_B)

    # STATION_A: full 3 cycles — will produce label + feature rows
    _add_prices(conn, STATION_A, _3_CYCLES)

    # STATION_B: 30 days only — too short for cycle detection; labels will be
    # produced (assemble_training_rows requires only forward data, not cycle data)
    # but compute_features will return None for all STATION_B rows.
    short = _sawtooth_series(0.7, start=_START)
    _add_prices(conn, STATION_B, short)

    df = assemble_feature_rows(conn, station_codes=[STATION_A, STATION_B])
    # All rows in output must be for STATION_A only
    assert len(df) > 0
    assert set(df["station_code"].unique()) == {STATION_A}


def test_assembler_columns(conn):
    """Output DataFrame has label columns followed by all FEATURE_COLUMNS."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)

    df = assemble_feature_rows(conn, station_codes=[STATION_A])
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
