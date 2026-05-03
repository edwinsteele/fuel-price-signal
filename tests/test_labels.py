"""Tests for fuel_signal.labels — label generation and training-row assembly."""

import datetime

import pytest

from fuel_signal.db import create_schema, open_db, upsert_daily_prices, upsert_stations
from fuel_signal.labels import assemble_training_rows, compute_label

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


def _d(offset: int) -> str:
    """Return YYYY-MM-DD for today + offset days."""
    return (datetime.date.today() + datetime.timedelta(days=offset)).isoformat()


def _station(conn, station_code: int) -> None:
    upsert_stations(conn, [{
        "station_code": station_code,
        "name": f"Station {station_code}",
        "address": f"{station_code} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": "2777",
        "brand": "Shell",
    }])


def _prices(conn, station_code: int, day_prices: list[tuple[int, float]]) -> None:
    """Insert (offset, price) pairs into daily_prices for station_code."""
    upsert_daily_prices(conn, [(station_code, "E10", _d(offset), price) for offset, price in day_prices])
    conn.commit()


# ---------------------------------------------------------------------------
# compute_label
# ---------------------------------------------------------------------------

def test_compute_label_drop(conn):
    """Forward min drops below threshold → label=1."""
    _station(conn, 1001)
    # today=200, forward min=196, threshold=3 → 196 < 197 → label=1
    _prices(conn, 1001, [(-15 + i, v) for i, v in enumerate([200, 199, 198, 197, 196, 197, 198, 199])])
    label = compute_label(conn, 1001, _d(-15), horizon_days=7, threshold_cents=3.0)
    assert label == 1


def test_compute_label_keep(conn):
    """Forward min stays above threshold → label=0."""
    _station(conn, 1001)
    # today=200, forward min=198, threshold=3 → 198 >= 197 → label=0
    _prices(conn, 1001, [(-15 + i, v) for i, v in enumerate([200, 199, 198, 198, 198, 198, 198, 198])])
    label = compute_label(conn, 1001, _d(-15), horizon_days=7, threshold_cents=3.0)
    assert label == 0


def test_compute_label_insufficient_forward_data(conn):
    """Only 5 forward days when horizon=7 → None."""
    _station(conn, 1001)
    # 6 rows total: today + 5 forward days (need 7)
    _prices(conn, 1001, [(-15 + i, 200.0 - i) for i in range(6)])
    label = compute_label(conn, 1001, _d(-15), horizon_days=7, threshold_cents=3.0)
    assert label is None


def test_compute_label_missing_today(conn):
    """No price row for the requested date → None."""
    _station(conn, 1001)
    # Insert prices starting the day after the anchor
    _prices(conn, 1001, [(-14 + i, 200.0) for i in range(8)])
    label = compute_label(conn, 1001, _d(-15), horizon_days=7, threshold_cents=3.0)
    assert label is None


# ---------------------------------------------------------------------------
# assemble_training_rows
# ---------------------------------------------------------------------------

def test_assemble_drop_label(conn):
    """Assembler produces row with label=1 and correct prices."""
    _station(conn, 1001)
    _prices(conn, 1001, [(-15 + i, v) for i, v in enumerate([200, 199, 198, 197, 196, 197, 198, 199])])
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0, station_codes=[1001])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["station_code"] == 1001
    assert row["label"] == 1
    assert row["today_price_cents"] == 200.0
    assert row["future_min_cents"] == 196.0


def test_assemble_keep_label(conn):
    """Assembler produces row with label=0 when price stays near today."""
    _station(conn, 1001)
    _prices(conn, 1001, [(-15 + i, v) for i, v in enumerate([200, 199, 198, 198, 198, 198, 198, 198])])
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0, station_codes=[1001])
    assert len(df) == 1
    assert df.iloc[0]["label"] == 0


def test_assemble_excludes_incomplete_horizon(conn):
    """Rows without a full horizon of forward data are excluded."""
    _station(conn, 1001)
    # 6 rows total → no row has 7 forward days
    _prices(conn, 1001, [(-15 + i, 200.0 - i) for i in range(6)])
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0, station_codes=[1001])
    assert df.empty


def test_assemble_multi_station(conn):
    """Assembler returns rows from both stations when station_codes covers them."""
    _station(conn, 1001)
    _station(conn, 1002)
    _prices(conn, 1001, [(-15 + i, 200.0 - i) for i in range(10)])
    _prices(conn, 1002, [(-15 + i, 180.0 + i) for i in range(10)])
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0, station_codes=[1001, 1002])
    assert set(df["station_code"]) == {1001, 1002}


def test_assemble_empty_station(conn):
    """Station with no prices yields an empty DataFrame with correct columns."""
    _station(conn, 1001)
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0, station_codes=[1001])
    assert df.empty
    assert list(df.columns) == ["station_code", "price_date", "today_price_cents", "future_min_cents", "label"]
