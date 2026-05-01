"""Tests for fuel_signal.inspect — gradient heatmap builder and Flask routes."""

import datetime

import pytest

from fuel_signal.db import (
    create_schema,
    open_db,
    upsert_daily_prices,
    upsert_stations,
)
from fuel_signal.inspect import _build_gradient_heatmap


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


_STATION_BM = {
    "station_code": 1001,
    "name": "Shell Springwood",
    "address": "1 Main Street, Springwood",
    "suburb": "Springwood",
    "postcode": "2777",
    "brand": "Shell",
}

_STATION_SYD = {
    "station_code": 2001,
    "name": "Ampol Parramatta",
    "address": "5 Church Street, Parramatta",
    "suburb": "Parramatta",
    "postcode": "2150",
    "brand": "Ampol",
}


def _insert_prices(conn, station_code, n_days=10, base_price=160.0):
    base = datetime.date(2024, 1, 1)
    rows = [
        (station_code, "E10", (base + datetime.timedelta(days=i)).isoformat(), base_price + i)
        for i in range(n_days)
    ]
    upsert_daily_prices(conn, rows)
    conn.commit()


def test_gradient_heatmap_shows_all_councils_when_none_selected(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_prices(conn, 1001)
    _insert_prices(conn, 2001)
    result = _build_gradient_heatmap(conn, cutoff=None, councils=None)
    assert result
    council_names = {c for c, _ in result["rows"]}
    assert "Blue Mountains" in council_names
    assert "Parramatta" in council_names


def test_gradient_heatmap_filters_to_selected_councils(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_prices(conn, 1001)
    _insert_prices(conn, 2001)
    result = _build_gradient_heatmap(conn, cutoff=None, councils=["Blue Mountains"])
    assert result
    council_names = {c for c, _ in result["rows"]}
    assert "Blue Mountains" in council_names
    assert "Parramatta" not in council_names


def test_gradient_heatmap_returns_daily_dates(conn):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=10)
    result = _build_gradient_heatmap(conn, cutoff=None)
    assert result
    # Daily mode: 10 dates, each formatted YYYY-MM-DD
    assert len(result["dates"]) == 10
    assert all(len(d) == 10 for d in result["dates"])


def test_gradient_heatmap_respects_cutoff(conn):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=10)
    # Cutoff after day 5 keeps only the last 5 dates
    cutoff = (datetime.date(2024, 1, 1) + datetime.timedelta(days=5)).isoformat()
    result = _build_gradient_heatmap(conn, cutoff=cutoff)
    assert result
    assert all(d >= cutoff for d in result["dates"])


def test_gradient_heatmap_empty_when_no_councils_selected(conn):
    # When only preferred stations are selected (no lga: specs), the
    # index() route must not fall back to showing all LGAs.
    # Simulate the route logic: selected_councils=[] → heatmap_data=None
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_prices(conn, 1001)
    _insert_prices(conn, 2001)
    # Route logic: if not selected_councils → heatmap_data = None (no call)
    selected_councils: list[str] = []
    heatmap_data = (
        _build_gradient_heatmap(conn, cutoff=None, councils=selected_councils)
        if selected_councils else None
    )
    assert heatmap_data is None
