"""Tests for fuel_signal.inspect — gradient heatmap builder and Flask routes."""

import datetime

import pytest

from fuel_signal import series as _series
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


def _resolve_specs(conn, specs):
    return [_series.resolve(conn, s) for s in specs]


def test_gradient_heatmap_filters_to_selected_lga(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_prices(conn, 1001)
    _insert_prices(conn, 2001)
    resolved = _resolve_specs(conn, ["lga:Blue Mountains"])
    result = _build_gradient_heatmap(resolved, cutoff=None)
    assert result
    labels = {label for label, _ in result["rows"]}
    assert any("Blue Mountains" in lab for lab in labels)
    assert not any("Parramatta" in lab for lab in labels)


def test_gradient_heatmap_includes_brand_and_station_rows(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_prices(conn, 1001)
    _insert_prices(conn, 2001)
    resolved = _resolve_specs(
        conn, ["sydney", "brand:Shell", "station:1001"]
    )
    result = _build_gradient_heatmap(resolved, cutoff=None)
    assert result
    labels = {label for label, _ in result["rows"]}
    assert any("Sydney" in lab for lab in labels)
    assert any("Shell" in lab for lab in labels)
    assert any("Springwood" in lab for lab in labels)


def test_gradient_heatmap_returns_daily_dates(conn):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=10)
    resolved = _resolve_specs(conn, ["station:1001"])
    result = _build_gradient_heatmap(resolved, cutoff=None)
    assert result
    assert len(result["dates"]) == 10
    assert all(len(d) == 10 for d in result["dates"])


def test_gradient_heatmap_respects_cutoff(conn):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=10)
    resolved = _resolve_specs(conn, ["station:1001"])
    cutoff = (datetime.date(2024, 1, 1) + datetime.timedelta(days=5)).isoformat()
    result = _build_gradient_heatmap(resolved, cutoff=cutoff)
    assert result
    assert all(d >= cutoff for d in result["dates"])


def test_gradient_heatmap_empty_when_no_resolved_series(conn):
    result = _build_gradient_heatmap([], cutoff=None)
    assert result == {}
