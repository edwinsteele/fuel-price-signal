"""Tests for fuel_signal.series — resolve(), resolve_members(), enumerate_groups()."""

import pytest

from fuel_signal.db import (
    create_schema,
    open_db,
    upsert_daily_prices,
    upsert_stations,
)
from fuel_signal.series import SeriesError, enumerate_groups, resolve, resolve_members

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


_ST_AMPOL = {
    "station_code": 100,
    "name": "Ampol Springwood",
    "address": "1 Great Western Hwy, Springwood",
    "suburb": "Springwood",
    "postcode": "2777",
    "brand": "Ampol",
}

_ST_SHELL = {
    "station_code": 200,
    "name": "Shell Blaxland",
    "address": "2 Old Bathurst Rd, Blaxland",
    "suburb": "Blaxland",
    "postcode": "2774",
    "brand": "Shell",
}

_ST_BP = {
    "station_code": 300,
    "name": "BP Valley Heights",
    "address": "3 Valley Road, Valley Heights",
    "suburb": "Valley Heights",
    "postcode": "2777",
    "brand": "BP",
}

_PENRITH_STATIONS = [
    {
        "station_code": 401 + i,
        "name": f"Ampol Penrith {i}",
        "address": f"{i+1} High Street, Penrith",
        "suburb": "Penrith",
        "postcode": "2750",
        "brand": "Ampol",
    }
    for i in range(4)
]


def _seed_stations(conn):
    upsert_stations(conn, [_ST_AMPOL, _ST_SHELL, _ST_BP])


def _seed_daily(conn, codes_and_prices: dict[int, float], dates=("2024-01-01", "2024-01-02")):
    rows = []
    for code, price in codes_and_prices.items():
        for d in dates:
            rows.append((code, "E10", d, price))
    upsert_daily_prices(conn, rows)
    conn.commit()


# ---------------------------------------------------------------------------
# resolve() — sydney
# ---------------------------------------------------------------------------

def test_resolve_sydney(conn):
    _seed_stations(conn)
    _seed_daily(conn, {100: 170.0, 200: 172.0})
    r = resolve(conn, "sydney")
    assert r.kind == "sydney"
    assert r.spec == "sydney"
    assert r.label == "Sydney E10 mean"
    assert len(r.points) == 2
    assert r.points[0][0] == "2024-01-01"
    assert r.points[0][1] == pytest.approx(171.0)


def test_resolve_sydney_case_insensitive(conn):
    _seed_stations(conn)
    r = resolve(conn, "SYDNEY")
    assert r.kind == "sydney"


# ---------------------------------------------------------------------------
# resolve() — lga:
# ---------------------------------------------------------------------------

def test_resolve_lga(conn):
    upsert_stations(conn, _PENRITH_STATIONS)
    rows = [(s["station_code"], "E10", "2024-03-01", 165.0) for s in _PENRITH_STATIONS]
    upsert_daily_prices(conn, rows)
    conn.commit()
    r = resolve(conn, "lga:Penrith")
    assert r.kind == "lga"
    assert "Penrith" in r.label
    assert len(r.points) > 0


def test_resolve_lga_case_insensitive(conn):
    upsert_stations(conn, _PENRITH_STATIONS)
    rows = [(s["station_code"], "E10", "2024-03-01", 165.0) for s in _PENRITH_STATIONS]
    upsert_daily_prices(conn, rows)
    conn.commit()
    r = resolve(conn, "lga:penrith")
    assert r.kind == "lga"


def test_resolve_council_alias(conn):
    upsert_stations(conn, _PENRITH_STATIONS)
    rows = [(s["station_code"], "E10", "2024-03-01", 165.0) for s in _PENRITH_STATIONS]
    upsert_daily_prices(conn, rows)
    conn.commit()
    r = resolve(conn, "council:Penrith")
    assert r.kind == "lga"


def test_resolve_lga_unknown_raises(conn):
    with pytest.raises(SeriesError, match="No LGA matching"):
        resolve(conn, "lga:NoSuchPlace")


def test_resolve_lga_ambiguous_raises(conn):
    with pytest.raises(SeriesError, match="Ambiguous"):
        resolve(conn, "lga:e")  # matches multiple LGAs


# ---------------------------------------------------------------------------
# resolve() — brand:
# ---------------------------------------------------------------------------

def test_resolve_brand(conn):
    # Need enough stations for distinct_brands min_stations threshold
    stations = [
        {**_ST_AMPOL, "station_code": 500 + i, "address": f"{i} Ampol St, Suburb{i}",
         "suburb": f"Suburb{i}", "postcode": "2000"}
        for i in range(4)
    ]
    upsert_stations(conn, stations)
    rows = [(s["station_code"], "E10", "2024-03-01", 168.0) for s in stations]
    upsert_daily_prices(conn, rows)
    conn.commit()
    r = resolve(conn, "brand:Ampol")
    assert r.kind == "brand"
    assert "Ampol" in r.label
    assert len(r.points) > 0


def test_resolve_brand_unknown_raises(conn):
    with pytest.raises(SeriesError, match="No brand matching"):
        resolve(conn, "brand:NoSuchBrand")


# ---------------------------------------------------------------------------
# resolve() — station: prefix and bare text
# ---------------------------------------------------------------------------

def test_resolve_station_by_code(conn):
    _seed_stations(conn)
    _seed_daily(conn, {100: 170.0})
    r = resolve(conn, "station:100")
    assert r.kind == "station"
    assert r.spec == "station:100"
    assert len(r.points) > 0


def test_resolve_station_bare_text(conn):
    _seed_stations(conn)
    _seed_daily(conn, {200: 172.0})
    r = resolve(conn, "Shell Blaxland")
    assert r.kind == "station"
    assert r.spec == "station:200"


def test_resolve_station_no_match_raises(conn):
    with pytest.raises(SeriesError, match="No station found"):
        resolve(conn, "station:99999")


def test_resolve_station_multiple_matches_raises(conn):
    """Two stations both containing 'Springwood' → disambiguation error."""
    s2 = {**_ST_SHELL, "station_code": 201, "name": "Springwood Other",
          "address": "99 Main St, Springwood", "suburb": "Springwood"}
    upsert_stations(conn, [_ST_AMPOL, s2])
    with pytest.raises(SeriesError, match="Multiple stations"):
        resolve(conn, "Springwood")


def test_resolve_station_exact_name_works(conn):
    _seed_stations(conn)
    _seed_daily(conn, {100: 170.0})
    r = resolve(conn, "Ampol Springwood")
    assert r.spec == "station:100"


# ---------------------------------------------------------------------------
# resolve_members()
# ---------------------------------------------------------------------------

def test_resolve_members_lga(conn):
    upsert_stations(conn, _PENRITH_STATIONS)
    rows = [(s["station_code"], "E10", "2024-03-01", 165.0) for s in _PENRITH_STATIONS]
    upsert_daily_prices(conn, rows)
    conn.commit()
    members = resolve_members(conn, "lga:Penrith")
    assert len(members) == 4
    assert all(m.kind == "station" for m in members)


def test_resolve_members_brand(conn):
    stations = [
        {**_ST_AMPOL, "station_code": 600 + i, "address": f"{i} Ampol Rd, Sub{i}",
         "suburb": f"Sub{i}", "postcode": "2000"}
        for i in range(3)
    ]
    upsert_stations(conn, stations)
    rows = [(s["station_code"], "E10", "2024-03-01", 168.0) for s in stations]
    upsert_daily_prices(conn, rows)
    conn.commit()
    members = resolve_members(conn, "brand:Ampol")
    assert len(members) == 3
    assert all(m.kind == "station" for m in members)


def test_resolve_members_sydney_returns_empty(conn):
    assert resolve_members(conn, "sydney") == []


def test_resolve_members_station_returns_empty(conn):
    _seed_stations(conn)
    _seed_daily(conn, {100: 170.0})
    assert resolve_members(conn, "station:100") == []


def test_resolve_members_unknown_lga_returns_empty(conn):
    assert resolve_members(conn, "lga:NoSuchPlace") == []


def test_resolve_members_only_returns_stations_with_data(conn):
    """Stations without daily_prices entries are excluded from members."""
    upsert_stations(conn, _PENRITH_STATIONS)
    # Only seed data for two of the four stations
    rows = [(s["station_code"], "E10", "2024-03-01", 165.0) for s in _PENRITH_STATIONS[:2]]
    upsert_daily_prices(conn, rows)
    conn.commit()
    members = resolve_members(conn, "lga:Penrith")
    assert len(members) == 2


# ---------------------------------------------------------------------------
# enumerate_groups()
# ---------------------------------------------------------------------------

def test_enumerate_groups_returns_lgas_with_stations(conn):
    upsert_stations(conn, _PENRITH_STATIONS)
    groups = enumerate_groups(conn)
    assert "Penrith" in groups["lgas"]


def test_enumerate_groups_lgas_excludes_empty(conn):
    groups = enumerate_groups(conn)
    # No stations → no LGAs
    assert groups["lgas"] == []


def test_enumerate_groups_brands_threshold(conn):
    # Only 2 Ampol stations — below default min_stations=3
    stations = [
        {**_ST_AMPOL, "station_code": 700 + i, "address": f"{i} Ampol St, Suburb{i}",
         "suburb": f"Suburb{i}", "postcode": "2000"}
        for i in range(2)
    ]
    upsert_stations(conn, stations)
    rows = [(s["station_code"], "E10", "2024-03-01", 168.0) for s in stations]
    upsert_daily_prices(conn, rows)
    conn.commit()
    groups = enumerate_groups(conn)
    assert "Ampol" not in groups["brands"]
