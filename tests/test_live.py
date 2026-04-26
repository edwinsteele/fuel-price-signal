"""Tests for fuel_signal.live — snapshot row construction and CSV persistence."""

import csv
import datetime
import pathlib

import pytest
import responses as rsps

from fuel_signal.live import (
    FUELAPI_PRICES_URL,
    FUELAPI_TOKEN_URL,
    SNAPSHOT_COLUMNS,
    _station_suburb_postcode,
    build_snapshot_rows,
    collect_snapshot,
    get_token,
    save_snapshot,
)

# ---------------------------------------------------------------------------
# Sample API response fixture
# ---------------------------------------------------------------------------

_API_RESPONSE = {
    "stations": [
        {
            "brandid": "1",
            "stationid": "s1",
            "brand": "Shell",
            "code": "1001",
            "name": "Shell Springwood",
            "address": "1 Main Street, Springwood NSW 2777",
            "location": {"latitude": -33.7, "longitude": 150.5},
        },
        {
            "brandid": "2",
            "stationid": "s2",
            "brand": "BP",
            "code": "2001",
            "name": "BP Blaxland",
            "address": "55 Old Bathurst Road, Blaxland NSW 2774",
            "location": {"latitude": -33.75, "longitude": 150.6},
        },
        {
            "brandid": "3",
            "stationid": "s3",
            "brand": "Ampol",
            "code": "3001",
            "name": "Ampol Broken Hill",
            "address": "1 Argent Street, Broken Hill NSW 2880",  # rural — postcode 2880 > 2799
            "location": {"latitude": -31.96, "longitude": 141.46},
        },
    ],
    "prices": [
        {"stationcode": "1001", "fueltype": "E10",  "price": 175.9, "lastupdated": "01/04/2024 09:00:00 AM"},
        {"stationcode": "1001", "fueltype": "U91",  "price": 179.9, "lastupdated": "01/04/2024 09:00:00 AM"},
        {"stationcode": "2001", "fueltype": "E10",  "price": 173.5, "lastupdated": "01/04/2024 10:00:00 AM"},
        {"stationcode": "3001", "fueltype": "E10",  "price": 180.0, "lastupdated": "01/04/2024 10:00:00 AM"},
    ],
}

_SNAPSHOT_DATE = datetime.date(2024, 4, 1)


# ---------------------------------------------------------------------------
# _station_suburb_postcode
# ---------------------------------------------------------------------------

def test_suburb_postcode_full_format():
    suburb, pc = _station_suburb_postcode("1 Main Street, Springwood NSW 2777")
    assert suburb == "Springwood"
    assert pc == "2777"


def test_suburb_postcode_multiword_suburb():
    suburb, pc = _station_suburb_postcode("55 Old Bathurst Road, BLAXLAND EAST NSW 2774")
    assert suburb == "Blaxland East"
    assert pc == "2774"


def test_suburb_postcode_no_state():
    _, pc = _station_suburb_postcode("1 Main Street, Springwood 2777")
    assert pc == "2777"


def test_suburb_postcode_multi_comma_address():
    # "Shop 1, Gilchrist Dr, Campbelltown NSW 2560" — must capture only last segment
    suburb, pc = _station_suburb_postcode("Shop 1, Gilchrist Dr, Campbelltown NSW 2560")
    assert suburb == "Campbelltown"
    assert pc == "2560"


def test_suburb_postcode_title_case():
    # API returns UPPERCASE suburbs — must be normalised to title case
    suburb, _ = _station_suburb_postcode("1 Main Street, SPRINGWOOD NSW 2777")
    assert suburb == "Springwood"


def test_suburb_postcode_no_comma_simple():
    # "123 Main Rd Suburb NSW 2000" — space instead of comma before suburb
    suburb, pc = _station_suburb_postcode("262 - 272 VICTORIA RD Rydalmere NSW 2116")
    assert suburb == "Rydalmere"
    assert pc == "2116"


def test_suburb_postcode_no_comma_highway():
    # Highway number + no comma — "33351 NEWELL HWY BOGGABILLA NSW 2409"
    suburb, pc = _station_suburb_postcode("33351 NEWELL HWY BOGGABILLA NSW 2409")
    assert suburb == "Boggabilla"
    assert pc == "2409"


def test_suburb_postcode_no_comma_corner():
    # Corner address with multiple street types — suburb is after the last one
    suburb, pc = _station_suburb_postcode(
        "Corner Pacific Hwy and Halls Road Coffs Harbour NSW 2450"
    )
    assert suburb == "Coffs Harbour"
    assert pc == "2450"


def test_suburb_postcode_no_comma_multiword_suburb():
    # "842-844 DAVID ST North Albury NSW 2640"
    suburb, pc = _station_suburb_postcode("842-844 DAVID ST North Albury NSW 2640")
    assert suburb == "North Albury"
    assert pc == "2640"


def test_suburb_postcode_no_match_returns_empty():
    suburb, pc = _station_suburb_postcode("no postcode here")
    assert suburb == ""
    assert pc == ""


# ---------------------------------------------------------------------------
# build_snapshot_rows
# ---------------------------------------------------------------------------

def test_build_filters_to_e10():
    rows = build_snapshot_rows(_API_RESPONSE, snapshot_date=_SNAPSHOT_DATE)
    # Station 1001 has both E10 and U91; only one E10 row should appear for it
    rows_for_1001 = [r for r in rows if r["station_code"] == 1001]
    assert len(rows_for_1001) == 1
    assert rows_for_1001[0]["price"] == 175.9


def test_build_excludes_rural():
    rows = build_snapshot_rows(_API_RESPONSE, snapshot_date=_SNAPSHOT_DATE)
    codes = {r["station_code"] for r in rows}
    assert 3001 not in codes  # Broken Hill (2880) — outside 2000–2799


def test_build_includes_metro_stations():
    rows = build_snapshot_rows(_API_RESPONSE, snapshot_date=_SNAPSHOT_DATE)
    codes = {r["station_code"] for r in rows}
    assert 1001 in codes
    assert 2001 in codes


def test_build_row_fields():
    rows = build_snapshot_rows(_API_RESPONSE, snapshot_date=_SNAPSHOT_DATE)
    row = next(r for r in rows if r["station_code"] == 1001)
    assert row["name"] == "Shell Springwood"
    assert row["brand"] == "Shell"
    assert row["price"] == 175.9
    assert row["date"] == "2024-04-01"
    assert row["postcode"] == "2777"
    assert row["suburb"] == "Springwood"


def test_build_uses_today_if_no_date():
    rows = build_snapshot_rows(_API_RESPONSE)
    today = datetime.date.today().isoformat()
    assert all(r["date"] == today for r in rows)


def test_build_custom_fuel_code():
    rows = build_snapshot_rows(_API_RESPONSE, fuel_code="U91", snapshot_date=_SNAPSHOT_DATE)
    assert len(rows) == 1
    assert rows[0]["station_code"] == 1001


def test_build_excludes_non_metro_postcodes():
    rows = build_snapshot_rows(
        _API_RESPONSE,
        snapshot_date=_SNAPSHOT_DATE,
        postcodes=frozenset({"2777"}),  # only Springwood
    )
    codes = {r["station_code"] for r in rows}
    assert codes == {1001}


# ---------------------------------------------------------------------------
# save_snapshot
# ---------------------------------------------------------------------------

def test_save_snapshot_writes_csv(tmp_path):
    rows = build_snapshot_rows(_API_RESPONSE, snapshot_date=_SNAPSHOT_DATE)
    path = save_snapshot(rows, _SNAPSHOT_DATE, tmp_path / "snapshots")
    assert path.exists()
    assert path.name == "2024-04-01.csv"
    assert path.parent.name == "04"
    assert path.parent.parent.name == "2024"


def test_save_snapshot_csv_columns(tmp_path):
    rows = build_snapshot_rows(_API_RESPONSE, snapshot_date=_SNAPSHOT_DATE)
    path = save_snapshot(rows, _SNAPSHOT_DATE, tmp_path / "snapshots")
    with open(path, newline="") as f:
        headers = next(csv.reader(f))
    assert headers == list(SNAPSHOT_COLUMNS)


def test_save_snapshot_row_count(tmp_path):
    rows = build_snapshot_rows(_API_RESPONSE, snapshot_date=_SNAPSHOT_DATE)
    path = save_snapshot(rows, _SNAPSHOT_DATE, tmp_path / "snapshots")
    with open(path, newline="") as f:
        data = list(csv.DictReader(f))
    assert len(data) == len(rows)


def test_save_snapshot_creates_parent_dirs(tmp_path):
    rows = build_snapshot_rows(_API_RESPONSE, snapshot_date=_SNAPSHOT_DATE)
    save_snapshot(rows, _SNAPSHOT_DATE, tmp_path / "deep" / "nested" / "snapshots")
    # No FileNotFoundError means dirs were created


# ---------------------------------------------------------------------------
# get_token (mocked HTTP)
# ---------------------------------------------------------------------------

@rsps.activate
def test_get_token_success():
    rsps.add(rsps.GET, FUELAPI_TOKEN_URL, json={"access_token": "tok123"})
    token = get_token("key", "secret")
    assert token == "tok123"


@rsps.activate
def test_get_token_raises_on_http_error():
    rsps.add(rsps.GET, FUELAPI_TOKEN_URL, status=401)
    with pytest.raises(Exception):
        get_token("bad", "creds")


# ---------------------------------------------------------------------------
# collect_snapshot (mocked HTTP end-to-end)
# ---------------------------------------------------------------------------

@rsps.activate
def test_collect_snapshot_writes_file(tmp_path):
    rsps.add(rsps.GET, FUELAPI_TOKEN_URL, json={"access_token": "tok123"})
    rsps.add(rsps.GET, FUELAPI_PRICES_URL, json=_API_RESPONSE)
    path = collect_snapshot(
        api_key="key",
        api_secret="secret",
        snapshots_dir=tmp_path / "snapshots",
        snapshot_date=_SNAPSHOT_DATE,
    )
    assert path.exists()
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    assert rows[0]["date"] == "2024-04-01"


@rsps.activate
def test_collect_snapshot_raises_without_creds(tmp_path):
    with pytest.raises(RuntimeError, match="FUELAPI_API_KEY"):
        collect_snapshot(api_key="", api_secret="", snapshots_dir=tmp_path)
