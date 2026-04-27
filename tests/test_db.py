"""Tests for fuel_signal.db — schema, address normalization, and load helpers."""

import csv
import pathlib

import pytest

from fuel_signal.db import (
    backfill_station_suburbs,
    create_schema,
    daily_average_e10,
    db_summary,
    insert_prices,
    is_file_loaded,
    load_all_cleaned,
    load_all_snapshots,
    load_cleaned_csv,
    load_snapshot_csv,
    mark_file_loaded,
    normalize_address,
    open_db,
    station_price_series,
    upsert_stations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


def _write_snapshot_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["station_code", "name", "address", "suburb", "postcode", "brand", "fuel_code", "price", "date"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def _write_cleaned_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ServiceStationName", "Address", "Suburb", "Postcode",
                           "Brand", "FuelCode", "PriceUpdatedDate", "Price"]
        )
        writer.writeheader()
        writer.writerows(rows)


_STATION = {
    "station_code": 1001,
    "name": "Shell Springwood",
    "address": "1 Main Street, Springwood",
    "suburb": "Springwood",
    "postcode": "2777",
    "brand": "Shell",
    "latitude": -33.7,
    "longitude": 150.5,
}


# ---------------------------------------------------------------------------
# normalize_address
# ---------------------------------------------------------------------------

def test_normalize_strips_state_postcode():
    assert normalize_address("283 MANNS RD, WEST GOSFORD NSW 2250") == "283 manns road west gosford"


def test_normalize_expands_rd():
    assert normalize_address("5 HIGH RD, PENRITH NSW 2750") == "5 high road penrith"


def test_normalize_expands_st_in_street_not_suburb():
    # "ST" in street portion → STREET; "ST MARYS" in suburb → untouched
    result = normalize_address("5 ADELAIDE ST, ST MARYS NSW 2760")
    assert result == "5 adelaide street st marys"


def test_normalize_expands_ave():
    assert normalize_address("10 PARK AVE, KATOOMBA NSW 2780") == "10 park avenue katoomba"


def test_normalize_expands_hwy():
    assert normalize_address("100 GREAT WESTERN HWY, BLAXLAND NSW 2774") == "100 great western highway blaxland"


def test_normalize_expands_cres():
    assert normalize_address("3 ROSE CRES, PENRITH NSW 2750") == "3 rose crescent penrith"


def test_normalize_api_format_no_state():
    # API addresses typically lack state/postcode suffix
    assert normalize_address("1 Main Street, Springwood") == "1 main street springwood"


def test_normalize_uppercase_lowercase_equivalent():
    a = normalize_address("283 Manns Road, West Gosford")
    b = normalize_address("283 MANNS ROAD, WEST GOSFORD NSW 2250")
    assert a == b


def test_normalize_strips_punctuation():
    # "ST." with trailing period in the street portion expands correctly
    assert normalize_address("12 Test St., Suburb NSW 2000") == "12 test street suburb"


# ---------------------------------------------------------------------------
# create_schema
# ---------------------------------------------------------------------------

def test_schema_creates_stations_table(conn):
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "stations" in tables
    assert "prices" in tables


def test_schema_idempotent(conn):
    create_schema(conn)  # second call should not raise
    count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------------------
# upsert_stations
# ---------------------------------------------------------------------------

def test_upsert_stations_inserts(conn):
    n = upsert_stations(conn, [_STATION])
    assert n == 1
    row = conn.execute("SELECT name, suburb, postcode FROM stations WHERE station_code=1001").fetchone()
    assert row == ("Shell Springwood", "Springwood", "2777")


def test_upsert_stations_normalizes_address(conn):
    upsert_stations(conn, [_STATION])
    addr = conn.execute("SELECT address_normalized FROM stations WHERE station_code=1001").fetchone()[0]
    assert addr == "1 main street springwood"


def test_upsert_stations_updates_name_on_rebrand(conn):
    upsert_stations(conn, [_STATION])
    rebranded = {**_STATION, "name": "Ampol Springwood", "brand": "Ampol"}
    upsert_stations(conn, [rebranded])
    row = conn.execute("SELECT name, brand FROM stations WHERE station_code=1001").fetchone()
    assert row == ("Ampol Springwood", "Ampol")


def test_upsert_stations_preserves_latlon_when_null(conn):
    upsert_stations(conn, [_STATION])
    # Second upsert has no lat/lon (like a snapshot CSV load)
    no_coords = {**_STATION, "latitude": None, "longitude": None}
    upsert_stations(conn, [no_coords])
    row = conn.execute("SELECT latitude, longitude FROM stations WHERE station_code=1001").fetchone()
    assert row == (-33.7, 150.5)


def test_upsert_stations_ignores_address_conflict(conn):
    """Two station_codes with the same address — second insert is silently dropped."""
    upsert_stations(conn, [_STATION])
    duplicate_addr = {**_STATION, "station_code": 9999, "name": "Other Station"}
    upsert_stations(conn, [duplicate_addr])  # should not raise
    count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    assert count == 1


def test_upsert_stations_populates_council_from_postcode(conn):
    upsert_stations(conn, [_STATION])  # postcode 2777 → Blue Mountains
    council = conn.execute(
        "SELECT council FROM stations WHERE station_code=1001"
    ).fetchone()[0]
    assert council == "Blue Mountains"


def test_upsert_stations_council_none_for_unknown_postcode(conn):
    unknown_pc = {**_STATION, "station_code": 2002, "postcode": "9999"}
    upsert_stations(conn, [unknown_pc])
    council = conn.execute(
        "SELECT council FROM stations WHERE station_code=2002"
    ).fetchone()[0]
    assert council is None


def test_upsert_stations_council_set_for_inner_sydney(conn):
    inner = {**_STATION, "station_code": 2003, "address": "5 Main St, Sydney", "postcode": "2000"}
    upsert_stations(conn, [inner])
    council = conn.execute(
        "SELECT council FROM stations WHERE station_code=2003"
    ).fetchone()[0]
    assert council == "Sydney"


# ---------------------------------------------------------------------------
# insert_prices
# ---------------------------------------------------------------------------

def test_insert_prices(conn):
    upsert_stations(conn, [_STATION])
    insert_prices(conn, [{"station_code": 1001, "fuel_code": "E10", "price_date": "2024-01-15", "price_cents": 180.0}])
    assert station_price_series(conn, 1001) == [("2024-01-15", 180.0)]
    source = conn.execute(
        "SELECT ps.code FROM prices p JOIN price_sources ps ON p.source_id = ps.id WHERE p.station_code=1001"
    ).fetchone()[0]
    assert source == "h"  # default source


def test_insert_prices_snapshot_source(conn):
    upsert_stations(conn, [_STATION])
    price = {"station_code": 1001, "fuel_code": "E10", "price_date": "2024-01-15", "price_cents": 180.0}
    insert_prices(conn, [price], source="s")
    source = conn.execute(
        "SELECT ps.code FROM prices p JOIN price_sources ps ON p.source_id = ps.id"
    ).fetchone()[0]
    assert source == "s"


def test_insert_prices_ignores_duplicates(conn):
    upsert_stations(conn, [_STATION])
    price = {"station_code": 1001, "fuel_code": "E10", "price_date": "2024-01-15", "price_cents": 180.0}
    insert_prices(conn, [price])
    insert_prices(conn, [price])  # duplicate — should be ignored, not raise
    count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# load_snapshot_csv
# ---------------------------------------------------------------------------

def test_load_snapshot_csv(conn, tmp_path):
    snap = tmp_path / "snap.csv"
    _write_snapshot_csv(snap, [{
        "station_code": 1001, "name": "Shell Springwood",
        "address": "1 Main Street, Springwood", "suburb": "Springwood",
        "postcode": "2777", "brand": "Shell", "fuel_code": "E10", "price": "175.9", "date": "2024-03-01",
    }])
    s, p = load_snapshot_csv(conn, snap)
    assert s == 1
    assert p == 1
    assert station_price_series(conn, 1001) == [("2024-03-01", 175.9)]
    source = conn.execute(
        "SELECT ps.code FROM prices p JOIN price_sources ps ON p.source_id = ps.id WHERE p.station_code=1001"
    ).fetchone()[0]
    assert source == "s"


def test_load_snapshot_csv_skips_price_for_duplicate_address_station(conn, tmp_path):
    """Two station_codes at the same normalised address: second is dropped from stations,
    its price row must not trigger a FK violation."""
    snap = tmp_path / "snap.csv"
    _write_snapshot_csv(snap, [
        {"station_code": 1001, "name": "Shell Springwood",
         "address": "1 Main Street, Springwood", "suburb": "Springwood",
         "postcode": "2777", "brand": "Shell", "fuel_code": "E10", "price": "175.9", "date": "2024-03-01"},
        # Same normalised address, different station_code — second will be dropped
        {"station_code": 1002, "name": "Other Springwood",
         "address": "1 Main Street, Springwood", "suburb": "Springwood",
         "postcode": "2777", "brand": "Other", "fuel_code": "E10", "price": "176.0", "date": "2024-03-01"},
    ])
    s, p = load_snapshot_csv(conn, snap)  # must not raise
    assert p == 1  # only the first station's price inserted


def test_load_snapshot_csv_skips_bad_price(conn, tmp_path):
    snap = tmp_path / "snap.csv"
    _write_snapshot_csv(snap, [{
        "station_code": 1001, "name": "Shell Springwood",
        "address": "1 Main Street, Springwood", "suburb": "Springwood",
        "postcode": "2777", "brand": "Shell", "fuel_code": "E10", "price": "NOT_A_PRICE", "date": "2024-03-01",
    }])
    _, p = load_snapshot_csv(conn, snap)
    assert p == 0


def test_load_snapshot_csv_reads_fuel_code_column(conn, tmp_path):
    snap = tmp_path / "snap.csv"
    _write_snapshot_csv(snap, [{
        "station_code": 1001, "name": "Shell Springwood",
        "address": "1 Main Street, Springwood", "suburb": "Springwood",
        "postcode": "2777", "brand": "Shell", "fuel_code": "U91", "price": "180.0", "date": "2024-03-01",
    }])
    _, p = load_snapshot_csv(conn, snap)
    assert p == 1
    # Price must be stored under U91, not the E10 fallback.
    from fuel_signal.db import station_price_series
    u91 = station_price_series(conn, 1001, fuel_code="U91")
    assert u91 == [("2024-03-01", 180.0)]



def test_load_snapshot_csv_postcode_filter(conn, tmp_path):
    snap = tmp_path / "snap.csv"
    _write_snapshot_csv(snap, [
        {"station_code": 1001, "name": "Shell Springwood", "address": "1 Main St, Springwood",
         "suburb": "Springwood", "postcode": "2777", "brand": "Shell",
         "fuel_code": "E10", "price": "175.9", "date": "2024-03-01"},
        {"station_code": 3001, "name": "Ampol Broken Hill", "address": "1 Argent St, Broken Hill",
         "suburb": "Broken Hill", "postcode": "2880", "brand": "Ampol",
         "fuel_code": "E10", "price": "190.0", "date": "2024-03-01"},
    ])
    s, p = load_snapshot_csv(conn, snap, postcodes=frozenset({"2777"}))
    assert s == 1
    assert p == 1
    codes = {r[0] for r in conn.execute("SELECT station_code FROM stations")}
    assert codes == {1001}


def test_load_snapshot_csv_fuel_codes_filter(conn, tmp_path):
    snap = tmp_path / "snap.csv"
    _write_snapshot_csv(snap, [
        {"station_code": 1001, "name": "Shell Springwood", "address": "1 Main St, Springwood",
         "suburb": "Springwood", "postcode": "2777", "brand": "Shell",
         "fuel_code": "E10", "price": "175.9", "date": "2024-03-01"},
        {"station_code": 1001, "name": "Shell Springwood", "address": "1 Main St, Springwood",
         "suburb": "Springwood", "postcode": "2777", "brand": "Shell",
         "fuel_code": "U91", "price": "180.0", "date": "2024-03-01"},
    ])
    _, p = load_snapshot_csv(conn, snap, fuel_codes={"E10"})
    assert p == 1
    price_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    assert price_count == 1


def test_load_all_snapshots(conn, tmp_path):
    snaps_dir = tmp_path / "snapshots"
    for i, date in enumerate(["2024-03-01", "2024-03-02"]):
        _write_snapshot_csv(snaps_dir / "2024" / "03" / f"{date}.csv", [{
            "station_code": 1001, "name": "Shell Springwood",
            "address": "1 Main Street, Springwood", "suburb": "Springwood",
            "postcode": "2777", "brand": "Shell", "fuel_code": "E10", "price": str(175.9 + i), "date": date,
        }])
    s, p = load_all_snapshots(conn, snaps_dir)
    assert p == 2


def test_load_all_snapshots_with_filters(conn, tmp_path):
    snaps_dir = tmp_path / "snapshots"
    # Two rows: one metro E10, one rural U91 — filter should admit only the metro E10.
    _write_snapshot_csv(snaps_dir / "2024" / "03" / "2024-03-01.csv", [
        {"station_code": 1001, "name": "Shell Springwood", "address": "1 Main St, Springwood",
         "suburb": "Springwood", "postcode": "2777", "brand": "Shell",
         "fuel_code": "E10", "price": "175.9", "date": "2024-03-01"},
        {"station_code": 3001, "name": "Ampol Broken Hill", "address": "1 Argent St, Broken Hill",
         "suburb": "Broken Hill", "postcode": "2880", "brand": "Ampol",
         "fuel_code": "U91", "price": "190.0", "date": "2024-03-01"},
    ])
    _, p = load_all_snapshots(conn, snaps_dir, postcodes=frozenset({"2777"}), fuel_codes={"E10"})
    assert p == 1


# ---------------------------------------------------------------------------
# load_cleaned_csv
# ---------------------------------------------------------------------------

def test_load_cleaned_csv_matches_by_address(conn, tmp_path):
    # Station pre-loaded in DB with API address
    upsert_stations(conn, [_STATION])
    # Historical CSV uses abbreviated address with state suffix
    cleaned = tmp_path / "hist.csv"
    _write_cleaned_csv(cleaned, [{
        "ServiceStationName": "Shell Springwood",
        "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
        "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
        "FuelCode": "E10", "PriceUpdatedDate": "2022-08-15 00:00:00", "Price": "168.5",
    }])
    inserted, skipped = load_cleaned_csv(conn, cleaned)
    assert inserted == 1
    assert skipped == 0
    source = conn.execute(
        "SELECT ps.code FROM prices p JOIN price_sources ps ON p.source_id = ps.id"
    ).fetchone()[0]
    assert source == "h"


def test_load_cleaned_csv_skips_unknown_address(conn, tmp_path):
    # No stations in DB → everything skipped
    cleaned = tmp_path / "hist.csv"
    _write_cleaned_csv(cleaned, [{
        "ServiceStationName": "Shell Springwood",
        "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
        "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
        "FuelCode": "E10", "PriceUpdatedDate": "2022-08-15 00:00:00", "Price": "168.5",
    }])
    inserted, skipped = load_cleaned_csv(conn, cleaned)
    assert inserted == 0
    assert skipped == 1


def test_load_cleaned_csv_truncates_datetime_to_date(conn, tmp_path):
    upsert_stations(conn, [_STATION])
    cleaned = tmp_path / "hist.csv"
    _write_cleaned_csv(cleaned, [{
        "ServiceStationName": "Shell Springwood",
        "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
        "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
        "FuelCode": "E10", "PriceUpdatedDate": "2022-08-15 09:30:00", "Price": "168.5",
    }])
    load_cleaned_csv(conn, cleaned)
    date = conn.execute("SELECT price_date FROM prices").fetchone()[0]
    assert date == 20220815  # stored as YYYYMMDD integer


def test_load_all_cleaned(conn, tmp_path):
    upsert_stations(conn, [_STATION])
    cleaned_dir = tmp_path / "cleaned"
    for i in range(3):
        _write_cleaned_csv(cleaned_dir / f"file{i}.csv", [{
            "ServiceStationName": "Shell Springwood",
            "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
            "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
            "FuelCode": "E10",
            "PriceUpdatedDate": f"2022-0{i+1}-15 00:00:00",
            "Price": str(160.0 + i),
        }])
    inserted, skipped = load_all_cleaned(conn, cleaned_dir)
    assert inserted == 3
    assert skipped == 0


# ---------------------------------------------------------------------------
# backfill_station_suburbs
# ---------------------------------------------------------------------------

def test_backfill_fills_blank_suburb(conn):
    station = {**_STATION, "suburb": ""}
    upsert_stations(conn, [station])
    n = backfill_station_suburbs(conn, {1001: "Springwood"})
    assert n == 1
    row = conn.execute("SELECT suburb FROM stations WHERE station_code = 1001").fetchone()
    assert row[0] == "Springwood"


def test_backfill_does_not_overwrite_existing_suburb(conn):
    upsert_stations(conn, [_STATION])  # suburb = "Springwood"
    n = backfill_station_suburbs(conn, {1001: "WrongSuburb"})
    assert n == 0
    row = conn.execute("SELECT suburb FROM stations WHERE station_code = 1001").fetchone()
    assert row[0] == "Springwood"


def test_backfill_returns_zero_for_empty_input(conn):
    assert backfill_station_suburbs(conn, {}) == 0


def test_load_cleaned_csv_populates_suburb_backfill(conn, tmp_path):
    station = {**_STATION, "suburb": ""}
    upsert_stations(conn, [station])
    cleaned = tmp_path / "file.csv"
    _write_cleaned_csv(cleaned, [{
        "ServiceStationName": "Shell Springwood",
        "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
        "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
        "FuelCode": "E10", "PriceUpdatedDate": "2022-01-15", "Price": "160.0",
    }])
    backfill: dict[int, str] = {}
    load_cleaned_csv(conn, cleaned, suburb_backfill=backfill)
    assert backfill == {1001: "Springwood"}


def test_load_cleaned_csv_backfill_first_seen_wins(conn, tmp_path):
    station = {**_STATION, "suburb": ""}
    upsert_stations(conn, [station])
    cleaned = tmp_path / "file.csv"
    _write_cleaned_csv(cleaned, [
        {"ServiceStationName": "Shell Springwood", "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
         "Suburb": "FirstSuburb", "Postcode": "2777", "Brand": "Shell",
         "FuelCode": "E10", "PriceUpdatedDate": "2022-01-15", "Price": "160.0"},
        {"ServiceStationName": "Shell Springwood", "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
         "Suburb": "SecondSuburb", "Postcode": "2777", "Brand": "Shell",
         "FuelCode": "E10", "PriceUpdatedDate": "2022-01-16", "Price": "161.0"},
    ])
    backfill: dict[int, str] = {}
    load_cleaned_csv(conn, cleaned, suburb_backfill=backfill)
    assert backfill[1001] == "FirstSuburb"


def test_load_all_cleaned_backfills_blank_suburbs(conn, tmp_path):
    station = {**_STATION, "suburb": ""}
    upsert_stations(conn, [station])
    cleaned_dir = tmp_path / "cleaned"
    _write_cleaned_csv(cleaned_dir / "file.csv", [{
        "ServiceStationName": "Shell Springwood",
        "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
        "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
        "FuelCode": "E10", "PriceUpdatedDate": "2022-01-15", "Price": "160.0",
    }])
    load_all_cleaned(conn, cleaned_dir)
    row = conn.execute("SELECT suburb FROM stations WHERE station_code = 1001").fetchone()
    assert row[0] == "Springwood"


# ---------------------------------------------------------------------------
# loaded_files tracking
# ---------------------------------------------------------------------------

def test_is_file_loaded_false_initially(conn):
    assert not is_file_loaded(conn, "2024-03-01.csv")


def test_mark_file_loaded_then_is_loaded(conn):
    mark_file_loaded(conn, "2024-03-01.csv")
    assert is_file_loaded(conn, "2024-03-01.csv")


def test_mark_file_loaded_idempotent(conn):
    mark_file_loaded(conn, "2024-03-01.csv")
    mark_file_loaded(conn, "2024-03-01.csv")  # should not raise
    assert is_file_loaded(conn, "2024-03-01.csv")


def test_load_all_snapshots_skips_already_loaded(conn, tmp_path):
    snaps_dir = tmp_path / "snapshots"
    _write_snapshot_csv(snaps_dir / "2024" / "03" / "2024-03-01.csv", [{
        "station_code": 1001, "name": "Shell Springwood",
        "address": "1 Main Street, Springwood", "suburb": "Springwood",
        "postcode": "2777", "brand": "Shell", "fuel_code": "E10", "price": "175.9", "date": "2024-03-01",
    }])
    load_all_snapshots(conn, snaps_dir)
    # Second load should skip the file and insert 0 new prices
    _, p = load_all_snapshots(conn, snaps_dir)
    assert p == 0


def test_load_all_snapshots_force_reloads(conn, tmp_path):
    snaps_dir = tmp_path / "snapshots"
    _write_snapshot_csv(snaps_dir / "2024" / "03" / "2024-03-01.csv", [{
        "station_code": 1001, "name": "Shell Springwood",
        "address": "1 Main Street, Springwood", "suburb": "Springwood",
        "postcode": "2777", "brand": "Shell", "fuel_code": "E10", "price": "175.9", "date": "2024-03-01",
    }])
    load_all_snapshots(conn, snaps_dir)
    # force=True bypasses the skip; INSERT OR IGNORE means 0 new prices but no error
    _, p = load_all_snapshots(conn, snaps_dir, force=True)
    assert p == 0  # already in DB via INSERT OR IGNORE, but file was not skipped


def test_load_all_snapshots_marks_file_loaded(conn, tmp_path):
    snaps_dir = tmp_path / "snapshots"
    _write_snapshot_csv(snaps_dir / "2024" / "03" / "2024-03-01.csv", [{
        "station_code": 1001, "name": "Shell Springwood",
        "address": "1 Main Street, Springwood", "suburb": "Springwood",
        "postcode": "2777", "brand": "Shell", "fuel_code": "E10", "price": "175.9", "date": "2024-03-01",
    }])
    load_all_snapshots(conn, snaps_dir)
    assert is_file_loaded(conn, "2024-03-01.csv")


def test_load_all_cleaned_skips_already_loaded(conn, tmp_path):
    upsert_stations(conn, [_STATION])
    cleaned_dir = tmp_path / "cleaned"
    _write_cleaned_csv(cleaned_dir / "hist.csv", [{
        "ServiceStationName": "Shell Springwood",
        "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
        "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
        "FuelCode": "E10", "PriceUpdatedDate": "2022-08-15 00:00:00", "Price": "168.5",
    }])
    load_all_cleaned(conn, cleaned_dir)
    inserted, _ = load_all_cleaned(conn, cleaned_dir)
    assert inserted == 0


def test_load_all_cleaned_force_reloads(conn, tmp_path):
    upsert_stations(conn, [_STATION])
    cleaned_dir = tmp_path / "cleaned"
    _write_cleaned_csv(cleaned_dir / "hist.csv", [{
        "ServiceStationName": "Shell Springwood",
        "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
        "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
        "FuelCode": "E10", "PriceUpdatedDate": "2022-08-15 00:00:00", "Price": "168.5",
    }])
    load_all_cleaned(conn, cleaned_dir)
    # force=True bypasses skip; INSERT OR IGNORE means 0 new prices but file was processed
    inserted, _ = load_all_cleaned(conn, cleaned_dir, force=True)
    assert inserted == 0  # already in DB, not a duplicate insert


def test_load_all_cleaned_marks_file_loaded(conn, tmp_path):
    upsert_stations(conn, [_STATION])
    cleaned_dir = tmp_path / "cleaned"
    _write_cleaned_csv(cleaned_dir / "hist.csv", [{
        "ServiceStationName": "Shell Springwood",
        "Address": "1 MAIN ST, SPRINGWOOD NSW 2777",
        "Suburb": "Springwood", "Postcode": "2777", "Brand": "Shell",
        "FuelCode": "E10", "PriceUpdatedDate": "2022-08-15 00:00:00", "Price": "168.5",
    }])
    load_all_cleaned(conn, cleaned_dir)
    assert is_file_loaded(conn, "hist.csv")


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _seed(conn):
    upsert_stations(conn, [_STATION])
    insert_prices(conn, [
        {"station_code": 1001, "fuel_code": "E10", "price_date": "2024-01-10", "price_cents": 175.0},
        {"station_code": 1001, "fuel_code": "E10", "price_date": "2024-01-11", "price_cents": 180.0},
        {"station_code": 1001, "fuel_code": "U91", "price_date": "2024-01-10", "price_cents": 177.0},
    ])


def test_daily_average_e10(conn):
    _seed(conn)
    result = daily_average_e10(conn)
    assert len(result) == 2
    assert result[0] == ("2024-01-10", 175.0)


def test_daily_average_e10_start_date(conn):
    _seed(conn)
    result = daily_average_e10(conn, start_date="2024-01-11")
    assert len(result) == 1
    assert result[0][0] == "2024-01-11"


def test_station_price_series(conn):
    _seed(conn)
    result = station_price_series(conn, 1001, "E10")
    assert result == [("2024-01-10", 175.0), ("2024-01-11", 180.0)]


def test_db_summary(conn):
    _seed(conn)
    s = db_summary(conn)
    assert s["station_count"] == 1
    assert s["price_count"] == 3
    assert s["earliest_date"] == "2024-01-10"
    assert s["latest_date"] == "2024-01-11"


def test_db_summary_empty(conn):
    s = db_summary(conn)
    assert s["station_count"] == 0
    assert s["earliest_date"] == "—"
