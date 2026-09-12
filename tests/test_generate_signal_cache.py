"""Tests for fuel_signal.generate_signal_cache — the #416 nightly precompute CLI."""

from __future__ import annotations

import datetime
import json

from click.testing import CliRunner

import fuel_signal.db as db
from fuel_signal.api_v1 import decode_cache_entry
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.generate_signal_cache import main as generate_signal_cache_cli

_CYCLE_LENGTH = 46


def _sawtooth_series(n_cycles: float = 4.0) -> list[tuple[str, float]]:
    start_date = datetime.date(2020, 1, 1)
    total_days = int(n_cycles * _CYCLE_LENGTH)
    result = []
    for day in range(total_days):
        pos = day % _CYCLE_LENGTH
        if pos < 3:
            price = 150.0 + 25.0 * (pos / 3)
        else:
            price = 150.0 + 25.0 * (1.0 - (pos - 3) / (_CYCLE_LENGTH - 3))
        result.append(((start_date + datetime.timedelta(days=day)).isoformat(), price))
    return result


def _build_db(db_path, *, create_schema: bool = True) -> list[tuple[str, float]]:
    conn = db.open_db(db_path)
    if create_schema:
        db.create_schema(conn)
    else:
        # Minimal pre-migration schema: enough for compute_signal, but no
        # signal_cache table — mimics a DB checked out before this PR.
        conn.executescript("""
            CREATE TABLE fuel_types (id INTEGER PRIMARY KEY, code TEXT NOT NULL UNIQUE);
            INSERT OR IGNORE INTO fuel_types (code) VALUES ('E10');
            CREATE TABLE stations (
                station_code INTEGER PRIMARY KEY, address_normalized TEXT NOT NULL UNIQUE,
                suburb TEXT NOT NULL, postcode TEXT NOT NULL, name TEXT NOT NULL,
                brand TEXT, council TEXT, latitude REAL, longitude REAL
            );
            CREATE TABLE daily_prices (
                station_code INTEGER NOT NULL, fuel_type_id INTEGER NOT NULL,
                price_date INTEGER NOT NULL, price_decicents INTEGER NOT NULL,
                PRIMARY KEY (station_code, fuel_type_id, price_date)
            );
        """)
    conn.execute(
        "INSERT INTO stations (station_code, address_normalized, suburb, postcode, name, brand)"
        " VALUES (9001, '1 main street springwood', 'Springwood', '2777', 'Shell Springwood', 'Shell')"
    )
    conn.commit()
    series = _sawtooth_series()
    fid = db.fuel_type_id(conn, "E10")
    conn.executemany(
        "INSERT INTO daily_prices (station_code, fuel_type_id, price_date, price_decicents)"
        " VALUES (9001, ?, ?, ?)",
        [(fid, db._date_to_int(d), round(p * 10)) for d, p in series],
    )
    conn.commit()
    conn.close()
    return series


def test_cli_creates_signal_cache_table_on_a_pre_migration_db(tmp_path):
    """sourcery-ai review on #425: a DB that predates `signal_cache` must not
    hard-fail the first cache-generation run."""
    db_path = tmp_path / "pre_migration.db"
    series = _build_db(db_path, create_schema=False)

    runner = CliRunner()
    result = runner.invoke(generate_signal_cache_cli, ["--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "Cached signal for" in result.output

    conn = db.open_db(db_path)
    row = db.read_signal_cache(conn)
    assert row is not None
    assert row["as_of_date"] == series[-1][0]
    conn.close()


def test_cli_stores_a_decodable_payload(tmp_path):
    db_path = tmp_path / "fresh.db"
    series = _build_db(db_path)
    as_of = series[-1][0]   # the latest date — no --force needed

    runner = CliRunner()
    result = runner.invoke(
        generate_signal_cache_cli, ["--db", str(db_path), "--as-of", as_of]
    )
    assert result.exit_code == 0, result.output

    conn = db.open_db(db_path)
    row = db.read_signal_cache(conn)
    assert row["as_of_date"] == as_of
    payload, gap_start, gap_end = decode_cache_entry(json.loads(row["payload_json"]))
    assert payload.as_of_date == as_of
    # compute_signal defaults to the real config's PREFERRED_STATIONS (like
    # `signal.py`'s own CLI) — none of these codes exist in the synthetic DB,
    # so every price comes back None, but the keys must still be present.
    assert set(payload.station_prices) == set(PREFERRED_STATIONS)
    assert all(price is None for price in payload.station_prices.values())
    conn.close()


def test_generated_at_has_second_precision_not_microseconds():
    """Claude review on #425: the contract's every worked example is
    second-precision; `datetime.isoformat()` defaults to microseconds, which
    an iOS `ISO8601DateFormatter` without `.withFractionalSeconds` rejects."""
    import re

    from fuel_signal.signal import _sydney_now
    generated_at = _sydney_now().isoformat(timespec="seconds")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}", generated_at)


# ---------------------------------------------------------------------------
# --as-of / --force — Claude review on #425
# ---------------------------------------------------------------------------

def test_cli_refuses_a_non_latest_as_of_without_force(tmp_path):
    db_path = tmp_path / "fresh.db"
    series = _build_db(db_path)
    stale_as_of = series[180][0]

    runner = CliRunner()
    result = runner.invoke(
        generate_signal_cache_cli, ["--db", str(db_path), "--as-of", stale_as_of]
    )
    assert result.exit_code != 0
    assert "not the latest available date" in result.output

    conn = db.open_db(db_path)
    assert db.read_signal_cache(conn) is None
    conn.close()


def test_cli_force_allows_a_non_latest_as_of(tmp_path):
    db_path = tmp_path / "fresh.db"
    series = _build_db(db_path)
    stale_as_of = series[180][0]

    runner = CliRunner()
    result = runner.invoke(
        generate_signal_cache_cli,
        ["--db", str(db_path), "--as-of", stale_as_of, "--force"],
    )
    assert result.exit_code == 0, result.output

    conn = db.open_db(db_path)
    row = db.read_signal_cache(conn)
    assert row["as_of_date"] == stale_as_of
    conn.close()
