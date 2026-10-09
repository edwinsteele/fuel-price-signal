"""POST /api/v1/devices (#435) — the `devices` table helpers in fuel_signal.db and
the route on the workbench app in fuel_signal.inspect.

Route tests go through Flask's test client, so they enter where the app does:
the raw request bytes and headers, not a pre-parsed body.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import zoneinfo

import pytest

from fuel_signal import db as _db
from fuel_signal.db import (
    MAX_DEVICES,
    cap_devices,
    create_schema,
    delete_device_if_older_than,
    list_devices,
    open_db,
    open_request_connection,
    register_device,
    upsert_device,
)
from fuel_signal.inspect import _create_app
from tests.contract_examples import contract_example

_SYDNEY = zoneinfo.ZoneInfo("Australia/Sydney")
_T0 = datetime.datetime(2026, 10, 10, 7, 1, 55, tzinfo=_SYDNEY)
_TOKEN = "a1" * 32


def _ts(minutes: int) -> str:
    return (_T0 + datetime.timedelta(minutes=minutes)).isoformat(timespec="seconds")


def _token(i: int) -> str:
    return f"{i:064x}"


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# db.py helpers
# ---------------------------------------------------------------------------


def test_create_schema_creates_devices_table(conn):
    cols = {row[1]: row for row in conn.execute("PRAGMA table_info(devices)")}
    assert set(cols) == {"token", "first_seen", "last_seen"}
    assert cols["token"][5] == 1  # primary key


def test_register_new_token_sets_first_and_last_seen(conn):
    register_device(conn, _TOKEN, _ts(0))
    assert list_devices(conn) == [{"token": _TOKEN, "first_seen": _ts(0), "last_seen": _ts(0)}]


def test_register_known_token_bumps_last_seen_keeps_first_seen(conn):
    register_device(conn, _TOKEN, _ts(0))
    register_device(conn, _TOKEN, _ts(5))
    assert list_devices(conn) == [{"token": _TOKEN, "first_seen": _ts(0), "last_seen": _ts(5)}]


def test_eleventh_token_evicts_oldest_last_seen(conn):
    for i in range(MAX_DEVICES):
        register_device(conn, _token(i), _ts(i))
    register_device(conn, _token(MAX_DEVICES), _ts(MAX_DEVICES))
    tokens = {d["token"] for d in list_devices(conn)}
    assert len(tokens) == MAX_DEVICES
    assert _token(0) not in tokens
    assert _token(MAX_DEVICES) in tokens


def test_eviction_is_by_last_seen_not_first_registration(conn):
    for i in range(MAX_DEVICES):
        register_device(conn, _token(i), _ts(i))
    register_device(conn, _token(0), _ts(20))  # the first-registered token re-registers
    register_device(conn, _token(99), _ts(21))
    tokens = {d["token"] for d in list_devices(conn)}
    assert _token(0) in tokens
    assert _token(1) not in tokens


def test_eviction_orders_by_instant_across_dst_fall_back(conn):
    # Sydney leaves DST at 03:00+11:00 on 2026-04-05, back to 02:00+10:00.
    # 02:15+10:00 (16:15Z) is LATER than 02:30+11:00 (15:30Z), though it sorts
    # first as a string.
    earlier, later = "2026-04-05T02:30:00+11:00", "2026-04-05T02:15:00+10:00"
    upsert_device(conn, "aa", earlier)
    upsert_device(conn, "bb", later)
    assert cap_devices(conn, max_devices=1) == 1
    assert [d["token"] for d in list_devices(conn)] == ["bb"]


def test_cap_never_evicts_the_token_being_registered(conn):
    # All 11 share one timestamp, so last_seen alone can't pick the survivor;
    # "ff…" sorts last by token, the tie-break that would otherwise evict it.
    for i in range(MAX_DEVICES):
        register_device(conn, _token(i), _ts(0))
    newcomer = "ff" * 32
    register_device(conn, newcomer, _ts(0))
    tokens = {d["token"] for d in list_devices(conn)}
    assert len(tokens) == MAX_DEVICES
    assert newcomer in tokens


def test_delete_if_older_than_deletes_a_token_last_seen_before(conn):
    register_device(conn, _TOKEN, _ts(0))
    assert delete_device_if_older_than(conn, _TOKEN, _ts(1)) is True
    assert list_devices(conn) == []


def test_delete_if_older_than_keeps_a_token_reregistered_since(conn):
    register_device(conn, _TOKEN, _ts(0))
    register_device(conn, _TOKEN, _ts(10))  # raced the push
    assert delete_device_if_older_than(conn, _TOKEN, _ts(5)) is False
    assert [d["token"] for d in list_devices(conn)] == [_TOKEN]


def test_delete_if_older_than_compares_instants_not_strings(conn):
    upsert_device(conn, _TOKEN, "2026-04-05T02:15:00+10:00")  # 16:15Z
    conn.commit()
    # 15:45Z: as a string "02:45" > "02:15", but it is the earlier instant.
    assert delete_device_if_older_than(conn, _TOKEN, "2026-04-05T02:45:00+11:00") is False
    assert delete_device_if_older_than(conn, _TOKEN, "2026-04-05T06:20:00Z") is True


def test_unreadable_timestamp_raises_instead_of_matching_nothing(conn):
    register_device(conn, _TOKEN, _ts(0))
    with pytest.raises(ValueError):
        delete_device_if_older_than(conn, _TOKEN, "yesterday")
    with pytest.raises(ValueError):
        upsert_device(conn, _TOKEN, "not-a-time")


def test_register_device_creates_table_on_a_db_that_predates_it(tmp_path):
    c = open_db(tmp_path / "old.db")  # no create_schema
    try:
        register_device(c, _TOKEN, _ts(0))
        assert [d["token"] for d in list_devices(c)] == [_TOKEN]
    finally:
        c.close()


def test_ensuring_an_existing_devices_table_needs_no_write_lock(conn, tmp_path):
    # register_device runs the CREATE TABLE IF NOT EXISTS on every request; it
    # must not queue behind the nightly load's write lock when the table exists.
    locker = sqlite3.connect(tmp_path / "test.db", isolation_level=None)
    w = open_request_connection(tmp_path / "test.db", busy_timeout_s=0.0)
    try:
        locker.execute("BEGIN IMMEDIATE")
        w.execute(_db._DEVICES_SCHEMA)  # no OperationalError
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            register_device(w, _TOKEN, _ts(0))
    finally:
        w.close()
        locker.close()


def test_open_request_connection_sets_busy_timeout(tmp_path):
    c = open_request_connection(tmp_path / "x.db", busy_timeout_s=2.5)
    try:
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 2500
    finally:
        c.close()


def test_database_file_rejects_in_memory_connection():
    c = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError):
            _db.database_file(c)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@pytest.fixture
def clock(monkeypatch):
    """Mutable stand-in for `_sydney_now`, so a test can advance request time."""
    state = {"now": _T0}
    monkeypatch.setattr("fuel_signal.inspect._sydney_now", lambda: state["now"])
    return state


def _client_for(c):
    app = _create_app(
        c, cd=None, today="2026-10-09", cycle_state=None,
        peak_data={}, summary={}, boundaries={},
    )
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def client(conn, clock):
    with _client_for(conn) as cl:
        yield cl


def _post(client, body, content_type="application/json"):
    data = body if isinstance(body, (str, bytes)) else json.dumps(body)
    return client.post("/api/v1/devices", data=data, content_type=content_type)


def test_post_new_token_returns_204_and_stores_it(client, conn):
    resp = _post(client, {"token": _TOKEN})
    assert resp.status_code == 204
    assert resp.data == b""
    assert list_devices(conn) == [
        {"token": _TOKEN, "first_seen": "2026-10-10T07:01:55+11:00", "last_seen": "2026-10-10T07:01:55+11:00"}
    ]


def test_post_repeat_token_returns_204_and_bumps_last_seen(client, conn, clock):
    assert _post(client, {"token": _TOKEN}).status_code == 204
    clock["now"] = _T0 + datetime.timedelta(hours=3)
    resp = _post(client, {"token": _TOKEN})
    assert resp.status_code == 204
    assert resp.data == b""
    assert list_devices(conn) == [
        {"token": _TOKEN, "first_seen": "2026-10-10T07:01:55+11:00", "last_seen": "2026-10-10T10:01:55+11:00"}
    ]


def test_post_mixed_case_is_stored_lowercased_as_one_row(client, conn):
    mixed = "A1b2C3d4" * 8
    assert _post(client, {"token": mixed}).status_code == 204
    assert _post(client, {"token": mixed.lower()}).status_code == 204
    assert _post(client, {"token": mixed.upper()}).status_code == 204
    assert [d["token"] for d in list_devices(conn)] == [mixed.lower()]


@pytest.mark.parametrize("token", ["ab", "0" * 200], ids=["one-byte", "100-bytes"])
def test_post_accepts_token_length_bounds(client, conn, token):
    assert _post(client, {"token": token}).status_code == 204
    assert [d["token"] for d in list_devices(conn)] == [token]


def test_post_accepts_json_content_type_with_charset(client):
    resp = _post(client, {"token": _TOKEN}, content_type="application/json; charset=utf-8")
    assert resp.status_code == 204


def test_post_ignores_extra_keys(client, conn):
    assert _post(client, {"token": _TOKEN, "platform": "ios"}).status_code == 204
    assert [d["token"] for d in list_devices(conn)] == [_TOKEN]


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        ({"token": "zz" * 32}, "application/json"),
        ({"token": "abc"}, "application/json"),
        ({"token": ""}, "application/json"),
        ({"token": "0" * 202}, "application/json"),
        ({"token": "ab\n"}, "application/json"),
        ({"token": "００"}, "application/json"),  # fullwidth digits
        ({}, "application/json"),
        ({"token": 1234}, "application/json"),
        ({"token": None}, "application/json"),
        ({"token": ["ab"]}, "application/json"),
        (["ab"], "application/json"),
        ("null", "application/json"),
        ('"ab"', "application/json"),
        ("{not json", "application/json"),
        ("", "application/json"),
        (b"\xff\xfe", "application/json"),
        (json.dumps({"token": _TOKEN}), None),
        (json.dumps({"token": _TOKEN}), "text/plain"),
        (json.dumps({"token": _TOKEN}), "application/vnd.api+json"),
        ("token=" + _TOKEN, "application/x-www-form-urlencoded"),
    ],
    ids=[
        "non-hex", "odd-length", "empty", "over-200", "trailing-newline", "non-ascii-digits",
        "missing-token", "number", "null-token", "list-token", "array-body", "null-body",
        "string-body", "malformed-json", "empty-body", "undecodable-bytes",
        "no-content-type", "text-plain", "json-suffix-type", "form-encoded",
    ],
)
def test_post_malformed_request_returns_400_and_stores_nothing(client, conn, body, content_type):
    resp = _post(client, body, content_type=content_type)
    assert resp.status_code == 400
    assert list_devices(conn) == []


def test_post_eleventh_token_evicts_oldest(client, conn, clock):
    for i in range(MAX_DEVICES + 1):
        clock["now"] = _T0 + datetime.timedelta(minutes=i)
        assert _post(client, {"token": _token(i)}).status_code == 204
    tokens = {d["token"] for d in list_devices(conn)}
    assert len(tokens) == MAX_DEVICES
    assert _token(0) not in tokens


def test_post_contract_example_is_accepted(client, conn):
    example = contract_example("devices-request")
    resp = _post(client, example)
    assert resp.status_code == 204
    assert [d["token"] for d in list_devices(conn)] == [example["token"].lower()]


def test_post_uses_its_own_connection_with_busy_timeout_and_closes_it(client, conn, monkeypatch):
    seen = {}
    real = _db.register_device

    def spy(c, token, seen_at, *args, **kwargs):
        seen["conn"] = c
        seen["busy_timeout"] = c.execute("PRAGMA busy_timeout").fetchone()[0]
        return real(c, token, seen_at, *args, **kwargs)

    monkeypatch.setattr("fuel_signal.db.register_device", spy)
    assert _post(client, {"token": _TOKEN}).status_code == 204
    assert seen["conn"] is not conn
    assert seen["busy_timeout"] == 5000
    with pytest.raises(sqlite3.ProgrammingError):
        seen["conn"].execute("SELECT 1")  # closed after the request
    assert not conn.in_transaction


def test_post_returns_503_while_another_writer_holds_the_lock(client, conn, tmp_path, monkeypatch):
    monkeypatch.setattr("fuel_signal.inspect._DEVICES_BUSY_TIMEOUT_S", 0.05)
    locker = sqlite3.connect(tmp_path / "test.db", isolation_level=None)
    try:
        locker.execute("BEGIN IMMEDIATE")
        assert _post(client, {"token": _TOKEN}).status_code == 503
        locker.execute("ROLLBACK")
        assert _post(client, {"token": _TOKEN}).status_code == 204
    finally:
        locker.close()
    assert [d["token"] for d in list_devices(conn)] == [_TOKEN]


def test_post_on_a_db_that_predates_the_devices_table(tmp_path, clock):
    c = open_db(tmp_path / "old.db")  # no create_schema: as viking's DB right after a deploy
    try:
        with _client_for(c) as cl:
            assert _post(cl, {"token": _TOKEN}).status_code == 204
        assert [d["token"] for d in list_devices(c)] == [_TOKEN]
    finally:
        c.close()


def test_no_get_route_tokens_are_never_readable(client):
    _post(client, {"token": _TOKEN})
    assert client.get("/api/v1/devices").status_code == 405
