"""APNs sender request and token-lifecycle tests; no network traffic."""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import jwt
import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from fuel_signal import db, send_push
from fuel_signal.send_push import _jwt as sign_apns_jwt
from tests.contract_examples import contract_example

TOKEN = "a" * 64
OTHER_TOKEN = "b" * 64
SEEN = "2026-10-11T10:00:00.000+11:00"


@pytest.fixture
def configured(monkeypatch, tmp_path):
    for name, value in {
        "APNS_KEY_PATH": str(tmp_path / "key.p8"),
        "APNS_KEY_ID": "KEYID",
        "APNS_TEAM_ID": "TEAMID",
        "APNS_TOPIC": "com.edwinsteele.FuelPriceSignal",
        "APNS_ENV": "sandbox",
    }.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "key.p8").write_text("test key")
    monkeypatch.setattr(send_push, "_jwt", lambda *args: "signed.jwt")
    return tmp_path


def _database(tmp_path, tokens=(TOKEN,)):
    path = tmp_path / "devices.db"
    conn = db.open_db(path)
    db.create_schema(conn)
    for token in tokens:
        db.register_device(conn, token, SEEN)
    conn.close()
    return path


def _mock_client(monkeypatch, handler):
    original = httpx.Client
    arguments = []

    def factory(**kwargs):
        arguments.append(kwargs)
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(send_push.httpx, "Client", factory)
    return arguments


def _run(path):
    return CliRunner().invoke(send_push.main, ["--db", str(path)])


@pytest.mark.parametrize("environment,host", [
    ("sandbox", "api.sandbox.push.apple.com"),
    ("production", "api.push.apple.com"),
])
def test_request_matches_contract(configured, monkeypatch, caplog, environment, host):
    monkeypatch.setenv("APNS_ENV", environment)
    path = _database(configured)
    requests = []
    options = _mock_client(monkeypatch, lambda request: (requests.append(request), httpx.Response(200))[1])

    with caplog.at_level("INFO"):
        result = _run(path)

    assert result.exit_code == 0, result.output
    assert options == [{"http2": True, "timeout": 10.0}]
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url == f"https://{host}/3/device/{TOKEN}"
    assert request.headers["authorization"] == "bearer signed.jwt"
    assert {name: value for name, value in request.headers.items() if name.startswith("apns-")} == {
        "apns-push-type": "background", "apns-priority": "5", "apns-topic": "com.edwinsteele.FuelPriceSignal",
    }
    assert request.content == b'{"aps":{"content-available":1}}'
    assert json.loads(request.content) == contract_example("push-payload")
    assert TOKEN not in result.output
    assert "signed.jwt" not in result.output
    assert TOKEN not in caplog.text
    assert "signed.jwt" not in caplog.text


def test_jwt_is_es256_with_key_id_and_team_claim(configured, monkeypatch):
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    key_path = configured / "key.p8"
    key_path.write_bytes(pem)
    monkeypatch.setattr(send_push.time, "time", lambda: 1234567890)

    signed = sign_apns_jwt(str(key_path), "KEYID", "TEAMID")

    assert jwt.get_unverified_header(signed)["alg"] == "ES256"
    assert jwt.get_unverified_header(signed)["kid"] == "KEYID"
    assert jwt.decode(signed, key.public_key(), algorithms=["ES256"], options={"verify_iat": False}) == {
        "iss": "TEAMID", "iat": 1234567890,
    }


@pytest.mark.parametrize("missing", ["APNS_KEY_PATH", "APNS_KEY_ID", "APNS_TEAM_ID", "APNS_TOPIC", "APNS_ENV"])
def test_missing_configuration_fails_before_send(configured, monkeypatch, missing):
    path = _database(configured)
    monkeypatch.delenv(missing)
    requests = []
    _mock_client(monkeypatch, lambda request: (requests.append(request), httpx.Response(200))[1])
    result = _run(path)
    assert result.exit_code != 0
    assert missing in result.output
    assert not requests


def test_unknown_environment_fails_before_send(configured, monkeypatch):
    path = _database(configured)
    monkeypatch.setenv("APNS_ENV", "auto")
    requests = []
    _mock_client(monkeypatch, lambda request: (requests.append(request), httpx.Response(200))[1])
    result = _run(path)
    assert result.exit_code != 0
    assert "APNS_ENV" in result.output
    assert not requests


@pytest.mark.parametrize("timestamp,removed", [
    ("2026-10-11T10:00:00.001+11:00", True),
    ("2026-10-11T09:59:59+11:00", False),
    ("2026-10-11T10:00:00+11:00", False),
])
def test_unregistered_deletes_only_older_registration(configured, monkeypatch, timestamp, removed):
    path = _database(configured)
    milliseconds = int(datetime.fromisoformat(timestamp).timestamp() * 1000)
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(410, json={"reason": "Unregistered", "timestamp": milliseconds}),
    )

    result = _run(path)

    assert result.exit_code == 0, result.output
    conn = db.open_db(path)
    assert (not db.list_devices(conn)) is removed
    conn.close()


@pytest.mark.parametrize("reason", ["BadDeviceToken", "DeviceTokenNotForTopic"])
def test_configuration_errors_log_and_keep_token(configured, monkeypatch, caplog, reason):
    path = _database(configured)
    requests = []
    _mock_client(
        monkeypatch,
        lambda request: (requests.append(request), httpx.Response(400, json={"reason": reason}))[1],
    )

    result = _run(path)

    assert result.exit_code != 0
    assert len(requests) == 1
    assert any(record.levelname == "ERROR" for record in caplog.records)
    assert TOKEN not in result.output
    conn = db.open_db(path)
    assert [item["token"] for item in db.list_devices(conn)] == [TOKEN]
    conn.close()


@pytest.mark.parametrize("first", ["transport", 429, 500, 503])
def test_retriable_failure_gets_one_retry(configured, monkeypatch, first):
    path = _database(configured)
    requests = []
    monkeypatch.setattr(send_push.time, "sleep", lambda seconds: None)

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if first == "transport":
                raise httpx.ConnectError("connection lost", request=request)
            return httpx.Response(first)
        return httpx.Response(200)

    _mock_client(monkeypatch, handler)
    result = _run(path)
    assert result.exit_code == 0, result.output
    assert len(requests) == 2


@pytest.mark.parametrize("status", [400, 403, 410, 429, 500])
def test_final_failure_is_nonzero_and_does_not_stop_other_tokens(configured, monkeypatch, status):
    path = _database(configured, (TOKEN, OTHER_TOKEN))
    requests = []
    monkeypatch.setattr(send_push.time, "sleep", lambda seconds: None)

    def handler(request):
        requests.append(request)
        if request.url.path.endswith(TOKEN):
            return httpx.Response(status, json={"reason": "Other"})
        return httpx.Response(200)

    _mock_client(monkeypatch, handler)
    result = _run(path)
    assert result.exit_code != 0
    assert len([r for r in requests if r.url.path.endswith(TOKEN)]) == (2 if status in (429, 500) else 1)
    assert any(r.url.path.endswith(OTHER_TOKEN) for r in requests)
    assert TOKEN not in result.output
    assert OTHER_TOKEN not in result.output


def test_no_devices_exits_zero_with_info_log(configured, monkeypatch, caplog):
    path = _database(configured, ())
    requests = []
    _mock_client(monkeypatch, lambda request: (requests.append(request), httpx.Response(200))[1])
    with caplog.at_level("INFO", logger="fuel_signal.send_push"):
        result = _run(path)
    assert result.exit_code == 0, result.output
    assert any(record.levelname == "INFO" and "No registered devices" in record.message for record in caplog.records)
    assert not requests
