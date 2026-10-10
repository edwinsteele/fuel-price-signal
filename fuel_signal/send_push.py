"""Send one APNs background wake to each registered device."""

from __future__ import annotations

import logging
import os
import pathlib
import sqlite3
import time
from datetime import UTC, datetime, timedelta

import click
import httpx
import jwt

from fuel_signal import db

logger = logging.getLogger(__name__)

_HOSTS = {
    "sandbox": "https://api.sandbox.push.apple.com",
    "production": "https://api.push.apple.com",
}
_PAYLOAD = b'{"aps":{"content-available":1}}'
_BACKOFF_SECONDS = 0.5
_TOKEN_LABEL = "[redacted]"


def _configuration() -> tuple[str, str, str, str, str]:
    names = ("APNS_KEY_PATH", "APNS_KEY_ID", "APNS_TEAM_ID", "APNS_TOPIC", "APNS_ENV")
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise click.ClickException(f"Missing APNs configuration: {', '.join(missing)}")
    if values["APNS_ENV"] not in _HOSTS:
        raise click.ClickException("APNS_ENV must be sandbox or production")
    return tuple(values[name] for name in names)


def _jwt(key_path: str, key_id: str, team_id: str) -> str:
    try:
        private_key = pathlib.Path(key_path).read_text(encoding="utf-8")
        return jwt.encode(
            {"iss": team_id, "iat": int(time.time())},
            private_key,
            algorithm="ES256",
            headers={"kid": key_id},
        )
    except (OSError, ValueError, jwt.PyJWTError):
        # Exceptions from key parsers may include key material; never echo them.
        raise click.ClickException("Could not load or sign with APNs key") from None


def _unregistered_at(response: httpx.Response) -> str | None:
    try:
        body = response.json()
        if body.get("reason") != "Unregistered":
            return None
        timestamp = body["timestamp"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            return None
        seconds, milliseconds = divmod(timestamp, 1000)
        at = datetime.fromtimestamp(seconds, tz=UTC) + timedelta(milliseconds=milliseconds)
        return at.isoformat(timespec="milliseconds")
    except (ValueError, KeyError, OverflowError, OSError, TypeError, AttributeError):
        return None


def _send_one(client: httpx.Client, host: str, token: str, headers: dict[str, str], conn) -> bool:
    # A prefix of one token can be the entirety of another registered token.
    label = _TOKEN_LABEL
    for attempt in range(2):
        try:
            response = client.post(f"{host}/3/device/{token}", headers=headers, content=_PAYLOAD)
        except httpx.TransportError:
            if attempt == 0:
                time.sleep(_BACKOFF_SECONDS)
                continue
            logger.error("APNs transport failure for token %s", label)
            return False

        if response.status_code == 200:
            logger.info("APNs accepted token %s", label)
            return True
        if response.status_code == 410:
            before = _unregistered_at(response)
            if before is not None:
                try:
                    removed = db.delete_device_if_older_than(conn, token, before)
                except sqlite3.Error:
                    # A failed commit may leave a pending delete; do not let a
                    # later successful cleanup accidentally commit it.
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
                    logger.error("APNs token cleanup failed for token %s", label)
                    return False
                logger.info("APNs unregistered token %s (removed=%s)", label, removed)
                return True
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 0:
                time.sleep(_BACKOFF_SECONDS)
                continue
        # Do not print the request, response body, or headers: they may contain tokens.
        logger.error("APNs rejected token %s (HTTP %s)", label, response.status_code)
        return False
    return False


@click.command("send_push")
@click.option(
    "--db", "db_path", default=str(db.DEFAULT_DB_PATH), show_default=True,
    help="Path to SQLite database.",
)
def main(db_path: str) -> None:
    """Send the nightly background wake to all registered devices."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # httpx's INFO request log includes the full /3/device/<token> URL.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    key_path, key_id, team_id, topic, environment = _configuration()
    path = pathlib.Path(db_path)
    if not path.is_file():
        raise click.ClickException(f"Database not found: {db_path}")
    bearer = _jwt(key_path, key_id, team_id)
    headers = {
        "authorization": f"bearer {bearer}",
        "apns-push-type": "background",
        "apns-priority": "5",
        "apns-topic": topic,
    }

    conn = db.open_db(path)
    try:
        devices = db.list_devices(conn)
        if not devices:
            logger.info("No registered devices; no APNs pushes sent")
            return
        failures = 0
        with httpx.Client(http2=True, timeout=10.0) as client:
            for device in devices:
                if not _send_one(client, _HOSTS[environment], device["token"], headers, conn):
                    failures += 1
        if failures:
            raise click.ClickException(f"APNs push failed for {failures} of {len(devices)} devices")
        logger.info("APNs accepted or retired %s device(s)", len(devices))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
