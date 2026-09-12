"""Precompute the expensive half of the signal for the /api/v1 blueprint.

Runs `compute_signal` (the ~29s `load_history` cost) once and stores the
result in `db.signal_cache`, so `inspect.py`'s blueprint can serve
`/api/v1/stations` and `/api/v1/recommendation` as a cheap read instead of
scoring on every request. Intended to run nightly, after the price load —
see docs/api-contract.md for the wire contract this feeds and
AGENTS.md/README.md for where this fits in the daily pipeline.

Usage:
    uv run python -m fuel_signal.generate_signal_cache
    uv run python -m fuel_signal.generate_signal_cache --as-of 2026-09-11
    uv run python -m fuel_signal.generate_signal_cache --db /path/to/fuel_signal.db
"""

from __future__ import annotations

import json
import logging
import pathlib

import click

import fuel_signal.db as db
from fuel_signal import api_v1
from fuel_signal.signal import (
    DEFAULT_MODEL_PATH,
    _gap_boundaries,
    _latest_daily_date,
    _sydney_now,
    compute_signal,
)

logger = logging.getLogger(__name__)


@click.command("generate_signal_cache")
@click.option(
    "--as-of",
    "as_of",
    default=None,
    metavar="DATE",
    help="Date to evaluate (YYYY-MM-DD). Defaults to latest date in daily_prices.",
)
@click.option(
    "--db",
    "db_path",
    default=str(db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option(
    "--model",
    "model_path",
    type=pathlib.Path,
    default=DEFAULT_MODEL_PATH,
    show_default=True,
    help="Calibrated model artifact. Falls back to rules-only if absent.",
)
def main(as_of: str | None, db_path: str, model_path: pathlib.Path) -> None:
    """Compute today's signal and store it for the HTTP API to serve."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )
    conn = db.open_db(path)
    try:
        # An existing DB from before this table was added won't have it yet;
        # create_schema is idempotent (CREATE TABLE IF NOT EXISTS), so this is
        # cheap and safe to run every time rather than requiring an operator
        # to separately re-run `python -m fuel_signal.db` first.
        db.create_schema(conn)
        as_of_date = as_of or _latest_daily_date(conn)
        payload = compute_signal(conn, as_of_date, model_path=model_path)
        gap_start, gap_end = _gap_boundaries(conn)
        blob = api_v1.encode_cache_entry(payload, gap_start, gap_end)
        generated_at = _sydney_now().isoformat()
        db.write_signal_cache(conn, as_of_date, generated_at, json.dumps(blob))
        click.echo(f"Cached signal for {as_of_date} (generated_at={generated_at}).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
