"""Precompute the expensive half of the signal for the /api/v1 blueprint.

Runs `compute_signal` (the ~29s `load_history` cost) once and stores the
result in `db.signal_cache`, so `inspect.py`'s blueprint can serve
`/api/v1/stations` and `/api/v1/recommendation` as a cheap read instead of
scoring on every request. Intended to run nightly, after the price load —
see docs/api-contract.md for the wire contract this feeds and
AGENTS.md/README.md for where this fits in the daily pipeline.

Usage:
    uv run python -m fuel_signal.generate_signal_cache
    uv run python -m fuel_signal.generate_signal_cache --db /path/to/fuel_signal.db

    # --as-of is for previewing a past date; it refuses anything but the
    # latest available date unless --force is also given (see main() below).
    uv run python -m fuel_signal.generate_signal_cache --as-of 2026-06-01 --force
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
    help="Date to evaluate (YYYY-MM-DD). Defaults to, and normally must equal, the "
         "latest date in daily_prices — see --force.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Allow --as-of to be earlier than the latest available date.",
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
def main(as_of: str | None, force: bool, db_path: str, model_path: pathlib.Path) -> None:
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
        latest = _latest_daily_date(conn)
        as_of_date = as_of or latest
        if as_of_date != latest and not force:
            # docs/api-contract.md deliberately has no `historical_view` field:
            # v1 has no as_of *request* parameter, so a live request can never
            # observe a stale cache — but that reasoning only holds if the
            # cached payload's as_of is always the latest date. `days_stale` is
            # measured from `last_real_price_date` (the DB's real current max,
            # independent of `as_of`), so a payload cached from an old --as-of
            # reads as fresh even though every priced/cycle field in it is old.
            raise click.ClickException(
                f"--as-of {as_of_date} is not the latest available date ({latest}). "
                "The API's freshness banner would not reflect this — pass --force "
                "if you really want to cache a historical date."
            )
        payload = compute_signal(conn, as_of_date, model_path=model_path)
        gap_start, gap_end = _gap_boundaries(conn)
        blob = api_v1.encode_cache_entry(payload, gap_start, gap_end)
        generated_at = _sydney_now().isoformat(timespec="seconds")
        db.write_signal_cache(conn, as_of_date, generated_at, json.dumps(blob))
        click.echo(f"Cached signal for {as_of_date} (generated_at={generated_at}).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
