"""Forward-fill daily price gaps and rebuild the daily_prices table."""

import logging
import pathlib
import sqlite3
from datetime import date, timedelta

import click

from fuel_signal.dates import date_from_int, date_to_int
from fuel_signal.db import (
    DEFAULT_DB_PATH,
    create_schema,
    fuel_type_id,
    open_db,
    station_price_series,
    upsert_daily_prices,
)

logger = logging.getLogger(__name__)

# Gaps wider than this (station closed / rebuilt) are left unfilled rather than
# forward-filled with a stale price that would pollute label computation.
MAX_GAP_FILL_DAYS = 28


def find_daily_gaps(
    price_rows: list[tuple[str, float]],
    end_date: str | None = None,
    max_gap_days: int = MAX_GAP_FILL_DAYS,
) -> list[tuple[str, float]]:
    """Return gap rows needed to make price_rows a complete daily series.

    price_rows:   sorted list of (date_str YYYY-MM-DD, price_cents) observations.
    end_date:     if provided, trail-fill from last observation+1 day to this date.
    max_gap_days: gaps spanning more than this many days are left unfilled entirely
                  (station likely closed or rebuilt). Gaps <= max_gap_days are filled.

    Returns only the gap dates — not the original observations.
    Same-day duplicates in price_rows: last price wins (matches DB INSERT OR IGNORE
    behaviour — in practice the prices table PK prevents duplicates, but handled
    defensively here for testability).
    """
    if max_gap_days < 0:
        raise ValueError("max_gap_days must be non-negative")
    gaps: list[tuple[str, float]] = []
    one_day = timedelta(days=1)
    last_date: date | None = None
    last_price: float | None = None

    for date_str, price in price_rows:
        entry_date = date.fromisoformat(date_str)

        if last_date is None:
            last_date = entry_date
            last_price = price
            continue

        if entry_date == last_date:
            last_price = price
            continue

        if (entry_date - last_date).days <= max_gap_days:
            fill = last_date + one_day
            while fill < entry_date:
                gaps.append((fill.isoformat(), last_price))
                fill += one_day

        last_date = entry_date
        last_price = price

    if last_date is not None and end_date is not None:
        end = date.fromisoformat(end_date)
        if (end - last_date).days <= max_gap_days:
            fill = last_date + one_day
            while fill <= end:
                gaps.append((fill.isoformat(), last_price))
                fill += one_day

    return gaps


def fill_all(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    end_date: str | None = None,
    max_gap_days: int = MAX_GAP_FILL_DAYS,
    since_date: str | None = None,
) -> int:
    """Rebuild daily_prices for all stations that have prices in fuel_code.

    Copies every observed price from prices, then forward-fills gaps between
    observations and from the last observation to end_date (default: today).
    Gaps wider than max_gap_days are left unfilled.

    since_date: if given, only dates >= since_date are recomputed — daily_prices
    rows before since_date are left as-is. Each station's forward-fill picks up
    from its most recent observation at or before since_date (if any), so gaps
    straddling the boundary still resolve correctly. Omit for a full rebuild.

    Returns the number of daily_prices rows written (all of them for a full
    rebuild, or just those from since_date onward for an incremental run).
    """
    if max_gap_days < 0:
        raise ValueError("max_gap_days must be non-negative")
    if end_date is None:
        end_date = date.today().isoformat()

    fid = fuel_type_id(conn, fuel_code)

    if since_date is None:
        conn.execute("DELETE FROM daily_prices WHERE fuel_type_id = ?", (fid,))
    else:
        conn.execute(
            "DELETE FROM daily_prices WHERE fuel_type_id = ? AND price_date >= ?",
            (fid, date_to_int(since_date)),
        )

    station_codes: list[int] = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT station_code FROM prices WHERE fuel_type_id = ? ORDER BY station_code",
            (fid,),
        )
    ]

    total = 0
    for station_code in station_codes:
        if since_date is None:
            raw = station_price_series(conn, station_code, fuel_code)
        else:
            anchor = conn.execute(
                "SELECT price_date FROM prices"
                " WHERE station_code = ? AND fuel_type_id = ? AND price_date <= ?"
                " ORDER BY price_date DESC LIMIT 1",
                (station_code, fid, date_to_int(since_date)),
            ).fetchone()
            window_start = date_from_int(anchor[0]) if anchor else since_date
            raw = station_price_series(conn, station_code, fuel_code, start_date=window_start)
        if not raw:
            continue

        gaps = find_daily_gaps(raw, end_date, max_gap_days=max_gap_days)
        all_rows = [(station_code, fuel_code, d, p) for d, p in raw + gaps]
        if since_date is not None:
            all_rows = [r for r in all_rows if r[2] >= since_date]
        upsert_daily_prices(conn, all_rows)
        total += len(all_rows)

    conn.commit()
    logger.info(
        "fill_all: %d stations, %d daily_prices rows written (%s, %s to %s)",
        len(station_codes), total, fuel_code, since_date or "full history", end_date,
    )
    return total


@click.command("fill")
@click.option(
    "--db",
    "db_path",
    default=str(DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option(
    "--max-gap-days",
    default=MAX_GAP_FILL_DAYS,
    show_default=True,
    help="Gaps wider than this many days are left unfilled (station likely closed).",
)
@click.option(
    "--since-date",
    default=None,
    help="Only recompute daily_prices from this date onward (YYYY-MM-DD). "
    "Omit for a full rebuild of the whole table.",
)
def main(db_path: str, max_gap_days: int, since_date: str | None) -> None:
    """Forward-fill daily price gaps and rebuild the daily_prices table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    conn = open_db(pathlib.Path(db_path))
    create_schema(conn)
    fill_all(conn, max_gap_days=max_gap_days, since_date=since_date)
    conn.close()


if __name__ == "__main__":
    main()
