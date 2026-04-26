"""Forward-fill daily price gaps and rebuild the daily_prices table."""

import logging
import pathlib
import sqlite3
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def find_daily_gaps(
    price_rows: list[tuple[str, float]],
    end_date: str | None = None,
) -> list[tuple[str, float]]:
    """Return gap rows needed to make price_rows a complete daily series.

    price_rows: sorted list of (date_str YYYY-MM-DD, price_cents) observations.
    end_date:   if provided, trail-fill from last observation+1 day to this date.

    Returns only the gap dates — not the original observations.
    Same-day duplicates in price_rows: last price wins (matches DB INSERT OR IGNORE
    behaviour — in practice the prices table PK prevents duplicates, but handled
    defensively here for testability).
    """
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

        fill = last_date + one_day
        while fill < entry_date:
            gaps.append((fill.isoformat(), last_price))
            fill += one_day

        last_date = entry_date
        last_price = price

    if last_date is not None and end_date is not None:
        end = date.fromisoformat(end_date)
        fill = last_date + one_day
        while fill <= end:
            gaps.append((fill.isoformat(), last_price))
            fill += one_day

    return gaps


def fill_all(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    end_date: str | None = None,
) -> int:
    """Rebuild daily_prices for all stations that have prices in fuel_code.

    Copies every observed price from prices, then forward-fills gaps between
    observations and from the last observation to end_date (default: today).

    Returns the total number of rows written to daily_prices.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    conn.execute("DELETE FROM daily_prices WHERE fuel_code = ?", (fuel_code,))

    station_codes: list[int] = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT station_code FROM prices WHERE fuel_code = ? ORDER BY station_code",
            (fuel_code,),
        )
    ]

    total = 0
    for station_code in station_codes:
        raw: list[tuple[str, float]] = [
            (r[0], r[1])
            for r in conn.execute(
                "SELECT price_date, price_cents FROM prices"
                " WHERE station_code = ? AND fuel_code = ? ORDER BY price_date",
                (station_code, fuel_code),
            )
        ]
        if not raw:
            continue

        gaps = find_daily_gaps(raw, end_date)
        all_rows = raw + gaps

        conn.executemany(
            "INSERT INTO daily_prices (station_code, fuel_code, price_date, price_cents)"
            " VALUES (?, ?, ?, ?)",
            [(station_code, fuel_code, d, p) for d, p in all_rows],
        )
        total += len(all_rows)

    conn.commit()
    logger.info(
        "fill_all: %d stations, %d total daily_prices rows written (%s, up to %s)",
        len(station_codes), total, fuel_code, end_date,
    )
    return total


if __name__ == "__main__":
    import sys

    from fuel_signal.db import DEFAULT_DB_PATH, create_schema, open_db

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    db_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB_PATH
    conn = open_db(db_path)
    create_schema(conn)
    fill_all(conn)
    conn.close()
