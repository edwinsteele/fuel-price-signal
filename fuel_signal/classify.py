"""Per-station daily classifier: Competitive / Sticky / Discount.

Classifies each station relative to its LGA's competitive price cluster using a
two-iteration median-premium approach. Results materialise into station_class and
classification_summary tables (one row per station per snapshot_date).

Run after fill.fill_all() has rebuilt daily_prices for the period of interest.
"""

import datetime
import logging
import pathlib
import sqlite3
import statistics

import click

from fuel_signal import db as _db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_DAYS: int = 45          # calendar days in the classification window
BAND_DECICENTS: int = 100      # ±10 cents = ±100 decicents; defines Sticky / Discount


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _int_to_date(v: int) -> datetime.date:
    s = str(v)
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))


def _date_to_int(d: datetime.date) -> int:
    return int(d.strftime("%Y%m%d"))


def _window_bounds(snapshot_date: int) -> tuple[int, int]:
    """Return (window_start_int, window_end_int) for the WINDOW_DAYS window ending at snapshot_date - 1."""
    d = _int_to_date(snapshot_date)
    window_end = d - datetime.timedelta(days=1)
    window_start = window_end - datetime.timedelta(days=WINDOW_DAYS - 1)
    return _date_to_int(window_start), _date_to_int(window_end)


def _median_int(values: list[int | float]) -> int:
    """Median of values, rounded to nearest integer."""
    return round(statistics.median(values))


def _classify_premium(median_premium_decicents: int) -> str:
    if median_premium_decicents > BAND_DECICENTS:
        return "Sticky"
    if median_premium_decicents < -BAND_DECICENTS:
        return "Discount"
    return "Competitive"


def _run_classification(
    station_obs: dict[int, list[tuple[int, int]]],
    reference_station_codes: set[int] | None = None,
) -> dict[int, tuple[str, int]]:
    """Classify stations against a cluster reference built from reference_station_codes.

    station_obs: {station_code: [(price_date_int, price_decicents), ...]}
    reference_station_codes: if None, use all stations in station_obs as reference.

    Returns {station_code: (class_str, median_premium_decicents)}.
    """
    ref_codes = reference_station_codes if reference_station_codes is not None else set(station_obs)

    # Build daily cluster reference: median of reference stations' prices for each day.
    day_ref_prices: dict[int, list[int]] = {}
    for sc in ref_codes:
        for day, price in station_obs.get(sc, []):
            day_ref_prices.setdefault(day, []).append(price)

    if not day_ref_prices:
        return {}

    daily_ref: dict[int, float] = {
        day: statistics.median(prices) for day, prices in day_ref_prices.items()
    }

    # For each station, compute median premium vs the daily reference.
    result: dict[int, tuple[str, int]] = {}
    for sc, obs in station_obs.items():
        premiums = [price - daily_ref[day] for day, price in obs if day in daily_ref]
        if not premiums:
            continue
        med_prem = _median_int(premiums)
        result[sc] = (_classify_premium(med_prem), med_prem)

    return result


def _classify_lga(
    station_obs: dict[int, list[tuple[int, int]]],
) -> tuple[dict[int, tuple[str, int]], int, int, int]:
    """Two-iteration classifier for one LGA on one snapshot date.

    Returns (iter2_classes, n_competitive, n_sticky, n_discount).
    iter2_classes is empty when zero Competitive stations emerge after iter 2
    (caller must still write a classification_summary row).
    """
    # Iter 1: all-station cluster reference.
    iter1 = _run_classification(station_obs)

    competitive_iter1 = {sc for sc, (cls, _) in iter1.items() if cls == "Competitive"}

    if not competitive_iter1:
        # Cannot build iter-2 reference; treat as zero-Competitive.
        n_comp = sum(1 for cls, _ in iter1.values() if cls == "Competitive")
        n_sticky = sum(1 for cls, _ in iter1.values() if cls == "Sticky")
        n_disc = sum(1 for cls, _ in iter1.values() if cls == "Discount")
        return {}, n_comp, n_sticky, n_disc

    # Iter 2: cluster reference from iter-1 Competitive stations only.
    iter2 = _run_classification(station_obs, reference_station_codes=competitive_iter1)

    n_comp = sum(1 for cls, _ in iter2.values() if cls == "Competitive")
    n_sticky = sum(1 for cls, _ in iter2.values() if cls == "Sticky")
    n_disc = sum(1 for cls, _ in iter2.values() if cls == "Discount")

    if n_comp == 0:
        return {}, n_comp, n_sticky, n_disc

    return iter2, n_comp, n_sticky, n_disc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_snapshot(
    conn: sqlite3.Connection,
    snapshot_date: int,
    fuel_code: str = "E10",
    fuel_type_id: int | None = None,
) -> tuple[int, int]:
    """Classify all stations for one snapshot_date using WINDOW_DAYS-day window.

    Returns (station_class_rows_written, lgas_processed).
    fuel_type_id: optional precomputed id — pass from classify_all to avoid repeated lookups.
    """
    window_start, window_end = _window_bounds(snapshot_date)

    fid = fuel_type_id if fuel_type_id is not None else _db.fuel_type_id(conn, fuel_code)

    # Load all forward-filled prices in the window with station → LGA mapping.
    raw = conn.execute(
        "SELECT dp.station_code, dp.price_date, dp.price_decicents, s.council"
        " FROM daily_prices dp JOIN stations s USING(station_code)"
        " WHERE dp.fuel_type_id = ? AND dp.price_date >= ? AND dp.price_date <= ?"
        " AND s.council IS NOT NULL",
        (fid, window_start, window_end),
    ).fetchall()

    # Group by LGA → station.
    lga_data: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for sc, day, price, council in raw:
        lga_data.setdefault(council, {}).setdefault(sc, []).append((day, price))

    station_class_rows: list[tuple[int, int, str, int]] = []
    summary_rows: list[tuple[int, str, int, int, int]] = []

    for lga, station_obs in lga_data.items():
        iter2_classes, n_comp, n_sticky, n_disc = _classify_lga(station_obs)

        summary_rows.append((snapshot_date, lga, n_comp, n_sticky, n_disc))

        if not iter2_classes:
            # Zero-Competitive: summary written above; no station_class rows.
            logger.warning(
                "classify: zero Competitive stations for LGA %r on %s", lga, snapshot_date
            )
            continue

        for sc, (cls, med_prem) in iter2_classes.items():
            station_class_rows.append((sc, snapshot_date, cls, med_prem))

    _db.upsert_station_class(conn, station_class_rows)
    _db.upsert_classification_summary(conn, summary_rows)
    conn.commit()

    logger.info(
        "classify: snapshot_date=%s, %d station_class rows, %d LGAs",
        snapshot_date, len(station_class_rows), len(lga_data),
    )
    return len(station_class_rows), len(lga_data)


def classify_all(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    """Classify all snapshot dates present in daily_prices.

    start_date / end_date: YYYY-MM-DD bounds (inclusive) on which snapshot_dates
    to classify. Each snapshot_date D uses daily_prices for [D-45, D-1]; the
    date bounds here filter which Ds are processed, not the underlying price data.
    Returns total station_class rows written.
    """
    fid = _db.fuel_type_id(conn, fuel_code)

    dates: list[int] = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT price_date FROM daily_prices WHERE fuel_type_id = ? ORDER BY price_date",
            (fid,),
        )
    ]

    if start_date:
        start_int = int(start_date.replace("-", ""))
        dates = [d for d in dates if d >= start_int]
    if end_date:
        end_int = int(end_date.replace("-", ""))
        dates = [d for d in dates if d <= end_int]

    total = 0
    for snapshot_date in dates:
        rows_written, _ = classify_snapshot(conn, snapshot_date, fuel_type_id=fid)
        total += rows_written

    logger.info("classify_all: %d total station_class rows written", total)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("classify")
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option("--fuel", default="E10", show_default=True, help="Fuel code to classify.")
@click.option(
    "--start",
    "start_date",
    default=None,
    help="First snapshot_date to classify (YYYY-MM-DD). Defaults to earliest date in daily_prices.",
)
@click.option(
    "--end",
    "end_date",
    default=None,
    help="Last snapshot_date to classify (YYYY-MM-DD). Defaults to latest date in daily_prices.",
)
@click.option("-v", "--verbose", is_flag=True)
def main(db_path: str, fuel: str, start_date: str | None, end_date: str | None, verbose: bool) -> None:
    """Classify stations as Competitive / Sticky / Discount and write to station_class table."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")
    conn = _db.open_db(pathlib.Path(db_path))
    _db.create_schema(conn)
    total = classify_all(conn, fuel_code=fuel, start_date=start_date, end_date=end_date)
    click.echo(f"Wrote {total} station_class rows.")
    conn.close()


if __name__ == "__main__":
    main()
