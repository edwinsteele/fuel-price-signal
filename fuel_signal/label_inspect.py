"""Interactive label inspector: show label decomposition for one station over N days.

Prints a table of the two label conditions side-by-side so you can see exactly
why each day is BUY or WAIT.

Usage:
    uv run python -m fuel_signal.label_inspect --station 585 --date 2024-03-01
    uv run python -m fuel_signal.label_inspect --station 585 --date 2024-03-01 --days 21
"""

import datetime
import pathlib

import click
import numpy as np

from fuel_signal import db as _db


def _to_date_int(s: str) -> int:
    return int(s.replace("-", ""))


def _from_date_int(v: int) -> str:
    s = str(v)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def inspect_station(
    conn,
    station_code: int,
    start_date: str,
    n_days: int = 21,
    horizon_days: int = 7,
    threshold_cents: float = 3.0,
    lookback_days: int = 90,
    percentile_pct: float = 33.0,
) -> None:
    fid = _db.fuel_type_id(conn, "E10")

    start = datetime.date.fromisoformat(start_date)
    fetch_start = start - datetime.timedelta(days=lookback_days + 1)
    fetch_end = start + datetime.timedelta(days=n_days + horizon_days)

    rows = conn.execute(
        "SELECT price_date, price_decicents FROM daily_prices"
        " WHERE station_code = ? AND fuel_type_id = ?"
        "   AND price_date >= ? AND price_date <= ?"
        " ORDER BY price_date",
        (
            station_code,
            fid,
            _to_date_int(fetch_start.isoformat()),
            _to_date_int(fetch_end.isoformat()),
        ),
    ).fetchall()

    series = {_from_date_int(r[0]): r[1] / 10 for r in rows}

    stn_row = conn.execute(
        "SELECT name, suburb FROM stations WHERE station_code = ?", (station_code,)
    ).fetchone()
    stn_label = f"{stn_row[0]} ({stn_row[1]})" if stn_row else f"station {station_code}"

    header = (
        f"{'Date':<12} {'Price':>7} {'FutMin':>7} {'P{:.0f}':>7} "
        f"{'Cheap?':>6} {'NoDrop?':>7} {'Label':>6}"
    ).format(percentile_pct)
    sep = "-" * len(header)
    click.echo(f"\nStation {station_code}: {stn_label}")
    click.echo(f"Params: horizon={horizon_days}d  threshold={threshold_cents}c  "
               f"lookback={lookback_days}d  percentile={percentile_pct:.0f}th")
    click.echo(sep)
    click.echo(header)
    click.echo(sep)

    for offset in range(n_days):
        d = start + datetime.timedelta(days=offset)
        date_str = d.isoformat()

        today_price = series.get(date_str)
        if today_price is None:
            click.echo(f"{date_str:<12} {'(no data)':>7}")
            continue

        forward = [
            series.get((d + datetime.timedelta(days=k)).isoformat())
            for k in range(1, horizon_days + 1)
        ]
        if any(v is None for v in forward):
            click.echo(f"{date_str:<12} {today_price:>7.1f} {'(insufficient forward)':>40}")
            continue

        past = [
            series.get((d - datetime.timedelta(days=k)).isoformat())
            for k in range(1, lookback_days + 1)
        ]
        if any(v is None for v in past):
            click.echo(f"{date_str:<12} {today_price:>7.1f} {'(insufficient history)':>40}")
            continue

        future_min = min(forward)
        p_threshold = float(np.percentile(past, percentile_pct))
        no_drop = future_min >= today_price - threshold_cents
        cheap = today_price <= p_threshold
        label = 1 if (no_drop and cheap) else 0

        label_str = click.style("BUY ", fg="green", bold=True) if label else "WAIT"
        cheap_str = click.style("Y", fg="green") if cheap else click.style("N", fg="red")
        drop_str = click.style("Y", fg="green") if no_drop else click.style("N", fg="red")

        click.echo(
            f"{date_str:<12} {today_price:>7.1f} {future_min:>7.1f} {p_threshold:>7.1f} "
            f"  {cheap_str}      {drop_str}    {label_str}"
        )

    click.echo(sep)


@click.command("label-inspect")
@click.option("--station", "station_code", default=585, show_default=True, type=int,
              help="FuelCheck station code.")
@click.option("--date", "start_date", required=True, help="Start date YYYY-MM-DD.")
@click.option("--days", "n_days", default=21, show_default=True, type=int,
              help="Number of days to display.")
@click.option("--horizon", default=7, show_default=True, type=int,
              help="Forward horizon (days).")
@click.option("--threshold", default=3.0, show_default=True,
              help="Min price drop (cents) to count as 'better deal'.")
@click.option("--lookback", default=90, show_default=True, type=int,
              help="Past days for percentile gate.")
@click.option("--percentile", default=33.0, show_default=True,
              help="Cheapness percentile gate.")
@click.option("--db", "db_path", default=str(_db.DEFAULT_DB_PATH), show_default=True,
              help="Path to SQLite DB.")
def main(
    station_code: int,
    start_date: str,
    n_days: int,
    horizon: int,
    threshold: float,
    lookback: int,
    percentile: float,
    db_path: str,
) -> None:
    """Show per-day label decomposition for one station."""
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(f"Database not found: {db_path}")

    conn = _db.open_db(path)
    inspect_station(
        conn,
        station_code=station_code,
        start_date=start_date,
        n_days=n_days,
        horizon_days=horizon,
        threshold_cents=threshold,
        lookback_days=lookback,
        percentile_pct=percentile,
    )
    conn.close()


if __name__ == "__main__":
    main()
