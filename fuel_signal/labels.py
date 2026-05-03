"""Label generation and training-row assembly for the ML signal model.

Labels look forward in time intentionally — forward queries are allowed here.
Point-in-time discipline applies to features (built in a later phase).
"""

import datetime
import pathlib
import sqlite3
from itertools import groupby

import click
import pandas as pd

from fuel_signal import db as _db


def _to_date_int(s: str) -> int:
    """'2024-01-15' → 20240115"""
    return int(s.replace("-", ""))


def _from_date_int(v: int) -> str:
    """20240115 → '2024-01-15'"""
    s = str(v)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def compute_label(
    conn: sqlite3.Connection,
    station_code: int,
    date_d: str,
    horizon_days: int = 7,
    threshold_cents: float = 3.0,
) -> int | None:
    """Return 1 if forward-min price drops below threshold, else 0.

    Returns None when today's price is missing or fewer than horizon_days of
    forward data exist — these rows are excluded from training.

    date_d: YYYY-MM-DD
    """
    fid = _db.fuel_type_id(conn, "E10")
    d_int = _to_date_int(date_d)

    row = conn.execute(
        "SELECT price_decicents FROM daily_prices"
        " WHERE station_code = ? AND fuel_type_id = ? AND price_date = ?",
        (station_code, fid, d_int),
    ).fetchone()
    if row is None:
        return None
    today_price = row[0] / 10

    d_end = datetime.date.fromisoformat(date_d) + datetime.timedelta(days=horizon_days)
    d_end_int = _to_date_int(d_end.isoformat())

    forward_rows = conn.execute(
        "SELECT price_decicents FROM daily_prices"
        " WHERE station_code = ? AND fuel_type_id = ? AND price_date > ? AND price_date <= ?"
        " ORDER BY price_date",
        (station_code, fid, d_int, d_end_int),
    ).fetchall()

    if len(forward_rows) < horizon_days:
        return None

    future_min = min(r[0] / 10 for r in forward_rows)
    return 1 if future_min < today_price - threshold_cents else 0


def assemble_training_rows(
    conn: sqlite3.Connection,
    horizon_days: int = 7,
    threshold_cents: float = 3.0,
    station_codes: list[int] | None = None,
) -> pd.DataFrame:
    """Build training rows with label for every (station, date) with full forward data.

    Columns: station_code, price_date, today_price_cents, future_min_cents, label.
    No feature columns — those are added in a later phase.
    station_codes=None processes all stations with E10 prices in daily_prices.
    """
    fid = _db.fuel_type_id(conn, "E10")

    if station_codes is not None:
        placeholders = ",".join("?" * len(station_codes))
        raw_rows = conn.execute(
            "SELECT station_code, price_date, price_decicents FROM daily_prices"
            f" WHERE fuel_type_id = ? AND station_code IN ({placeholders})"
            " ORDER BY station_code, price_date",
            [fid, *station_codes],
        ).fetchall()
    else:
        raw_rows = conn.execute(
            "SELECT station_code, price_date, price_decicents FROM daily_prices"
            " WHERE fuel_type_id = ? ORDER BY station_code, price_date",
            (fid,),
        ).fetchall()

    records: list[dict] = []
    for station_code, station_iter in groupby(raw_rows, key=lambda r: r[0]):
        series = [(r[1], r[2] / 10) for r in station_iter]  # [(date_int, price_cents)]
        n = len(series)
        for i in range(n - horizon_days):
            date_int, today_price = series[i]
            forward_prices = [p for _, p in series[i + 1 : i + 1 + horizon_days]]
            future_min = min(forward_prices)
            label = 1 if future_min < today_price - threshold_cents else 0
            records.append({
                "station_code": station_code,
                "price_date": _from_date_int(date_int),
                "today_price_cents": today_price,
                "future_min_cents": future_min,
                "label": label,
            })

    cols = ["station_code", "price_date", "today_price_cents", "future_min_cents", "label"]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records, columns=cols)


@click.command("labels")
@click.option(
    "--output",
    default="data/labels.csv",
    show_default=True,
    help="Output CSV path.",
)
@click.option("--horizon", default=7, show_default=True, help="Forward horizon in days.")
@click.option(
    "--threshold",
    default=3.0,
    show_default=True,
    help="Minimum price drop (cents) to label as 1.",
)
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite DB.",
)
def main(output: str, horizon: int, threshold: float, db_path: str) -> None:
    """Assemble ML training rows with buy-signal labels."""
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )

    conn = _db.open_db(path)
    df = assemble_training_rows(conn, horizon_days=horizon, threshold_cents=threshold)
    conn.close()

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    click.echo(f"Wrote {len(df):,} rows ({int(df['label'].sum()):,} positive) to {out_path}")


if __name__ == "__main__":
    main()
