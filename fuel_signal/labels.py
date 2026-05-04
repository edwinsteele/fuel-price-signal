"""Label generation and training-row assembly for the ML signal model.

## Label definition

    label = 1 (BUY) if ALL of:
        future_min >= today_price - threshold_cents          # condition 1: no better deal coming
        AND today_price <= percentile(past lookback_days, percentile_pct)  # condition 2: price is cheap

    label = 0 (WAIT/DON'T BUY) otherwise.

## Why BUY=1, not WAIT=1

Boolean true means "do the thing." WAIT=1 is the natural enemy of readable code and
correct label construction — it inverts intuition at every turn. All label design discussions
should treat 1 as BUY.

## Why two conditions are necessary

Condition 1 alone (no drop coming) fails on high-price plateaus. If prices are stuck at
200c for two weeks with no drop predicted, condition 1 fires (label=1, BUY) — but 200c is
expensive and a bad time to buy. The model has no way to distinguish a cheap trough from an
expensive plateau using only directional information.

Condition 2 (absolute price gate) fixes this. A 200c plateau day fails the percentile gate
and gets label=0 even if no drop is predicted within the horizon. A 160c trough day passes
both conditions and gets label=1.

## Design goal: avoid the top, not find the bottom

The Sydney E10 price cycle is ~45 days. Most users refuel every 1–4 weeks and cannot hold
out for a full cycle to catch the exact trough. Optimising for the trough would produce a
model that tells most users to wait indefinitely.

The label instead targets: "was this price objectively cheap, and was there no obviously
better deal available shortly afterward?" This is a practical, actionable definition that
works across different refueling cadences.

## What the label does NOT encode (decision-layer concerns)

The label is a price question, not a personal-circumstances question. The following factors
belong in the decision layer (the rule that interprets model output), not in the label:

- Tank level / urgency: if the tank is nearly empty, the user may need to buy regardless of
  signal. The label cannot know the user's tank on any historical day.
- Refueling cadence: whether someone refuels weekly or monthly does not change whether a
  historical price was objectively cheap. The horizon (H days) is a structural parameter
  about what counts as "soon", not a user-specific constraint.
- Station availability: the user's preferred station may not be on their route today.
- Budget constraints, upcoming long drives, vehicle changes, etc.

These are real factors — they should be applied as filters on top of P(BUY) at decision
time, not baked into the training target.

## Parameter rationale

horizon_days (default 7):
    Defines "no better deal coming soon." 7 days is a reasonable lower bound for
    "actionable" — if a better price arrives in 3 days, most users could act on it.
    See GitHub issue #27 to revisit (14d was considered for better plateau coverage).

threshold_cents (default 3.0):
    The minimum price improvement that counts as "a better deal." Below this, the
    difference is noise or not worth waiting for.

lookback_days (default 90 ≈ 2 cycles):
    The window over which "cheap" is defined. Deliberately set to ~2 cycle lengths
    rather than a full year. A 365-day lookback is contaminated by long-run price
    drift (general inflation, supply shocks) — prices that look cheap vs a year ago
    may be expensive relative to the current price regime.

percentile_pct (default 33.0):
    The cheapness gate. 33rd percentile ≈ 15 of 45 cycle days count as cheap.
    25th percentile ≈ 11 days. Starting at 33rd for more training signal.
    See GitHub issue #27 to tune this against backtest outcomes.

## Point-in-time discipline

Labels look forward in time intentionally — forward queries are valid here because labels
are computed in hindsight from historical data. Point-in-time discipline (no lookahead)
applies strictly to features, which are built in a later phase.
"""

import datetime
import pathlib
import sqlite3
from itertools import groupby

import click
import numpy as np
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
    lookback_days: int = 90,
    percentile_pct: float = 33.0,
) -> int | None:
    """Return 1 (BUY) if price is cheap and no better deal is coming, else 0.

    Returns None when today's price is missing, fewer than horizon_days of
    forward data exist, or fewer than lookback_days of past data exist.

    date_d: YYYY-MM-DD
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    fid = _db.fuel_type_id(conn, "E10")
    d_int = _to_date_int(date_d)
    d = datetime.date.fromisoformat(date_d)

    row = conn.execute(
        "SELECT price_decicents FROM daily_prices"
        " WHERE station_code = ? AND fuel_type_id = ? AND price_date = ?",
        (station_code, fid, d_int),
    ).fetchone()
    if row is None:
        return None
    today_price = row[0] / 10

    d_end_int = _to_date_int((d + datetime.timedelta(days=horizon_days)).isoformat())
    forward_rows = conn.execute(
        "SELECT price_decicents FROM daily_prices"
        " WHERE station_code = ? AND fuel_type_id = ? AND price_date > ? AND price_date <= ?"
        " ORDER BY price_date",
        (station_code, fid, d_int, d_end_int),
    ).fetchall()
    if len(forward_rows) < horizon_days:
        return None
    future_min = min(r[0] / 10 for r in forward_rows)

    d_start_int = _to_date_int((d - datetime.timedelta(days=lookback_days)).isoformat())
    past_rows = conn.execute(
        "SELECT price_decicents FROM daily_prices"
        " WHERE station_code = ? AND fuel_type_id = ? AND price_date >= ? AND price_date < ?",
        (station_code, fid, d_start_int, d_int),
    ).fetchall()
    if len(past_rows) < lookback_days:
        return None
    price_threshold = float(np.percentile([r[0] / 10 for r in past_rows], percentile_pct))

    no_better_deal = future_min >= today_price - threshold_cents
    price_is_cheap = today_price <= price_threshold
    return 1 if (no_better_deal and price_is_cheap) else 0


def assemble_training_rows(
    conn: sqlite3.Connection,
    horizon_days: int = 7,
    threshold_cents: float = 3.0,
    lookback_days: int = 90,
    percentile_pct: float = 33.0,
    station_codes: list[int] | None = None,
) -> pd.DataFrame:
    """Build training rows with label for every (station, date) with full forward and lookback data.

    Columns: station_code, price_date, today_price_cents, future_min_cents, label.
    No feature columns — those are added in a later phase.
    station_codes=None processes all stations with E10 prices in daily_prices.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")

    cols = ["station_code", "price_date", "today_price_cents", "future_min_cents", "label"]
    if station_codes is not None and len(station_codes) == 0:
        return pd.DataFrame(columns=cols)

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
        for i in range(lookback_days, n - horizon_days):
            date_int, today_price = series[i]
            forward_prices = [p for _, p in series[i + 1 : i + 1 + horizon_days]]
            future_min = min(forward_prices)
            past_prices = [p for _, p in series[i - lookback_days : i]]
            price_threshold = float(np.percentile(past_prices, percentile_pct))

            no_better_deal = future_min >= today_price - threshold_cents
            price_is_cheap = today_price <= price_threshold
            label = 1 if (no_better_deal and price_is_cheap) else 0
            records.append({
                "station_code": station_code,
                "price_date": _from_date_int(date_int),
                "today_price_cents": today_price,
                "future_min_cents": future_min,
                "label": label,
            })

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
@click.option("--horizon", type=click.IntRange(min=1), default=7, show_default=True, help="Forward horizon in days.")
@click.option("--threshold", default=3.0, show_default=True, help="Minimum price drop (cents) to signal a better deal.")
@click.option("--lookback", default=90, show_default=True, help="Past days for price percentile (~2 cycles).")
@click.option("--percentile", default=33.0, show_default=True, help="Percentile gate for 'price is cheap' condition.")
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite DB.",
)
def main(output: str, horizon: int, threshold: float, lookback: int, percentile: float, db_path: str) -> None:
    """Assemble ML training rows with BUY-signal labels (label=1 means buy)."""
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )

    conn = _db.open_db(path)
    df = assemble_training_rows(
        conn,
        horizon_days=horizon,
        threshold_cents=threshold,
        lookback_days=lookback,
        percentile_pct=percentile,
    )
    conn.close()

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    click.echo(f"Wrote {len(df):,} rows ({int(df['label'].sum()):,} positive) to {out_path}")


if __name__ == "__main__":
    main()
