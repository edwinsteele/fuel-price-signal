"""Diagnostic: empirical FN cost from label=1 rows.

A false negative (FN) occurs when the model says WAIT on a label=1 day — a day
where the price was objectively cheap and no better deal was coming within the
horizon. The user defers the purchase and buys delay_days later at whatever
price their station charges then.

FN cost (damage) for a given row:

    damage = price_delay_days_later - today_price_cents

Positive damage: price rose — you paid more by waiting.
Negative damage: price fell — waiting was actually cheaper (rare on label=1 days,
since condition 1 says no drop of > threshold_cents was coming within 7 days;
a fall after day 7 is possible).

Unlike FP cost (which is fully contained in the features CSV), FN cost requires
a DB query because the price delay_days after each label=1 row is not pre-computed.
All daily_prices are loaded in one bulk SELECT and joined in memory — one query
regardless of how many label=1 rows there are.

Usage
-----
    uv run python -m fuel_signal.fn_cost
    uv run python -m fuel_signal.fn_cost --delay 14 --plot data/fn_cost_14d.png
"""

from __future__ import annotations

import datetime
import pathlib
import sqlite3

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal import db as _db

_CURRENT_FN_PENALTY: float = 0.0
DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_PLOT_PATH = pathlib.Path("data/fn_cost_distribution.png")
DEFAULT_DELAY_DAYS: int = 7


def _date_plus(iso_date: str, days: int) -> str:
    """'2024-06-15' + 7 → '2024-06-22'"""
    return (datetime.date.fromisoformat(iso_date) + datetime.timedelta(days=days)).isoformat()


def compute_fn_damage(
    conn: sqlite3.Connection,
    features_df: pd.DataFrame,
    delay_days: int = DEFAULT_DELAY_DAYS,
) -> pd.DataFrame:
    """Return label=1 rows augmented with damage = price_delay_days_later - today_price.

    Rows where the future price is missing from daily_prices are dropped.
    All daily_prices are loaded in one bulk SELECT; no per-row DB queries.
    """
    fn = features_df[features_df["label"] == 1][
        ["station_code", "price_date", "today_price_cents"]
    ].copy()

    if fn.empty:
        fn["future_price_cents"] = pd.Series(dtype=float)
        fn["damage"] = pd.Series(dtype=float)
        return fn

    fid = _db.fuel_type_id(conn, "E10")

    # One bulk SELECT → dict[(station_code, iso_date)] → price_cents
    price_by_key: dict[tuple[int, str], float] = {}
    for sc, date_int, decicents in conn.execute(
        "SELECT station_code, price_date, price_decicents FROM daily_prices"
        " WHERE fuel_type_id = ?",
        (fid,),
    ):
        s = str(date_int)
        iso = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        price_by_key[(sc, iso)] = decicents / 10

    future_prices = []
    for _, row in fn.iterrows():
        target = _date_plus(row["price_date"], delay_days)
        future_prices.append(price_by_key.get((int(row["station_code"]), target)))

    fn["future_price_cents"] = future_prices
    fn = fn.dropna(subset=["future_price_cents"])
    fn["damage"] = fn["future_price_cents"] - fn["today_price_cents"]
    return fn.reset_index(drop=True)


def _stats(series: pd.Series) -> dict:
    if series.empty:
        return {k: float("nan") for k in ("n", "mean", "median", "p25", "p75", "p90")}
    return {
        "n": len(series),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
    }


def format_summary(fn: pd.DataFrame, delay_days: int = DEFAULT_DELAY_DAYS) -> str:
    s = _stats(fn["damage"])
    w = 13

    def _fmt_n(d: dict) -> str:
        return f"{int(d['n']):>{w},}" if not np.isnan(d["n"]) else f"{'—':>{w}}"

    def _fmt_c(d: dict, key: str) -> str:
        v = d[key]
        return f"{v:>{w - 1}.2f}c" if not np.isnan(v) else f"{'—':>{w}}"

    med = s["median"]
    suggestion = (
        f"  → Suggested FN penalty: {med:.2f}c (median; delay assumption = {delay_days}d)"
        if not np.isnan(med)
        else "  → No data to suggest FN penalty."
    )
    if not np.isnan(med) and abs(med - _CURRENT_FN_PENALTY) > 0.5:
        suggestion += f"  ** differs from current {_CURRENT_FN_PENALTY:.1f}c **"

    rows = [
        f"FN cost analysis — {len(fn):,} label=1 rows with {delay_days}d future price",
        f"  Delay assumption: {delay_days} days  |  Current FN penalty: {_CURRENT_FN_PENALTY:.1f}c",
        "",
        f"  {'':22s}{'label=1 rows':>{w}}",
        "  " + "-" * (22 + w),
        f"  {'rows':<22s}{_fmt_n(s)}",
        f"  {'mean damage':<22s}{_fmt_c(s, 'mean')}",
        f"  {'median damage':<22s}{_fmt_c(s, 'median')}",
        f"  {'p25 damage':<22s}{_fmt_c(s, 'p25')}",
        f"  {'p75 damage':<22s}{_fmt_c(s, 'p75')}",
        f"  {'p90 damage':<22s}{_fmt_c(s, 'p90')}",
        "  " + "-" * (22 + w),
        suggestion,
    ]
    return "\n".join(rows)


def plot_fn_distribution(
    fn: pd.DataFrame,
    delay_days: int = DEFAULT_DELAY_DAYS,
    out_path: pathlib.Path = DEFAULT_PLOT_PATH,
) -> None:
    """Save histogram of FN damage to out_path."""
    if fn.empty:
        return

    damage = fn["damage"]
    lo = float(damage.quantile(0.005))
    hi = float(damage.quantile(0.995))
    if np.isnan(lo) or np.isnan(hi) or lo >= hi:
        lo = float(damage.min()) - 0.5
        hi = float(damage.max()) + 0.5
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(damage, bins=bins, alpha=0.75, color="steelblue",
            label=f"label=1 rows ({len(fn):,})")

    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.2, label="0c (break-even)")
    ax.axvline(_CURRENT_FN_PENALTY, color="red", linestyle="--", linewidth=1.5,
               label=f"Current FN penalty ({_CURRENT_FN_PENALTY:.1f}c)")

    if not damage.empty:
        med = float(damage.median())
        ax.axvline(med, color="darkorange", linestyle="-", linewidth=1.8,
                   label=f"Median ({med:.2f}c)")

    ax.set_xlabel("price_delay_days_later − today_price  (cents)")
    ax.set_ylabel("Row count")
    ax.set_title(
        f"FN damage distribution — label=1 rows, {delay_days}-day delay assumption\n"
        "(cost of a wrong WAIT decision; positive = price rose, negative = price fell)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command("fn-cost")
@click.option(
    "--features-csv", "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Features CSV produced by 'python -m fuel_signal.features'.",
)
@click.option(
    "--db", "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite DB (needed to look up future prices).",
)
@click.option(
    "--plot", "plot_path",
    default=str(DEFAULT_PLOT_PATH),
    show_default=True,
    help="Output PNG path for the damage distribution plot.",
)
@click.option(
    "--delay", "delay_days",
    default=DEFAULT_DELAY_DAYS,
    show_default=True,
    type=click.IntRange(min=1),
    help="Days after the label=1 date at which the user is assumed to buy.",
)
def main(features_csv: str, db_path: str, plot_path: str, delay_days: int) -> None:
    """Empirical FN cost diagnostic: price rise after missed BUY signals.

    For each label=1 row, computes the cost of saying WAIT: price delay_days
    later minus today's price. Reports the distribution and suggests an FN
    penalty for use in the cost model (score_phase2.py).
    """
    src = pathlib.Path(features_csv)
    if not src.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )
    db = pathlib.Path(db_path)
    if not db.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. "
            "Run 'uv run python -m fuel_signal.db' first."
        )

    required = {"station_code", "price_date", "today_price_cents", "label"}
    df = pd.read_csv(src)
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(
            f"Features CSV is missing columns: {sorted(missing)}. "
            "Re-run 'uv run python -m fuel_signal.features' to regenerate."
        )

    conn = _db.open_db(db)
    fn = compute_fn_damage(conn, df, delay_days)
    conn.close()

    click.echo(format_summary(fn, delay_days))
    click.echo()

    out = pathlib.Path(plot_path)
    plot_fn_distribution(fn, delay_days, out)
    click.echo(f"Wrote plot to {out}")


if __name__ == "__main__":
    main()
