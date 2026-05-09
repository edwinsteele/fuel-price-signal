"""Diagnostic: empirical TP benefit from label=1 rows.

A true positive (TP) occurs when the model says BUY on a label=1 day — a day
where the price was objectively cheap and no better deal was coming within the
horizon. The user buys today at today_price_cents.

TP benefit for a given row:

    benefit = mean(price at same station over next horizon_days) - today_price_cents

Positive benefit: future prices are higher than today — buying now saves that
many cents compared to waiting a random number of days in the horizon.
Near-zero benefit: price was flat — buying now vs. waiting made little difference.

This diagnostic grounds the TP reward in the cost model. The current value (3.0c)
is the label threshold floor — the minimum qualifying saving. If the empirical
mean benefit is substantially higher, the cost model understates TP value and
inflates the relative weight of FP/FN penalties, pushing tau too low.

Usage
-----
    uv run python -m fuel_signal.tp_benefit
    uv run python -m fuel_signal.tp_benefit --horizon 14 --plot data/tp_benefit_14d.png
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

_CURRENT_TP_REWARD: float = 3.0
DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_PLOT_PATH = pathlib.Path("data/tp_benefit_distribution.png")
DEFAULT_HORIZON_DAYS: int = 7


def _date_plus(iso_date: str, days: int) -> str:
    return (datetime.date.fromisoformat(iso_date) + datetime.timedelta(days=days)).isoformat()


def compute_tp_benefit(
    conn: sqlite3.Connection,
    features_df: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Return label=1 rows augmented with benefit = mean_future_price - today_price.

    The mean is taken over the next horizon_days days at the same station.
    Rows where fewer than horizon_days future prices exist in daily_prices are dropped.
    All daily_prices are loaded in one bulk SELECT; no per-row DB queries.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    tp = features_df[features_df["label"] == 1][
        ["station_code", "price_date", "today_price_cents"]
    ].copy()

    if tp.empty:
        tp["future_avg_cents"] = pd.Series(dtype=float)
        tp["benefit"] = pd.Series(dtype=float)
        return tp

    fid = _db.fuel_type_id(conn, "E10")

    price_by_key: dict[tuple[int, str], float] = {}
    for sc, date_int, decicents in conn.execute(
        "SELECT station_code, price_date, price_decicents FROM daily_prices"
        " WHERE fuel_type_id = ?",
        (fid,),
    ):
        s = str(date_int)
        iso = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        price_by_key[(sc, iso)] = decicents / 10

    future_avgs = []
    for _, row in tp.iterrows():
        station = int(row["station_code"])
        prices = [
            price_by_key.get((station, _date_plus(row["price_date"], d)))
            for d in range(1, horizon_days + 1)
        ]
        if any(p is None for p in prices):
            future_avgs.append(None)
        else:
            future_avgs.append(sum(prices) / len(prices))  # type: ignore[arg-type]

    tp["future_avg_cents"] = future_avgs
    tp = tp.dropna(subset=["future_avg_cents"])
    tp["benefit"] = tp["future_avg_cents"] - tp["today_price_cents"]
    return tp.reset_index(drop=True)


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


def format_summary(tp: pd.DataFrame, horizon_days: int = DEFAULT_HORIZON_DAYS) -> str:
    s = _stats(tp["benefit"])
    w = 13

    def _fmt_n(d: dict) -> str:
        return f"{int(d['n']):>{w},}" if not np.isnan(d["n"]) else f"{'—':>{w}}"

    def _fmt_c(d: dict, key: str) -> str:
        v = d[key]
        return f"{v:>{w - 1}.2f}c" if not np.isnan(v) else f"{'—':>{w}}"

    if not np.isnan(s["mean"]):
        mean_val = s["mean"]
        suggestion = f"  → Suggested TP reward: {mean_val:.2f}c (mean; horizon = {horizon_days}d)"
        if abs(mean_val - _CURRENT_TP_REWARD) > 0.5:
            suggestion += f"  ** differs from current {_CURRENT_TP_REWARD:.1f}c **"
    else:
        suggestion = "  → No data to suggest TP reward."

    rows = [
        f"TP benefit analysis — {len(tp):,} label=1 rows with {horizon_days}d horizon",
        f"  Horizon: {horizon_days} days  |  Current TP reward: {_CURRENT_TP_REWARD:.1f}c",
        "",
        f"  {'':22s}{'label=1 rows':>{w}}",
        "  " + "-" * (22 + w),
        f"  {'rows':<22s}{_fmt_n(s)}",
        f"  {'mean benefit':<22s}{_fmt_c(s, 'mean')}",
        f"  {'median benefit':<22s}{_fmt_c(s, 'median')}",
        f"  {'p25 benefit':<22s}{_fmt_c(s, 'p25')}",
        f"  {'p75 benefit':<22s}{_fmt_c(s, 'p75')}",
        f"  {'p90 benefit':<22s}{_fmt_c(s, 'p90')}",
        "  " + "-" * (22 + w),
        suggestion,
    ]
    return "\n".join(rows)


def plot_tp_distribution(
    tp: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    out_path: pathlib.Path = DEFAULT_PLOT_PATH,
) -> pathlib.Path | None:
    """Save histogram of TP benefit to out_path. Returns out_path on success, None if skipped."""
    if tp.empty:
        return None

    benefit = tp["benefit"]
    lo = float(benefit.quantile(0.005))
    hi = float(benefit.quantile(0.995))
    if np.isnan(lo) or np.isnan(hi) or lo >= hi:
        lo = float(benefit.min()) - 0.5
        hi = float(benefit.max()) + 0.5
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(benefit, bins=bins, alpha=0.75, color="seagreen",
            label=f"label=1 rows ({len(tp):,})")

    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.2, label="0c (break-even)")
    ax.axvline(_CURRENT_TP_REWARD, color="red", linestyle="--", linewidth=1.5,
               label=f"Current TP reward ({_CURRENT_TP_REWARD:.1f}c)")

    if not benefit.empty:
        mean_val = float(benefit.mean())
        ax.axvline(mean_val, color="darkorange", linestyle="-", linewidth=1.8,
                   label=f"Mean ({mean_val:.2f}c)")

    ax.set_xlabel("mean_future_price − today_price  (cents)")
    ax.set_ylabel("Row count")
    ax.set_title(
        f"TP benefit distribution — label=1 rows, {horizon_days}-day horizon\n"
        "(saving from a correct BUY vs. waiting a random day in the horizon; positive = future is higher)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


@click.command("tp-benefit")
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
    help="Output PNG path for the benefit distribution plot.",
)
@click.option(
    "--horizon", "horizon_days",
    default=DEFAULT_HORIZON_DAYS,
    show_default=True,
    type=click.IntRange(min=1),
    help="Days over which future average price is computed (match labels.py --horizon).",
)
def main(features_csv: str, db_path: str, plot_path: str, horizon_days: int) -> None:
    """Empirical TP benefit diagnostic: saving from correct BUY signals on label=1 days.

    For each label=1 row, computes the benefit of saying BUY: the mean price over
    the next horizon_days at the same station minus today's price. Reports the
    distribution and suggests a TP reward for use in the cost model (score_phase2.py).
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
    try:
        tp = compute_tp_benefit(conn, df, horizon_days)
    finally:
        conn.close()

    click.echo(format_summary(tp, horizon_days))
    click.echo()

    out = pathlib.Path(plot_path)
    written = plot_tp_distribution(tp, horizon_days, out)
    if written:
        click.echo(f"Wrote plot to {out}")
    else:
        click.echo("Skipped plot — no label=1 rows with full future price coverage.")


if __name__ == "__main__":
    main()
