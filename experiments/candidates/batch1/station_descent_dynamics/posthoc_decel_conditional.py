"""Post-hoc: does station_descent_decel actually predict restorations? (2026-08-25)

NOT pipeline-generated. Run interactively while reviewing the fps-6to dossier, to
answer a question the graded run could not: the headline sits inside the noise band,
but is that because the columns are inert, or because the edge is small with a
fat-tailed payoff that 14 folds cannot resolve?

Method: read this run's own rowpreds.parquet (all 714 stations, 771k rows -- far
wider than the 5 preferred stations the backtest fills cover). Forward 3-day price
change is recovered from the BACKWARD change observed three days later:
station_px_change_3d at t+3 == price(t+3) - price(t). No new data is loaded and no
model is refit; this is a conditional-distribution read of columns already computed
and PIT-validated by the graded run.

Caveat: forward return != realised saving. It ignores the tank/forced-fill mechanics
that amplified fold 13, so the edge measured here is a LOWER bound on what acting on
the signal is worth, and says nothing about whether the model can capture it.

Usage:
    PYTHONPATH=. uv run python \
      experiments/candidates/batch1/station_descent_dynamics/posthoc_decel_conditional.py
"""
from __future__ import annotations

import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent
STEEP_DESCENT_CENTS = -15  # 14-day change at or below this counts as a steep descent
CLIFF_CENTS = 25  # forward 3d jump at or above this is a restoration cliff


def load_rows() -> pd.DataFrame:
    """One row per (fold, station, date) with the candidate columns and a forward return."""
    df = pd.read_parquet(
        HERE / "rowpreds.parquet",
        columns=[
            "fold", "station_code", "price_date", "label",
            "station_px_change_3d", "station_px_change_14d",
            "station_descent_decel", "run", "seed",
        ],
    )
    # The candidate columns are a property of the frame, not the arm -- any single
    # (run, seed) gives the full row set exactly once.
    df = df[(df.run == "R0") & (df.seed == 42)].drop(columns=["run", "seed"])
    df["price_date"] = pd.to_datetime(df.price_date)
    df = df.sort_values(["fold", "station_code", "price_date"])

    key = ["fold", "station_code", "price_date"]
    backward = df.set_index(key).station_px_change_3d
    shifted = backward.copy()
    shifted.index = pd.MultiIndex.from_arrays([
        shifted.index.get_level_values(0),
        shifted.index.get_level_values(1),
        shifted.index.get_level_values(2) - pd.Timedelta(days=3),
    ])
    df["fwd_px_3d"] = df.set_index(key).index.map(shifted)
    return df.dropna(subset=["fwd_px_3d"])


def summarise(label: str, rows: pd.DataFrame, base_mean: float) -> None:
    if len(rows) < 200:
        print(f"  {label:<34} n={len(rows)} (too few)")
        return
    fwd = rows.fwd_px_3d
    print(
        f"  {label:<34} n={len(rows):6d}  mean={fwd.mean():+6.2f}"
        f"  edge={fwd.mean() - base_mean:+6.2f}"
        f"  P(cliff>+{CLIFF_CENTS}c)={100 * (fwd > CLIFF_CENTS).mean():5.2f}%"
        f"  p95={fwd.quantile(0.95):+6.1f}  label={rows.label.mean():.3f}"
    )


def main() -> None:
    df = load_rows()
    base_mean = df.fwd_px_3d.mean()
    print(f"rows={len(df)}  stations={df.station_code.nunique()}  "
          f"dates {df.price_date.min().date()}..{df.price_date.max().date()}")
    print(f"\nBASE RATE  mean={base_mean:+.2f}  "
          f"P(cliff>+{CLIFF_CENTS}c)={100 * (df.fwd_px_3d > CLIFF_CENTS).mean():.2f}%")

    steep = df[df.station_px_change_14d <= STEEP_DESCENT_CENTS]
    print(f"\nWITHIN STEEP DESCENTS (station_px_change_14d <= {STEEP_DESCENT_CENTS}), "
          f"by station_descent_decel:")
    for lo, hi in [(-99, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 99)]:
        bucket = steep[steep.station_descent_decel.between(lo, hi, inclusive="left")]
        summarise(f"decel [{lo:+3d},{hi:+3d})", bucket, base_mean)

    print("\nThe two decisive fold-13 buys sat at decel +4.93 (18517, 2024-10-20) "
          "and +5.36 (585, 2024-11-20).")


if __name__ == "__main__":
    main()
