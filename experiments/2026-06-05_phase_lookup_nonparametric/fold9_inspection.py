"""Inspect fold 9 (2023-10-26 → 2024-01-23) cycle structure.

Hypothesis (user observation): fold 9 starts with an extended downtick where
cycle phase runs longer than normal. Quantify and visualise.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

HERE = pathlib.Path(__file__).parent

FOLD9_START = pd.Timestamp("2023-10-26")
FOLD9_END = pd.Timestamp("2024-01-23")

# Comparison: surrounding 90-day windows + the full dataset for reference.
PRE_START = FOLD9_START - pd.Timedelta(days=90)
POST_END = FOLD9_END + pd.Timedelta(days=90)


def summarise(df: pd.DataFrame, label: str) -> dict:
    """One-line summary of cycle structure for a date-bounded window."""
    if df.empty:
        return {"label": label, "n_rows": 0}
    pct = df["cycle_pct_through"].to_numpy(dtype=float)
    dsp = df["cycle_days_since_peak"].to_numpy(dtype=float)
    mcl = df["cycle_mean_length"].to_numpy(dtype=float)
    last_min = df["cycle_last_min_cents"].to_numpy(dtype=float)
    last_max = df["cycle_last_max_cents"].to_numpy(dtype=float)
    amp = last_max - last_min
    station = df["station_price_cents"].to_numpy(dtype=float)
    return {
        "label": label,
        "n_rows": len(df),
        "n_unique_dates": df["price_date"].nunique(),
        "pct_through_mean": pct.mean(),
        "pct_through_median": float(np.median(pct)),
        "pct_through_p90": float(np.quantile(pct, 0.90)),
        "pct_through_p99": float(np.quantile(pct, 0.99)),
        "frac_pct_gt_1": float((pct > 1.0).mean()),
        "frac_pct_gt_1_5": float((pct > 1.5).mean()),
        "days_since_peak_mean": dsp.mean(),
        "days_since_peak_p90": float(np.quantile(dsp, 0.90)),
        "mean_cycle_length_mean": mcl.mean(),
        "amplitude_mean": float(np.mean(amp[amp > 0])),
        "station_minus_min_mean": float(np.mean(station - last_min)),
    }


def main() -> None:
    print("Loading features …")
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    print(f"  total rows: {len(df):,}  date range: {df['price_date'].min().date()} → {df['price_date'].max().date()}")

    masks = {
        "FULL DATASET": np.ones(len(df), dtype=bool),
        "PRE  (Jul–Oct 2023, 90d before fold 9)": (df["price_date"] >= PRE_START) & (df["price_date"] < FOLD9_START),
        "FOLD 9  (Oct 2023 – Jan 2024)": (df["price_date"] >= FOLD9_START) & (df["price_date"] <= FOLD9_END),
        "POST (Jan–Apr 2024, 90d after fold 9)": (df["price_date"] > FOLD9_END) & (df["price_date"] <= POST_END),
    }

    summaries = []
    for label, mask in masks.items():
        summaries.append(summarise(df[mask], label))

    out = pd.DataFrame(summaries)
    print("\nCycle-structure summary by window:")
    print(out.to_string(index=False))
    out.to_csv(HERE / "fold9_summary.csv", index=False)

    # Daily Sydney-average price for plot — 60d before fold 9 through 60d after.
    plot_start = PRE_START - pd.Timedelta(days=30)
    plot_end = POST_END + pd.Timedelta(days=30)
    daily = (df[(df["price_date"] >= plot_start) & (df["price_date"] <= plot_end)]
             .groupby("price_date", as_index=False)
             .agg(price=("station_price_cents", "mean"),
                  pct=("cycle_pct_through", "mean"),
                  dsp=("cycle_days_since_peak", "mean"),
                  mcl=("cycle_mean_length", "mean")))
    daily = daily.sort_values("price_date")

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(daily["price_date"], daily["price"], lw=1.2, color="black")
    axes[0].axvspan(FOLD9_START, FOLD9_END, alpha=0.12, color="red", label="fold 9 val window")
    axes[0].set_ylabel("Sydney avg price (cents)")
    axes[0].set_title("Fold 9 inspection — price, phase, days_since_peak")
    axes[0].legend(loc="upper right")

    axes[1].plot(daily["price_date"], daily["pct"], lw=1.2, color="tab:blue")
    axes[1].axhline(1.0, color="grey", linestyle=":", alpha=0.6, label="pct=1 (cycle length elapsed)")
    axes[1].axhline(1.5, color="grey", linestyle="--", alpha=0.6, label="pct=1.5 (lookup clip)")
    axes[1].axvspan(FOLD9_START, FOLD9_END, alpha=0.12, color="red")
    axes[1].set_ylabel("mean cycle_pct_through")
    axes[1].legend(loc="upper right")

    axes[2].plot(daily["price_date"], daily["dsp"], lw=1.2, color="tab:green", label="days_since_peak (avg)")
    axes[2].plot(daily["price_date"], daily["mcl"], lw=1.0, color="tab:orange",
                 alpha=0.7, label="mean_cycle_length (avg)")
    axes[2].axvspan(FOLD9_START, FOLD9_END, alpha=0.12, color="red")
    axes[2].set_ylabel("days")
    axes[2].set_xlabel("date")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(HERE / "fold9_inspection.png", dpi=120)
    print(f"\nSaved: {HERE / 'fold9_inspection.png'}")
    print(f"Saved: {HERE / 'fold9_summary.csv'}")


if __name__ == "__main__":
    main()
