"""Oracle diagnostic: d(network_px_std)/dt by cycle type, train-only.

Pre-classifies each cycle by its eventual full-cycle properties (length and
descent slope at trough — uses future info, oracle view) and plots the
dispersion-rate trajectory through phase for each cycle type.

Restricted to data before fold 1's val_start (2021-11-05) so the diagnostic
sees only the walk-forward training segment. Earliest-val_start cutoff:
``TRAIN_END_EXCL = "2021-11-01"`` — strict less-than-this excludes every
fold's val window.

This is exploratory only. The cycle-type labels use the full cycle, so any
signal found cannot directly become a PIT-safe feature. The diagnostic
answers "does the information exist in the train data at all?" — not "what
feature should we engineer?".

Read this plot for:
- Lines diverge from phase 0 onwards → there's an early-detectable signal
  in d/dt; search for a PIT-safe proxy.
- Lines only diverge at phase >= 1 → the cycle type is only knowable from
  d/dt once elongation is already observable from cycle_days_since_peak
  alone. The elongation feature was the right idea; the model just
  couldn't use it.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

OUT = pathlib.Path(__file__).parent
TRAIN_END_EXCL = pd.Timestamp("2021-11-01")
SHALLOW_SLOPE_THRESHOLD = -0.9  # matches #214's row-wise definition
MIN_BIN_COUNT = 5  # train segment is ~1800 dates split 3 ways; coarse but acceptable


def main() -> None:
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])

    # Per-date frame — cycle vars are network-wide so dedup is lossless for
    # cycle reconstruction. Recover sydney_avg from the existing baseline
    # difference column.
    per_date = (
        df.drop_duplicates("price_date")
        .loc[:, [
            "price_date", "station_price_cents", "station_minus_sydney_avg_cents",
            "cycle_days_since_peak", "cycle_last_max_cents",
            "cycle_pct_through", "network_px_std", "network_px_std_delta_3d",
        ]]
        .sort_values("price_date")
        .reset_index(drop=True)
        .copy()
    )
    per_date["sydney_avg"] = (
        per_date["station_price_cents"] - per_date["station_minus_sydney_avg_cents"]
    )
    per_date = per_date[per_date["price_date"] < TRAIN_END_EXCL].reset_index(drop=True)
    print(f"Train segment: {len(per_date):,} dates "
          f"({per_date['price_date'].min().date()} .. {per_date['price_date'].max().date()})",
          flush=True)

    # --- Cycle boundary detection ---
    # cycle_days_since_peak resets (drops) when a new peak is confirmed. Each
    # reset marks the start of a new cycle's days_since counter.
    per_date["dsp_prev"] = per_date["cycle_days_since_peak"].shift(1)
    per_date["is_cycle_start"] = (
        per_date["cycle_days_since_peak"] < per_date["dsp_prev"] - 1
    ).fillna(False)
    # First valid row also starts a cycle.
    per_date.loc[0, "is_cycle_start"] = True
    per_date["cycle_id"] = per_date["is_cycle_start"].cumsum()

    # --- Per-cycle summary ---
    cyc = (
        per_date.groupby("cycle_id")
        .agg(
            start_date=("price_date", "min"),
            end_date=("price_date", "max"),
            length_days=("price_date", "size"),
            peak_price=("cycle_last_max_cents", "first"),
            trough_price=("sydney_avg", "min"),
        )
        .reset_index()
    )
    # Trough day = the day of the trough relative to the cycle's start.
    trough_day = (
        per_date.merge(
            per_date.groupby("cycle_id")["sydney_avg"].min().reset_index()
            .rename(columns={"sydney_avg": "_min"}),
            on="cycle_id",
        )
        .loc[lambda d: d["sydney_avg"] == d["_min"]]
        .groupby("cycle_id")["price_date"].min()
        .rename("trough_date")
        .reset_index()
    )
    cyc = cyc.merge(trough_day, on="cycle_id")
    cyc["days_to_trough"] = (cyc["trough_date"] - cyc["start_date"]).dt.days
    cyc = cyc[cyc["days_to_trough"] > 0].copy()
    cyc["descent_slope"] = (
        (cyc["trough_price"] - cyc["peak_price"]) / cyc["days_to_trough"]
    )

    # Drop the in-progress trailing cycle if its length is small relative to
    # the last completed one (likely incomplete on the cutoff boundary).
    median_len = float(cyc["length_days"].median())
    last_row = cyc.iloc[-1]
    if last_row["length_days"] < 0.6 * median_len:
        cyc = cyc.iloc[:-1].copy()

    # Also drop the first cycle if it's a partial — the CycleDetector takes
    # a couple of cycles to confirm the first peak, so the earliest "cycle"
    # in the data is often a stub.
    if cyc.iloc[0]["length_days"] < 0.6 * median_len:
        cyc = cyc.iloc[1:].copy()

    print(f"Cycles found in train segment: {len(cyc)}  "
          f"(median length {cyc['length_days'].median():.0f}d, "
          f"IQR {cyc['length_days'].quantile(0.25):.0f}..{cyc['length_days'].quantile(0.75):.0f}d)",
          flush=True)
    print(f"  median descent_slope = {cyc['descent_slope'].median():.2f} c/day",
          flush=True)

    # --- Classify cycles ---
    elongation_threshold = float(cyc["length_days"].median())  # data-driven, train-only
    is_elongated = cyc["length_days"] > elongation_threshold
    is_shallow = cyc["descent_slope"] > SHALLOW_SLOPE_THRESHOLD

    cyc["cycle_type"] = "normal"
    cyc.loc[is_elongated & ~is_shallow, "cycle_type"] = "elongated_steep"
    cyc.loc[is_elongated & is_shallow, "cycle_type"] = "elongated_shallow"

    print(f"  elongation cutoff = > {elongation_threshold:.0f}d (train median)",
          flush=True)
    print(f"  shallow cutoff    = descent_slope > {SHALLOW_SLOPE_THRESHOLD} c/day",
          flush=True)
    for t in ("normal", "elongated_steep", "elongated_shallow"):
        sub = cyc[cyc["cycle_type"] == t]
        if len(sub):
            print(f"  {t}: {len(sub)} cycles  "
                  f"length {sub['length_days'].median():.0f}d (med)  "
                  f"slope {sub['descent_slope'].median():.2f} c/day (med)",
                  flush=True)

    # --- Tag each row with its parent cycle's type ---
    cyc_lookup = cyc.set_index("cycle_id")["cycle_type"]
    per_date["cycle_type"] = per_date["cycle_id"].map(cyc_lookup)
    per_date = per_date.dropna(subset=["cycle_type", "cycle_pct_through",
                                       "network_px_std", "network_px_std_delta_3d"])

    # --- Phase-binned aggregation ---
    # Coarser bins than the row-wise plot because train segment is smaller.
    bins = np.concatenate(
        [np.arange(0.0, 1.0, 0.05), np.arange(1.0, 2.01, 0.1)]
    )
    centres = (bins[:-1] + bins[1:]) / 2
    per_date["phase_bin"] = pd.cut(
        per_date["cycle_pct_through"], bins, labels=centres, include_lowest=True,
    )

    types = ("normal", "elongated_steep", "elongated_shallow")
    colours = {
        "normal": "#2980b9",
        "elongated_steep": "#16a085",
        "elongated_shallow": "#c0392b",
    }

    fig, (ax_level, ax_delta) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    for t in types:
        sub = per_date[per_date["cycle_type"] == t]
        if sub.empty:
            continue
        agg = (
            sub.groupby("phase_bin", observed=True)
            .agg(
                n=("network_px_std", "count"),
                level_med=("network_px_std", "median"),
                level_q25=("network_px_std", lambda s: float(np.quantile(s, 0.25))),
                level_q75=("network_px_std", lambda s: float(np.quantile(s, 0.75))),
                delta_med=("network_px_std_delta_3d", "median"),
                delta_q25=("network_px_std_delta_3d", lambda s: float(np.quantile(s, 0.25))),
                delta_q75=("network_px_std_delta_3d", lambda s: float(np.quantile(s, 0.75))),
            )
            .reset_index()
        )
        agg["phase"] = agg["phase_bin"].astype(float)
        agg = agg[agg["n"] >= MIN_BIN_COUNT]
        if agg.empty:
            continue
        c = colours[t]
        n_cycles = int(cyc[cyc["cycle_type"] == t].shape[0])
        ax_level.plot(agg["phase"], agg["level_med"], color=c, lw=2.5,
                      label=f"{t} ({n_cycles} cycles)", marker="o",
                      markersize=4, alpha=0.95)
        ax_delta.plot(agg["phase"], agg["delta_med"], color=c, lw=2.5,
                      label=t, marker="o", markersize=4, alpha=0.95)

        # Print numerical separation at key phases for the user.
        print(f"\n  {t}: median network_px_std + Δ3d at key phases:", flush=True)
        for phase_lo, phase_hi, label in [
            (0.10, 0.30, "early descent  (phase 0.1-0.3)"),
            (0.40, 0.55, "trough zone    (phase 0.4-0.55)"),
            (0.70, 0.95, "approach peak  (phase 0.7-0.95)"),
            (1.00, 1.50, "elongation tail (phase 1.0-1.5)"),
        ]:
            sl = agg[(agg["phase"] >= phase_lo) & (agg["phase"] < phase_hi)]
            if sl.empty:
                print(f"    {label}: (no data)", flush=True)
            else:
                lvl = float(np.median(sl["level_med"]))
                d3d = float(np.median(sl["delta_med"]))
                print(f"    {label}: level={lvl:5.2f}c  Δ3d={d3d:+5.2f}c",
                      flush=True)

    ax_level.axvspan(0.4, 0.55, color="grey", alpha=0.10, label="trough zone (typical)")
    ax_level.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax_level.set_ylabel("median network_px_std (cents)")
    ax_level.set_title(
        "ORACLE diagnostic — cycle pre-classified by full-cycle length + descent slope "
        f"(train segment, < {TRAIN_END_EXCL.date()})"
    )
    ax_level.grid(alpha=0.3)
    ax_level.legend(loc="upper right", fontsize=9)

    ax_delta.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax_delta.axvspan(0.4, 0.55, color="grey", alpha=0.10)
    ax_delta.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax_delta.set_xlabel("cycle_pct_through (days_since_peak / mean_cycle_length)")
    ax_delta.set_ylabel("median d(network_px_std)/dt (3d, cents)")
    ax_delta.grid(alpha=0.3)
    ax_delta.legend(loc="upper right", fontsize=9)

    out_path = OUT / "phase_oracle_cycles.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
