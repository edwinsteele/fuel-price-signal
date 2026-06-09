"""Diagnostic: does A's level + rate look different across descent regimes?

Splits rows into three descent populations and plots median network_px_std
(level) and median network_px_std_delta_3d (rate) vs cycle_pct_through for
each. Same x-axis as ``phase_diagnostic.png``; the new axis is the regime
overlay.

Bucket definitions (matching #214 + step5):

- normal_descent       : elongation_ratio <= 1.0  AND slope < 0
- ext_descent_steep    : elongation_ratio >  1.0  AND slope <= -0.9
- ext_descent_shallow  : elongation_ratio >  1.0  AND -0.9 < slope < 0

Bucket labels reuse the in-script computed elongation_ratio + slope
(frozen 730d baseline, closed='left'), exactly as the paired_wfcv harness
saw them.

Read this plot for:
- Coincident lines → A is information-limited in the failure regime
  (replace A; pivot toward #215 external data).
- Diverging lines → A carries discriminating info; failure is in how the
  model combines it with context (wrap A: gating, regime-conditional
  normalisation, or an explicit interaction term).
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

OUT = pathlib.Path(__file__).parent
CML_BASELINE_WINDOW_DAYS = 730
SHALLOW_SLOPE_THRESHOLD = -0.9
EXT_DESCENT_ELONGATION_THRESHOLD = 1.0
MIN_BIN_COUNT = 100


def main() -> None:
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    print(f"Loaded {len(df):,} rows", flush=True)

    # --- Compute elongation_ratio + slope (mirrors paired_wfcv) ---
    cml_by_date = (
        df.dropna(subset=["cycle_mean_length"])
        .drop_duplicates("price_date")
        .set_index("price_date")["cycle_mean_length"]
        .sort_index()
    )
    full_idx = pd.date_range(cml_by_date.index.min(), cml_by_date.index.max(), freq="D")
    cml_daily = cml_by_date.reindex(full_idx)
    baseline = (
        cml_daily.rolling(f"{CML_BASELINE_WINDOW_DAYS}D", closed="left", min_periods=1)
        .median()
        .rename("station_baseline_cml")
    )
    df = df.join(baseline, on="price_date")

    safe = df["station_baseline_cml"] > 0
    df["elongation_ratio"] = np.where(
        safe, df["cycle_days_since_peak"] / df["station_baseline_cml"], np.nan,
    )
    nonzero = df["cycle_days_since_peak"] > 0
    df["cycle_descent_slope_so_far"] = np.where(
        nonzero,
        (df["station_price_cents"] - df["cycle_last_max_cents"]) / df["cycle_days_since_peak"],
        np.nan,
    )

    # --- Bucket assignment ---
    is_ext = df["elongation_ratio"] > EXT_DESCENT_ELONGATION_THRESHOLD
    is_descent = df["cycle_descent_slope_so_far"] < 0  # still descending
    is_shallow = df["cycle_descent_slope_so_far"] > SHALLOW_SLOPE_THRESHOLD

    df["bucket"] = "other"
    df.loc[~is_ext & is_descent, "bucket"] = "normal_descent"
    df.loc[is_ext & is_descent & is_shallow, "bucket"] = "ext_descent_shallow"
    df.loc[is_ext & is_descent & ~is_shallow, "bucket"] = "ext_descent_steep"

    for b in ("normal_descent", "ext_descent_steep", "ext_descent_shallow", "other"):
        n = int((df["bucket"] == b).sum())
        print(f"  {b}: {n:,} rows ({n / len(df) * 100:.1f}%)", flush=True)

    # --- Per-(bucket, phase_bin) aggregation ---
    bins = np.concatenate(
        [np.arange(0.0, 1.0, 0.025), np.arange(1.0, 2.01, 0.05)]
    )
    centres = (bins[:-1] + bins[1:]) / 2
    df["phase_bin"] = pd.cut(
        df["cycle_pct_through"], bins, labels=centres, include_lowest=True,
    )

    buckets = ("normal_descent", "ext_descent_steep", "ext_descent_shallow")
    colours = {
        "normal_descent": "#2980b9",     # blue — A's good bucket
        "ext_descent_steep": "#16a085",  # green — elongated but steep
        "ext_descent_shallow": "#c0392b", # red — the failure regime
    }

    fig, (ax_level, ax_delta) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    for b in buckets:
        sub = df[(df["bucket"] == b) & df["network_px_std"].notna()]
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
        c = colours[b]
        ax_level.fill_between(agg["phase"], agg["level_q25"], agg["level_q75"],
                              color=c, alpha=0.15)
        ax_level.plot(agg["phase"], agg["level_med"], color=c, lw=2,
                      label=f"{b} (n={len(sub):,})")
        ax_delta.fill_between(agg["phase"], agg["delta_q25"], agg["delta_q75"],
                              color=c, alpha=0.15)
        ax_delta.plot(agg["phase"], agg["delta_med"], color=c, lw=2, label=b)

    ax_level.axvspan(0.4, 0.55, color="grey", alpha=0.10, label="trough zone (typical)")
    ax_level.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax_level.set_ylabel("median network_px_std (cents)")
    ax_level.set_title("A's level + rate-of-change by descent regime "
                       "(shading = IQR within bucket × phase bin)")
    ax_level.grid(alpha=0.3)
    ax_level.legend(loc="upper right", fontsize=9)

    ax_delta.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax_delta.axvspan(0.4, 0.55, color="grey", alpha=0.10)
    ax_delta.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax_delta.set_xlabel("cycle_pct_through (days_since_peak / mean_cycle_length)")
    ax_delta.set_ylabel("median d(network_px_std)/dt (3d, cents)")
    ax_delta.grid(alpha=0.3)
    ax_delta.legend(loc="upper right", fontsize=9)

    out_path = OUT / "phase_by_regime.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
