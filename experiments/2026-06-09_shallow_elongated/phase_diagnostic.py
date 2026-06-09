"""Diagnostic: d(network_px_std)/dt vs cycle phase.

Tests the trough-detection-via-rate-of-change hypothesis: does the existing
``network_px_std_delta_3d`` have a clean signature near the trough that's
visible in the aggregate phase shape?

Both network_px_std and cycle_pct_through are network-wide per-date
quantities (Sydney-avg-derived), so we de-dup to one row per date before
binning. Phase semantics (per project_cycle_pct_through_semantics):

  pct = 0    → peak (start of cycle)
  pct ≈ 0.5  → trough
  pct ≈ 1.0  → next peak (typical)
  pct > 1.0  → elongation territory (current cycle longer than mean)

If rate-of-change carries a clean trough signature regardless of elongation,
we'd expect d/dt to dip (compression — laggards catching down) somewhere
just before pct ≈ 0.5, and the same signature to recur — possibly stretched
— in the pct > 1 elongation band before the late trough.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

OUT = pathlib.Path(__file__).parent


def main() -> None:
    df = load_features()
    print(f"Loaded {len(df):,} rows", flush=True)

    # network_px_std + cycle_pct_through are network-wide per-date — de-dup
    # to one row per date so each date contributes equally rather than being
    # weighted by station count.
    per_date = (
        df.drop_duplicates("price_date")
        .loc[:, ["price_date", "cycle_pct_through", "network_px_std",
                 "network_px_std_delta_3d"]]
        .dropna()
        .copy()
    )
    print(f"  per-date rows after dedup + dropna: {len(per_date):,}", flush=True)
    print(f"  phase range: {per_date['cycle_pct_through'].min():.2f} "
          f".. {per_date['cycle_pct_through'].max():.2f}", flush=True)

    # Bin by phase. Tight bins through the normal cycle (0..1), coarser in
    # the elongation tail (1..2) where per-bin sample sizes thin.
    bins = np.concatenate([np.arange(0.0, 1.0, 0.025), np.arange(1.0, 2.01, 0.05)])
    centres = (bins[:-1] + bins[1:]) / 2
    per_date["phase_bin"] = pd.cut(per_date["cycle_pct_through"], bins,
                                   labels=centres, include_lowest=True)
    agg = (
        per_date.groupby("phase_bin", observed=True)
        .agg(
            n=("network_px_std_delta_3d", "count"),
            level_med=("network_px_std", "median"),
            delta_med=("network_px_std_delta_3d", "median"),
            delta_q25=("network_px_std_delta_3d", lambda s: float(np.quantile(s, 0.25))),
            delta_q75=("network_px_std_delta_3d", lambda s: float(np.quantile(s, 0.75))),
        )
        .reset_index()
    )
    agg["phase"] = agg["phase_bin"].astype(float)
    agg = agg[agg["n"] >= 10]  # drop sparse tail bins

    print(f"  bins with n>=10: {len(agg)}  (n range {agg['n'].min()}..{agg['n'].max()})",
          flush=True)

    # --- Plot ---
    fig, (ax_level, ax_delta) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4]},
    )

    # Top: level (median network_px_std by phase)
    ax_level.plot(agg["phase"], agg["level_med"], color="C0", lw=2,
                  label="median network_px_std")
    ax_level.axvspan(0.4, 0.55, color="C2", alpha=0.10, label="trough zone (typical)")
    ax_level.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax_level.set_ylabel("network_px_std (cents)")
    ax_level.set_title("Dispersion level + rate-of-change vs cycle phase "
                       "(de-dup to one row per date)")
    ax_level.grid(alpha=0.3)
    ax_level.legend(loc="upper right", fontsize=9)

    # Bottom: delta (median + IQR shading)
    ax_delta.fill_between(agg["phase"], agg["delta_q25"], agg["delta_q75"],
                          color="C1", alpha=0.20, label="IQR (25-75th)")
    ax_delta.plot(agg["phase"], agg["delta_med"], color="C1", lw=2,
                  label="median d(network_px_std)/dt (3d)")
    ax_delta.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax_delta.axvspan(0.4, 0.55, color="C2", alpha=0.10,
                     label="trough zone (typical)")
    ax_delta.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6,
                     label="phase = 1 (typical next peak)")
    ax_delta.set_xlabel("cycle_pct_through (days_since_peak / mean_cycle_length)")
    ax_delta.set_ylabel("Δ over 3 calendar days (cents)")
    ax_delta.grid(alpha=0.3)
    ax_delta.legend(loc="upper right", fontsize=9)

    out_path = OUT / "phase_diagnostic.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
