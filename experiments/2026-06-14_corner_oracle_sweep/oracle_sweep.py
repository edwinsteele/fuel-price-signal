"""Oracle-diagnostic sweep (#237): does ANY other intra-series signal carry
the shallow-vs-steep-within-elongated distinction that A (network_px_std) does
not?

Train-only oracle existence check (per feedback_oracle_diagnostic_pattern). NO
model fitting. Cycles are pre-classified by their EVENTUAL full-cycle shape
(length + descent slope — future info, oracle view) into normal /
elongated_steep / elongated_shallow, then each candidate signal's median
trajectory is plotted against cycle_pct_through, one line per class.

The read (per #237): a candidate is a LEAD only if it separates
``elongated_shallow`` from ``elongated_steep`` in the phase range where
prediction actually matters — the late-descent / trough zone, pct ~ 0.15-0.60 —
NOT only at pct > 1.4, where A finally separated (too late to be useful). The
oracle label never becomes a feature; a hit must subsequently earn a PIT-safe
proxy and pass paired-WFCV on untouched folds (separate follow-up).

Candidate families (issue #237):
  F1 trough-proximity (intra-series, on the Sydney-avg series)
     - days_since_meaningful_drop : days since the last >=0.3c daily fall
     - down_run_length            : length of the current consecutive-fall run
     - px_change_5d               : net 5-calendar-day change in Sydney-avg
  F2 cross-LGA / cross-brand late-descent consensus
     - lga_phase_std              : std of days_since_trough across 35 LGAs (= triplet C)
     - lga_trough_fraction        : fraction of 35 LGAs that recently entered trough
  F3 triplet B / C re-examined inside the corner
     - network_disc_gap           : comp_median - disc_median per date (triplet B)
     - network_disc_gap_delta_3d  : 3d delta of the gap

Run:
  PYTHONPATH=. uv run python experiments/2026-06-14_corner_oracle_sweep/oracle_sweep.py \
    2>&1 | tee experiments/2026-06-14_corner_oracle_sweep/run.log
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.lib.features.cycle_shape import label_cycle_shape
from experiments.lib.features.deltas import calendar_aware_delta
from experiments.lib.features.dispersion import cohort_agg_diff_by_date
from experiments.lib.io import write_meta
from experiments.lib.timing import time_block
from fuel_signal.features import load_features
from fuel_signal.lga_leadership import lga_feature_columns

OUT = pathlib.Path(__file__).parent

# Train-only cutoff: strictly < this excludes every fold's val window (fold 1
# val_start = 2021-11-05). Identical to #214's phase_oracle_cycles.py so the
# two oracle diagnostics see the same train segment.
TRAIN_END_EXCL = pd.Timestamp("2021-11-01")

# Candidate-signal parameters.
COMP_BAND_CENTS = 5.0   # competitive cohort: |stickiness_score| <= 5 (triplet B)
DISC_THRESH = -5.0      # discount cohort:    stickiness_score  < -5 (triplet B)
DROP_CENTS = 0.3        # a "meaningful" daily fall in Sydney-avg
TROUGH_RECENT_DAYS = 3  # an LGA "recently entered trough" if days_since_trough <= 3

MIN_BIN_COUNT = 5       # coarse bins — train segment is ~1800 dates / 3 classes

# Phase windows for the numerical read. The predictive zone (where the buy/sell
# objective carries its uncertainty) is the late-descent / trough band.
PREDICTIVE_ZONE = (0.15, 0.60)
LATE_TAIL_ZONE = (1.0, 1.5)

# label -> (lo, hi) for printed separation table.
PHASE_WINDOWS = {
    "early-mid descent (0.15-0.40)": (0.15, 0.40),
    "trough zone (0.40-0.60)": (0.40, 0.60),
    "approach peak (0.70-0.95)": (0.70, 0.95),
    "elongation tail (1.0-1.5)": (1.0, 1.5),
}

CANDIDATES = [
    "days_since_meaningful_drop",
    "down_run_length",
    "px_change_5d",
    "lga_phase_std",
    "lga_trough_fraction",
    "network_disc_gap",
    "network_disc_gap_delta_3d",
]
CLASSES = ("normal", "elongated_steep", "elongated_shallow")
COLOURS = {
    "normal": "#2980b9",
    "elongated_steep": "#16a085",
    "elongated_shallow": "#c0392b",
}


def compute_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all candidate signals and return a per-date frame.

    F3 (B) needs the full per-row frame (cohort medians across stations);
    everything else lives on the per-date Sydney-avg series. All signals are
    backward-looking (PIT-honest as eventual proxies) — only the oracle CLASS
    label uses future info, and that is applied downstream.
    """
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])

    # --- F3 / B: comp_median - disc_median per date (from full per-row frame) ---
    comp_mask = df["stickiness_score"].abs() <= COMP_BAND_CENTS
    disc_mask = df["stickiness_score"] < DISC_THRESH
    b = cohort_agg_diff_by_date(
        df, comp_mask, disc_mask, value_col="station_price_cents", agg="median",
    ).rename("network_disc_gap")
    b_delta = calendar_aware_delta(b, 3).rename("network_disc_gap_delta_3d")

    # --- Per-date frame (cycle vars are network-wide; dedup is lossless) ---
    lga_cols = lga_feature_columns()  # 35 LGA days_since_trough_entry_* columns
    per_date = (
        df.drop_duplicates("price_date")
        .loc[:, [
            "price_date", "station_price_cents", "station_minus_sydney_avg_cents",
            "cycle_days_since_peak", "cycle_pct_through", "cycle_last_max_cents",
            "lga_phase_std", *lga_cols,
        ]]
        .sort_values("price_date")
        .reset_index(drop=True)
        .copy()
    )
    per_date["sydney_avg"] = (
        per_date["station_price_cents"] - per_date["station_minus_sydney_avg_cents"]
    )

    # --- F2 / lga_trough_fraction: share of LGAs that recently entered trough ---
    lga = per_date[lga_cols]
    per_date["lga_trough_fraction"] = (
        (lga <= TROUGH_RECENT_DAYS).sum(axis=1) / lga.notna().sum(axis=1)
    )

    # --- F1: trough-proximity on a contiguous daily Sydney-avg series ---
    s = per_date.set_index("price_date")["sydney_avg"].sort_index()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    diff = s.diff()

    drop_day = (diff <= -DROP_CENTS).fillna(False)
    # Days since the most recent meaningful fall (0 on a fall day itself).
    days_since_drop = (~drop_day).groupby(drop_day.cumsum()).cumcount()
    days_since_drop = days_since_drop.rename("days_since_meaningful_drop")

    down_day = (diff < 0).fillna(False)
    # Length of the current consecutive-fall run (0 if today is not a fall).
    run = down_day.groupby((~down_day).cumsum()).cumcount() + 1
    down_run = (run * down_day).rename("down_run_length")

    px_change_5d = calendar_aware_delta(s, 5).rename("px_change_5d")

    f1 = pd.concat([days_since_drop, down_run, px_change_5d], axis=1)
    f1.index.name = "price_date"

    # --- Join everything onto per_date ---
    per_date = (
        per_date.join(f1, on="price_date")
        .join(b, on="price_date")
        .join(b_delta, on="price_date")
    )
    return per_date


def main() -> None:
    with time_block("load_features"):
        df = load_features()
    print(f"  rows={len(df):,}", flush=True)

    with time_block("compute_candidates"):
        per_date = compute_candidates(df)

    per_date = per_date[per_date["price_date"] < TRAIN_END_EXCL].reset_index(drop=True)
    print(
        f"Train segment: {len(per_date):,} dates "
        f"({per_date['price_date'].min().date()} .. {per_date['price_date'].max().date()})",
        flush=True,
    )

    with time_block("label_cycle_shape"):
        per_date, cyc = label_cycle_shape(per_date)

    print(
        f"Cycles: {len(cyc)} (median length {cyc['length_days'].median():.0f}d, "
        f"median descent_slope {cyc['descent_slope'].median():.2f} c/day)",
        flush=True,
    )
    for t in CLASSES:
        sub = cyc[cyc["cycle_type"] == t]
        if len(sub):
            print(
                f"  {t}: {len(sub)} cycles  length {sub['length_days'].median():.0f}d (med)  "
                f"slope {sub['descent_slope'].median():.2f} c/day (med)",
                flush=True,
            )

    per_date = per_date.dropna(subset=["cycle_type", "cycle_pct_through"])

    # --- Phase binning ---
    bins = np.concatenate([np.arange(0.0, 1.0, 0.05), np.arange(1.0, 2.01, 0.1)])
    centres = (bins[:-1] + bins[1:]) / 2
    per_date["phase_bin"] = pd.cut(
        per_date["cycle_pct_through"], bins, labels=centres, include_lowest=True,
    )

    # --- Per-candidate aggregation, plot, and numerical read ---
    n = len(CANDIDATES)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    summary: dict[str, dict] = {}
    for ax, cand in zip(axes, CANDIDATES, strict=False):
        # Per (class, phase_bin) median of the candidate.
        class_curves: dict[str, pd.DataFrame] = {}
        for t in CLASSES:
            sub = per_date[per_date["cycle_type"] == t].dropna(subset=[cand])
            if sub.empty:
                continue
            agg = (
                sub.groupby("phase_bin", observed=True)[cand]
                .agg(["median", "count"])
                .reset_index()
            )
            agg["phase"] = agg["phase_bin"].astype(float)
            agg = agg[agg["count"] >= MIN_BIN_COUNT]
            if agg.empty:
                continue
            class_curves[t] = agg
            ax.plot(
                agg["phase"], agg["median"], color=COLOURS[t], lw=2.2,
                marker="o", markersize=3, alpha=0.95, label=t,
            )

        ax.axvspan(*PREDICTIVE_ZONE, color="orange", alpha=0.08)
        ax.axvline(1.0, color="grey", ls="--", lw=0.7, alpha=0.6)
        ax.set_title(cand, fontsize=10)
        ax.grid(alpha=0.3)

        # Numerical read: shallow vs steep separation, predictive zone vs late tail.
        def _zone_med(t: str, lo: float, hi: float) -> float:
            c = class_curves.get(t)
            if c is None:
                return float("nan")
            sl = c[(c["phase"] >= lo) & (c["phase"] < hi)]
            return float(np.nanmedian(sl["median"])) if not sl.empty else float("nan")

        sep_pred = abs(
            _zone_med("elongated_shallow", *PREDICTIVE_ZONE)
            - _zone_med("elongated_steep", *PREDICTIVE_ZONE)
        )
        sep_late = abs(
            _zone_med("elongated_shallow", *LATE_TAIL_ZONE)
            - _zone_med("elongated_steep", *LATE_TAIL_ZONE)
        )
        summary[cand] = {
            "shallow_vs_steep_sep_predictive_zone": sep_pred,
            "shallow_vs_steep_sep_late_tail": sep_late,
            "windows": {
                label: {t: _zone_med(t, lo, hi) for t in CLASSES}
                for label, (lo, hi) in PHASE_WINDOWS.items()
            },
        }
        print(f"\n=== {cand} ===", flush=True)
        print(
            f"  shallow-vs-steep |Δmedian|: predictive zone {PREDICTIVE_ZONE} = "
            f"{sep_pred:.3f}   late tail {LATE_TAIL_ZONE} = {sep_late:.3f}",
            flush=True,
        )
        for label, (lo, hi) in PHASE_WINDOWS.items():
            vals = "  ".join(
                f"{t.split('_')[-1]}={_zone_med(t, lo, hi):.2f}" for t in CLASSES
            )
            print(f"    {label}: {vals}", flush=True)

    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(loc="best", fontsize=8)
    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel("cycle_pct_through")
    fig.suptitle(
        "ORACLE corner sweep (#237) — candidate median vs phase by eventual cycle shape "
        f"(train segment < {TRAIN_END_EXCL.date()})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png = OUT / "oracle_sweep.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nWrote {out_png}", flush=True)

    # --- Ranked verdict aid ---
    ranked = sorted(
        summary.items(),
        key=lambda kv: kv[1]["shallow_vs_steep_sep_predictive_zone"],
        reverse=True,
    )
    print("\n=== Candidates ranked by predictive-zone shallow/steep separation ===", flush=True)
    for cand, m in ranked:
        print(
            f"  {cand:<30} pred={m['shallow_vs_steep_sep_predictive_zone']:.3f}  "
            f"late={m['shallow_vs_steep_sep_late_tail']:.3f}",
            flush=True,
        )

    write_meta(OUT, {
        "issue": 237,
        "kind": "train-only oracle existence sweep (no model fitting)",
        "train_end_excl": str(TRAIN_END_EXCL.date()),
        "params": {
            "COMP_BAND_CENTS": COMP_BAND_CENTS,
            "DISC_THRESH": DISC_THRESH,
            "DROP_CENTS": DROP_CENTS,
            "TROUGH_RECENT_DAYS": TROUGH_RECENT_DAYS,
            "MIN_BIN_COUNT": MIN_BIN_COUNT,
            "predictive_zone": PREDICTIVE_ZONE,
            "late_tail_zone": LATE_TAIL_ZONE,
        },
        "n_cycles": {t: int((cyc["cycle_type"] == t).sum()) for t in CLASSES},
        "candidates": CANDIDATES,
        "separation": summary,
        "note": (
            "shallow_vs_steep_sep_* are |Δmedian| between elongated_shallow and "
            "elongated_steep within the phase zone. A LEAD requires meaningful "
            "predictive-zone separation; separation only in the late tail repeats "
            "A's failure mode. Oracle hit is a lead only — PIT-safe proxy + "
            "paired-WFCV on untouched folds is the non-negotiable real test."
        ),
    })


if __name__ == "__main__":
    main()
