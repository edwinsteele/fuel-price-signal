"""
Fold-level (elongation, gradient) 2D diagnostic for the late-descent triplet.

Question (user, 2026-06-07): the single-axis elongation hypothesis didn't
generalise across 14 folds. Is the regression on folds {7, 9, 13} explainable
by an elongation x descent-gradient interaction? Two operational uses:
  - a constraint on A/C that suppresses the regression on extended-descent
    rows;
  - a broad-population signal where {7, 9, 13} are just the loudest
    exemplars.

Step 1 (this script): cheap fold-level diagnostic. Place each of 14 folds
on the 2D (elongation, gradient) plane, colour by per-fold delta_hard25
(median across seeds) for R1_ABC, R5_A_only, R6_B_only, R7_C_only.

Definitions
-----------
elongation_score (per fold):
    For each val row, ratio of `cycle_days_since_peak` to a frozen baseline
    of `cycle_mean_length` taken as the median over the 730d window ending
    at (val_start - 1), per station. Fold score = median of per-row ratios
    over val rows. Frozen baseline avoids the adaptivity issue
    (project_late_descent_elongation_regime, attempt 1).

gradient_score (per fold):
    For each station-date in val window, 14d backward slope of
    station_price_cents in cents/day. Keep only rows where the slope is
    negative (descending arm). Fold score = median of those negative slopes.
    More negative = steeper descent; closer to zero = shallower.

per-fold delta (per run, per cohort):
    For each (fold, run) cell, median across the 5 seeds, then subtract
    median-across-seeds baseline. (Per feedback_check_seed_variance_before_
    trusting_mean — mean is contaminated by the fold-2 s44 outlier.)

Output
------
- step4_fold_scores.csv: (fold, elongation, gradient, delta_hard25_R1,
  delta_hard25_R5, delta_hard25_R6, delta_hard25_R7)
- step4_2d_panels.png: 2x2 panels, one per run, dot per fold.
"""

from __future__ import annotations

import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow `python experiments/.../step4_...py` invocation.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuel_signal.features import load_features  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
RUNS_CSV = OUT / "step2_runs.csv"
TARGET_RUNS = ("R1_ABC", "R5_A_only", "R6_B_only", "R7_C_only")
BASELINE_RUN = "R0_baseline"
BACKWARD_LAG_DAYS = 14
BASELINE_WINDOW_DAYS = 730


def per_fold_deltas(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Median across seeds per (fold, run), then delta vs baseline per fold."""
    med = (
        runs_df.groupby(["fold", "run"], as_index=False)["ll_hard25"]
        .median()
        .rename(columns={"ll_hard25": "ll_hard25_median"})
    )
    base = med[med["run"] == BASELINE_RUN][["fold", "ll_hard25_median"]].rename(
        columns={"ll_hard25_median": "ll_hard25_base"}
    )
    med = med.merge(base, on="fold")
    med["delta_hard25"] = med["ll_hard25_median"] - med["ll_hard25_base"]
    return med[med["run"].isin(TARGET_RUNS)].copy()


def fold_windows(runs_df: pd.DataFrame) -> pd.DataFrame:
    return (
        runs_df[["fold", "val_start", "val_end"]]
        .drop_duplicates()
        .assign(
            val_start=lambda d: pd.to_datetime(d["val_start"]),
            val_end=lambda d: pd.to_datetime(d["val_end"]),
        )
        .sort_values("fold")
        .reset_index(drop=True)
    )


def elongation_score(features: pd.DataFrame, val_start: pd.Timestamp,
                     val_end: pd.Timestamp) -> float:
    """Median per-row (days_since_peak / frozen baseline mean cycle length).

    Frozen baseline: median `cycle_mean_length` over the 730d window ending
    at (val_start - 1) per station.
    """
    base_lo = val_start - pd.Timedelta(days=BASELINE_WINDOW_DAYS)
    base_hi = val_start - pd.Timedelta(days=1)

    base_mask = (features["price_date"] >= base_lo) & (features["price_date"] <= base_hi)
    base = (
        features.loc[base_mask, ["station_code", "cycle_mean_length"]]
        .dropna()
        .groupby("station_code", as_index=False)["cycle_mean_length"]
        .median()
        .rename(columns={"cycle_mean_length": "cml_base"})
    )

    val_mask = (features["price_date"] >= val_start) & (features["price_date"] <= val_end)
    val = features.loc[val_mask, ["station_code", "cycle_days_since_peak"]].dropna()
    val = val.merge(base, on="station_code", how="inner")
    val = val[val["cml_base"] > 0]
    if val.empty:
        return float("nan")
    val["elong"] = val["cycle_days_since_peak"] / val["cml_base"]
    return float(val["elong"].median())


def gradient_score(price_panel: pd.DataFrame, val_start: pd.Timestamp,
                   val_end: pd.Timestamp) -> float:
    """Median 14d backward slope (cents/day) over descending station-date rows.

    `price_panel` is sorted by (station_code, price_date) with a precomputed
    `price_lag14` column.
    """
    val_mask = (price_panel["price_date"] >= val_start) & (price_panel["price_date"] <= val_end)
    val = price_panel.loc[val_mask, ["station_price_cents", "price_lag14"]].dropna()
    slope = (val["station_price_cents"] - val["price_lag14"]) / BACKWARD_LAG_DAYS
    descending = slope[slope < 0]
    if descending.empty:
        return float("nan")
    return float(descending.median())


def descent_fraction(features: pd.DataFrame, val_start: pd.Timestamp,
                     val_end: pd.Timestamp) -> float:
    """Fraction of val rows on the descending arm (cycle_pct_through < 0.5).

    Per project_cycle_pct_through_semantics: peak-anchored, non-monotonic,
    empirical shape peak -> trough (~0.5) -> peak. So pct < 0.5 = descending.
    """
    val_mask = (features["price_date"] >= val_start) & (features["price_date"] <= val_end)
    val = features.loc[val_mask, ["cycle_pct_through"]].dropna()
    if val.empty:
        return float("nan")
    return float((val["cycle_pct_through"] < 0.5).mean())


def cycle_descent_slope(features: pd.DataFrame, val_start: pd.Timestamp,
                        val_end: pd.Timestamp) -> float:
    """Median cycle-anchored descent slope over descending val rows.

    Slope = (station_price_cents - cycle_last_max_cents) / cycle_days_since_peak.
    Negative for genuine descent (price below last peak). Restricted to
    cycle_pct_through < 0.5 rows so ascent rows don't dilute. Median across
    rows. More negative = steeper descent during the current cycle so far.
    """
    val_mask = (features["price_date"] >= val_start) & (features["price_date"] <= val_end)
    cols = ["station_price_cents", "cycle_last_max_cents", "cycle_days_since_peak",
            "cycle_pct_through"]
    val = features.loc[val_mask, cols].dropna()
    val = val[(val["cycle_pct_through"] < 0.5) & (val["cycle_days_since_peak"] > 0)]
    if val.empty:
        return float("nan")
    slope = (val["station_price_cents"] - val["cycle_last_max_cents"]) / val["cycle_days_since_peak"]
    return float(slope.median())


def build_price_panel(features: pd.DataFrame) -> pd.DataFrame:
    panel = features[["station_code", "price_date", "station_price_cents"]].sort_values(
        ["station_code", "price_date"]
    )
    panel["price_lag14"] = panel.groupby("station_code")["station_price_cents"].shift(BACKWARD_LAG_DAYS)
    return panel


def main() -> None:
    t0 = time.perf_counter()
    print("Loading features...", flush=True)
    features = load_features()
    features["price_date"] = pd.to_datetime(features["price_date"])
    print(f"  loaded {len(features):,} rows ({time.perf_counter()-t0:.1f}s)", flush=True)

    runs_df = pd.read_csv(RUNS_CSV)
    folds = fold_windows(runs_df)
    deltas = per_fold_deltas(runs_df)
    delta_wide = deltas.pivot(index="fold", columns="run", values="delta_hard25")

    t1 = time.perf_counter()
    print("Building price panel for gradient...", flush=True)
    panel = build_price_panel(features)
    print(f"  built in {time.perf_counter()-t1:.1f}s", flush=True)

    rows = []
    t2 = time.perf_counter()
    for _, fr in folds.iterrows():
        f = int(fr["fold"])
        vs, ve = fr["val_start"], fr["val_end"]
        elong = elongation_score(features, vs, ve)
        grad = gradient_score(panel, vs, ve)
        dfrac = descent_fraction(features, vs, ve)
        cslope = cycle_descent_slope(features, vs, ve)
        row = {"fold": f, "val_start": vs.date(), "val_end": ve.date(),
               "elongation": elong, "gradient": grad,
               "descent_frac": dfrac, "cycle_descent_slope": cslope}
        for run in TARGET_RUNS:
            row[f"delta_hard25_{run}"] = float(delta_wide.loc[f, run]) if f in delta_wide.index else float("nan")
        rows.append(row)
        print(f"  fold {f:>2}  elong={elong:.3f}  grad={grad:.3f}  "
              f"dfrac={dfrac:.2f}  cyc_slope={cslope:.3f}  "
              f"d_R1={row['delta_hard25_R1_ABC']:+.3f}  "
              f"d_R5={row['delta_hard25_R5_A_only']:+.3f}",
              flush=True)
    print(f"Per-fold scores computed in {time.perf_counter()-t2:.1f}s", flush=True)

    out_df = pd.DataFrame(rows)
    out_csv = OUT / "step4_fold_scores.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Sanity check: report Pearson r per run against handover values
    # (R1 -0.21, R5_A -0.25, R6_B -0.37, R7_C +0.16 on hard25).
    print("\nPearson r(elongation, delta_hard25) [all 14 folds]:")
    for run in TARGET_RUNS:
        r = out_df[["elongation", f"delta_hard25_{run}"]].dropna().corr().iloc[0, 1]
        print(f"  {run:<12}  r={r:+.3f}")
    print("(Handover quoted: R1 -0.21, R5 -0.25, R6 -0.37, R7 +0.16 on hard25.)")

    # User insight (2026-06-07): fold 1 is in a generally price-INCREASING
    # regime, so its "A helps" verdict averages over mostly-ascending rows.
    # Recompute correlations on descent-dominated folds only (descent_frac
    # >= 0.5). cycle_descent_slope is the cleaner gradient measure on this
    # subset.
    descent_mask = out_df["descent_frac"] >= 0.5
    descent_only = out_df[descent_mask].copy()
    print(f"\nDescent-dominated folds (descent_frac >= 0.5): "
          f"{sorted(descent_only['fold'].tolist())}  "
          f"({len(descent_only)}/{len(out_df)})")
    print("Pearson r(cycle_descent_slope, delta_hard25) [descent-dominated subset]:")
    for run in TARGET_RUNS:
        sub = descent_only[["cycle_descent_slope", f"delta_hard25_{run}"]].dropna()
        if len(sub) < 3:
            print(f"  {run:<12}  n={len(sub)} (too few)")
            continue
        r = sub.corr().iloc[0, 1]
        print(f"  {run:<12}  r={r:+.3f}  n={len(sub)}")
    print("Pearson r(elongation, delta_hard25) [descent-dominated subset]:")
    for run in TARGET_RUNS:
        sub = descent_only[["elongation", f"delta_hard25_{run}"]].dropna()
        if len(sub) < 3:
            print(f"  {run:<12}  n={len(sub)} (too few)")
            continue
        r = sub.corr().iloc[0, 1]
        print(f"  {run:<12}  r={r:+.3f}  n={len(sub)}")

    # 2x2 panels. y-axis = cycle_descent_slope (descent-anchored slope), per
    # the user's clarification that fold 1 is ascent-dominated and shouldn't
    # be measured the same way. Marker SHAPE encodes descent-domination:
    # filled circle = descent-dominated (descent_frac >= 0.5), open square
    # = ascent-dominated.
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    vmax = float(np.nanmax(np.abs(out_df[[f"delta_hard25_{r}" for r in TARGET_RUNS]].values)))
    for ax, run in zip(axes.flat, TARGET_RUNS):
        d = out_df[f"delta_hard25_{run}"]
        # Descent-dominated folds (filled circle)
        d_mask = out_df["descent_frac"] >= 0.5
        if d_mask.any():
            sc = ax.scatter(out_df.loc[d_mask, "elongation"],
                            out_df.loc[d_mask, "cycle_descent_slope"],
                            c=d[d_mask], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            s=170, edgecolor="black", linewidth=0.9,
                            marker="o", label="descent-dominated")
        # Ascent-dominated folds (open square — outline only)
        if (~d_mask).any():
            sc = ax.scatter(out_df.loc[~d_mask, "elongation"],
                            out_df.loc[~d_mask, "cycle_descent_slope"],
                            c=d[~d_mask], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            s=170, edgecolor="black", linewidth=0.9,
                            marker="s", label="ascent-dominated")
        for _, r in out_df.iterrows():
            ax.annotate(f"f{int(r['fold'])}",
                        (r["elongation"], r["cycle_descent_slope"]),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=9)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_title(f"{run}  delta_hard25 (median across seeds)")
        ax.set_xlabel("Per-fold elongation score")
        ax.set_ylabel("Per-fold cycle-anchored descent slope (cents/day)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
        fig.colorbar(sc, ax=ax, label="delta (positive = regression)")
    fig.suptitle(
        "Fold-level 2D — x=elongation, y=cycle descent slope, "
        "shape=regime; red = regression, blue = helps",
        y=0.995, fontsize=12,
    )
    fig.tight_layout()
    out_png = OUT / "step4_2d_panels.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"Wrote {out_png}")

    print(f"\nTotal wall: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
