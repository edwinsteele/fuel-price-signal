"""Step 5 — row-level diagnostic for the extended-descent hypothesis.

Test: within each fold, does A's (network_px_std + delta_3d) per-row error
concentrate on extended-descent rows? If yes across many folds, the
regression on fold 7 is the loudest exemplar of a population-wide failure
mode, and a constraint feature (e.g. is_extended_descent x A interaction)
has a fighting chance. If only fold 7's extended-descent rows light up,
it's a localised pattern, constraint design is low-value.

Inputs:
- step5_rowpreds.parquet (per (fold, run, seed, row) predicted_proba)
- step5_fold_meta.csv  (per-(fold, run, seed) summary; for val_start lookup)
- data/features.csv via load_features() for per-row covariates

Row-level definitions:
- delta_logloss = median_R5_logloss - median_R0_logloss (median across 5 seeds)
- descent_arm = (5d backward price change < 0)
  Price-based, NOT pct-based, because cycle_pct_through is peak-anchored and
  rows in an elongated cycle can have pct >= 1.0 even while still descending
  toward a delayed trough. Using pct would silently exclude the exact rows
  the hypothesis targets.
- is_extended_descent = descent_arm AND
  (cycle_days_since_peak / station_frozen_baseline > 1.3)
  where station_frozen_baseline = median cycle_mean_length over the 730d
  window ending at (val_start - 1) for that station.

Outputs:
- step5_rowdelta.parquet   (per-row delta_logloss + cohort flags)
- step5_rowlevel_summary.csv  (stratified means by fold x cohort)
- step5_rowlevel.png   (visualisations)
- console summary
"""
from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

OUT = pathlib.Path(__file__).parent
EPS = 1e-15
ELONGATION_TAU = 1.3       # multiplier on baseline cycle length
BASELINE_WINDOW_DAYS = 730


def per_row_log_loss(label: np.ndarray, proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, EPS, 1 - EPS)
    return -(label * np.log(p) + (1 - label) * np.log(1 - p))


def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading rowpreds + features ...")
    rp = pd.read_parquet(OUT / "step5_rowpreds.parquet")
    meta = pd.read_csv(OUT / "step5_fold_meta.csv")
    feats = load_features()
    feats["price_date"] = pd.to_datetime(feats["price_date"])
    print(f"  rp: {len(rp):,} rows  features: {len(feats):,} rows  "
          f"({time.perf_counter()-overall_t0:.1f}s)")

    # Per-(fold, run, station, date): median proba across seeds.
    rp["price_date"] = pd.to_datetime(rp["price_date"])
    rp["ll"] = per_row_log_loss(rp["label"].to_numpy(), rp["proba"].to_numpy())
    agg = (
        rp.groupby(["fold", "run", "station_code", "price_date"], as_index=False)
        .agg(label=("label", "first"),
             is_hard25=("is_hard25", "first"),
             is_hard10=("is_hard10", "first"),
             ll_median=("ll", "median"))
    )

    # Pivot to wide on run: R0 vs R5 per row.
    wide = agg.pivot_table(
        index=["fold", "station_code", "price_date", "label", "is_hard25", "is_hard10"],
        columns="run", values="ll_median",
    ).reset_index()
    wide["delta_ll"] = wide["R5_A_only"] - wide["R0_baseline"]
    print(f"Per-row delta_ll rows: {len(wide):,}")

    # ---- Per-row covariates for stratification ----
    # Add 5d backward price change per (station_code, price_date) — used as
    # the descent indicator. Calendar-aware: look up same station 5 days
    # earlier; NaN if no observation at that exact date (e.g. across a
    # fill-pipeline gap > max_gap_days). Mirrors step2's _px_5d_change.
    print("Computing 5d backward price change ...")
    t_px = time.perf_counter()
    lookup = feats[["station_code", "price_date", "station_price_cents"]].rename(
        columns={"price_date": "_lookup_date", "station_price_cents": "_px_5d_ago"}
    )
    feats_with_px5 = feats.copy()
    feats_with_px5["_lookup_date"] = feats_with_px5["price_date"] - pd.Timedelta(days=5)
    feats_with_px5 = feats_with_px5.merge(
        lookup, on=["station_code", "_lookup_date"], how="left", validate="m:1"
    )
    feats_with_px5["px_change_5d"] = (
        feats_with_px5["station_price_cents"] - feats_with_px5["_px_5d_ago"]
    )
    print(f"  px_change_5d nulls: "
          f"{feats_with_px5['px_change_5d'].isna().mean()*100:.1f}%  "
          f"({time.perf_counter()-t_px:.1f}s)")

    covars = feats_with_px5[["station_code", "price_date",
                              "cycle_pct_through", "cycle_days_since_peak",
                              "cycle_mean_length", "px_change_5d"]]
    wide = wide.merge(covars, on=["station_code", "price_date"], how="left",
                      validate="m:1")

    # ---- Frozen baseline cycle_mean_length per (fold, station) ----
    # Window: [val_start - 730, val_start - 1] per fold.
    fold_starts = (meta[["fold", "val_start"]].drop_duplicates()
                   .assign(val_start=lambda d: pd.to_datetime(d["val_start"])))
    base_rows = []
    print("Computing frozen baselines per (fold, station) ...")
    t0 = time.perf_counter()
    for _, fr in fold_starts.iterrows():
        f, vs = int(fr["fold"]), fr["val_start"]
        lo = vs - pd.Timedelta(days=BASELINE_WINDOW_DAYS)
        hi = vs - pd.Timedelta(days=1)
        sub = feats.loc[
            (feats["price_date"] >= lo) & (feats["price_date"] <= hi),
            ["station_code", "cycle_mean_length"],
        ].dropna()
        if sub.empty:
            continue
        bsl = sub.groupby("station_code", as_index=False)["cycle_mean_length"].median()
        bsl["fold"] = f
        bsl = bsl.rename(columns={"cycle_mean_length": "baseline_cml"})
        base_rows.append(bsl)
    baselines = pd.concat(base_rows, ignore_index=True)
    print(f"  baselines: {len(baselines):,} (fold, station) rows  "
          f"({time.perf_counter()-t0:.1f}s)")

    wide = wide.merge(baselines, on=["fold", "station_code"], how="left")

    # ---- Cohort flags ----
    # Descent indicator: price-based, not pct-based (per docstring rationale).
    wide["descent_arm"] = (wide["px_change_5d"] < 0)
    wide["is_elongated"] = (
        (wide["cycle_days_since_peak"] / wide["baseline_cml"]) > ELONGATION_TAU
    )
    wide["bucket"] = np.where(
        wide["is_elongated"] & wide["descent_arm"], "ext_descent",
        np.where(wide["descent_arm"], "normal_descent",
                 np.where(wide["is_elongated"], "elong_ascent", "normal_ascent")),
    )
    # NaN rows (no px_change_5d or no baseline) get NaN bucket via the np.where
    # cascade — drop them for stratification.
    has_data = wide["px_change_5d"].notna() & wide["baseline_cml"].notna()
    print(f"\nRows with covariate data: {has_data.sum():,} / {len(wide):,} "
          f"({has_data.mean()*100:.1f}%)")
    wide = wide[has_data].copy()

    # ---- All-fold stratification ----
    print("\n=== ALL FOLDS — per-row delta_ll by bucket ===")
    overall = (
        wide.groupby("bucket")
        .agg(n=("delta_ll", "size"),
             mean_delta=("delta_ll", "mean"),
             median_delta=("delta_ll", "median"),
             pct_positive=("delta_ll", lambda s: float((s > 0).mean())),
             n_label1=("label", "sum"))
        .reset_index()
        .sort_values("mean_delta", ascending=False)
    )
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Same but on hard25 cohort
    print("\n=== HARD25 ROWS — per-row delta_ll by bucket ===")
    hard = wide[wide["is_hard25"] == 1]
    hard_strat = (
        hard.groupby("bucket")
        .agg(n=("delta_ll", "size"),
             mean_delta=("delta_ll", "mean"),
             median_delta=("delta_ll", "median"),
             pct_positive=("delta_ll", lambda s: float((s > 0).mean())))
        .reset_index()
        .sort_values("mean_delta", ascending=False)
    )
    print(hard_strat.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- Per-fold breakdown for ext_descent rows ----
    print("\n=== PER FOLD — ext_descent rows: n + mean delta_ll (all rows) ===")
    ext = wide[wide["bucket"] == "ext_descent"]
    pf_ext = (
        ext.groupby("fold")
        .agg(n=("delta_ll", "size"),
             mean_delta=("delta_ll", "mean"),
             median_delta=("delta_ll", "median"),
             pct_positive=("delta_ll", lambda s: float((s > 0).mean())))
        .reset_index()
        .sort_values("fold")
    )
    print(pf_ext.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Compare: per-fold mean delta_ll on ext_descent vs normal_descent
    pf_all = (
        wide.groupby(["fold", "bucket"])
        .agg(n=("delta_ll", "size"), mean_delta=("delta_ll", "mean"))
        .reset_index()
    )
    pivot = pf_all.pivot(index="fold", columns="bucket", values="mean_delta")
    pivot["ext_minus_normal_descent"] = pivot.get("ext_descent") - pivot.get("normal_descent")
    print("\n=== PER FOLD — mean delta_ll, ext_descent - normal_descent ===")
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ---- Fold 7 deep-dive ----
    print("\n=== FOLD 7 deep-dive ===")
    f7 = wide[wide["fold"] == 7]
    f7_strat = (
        f7.groupby("bucket")
        .agg(n=("delta_ll", "size"),
             mean_delta=("delta_ll", "mean"),
             contribution=("delta_ll", lambda s: float(s.sum())))
        .reset_index()
        .sort_values("mean_delta", ascending=False)
    )
    f7_strat["pct_of_fold_sum"] = f7_strat["contribution"] / f7_strat["contribution"].sum()
    print(f7_strat.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- Save artefacts ----
    out_parquet = OUT / "step5_rowdelta.parquet"
    wide.to_parquet(out_parquet, index=False, compression="zstd")
    print(f"\nWrote {out_parquet}")
    out_csv = OUT / "step5_rowlevel_summary.csv"
    pivot.to_csv(out_csv)
    print(f"Wrote {out_csv}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: distribution of delta_ll per bucket (all rows)
    ax = axes[0]
    buckets = ["normal_ascent", "normal_descent", "elong_ascent", "ext_descent"]
    data = [wide.loc[wide["bucket"] == b, "delta_ll"].dropna() for b in buckets]
    bp = ax.boxplot(data, tick_labels=buckets, showfliers=False)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel("Per-row delta_ll (R5 - R0; positive = A makes it worse)")
    ax.set_title("All folds — per-row delta_ll by bucket (boxplot, fliers hidden)")
    ax.tick_params(axis="x", rotation=20)
    # Annotate n
    for i, b in enumerate(buckets, start=1):
        n = int((wide["bucket"] == b).sum())
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n:,}",
                ha="center", fontsize=8, color="dimgrey")

    # Plot 2: per-fold ext_descent delta vs normal_descent delta
    ax = axes[1]
    p = pivot.reset_index()
    if "ext_descent" in p.columns and "normal_descent" in p.columns:
        ax.scatter(p["normal_descent"], p["ext_descent"], s=120, edgecolor="black")
        for _, r in p.iterrows():
            ax.annotate(f"f{int(r['fold'])}",
                        (r["normal_descent"], r["ext_descent"]),
                        textcoords="offset points", xytext=(7, 4), fontsize=9)
        lim = float(np.nanmax(np.abs(p[["normal_descent", "ext_descent"]].to_numpy())))
        ax.plot([-lim, lim], [-lim, lim], ls=":", color="grey", lw=0.7)
        ax.axhline(0, color="grey", lw=0.4)
        ax.axvline(0, color="grey", lw=0.4)
        ax.set_xlabel("mean delta_ll on normal_descent rows")
        ax.set_ylabel("mean delta_ll on ext_descent rows")
        ax.set_title("Per-fold — does A regress more on ext_descent than normal_descent?")
    fig.tight_layout()
    out_png = OUT / "step5_rowlevel.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"Wrote {out_png}")

    print(f"\nTotal wall: {time.perf_counter() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
