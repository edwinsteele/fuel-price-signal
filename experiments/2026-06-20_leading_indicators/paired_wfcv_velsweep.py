"""TGP velocity horizon sweep — leading-indicator exploration (#215).

paired_wfcv_velocity.py established that TGP velocity (tgp_delta_7d) carries
strong independent signal and rescues the shock regime where the raw gap fails;
gap-only is unhelpful. This sweeps the velocity HORIZON to find the best period.

Velocity = point-to-point N-day change of PIT Sydney ULP TGP: TGP_t - TGP_{t-N}
(an N-day momentum, NOT a moving average). Same form as the proven vel7 — only
the period N varies, so the arms are directly comparable.

Run grid (each arm = velocity-only at one horizon, vs the same baseline):
  R0     54-feat baseline
  vel3   + tgp_delta_3d
  vel7   + tgp_delta_7d   (the proven horizon)
  vel14  + tgp_delta_14d

Question: which horizon maximises the SHOCK gain (where the #262 money is)
without losing the normal-fold gain? Hypothesis (user): shocks move fast, so a
shorter horizon may help shocks more.

4 runs × 14 folds × 5 seeds = 280 LightGBM fits.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-20_leading_indicators/paired_wfcv_velsweep.py \\
    2>&1 | tee experiments/2026-06-20_leading_indicators/run_velsweep.log
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd

from experiments.lib.aggregate import aggregate_with_deltas
from experiments.lib.cohorts import hard_quantile_mask
from experiments.lib.constants import SEEDS, SHOCK_FOLDS
from experiments.lib.fit import fit_score, per_row_log_loss
from experiments.lib.folds import iter_folds_with_baseline_fit
from experiments.lib.gates import seed_variance_gate
from experiments.lib.io import write_meta
from experiments.lib.rowpreds import RowPredCollector
from experiments.lib.timing import time_block
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    load_features,
)

OUT = pathlib.Path(__file__).parent
TGP_XLSX = OUT / "data" / "AIP_TGP_2026-06-19.xlsx"

HORIZONS = [3, 7, 14]
RUNS: dict[str, list[str]] = {"R0": []}
for _n in HORIZONS:
    RUNS[f"vel{_n}"] = [f"tgp_delta_{_n}d"]
CANDIDATE_COLS = [c for cols in RUNS.values() for c in cols]


def _load_tgp_pit() -> pd.Series:
    """Daily PIT Sydney ULP TGP (c/L): weekday series, ffill weekends, lag 1 day."""
    tgp = pd.read_excel(TGP_XLSX, sheet_name="Petrol TGP")
    dcol = tgp.columns[0]
    tgp = tgp[[dcol, "Sydney"]].rename(columns={dcol: "date", "Sydney": "tgp"})
    tgp["date"] = pd.to_datetime(tgp["date"], errors="coerce")
    s = tgp.dropna(subset=["date"]).set_index("date")["tgp"].sort_index()
    return s.asfreq("D").ffill().shift(1)


def add_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])
    pit = _load_tgp_pit()
    for n in HORIZONS:
        df[f"tgp_delta_{n}d"] = df["price_date"].map(pit - pit.shift(n))
    return df


def main() -> None:
    overall_t0 = time.perf_counter()

    print("Loading features ...", flush=True)
    with time_block("load_features"):
        df = load_features()
    print(f"  rows={len(df):,}", flush=True)

    print("Computing candidate features in-script ...", flush=True)
    with time_block("add_candidate_columns"):
        df = add_candidate_columns(df)
    for c in CANDIDATE_COLS:
        s = df[c]
        print(
            f"  {c}: null={s.isna().mean():.4%}  mean={s.mean():.2f}  sd={s.std():.2f}  "
            f"p1={s.quantile(.01):.1f}  p99={s.quantile(.99):.1f}",
            flush=True,
        )

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    assert len(baseline_cols) == 54, f"expected 54, got {len(baseline_cols)}"
    print(f"\nBaseline features: {len(baseline_cols)}", flush=True)
    print(f"Run grid: {list(RUNS.keys())}", flush=True)
    print(f"Seeds: {SEEDS} (n={len(SEEDS)})", flush=True)

    print(
        f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  "
        f"{'val_rows':>8}  {'run':<6}  {'seed':>4}  {'ll_all':>7}  {'ll_h25':>7}  {'fit_s':>6}",
        flush=True,
    )
    print("-" * 100, flush=True)

    rows: list[dict] = []
    collector = RowPredCollector(pd.DataFrame())

    for fold_idx, regime, train_df, val_df, ll0, p0, t0, prl0 in iter_folds_with_baseline_fit(
        df, baseline_cols
    ):
        vd = pd.to_datetime(val_df["price_date"])
        val_start, val_end = vd.min(), vd.max()
        y = val_df["label"].to_numpy(dtype=int)
        hard25_mask = hard_quantile_mask(prl0, 0.75)

        collector.ident_base = pd.DataFrame({
            "fold": np.int8(fold_idx),
            "station_code": val_df["station_code"].to_numpy(),
            "price_date": vd.to_numpy(),
            "label": y.astype(np.int8),
            "is_hard25": hard25_mask.astype(np.int8),
        })

        for run_name, extra in RUNS.items():
            cols = baseline_cols + extra
            for seed in SEEDS:
                if run_name == "R0" and seed == SEEDS[0]:
                    ll, p, t = ll0, p0, t0
                else:
                    ll, p, t = fit_score(train_df, val_df, cols, seed)
                prl = per_row_log_loss(y, p)
                ll_hard25 = float(prl[hard25_mask].mean()) if hard25_mask.any() else float("nan")
                rows.append({
                    "fold": fold_idx, "regime": regime,
                    "val_start": val_start.strftime("%Y-%m-%d"),
                    "val_end": val_end.strftime("%Y-%m-%d"),
                    "val_rows": len(val_df),
                    "run": run_name, "n_features": len(cols), "seed": seed,
                    "ll_all": ll, "ll_hard25": ll_hard25, "fit_s": t,
                })
                collector.add(run_name, seed, p)
                print(
                    f"{fold_idx:>4}  {regime:>6}  {val_start.strftime('%Y-%m-%d'):>10}  "
                    f"{val_end.strftime('%Y-%m-%d'):>10}  {len(val_df):>8,}  {run_name:<6}  "
                    f"{seed:>4}  {ll:>7.4f}  {ll_hard25:>7.4f}  {t:>5.1f}s",
                    flush=True,
                )

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "runs_velsweep.csv", index=False)
    print(f"\nPer-(fold,run,seed) results: {OUT / 'runs_velsweep.csv'}", flush=True)
    collector.to_parquet(OUT / "rowpreds_velsweep.parquet")

    cohort_ll = {"all": "ll_all", "hard25": "ll_hard25"}
    seed_var_summary, seed_var_flags = seed_variance_gate(df_rows, cohort_ll)
    fold_run = aggregate_with_deltas(df_rows, cohort_ll)
    fold_run.to_csv(OUT / "fold_run_velsweep.csv", index=False)

    # ── HEADLINE: regime summary (mean Δll_all by regime), per horizon ────────
    print("\n=== HEADLINE — mean Δll_all (median across seeds), by regime ===", flush=True)
    print("    (negative = better; best horizon = most negative SHOCK without losing normal)", flush=True)
    print(f"    {'run':<6}  {'normal':>9}  {'shock':>9}  {'pooled':>9}", flush=True)
    nonbase = [r for r in RUNS if r != "R0"]
    for run_name in nonbase:
        sub = fold_run[fold_run["run"] == run_name].set_index("fold")["delta_ll_all_median"]
        normal = sub[[f for f in sub.index if f not in SHOCK_FOLDS]].mean()
        shock = sub[[f for f in sub.index if f in SHOCK_FOLDS]].mean()
        print(f"    {run_name:<6}  {normal:>+9.4f}  {shock:>+9.4f}  {sub.mean():>+9.4f}", flush=True)

    # ── per-fold Δll_all matrix (fold × horizon) ──────────────────────────────
    print("\n=== Per-fold Δll_all (R* − R0), median across seeds ===", flush=True)
    header = "    " + f"{'fold':>4}  {'regime':>6}  " + "  ".join(f"{r:>9}" for r in nonbase)
    print(header, flush=True)
    piv = fold_run.pivot_table(index=["fold", "regime"], columns="run",
                               values="delta_ll_all_median")
    for (fold, regime), r in piv.iterrows():
        shock = " *shock" if int(fold) in SHOCK_FOLDS else ""
        cells = "  ".join(f"{r[run]:>+9.4f}" for run in nonbase)
        print(f"    {int(fold):>4}  {regime:>6}  {cells}{shock}", flush=True)

    if seed_var_flags:
        print(f"\n(seed-variance: {len(seed_var_flags)} flagged cells — see meta.json)", flush=True)

    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "n_baseline_features": len(baseline_cols),
        "horizons": HORIZONS,
        "candidate_columns": CANDIDATE_COLS,
        "definitions": {
            f"tgp_delta_{n}d": f"PIT Sydney ULP TGP - same {n} days prior (c/L); N-day momentum"
            for n in HORIZONS
        },
        "run_grid": dict(RUNS),
        "cohort_definitions": {
            "all": "full val set",
            "hard25": "top quartile baseline per-row log-loss per fold",
        },
        "seed_variance_gate": {
            "rule": "ratio = seed_std / median(seed_std across cohort cells); flag > 5×",
            "per_cohort": seed_var_summary, "flagged_cells": seed_var_flags,
        },
        "note": (
            "Velocity horizon sweep (velocity-only, point-to-point N-day momentum). "
            "Headline = regime split of Δll_all. R3 (gap+velocity) deferred."
        ),
        "total_wall_seconds": time.perf_counter() - overall_t0,
    }
    write_meta(OUT, meta)
    print(f"[total wall] {time.perf_counter() - overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
