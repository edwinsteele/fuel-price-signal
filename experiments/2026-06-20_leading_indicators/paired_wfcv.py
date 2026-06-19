"""TGP floor-anchor feature — leading-indicator exploration (#215 thread).

Tests whether `station_minus_tgp_cents` (current pump price minus the AIP Sydney
ULP Terminal Gate Price — the wholesale floor) adds signal over the 54-feature
baseline. Motivation (this experiment's overlay + oracle diagnostic):
  - pump cycle troughs "kiss" TGP (kiss-gap mean -0.3, sd 3.1 c/L);
  - gap-to-TGP explains depth-remaining (r2=0.63) — a forward floor anchor the
    model lacks. Its only current anchor, `station_minus_last_min_cents`, is the
    PREVIOUS cycle's trough — stale exactly in shocks (2020/2022), where the
    #262 headroom map says the recoverable money is.

This is the WFCV log-loss SCREEN. The arbiter (realised backtest) needs TGP
plumbed into PriceHistory.decide() (follow-up) — see the experiment README.

Candidate columns:
  station_minus_tgp_cents — station_price_cents - TGP_sydney(price_date), PIT.

Run grid:
  R0   54-feat baseline
  R1   + station_minus_tgp_cents

2 runs × 14 folds × 5 seeds = 140 LightGBM fits.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-20_leading_indicators/paired_wfcv.py \\
    2>&1 | tee experiments/2026-06-20_leading_indicators/run.log
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
from experiments.lib.gates import GateSpec, evaluate_gates, seed_variance_gate
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

CANDIDATE_COL = "station_minus_tgp_cents"

RUNS: dict[str, list[str]] = {
    "R0": [],
    "R1": [CANDIDATE_COL],
}

# Sign convention (single-sourced in evaluate_gates): Δ = run − R0; negative is
# better. A floor-anchor should help broadly and most where last_min goes stale —
# the COVID shock fold (4). Modest target; the real read is the per-fold table +
# the realised arbiter (follow-up).
GATE = GateSpec(
    cohort_col="delta_ll_hard25_median",
    pop_col="delta_ll_all_median",
    target_fold=4,           # COVID shock — clearest stale-anchor divergence
    target_max=-0.01,        # ≥0.01 improvement there
    worst_fold_max=0.02,     # no fold may regress by more than this
    net_pop_max=0.0,         # mean Δ across all rows must be ≤ 0
)


def _load_tgp_pit() -> pd.Series:
    """Daily PIT Sydney ULP TGP (c/L): weekday series, ffill weekends, lag 1 day.

    The 1-day lag is conservative: TGP is published each weekday morning, so
    same-day is arguably available at decision time, but lagging removes any
    intraday-timing doubt for the screen (cost ≈ slow drift, ~1 c/L).
    """
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
    tgp_pit = df["price_date"].map(pit)
    df[CANDIDATE_COL] = df["station_price_cents"] - tgp_pit
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
    null_rate = df[CANDIDATE_COL].isna().mean()
    print(
        f"  {CANDIDATE_COL}: null_rate={null_rate:.4%}  "
        f"mean={df[CANDIDATE_COL].mean():.2f}  sd={df[CANDIDATE_COL].std():.2f}  "
        f"p1={df[CANDIDATE_COL].quantile(.01):.1f}  p99={df[CANDIDATE_COL].quantile(.99):.1f}",
        flush=True,
    )

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    assert len(baseline_cols) == 54, f"expected 54, got {len(baseline_cols)}"
    print(f"\nBaseline features: {len(baseline_cols)}", flush=True)
    print(f"Run grid: {list(RUNS.keys())}", flush=True)
    print(f"Seeds: {SEEDS} (n={len(SEEDS)})", flush=True)

    print(
        f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  "
        f"{'val_rows':>8}  {'run':<10}  {'seed':>4}  "
        f"{'ll_all':>7}  {'ll_h25':>7}  {'fit_s':>6}",
        flush=True,
    )
    print("-" * 110, flush=True)

    rows: list[dict] = []
    collector = RowPredCollector(pd.DataFrame())

    for fold_idx, regime, train_df, val_df, ll0, p0, t0, prl0 in iter_folds_with_baseline_fit(
        df, baseline_cols
    ):
        vd = pd.to_datetime(val_df["price_date"])
        val_start = vd.min()
        val_end = vd.max()
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
                    "run": run_name, "n_features": len(cols),
                    "seed": seed,
                    "ll_all": ll, "ll_hard25": ll_hard25,
                    "fit_s": t,
                })
                collector.add(run_name, seed, p)

                print(
                    f"{fold_idx:>4}  {regime:>6}  "
                    f"{val_start.strftime('%Y-%m-%d'):>10}  "
                    f"{val_end.strftime('%Y-%m-%d'):>10}  "
                    f"{len(val_df):>8,}  {run_name:<10}  {seed:>4}  "
                    f"{ll:>7.4f}  {ll_hard25:>7.4f}  {t:>5.1f}s",
                    flush=True,
                )

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "runs.csv", index=False)
    print(f"\nPer-(fold,run,seed) results: {OUT / 'runs.csv'}", flush=True)

    collector.to_parquet(OUT / "rowpreds.parquet")

    cohort_ll = {"all": "ll_all", "hard25": "ll_hard25"}
    seed_var_summary, seed_var_flags = seed_variance_gate(df_rows, cohort_ll)
    fold_run = aggregate_with_deltas(df_rows, cohort_ll)
    fold_run.to_csv(OUT / "fold_run.csv", index=False)

    gate_results: dict[str, list[dict]] = {}
    print("\n=== Gate evaluation (sign: Δ = run − R0; negative is better) ===", flush=True)
    print(
        f"    {'run':<10}  {'Δh25 (med)':>12}  {'Δall med':>10}  "
        f"{'helps_h25':>10}  {'gates'}",
        flush=True,
    )
    for run_name in RUNS:
        sub = fold_run[fold_run["run"] == run_name]
        n_folds = len(sub)
        if run_name == "R0":
            print(
                f"    {run_name:<10}  baseline "
                f"(ll_h25 median across folds = "
                f"{float(np.nanmedian(sub['ll_hard25_median'])):.4f})",
                flush=True,
            )
            continue
        d_h25_med = sub["delta_ll_hard25_median"].to_numpy()
        d_all_med = sub["delta_ll_all_median"].to_numpy()
        gates = evaluate_gates(fold_run, GATE, run_name)
        gate_results[run_name] = gates
        verdict = "PASS" if all(g["passed"] for g in gates) else "FAIL"
        failed = [g["name"] for g in gates if not g["passed"]]
        print(
            f"    {run_name:<10}  "
            f"{float(d_h25_med.mean()):>+12.4f}  "
            f"{float(d_all_med.mean()):>+10.4f}  "
            f"{(d_h25_med < 0).sum():>4}/{n_folds:<5}  "
            f"{verdict}" + (f"  (failed: {', '.join(failed)})" if failed else ""),
            flush=True,
        )

    # Per-fold Δ table (the real read for a regime-targeted feature)
    print("\n=== Per-fold Δ (R1 − R0), median across seeds ===", flush=True)
    print(f"    {'fold':>4}  {'regime':>6}  {'Δll_all':>9}  {'Δll_h25':>9}", flush=True)
    r1 = fold_run[fold_run["run"] == "R1"].sort_values("fold")
    for _, r in r1.iterrows():
        shock = " *shock" if int(r["fold"]) in SHOCK_FOLDS else ""
        print(
            f"    {int(r['fold']):>4}  {r['regime']:>6}  "
            f"{r['delta_ll_all_median']:>+9.4f}  {r['delta_ll_hard25_median']:>+9.4f}{shock}",
            flush=True,
        )

    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "n_baseline_features": len(baseline_cols),
        "candidate_columns": [CANDIDATE_COL],
        "definitions": {
            CANDIDATE_COL: (
                "station_price_cents - PIT Sydney ULP TGP (c/L, GST incl., AIP "
                "AIP_TGP_2026-06-19.xlsx 'Petrol TGP' Sydney col); TGP ffilled "
                "across weekends and lagged 1 day for PIT safety."
            ),
        },
        "run_grid": dict(RUNS),
        "gate_spec": {
            "cohort_col": GATE.cohort_col,
            "pop_col": GATE.pop_col,
            "target_fold": GATE.target_fold,
            "target_max": GATE.target_max,
            "worst_fold_max": GATE.worst_fold_max,
            "net_pop_max": GATE.net_pop_max,
        },
        "cohort_definitions": {
            "all": "full val set",
            "hard25": "top quartile baseline per-row log-loss per fold",
        },
        "aggregation_convention": (
            "Headline = median across 5 seeds per (fold, run); summary then "
            "averages those medians across 14 folds. Mean shown alongside."
        ),
        "seed_variance_gate": {
            "rule": "ratio = seed_std / median(seed_std across cohort cells); flag > 5×",
            "per_cohort": seed_var_summary,
            "flagged_cells": seed_var_flags,
        },
        "gate_results": gate_results,
        "candidate_null_rate": float(null_rate),
        "note": (
            "WFCV log-loss SCREEN only. Realised backtest (arbiter) requires TGP "
            "plumbed into PriceHistory.decide() — follow-up if this passes."
        ),
        "total_wall_seconds": time.perf_counter() - overall_t0,
    }

    write_meta(OUT, meta)
    print(f"[total wall] {time.perf_counter() - overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
