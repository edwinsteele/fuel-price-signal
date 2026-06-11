"""<Experiment name> — issue #N.

<Hypothesis: what you're testing and why, 2-3 sentences.>

Candidate columns:
  <CANDIDATE_COL>  — <definition>

Run grid:
  R0   <N>-feat baseline
  R1   + <CANDIDATE_COL>

N runs × 14 folds × 5 seeds = N*14*5 LightGBM fits.

Usage:
  PYTHONPATH=. uv run python experiments/<date>_<slug>/paired_wfcv.py \\
    2>&1 | tee experiments/<date>_<slug>/run.log
"""
# ── HOW TO USE THIS TEMPLATE ─────────────────────────────────────────────────
#
# IN-SCRIPT (you own these — change them per experiment):
#   • add_candidate_columns()   compute new columns from features.csv columns
#   • RUNS dict                 run grid: R0 = baseline, R1+ add candidate cols
#   • GATE GateSpec             pass/fail thresholds for this experiment
#   • cohort / bucket masks     any val-set boolean masks beyond hard25
#   • meta["definitions"] etc.  human-readable column/bucket descriptions
#
# LIB (do not inline — always import from experiments.lib):
#   • fold iteration            iter_folds_with_baseline_fit
#   • fitting + per-row loss    fit_score, per_row_log_loss
#   • hard-cohort mask          hard_quantile_mask
#   • row-pred collection       RowPredCollector
#   • seed-variance gate        seed_variance_gate
#   • fold/run aggregation      aggregate_with_deltas
#   • gate evaluation           GateSpec, evaluate_gates
#   • meta serialisation        write_meta
#   • timing                    time_block
#   • shared constants          SEEDS, SHOCK_FOLDS, LGBM_DEFAULTS
#
# PROMOTION RULE:
#   If an add_candidate_columns block is copied across 2+ experiments,
#   extract the primitive into experiments/lib/features/ and import it.
#
# ─────────────────────────────────────────────────────────────────────────────
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

# ── TODO: rename / add candidate column names ─────────────────────────────────
CANDIDATE_COL = "my_candidate_col"

# ── TODO: define the run grid ────────────────────────────────────────────────
RUNS: dict[str, list[str]] = {
    "R0": [],
    "R1": [CANDIDATE_COL],
}

# ── TODO: set gate thresholds for this experiment ────────────────────────────
# Sign convention (single-sourced in evaluate_gates): Δ = run − R0;
# negative is better. A gate passes when value <= threshold.
GATE = GateSpec(
    cohort_col="delta_ll_hard25_median",
    pop_col="delta_ll_all_median",
    target_fold=7,          # fold that must show improvement
    target_max=-0.04,       # Δ must be at most this (improvement of ≥ 0.04)
    worst_fold_max=0.01,    # no fold may regress by more than this
    net_pop_max=0.0,        # mean Δ across all rows must be ≤ 0
)


# ── TODO: implement candidate-column computation ──────────────────────────────
def add_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candidate columns from columns already present in features.csv.

    Use experiments.lib.features.* helpers for PIT-safe rolling/delta work:
      rolling_baseline(series, window_days)          closed='left' by default
      calendar_aware_delta(series, lag_days)         reindexes to daily grid
      cohort_std_by_date(df, mask)                   same-date std
      cohort_agg_diff_by_date(df, mask_a, mask_b)    diff of two cohort aggs
    """
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])
    # TODO: compute df[CANDIDATE_COL]
    return df


def main() -> None:
    overall_t0 = time.perf_counter()

    # ── Load + compute features ───────────────────────────────────────────────
    print("Loading features ...", flush=True)
    with time_block("load_features"):
        df = load_features()
    print(f"  rows={len(df):,}", flush=True)

    print("Computing candidate features in-script ...", flush=True)
    with time_block("add_candidate_columns"):
        df = add_candidate_columns(df)
    # TODO: print null rates / coverage stats for each new column

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    # TODO: update expected count if the baseline has changed
    assert len(baseline_cols) == 54, f"expected 54, got {len(baseline_cols)}"
    print(f"\nBaseline features: {len(baseline_cols)}", flush=True)
    print(f"Run grid: {list(RUNS.keys())}", flush=True)
    print(f"Seeds: {SEEDS} (n={len(SEEDS)})", flush=True)

    # ── Per-(fold, run, seed) loop ────────────────────────────────────────────
    print(
        f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  "
        f"{'val_rows':>8}  {'run':<10}  {'seed':>4}  "
        f"{'ll_all':>7}  {'ll_h25':>7}  {'fit_s':>6}",
        flush=True,
    )
    print("-" * 110, flush=True)

    rows: list[dict] = []
    collector = RowPredCollector(
        # ident_base is updated at the start of each fold via collector.ident_base = ident
        pd.DataFrame()
    )

    for fold_idx, regime, train_df, val_df, ll0, p0, t0, prl0 in iter_folds_with_baseline_fit(
        df, baseline_cols
    ):
        vd = pd.to_datetime(val_df["price_date"])
        val_start = vd.min()
        val_end = vd.max()
        y = val_df["label"].to_numpy(dtype=int)
        hard25_mask = hard_quantile_mask(prl0, 0.75)

        # TODO: compute any experiment-specific bucket masks from val_df here
        # e.g.:  is_my_bucket = (val_df["some_col"] > threshold).to_numpy()

        # Build the row-identity block for this fold (all runs/seeds share it)
        collector.ident_base = pd.DataFrame({
            "fold": np.int8(fold_idx),
            "station_code": val_df["station_code"].to_numpy(),
            "price_date": vd.to_numpy(),
            "label": y.astype(np.int8),
            "is_hard25": hard25_mask.astype(np.int8),
            # TODO: add experiment-specific bucket columns as int8 flags here
            # "is_my_bucket": is_my_bucket.astype(np.int8),
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

    # ── Save raw per-(fold, run, seed) results ────────────────────────────────
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT / "runs.csv", index=False)
    print(f"\nPer-(fold,run,seed) results: {OUT / 'runs.csv'}", flush=True)

    collector.to_parquet(OUT / "rowpreds.parquet")

    # ── Aggregate + gates ─────────────────────────────────────────────────────
    cohort_ll = {"all": "ll_all", "hard25": "ll_hard25"}
    seed_var_summary, seed_var_flags = seed_variance_gate(df_rows, cohort_ll)
    fold_run = aggregate_with_deltas(df_rows, cohort_ll)
    fold_run.to_csv(OUT / "fold_run.csv", index=False)

    # Verdict table
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

    # ── Write meta ────────────────────────────────────────────────────────────
    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "n_baseline_features": len(baseline_cols),
        "candidate_columns": [CANDIDATE_COL],  # TODO: list all candidate cols
        "definitions": {
            # TODO: human-readable formula for each candidate column
            CANDIDATE_COL: "TODO: describe the formula",
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
        "total_wall_seconds": time.perf_counter() - overall_t0,
    }

    write_meta(OUT, meta)
    print(f"[total wall] {time.perf_counter() - overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
