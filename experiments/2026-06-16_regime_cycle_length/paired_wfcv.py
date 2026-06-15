"""Regime-local cycle-length denominator — paired walk-forward CV — issue #254.

`cycle_mean_length` (and therefore `cycle_pct_through = days_since_last_peak /
mean_cycle_length`) is currently an expanding all-history mean from 2016 — it
averages ACROSS the COVID structural break, where cycle length steps ~28d ->
~41d. Effect: `cycle_pct_through` inflated ~13% on average, worst exactly in
fold 7's era (the #214/#231 corner-failure fold). This experiment swaps the
denominator for a regime-local, break-floored shrunk median (see cycle_regime.py)
and asks whether the corrected phase axis improves the model.

Cycle features are RECOMPUTED live through cycle.py (baseline = unpatched HEAD,
regime = RegimeCycleDetector) — the cached features.csv cycle columns are NOT
reused (only the 52 non-cycle-length columns come from the cache; they are
unaffected by the denominator). Both arms therefore differ in cycle_mean_length
and cycle_pct_through ONLY.

Arms:
  R0   baseline           54 feat, cycle denominator = unpatched expanding mean
  R1   + regime denom     54 feat, cycle_mean_length/pct_through swapped to regime
  R2   + is_post_covid     54 feat (baseline) + is_post_covid dummy (let the tree
                            learn the regime split itself; denominator unchanged)

3 runs × 14 folds × 5 seeds = 210 LightGBM fits (R0/seed0 reused per fold).

Decision rule (#254): normal-fold MEDIAN Δlogloss improves AND fold 7 does NOT
regress (the falsifiable claim), with a bounded shock-fold worst case.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-16_regime_cycle_length/paired_wfcv.py \
    2>&1 | tee experiments/2026-06-16_regime_cycle_length/run.log
"""
from __future__ import annotations

import pathlib
import sys
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
from fuel_signal import db as _db
from fuel_signal.cycle import CycleDetector
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    load_features,
)

# Sibling module in this dir. The dated dir can't be an importable package (its
# name starts with a digit), so add it to sys.path and import by bare name.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cycle_regime import BREAK_DATE, RegimeCycleDetector  # noqa: E402

OUT = pathlib.Path(__file__).parent

# Cycle-length-dependent columns (the only two the denominator touches).
MEAN_COL = "cycle_mean_length"
PCT_COL = "cycle_pct_through"
REGIME_MEAN_COL = "cycle_mean_length_regime"
REGIME_PCT_COL = "cycle_pct_through_regime"
DUMMY_COL = "is_post_covid"

# ── Decision gates (issue #254) ───────────────────────────────────────────────
# Machine gates on the high-leverage trough cohort (hard25 = top-quartile
# baseline per-row loss per fold — where a wrong phase axis bites hardest).
# Sign: Δ = run − R0; negative is better; a gate passes when value <= threshold.
#   target_fold=7, target_max=+0.005 -> fold 7 must NOT regress (small tolerance)
#   worst_fold_max -> overall worst-fold cap (shock folds also reported separately)
#   net_pop_max=0.0 -> mean all-rows Δ across folds must not increase loss
# The PRIMARY #254 claim (normal-fold MEDIAN Δ improves) is evaluated as a custom
# verdict below, since GateSpec has no normal-fold-median slot.
GATE = GateSpec(
    cohort_col="delta_ll_hard25_median",
    pop_col="delta_ll_all_median",
    target_fold=7,
    target_max=0.005,
    worst_fold_max=0.03,
    net_pop_max=0.0,
)


def recompute_cycle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute cycle_mean_length / cycle_pct_through live through cycle.py.

    The cycle state is a per-DATE quantity (single metro-average series shared
    across all stations), so we build one value per unique date and broadcast.
    Baseline values come from the unpatched ``CycleDetector`` (post-#250 HEAD);
    regime values from ``RegimeCycleDetector``. The cached features.csv cycle
    columns are overwritten with the freshly computed baseline so R0 is the
    honest live baseline and R0/R1 differ ONLY in the denominator.
    """
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])

    conn = _db.open_db()
    try:
        series = _db.average_price_series(conn)  # read-only SELECT
    finally:
        conn.close()

    base_det = CycleDetector(series)
    regime_det = RegimeCycleDetector(series)

    date_strs = sorted(df["price_date"].dt.strftime("%Y-%m-%d").unique())
    base_ml, base_pct, reg_ml, reg_pct = {}, {}, {}, {}
    for d in date_strs:
        bs = base_det.detect(d)
        if bs is not None:
            base_ml[d] = bs.mean_cycle_length
            base_pct[d] = bs.pct_through_cycle
        rs = regime_det.detect(d)
        if rs is not None:
            reg_ml[d] = rs.mean_cycle_length
            reg_pct[d] = rs.pct_through_cycle

    key = df["price_date"].dt.strftime("%Y-%m-%d")
    df[MEAN_COL] = key.map(base_ml)
    df[PCT_COL] = key.map(base_pct)
    df[REGIME_MEAN_COL] = key.map(reg_ml)
    df[REGIME_PCT_COL] = key.map(reg_pct)
    df[DUMMY_COL] = (df["price_date"] >= BREAK_DATE).astype(float)
    return df


def main() -> None:
    overall_t0 = time.perf_counter()

    print("Loading features ...", flush=True)
    with time_block("load_features"):
        df = load_features()
    print(f"  rows={len(df):,}", flush=True)

    print("Recomputing cycle columns through cycle.py (baseline + regime) ...", flush=True)
    with time_block("recompute_cycle_columns"):
        df = recompute_cycle_columns(df)
    n_changed = int((df[MEAN_COL] != df[REGIME_MEAN_COL]).sum())
    print(
        f"  baseline {MEAN_COL}: median={df[MEAN_COL].median():.2f}d  "
        f"regime: median={df[REGIME_MEAN_COL].median():.2f}d  "
        f"rows where they differ: {n_changed:,}/{len(df):,}",
        flush=True,
    )
    print(
        f"  {DUMMY_COL}: post-COVID rows {int(df[DUMMY_COL].sum()):,}/{len(df):,}",
        flush=True,
    )

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    assert len(baseline_cols) == 54, f"expected 54, got {len(baseline_cols)}"

    # R1 swaps the two denominator-dependent columns in place (same model inputs,
    # regime values). R2 adds the dummy. R0 is the baseline column set verbatim.
    r1_cols = [
        REGIME_MEAN_COL if c == MEAN_COL else REGIME_PCT_COL if c == PCT_COL else c
        for c in baseline_cols
    ]
    run_cols: dict[str, list[str]] = {
        "R0": baseline_cols,
        "R1": r1_cols,
        "R2": baseline_cols + [DUMMY_COL],
    }
    print(f"\nBaseline features: {len(baseline_cols)}", flush=True)
    print(f"Run grid: {list(run_cols.keys())}", flush=True)
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

        for run_name, cols in run_cols.items():
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
                    f"{val_start.strftime('%Y-%m-%d'):>10}  {val_end.strftime('%Y-%m-%d'):>10}  "
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

    # ── Verdict ───────────────────────────────────────────────────────────────
    # #254 primary claim is on NORMAL folds; shock folds (1,4,9,13) reported but
    # not allowed to sink the verdict (the project has no systematic shock layer).
    normal_mask = ~fold_run["fold"].isin(SHOCK_FOLDS)
    custom_verdict: dict[str, dict] = {}
    gate_results: dict[str, list[dict]] = {}

    print("\n=== Gate evaluation (sign: Δ = run − R0; negative is better) ===", flush=True)
    print(
        f"    {'run':<10}  {'normal med Δh25':>15}  {'normal med Δall':>15}  "
        f"{'fold7 Δh25':>11}  {'shock worst Δh25':>16}  {'gates'}",
        flush=True,
    )
    for run_name in run_cols:
        sub = fold_run[fold_run["run"] == run_name]
        if run_name == "R0":
            base_h25 = float(np.nanmedian(sub.loc[normal_mask, "ll_hard25_median"]))
            print(f"    {run_name:<10}  baseline (normal-fold median ll_h25 = {base_h25:.4f})",
                  flush=True)
            continue

        norm = sub[normal_mask]
        normal_med_h25 = float(np.nanmedian(norm["delta_ll_hard25_median"]))
        normal_med_all = float(np.nanmedian(norm["delta_ll_all_median"]))
        fold7 = sub.loc[sub["fold"] == 7, "delta_ll_hard25_median"]
        fold7_h25 = float(fold7.iloc[0]) if len(fold7) else float("nan")
        shock = sub[sub["fold"].isin(SHOCK_FOLDS)]
        shock_worst_h25 = float(shock["delta_ll_hard25_median"].max()) if len(shock) else float("nan")

        # #254 falsifiable claim: normal-fold median improves AND fold 7 ≤ +tol.
        claim_normal_improves = normal_med_h25 < 0
        claim_fold7_ok = fold7_h25 <= GATE.target_max
        passes_claim = claim_normal_improves and claim_fold7_ok

        gates = evaluate_gates(fold_run, GATE, run_name)
        gate_results[run_name] = gates
        custom_verdict[run_name] = {
            "normal_fold_median_delta_h25": normal_med_h25,
            "normal_fold_median_delta_all": normal_med_all,
            "fold7_delta_h25": fold7_h25,
            "shock_worst_delta_h25": shock_worst_h25,
            "claim_normal_median_improves": bool(claim_normal_improves),
            "claim_fold7_not_regressing": bool(claim_fold7_ok),
            "passes_254_claim": bool(passes_claim),
            "machine_gates_pass": bool(all(g["passed"] for g in gates)),
        }
        verdict = "PASS" if passes_claim else "FAIL"
        print(
            f"    {run_name:<10}  {normal_med_h25:>+15.4f}  {normal_med_all:>+15.4f}  "
            f"{fold7_h25:>+11.4f}  {shock_worst_h25:>+16.4f}  {verdict}"
            f"  (machine gates: {'pass' if all(g['passed'] for g in gates) else 'fail'})",
            flush=True,
        )

    meta = {
        "issue": 254,
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "n_baseline_features": len(baseline_cols),
        "break_date": str(BREAK_DATE.date()),
        "candidate_columns": [REGIME_MEAN_COL, REGIME_PCT_COL, DUMMY_COL],
        "definitions": {
            REGIME_MEAN_COL: (
                "regime-local cycle-length denominator: expanding median of "
                "post-break confirmed-peak cycle lengths, hard-floored at "
                f"{BREAK_DATE.date()}, pseudo-count k=2 shrinkage toward the "
                "pre-COVID median during warm-up (median-augment form). "
                "Cycles stamped at closing peak."
            ),
            REGIME_PCT_COL: "cycle_days_since_peak / cycle_mean_length_regime",
            DUMMY_COL: f"1.0 if price_date >= {BREAK_DATE.date()} else 0.0",
        },
        "run_grid": {k: v for k, v in run_cols.items()},
        "gate_spec": {
            "cohort_col": GATE.cohort_col, "pop_col": GATE.pop_col,
            "target_fold": GATE.target_fold, "target_max": GATE.target_max,
            "worst_fold_max": GATE.worst_fold_max, "net_pop_max": GATE.net_pop_max,
        },
        "decision_rule_254": (
            "PRIMARY: normal-fold median Δ(ll_hard25) < 0 AND fold-7 Δ(ll_hard25) "
            "<= +0.005 (does not regress). Shock-fold worst case reported, bounded."
        ),
        "cohort_definitions": {
            "all": "full val set",
            "hard25": "top quartile baseline per-row log-loss per fold",
        },
        "aggregation_convention": (
            "Headline = median across 5 seeds per (fold, run); normal-fold verdict "
            "then takes the median of those across the 10 normal folds. Mean shown alongside."
        ),
        "seed_variance_gate": {
            "rule": "ratio = seed_std / median(seed_std across cohort cells); flag > 5×",
            "per_cohort": seed_var_summary,
            "flagged_cells": seed_var_flags,
        },
        "verdict_254": custom_verdict,
        "machine_gate_results": gate_results,
        "total_wall_seconds": time.perf_counter() - overall_t0,
    }
    write_meta(OUT, meta)
    print(f"[total wall] {time.perf_counter() - overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
