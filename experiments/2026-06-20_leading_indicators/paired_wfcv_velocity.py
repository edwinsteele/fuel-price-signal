"""TGP floor-anchor + velocity redesign — leading-indicator exploration (#215).

The gap-alone screen (paired_wfcv.py) showed station_minus_tgp_cents HELPS calm
folds (normal mean Δll_all -0.012) but HURTS shocks (shock mean +0.011) — the
inverse of the hypothesis. Mechanism: the kiss is to the floor AT TROUGH time;
today's TGP is an accurate proxy when the floor is stable (calm) and misleading
when it's moving (shock). Fix = hand the model the floor's motion so it can
discount the anchor when TGP is sliding.

Velocity horizon = 7d (not 14d): the feature's job is to FLAG that the floor is
moving, not to estimate time-to-trough. Shocks move fast, so a 7d gradient spikes
at shock onset while 14d smooths over it and lags — the faster sensor is the right
one for a fast-shock failure mode.

Run grid (near-factorial — one change per arm for clean attribution):
  R0   54-feat baseline
  R1   + station_minus_tgp_cents                 (gap)
  R2   + tgp_delta_7d                            (velocity STANDALONE — independent signal?)
  R3   + gap + tgp_delta_7d                      (velocity rescue, given gap)
  R4   + gap + tgp_delta_7d + tgp_gap_x_vel7     (explicit interaction)

Questions: (a) does velocity carry independent signal (R2−R0)? (b) does it move
the SHOCK mean Δll_all from R1's +0.011 toward 0/negative without giving back the
normal-fold wins (R3 vs R1)? (c) is an explicit interaction needed (R4−R3)?

5 runs × 14 folds × 5 seeds = 350 LightGBM fits.

Usage:
  PYTHONPATH=. uv run python experiments/2026-06-20_leading_indicators/paired_wfcv_velocity.py \\
    2>&1 | tee experiments/2026-06-20_leading_indicators/run_velocity.log
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

GAP = "station_minus_tgp_cents"
VEL7 = "tgp_delta_7d"
GAPxVEL = "tgp_gap_x_vel7"
CANDIDATE_COLS = [GAP, VEL7, GAPxVEL]

RUNS: dict[str, list[str]] = {
    "R0": [],
    "R1": [GAP],
    "R2": [VEL7],
    "R3": [GAP, VEL7],
    "R4": [GAP, VEL7, GAPxVEL],
}

# Gate machinery left as-is per project decision (hard25); the verdict is read
# from the regime summary (normal vs shock Δll_all) + the realised arbiter, not
# this pass/fail. See memory feedback-hard25-not-default.
GATE = GateSpec(
    cohort_col="delta_ll_hard25_median",
    pop_col="delta_ll_all_median",
    target_fold=4,
    target_max=-0.01,
    worst_fold_max=0.02,
    net_pop_max=0.0,
)


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
    vel7 = pit - pit.shift(7)
    df[GAP] = df["station_price_cents"] - df["price_date"].map(pit)
    df[VEL7] = df["price_date"].map(vel7)
    df[GAPxVEL] = df[GAP] * df[VEL7]
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
        print(
            f"  {c}: null={df[c].isna().mean():.4%}  mean={df[c].mean():.2f}  "
            f"sd={df[c].std():.2f}  p1={df[c].quantile(.01):.1f}  p99={df[c].quantile(.99):.1f}",
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
    df_rows.to_csv(OUT / "runs_velocity.csv", index=False)
    print(f"\nPer-(fold,run,seed) results: {OUT / 'runs_velocity.csv'}", flush=True)
    collector.to_parquet(OUT / "rowpreds_velocity.parquet")

    cohort_ll = {"all": "ll_all", "hard25": "ll_hard25"}
    seed_var_summary, seed_var_flags = seed_variance_gate(df_rows, cohort_ll)
    fold_run = aggregate_with_deltas(df_rows, cohort_ll)
    fold_run.to_csv(OUT / "fold_run_velocity.csv", index=False)

    # ── HEADLINE: regime summary (mean Δll_all by regime) ─────────────────────
    print("\n=== HEADLINE — mean Δll_all (median across seeds), by regime ===", flush=True)
    print("    (negative = better; the rescue question is whether shock moves toward ≤0)", flush=True)
    print(f"    {'run':<6}  {'normal':>9}  {'shock':>9}  {'pooled':>9}", flush=True)
    for run_name in RUNS:
        if run_name == "R0":
            continue
        sub = fold_run[fold_run["run"] == run_name].set_index("fold")["delta_ll_all_median"]
        normal = sub[[f for f in sub.index if f not in SHOCK_FOLDS]].mean()
        shock = sub[[f for f in sub.index if f in SHOCK_FOLDS]].mean()
        print(
            f"    {run_name:<6}  {normal:>+9.4f}  {shock:>+9.4f}  {sub.mean():>+9.4f}",
            flush=True,
        )

    # ── per-fold Δll_all matrix (fold × run) ──────────────────────────────────
    print("\n=== Per-fold Δll_all (R* − R0), median across seeds ===", flush=True)
    nonbase = [r for r in RUNS if r != "R0"]
    header = "    " + f"{'fold':>4}  {'regime':>6}  " + "  ".join(f"{r:>9}" for r in nonbase)
    print(header, flush=True)
    piv = fold_run.pivot_table(index=["fold", "regime"], columns="run",
                               values="delta_ll_all_median")
    for (fold, regime), r in piv.iterrows():
        shock = " *shock" if int(fold) in SHOCK_FOLDS else ""
        cells = "  ".join(f"{r[run]:>+9.4f}" for run in nonbase)
        print(f"    {int(fold):>4}  {regime:>6}  {cells}{shock}", flush=True)

    # ── gate table (secondary; hard25 left as-is) ─────────────────────────────
    gate_results: dict[str, list[dict]] = {}
    print("\n=== Gate table (secondary read — hard25, machinery unchanged) ===", flush=True)
    for run_name in nonbase:
        gates = evaluate_gates(fold_run, GATE, run_name)
        gate_results[run_name] = gates
        verdict = "PASS" if all(g["passed"] for g in gates) else "FAIL"
        failed = [g["name"] for g in gates if not g["passed"]]
        print(f"    {run_name}: {verdict}" + (f"  (failed: {', '.join(failed)})" if failed else ""),
              flush=True)

    meta = {
        "seeds": list(SEEDS),
        "shock_folds": sorted(SHOCK_FOLDS),
        "n_baseline_features": len(baseline_cols),
        "candidate_columns": CANDIDATE_COLS,
        "definitions": {
            GAP: "station_price_cents - PIT Sydney ULP TGP (c/L)",
            VEL7: "PIT TGP - PIT TGP 7 days prior (c/L)",
            GAPxVEL: f"{GAP} * {VEL7} (explicit gap×velocity interaction)",
        },
        "run_grid": dict(RUNS),
        "gate_spec": {
            "cohort_col": GATE.cohort_col, "pop_col": GATE.pop_col,
            "target_fold": GATE.target_fold, "target_max": GATE.target_max,
            "worst_fold_max": GATE.worst_fold_max, "net_pop_max": GATE.net_pop_max,
        },
        "cohort_definitions": {
            "all": "full val set",
            "hard25": "top quartile baseline per-row log-loss per fold",
        },
        "seed_variance_gate": {
            "rule": "ratio = seed_std / median(seed_std across cohort cells); flag > 5×",
            "per_cohort": seed_var_summary, "flagged_cells": seed_var_flags,
        },
        "gate_results": gate_results,
        "note": (
            "Velocity-rescue redesign. Headline = regime split of Δll_all; hard25 "
            "gate is secondary. Realised backtest (arbiter) still needs PriceHistory "
            "plumbing for any winning column."
        ),
        "total_wall_seconds": time.perf_counter() - overall_t0,
    }
    write_meta(OUT, meta)
    print(f"[total wall] {time.perf_counter() - overall_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
