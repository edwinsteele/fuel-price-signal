"""Step 5: paired walk-forward CV restricted to R0 + R5 (A only), saving
per-(fold, run, seed, row) predicted probabilities for row-level diagnostic.

Goal: enable row-level test of the "broad-population signal" hypothesis —
within each fold, do A's per-row prediction errors concentrate on extended-
descent rows? If yes across many folds, the regression on fold 7 is just the
loudest exemplar of a population-wide failure mode and constraint design is
justified.

Mirrors step2_paired_wfcv.py's harness exactly (same SEEDS, same walk-forward
folds, same A feature definition, same hard25/hard10 mask derivation) so the
ll_all / ll_hard25 numbers reproduce step2_runs.csv for R0 + R5.

Outputs:
- step5_rowpreds.parquet: (fold, run, seed, station_code, price_date, label,
  proba, is_hard25, is_hard10). One row per (fold, run, seed, val_row).
- step5_fold_meta.csv: fold-level summary (val window, n rows, hard25/hard10
  thresholds, n_lated, ll_all per run × seed).

Usage:
    PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step5_paired_wfcv_rowpreds.py
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, load_features

OUT = pathlib.Path(__file__).parent
SEEDS = (42, 43, 44, 45, 46)
SHOCK_FOLDS = frozenset({1, 4, 9, 13})

# Signal A only (matches step2_paired_wfcv.py).
A_LEVEL = "network_px_std"
A_DELTA = "network_px_std_delta_3d"
A_COLS = [A_LEVEL, A_DELTA]
COMP_BAND_CENTS = 5.0
DELTA_LAG_DAYS = 3


def compute_signal_a(df: pd.DataFrame) -> pd.DataFrame:
    """Compute network_px_std + delta_3d. Matches step2_paired_wfcv.py § A."""
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])
    comp = df[df["stickiness_score"].abs() <= COMP_BAND_CENTS]
    a_by_date = comp.groupby("price_date")["station_price_cents"].std().rename(A_LEVEL)
    df = df.join(a_by_date, on="price_date")
    per_date_level = df.drop_duplicates("price_date").set_index("price_date")[A_LEVEL].sort_index()
    full_idx = pd.date_range(per_date_level.index.min(), per_date_level.index.max(), freq="D")
    s = per_date_level.reindex(full_idx)
    delta = (s - s.shift(DELTA_LAG_DAYS)).rename(A_DELTA)
    df = df.join(delta, on="price_date")
    return df


def fit_score(train_df: pd.DataFrame, val_df: pd.DataFrame,
              cols: list[str], seed: int) -> tuple[float, np.ndarray, float]:
    t0 = time.perf_counter()
    model = LGBMClassifier(random_state=seed, verbose=-1, subsample=0.8, subsample_freq=1)
    model.fit(train_df[cols], train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols])[:, 1]
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, p, time.perf_counter() - t0


def per_row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading features ...")
    t0 = time.perf_counter()
    df = load_features()
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}")

    print("Computing Signal A ...")
    t0 = time.perf_counter()
    df = compute_signal_a(df)
    print(f"  [compute_A] {time.perf_counter() - t0:.1f}s")
    for c in A_COLS:
        print(f"    {c}: nulls = {df[c].isna().mean()*100:.2f}%")

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    assert len(baseline_cols) == 50, f"expected 50, got {len(baseline_cols)}"
    runs: dict[str, list[str]] = {
        "R0_baseline": baseline_cols,
        "R5_A_only": baseline_cols + A_COLS,
    }
    print(f"\nRuns: {list(runs.keys())}  Seeds: {SEEDS}")

    folds = list(_ev.walk_forward_folds(df, train_min_days=1825, val_days=90, step_days=90))
    print(f"Walk-forward folds: {len(folds)}\n")

    pred_rows: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    print(f"{'fold':>4}  {'regime':>6}  {'val_start':>10}  {'val_end':>10}  "
          f"{'run':<12}  {'seed':>4}  {'ll_all':>7}  {'ll_h25':>7}  {'fit_s':>5}")
    print("-" * 100)

    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        vd = pd.to_datetime(val_df["price_date"])
        val_start = vd.min().strftime("%Y-%m-%d")
        val_end = vd.max().strftime("%Y-%m-%d")
        regime = "shock" if i in SHOCK_FOLDS else "normal"
        y = val_df["label"].to_numpy(dtype=int)

        # Derive hard25 / hard10 masks from baseline seed-0 predictions
        # (matches step2_paired_wfcv.py exactly).
        ll0, p0, t0_fit = fit_score(train_df, val_df, baseline_cols, SEEDS[0])
        prl0 = per_row_log_loss(y, p0)
        hard25_thresh = float(np.quantile(prl0, 0.75))
        hard10_thresh = float(np.quantile(prl0, 0.90))
        is_hard25 = (prl0 >= hard25_thresh).astype(np.int8)
        is_hard10 = (prl0 >= hard10_thresh).astype(np.int8)

        # Identity columns for the val rows. price_date as date32 in parquet
        # via pandas datetime64[ns].
        ident = pd.DataFrame({
            "fold": np.int8(i),
            "station_code": val_df["station_code"].to_numpy(),
            "price_date": pd.to_datetime(val_df["price_date"]).to_numpy(),
            "label": y.astype(np.int8),
            "is_hard25": is_hard25,
            "is_hard10": is_hard10,
        })

        for run_name, cols in runs.items():
            for seed in SEEDS:
                if run_name == "R0_baseline" and seed == SEEDS[0]:
                    ll, p, fit_s = ll0, p0, t0_fit
                else:
                    ll, p, fit_s = fit_score(train_df, val_df, cols, seed)
                prl = per_row_log_loss(y, p)
                ll_h25 = float(prl[is_hard25.astype(bool)].mean())

                block = ident.copy()
                block["run"] = run_name
                block["seed"] = np.int8(seed)
                block["proba"] = p.astype(np.float32)
                pred_rows.append(block)

                meta_rows.append({
                    "fold": i, "regime": regime,
                    "val_start": val_start, "val_end": val_end,
                    "val_rows": len(val_df),
                    "run": run_name, "seed": seed,
                    "ll_all": ll, "ll_hard25": ll_h25,
                    "hard25_thresh": hard25_thresh,
                    "hard10_thresh": hard10_thresh,
                    "fit_s": fit_s,
                })
                print(f"{i:>4}  {regime:>6}  {val_start:>10}  {val_end:>10}  "
                      f"{run_name:<12}  {seed:>4}  {ll:>7.4f}  {ll_h25:>7.4f}  "
                      f"{fit_s:>5.1f}s", flush=True)

    print(f"\nWriting outputs ...")
    pred_df = pd.concat(pred_rows, ignore_index=True)
    out_parquet = OUT / "step5_rowpreds.parquet"
    pred_df.to_parquet(out_parquet, index=False, compression="zstd")
    print(f"  {out_parquet}  ({len(pred_df):,} rows)")

    meta_df = pd.DataFrame(meta_rows)
    out_meta = OUT / "step5_fold_meta.csv"
    meta_df.to_csv(out_meta, index=False)
    print(f"  {out_meta}  ({len(meta_df)} rows)")

    print(f"\nTotal wall: {time.perf_counter() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
