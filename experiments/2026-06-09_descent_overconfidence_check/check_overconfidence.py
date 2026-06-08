"""Diagnostic: predicted vs actual BUY rate by descent-stage bucket.

Compares the 50-feat Phase 4 model (lgbm_phase4.joblib) against the new
54-feat RAC_full model (lgbm.joblib) to see whether late-descent
overconfidence changed after adding network_px_std and friends.

Buckets derived from existing feature columns (no new DB queries needed):
  normal_descent  : cycle_days_since_peak / cycle_mean_length <= 1.0
  ext_descent     : ratio > 1.0
  ext_shallow     : ext_descent AND descent_slope >= SHALLOW_THRESHOLD c/day
  ext_steep       : ext_descent AND descent_slope <  SHALLOW_THRESHOLD c/day

descent_slope = (station_price_cents - cycle_last_max_cents) / cycle_days_since_peak
  (negative for descent; less negative = shallower)

Usage:
    PYTHONPATH=. uv run python experiments/2026-06-09_descent_overconfidence_check/check_overconfidence.py
"""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from fuel_signal.features import load_features

MODELS = {
    "50-feat (phase4)": "data/models/lgbm_phase4.joblib",
    "54-feat (RAC_full)": "data/models/lgbm.joblib",
}
SHALLOW_THRESHOLD = -0.9  # c/day; >= this → shallow descent

# ---------------------------------------------------------------------------
t0 = time.time()

print("Loading features…")
df = load_features()
print(f"  {len(df):,} rows  pos_rate={df['label'].mean():.3f}  [{time.time()-t0:.1f}s]")

# Descent-stage flags (computed once from existing columns)
valid = df["cycle_days_since_peak"] > 0
df["elongation_ratio"] = np.where(
    valid,
    df["cycle_days_since_peak"] / df["cycle_mean_length"],
    np.nan,
)
df["descent_slope"] = np.where(
    valid,
    (df["station_price_cents"] - df["cycle_last_max_cents"]) / df["cycle_days_since_peak"],
    np.nan,
)

ext     = df["elongation_ratio"] > 1.0
normal  = valid & ~ext
shallow = ext & (df["descent_slope"] >= SHALLOW_THRESHOLD)
steep   = ext & (df["descent_slope"] <  SHALLOW_THRESHOLD)

buckets: list[tuple[str, pd.Series]] = [
    ("all_rows",       pd.Series([True] * len(df), index=df.index)),
    ("normal_descent", normal),
    ("ext_descent",    ext),
    ("ext_shallow",    shallow),
    ("ext_steep",      steep),
]

# ---------------------------------------------------------------------------

def _header() -> None:
    print(f"\n  {'bucket':<20}  {'n':>8}  {'actual':>6}  {'pred':>6}  {'gap':>6}  {'|gap|':>6}")
    print("  " + "-" * 62)


def _row(label: str, mask: pd.Series, pred_col: str) -> None:
    sub = df[mask]
    if len(sub) == 0:
        print(f"  {label:<20}  {'0':>8}")
        return
    actual   = sub["label"].mean()
    pred     = sub[pred_col].mean()
    gap      = pred - actual
    print(
        f"  {label:<20}  {len(sub):>8,}  {actual:>6.3f}  {pred:>6.3f}  {gap:>+6.3f}  {abs(gap):>6.3f}"
    )


# ---------------------------------------------------------------------------
for model_label, model_path in MODELS.items():
    print(f"\n{'='*65}")
    print(f"Model: {model_label}  ({model_path})")
    print(f"{'='*65}")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feat_cols = artifact["feature_columns"]

    t1 = time.time()
    X = df[feat_cols]          # DataFrame — preserves feature names for LightGBM
    pred_col = f"pred_{model_label.split()[0]}"
    df[pred_col] = pipeline.predict_proba(X)[:, 1]
    print(f"  Scored {len(df):,} rows in {time.time()-t1:.1f}s")

    _header()
    for bucket_label, mask in buckets:
        _row(bucket_label, mask, pred_col)

# ---------------------------------------------------------------------------
# Delta table: gap(54-feat) - gap(50-feat) per bucket
print(f"\n{'='*65}")
print("Gap delta: 54-feat minus 50-feat  (negative = less overconfident)")
print(f"{'='*65}")
print(f"\n  {'bucket':<20}  {'n':>8}  {'gap_50':>7}  {'gap_54':>7}  {'Δgap':>7}")
print("  " + "-" * 55)

col50 = "pred_50-feat"
col54 = "pred_54-feat"
for bucket_label, mask in buckets:
    sub = df[mask]
    if len(sub) == 0:
        continue
    actual  = sub["label"].mean()
    gap50   = sub[col50].mean() - actual
    gap54   = sub[col54].mean() - actual
    delta   = gap54 - gap50
    print(
        f"  {bucket_label:<20}  {len(sub):>8,}  {gap50:>+7.3f}  {gap54:>+7.3f}  {delta:>+7.3f}"
    )

print(f"\nTotal wall time: {time.time()-t0:.1f}s")
