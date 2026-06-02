"""L1 experiment: paired walk-forward CV — Phase 4 (50 broadcast) vs L1 (16 per-row).

Question: does a single per-row `days_since_my_lga_trough` column (looked up at
feature-build time via stations.council) match the 29-of-35 broadcast LGA columns
the Phase 4 model uses?

Phase 4 baseline: FEATURE_COLUMNS + LGA_FEATURE_COLUMNS (50 feats, broadcast).
L1 candidate:    FEATURE_COLUMNS + [days_since_my_lga_trough] (16 feats, per-row).

Derives the per-row column from the existing features.csv: for each row, looks up
the broadcast value for that row's station-council. This is exact-equivalent to
computing days_since_my_lga_trough = compute_pit_strict_days_since_trough[
(date_d, stations.council[station_code])] — the broadcast column already stores
the same value, just replicated across every row on each date.

Single seed (42); 14-fold paired walk-forward CV mirroring cv_compare_phase4.
Reports per-fold deltas + median/mean Δ + named regressions.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import db as _db
from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS
from fuel_signal.lga_leadership import lga_slug

SEED = 42
PER_ROW_COL = "days_since_my_lga_trough"

FEATS_PHASE4 = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS  # 50 cols, broadcast
FEATS_L1 = FEATURE_COLUMNS + [PER_ROW_COL]            # 16 cols, per-row


def _build_pipeline() -> LGBMClassifier:
    # Mirrors fuel_signal.train_lgbm.build_pipeline for parity with the locked model.
    return LGBMClassifier(
        random_state=SEED,
        verbose=-1,
        subsample=0.8,
        subsample_freq=1,
    )


def _fit_score(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str]) -> float:
    model = _build_pipeline()
    model.fit(train_df[cols].to_numpy(dtype=float), train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols].to_numpy(dtype=float))[:, 1]
    return float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))


def derive_per_row_column(df: pd.DataFrame, conn) -> pd.DataFrame:
    """Add days_since_my_lga_trough to df by picking each row's broadcast column.

    Rows whose station has no council (or a council not in LGA_FEATURE_COLUMNS)
    get NaN, matching the broadcast semantics.
    """
    council_by_station: dict[int, str | None] = {
        sc: council for sc, council in conn.execute(
            "SELECT station_code, council FROM stations"
        )
    }
    council_series = df["station_code"].map(council_by_station)
    df = df.copy()
    df[PER_ROW_COL] = np.nan
    for council, idx in df.groupby(council_series, dropna=True).groups.items():
        col = f"days_since_trough_entry_{lga_slug(council)}"
        if col in df.columns:
            df.loc[idx, PER_ROW_COL] = df.loc[idx, col].to_numpy()
    return df


def main() -> None:
    print(f"Phase 4 features: {len(FEATS_PHASE4)} cols (15 base + {len(LGA_FEATURE_COLUMNS)} broadcast LGA)")
    print(f"L1 features:      {len(FEATS_L1)} cols (15 base + 1 per-row)\n")

    df = pd.read_csv("data/features.csv")
    print(f"Loaded {len(df):,} rows from data/features.csv")

    conn = _db.open_db(_db.DEFAULT_DB_PATH)
    df = derive_per_row_column(df, conn)
    conn.close()

    n_nan = int(df[PER_ROW_COL].isna().sum())
    n_total = len(df)
    print(f"days_since_my_lga_trough: {n_total - n_nan:,} non-NaN / {n_total:,} rows "
          f"({(n_total - n_nan) / n_total * 100:.1f}%)\n")

    folds = list(_ev.walk_forward_folds(df, train_min_days=1825, val_days=90, step_days=90))
    print(f"Walk-forward folds: {len(folds)}\n")

    print(f"{'fold':>4}  {'val_start':>10}  {'val_end':>10}  "
          f"{'val_rows':>8}  {'BUY%':>5}  "
          f"{'ll_p4':>7}  {'ll_l1':>7}  {'Δ (L1-P4)':>9}")
    print("-" * 88)

    rows = []
    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        ll_p4 = _fit_score(train_df, val_df, FEATS_PHASE4)
        ll_l1 = _fit_score(train_df, val_df, FEATS_L1)
        delta = ll_l1 - ll_p4
        vd = pd.to_datetime(val_df["price_date"])
        row = {
            "fold": i,
            "val_start": vd.min().strftime("%Y-%m-%d"),
            "val_end":   vd.max().strftime("%Y-%m-%d"),
            "val_rows":  len(val_df),
            "buy_rate":  float(val_df["label"].mean()),
            "ll_phase4": ll_p4,
            "ll_l1":     ll_l1,
            "delta":     delta,
        }
        rows.append(row)
        print(f"{row['fold']:>4}  {row['val_start']:>10}  {row['val_end']:>10}  "
              f"{row['val_rows']:>8,}  {row['buy_rate']*100:>4.1f}%  "
              f"{ll_p4:>7.4f}  {ll_l1:>7.4f}  {delta:>+9.4f}")

    if not rows:
        print("No folds produced.")
        return

    deltas = np.array([r["delta"] for r in rows])
    n_helps = int((deltas < 0).sum())   # L1 better
    n_hurts = int((deltas > 0).sum())   # L1 worse
    print("-" * 88)
    print(f"folds: {len(rows)}    Phase 4 mean: {np.mean([r['ll_phase4'] for r in rows]):.4f}    "
          f"L1 mean: {np.mean([r['ll_l1'] for r in rows]):.4f}")
    print(f"Δ (L1 − P4): mean {deltas.mean():+.4f}  median {np.median(deltas):+.4f}  "
          f"std {deltas.std(ddof=1):.4f}  min {deltas.min():+.4f}  max {deltas.max():+.4f}")
    print(f"Folds where L1 is BETTER (Δ<0): {n_helps}/{len(rows)}  "
          f"WORSE (Δ>0): {n_hurts}/{len(rows)}")

    # Name folds that regress beyond the seed-std band (~0.0075 from
    # feedback_seed_discipline; 3σ ≈ 0.0225). Mirrors the threshold cv_report uses.
    regressions = [r for r in rows if r["delta"] > 0.0225]
    if regressions:
        print("\nRegressions (Δ > +0.0225, ~3σ of seed std):")
        for r in regressions:
            print(f"  fold {r['fold']} ({r['val_start']}→{r['val_end']}): "
                  f"Δ={r['delta']:+.4f}  P4={r['ll_phase4']:.4f}  L1={r['ll_l1']:.4f}")

    rescues = [r for r in rows if r["delta"] < -0.0225]
    if rescues:
        print("\nRescues (Δ < -0.0225, L1 substantially better):")
        for r in rescues:
            print(f"  fold {r['fold']} ({r['val_start']}→{r['val_end']}): "
                  f"Δ={r['delta']:+.4f}  P4={r['ll_phase4']:.4f}  L1={r['ll_l1']:.4f}")

    out = pathlib.Path(__file__).parent / "results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
