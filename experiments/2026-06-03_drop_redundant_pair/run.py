"""Drop-one ablation: station_price_cents vs station_minus_last_max_cents.

Multi-seed val-only ablation (no walk-forward CV yet). Trains the 50-feat
Phase 4 LGBM in three configurations across SEEDS seeds each:
  - baseline: full 50 features
  - drop_price: drop `station_price_cents`
  - drop_minus_max: drop `station_minus_last_max_cents`

Reports per-(config, seed) val logloss + brier, plus per-config mean ± std and
Δ vs baseline mean. Goal: see whether the single-seed Δ from the v1 run
exceeds 3× cross-seed std on this val window.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
)
from fuel_signal.train_lgbm import train_and_evaluate

HERE = pathlib.Path(__file__).parent
FEATURES_CSV = pathlib.Path("data/features.csv")
SEEDS = [0, 1, 2, 3, 42]

DROP_PRICE = "station_price_cents"
DROP_MINUS_MAX = "station_minus_last_max_cents"


def main() -> None:
    full_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    assert DROP_PRICE in full_cols, DROP_PRICE
    assert DROP_MINUS_MAX in full_cols, DROP_MINUS_MAX

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features.csv: {len(df):,} rows, {len(full_cols)} features")

    configs = {
        "baseline": full_cols,
        "drop_price": [c for c in full_cols if c != DROP_PRICE],
        "drop_minus_max": [c for c in full_cols if c != DROP_MINUS_MAX],
    }

    rows = []
    for name, cols in configs.items():
        for seed in SEEDS:
            print(f"\n=== {name} (seed={seed}, {len(cols)} features) ===")
            res = train_and_evaluate(df, feature_columns=cols, random_state=seed)
            print(
                f"  train={res['train_size']:,}  val={res['val_size']:,}  "
                f"val_logloss={res['val_logloss']:.6f}  val_brier={res['val_brier']:.6f}"
            )
            rows.append({
                "config": name,
                "seed": seed,
                "n_features": len(cols),
                "train_size": res["train_size"],
                "val_size": res["val_size"],
                "val_logloss": res["val_logloss"],
                "val_brier": res["val_brier"],
            })

    raw = pd.DataFrame(rows)
    raw.to_csv(HERE / "results_per_seed.csv", index=False)

    summary = (
        raw.groupby("config")
        .agg(
            n_seeds=("seed", "count"),
            n_features=("n_features", "first"),
            val_logloss_mean=("val_logloss", "mean"),
            val_logloss_std=("val_logloss", "std"),
            val_brier_mean=("val_brier", "mean"),
            val_brier_std=("val_brier", "std"),
        )
        .reset_index()
    )

    base_ll_mean = float(summary.loc[summary["config"] == "baseline", "val_logloss_mean"].iloc[0])
    base_br_mean = float(summary.loc[summary["config"] == "baseline", "val_brier_mean"].iloc[0])
    summary["delta_logloss_vs_baseline_mean"] = summary["val_logloss_mean"] - base_ll_mean
    summary["delta_brier_vs_baseline_mean"] = summary["val_brier_mean"] - base_br_mean

    summary.to_csv(HERE / "results_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("PER-SEED RESULTS")
    print("=" * 78)
    print(raw.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n" + "=" * 78)
    print("SUMMARY (mean ± std across seeds " + str(SEEDS) + ")")
    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Discipline check: Δ vs baseline_mean compared to 3× pooled std across seeds.
    pooled_std = float(np.sqrt((raw.groupby("config")["val_logloss"].var()).mean()))
    print(f"\nPooled cross-seed std (val_logloss): {pooled_std:.6f}")
    print(f"Decision threshold (3×): {3 * pooled_std:.6f}")
    for _, r in summary.iterrows():
        if r["config"] == "baseline":
            continue
        d = float(r["delta_logloss_vs_baseline_mean"])
        verdict = "REAL" if abs(d) > 3 * pooled_std else "within noise"
        print(f"  {r['config']:<16} Δ={d:+.6f}  ({verdict})")

    (HERE / "params.json").write_text(json.dumps({
        "features_csv": str(FEATURES_CSV),
        "seeds": SEEDS,
        "n_features_full": len(full_cols),
        "drop_candidates": [DROP_PRICE, DROP_MINUS_MAX],
    }, indent=2))

    print(f"\nSaved {HERE / 'results_per_seed.csv'}")
    print(f"Saved {HERE / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
