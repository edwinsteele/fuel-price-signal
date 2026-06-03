"""Step 2 — paired walk-forward CV: baseline vs drop_minus_max.

Builds two joblib artifacts in this dir (gitignored), then invokes
fuel_signal.cv_report.run_paired_cv across the standard walk-forward folds.

Both artifacts share the same pipeline (LightGBM, seed 42) — the only
difference is `feature_columns`. cv_report clones the pipeline and re-fits
each fold using the artifact's `feature_columns`, so we don't carry over any
training-set contamination across splits.
"""

from __future__ import annotations

import pathlib

import joblib
import numpy as np
import pandas as pd

from fuel_signal.cv_report import run_paired_cv
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS
from fuel_signal.train_lgbm import build_pipeline

HERE = pathlib.Path(__file__).parent
FEATURES_CSV = pathlib.Path("data/features.csv")
SEED = 42
DROP_MINUS_MAX = "station_minus_last_max_cents"

BASELINE_ARTIFACT = HERE / "baseline.joblib"
DROP_ARTIFACT = HERE / "drop_minus_max.joblib"


def _save_artifact(path: pathlib.Path, feature_columns: list[str]) -> None:
    """Save an unfitted pipeline + feature_columns. cv_report.run_paired_cv
    clones+fits per fold, so the pipeline doesn't need to be pre-fit."""
    joblib.dump(
        {"pipeline": build_pipeline(random_state=SEED), "feature_columns": feature_columns},
        path,
    )


def main() -> None:
    full_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    drop_cols = [c for c in full_cols if c != DROP_MINUS_MAX]
    assert len(drop_cols) == len(full_cols) - 1

    _save_artifact(BASELINE_ARTIFACT, full_cols)
    _save_artifact(DROP_ARTIFACT, drop_cols)
    print(f"Wrote {BASELINE_ARTIFACT} ({len(full_cols)} features)")
    print(f"Wrote {DROP_ARTIFACT} ({len(drop_cols)} features)")

    print(f"\nLoading {FEATURES_CSV} ...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df):,} rows")

    print("\nRunning paired walk-forward CV (model=drop_minus_max, baseline=full)...")
    results = run_paired_cv(
        df,
        model_path=DROP_ARTIFACT,
        baseline_path=BASELINE_ARTIFACT,
        seed=SEED,
    )

    out = pd.DataFrame(results)
    out.to_csv(HERE / "step2_cv_results.csv", index=False)

    print("\n" + "=" * 84)
    print("PER-FOLD RESULTS  (Δ = drop_minus_max − baseline; negative = drop helps)")
    print("=" * 84)
    for r in results:
        print(
            f"fold {r['fold_idx']:>3}  "
            f"val {r['val_start']}→{r['val_end']}  "
            f"n={r['n_val']:>6,}  "
            f"baseline={r['baseline_logloss']:.4f}  "
            f"model={r['model_logloss']:.4f}  "
            f"Δ={r['delta']:+.4f}"
        )

    deltas = np.array([r["delta"] for r in results])
    n_wins = int((deltas < 0).sum())
    regressions = [r for r in results if r["delta"] > 0.05]
    print("─" * 84)
    print(
        f"folds: {len(results)}  wins (Δ<0): {n_wins}/{len(results)}  "
        f"median Δ={np.median(deltas):+.4f}  mean Δ={deltas.mean():+.4f}  "
        f"std Δ={deltas.std(ddof=1):.4f}"
    )
    if regressions:
        names = ", ".join(
            f"fold {r['fold_idx']} (Δ={r['delta']:+.4f})" for r in regressions
        )
        print(f"regressions (Δ>+0.05): {names}")
    print(f"\nSaved {HERE / 'step2_cv_results.csv'}")


if __name__ == "__main__":
    main()
