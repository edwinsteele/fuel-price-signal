"""Throwaway: bank a raw-test-logloss seed vector for the post-#144 Phase 4 lock.

Stopgap until issue #145 ships a proper --seeds flag + results.csv schema.
Refits raw (uncalibrated) 50-feat LightGBM at the standard seed set and prints
the per-seed test-logloss vector + mean/std, matching the metric used for the
lgbm_council_fix row so the two are directly comparable.

    uv run python experiments/seed_bank_phase4/run.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS

SEEDS = [1, 7, 42, 99, 2024]
FEATURES_CSV = "data/features.csv"


def main() -> None:
    cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    df = pd.read_csv(FEATURES_CSV)
    train, _val, test = _ev.split(df)

    X_train = train[cols].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    X_test = test[cols].to_numpy(dtype=float)
    y_test = test["label"].to_numpy(dtype=int)

    scores: list[float] = []
    for seed in SEEDS:
        model = LGBMClassifier(random_state=seed, verbose=-1)
        model.fit(X_train, y_train)
        p_test = model.predict_proba(X_test)[:, 1]
        ll = float(_ev.log_loss(y_test, p_test))
        scores.append(ll)
        print(f"  seed {seed:>4} : raw test logloss {ll:.4f}")

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    print(f"\n  n_features            : {len(cols)}")
    print(f"  seed5_raw_test_logloss: {[round(s, 4) for s in scores]}@seeds{SEEDS}")
    print(f"  seed5_raw_mean        : {mean:.4f}")
    print(f"  seed5_raw_std         : {std:.4f}  (3sigma={3 * std:.4f})")


if __name__ == "__main__":
    main()
