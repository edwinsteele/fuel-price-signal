from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev

from .constants import LGBM_DEFAULTS


def fit_score(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cols: list[str],
    seed: int,
) -> tuple[float, np.ndarray, float]:
    t0 = time.perf_counter()
    model = LGBMClassifier(random_state=seed, **LGBM_DEFAULTS)
    model.fit(train_df[cols], train_df["label"].to_numpy(dtype=int))
    p = model.predict_proba(val_df[cols])[:, 1]
    ll = float(_ev.log_loss(val_df["label"].to_numpy(dtype=int), p))
    return ll, p, time.perf_counter() - t0


def per_row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
