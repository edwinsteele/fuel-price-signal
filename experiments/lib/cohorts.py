from __future__ import annotations

import numpy as np


def hard_quantile_mask(prl: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of rows in the hardest (1-q) fraction by per-row log-loss."""
    return prl >= float(np.quantile(prl, q))
