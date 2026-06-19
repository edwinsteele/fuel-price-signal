"""Cycle-zone helpers shared across realised-fill ledger experiments."""
from __future__ import annotations

import math

import pandas as pd

# Three-band cut on cycle_pct_through (= days_since_peak / mean_cycle_length).
CYCLE_REGIME_BANDS: list[tuple[str, float, float]] = [
    ("normal", 0.0, 0.6),
    ("late_descent", 0.6, 1.0),
    ("overdue", 1.0, math.inf),
]


def assign_regime(pct: float) -> str:
    """Map a cycle_pct_through value to its regime name.

    Returns 'unmatched' for NaN (fill date had no prior feature row via as-of join).
    Falls back to 'normal' for any value that escapes the band table.
    """
    if pd.isna(pct):
        return "unmatched"
    for name, lo, hi in CYCLE_REGIME_BANDS:
        if lo <= pct < hi:
            return name
    return "normal"


def pooled_cpl(fills: pd.DataFrame) -> float:
    """Pooled cost-per-litre from a fill ledger (spend_cents / litres).

    Returns NaN when the ledger has zero litres — avoids a silent divide-by-zero.
    """
    litres = fills["litres"].sum()
    return fills["spend_cents"].sum() / litres if litres > 0 else float("nan")
