"""Tests for experiments/lib/zones — CYCLE_REGIME_BANDS, assign_regime, pooled_cpl."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from experiments.lib.zones import CYCLE_REGIME_BANDS, assign_regime, pooled_cpl

# ---------------------------------------------------------------------------
# assign_regime — band edges and special cases
# ---------------------------------------------------------------------------

def test_assign_regime_normal_interior():
    assert assign_regime(0.3) == "normal"


def test_assign_regime_normal_at_zero():
    assert assign_regime(0.0) == "normal"


def test_assign_regime_late_descent_at_lower_edge():
    assert assign_regime(0.6) == "late_descent"


def test_assign_regime_late_descent_interior():
    assert assign_regime(0.8) == "late_descent"


def test_assign_regime_overdue_at_lower_edge():
    # >= 1.0 → overdue tail
    assert assign_regime(1.0) == "overdue"


def test_assign_regime_overdue_beyond():
    assert assign_regime(2.5) == "overdue"


def test_assign_regime_nan_returns_unmatched():
    assert assign_regime(float("nan")) == "unmatched"


# ---------------------------------------------------------------------------
# assign_regime — CYCLE_REGIME_BANDS constant is the source of truth
# ---------------------------------------------------------------------------

def test_cycle_regime_bands_overdue_hi_is_inf():
    _, _, hi = CYCLE_REGIME_BANDS[-1]
    assert hi == math.inf


# ---------------------------------------------------------------------------
# pooled_cpl — normal and zero-litres cases
# ---------------------------------------------------------------------------

def _fills(spend: list[float], litres: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"spend_cents": spend, "litres": litres})


def test_pooled_cpl_basic():
    df = _fills([200.0, 400.0], [2.0, 2.0])
    assert pooled_cpl(df) == pytest.approx(150.0)


def test_pooled_cpl_zero_litres_returns_nan():
    df = _fills([100.0], [0.0])
    result = pooled_cpl(df)
    assert math.isnan(result)


def test_pooled_cpl_empty_frame_returns_nan():
    df = _fills([], [])
    result = pooled_cpl(df)
    assert math.isnan(result)
