"""Tests for experiments/lib/flips.py — diff_fills, summarise_flips (fps-gez)."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments.lib.flips import diff_fills, summarise_flips

BASELINE = "R0"
CANDIDATE = "candidate"


def _fill(fold, arm, station, date, price, litres=10.0):
    return {
        "fold": fold, "arm": arm, "own_tau": 0.25, "date": date,
        "station_code": station, "price": price, "litres": litres,
        "spend_cents": price * litres, "emergency": False,
    }


def test_diff_fills_no_divergence_is_empty():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, CANDIDATE, 100, "2026-01-01", 150.0),
    ])
    flips = diff_fills(fills, BASELINE, CANDIDATE)
    assert flips.empty
    assert list(flips.columns) == ["fold", "station_code", "date", "price", "litres", "spend_cents", "bought_by"]


def test_diff_fills_flags_baseline_only_and_candidate_only():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),   # baseline-only
        _fill(1, CANDIDATE, 100, "2026-01-02", 140.0),  # candidate-only
        _fill(1, BASELINE, 200, "2026-01-05", 160.0),   # matches — not a flip
        _fill(1, CANDIDATE, 200, "2026-01-05", 160.0),
    ])
    flips = diff_fills(fills, BASELINE, CANDIDATE)
    assert len(flips) == 2
    by_side = dict(zip(flips["bought_by"], flips["date"]))
    assert by_side["baseline"] == "2026-01-01"
    assert by_side["candidate"] == "2026-01-02"


def test_diff_fills_different_station_same_date_both_flip():
    # Same date, different station: two independent decisions, neither has a
    # counterpart on the other arm at that key, so BOTH are flips.
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, CANDIDATE, 200, "2026-01-01", 150.0),
    ])
    flips = diff_fills(fills, BASELINE, CANDIDATE)
    assert len(flips) == 2


def test_summarise_flips_zero_flips():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, CANDIDATE, 100, "2026-01-01", 150.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set())
    assert summary["n_flips"] == 0
    assert summary["rows"] == []
    assert summary["per_fold"] == [
        {
            "fold": 1, "regime": "normal", "n_baseline_only": 0, "n_candidate_only": 0,
            "flip_cpl_baseline": None, "flip_cpl_candidate": None, "flip_cpl_delta": None,
        }
    ]


def test_summarise_flips_computes_flip_only_cpl_and_delta():
    fills = pd.DataFrame([
        # baseline buys expensive, candidate buys the same litres cheaper — a real
        # flip-only edge, distinct from any other fill in the fold.
        _fill(1, BASELINE, 100, "2026-01-01", price=200.0, litres=10.0),
        _fill(1, CANDIDATE, 100, "2026-01-02", price=150.0, litres=10.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds={1})
    row = summary["per_fold"][0]
    assert row["regime"] == "shock"
    assert row["n_baseline_only"] == 1
    assert row["n_candidate_only"] == 1
    assert row["flip_cpl_baseline"] == pytest.approx(200.0)
    assert row["flip_cpl_candidate"] == pytest.approx(150.0)
    assert row["flip_cpl_delta"] == pytest.approx(-50.0)
    assert summary["n_flips"] == 2


def test_summarise_flips_per_fold_covers_every_fold_present_in_fills():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, CANDIDATE, 100, "2026-01-01", 150.0),  # fold 1: no flips
        _fill(2, BASELINE, 100, "2026-01-08", 150.0),
        _fill(2, CANDIDATE, 100, "2026-01-09", 140.0),  # fold 2: one flip each side
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set())
    folds = {row["fold"]: row for row in summary["per_fold"]}
    assert folds[1]["n_baseline_only"] == 0
    assert folds[2]["n_baseline_only"] == 1
    assert folds[2]["n_candidate_only"] == 1
