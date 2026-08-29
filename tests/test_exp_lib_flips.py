"""Tests for experiments/lib/flips.py — diff_fills, summarise_flips (fps-gez, fps-e1w)."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments.lib.flips import (
    MIN_FOLD_LOCAL_DISPERSION_N,
    cascade_window_days,
    diff_fills,
    summarise_flips,
)

BASELINE = "R0"
CANDIDATE = "candidate"

# This run's own tank_params ("50/3.571/1d/10%", the fps-6yi batch) derives a 7-day cascade
# window — see cascade_window_days's docstring. Used as the default across these tests so a
# single station's flips more than a week apart are never accidentally collapsed together.
WINDOW = cascade_window_days("50/3.571/1d/10%")


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


# ── cascade_window_days (fps-e1w change 3) ──────────────────────────────────────

def test_cascade_window_days_derives_half_tank_life_not_hardcoded():
    # tank_size_litres / daily_consumption_litres / 2, rounded — the run's own fps-6yi
    # tank_params (50 / (50/14)) gives exactly 7, the value the post-dossier review used
    # by hand. Derived here from the run's own parameters, not copied as a literal.
    assert cascade_window_days("50/3.571/1d/10%") == 7


def test_cascade_window_days_scales_with_a_different_tank():
    # A 20L tank at 2L/day empties in 10 days -> half-life window of 5.
    assert cascade_window_days("20/2.000/1d/10%") == 5


# ── summarise_flips: baseline shape ─────────────────────────────────────────────

def test_summarise_flips_zero_flips():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, CANDIDATE, 100, "2026-01-01", 150.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=WINDOW)
    assert summary["n_flips"] == 0
    assert summary["cascade_window_days"] == WINDOW
    assert summary["rows"] == []
    row = summary["per_fold"][0]
    assert row["fold"] == 1
    assert row["regime"] == "normal"
    assert row["n_baseline_only"] == 0
    assert row["n_candidate_only"] == 0
    assert row["n_flips"] == 0
    assert row["n_decisions"] == 0
    assert row["litres_baseline"] == 0.0
    assert row["litres_candidate"] == 0.0
    assert row["flip_cpl_baseline"] is None
    assert row["flip_cpl_candidate"] is None
    assert row["flip_cpl_delta"] is None
    assert row["flip_cpl_delta_reason"] == "no flips"
    assert row["flip_cpl_delta_se"] is None
    assert row["flip_cpl_delta_interval"] is None
    assert row["flip_cpl_delta_inside_own_se"] is None


def test_summarise_flips_one_arm_only_has_its_own_reason():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, CANDIDATE, 100, "2026-01-01", 150.0),  # matches, not a flip
        _fill(1, CANDIDATE, 200, "2026-01-05", 140.0),  # candidate-only flip
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=WINDOW)
    row = summary["per_fold"][0]
    assert row["n_baseline_only"] == 0
    assert row["n_candidate_only"] == 1
    assert row["flip_cpl_delta"] is None
    assert row["flip_cpl_delta_reason"] == "one arm only"
    assert row["litres_candidate"] == pytest.approx(10.0)
    assert row["litres_baseline"] == 0.0


def test_summarise_flips_computes_flip_only_cpl_and_delta():
    fills = pd.DataFrame([
        # baseline buys expensive, candidate buys the same litres cheaper — a real
        # flip-only edge, distinct from any other fill in the fold.
        _fill(1, BASELINE, 100, "2026-01-01", price=200.0, litres=10.0),
        _fill(1, CANDIDATE, 100, "2026-01-02", price=150.0, litres=10.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds={1}, cascade_window_days=WINDOW)
    row = summary["per_fold"][0]
    assert row["regime"] == "shock"
    assert row["n_baseline_only"] == 1
    assert row["n_candidate_only"] == 1
    assert row["flip_cpl_baseline"] == pytest.approx(200.0)
    assert row["flip_cpl_candidate"] == pytest.approx(150.0)
    assert row["flip_cpl_delta"] == pytest.approx(-50.0)
    assert row["flip_cpl_delta_reason"] is None
    assert summary["n_flips"] == 2
    # Below MIN_FOLD_LOCAL_DISPERSION_N (only 2 flip fills) -> falls back to run-wide, which
    # here is the same 2 rows, so dispersion is still estimable (prices 200 and 150 differ).
    assert row["flip_cpl_delta_se"] is not None
    assert row["flip_cpl_delta_interval"] is not None
    lo, hi = row["flip_cpl_delta_interval"]
    assert lo < row["flip_cpl_delta"] < hi


def test_summarise_flips_per_fold_covers_every_fold_present_in_fills():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, CANDIDATE, 100, "2026-01-01", 150.0),  # fold 1: no flips
        _fill(2, BASELINE, 100, "2026-01-08", 150.0),
        _fill(2, CANDIDATE, 100, "2026-01-09", 140.0),  # fold 2: one flip each side
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=WINDOW)
    folds = {row["fold"]: row for row in summary["per_fold"]}
    assert folds[1]["n_baseline_only"] == 0
    assert folds[2]["n_baseline_only"] == 1
    assert folds[2]["n_candidate_only"] == 1


# ── n_decisions: cascade collapse (fps-e1w change 3) ────────────────────────────

def test_cascade_collapse_groups_same_station_flips_within_window():
    # Four flips at the same station, each one day apart -> well within a 7-day window ->
    # one cascade, one decision, even though n_flips is 4.
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, f"2026-01-0{d}", 150.0 + d) for d in range(1, 5)
    ] + [
        _fill(1, CANDIDATE, 100, "2026-01-01", 150.0),  # matches the first baseline fill
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=7)
    row = summary["per_fold"][0]
    assert row["n_flips"] == 3  # baseline's 01-02, 01-03, 01-04 (01-01 matched)
    assert row["n_decisions"] == 1


def test_cascade_collapse_gap_beyond_window_starts_a_new_decision():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, BASELINE, 100, "2026-01-20", 155.0),  # 19 days later — a new decision
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=7)
    row = summary["per_fold"][0]
    assert row["n_flips"] == 2
    assert row["n_decisions"] == 2


def test_cascade_collapse_is_per_station_not_per_fold():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, BASELINE, 200, "2026-01-01", 150.0),  # same date, different station
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=7)
    row = summary["per_fold"][0]
    assert row["n_flips"] == 2
    assert row["n_decisions"] == 2


def test_cascade_collapse_spans_both_arms_at_one_station():
    # A baseline flip followed three days later by a candidate flip at the SAME station is one
    # shared timeline of divergence, not two independent decisions.
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, CANDIDATE, 100, "2026-01-04", 145.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=7)
    row = summary["per_fold"][0]
    assert row["n_flips"] == 2
    assert row["n_decisions"] == 1


# ── SE / interval and the degeneracy guard (fps-e1w change 2 / design note) ─────

def test_flip_cpl_delta_interval_widens_with_thin_effective_n():
    # A fold with a large price spread and small litres-weighted effective n per arm should
    # print a wide interval that contains the delta.
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", price=200.0, litres=10.0),
        _fill(1, BASELINE, 101, "2026-01-02", price=210.0, litres=10.0),
        _fill(1, BASELINE, 102, "2026-01-03", price=190.0, litres=10.0),
        _fill(1, BASELINE, 103, "2026-01-04", price=220.0, litres=10.0),
        _fill(1, CANDIDATE, 200, "2026-01-05", price=140.0, litres=10.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=7)
    row = summary["per_fold"][0]
    assert row["flip_cpl_delta"] is not None
    lo, hi = row["flip_cpl_delta_interval"]
    assert lo < row["flip_cpl_delta"] < hi
    assert row["flip_cpl_delta_inside_own_se"] is True  # thin single-fill candidate side


def test_degenerate_price_dispersion_prints_no_interval_not_a_hairline_one():
    # Every flip fill in this fold (and in the whole run) lands at the identical price —
    # np.std would return a tiny non-zero float (binary rounding noise, fps-tnz's failure
    # mode), not exact 0.0, which a bare `> 0` check would wrongly treat as a real, tight
    # ruler. The fix must reject it via np.ptp and print no interval, not a hairline one.
    price = 184.9
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", price=price, litres=10.0),
        _fill(1, CANDIDATE, 100, "2026-01-02", price=price, litres=10.0),
        _fill(2, BASELINE, 200, "2026-02-01", price=price, litres=5.0),
        _fill(2, CANDIDATE, 200, "2026-02-02", price=price, litres=5.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=7)
    for row in summary["per_fold"]:
        assert row["flip_cpl_delta"] == pytest.approx(0.0)
        assert row["flip_cpl_delta_se"] is None
        assert row["flip_cpl_delta_interval"] is None
        assert row["flip_cpl_delta_inside_own_se"] is None
        assert "not estimable" in row["flip_cpl_delta_interval_reason"]


def test_degenerate_fold_local_dispersion_falls_back_to_usable_run_wide():
    # Fold 1's flip prices are all identical (would degenerate fold-locally) but fold 2 has
    # real spread, so run-wide dispersion IS usable and fold 1 must fall back to it rather
    # than printing no interval at all — degeneracy falls through by VALUE, not by count.
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", price=150.0, litres=10.0),
        _fill(1, BASELINE, 101, "2026-01-02", price=150.0, litres=10.0),
        _fill(1, BASELINE, 102, "2026-01-03", price=150.0, litres=10.0),
        _fill(1, BASELINE, 103, "2026-01-04", price=150.0, litres=10.0),
        _fill(1, CANDIDATE, 200, "2026-01-05", price=140.0, litres=10.0),
        _fill(2, BASELINE, 300, "2026-02-01", price=200.0, litres=10.0),
        _fill(2, CANDIDATE, 300, "2026-02-02", price=120.0, litres=10.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), cascade_window_days=7)
    fold1 = next(r for r in summary["per_fold"] if r["fold"] == 1)
    assert fold1["flip_cpl_delta"] is not None
    assert fold1["flip_cpl_delta_se"] is not None
    assert fold1["flip_cpl_delta_interval"] is not None


def test_min_fold_local_dispersion_n_is_the_documented_threshold():
    assert MIN_FOLD_LOCAL_DISPERSION_N == 4
