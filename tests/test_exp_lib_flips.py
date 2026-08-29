"""Tests for experiments/lib/flips.py — diff_fills, summarise_flips (fps-gez, fps-e1w),
summarise_regret (fps-2js)."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments.lib.flips import (
    MIN_FOLD_LOCAL_DISPERSION_N,
    cascade_window_days,
    diff_fills,
    parse_tank_params,
    regret_horizon_days,
    summarise_flips,
    summarise_regret,
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
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=WINDOW)
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
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=WINDOW)
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
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds={1}, window_days=WINDOW)
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
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=WINDOW)
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
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
    row = summary["per_fold"][0]
    assert row["n_flips"] == 3  # baseline's 01-02, 01-03, 01-04 (01-01 matched)
    assert row["n_decisions"] == 1


def test_cascade_collapse_gap_beyond_window_starts_a_new_decision():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, BASELINE, 100, "2026-01-20", 155.0),  # 19 days later — a new decision
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
    row = summary["per_fold"][0]
    assert row["n_flips"] == 2
    assert row["n_decisions"] == 2


def test_cascade_collapse_is_per_station_not_per_fold():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", 150.0),
        _fill(1, BASELINE, 200, "2026-01-01", 150.0),  # same date, different station
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
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
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
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
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
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
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
    for row in summary["per_fold"]:
        assert row["flip_cpl_delta"] == pytest.approx(0.0)
        assert row["flip_cpl_delta_se"] is None
        assert row["flip_cpl_delta_interval"] is None
        assert row["flip_cpl_delta_inside_own_se"] is None
        assert "not estimable" in row["flip_cpl_delta_interval_reason"]


def test_degenerate_fold_local_dispersion_falls_back_to_usable_run_wide():
    # Fold 1's flip prices are ALL identical, on BOTH arms (ptp=0 across the whole fold-local
    # scope _price_dispersion actually receives) -> genuinely degenerate fold-locally, at
    # n=5 >= MIN_FOLD_LOCAL_DISPERSION_N, so this can only pass via the run-wide fallback.
    # Fold 2 supplies the real spread that makes run-wide dispersion usable. (PR #347 review
    # finding #5: an earlier version of this test included a candidate row at a different
    # price WITHIN fold 1, so fold-local dispersion was never actually degenerate and the
    # fold-local branch silently satisfied the test on its own — deleting the run-wide
    # fallback entirely would not have failed it.)
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", price=150.0, litres=10.0),
        _fill(1, BASELINE, 101, "2026-01-02", price=150.0, litres=10.0),
        _fill(1, BASELINE, 102, "2026-01-03", price=150.0, litres=10.0),
        _fill(1, BASELINE, 103, "2026-01-04", price=150.0, litres=10.0),
        _fill(1, CANDIDATE, 104, "2026-01-05", price=150.0, litres=5.0),
        _fill(2, BASELINE, 300, "2026-02-01", price=200.0, litres=10.0),
        _fill(2, CANDIDATE, 300, "2026-02-02", price=120.0, litres=10.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
    fold1 = next(r for r in summary["per_fold"] if r["fold"] == 1)
    assert fold1["flip_cpl_delta"] == pytest.approx(0.0)
    assert fold1["flip_cpl_delta_se"] is not None
    assert fold1["flip_cpl_delta_interval"] is not None
    # The SE must actually be the RUN-WIDE spread (prices 150 x5, 200, 120 -> std(ddof=1) of
    # those 7 values), not a degenerate fold-local one — pin the exact number so a future
    # regression back to fold-local can't silently pass. Base side is 4 single-fill decision
    # groups (4 distinct stations, no cascading) -> n_eff=4; candidate side is 1 fill -> n_eff=1.
    import math

    import numpy as np

    run_wide_s = float(np.std([150.0, 150.0, 150.0, 150.0, 150.0, 200.0, 120.0], ddof=1))
    expected_se = math.sqrt((run_wide_s / math.sqrt(4)) ** 2 + (run_wide_s / math.sqrt(1)) ** 2)
    assert fold1["flip_cpl_delta_se"] == pytest.approx(expected_se)


def test_min_fold_local_dispersion_n_is_the_documented_threshold():
    assert MIN_FOLD_LOCAL_DISPERSION_N == 4


# ── decision-level SE, not fill-level (PR #347 review finding #1) ──────────────

def test_se_uses_decision_level_litres_not_per_fill_litres():
    # One arm's flip litres split across a 5-fill cascade at ONE station (all one decision);
    # the other arm's identical total litres as a single fill at a DIFFERENT station (also one
    # decision). Per-fill weighting would treat the cascaded side as 5 independent draws
    # (n_eff~5, a NARROW SE); decision-level weighting must treat it as 1 draw (n_eff=1, same
    # as the other side) since n_decisions says so.
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", price=200.0, litres=2.0),
        _fill(1, BASELINE, 100, "2026-01-02", price=205.0, litres=2.0),
        _fill(1, BASELINE, 100, "2026-01-03", price=195.0, litres=2.0),
        _fill(1, BASELINE, 100, "2026-01-04", price=210.0, litres=2.0),
        _fill(1, BASELINE, 100, "2026-01-05", price=190.0, litres=2.0),
        _fill(1, CANDIDATE, 200, "2026-01-01", price=140.0, litres=10.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
    row = summary["per_fold"][0]
    assert row["n_flips"] == 6
    assert row["n_decisions"] == 2  # one cascade (5 fills, 1 station) + one single fill

    # Recompute the fill-level (WRONG, pre-fix) SE by hand and confirm the actual SE is wider.
    import math

    prices = [200.0, 205.0, 195.0, 210.0, 190.0, 140.0]
    s = float(pd.Series(prices).std(ddof=1))
    fill_level_n_eff_base = (10.0 ** 2) / (5 * (2.0 ** 2))  # 5 equal-litres fills -> n_eff=5
    fill_level_se = math.sqrt((s / math.sqrt(fill_level_n_eff_base)) ** 2 + (s / math.sqrt(1.0)) ** 2)
    decision_level_se = row["flip_cpl_delta_se"]
    assert decision_level_se > fill_level_se
    # Decision-level: base side collapses to ONE decision (n_eff=1, same as candidate's own
    # single fill) -> SE = s * sqrt(2).
    assert decision_level_se == pytest.approx(s * math.sqrt(2))


def test_top_level_n_decisions_sums_the_per_fold_column():
    fills = pd.DataFrame([
        _fill(1, BASELINE, 100, "2026-01-01", price=200.0, litres=10.0),
        _fill(1, CANDIDATE, 100, "2026-01-02", price=150.0, litres=10.0),
        _fill(2, BASELINE, 200, "2026-02-01", price=200.0, litres=10.0),
        _fill(2, BASELINE, 200, "2026-02-02", price=195.0, litres=10.0),  # cascades w/ above
        _fill(2, CANDIDATE, 300, "2026-02-01", price=180.0, litres=10.0),
    ])
    summary = summarise_flips(fills, BASELINE, CANDIDATE, shock_folds=set(), window_days=7)
    assert summary["n_decisions"] == sum(row["n_decisions"] for row in summary["per_fold"])
    assert summary["n_decisions"] == 3  # fold1: 1 decision; fold2: 2 (cascade + single)


# ── cascade_window_days: floored above the run's own cadence (PR #347 review finding #2) ────

def test_cascade_window_days_floored_above_cadence_when_half_life_would_collide():
    # This project's own real batch0/tgp_delta_7d tank_params: half-life is exactly 7, and so
    # is the cadence -- an unguarded window of 7 would make EVERY on-grid gap (always exactly
    # the cadence for a station bought every eligible week) collapse without limit, and,
    # symmetrically, a half-life BELOW the cadence would make the window narrower than any
    # possible gap, permanently disabling cascade collapse for that run. Both are avoided by
    # flooring strictly above the cadence.
    assert cascade_window_days("50/3.571/7d/10%") == 8  # half-life 7, cadence+1 = 8


def test_cascade_window_days_floor_does_not_change_a_run_where_half_life_already_dominates():
    # fps-6yi's own tank_params: half-life (7) is already well above the 1-day cadence, so the
    # floor is a no-op here -- pinning this guards against the floor accidentally widening a
    # window that didn't need it.
    assert cascade_window_days("50/3.571/1d/10%") == 7


# ── fps-2js: timing regret ──────────────────────────────────────────────────────

class FakePrices:
    """A `flips.StationPriceSource` over a dict, reproducing PriceHistory.station_price_at's
    forward-fill: `price_at` returns the latest price on or before `as_of`, so a date the
    station never reported resolves to the carried one exactly as the tank simulator saw it."""

    def __init__(self, observed: dict[int, dict[str, float]]):
        self._observed = observed

    def price_at(self, station_code, as_of):
        series = self._observed.get(int(station_code)) or {}
        earlier = [d for d in series if d <= as_of]
        return series[max(earlier)] if earlier else None

    def is_observed(self, station_code, as_of):
        return as_of in (self._observed.get(int(station_code)) or {})


def _flip(fold, station, date, price, litres, bought_by):
    return {"fold": fold, "station_code": station, "date": date, "price": price,
            "litres": litres, "spend_cents": price * litres, "bought_by": bought_by}


def test_parse_tank_params_returns_floor_as_a_fraction_not_a_percentage():
    # The stamp writes "10%" but TankParams.floor_fraction is 0.10 — regret_horizon_days
    # multiplies by (1 - floor), so a percentage leaking through here would produce a
    # negative horizon rather than an obviously-wrong one.
    assert parse_tank_params("50/3.571/1d/10%") == (50.0, 3.571, 1, 0.10)
    assert parse_tank_params("60/4.000/7d/25%") == (60.0, 4.0, 7, 0.25)


def test_regret_horizon_is_the_tanks_own_feasible_wait():
    # fps-6yi's tank: 0.9 * 50 / 3.571 = 12.60 -> 13, one day inside the 14.0-day
    # full-to-empty life. Pinned because the whole measure's validity rests on the horizon
    # being a REACHABLE wait (see regret_horizon_days' sweep table).
    assert regret_horizon_days("50/3.571/1d/10%") == 13
    # A bigger floor leaves less usable range, so the horizon must SHRINK, never grow.
    assert regret_horizon_days("50/3.571/1d/50%") == 7
    # Independent of cadence — cadence sets the step within the window, not its length.
    assert regret_horizon_days("50/3.571/7d/10%") == 13


def test_summarise_regret_scores_price_paid_minus_best_reachable():
    flips = pd.DataFrame([
        _flip(1, 100, "2026-01-01", 180.0, 10.0, "baseline"),
        _flip(1, 100, "2026-01-20", 150.0, 10.0, "candidate"),
    ])
    prices = FakePrices({100: {
        "2026-01-01": 180.0, "2026-01-03": 160.0,   # baseline could have waited 2 days
        "2026-01-20": 150.0, "2026-01-22": 155.0,   # candidate already bought the low
    }})
    out = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1, window_days=7)
    assert out["all"]["regret_baseline"] == pytest.approx(20.0)
    assert out["all"]["regret_candidate"] == pytest.approx(0.0)
    assert out["all"]["regret_delta"] == pytest.approx(-20.0)
    assert out["n_scored"] == 2 and out["n_unscored"] == 0


def test_summarise_regret_horizon_bounds_what_counts_as_reachable():
    """A cheaper price beyond the tank's reach must NOT be charged as regret — the whole
    point of deriving the horizon from the tank rather than from cycle length."""
    flips = pd.DataFrame([_flip(1, 100, "2026-01-01", 180.0, 10.0, "baseline")])
    prices = FakePrices({100: {"2026-01-01": 180.0, "2026-01-20": 120.0}})
    inside = summarise_regret(flips, prices, set(), horizon_days=30, cadence_days=1, window_days=7)
    outside = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1, window_days=7)
    assert inside["all"]["regret_baseline"] == pytest.approx(60.0)
    assert outside["all"]["regret_baseline"] == pytest.approx(0.0)


def test_summarise_regret_walks_the_runs_own_cadence_grid():
    """At a 7-day cadence only every 7th day is a decision point, so a low price on an
    off-grid day was never actually buyable and must not register as regret."""
    flips = pd.DataFrame([_flip(1, 100, "2026-01-01", 180.0, 10.0, "baseline")])
    prices = FakePrices({100: {"2026-01-01": 180.0, "2026-01-03": 100.0, "2026-01-08": 170.0}})
    daily = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1, window_days=8)
    weekly = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=7, window_days=8)
    assert daily["all"]["regret_baseline"] == pytest.approx(80.0)
    # 2026-01-08 carries 170.0; the 100.0 on the 3rd is off-grid and unreachable.
    assert weekly["all"]["regret_baseline"] == pytest.approx(10.0)


def test_summarise_regret_forward_fills_dark_days_and_counts_them():
    """A dark fill day is scored (at the carried price the simulator itself bought at), never
    dropped — dropping is what would reintroduce the disjoint-basket defect."""
    flips = pd.DataFrame([
        _flip(1, 100, "2026-01-05", 170.0, 10.0, "candidate"),   # station reported on 01-01, not 01-05
    ])
    prices = FakePrices({100: {"2026-01-01": 170.0, "2026-01-06": 150.0}})
    out = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1, window_days=7)
    assert out["dark_fill_days"] == 1
    assert out["dark_fill_folds"] == [1]
    assert out["n_scored"] == 1 and out["n_unscored"] == 0
    assert out["all"]["regret_candidate"] == pytest.approx(20.0)
    assert out["n_price_mismatch"] == 0


def test_summarise_regret_flags_a_ledger_price_that_disagrees_with_the_db():
    """The executed price and the carried DB price agreed on all 303 fps-6yi flips; if that
    ever stops being true the count has to surface rather than vanish into a floored zero."""
    flips = pd.DataFrame([_flip(1, 100, "2026-01-01", 180.0, 10.0, "baseline")])
    prices = FakePrices({100: {"2026-01-01": 999.0}})
    out = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1, window_days=7)
    assert out["n_price_mismatch"] == 1
    # Floored at the price actually paid, so a bad DB row can't manufacture negative regret.
    assert out["all"]["regret_baseline"] == pytest.approx(0.0)


def test_summarise_regret_one_arm_only_gets_a_reason_not_a_silent_null():
    flips = pd.DataFrame([_flip(1, 100, "2026-01-01", 180.0, 10.0, "candidate")])
    prices = FakePrices({100: {"2026-01-01": 180.0}})
    out = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1, window_days=7)
    row = out["per_fold"][0]
    assert row["regret_delta"] is None
    assert row["regret_delta_reason"] == "one arm only"
    assert row["regret_delta_inside_own_se"] is None


def test_summarise_regret_interval_is_sized_on_decisions_not_fills():
    """The SE must widen when a fold's flips collapse into one cascade — the same defect PR
    #347's review caught in flip_cpl_delta's SE, which this measure must not repeat.

    A/B on the cascade window alone, holding the DATA fixed: at window_days=7 the six
    consecutive candidate flips are one decision, at window_days=0 every 1-day gap exceeds the
    window so they are six. If the SE were sized on per-fill litres (the defect), both arms of
    this comparison would return the SAME interval and the assertion below would fail.
    """
    cascade = [_flip(1, 100, f"2026-01-0{d}", 180.0 + d, 10.0, "candidate") for d in range(1, 7)]
    spread = [_flip(1, 200, f"2026-0{m}-01", 180.0 + m, 10.0, "baseline") for m in range(1, 7)]
    prices = FakePrices({
        100: {f"2026-01-0{d}": 180.0 + d for d in range(1, 7)} | {"2026-01-08": 100.0},
        200: {f"2026-0{m}-01": 180.0 + m for m in range(1, 7)} | {"2026-07-01": 100.0},
    })
    flips = pd.DataFrame(cascade + spread)
    collapsed = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1,
                                 window_days=7)["per_fold"][0]
    independent = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1,
                                   window_days=0)["per_fold"][0]
    assert collapsed["n_decisions"] == 7      # 1 cascaded candidate group + 6 baseline days
    assert independent["n_decisions"] == 12   # every flip its own decision
    # Same rows, same regrets — only the decision grouping differs.
    assert collapsed["regret_delta"] == pytest.approx(independent["regret_delta"])
    assert collapsed["regret_delta_se"] > independent["regret_delta_se"]


def test_summarise_regret_degenerate_dispersion_prints_no_interval():
    """Every flip landing at zero regret leaves no ruler to size an SE against — print no
    interval and say why, never a hairline one (same contract as _price_dispersion)."""
    flips = pd.DataFrame([
        _flip(1, 100, "2026-01-01", 180.0, 10.0, "baseline"),
        _flip(1, 100, "2026-02-01", 180.0, 10.0, "candidate"),
    ])
    prices = FakePrices({100: {"2026-01-01": 180.0, "2026-02-01": 180.0}})
    out = summarise_regret(flips, prices, set(), horizon_days=13, cadence_days=1, window_days=7)
    row = out["per_fold"][0]
    assert row["regret_delta"] == pytest.approx(0.0)
    assert row["regret_delta_se"] is None
    assert row["regret_delta_interval"] is None
    assert "not estimable" in row["regret_delta_interval_reason"]
