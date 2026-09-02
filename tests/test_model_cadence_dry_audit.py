"""Tests for the re-aimed run-dry gate in
`experiments/2026-08-20_cadence_ceiling/model_cadence.py` (fps-32h).

Post PR #358, `run_oracle_backtest` clamps-and-accounts through a no-price date
exactly like `run_backtest` does, so a dry event alone no longer proves a cadence
is unsafe — a real station closure produces dry events in both arms even when
nothing about the cadence choice is unsafe. `_dry_audit` must instead classify
each dry event as darkness (forgiven) or stranding (a genuine cadence failure),
and `_symmetry_excess_litres` must catch the two arms disagreeing by more than
the physical bound. Experiment scripts are conventionally untested print-and-exit
code, but this gate is exactly the kind of estimator-with-a-known-answer that
earns a test (see test_texture_icc_estimator.py's docstring for the precedent).

The script lives outside any package (dated, hyphenated directory), so it is
loaded by path.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pandas as pd

from fuel_signal.backtest import PriceHistory, TankParams

EXPERIMENT_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "experiments" / "2026-08-20_cadence_ceiling"
)


def _load(name: str):
    sys.path.insert(0, str(EXPERIMENT_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"_cadence_{name}", EXPERIMENT_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT_DIR))


model_cadence = _load("model_cadence")

EMPTY_FILLS = pd.DataFrame(columns=["fold", "station_code", "date", "litres"])
TANK = TankParams(
    tank_size_litres=10.0, daily_consumption_litres=4.0, evaluation_interval_days=1,
    floor_fraction=0.10,
)
# Three eval dates at 1d cadence: 01-05, 01-06, 01-07. Level starts at 5.0
# (0.5 * size); with no fills, day1 leaves 1.0 (5 - 4, not dry) and day2 goes to
# -3.0 (1 - 4, dry) — the same shape for every station below, only the price
# history around day2 differs.
PLAN = types.SimpleNamespace(fold=1, val_start="2024-01-05", val_end="2024-01-07")

GRACE_STATION = 1     # dry event on the date price RESUMES after a gap
DARK_STATION = 2      # dry event on a date with no price at all
STRANDED_STATION = 3  # dry event on a date with a price, no adjacent gap


def _history() -> PriceHistory:
    avg_series = [("2024-01-05", 180.0), ("2024-01-06", 180.0), ("2024-01-07", 180.0)]
    return PriceHistory(
        avg_series=avg_series,
        station_prices={
            # Gap 2023-11-01 -> 2024-01-07 is 67 days (> MAX_GAP_FILL_DAYS=28), so
            # 01-05 and 01-06 resolve to None and 01-07 resolves to 175.0 exactly
            # (it IS an observation) — the one-step grace day.
            GRACE_STATION: [("2023-11-01", 180.0), ("2024-01-07", 175.0)],
            # Gap 2023-11-01 -> 2024-03-01 spans the whole window: every eval date
            # resolves to None.
            DARK_STATION: [("2023-11-01", 180.0), ("2024-03-01", 180.0)],
            # Priced every eval date: no gap anywhere.
            STRANDED_STATION: [
                ("2024-01-05", 180.0), ("2024-01-06", 180.0), ("2024-01-07", 180.0),
            ],
        },
    )


def test_transition_day_dry_event_is_darkness_not_stranding():
    """The one-step grace: price resumes on the dry date itself, but yesterday had
    none — run_backtest doesn't raise there either, so this must not count as
    stranding."""
    audit = model_cadence._dry_audit(
        EMPTY_FILLS, [PLAN], 1, TANK, [GRACE_STATION], _history()
    )
    assert audit["dry_events"] == 1
    assert audit["stranded_events"] == 0
    assert audit["dry_litres"] == 3.0


def test_no_price_dry_event_is_darkness():
    """A dry event whose date has no resolvable price at all is the direct
    darkness case."""
    audit = model_cadence._dry_audit(
        EMPTY_FILLS, [PLAN], 1, TANK, [DARK_STATION], _history()
    )
    assert audit["dry_events"] == 1
    assert audit["stranded_events"] == 0
    assert audit["dry_litres"] == 3.0


def test_dry_event_with_a_price_and_no_adjacent_gap_is_stranding():
    """A price was available today AND yesterday, so run_backtest would raise on
    this transition — a real cadence failure the gate must still catch."""
    audit = model_cadence._dry_audit(
        EMPTY_FILLS, [PLAN], 1, TANK, [STRANDED_STATION], _history()
    )
    assert audit["dry_events"] == 1
    assert audit["stranded_events"] == 1
    assert audit["stranded_litres"] == 3.0


def test_dry_audit_pools_all_three_stations_correctly():
    audit = model_cadence._dry_audit(
        EMPTY_FILLS, [PLAN], 1, TANK, [GRACE_STATION, DARK_STATION, STRANDED_STATION],
        _history(),
    )
    assert audit["dry_events"] == 3
    assert audit["stranded_events"] == 1
    assert audit["dry_litres"] == 9.0
    assert audit["stranded_litres"] == 3.0
    assert audit["gaps_by_key"][(1, GRACE_STATION)] == 1
    assert audit["gaps_by_key"][(1, DARK_STATION)] == 1
    assert (1, STRANDED_STATION) not in audit["gaps_by_key"]


def test_symmetry_excess_zero_within_one_tank_size_per_gap():
    model_audit = {"litres_by_key": {(1, 100): 20.0}, "gaps_by_key": {(1, 100): 1}}
    oracle_audit = {"litres_by_key": {(1, 100): 25.0}, "gaps_by_key": {(1, 100): 1}}
    excess = model_cadence._symmetry_excess_litres(model_audit, oracle_audit, tank_size=10.0)
    assert excess <= 0.0


def test_symmetry_excess_positive_beyond_one_tank_size_per_gap():
    model_audit = {"litres_by_key": {(1, 100): 5.0}, "gaps_by_key": {(1, 100): 1}}
    oracle_audit = {"litres_by_key": {(1, 100): 20.0}, "gaps_by_key": {(1, 100): 1}}
    excess = model_cadence._symmetry_excess_litres(model_audit, oracle_audit, tank_size=10.0)
    assert excess > 0.0
