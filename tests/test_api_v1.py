"""Tests for fuel_signal.api_v1 — the /api/v1 wire-format serializers.

Several tests reproduce the worked examples from docs/api-contract.md verbatim
(rev 3, agreed with the iOS-app session) and assert an exact match against the
literal JSON in that document, rather than just spot-checking fields.
"""

from __future__ import annotations

import datetime
import math

import pytest

from fuel_signal import api_v1
from fuel_signal.cycle import CycleState
from fuel_signal.signal import (
    CombinedVerdict,
    SignalEvaluation,
    SignalPayload,
    SignalRecommendation,
    StationView,
)

_CONTRACT_EVALUATIONS = [
    SignalEvaluation(
        "AverageCycleTimeSignal",
        SignalRecommendation.BUY,
        "cycle ending soon (86% through cycle; day 40 / 46.3)",
    ),
    SignalEvaluation(
        "AverageGradientAfterPeakSignal",
        SignalRecommendation.NEUTRAL,
        "price has not flatlined (last 3 gradients: [-0.9, -0.8, -0.7])",
    ),
    SignalEvaluation(
        "AverageNearPreviousMinMaxSignal",
        SignalRecommendation.WAIT,
        "price in middle of last cycle (current 178.4c; last cycle min 155.3c, max 201.7c)",
    ),
    SignalEvaluation(
        "FavouriteServiceStationPriceGradientSignal",
        SignalRecommendation.NEUTRAL,
        "no preferred stations raising sharply (big raisers: none; non-raisers: "
        "BP Springwood @ -0.8, 7-Eleven Penrith South @ -0.6)",
    ),
]


def _make_payload(
    *,
    as_of_date: str = "2026-09-11",
    preferred_stations: dict[int, str],
    station_prices: dict[int, float | None],
    station_probs: dict[int, float] | None = None,
    avg_current_price: float = 178.4,
    drift: float | None = -0.8,
    last_real_price_date: str | None = "2026-09-11",
    rule_verdict: CombinedVerdict | None = None,
    evaluations: list[SignalEvaluation] | None = None,
    days_since_last_peak: int = 40,
    mean_cycle_length: float = 46.33,
) -> SignalPayload:
    state = CycleState(
        as_of_date=as_of_date,
        days_since_last_peak=days_since_last_peak,
        mean_cycle_length=mean_cycle_length,
        pct_through_cycle=days_since_last_peak / mean_cycle_length,
        last_cycle_min=155.3,
        last_cycle_max=201.7,
        last_3_gradients=[-0.9, -0.8, -0.7],
        peak_count=5,
    )
    return SignalPayload(
        as_of_date=as_of_date,
        preferred_stations=preferred_stations,
        state=state,
        avg_current_price=avg_current_price,
        drift=drift,
        station_prices=station_prices,
        station_probs=station_probs or {},
        rule_verdict=rule_verdict or CombinedVerdict("BUY ", "BUY", 0.5),
        evaluations=evaluations or _CONTRACT_EVALUATIONS,
        last_real_price_date=last_real_price_date,
    )


_CONTRACT_STATIONS = {414: "BP Springwood", 261: "7-Eleven Penrith South", 585: "EG Ampol Emu Heights"}
_CONTRACT_PRICES = {414: 161.9, 261: 157.7, 585: None}
_CONTRACT_PROBS = {414: 0.823}
_ROUTING_DAY = datetime.date(2026, 9, 12)
_GENERATED_AT = "2026-09-11T22:04:13+10:00"
_SERVED_AT = "2026-09-12T07:01:55+10:00"


def _contract_payload(**overrides) -> SignalPayload:
    kwargs = dict(
        preferred_stations=_CONTRACT_STATIONS,
        station_prices=_CONTRACT_PRICES,
        station_probs=_CONTRACT_PROBS,
    )
    kwargs.update(overrides)
    return _make_payload(**kwargs)


# ---------------------------------------------------------------------------
# encode/decode round trip
# ---------------------------------------------------------------------------


def test_encode_decode_round_trips_a_payload():
    payload = _contract_payload()
    blob = api_v1.encode_cache_entry(payload, gap_start=None, gap_end=None)
    decoded_payload, gap_start, gap_end = api_v1.decode_cache_entry(blob)
    assert decoded_payload == payload
    assert gap_start is None
    assert gap_end is None


def test_encode_decode_round_trips_gap_boundaries():
    payload = _contract_payload()
    blob = api_v1.encode_cache_entry(payload, gap_start="2021-03-01", gap_end="2021-03-15")
    _, gap_start, gap_end = api_v1.decode_cache_entry(blob)
    assert (gap_start, gap_end) == ("2021-03-01", "2021-03-15")


def test_encode_decode_round_trips_nan_mean_value():
    """`combine_signals` returns NaN when every signal is NEUTRAL — the one
    NaN-capable field on the wire (contract § "What can and cannot be NaN")."""
    payload = _contract_payload(rule_verdict=CombinedVerdict("WAIT", "WAIT", float("nan")))
    blob = api_v1.encode_cache_entry(payload, gap_start=None, gap_end=None)
    encoded_mean = blob["payload"]["rule_verdict"]["mean_value"]
    assert encoded_mean is None   # JSON has no NaN; must serialize as null
    decoded_payload, _, _ = api_v1.decode_cache_entry(blob)
    assert math.isnan(decoded_payload.rule_verdict.mean_value)


# ---------------------------------------------------------------------------
# GET /api/v1/stations — exact match against the contract's worked example
# ---------------------------------------------------------------------------


def test_build_stations_response_matches_contract_worked_example():
    payload = _contract_payload()
    body = api_v1.build_stations_response(
        payload, gap_start=None, gap_end=None, generated_at=_GENERATED_AT,
        routing_day=_ROUTING_DAY, served_at=_SERVED_AT, now=_ROUTING_DAY,
    )

    assert body["as_of"] == "2026-09-11"
    assert body["generated_at"] == _GENERATED_AT
    assert body["routing_day"] == "2026-09-12"
    assert body["served_at"] == _SERVED_AT
    assert body["freshness"] == {
        "latest_observation_date": "2026-09-11",
        "days_stale": 1,
        "expected_lag_days": 1,
        "fill_gap": None,
    }
    assert body["network"] == {
        "average_price": 178.4,
        "drift_cents_per_day": -0.8,
        "drift_label": "falling",
    }

    on_route, off_route, unpriced = body["stations"]
    assert on_route == {
        "code": 414,
        "label": "BP Springwood",
        "group": "on_route",
        "price": 161.9,
        "delta_vs_network": -16.5,
        "probability_buy": 0.823,
        "route": {
            "restricted": False,
            "days": None,
            "reachable_on_routing_day": True,
            "next_reachable_date": "2026-09-12",
            "days_until_reachable": 0,
        },
        "diversion": None,
        "cheapest_on_route": True,
    }
    assert off_route == {
        "code": 261,
        "label": "7-Eleven Penrith South",
        "group": "off_route",
        "price": 157.7,
        "delta_vs_network": -20.7,
        "probability_buy": None,
        "route": {
            "restricted": True,
            "days": [2, 6],
            "reachable_on_routing_day": False,
            "next_reachable_date": "2026-09-13",
            "days_until_reachable": 1,
        },
        "diversion": {
            "delta_vs_cheapest_on_route": -4.2,
            "worth_diverting": True,
            "threshold_cents": 3.0,
        },
        "cheapest_on_route": False,
    }
    assert unpriced == {
        "code": 585,
        "label": "EG Ampol Emu Heights",
        "group": "unpriced",
        "price": None,
        "delta_vs_network": None,
        "probability_buy": None,
        "route": {
            "restricted": False,
            "days": None,
            "reachable_on_routing_day": True,
            "next_reachable_date": "2026-09-12",
            "days_until_reachable": 0,
        },
        "diversion": None,
        "cheapest_on_route": False,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/recommendation — exact match against the contract's worked example
# ---------------------------------------------------------------------------


def test_build_recommendation_response_matches_contract_worked_example():
    payload = _contract_payload()
    body = api_v1.build_recommendation_response(
        payload, gap_start=None, gap_end=None, generated_at=_GENERATED_AT,
        routing_day=_ROUTING_DAY, served_at=_SERVED_AT, now=_ROUTING_DAY,
    )

    assert body["status"] == "ok"
    assert body["verdict"] == "BUY"
    assert body["fill_size"] == "bridge"
    assert body["source"] == "model"
    assert body["headline"] == "WORTH A STOP — BP Springwood @ 161.9c, but bridge, don't brim."
    assert body["reason"] == (
        "Good local price, but the network is falling 0.8c/day — buy enough to "
        "get by and keep some tank for later."
    )
    assert body["target_station"] == {"code": 414, "label": "BP Springwood", "price": 161.9}
    assert body["nearest_off_route"] is None
    assert body["cycle"] == {
        "day": 41,
        "length": 46.33,
        "beyond_expected_length": False,
        "pct_through": pytest.approx(0.8633714655730629),
        "last_cycle_min": 155.3,
        "last_cycle_max": 201.7,
    }
    assert body["rules"] == {
        "verdict": "BUY",
        "mean_value": 0.5,
        "signals": [
            {"name": "AverageCycleTimeSignal", "recommendation": "BUY",
             "description": "cycle ending soon (86% through cycle; day 40 / 46.3)"},
            {"name": "AverageGradientAfterPeakSignal", "recommendation": "NEUTRAL",
             "description": "price has not flatlined (last 3 gradients: [-0.9, -0.8, -0.7])"},
            {"name": "AverageNearPreviousMinMaxSignal", "recommendation": "WAIT",
             "description": "price in middle of last cycle (current 178.4c; last cycle min 155.3c, max 201.7c)"},
            {"name": "FavouriteServiceStationPriceGradientSignal", "recommendation": "NEUTRAL",
             "description": "no preferred stations raising sharply (big raisers: none; non-raisers: "
             "BP Springwood @ -0.8, 7-Eleven Penrith South @ -0.6)"},
        ],
    }


def test_recommendation_dont_buy_is_remapped_from_rules_verdict():
    """The server's verbatim "DON'T BUY" is a poor wire token — remapped to
    DONT_BUY, a deliberate exception noted in the contract."""
    payload = _make_payload(
        preferred_stations={414: "BP Springwood"},
        station_prices={414: 161.9},
        station_probs={},   # unscored -> source: rules
        rule_verdict=CombinedVerdict("DONT", "DON'T BUY", -1.0),
    )
    body = api_v1.build_recommendation_response(
        payload, gap_start=None, gap_end=None, generated_at=_GENERATED_AT,
        routing_day=_ROUTING_DAY, served_at=_SERVED_AT, now=_ROUTING_DAY,
    )
    assert body["status"] == "ok"
    assert body["source"] == "rules"
    assert body["verdict"] == "DONT_BUY"
    assert body["fill_size"] is None
    assert body["rules"]["verdict"] == "BUY" or True  # rules.verdict is independent; see next assert
    # rules.verdict tracks the SAME CombinedVerdict, so it must also be remapped.
    assert body["rules"]["verdict"] == "DONT_BUY"


def test_recommendation_status_no_priced_station_on_route():
    """261 is priced but only reachable Wed/Sun (real STATION_ROUTE_DAYS); route
    on a day it's not passed and there is no other priced station."""
    monday = datetime.date(2026, 9, 7)
    assert monday.weekday() == 0
    payload = _make_payload(
        preferred_stations={261: "7-Eleven Penrith South"},
        station_prices={261: 157.7},
        station_probs={},
    )
    body = api_v1.build_recommendation_response(
        payload, gap_start=None, gap_end=None, generated_at=_GENERATED_AT,
        routing_day=monday, served_at=_SERVED_AT, now=monday,
    )
    assert body["status"] == "no_priced_station_on_route"
    assert body["verdict"] is None
    assert body["fill_size"] is None
    assert body["source"] is None
    assert body["headline"] == "no priced station is reachable today"
    assert body["target_station"] is None
    assert body["nearest_off_route"] == {
        "code": 261,
        "label": "7-Eleven Penrith South",
        "price": 157.7,
        "next_reachable_date": "2026-09-09",   # next Wed/Sun on/after Monday 2026-09-07
        "days_until_reachable": 2,
    }
    assert "7-Eleven Penrith South" in body["reason"]
    assert "157.7c" in body["reason"]


def test_recommendation_status_no_price_data():
    payload = _make_payload(
        preferred_stations={414: "BP Springwood"},
        station_prices={414: None},
        station_probs={},
    )
    body = api_v1.build_recommendation_response(
        payload, gap_start=None, gap_end=None, generated_at=_GENERATED_AT,
        routing_day=_ROUTING_DAY, served_at=_SERVED_AT, now=_ROUTING_DAY,
    )
    assert body["status"] == "no_price_data"
    assert body["headline"] == "no price data for any preferred station"
    assert body["reason"] is None
    assert body["verdict"] is None
    assert body["fill_size"] is None
    assert body["source"] is None
    assert body["target_station"] is None
    assert body["nearest_off_route"] is None


def test_freshness_fill_gap_populated_when_as_of_falls_in_the_gap():
    payload = _contract_payload(as_of_date="2021-03-10")
    body = api_v1.build_stations_response(
        payload, gap_start="2021-03-01", gap_end="2021-03-15", generated_at=_GENERATED_AT,
        routing_day=_ROUTING_DAY, served_at=_SERVED_AT, now=_ROUTING_DAY,
    )
    assert body["freshness"]["fill_gap"] == {"start": "2021-03-01", "end": "2021-03-15"}


# ---------------------------------------------------------------------------
# _fill_decision — the contract's complete branch table
# ---------------------------------------------------------------------------


def _view(code: int, price: float, prob: float | None) -> StationView:
    return StationView(code=code, label=f"S{code}", price=price, prob=prob, route_days=None)


@pytest.mark.parametrize(
    "on_route,drift,scored,rule_long_label,expected",
    [
        ([_view(1, 150.0, 0.9)], 0.0, True, "BUY", ("model", "BUY", "brim")),
        ([_view(1, 150.0, 0.9)], -0.8, True, "BUY", ("model", "BUY", "bridge")),
        ([_view(1, 150.0, 0.1), _view(2, 151.0, 0.9)], 0.0, True, "BUY", ("model", "WAIT", None)),
        ([_view(1, 150.0, 0.1), _view(2, 151.0, 0.1)], 0.0, True, "BUY", ("model", "WAIT", "minimum")),
        ([_view(1, 150.0, None)], 0.0, False, "BUY", ("rules", "BUY", "brim")),
        ([_view(1, 150.0, None)], -0.8, False, "BUY", ("rules", "BUY", "bridge")),
        ([_view(1, 150.0, None)], 0.0, False, "WAIT", ("rules", "WAIT", None)),
        ([_view(1, 150.0, None)], 0.0, False, "DON'T BUY", ("rules", "DONT_BUY", None)),
    ],
)
def test_fill_decision_matches_contract_branch_table(on_route, drift, scored, rule_long_label, expected):
    rule_verdict = CombinedVerdict("x", rule_long_label, 0.0)
    result = api_v1._fill_decision(
        on_route, drift, 0.25, scored=scored, rule_verdict=rule_verdict
    )
    assert result == expected


# ---------------------------------------------------------------------------
# Cache format versioning — Claude review on #425
# ---------------------------------------------------------------------------


def test_encode_cache_entry_stamps_the_format_version():
    payload = _contract_payload()
    blob = api_v1.encode_cache_entry(payload, gap_start=None, gap_end=None)
    assert blob["version"] == api_v1._CACHE_FORMAT_VERSION


def test_decode_cache_entry_rejects_a_missing_version():
    """A row written before versioning existed — must fail loudly (the /api/v1
    blueprint catches this and degrades to 503, not a bare 500)."""
    payload = _contract_payload()
    blob = api_v1.encode_cache_entry(payload, gap_start=None, gap_end=None)
    del blob["version"]
    with pytest.raises(ValueError, match="unsupported signal_cache format version"):
        api_v1.decode_cache_entry(blob)


def test_decode_cache_entry_rejects_a_future_version():
    payload = _contract_payload()
    blob = api_v1.encode_cache_entry(payload, gap_start=None, gap_end=None)
    blob["version"] = api_v1._CACHE_FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="unsupported signal_cache format version"):
        api_v1.decode_cache_entry(blob)


def test_decode_cache_entry_raises_keyerror_on_a_missing_field():
    """A field renamed/removed since the row was written — same failure shape,
    different exception type; the blueprint catches both."""
    payload = _contract_payload()
    blob = api_v1.encode_cache_entry(payload, gap_start=None, gap_end=None)
    del blob["payload"]["last_real_price_date"]
    with pytest.raises(KeyError):
        api_v1.decode_cache_entry(blob)


# ---------------------------------------------------------------------------
# A station with an empty (not None) route_days — Claude review on #425
# ---------------------------------------------------------------------------


def test_route_block_handles_a_station_with_no_reachable_day():
    """`route_days=frozenset()` is a real (if unusual) config state — genuinely
    never reachable — distinct from `route_days=None` (always reachable).
    `StationView.next_reachable` returns None for it, same as the unrestricted
    case; the two must not be conflated."""
    v = StationView(code=1, label="Never", price=150.0, prob=None, route_days=frozenset())
    block = api_v1._route_block(v, _ROUTING_DAY)
    assert block == {
        "restricted": True,
        "days": [],
        "reachable_on_routing_day": False,
        "next_reachable_date": None,
        "days_until_reachable": None,
    }


def test_recommendation_handles_an_off_route_station_with_no_reachable_day():
    payload = _make_payload(
        preferred_stations={999: "Never Reachable"},
        station_prices={999: 150.0},
        station_probs={},
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(api_v1, "STATION_ROUTE_DAYS", {999: frozenset()})
        body = api_v1.build_recommendation_response(
            payload, gap_start=None, gap_end=None, generated_at=_GENERATED_AT,
            routing_day=_ROUTING_DAY, served_at=_SERVED_AT, now=_ROUTING_DAY,
        )
    assert body["status"] == "no_priced_station_on_route"
    assert body["nearest_off_route"] == {
        "code": 999,
        "label": "Never Reachable",
        "price": 150.0,
        "next_reachable_date": None,
        "days_until_reachable": None,
    }
    assert "no reachable day configured" in body["reason"]
