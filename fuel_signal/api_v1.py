"""JSON encoding for the /api/v1 blueprint — see docs/api-contract.md (rev 3).

Two halves, matching the cached/live split in `signal.SignalPayload`:

- `encode_cache_entry` / `decode_cache_entry` round-trip a `SignalPayload` (plus
  the gap boundaries, themselves a pure function of the DB) through the JSON
  blob stored in `db.signal_cache` by the nightly generator
  (`fuel_signal.generate_signal_cache`).
- `build_stations_response` / `build_recommendation_response` take a decoded
  payload plus `routing_day`/`now` — fresh at request time, never cached — and
  produce the exact wire shapes the contract defines.
"""

from __future__ import annotations

import datetime
import math

from fuel_signal import signal
from fuel_signal.config import STATION_ROUTE_DAYS

# Bump when `_encode_signal_payload`/`_decode_signal_payload`'s field set changes
# in a way an older/newer reader can't tolerate. See `decode_cache_entry`.
_CACHE_FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# SignalPayload <-> JSON
# ---------------------------------------------------------------------------


def _encode_signal_payload(payload: signal.SignalPayload) -> dict:
    return {
        "as_of_date": payload.as_of_date,
        "preferred_stations": {str(k): v for k, v in payload.preferred_stations.items()},
        "state": {
            "as_of_date": payload.state.as_of_date,
            "days_since_last_peak": payload.state.days_since_last_peak,
            "mean_cycle_length": payload.state.mean_cycle_length,
            "pct_through_cycle": payload.state.pct_through_cycle,
            "last_cycle_min": payload.state.last_cycle_min,
            "last_cycle_max": payload.state.last_cycle_max,
            "last_3_gradients": payload.state.last_3_gradients,
            "peak_count": payload.state.peak_count,
        },
        "avg_current_price": payload.avg_current_price,
        "drift": payload.drift,
        "station_prices": {str(k): v for k, v in payload.station_prices.items()},
        "station_probs": {str(k): v for k, v in payload.station_probs.items()},
        "rule_verdict": {
            "label": payload.rule_verdict.label,
            "long_label": payload.rule_verdict.long_label,
            "mean_value": (
                None
                if math.isnan(payload.rule_verdict.mean_value)
                else payload.rule_verdict.mean_value
            ),
        },
        "evaluations": [
            {
                "name": e.name,
                "recommendation": e.recommendation.name,
                "description": e.description,
            }
            for e in payload.evaluations
        ],
        "last_real_price_date": payload.last_real_price_date,
    }


def _decode_signal_payload(data: dict) -> signal.SignalPayload:
    state_d = data["state"]
    state = signal.CycleState(
        as_of_date=state_d["as_of_date"],
        days_since_last_peak=state_d["days_since_last_peak"],
        mean_cycle_length=state_d["mean_cycle_length"],
        pct_through_cycle=state_d["pct_through_cycle"],
        last_cycle_min=state_d["last_cycle_min"],
        last_cycle_max=state_d["last_cycle_max"],
        last_3_gradients=state_d["last_3_gradients"],
        peak_count=state_d["peak_count"],
    )
    rv_d = data["rule_verdict"]
    rule_verdict = signal.CombinedVerdict(
        label=rv_d["label"],
        long_label=rv_d["long_label"],
        mean_value=float("nan") if rv_d["mean_value"] is None else rv_d["mean_value"],
    )
    evaluations = [
        signal.SignalEvaluation(
            name=e["name"],
            recommendation=signal.SignalRecommendation[e["recommendation"]],
            description=e["description"],
        )
        for e in data["evaluations"]
    ]
    return signal.SignalPayload(
        as_of_date=data["as_of_date"],
        preferred_stations={int(k): v for k, v in data["preferred_stations"].items()},
        state=state,
        avg_current_price=data["avg_current_price"],
        drift=data["drift"],
        station_prices={int(k): v for k, v in data["station_prices"].items()},
        station_probs={int(k): v for k, v in data["station_probs"].items()},
        rule_verdict=rule_verdict,
        evaluations=evaluations,
        last_real_price_date=data["last_real_price_date"],
    )


def encode_cache_entry(
    payload: signal.SignalPayload, gap_start: str | None, gap_end: str | None
) -> dict:
    """The full JSON-safe blob stored in `db.signal_cache.payload_json`."""
    return {
        "version": _CACHE_FORMAT_VERSION,
        "payload": _encode_signal_payload(payload),
        "gap_start": gap_start,
        "gap_end": gap_end,
    }


def decode_cache_entry(data: dict) -> tuple[signal.SignalPayload, str | None, str | None]:
    """Raises (KeyError, TypeError, ValueError) on a blob this code can't read.

    `SignalPayload` is explicitly designed to grow new fields over time (see its
    docstring), and a cache row persists across a `git pull` + waitress restart
    until the next nightly `generate_signal_cache` run overwrites it — so a
    field rename/addition can leave a row on disk that predates the code
    reading it. Callers (the /api/v1 blueprint) must catch these and degrade to
    the documented 503 "not_generated", not let them surface as a bare 500.
    """
    if data.get("version") != _CACHE_FORMAT_VERSION:
        raise ValueError(f"unsupported signal_cache format version: {data.get('version')!r}")
    return (
        _decode_signal_payload(data["payload"]),
        data.get("gap_start"),
        data.get("gap_end"),
    )


# ---------------------------------------------------------------------------
# Shared wire-format helpers
# ---------------------------------------------------------------------------


def _round1(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


def _drift_label(drift: float | None) -> str:
    if drift is None:
        return "unknown"
    if drift > 0.05:
        return "rising"
    if drift < -0.05:
        return "falling"
    return "flat"


def _map_verdict(long_label: str) -> str:
    """`CombinedVerdict.long_label` -> the wire enum. Only DON'T BUY is remapped."""
    return "DONT_BUY" if long_label == "DON'T BUY" else long_label


def _station_views(payload: signal.SignalPayload) -> list[signal.StationView]:
    return [
        signal.StationView(
            code=code,
            label=label,
            price=payload.station_prices.get(code),
            prob=payload.station_probs.get(code),
            route_days=STATION_ROUTE_DAYS.get(code),
        )
        for code, label in payload.preferred_stations.items()
    ]


def _partition(
    views: list[signal.StationView], routing_day: datetime.date
) -> tuple[list[signal.StationView], list[signal.StationView], list[signal.StationView]]:
    """(on_route, off_route, unpriced) — price partitions first, route second.

    A station with no price is `unpriced` regardless of reachability (contract
    § "The `group` discriminator is not a routing partition"); `on_route` and
    `off_route` therefore only ever hold priced stations.
    """
    on_route = sorted(
        (v for v in views if v.price is not None and v.reachable_on(routing_day)),
        key=lambda v: v.price,
    )
    off_route = sorted(
        (v for v in views if v.price is not None and not v.reachable_on(routing_day)),
        key=lambda v: v.price,
    )
    unpriced = sorted((v for v in views if v.price is None), key=lambda v: v.label)
    return on_route, off_route, unpriced


def _route_block(v: signal.StationView, routing_day: datetime.date) -> dict:
    if v.route_days is None:
        return {
            "restricted": False,
            "days": None,
            "reachable_on_routing_day": True,
            "next_reachable_date": routing_day.isoformat(),
            "days_until_reachable": 0,
        }
    nxt = v.next_reachable(routing_day)
    if nxt is None:
        # `route_days` is a non-None but EMPTY set — a station configured as
        # never reachable, not the same as unrestricted. `next_reachable`
        # returns None here too (its `range(0, 8)` search matches nothing), so
        # this is not covered by the None-means-unrestricted branch above.
        return {
            "restricted": True,
            "days": sorted(v.route_days),
            "reachable_on_routing_day": False,
            "next_reachable_date": None,
            "days_until_reachable": None,
        }
    return {
        "restricted": True,
        "days": sorted(v.route_days),
        "reachable_on_routing_day": v.reachable_on(routing_day),
        "next_reachable_date": nxt[0].isoformat(),
        "days_until_reachable": nxt[1],
    }


def _header(as_of_date: str, generated_at: str, routing_day: datetime.date, served_at: str) -> dict:
    return {
        "as_of": as_of_date,
        "generated_at": generated_at,
        "routing_day": routing_day.isoformat(),
        "served_at": served_at,
    }


def _freshness_block(
    payload: signal.SignalPayload,
    gap_start: str | None,
    gap_end: str | None,
    now: datetime.date,
) -> dict:
    latest = payload.last_real_price_date or payload.as_of_date
    fill_gap = None
    if gap_start and gap_end and gap_start <= payload.as_of_date <= gap_end:
        fill_gap = {"start": gap_start, "end": gap_end}
    return {
        "latest_observation_date": latest,
        "days_stale": (now - datetime.date.fromisoformat(latest)).days,
        "expected_lag_days": signal.EXPECTED_LAG_DAYS,
        "fill_gap": fill_gap,
    }


def _network_block(payload: signal.SignalPayload) -> dict:
    return {
        "average_price": _round1(payload.avg_current_price),
        "drift_cents_per_day": _round1(payload.drift),
        "drift_label": _drift_label(payload.drift),
    }


# ---------------------------------------------------------------------------
# GET /api/v1/stations
# ---------------------------------------------------------------------------


def build_stations_response(
    payload: signal.SignalPayload,
    gap_start: str | None,
    gap_end: str | None,
    generated_at: str,
    routing_day: datetime.date,
    served_at: str,
    now: datetime.date,
) -> dict:
    on_route, off_route, unpriced = _partition(_station_views(payload), routing_day)
    cheapest_code = on_route[0].code if on_route else None

    def entry(v: signal.StationView, group: str) -> dict:
        diversion = None
        if group == "off_route" and on_route:
            delta = v.price - on_route[0].price
            diversion = {
                "delta_vs_cheapest_on_route": _round1(delta),
                "worth_diverting": delta <= -signal.DIVERSION_WORTH_CENTS,
                "threshold_cents": signal.DIVERSION_WORTH_CENTS,
            }
        return {
            "code": v.code,
            "label": v.label,
            "group": group,
            "price": _round1(v.price),
            "delta_vs_network": (
                _round1(v.price - payload.avg_current_price) if v.price is not None else None
            ),
            "probability_buy": None if v.prob is None else round(v.prob, 3),
            "route": _route_block(v, routing_day),
            "diversion": diversion,
            "cheapest_on_route": v.code == cheapest_code,
        }

    stations = (
        [entry(v, "on_route") for v in on_route]
        + [entry(v, "off_route") for v in off_route]
        + [entry(v, "unpriced") for v in unpriced]
    )
    return {
        **_header(payload.as_of_date, generated_at, routing_day, served_at),
        "freshness": _freshness_block(payload, gap_start, gap_end, now),
        "network": _network_block(payload),
        "stations": stations,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/recommendation
# ---------------------------------------------------------------------------


def _fill_decision(
    on_route: list[signal.StationView],
    drift: float | None,
    threshold: float,
    *,
    scored: bool,
    rule_verdict: signal.CombinedVerdict,
) -> tuple[str, str, str | None]:
    """(source, verdict, fill_size) per the contract's complete branch table."""
    falling = drift is not None and drift < signal.FALLING_CENTS_PER_DAY
    if not scored:
        verdict = _map_verdict(rule_verdict.long_label)
        if verdict == "BUY":
            return "rules", "BUY", ("bridge" if falling else "brim")
        return "rules", verdict, None

    best = on_route[0]
    buying = best.prob is not None and best.prob >= threshold
    if buying:
        return "model", "BUY", ("bridge" if falling else "brim")
    others = [v for v in on_route[1:] if v.prob is not None and v.prob >= threshold]
    return ("model", "WAIT", None) if others else ("model", "WAIT", "minimum")


def _rules_block(payload: signal.SignalPayload) -> dict:
    mean = payload.rule_verdict.mean_value
    return {
        "verdict": _map_verdict(payload.rule_verdict.long_label),
        "mean_value": None if math.isnan(mean) else mean,
        "signals": [
            {"name": e.name, "recommendation": e.recommendation.name, "description": e.description}
            for e in payload.evaluations
        ],
    }


def _cycle_block(state: signal.CycleState) -> dict:
    day = state.days_since_last_peak + 1
    return {
        "day": day,
        "length": state.mean_cycle_length,
        "beyond_expected_length": day > round(state.mean_cycle_length),
        "pct_through": state.pct_through_cycle,
        "last_cycle_min": _round1(state.last_cycle_min),
        "last_cycle_max": _round1(state.last_cycle_max),
    }


def build_recommendation_response(
    payload: signal.SignalPayload,
    gap_start: str | None,
    gap_end: str | None,
    generated_at: str,
    routing_day: datetime.date,
    served_at: str,
    now: datetime.date,
) -> dict:
    on_route, off_route, _unpriced = _partition(_station_views(payload), routing_day)
    scored = bool(payload.station_probs)

    if on_route:
        status = "ok"
        source, verdict, fill_size = _fill_decision(
            on_route, payload.drift, signal.MODEL_THRESHOLD,
            scored=scored, rule_verdict=payload.rule_verdict,
        )
        headline, reason = signal._fill_advice(
            on_route, payload.drift, signal.MODEL_THRESHOLD,
            scored=scored, rule_verdict=payload.rule_verdict,
        )
        best = on_route[0]
        target_station = {"code": best.code, "label": best.label, "price": _round1(best.price)}
        nearest_off_route = None
    elif off_route:
        status = "no_priced_station_on_route"
        source = verdict = fill_size = None
        headline = "no priced station is reachable today"
        nearest = off_route[0]
        nxt = nearest.next_reachable(routing_day)
        # `nxt` is None for a station configured with an empty (not None)
        # `route_days` — genuinely never reachable, not the common case this
        # branch is named for, but real input, not a code invariant to assert.
        if nxt is None:
            next_reachable_date = None
            days_until_reachable = None
            reason = (
                f"Cheapest preferred station is {nearest.label} @ {nearest.price:.1f}c, "
                "but it has no reachable day configured."
            )
        else:
            next_reachable_date = nxt[0].isoformat()
            days_until_reachable = nxt[1]
            reason = (
                f"Cheapest preferred station is {nearest.label} @ {nearest.price:.1f}c, "
                f"next passed in {nxt[1]}d."
            )
        target_station = None
        nearest_off_route = {
            "code": nearest.code,
            "label": nearest.label,
            "price": _round1(nearest.price),
            "next_reachable_date": next_reachable_date,
            "days_until_reachable": days_until_reachable,
        }
    else:
        status = "no_price_data"
        source = verdict = fill_size = None
        headline = "no price data for any preferred station"
        reason = None
        target_station = None
        nearest_off_route = None

    return {
        **_header(payload.as_of_date, generated_at, routing_day, served_at),
        "freshness": _freshness_block(payload, gap_start, gap_end, now),
        "status": status,
        "verdict": verdict,
        "fill_size": fill_size,
        "source": source,
        "headline": headline,
        "reason": reason,
        "target_station": target_station,
        "nearest_off_route": nearest_off_route,
        "cycle": _cycle_block(payload.state),
        "network": _network_block(payload),
        "rules": _rules_block(payload),
    }
