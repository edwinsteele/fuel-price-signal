"""Tests for fuel_signal.inspect — gradient heatmap builder and Flask routes."""

import datetime
import re

import pytest

from fuel_signal import series as _series
from fuel_signal.db import (
    create_schema,
    db_summary,
    insert_prices,
    open_db,
    upsert_daily_prices,
    upsert_stations,
)
from fuel_signal.inspect import (
    _build_coverage_heatmap,
    _build_gradient_heatmap,
    _build_line_spec,
    _create_app,
    _slice_points,
)


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


_STATION_BM = {
    "station_code": 1001,
    "name": "Shell Springwood",
    "address": "1 Main Street, Springwood",
    "suburb": "Springwood",
    "postcode": "2777",
    "brand": "Shell",
}

_STATION_SYD = {
    "station_code": 2001,
    "name": "Ampol Parramatta",
    "address": "5 Church Street, Parramatta",
    "suburb": "Parramatta",
    "postcode": "2150",
    "brand": "Ampol",
}


def _insert_prices(conn, station_code, n_days=10, base_price=160.0):
    base = datetime.date(2024, 1, 1)
    rows = [
        (station_code, "E10", (base + datetime.timedelta(days=i)).isoformat(), base_price + i)
        for i in range(n_days)
    ]
    upsert_daily_prices(conn, rows)
    conn.commit()


def _resolve_specs(conn, specs):
    return [_series.resolve(conn, s) for s in specs]


def test_gradient_heatmap_filters_to_selected_lga(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_prices(conn, 1001)
    _insert_prices(conn, 2001)
    resolved = _resolve_specs(conn, ["lga:Blue Mountains"])
    result = _build_gradient_heatmap(resolved, cutoff=None)
    assert result
    labels = {label for label, _ in result["rows"]}
    assert any("Blue Mountains" in lab for lab in labels)
    assert not any("Parramatta" in lab for lab in labels)


def test_gradient_heatmap_includes_brand_and_station_rows(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_prices(conn, 1001)
    _insert_prices(conn, 2001)
    resolved = _resolve_specs(
        conn, ["sydney", "brand:Shell", "station:1001"]
    )
    result = _build_gradient_heatmap(resolved, cutoff=None)
    assert result
    labels = {label for label, _ in result["rows"]}
    assert any("Sydney" in lab for lab in labels)
    assert any("Shell" in lab for lab in labels)
    assert any("Springwood" in lab for lab in labels)


def test_gradient_heatmap_returns_daily_dates(conn):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=10)
    resolved = _resolve_specs(conn, ["station:1001"])
    result = _build_gradient_heatmap(resolved, cutoff=None)
    assert result
    assert len(result["dates"]) == 10
    assert all(len(d) == 10 for d in result["dates"])


def test_gradient_heatmap_respects_cutoff(conn):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=10)
    resolved = _resolve_specs(conn, ["station:1001"])
    cutoff = (datetime.date(2024, 1, 1) + datetime.timedelta(days=5)).isoformat()
    result = _build_gradient_heatmap(resolved, cutoff=cutoff)
    assert result
    assert all(d >= cutoff for d in result["dates"])


def test_gradient_heatmap_empty_when_no_resolved_series(conn):
    result = _build_gradient_heatmap([], cutoff=None)
    assert result == {}


# ---------------------------------------------------------------------------
# Coverage heatmap tests
# ---------------------------------------------------------------------------

def _insert_raw_prices(conn, station_code, n_days=5, base_price=160.0):
    # Use recent dates so coverage_matrix's 24-month window includes them.
    base = datetime.date.today() - datetime.timedelta(days=30)
    rows = [
        {
            "station_code": station_code,
            "fuel_code": "E10",
            "price_date": (base + datetime.timedelta(days=i)).isoformat(),
            "price_cents": base_price + i,
        }
        for i in range(n_days)
    ]
    insert_prices(conn, rows)
    conn.commit()


def test_coverage_heatmap_filters_to_station_codes(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_raw_prices(conn, 1001)
    _insert_raw_prices(conn, 2001)
    result = _build_coverage_heatmap(conn, cutoff=None, station_codes={1001})
    assert result
    row_names = {name for name, _ in result["rows"]}
    assert "Shell Springwood" in row_names
    assert "Ampol Parramatta" not in row_names


def test_coverage_heatmap_no_filter_shows_all_stations(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_raw_prices(conn, 1001)
    _insert_raw_prices(conn, 2001)
    result = _build_coverage_heatmap(conn, cutoff=None, station_codes=None)
    assert result
    row_names = {name for name, _ in result["rows"]}
    assert "Shell Springwood" in row_names
    assert "Ampol Parramatta" in row_names


def test_coverage_heatmap_empty_station_codes_returns_empty(conn):
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    _insert_raw_prices(conn, 1001)
    _insert_raw_prices(conn, 2001)
    result = _build_coverage_heatmap(conn, cutoff=None, station_codes=set())
    assert result == {}


# ---------------------------------------------------------------------------
# _slice_points date-range tests
# ---------------------------------------------------------------------------

_POINTS = [
    ("2024-01-01", 150.0),
    ("2024-02-01", 155.0),
    ("2024-03-01", 160.0),
    ("2024-04-01", 165.0),
]


def test_slice_points_lower_bound_only():
    result = _slice_points(_POINTS, cutoff="2024-02-01")
    assert [d for d, _ in result] == ["2024-02-01", "2024-03-01", "2024-04-01"]


def test_slice_points_upper_bound_only():
    result = _slice_points(_POINTS, cutoff=None, end="2024-03-01")
    assert [d for d, _ in result] == ["2024-01-01", "2024-02-01", "2024-03-01"]


def test_slice_points_both_bounds():
    result = _slice_points(_POINTS, cutoff="2024-02-01", end="2024-03-01")
    assert [d for d, _ in result] == ["2024-02-01", "2024-03-01"]


def test_slice_points_no_bounds_returns_all():
    result = _slice_points(_POINTS, cutoff=None)
    assert result == _POINTS


def test_slice_points_empty_window_returns_empty():
    result = _slice_points(_POINTS, cutoff="2024-05-01", end="2024-06-01")
    assert result == []


# ---------------------------------------------------------------------------
# _build_line_spec: show_annotations flag
# ---------------------------------------------------------------------------

_PEAK_DATA_EMPTY = {
    "peak_dates": [],
    "plateau_peak_date": None,
    "last_cycle_start": None,
    "last_cycle_end": None,
}


def test_build_line_spec_annotations_on_flag_true(conn):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=10)
    resolved = _resolve_specs(conn, ["station:1001"])
    peak_date = resolved[0].points[3][0]  # a date within the series
    peak_data = {**_PEAK_DATA_EMPTY, "peak_dates": [peak_date]}
    result = _build_line_spec(resolved, peak_data, {}, show_annotations=True)
    assert "pk0" in result["annotations"]


def test_build_line_spec_annotations_off_flag_false(conn):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=10)
    resolved = _resolve_specs(conn, ["station:1001"])
    peak_date = resolved[0].points[3][0]
    peak_data = {**_PEAK_DATA_EMPTY, "peak_dates": [peak_date]}
    result = _build_line_spec(resolved, peak_data, {}, show_annotations=False)
    assert result["annotations"] == {}


# ---------------------------------------------------------------------------
# Route-level tests: start/end query params
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_client(conn):
    """Minimal Flask test client with one station and a few price rows."""
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=14)
    app = _create_app(
        conn,
        cd=None,
        today="2024-01-14",
        cycle_state=None,
        peak_data={},
        summary=db_summary(conn),
        boundaries={},
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_route_start_end_override_persists_in_response(flask_client):
    """start= and end= params should appear in the rendered form inputs."""
    resp = flask_client.get(
        "/?start=2024-01-03&end=2024-01-07&series=station:1001&chart=heatmap-coverage"
    )
    assert resp.status_code == 200
    html = resp.data.decode()
    assert 'value="2024-01-03"' in html
    assert 'value="2024-01-07"' in html


def test_route_invalid_start_returns_400(flask_client):
    """Invalid start date should return a 400 with an error in the page."""
    resp = flask_client.get("/?start=not-a-date&series=station:1001")
    assert resp.status_code == 400
    assert b"Invalid start date" in resp.data


def test_route_invalid_end_returns_400(flask_client):
    """Invalid end date should return a 400 with an error in the page."""
    resp = flask_client.get("/?end=2024-13-01&series=station:1001")
    assert resp.status_code == 400
    assert b"Invalid end date" in resp.data


def test_route_fresh_load_annotations_checkbox_checked(flask_client):
    """Fresh load (no query params) should render the annotations checkbox checked."""
    resp = flask_client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert 'name="annotations"' in html
    # Checkbox must carry the checked attribute on fresh load.
    assert re.search(r'<input[^>]*name="annotations"[^>]*\bchecked\b', html)


def test_route_form_submit_without_annotations_param_unchecked(flask_client):
    """Submitting the form without the annotations param should leave the checkbox unchecked."""
    resp = flask_client.get("/?series=station:1001&chart=line")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert 'name="annotations"' in html
    assert not re.search(r'<input[^>]*name="annotations"[^>]*\bchecked\b', html)


# ---------------------------------------------------------------------------
# Coverage heatmap: LGA/brand/station selections honoured alongside sydney
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_client_two_lgas(conn):
    """Flask test client with one Blue Mountains station and one Parramatta station."""
    upsert_stations(conn, [_STATION_BM, _STATION_SYD])
    # Populate prices (for coverage_matrix) and daily_prices (for resolve_members).
    _insert_raw_prices(conn, 1001)
    _insert_raw_prices(conn, 2001)
    base = datetime.date.today() - datetime.timedelta(days=30)
    for code in (1001, 2001):
        upsert_daily_prices(conn, [
            (code, "E10", (base + datetime.timedelta(days=i)).isoformat(), 160.0 + i)
            for i in range(5)
        ])
    conn.commit()
    app = _create_app(
        conn,
        cd=None,
        today=datetime.date.today().isoformat(),
        cycle_state=None,
        peak_data={},
        summary=db_summary(conn),
        boundaries={},
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_route_coverage_heatmap_honours_lga_when_sydney_also_selected(flask_client_two_lgas):
    """LGA filter applies to coverage heatmap even when sydney is also selected."""
    resp = flask_client_two_lgas.get(
        "/?series=sydney&series=lga:Blue+Mountains&chart=heatmap-coverage"
    )
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "Shell Springwood" in html
    assert "Ampol Parramatta" not in html


def test_route_coverage_heatmap_sydney_only_shows_all(flask_client_two_lgas):
    """When only sydney is selected, all stations appear in the coverage heatmap."""
    resp = flask_client_two_lgas.get(
        "/?series=sydney&chart=heatmap-coverage"
    )
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "Shell Springwood" in html
    assert "Ampol Parramatta" in html
