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
    _apply_hybrid_cutoff,
    _build_coverage_heatmap,
    _build_gradient_heatmap,
    _build_line_spec,
    _compute_interaction_budget_ranks,
    _create_app,
    _load_partner_scores,
    _load_shap_summary,
    _slice_points,
    _sort_shap_rows,
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


# ---------------------------------------------------------------------------
# _load_shap_summary
# ---------------------------------------------------------------------------

def test_load_shap_summary_returns_none_when_missing(tmp_path):
    assert _load_shap_summary(tmp_path / "no_such_dir") is None


def test_load_shap_summary_returns_none_when_csv_absent(tmp_path):
    (tmp_path / "shap").mkdir()
    assert _load_shap_summary(tmp_path / "shap") is None


def test_load_shap_summary_returns_rows(tmp_path):
    import csv
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    with open(shap_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_shap", "rank", "r", "nan_fraction"])
        writer.writeheader()
        writer.writerow({"feature": "feat_a", "mean_abs_shap": 0.05, "rank": 1, "r": 1.0, "nan_fraction": 0.0})
        writer.writerow({"feature": "feat_b", "mean_abs_shap": 0.02, "rank": 2, "r": -1.0, "nan_fraction": 0.1})
    rows = _load_shap_summary(shap_dir)
    assert rows is not None
    assert len(rows) == 2
    assert rows[0]["feature"] == "feat_a"


# ---------------------------------------------------------------------------
# _sort_shap_rows
# ---------------------------------------------------------------------------

_SAMPLE_ROWS = [
    {"feature": "zeta", "mean_abs_shap": 0.01, "rank": 3, "r": 1.0, "nan_fraction": 0.0},
    {"feature": "alpha", "mean_abs_shap": 0.05, "rank": 1, "r": -1.0, "nan_fraction": 0.0},
    {"feature": "mu",    "mean_abs_shap": 0.03, "rank": 2, "r": float("nan"), "nan_fraction": 0.5},
]


def test_sort_shap_rows_shap_descending():
    rows = _sort_shap_rows(_SAMPLE_ROWS, "shap")
    vals = [r["mean_abs_shap"] for r in rows]
    assert vals == sorted(vals, reverse=True)


def test_sort_shap_rows_alpha():
    rows = _sort_shap_rows(_SAMPLE_ROWS, "alpha")
    names = [r["feature"] for r in rows]
    assert names == sorted(names)


def test_sort_shap_rows_sign_groups_by_sign():
    rows = _sort_shap_rows(_SAMPLE_ROWS, "sign")
    # Positive-sign features rank first (key = -1), then NaN (key = 0), then negative (key = 1).
    signs = [r["r"] for r in rows]
    non_nan_signs = [s for s in signs if s == s]
    assert non_nan_signs == sorted(non_nan_signs, reverse=True)


# ---------------------------------------------------------------------------
# /features route
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_client_with_shap(conn, tmp_path):
    """Flask test client wired to a minimal shap_dir with summary.csv."""
    import csv
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=14)
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    (shap_dir / "dependence").mkdir()

    # Write summary.csv
    with open(shap_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_shap", "rank", "r", "nan_fraction"])
        writer.writeheader()
        writer.writerow({"feature": "cycle_pct_through", "mean_abs_shap": 0.08,
                          "rank": 1, "r": 1.0, "nan_fraction": 0.0})
        writer.writerow({"feature": "stickiness_score", "mean_abs_shap": 0.03,
                          "rank": 2, "r": -1.0, "nan_fraction": 0.2})

    # Write a dummy PNG for one feature
    (shap_dir / "dependence" / "cycle_pct_through.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 20  # minimal PNG-like header
    )

    app = _create_app(
        conn,
        cd=None,
        today="2024-01-14",
        cycle_state=None,
        peak_data={},
        summary=db_summary(conn),
        boundaries={},
        shap_dir=shap_dir,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_features_route_returns_200(flask_client_with_shap):
    resp = flask_client_with_shap.get("/features")
    assert resp.status_code == 200


def test_features_route_shows_ranked_table(flask_client_with_shap):
    resp = flask_client_with_shap.get("/features")
    html = resp.data.decode()
    assert "cycle_pct_through" in html
    assert "stickiness_score" in html


def test_features_route_no_artifact_shows_banner(conn, tmp_path):
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=14)
    empty_dir = tmp_path / "empty_shap"
    empty_dir.mkdir()
    app = _create_app(
        conn,
        cd=None,
        today="2024-01-14",
        cycle_state=None,
        peak_data={},
        summary=db_summary(conn),
        boundaries={},
        shap_dir=empty_dir,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get("/features")
    assert resp.status_code == 200
    assert b"summary.csv" in resp.data


def test_features_route_sort_alpha(flask_client_with_shap):
    resp = flask_client_with_shap.get("/features?sort=alpha")
    html = resp.data.decode()
    # Both features should appear
    assert "cycle_pct_through" in html
    assert "stickiness_score" in html


def test_features_route_drill_down_shows_plot(flask_client_with_shap):
    resp = flask_client_with_shap.get("/features?feature=cycle_pct_through")
    html = resp.data.decode()
    assert "/features/plot/cycle_pct_through" in html


def test_features_route_unknown_feature_ignored(flask_client_with_shap):
    resp = flask_client_with_shap.get("/features?feature=nonexistent_feature")
    assert resp.status_code == 200
    assert b"/features/plot/nonexistent_feature" not in resp.data


def test_features_plot_serves_png(flask_client_with_shap):
    resp = flask_client_with_shap.get("/features/plot/cycle_pct_through")
    assert resp.status_code == 200
    assert resp.content_type == "image/png"


def test_features_plot_missing_returns_404(flask_client_with_shap):
    resp = flask_client_with_shap.get("/features/plot/stickiness_score")
    assert resp.status_code == 404


def test_features_in_nav(flask_client_with_shap):
    resp = flask_client_with_shap.get("/features")
    html = resp.data.decode()
    assert 'href="/features"' in html


# ---------------------------------------------------------------------------
# _load_partner_scores
# ---------------------------------------------------------------------------

def test_load_partner_scores_returns_none_when_missing(tmp_path):
    assert _load_partner_scores(tmp_path / "no_such_dir") is None


def test_load_partner_scores_returns_none_when_csv_absent(tmp_path):
    (tmp_path / "shap").mkdir()
    assert _load_partner_scores(tmp_path / "shap") is None


def test_load_partner_scores_returns_dict(tmp_path):
    import csv
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    with open(shap_dir / "partner_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "partner", "score", "pct_of_top", "pct_of_total"])
        writer.writeheader()
        writer.writerow({
            "feature": "feat_a", "partner": "feat_b", "score": 2.0, "pct_of_top": 1.0, "pct_of_total": 0.6
        })
        writer.writerow({
            "feature": "feat_a", "partner": "feat_c", "score": 1.0, "pct_of_top": 0.5, "pct_of_total": 0.3
        })
        writer.writerow({
            "feature": "feat_b", "partner": "feat_a", "score": 1.5, "pct_of_top": 1.0, "pct_of_total": 1.0
        })
    result = _load_partner_scores(shap_dir)
    assert result is not None
    assert "feat_a" in result
    assert len(result["feat_a"]) == 2
    assert result["feat_a"][0]["partner"] == "feat_b"  # sorted desc by score


# ---------------------------------------------------------------------------
# _apply_hybrid_cutoff
# ---------------------------------------------------------------------------

def _make_partners(scores):
    top = max(scores)
    total = sum(scores)
    return [
        {"partner": f"p{i}", "score": s, "pct_of_top": s / top, "pct_of_total": s / total}
        for i, s in enumerate(scores)
    ]


def test_apply_hybrid_cutoff_returns_top_n_when_wider():
    partners = _make_partners([10, 9, 8, 7, 6, 5, 4, 3])  # threshold (>=50%) gives 2; top-6 gives 6
    result = _apply_hybrid_cutoff(partners, top_n=6, threshold_pct=0.5)
    assert len(result) == 6


def test_apply_hybrid_cutoff_returns_threshold_when_wider():
    # All 8 scores are >= 80% of top → threshold set has 8; top-6 has 6
    partners = _make_partners([10, 9, 9, 8, 8, 9, 9, 8])
    result = _apply_hybrid_cutoff(partners, top_n=6, threshold_pct=0.5)
    assert len(result) == 8


def test_apply_hybrid_cutoff_empty_input():
    assert _apply_hybrid_cutoff([]) == []


# ---------------------------------------------------------------------------
# _compute_interaction_budget_ranks
# ---------------------------------------------------------------------------

def test_compute_interaction_budget_ranks():
    scores = {
        "feat_a": [{"score": 3.0}, {"score": 2.0}],  # total=5
        "feat_b": [{"score": 1.0}],                    # total=1
        "feat_c": [{"score": 4.0}, {"score": 1.0}],  # total=5
    }
    ranks = _compute_interaction_budget_ranks(scores)
    # feat_a and feat_c both total 5, feat_b totals 1
    assert ranks["feat_b"][0] == 3  # lowest
    assert ranks["feat_b"][1] == 3  # 3 features
    # feat_a and feat_c should be ranks 1 and 2 (order may vary)
    assert ranks["feat_a"][0] in (1, 2)
    assert ranks["feat_c"][0] in (1, 2)


# ---------------------------------------------------------------------------
# /features route: partner dropdown and interaction
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_client_with_shap_and_partners(conn, tmp_path):
    """Flask test client with summary.csv and partner_scores.csv."""
    import csv
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=14)
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    dep_dir = shap_dir / "dependence"
    dep_dir.mkdir()

    with open(shap_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_shap", "rank", "r", "nan_fraction"])
        writer.writeheader()
        writer.writerow({
            "feature": "cycle_pct_through", "mean_abs_shap": 0.08, "rank": 1, "r": 0.83, "nan_fraction": 0.0
        })
        writer.writerow({
            "feature": "stickiness_score", "mean_abs_shap": 0.03, "rank": 2, "r": -0.62, "nan_fraction": 0.2
        })

    with open(shap_dir / "partner_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "partner", "score", "pct_of_top", "pct_of_total"])
        writer.writeheader()
        writer.writerow({"feature": "cycle_pct_through", "partner": "stickiness_score",
                          "score": 2.0, "pct_of_top": 1.0, "pct_of_total": 0.7})
        writer.writerow({"feature": "stickiness_score", "partner": "cycle_pct_through",
                          "score": 1.5, "pct_of_top": 1.0, "pct_of_total": 1.0})

    (dep_dir / "cycle_pct_through.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    (dep_dir / "stickiness_score.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

    app = _create_app(
        conn, cd=None, today="2024-01-14", cycle_state=None,
        peak_data={}, summary=db_summary(conn), boundaries={}, shap_dir=shap_dir,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_features_route_shows_partners_dropdown(flask_client_with_shap_and_partners):
    resp = flask_client_with_shap_and_partners.get("/features?feature=cycle_pct_through")
    html = resp.data.decode()
    assert "<select" in html
    assert "stickiness_score" in html


def test_features_route_missing_partner_scores_shows_banner(conn, tmp_path):
    import csv
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=14)
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    with open(shap_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_shap", "rank", "r", "nan_fraction"])
        writer.writeheader()
        writer.writerow({"feature": "feat_a", "mean_abs_shap": 0.05, "rank": 1, "r": 0.5, "nan_fraction": 0.0})
    app = _create_app(
        conn, cd=None, today="2024-01-14", cycle_state=None,
        peak_data={}, summary=db_summary(conn), boundaries={}, shap_dir=shap_dir,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get("/features")
    assert resp.status_code == 200
    assert b"partner_scores.csv" in resp.data


def test_features_route_interaction_param_renders_interaction_url(flask_client_with_shap_and_partners):
    resp = flask_client_with_shap_and_partners.get(
        "/features?feature=cycle_pct_through&interaction=stickiness_score"
    )
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "interaction=stickiness_score" in html


def test_features_route_active_interaction_shows_reset_link(flask_client_with_shap_and_partners):
    resp = flask_client_with_shap_and_partners.get(
        "/features?feature=cycle_pct_through&interaction=stickiness_score"
    )
    html = resp.data.decode()
    assert "Reset to auto" in html


def test_features_route_no_interaction_no_reset_link(flask_client_with_shap_and_partners):
    resp = flask_client_with_shap_and_partners.get(
        "/features?feature=cycle_pct_through"
    )
    html = resp.data.decode()
    assert "Reset to auto" not in html


def test_features_route_interaction_budget_rank_in_side_panel(flask_client_with_shap_and_partners):
    resp = flask_client_with_shap_and_partners.get(
        "/features?feature=cycle_pct_through"
    )
    html = resp.data.decode()
    assert "interaction-budget rank" in html


def test_features_route_staleness_banner(conn, tmp_path):
    import csv
    import time
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=14)
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    (shap_dir / "dependence").mkdir()
    # Write shap_values.npy first (older)
    sv_path = shap_dir / "shap_values.npy"
    sv_path.write_bytes(b"\x00" * 8)
    time.sleep(0.05)
    # Write model file after (newer)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"\x00" * 8)
    with open(shap_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_shap", "rank", "r", "nan_fraction"])
        writer.writeheader()
        writer.writerow({"feature": "feat_a", "mean_abs_shap": 0.05, "rank": 1, "r": 0.5, "nan_fraction": 0.0})
    app = _create_app(
        conn, cd=None, today="2024-01-14", cycle_state=None,
        peak_data={}, summary=db_summary(conn), boundaries={},
        shap_dir=shap_dir, model_path=model_path,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get("/features")
    assert resp.status_code == 200
    assert b"Stale SHAP" in resp.data


def test_features_plot_interaction_missing_arrays_returns_503(conn, tmp_path):
    import csv
    upsert_stations(conn, [_STATION_BM])
    _insert_prices(conn, 1001, n_days=14)
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    (shap_dir / "dependence").mkdir()
    # summary.csv exists but no X_val.npy / shap_values.npy / feature_columns.json
    with open(shap_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_shap", "rank", "r", "nan_fraction"])
        writer.writeheader()
        writer.writerow({"feature": "feat_a", "mean_abs_shap": 0.05, "rank": 1, "r": 0.5, "nan_fraction": 0.0})
    app = _create_app(
        conn, cd=None, today="2024-01-14", cycle_state=None,
        peak_data={}, summary=db_summary(conn), boundaries={}, shap_dir=shap_dir,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get("/features/plot/feat_a?interaction=feat_b")
    assert resp.status_code == 503
