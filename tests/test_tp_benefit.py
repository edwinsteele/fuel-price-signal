"""Tests for fuel_signal.tp_benefit."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest
from click.testing import CliRunner

from fuel_signal.db import create_schema, open_db, upsert_daily_prices, upsert_stations
from fuel_signal.tp_benefit import (
    compute_tp_benefit,
    format_summary,
    main,
    plot_tp_distribution,
)

STATION = 999
FUEL = "E10"


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    upsert_stations(c, [{"station_code": STATION, "address": "999 Test Street, Testville",
                         "suburb": "Testville", "postcode": "2000", "name": "Test Station",
                         "brand": "TestBrand"}])
    yield c
    c.close()


def _dates(start: str, n: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n)]


def _load_prices(conn, start: str, prices_cents: list[float]) -> None:
    rows = [
        (STATION, FUEL, d, p)
        for d, p in zip(_dates(start, len(prices_cents)), prices_cents)
    ]
    upsert_daily_prices(conn, rows)
    conn.commit()


def _features_df(label1_dates: list[str], today_price: float = 160.0) -> pd.DataFrame:
    return pd.DataFrame([
        {"station_code": STATION, "price_date": d, "today_price_cents": today_price, "label": 1}
        for d in label1_dates
    ])


# ---------------------------------------------------------------------------
# compute_tp_benefit
# ---------------------------------------------------------------------------

def test_positive_benefit_prices_rose(conn):
    # today=160, next 7 days all 170 → benefit = 170 - 160 = 10
    _load_prices(conn, "2024-01-01", [160.0] + [170.0] * 7)
    df = _features_df(["2024-01-01"])
    tp = compute_tp_benefit(conn, df, horizon_days=7)
    assert len(tp) == 1
    assert tp["benefit"].iloc[0] == pytest.approx(10.0)


def test_zero_benefit_flat_prices(conn):
    # today=160, next 7 days all 160 → benefit = 0
    _load_prices(conn, "2024-01-01", [160.0] * 8)
    df = _features_df(["2024-01-01"])
    tp = compute_tp_benefit(conn, df, horizon_days=7)
    assert tp["benefit"].iloc[0] == pytest.approx(0.0)


def test_negative_benefit_prices_fell(conn):
    # today=160, next 7 days all 150 → benefit = -10 (buying today was worse)
    _load_prices(conn, "2024-01-01", [160.0] + [150.0] * 7)
    df = _features_df(["2024-01-01"])
    tp = compute_tp_benefit(conn, df, horizon_days=7)
    assert tp["benefit"].iloc[0] == pytest.approx(-10.0)


def test_benefit_is_average_not_endpoint(conn):
    # today=160; next 7 days: 4 days at 170 + 3 days at 180 → avg = (4*170+3*180)/7
    _load_prices(conn, "2024-01-01", [160.0] + [170.0] * 4 + [180.0] * 3)
    df = _features_df(["2024-01-01"])
    tp = compute_tp_benefit(conn, df, horizon_days=7)
    expected_avg = (4 * 170.0 + 3 * 180.0) / 7
    assert tp["benefit"].iloc[0] == pytest.approx(expected_avg - 160.0)


def test_excludes_label_0_rows(conn):
    _load_prices(conn, "2024-01-01", [160.0] * 8)
    df = pd.DataFrame([
        {"station_code": STATION, "price_date": "2024-01-01", "today_price_cents": 160.0, "label": 0},
        {"station_code": STATION, "price_date": "2024-01-01", "today_price_cents": 160.0, "label": 1},
    ])
    tp = compute_tp_benefit(conn, df, horizon_days=7)
    assert len(tp) == 1


def test_missing_future_price_drops_row(conn):
    # Only 3 days of data; horizon=7 requires 7 future days → row dropped
    _load_prices(conn, "2024-01-01", [160.0] * 3)
    df = _features_df(["2024-01-01"])
    tp = compute_tp_benefit(conn, df, horizon_days=7)
    assert len(tp) == 0


def test_partial_future_coverage_drops_row(conn):
    # 5 future days available but horizon=7 → row dropped (strict coverage required)
    _load_prices(conn, "2024-01-01", [160.0] * 6)
    df = _features_df(["2024-01-01"])
    tp = compute_tp_benefit(conn, df, horizon_days=7)
    assert len(tp) == 0


def test_multiple_rows(conn):
    # Prices: [160, 170, 170, 170, 170, 170, 170, 170, 180] (indices 0-8)
    # Row 0 (2024-01-01): horizon days 1-7 = indices 1-7 → all 170, avg=170, benefit=10
    # Row 1 (2024-01-02): horizon days 1-7 = indices 2-8 → 6*170+180, avg≈171.43, benefit≈11.43
    _load_prices(conn, "2024-01-01", [160.0] + [170.0] * 7 + [180.0])
    df = _features_df(["2024-01-01", "2024-01-02"])
    tp = compute_tp_benefit(conn, df, horizon_days=7)
    assert len(tp) == 2
    assert tp["benefit"].iloc[0] == pytest.approx(170.0 - 160.0)
    expected_avg_row1 = (6 * 170.0 + 180.0) / 7
    assert tp["benefit"].iloc[1] == pytest.approx(expected_avg_row1 - 160.0)


def test_custom_horizon(conn):
    # horizon=3; next 3 days avg = (170+180+190)/3 = 180
    _load_prices(conn, "2024-01-01", [160.0, 170.0, 180.0, 190.0])
    df = _features_df(["2024-01-01"])
    tp = compute_tp_benefit(conn, df, horizon_days=3)
    assert tp["benefit"].iloc[0] == pytest.approx(180.0 - 160.0)


def test_empty_features_df(conn):
    df = pd.DataFrame(columns=["station_code", "price_date", "today_price_cents", "label"])
    tp = compute_tp_benefit(conn, df)
    assert tp.empty


def test_zero_horizon_raises(conn):
    df = pd.DataFrame(columns=["station_code", "price_date", "today_price_cents", "label"])
    with pytest.raises(ValueError, match="horizon_days"):
        compute_tp_benefit(conn, df, horizon_days=0)


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

def test_format_summary_smoke(conn):
    _load_prices(conn, "2024-01-01", [160.0] + [170.0] * 7)
    tp = compute_tp_benefit(conn, _features_df(["2024-01-01"]))
    out = format_summary(tp)
    assert "Suggested TP reward" in out
    assert "trimmed mean" in out
    assert "10.00c" in out


def test_format_summary_empty():
    tp = pd.DataFrame(columns=["station_code", "price_date", "today_price_cents", "benefit"])
    out = format_summary(tp)
    assert "No data" in out


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def test_plot_no_crash(conn, tmp_path):
    _load_prices(conn, "2024-01-01", [160.0] + [170.0] * 7)
    tp = compute_tp_benefit(conn, _features_df(["2024-01-01"]))
    plot_tp_distribution(tp, out_path=tmp_path / "out.png")
    assert (tmp_path / "out.png").exists()


def test_plot_empty_no_crash(tmp_path):
    tp = pd.DataFrame(columns=["station_code", "price_date", "today_price_cents", "benefit"])
    result = plot_tp_distribution(tp, out_path=tmp_path / "out.png")
    assert result is None
    assert not (tmp_path / "out.png").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_missing_features_csv(tmp_path):
    result = CliRunner().invoke(main, [
        "--features-csv", str(tmp_path / "missing.csv"),
        "--db", str(tmp_path / "fuel_signal.db"),
    ])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_cli_missing_db(tmp_path):
    csv = tmp_path / "features.csv"
    pd.DataFrame(columns=["station_code", "price_date", "today_price_cents", "label"]).to_csv(csv, index=False)
    result = CliRunner().invoke(main, [
        "--features-csv", str(csv),
        "--db", str(tmp_path / "missing.db"),
    ])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_cli_empty_result_skips_plot(tmp_path):
    db_path = tmp_path / "fuel_signal.db"
    conn = open_db(db_path)
    create_schema(conn)
    upsert_stations(conn, [{"station_code": STATION, "address": "999 Test Street, Testville",
                            "suburb": "Testville", "postcode": "2000", "name": "Test Station",
                            "brand": "TestBrand"}])
    _load_prices(conn, "2024-01-01", [160.0] * 3)  # only 3 days — horizon=7 not satisfied
    conn.close()

    csv_path = tmp_path / "features.csv"
    pd.DataFrame([
        {"station_code": STATION, "price_date": "2024-01-01", "today_price_cents": 160.0, "label": 1},
    ]).to_csv(csv_path, index=False)

    plot_path = tmp_path / "out.png"
    result = CliRunner().invoke(main, [
        "--features-csv", str(csv_path),
        "--db", str(db_path),
        "--plot", str(plot_path),
    ])
    assert result.exit_code == 0, result.output
    assert not plot_path.exists()
    assert "skipped plot" in result.output.lower()


def test_cli_writes_plot(tmp_path):
    db_path = tmp_path / "fuel_signal.db"
    conn = open_db(db_path)
    create_schema(conn)
    upsert_stations(conn, [{"station_code": STATION, "address": "999 Test Street, Testville",
                            "suburb": "Testville", "postcode": "2000", "name": "Test Station",
                            "brand": "TestBrand"}])
    _load_prices(conn, "2024-01-01", [160.0] + [170.0] * 14)
    conn.close()

    csv_path = tmp_path / "features.csv"
    pd.DataFrame([
        {"station_code": STATION, "price_date": "2024-01-01", "today_price_cents": 160.0, "label": 1},
        {"station_code": STATION, "price_date": "2024-01-02", "today_price_cents": 160.0, "label": 0},
    ]).to_csv(csv_path, index=False)

    plot_path = tmp_path / "out.png"
    result = CliRunner().invoke(main, [
        "--features-csv", str(csv_path),
        "--db", str(db_path),
        "--plot", str(plot_path),
    ])
    assert result.exit_code == 0, result.output
    assert plot_path.exists()
