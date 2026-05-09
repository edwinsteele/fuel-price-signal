"""Tests for fuel_signal.fn_cost."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest
from click.testing import CliRunner

from fuel_signal.db import create_schema, open_db, upsert_daily_prices, upsert_stations
from fuel_signal.fn_cost import (
    compute_fn_damage,
    format_summary,
    main,
    plot_fn_distribution,
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
# compute_fn_damage
# ---------------------------------------------------------------------------

def test_positive_damage_price_rose(conn):
    # Today=160, day+7=170 → damage=+10
    _load_prices(conn, "2024-01-01", [160.0] * 7 + [170.0])
    df = _features_df(["2024-01-01"])
    fn = compute_fn_damage(conn, df, delay_days=7)
    assert len(fn) == 1
    assert fn["damage"].iloc[0] == pytest.approx(10.0)


def test_negative_damage_price_fell(conn):
    # Today=160, day+7=150 → damage=−10
    _load_prices(conn, "2024-01-01", [160.0] * 7 + [150.0])
    df = _features_df(["2024-01-01"])
    fn = compute_fn_damage(conn, df, delay_days=7)
    assert fn["damage"].iloc[0] == pytest.approx(-10.0)


def test_zero_damage_price_unchanged(conn):
    _load_prices(conn, "2024-01-01", [160.0] * 8)
    df = _features_df(["2024-01-01"])
    fn = compute_fn_damage(conn, df, delay_days=7)
    assert fn["damage"].iloc[0] == pytest.approx(0.0)


def test_excludes_label_0_rows(conn):
    _load_prices(conn, "2024-01-01", [160.0] * 8)
    df = pd.DataFrame([
        {"station_code": STATION, "price_date": "2024-01-01", "today_price_cents": 160.0, "label": 0},
        {"station_code": STATION, "price_date": "2024-01-01", "today_price_cents": 160.0, "label": 1},
    ])
    fn = compute_fn_damage(conn, df, delay_days=7)
    assert len(fn) == 1


def test_missing_future_price_dropped(conn):
    # Only 3 days of data; day+7 doesn't exist → row dropped
    _load_prices(conn, "2024-01-01", [160.0] * 3)
    df = _features_df(["2024-01-01"])
    fn = compute_fn_damage(conn, df, delay_days=7)
    assert len(fn) == 0


def test_multiple_rows(conn):
    # Both rows use today_price=160.0 (helper default); day+7 for row 0 is index 7,
    # day+7 for row 1 is index 8.
    _load_prices(conn, "2024-01-01", [160.0, 160.0, 160.0, 160.0, 160.0, 160.0, 160.0, 175.0, 180.0])
    df = _features_df(["2024-01-01", "2024-01-02"])
    fn = compute_fn_damage(conn, df, delay_days=7)
    assert len(fn) == 2
    assert fn["damage"].iloc[0] == pytest.approx(175.0 - 160.0)
    assert fn["damage"].iloc[1] == pytest.approx(180.0 - 160.0)


def test_custom_delay(conn):
    # day+14=180 → damage=20
    _load_prices(conn, "2024-01-01", [160.0] * 14 + [180.0])
    df = _features_df(["2024-01-01"])
    fn = compute_fn_damage(conn, df, delay_days=14)
    assert fn["damage"].iloc[0] == pytest.approx(20.0)


def test_empty_features_df(conn):
    df = pd.DataFrame(columns=["station_code", "price_date", "today_price_cents", "label"])
    fn = compute_fn_damage(conn, df)
    assert fn.empty


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

def test_format_summary_smoke(conn):
    _load_prices(conn, "2024-01-01", [160.0] * 7 + [170.0])
    fn = compute_fn_damage(conn, _features_df(["2024-01-01"]))
    out = format_summary(fn)
    assert "Suggested FN penalty" in out
    assert "10.00c" in out


def test_format_summary_empty():
    fn = pd.DataFrame(columns=["station_code", "price_date", "today_price_cents", "damage"])
    out = format_summary(fn)
    assert "No data" in out


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def test_plot_no_crash(conn, tmp_path):
    _load_prices(conn, "2024-01-01", [160.0] * 7 + [170.0])
    fn = compute_fn_damage(conn, _features_df(["2024-01-01"]))
    plot_fn_distribution(fn, out_path=tmp_path / "out.png")
    assert (tmp_path / "out.png").exists()


def test_plot_empty_no_crash(tmp_path):
    fn = pd.DataFrame(columns=["station_code", "price_date", "today_price_cents", "damage"])
    plot_fn_distribution(fn, out_path=tmp_path / "out.png")
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


def test_cli_writes_plot(tmp_path):
    db_path = tmp_path / "fuel_signal.db"
    conn = open_db(db_path)
    create_schema(conn)
    upsert_stations(conn, [{"station_code": STATION, "address": "999 Test Street, Testville",
                            "suburb": "Testville", "postcode": "2000", "name": "Test Station",
                            "brand": "TestBrand"}])
    _load_prices(conn, "2024-01-01", [160.0] * 7 + [170.0] * 10)
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
