import csv
import pathlib

from click.testing import CliRunner

from fuel_signal.snapshot_retire import (
    asof_price,
    build_historical_index,
    committed_snapshot_months,
    compare_month,
    main,
    parse_resource_month,
)

# ---------------------------------------------------------------------------
# parse_resource_month
# ---------------------------------------------------------------------------

def test_parse_resource_month_full_month_four_digit_year():
    assert parse_resource_month("fuelcheck_pricehistory_apr2026.xlsx") == (2026, 4)


def test_parse_resource_month_full_name_two_digit_year():
    assert parse_resource_month("price_history_checks_june23.xlsx") == (2023, 6)


def test_parse_resource_month_sept_variant():
    assert parse_resource_month("price_history_checks_sept2025.csv") == (2025, 9)


def test_parse_resource_month_unparseable_returns_none():
    assert parse_resource_month("service-station-history.xlsx") is None


# ---------------------------------------------------------------------------
# committed_snapshot_months
# ---------------------------------------------------------------------------

def test_committed_snapshot_months_finds_populated_dirs(tmp_path):
    (tmp_path / "2026" / "04").mkdir(parents=True)
    (tmp_path / "2026" / "04" / "2026-04-27.csv").write_text("x")
    (tmp_path / "2026" / "08").mkdir(parents=True)  # empty — no csv

    months = committed_snapshot_months(tmp_path)
    assert months == {(2026, 4): tmp_path / "2026" / "04"}


# ---------------------------------------------------------------------------
# build_historical_index / asof_price
# ---------------------------------------------------------------------------

def _write_cleaned_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fieldnames = ["ServiceStationName", "Address", "Suburb", "Postcode",
                  "Brand", "FuelCode", "PriceUpdatedDate", "Price"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_asof_price_picks_most_recent_event_at_or_before_date(tmp_path):
    cleaned = tmp_path / "cleaned.csv"
    _write_cleaned_csv(cleaned, [
        {"ServiceStationName": "BP Test", "Address": "1 Main St, Sydney NSW 2000",
         "Suburb": "Sydney", "Postcode": "2000", "Brand": "BP", "FuelCode": "E10",
         "PriceUpdatedDate": "2026-04-01 09:00:00", "Price": "180.0"},
        {"ServiceStationName": "BP Test", "Address": "1 Main St, Sydney NSW 2000",
         "Suburb": "Sydney", "Postcode": "2000", "Brand": "BP", "FuelCode": "E10",
         "PriceUpdatedDate": "2026-04-05 09:00:00", "Price": "190.0"},
    ])
    idx = build_historical_index(cleaned)
    addr = "1 main street sydney"

    assert asof_price(idx, addr, "E10", "2026-04-03") == 180.0
    assert asof_price(idx, addr, "E10", "2026-04-05") == 190.0
    assert asof_price(idx, addr, "E10", "2026-04-10") == 190.0
    assert asof_price(idx, addr, "E10", "2026-03-30") is None
    assert asof_price(idx, "nowhere", "E10", "2026-04-10") is None


# ---------------------------------------------------------------------------
# compare_month
# ---------------------------------------------------------------------------

def _write_snapshot_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fieldnames = ["station_code", "name", "address", "suburb", "postcode",
                  "brand", "fuel_code", "price", "date"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_compare_month_agreement_and_divergence(tmp_path):
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    _write_snapshot_csv(snapshot_dir / "2026-04-01.csv", [
        {"station_code": 1, "name": "BP Test", "address": "1 Main St, Sydney NSW 2000",
         "suburb": "Sydney", "postcode": "2000", "brand": "BP", "fuel_code": "E10",
         "price": "180.0", "date": "2026-04-01"},
        {"station_code": 2, "name": "Shell Test", "address": "2 Other St, Sydney NSW 2000",
         "suburb": "Sydney", "postcode": "2000", "brand": "Shell", "fuel_code": "E10",
         "price": "175.0", "date": "2026-04-01"},
    ])

    cleaned = tmp_path / "cleaned.csv"
    _write_cleaned_csv(cleaned, [
        {"ServiceStationName": "BP Test", "Address": "1 Main St, Sydney NSW 2000",
         "Suburb": "Sydney", "Postcode": "2000", "Brand": "BP", "FuelCode": "E10",
         "PriceUpdatedDate": "2026-04-01 20:00:00", "Price": "180.0"},
        {"ServiceStationName": "Shell Test", "Address": "2 Other St, Sydney NSW 2000",
         "Suburb": "Sydney", "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2026-04-01 20:00:00", "Price": "169.0"},
    ])

    report = compare_month(2026, 4, snapshot_dir, cleaned, tolerance=0.05)
    assert report.rows_total == 2
    assert report.rows_comparable == 2
    assert report.rows_agree == 1
    assert report.agreement_ratio == 0.5
    assert len(report.diverging) == 1
    assert report.diverging[0][:3] == ("2026-04-01", "Shell Test", "E10")
    assert not report.eligible(min_agreement=0.95)
    assert report.eligible(min_agreement=0.5)


def test_compare_month_no_historical_coverage_is_not_eligible(tmp_path):
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    _write_snapshot_csv(snapshot_dir / "2026-04-01.csv", [
        {"station_code": 1, "name": "BP Test", "address": "1 Main St, Sydney NSW 2000",
         "suburb": "Sydney", "postcode": "2000", "brand": "BP", "fuel_code": "E10",
         "price": "180.0", "date": "2026-04-01"},
    ])
    cleaned = tmp_path / "cleaned.csv"
    _write_cleaned_csv(cleaned, [])

    report = compare_month(2026, 4, snapshot_dir, cleaned, tolerance=0.05)
    assert report.rows_comparable == 0
    assert not report.eligible(min_agreement=0.95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_reports_and_applies_deletion(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "snapshots" / "2026" / "04"
    snapshot_dir.mkdir(parents=True)
    snap_file = snapshot_dir / "2026-04-01.csv"
    _write_snapshot_csv(snap_file, [
        {"station_code": 1, "name": "BP Test", "address": "1 Main St, Sydney NSW 2000",
         "suburb": "Sydney", "postcode": "2000", "brand": "BP", "fuel_code": "E10",
         "price": "180.0", "date": "2026-04-01"},
    ])

    cleaned = tmp_path / "cleaned.csv"
    _write_cleaned_csv(cleaned, [
        {"ServiceStationName": "BP Test", "Address": "1 Main St, Sydney NSW 2000",
         "Suburb": "Sydney", "Postcode": "2000", "Brand": "BP", "FuelCode": "E10",
         "PriceUpdatedDate": "2026-04-01 20:00:00", "Price": "180.0"},
    ])

    monkeypatch.setattr(
        "fuel_signal.snapshot_retire.find_candidate_months",
        lambda raw_dir, cleaned_dir, snapshots_dir: [(2026, 4, snapshot_dir, cleaned)],
    )

    runner = CliRunner()
    # Dry run: report only, file must survive.
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "ELIGIBLE" in result.output
    assert snap_file.exists()

    # --apply: file should be deleted.
    result = runner.invoke(main, ["--apply"])
    assert result.exit_code == 0
    assert not snap_file.exists()


def test_main_no_candidates(monkeypatch):
    monkeypatch.setattr(
        "fuel_signal.snapshot_retire.find_candidate_months",
        lambda raw_dir, cleaned_dir, snapshots_dir: [],
    )
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "No committed snapshot months" in result.output
