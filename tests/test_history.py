import io
import pathlib
from datetime import datetime

import openpyxl
import responses as rsps

from fuel_signal.history import (
    ResourceFetcher,
    Transformer,
    _format_cell,
    clean_all_resources,
    discover_price_resources,
    download_all,
)

DATASET_URL = "https://data.nsw.gov.au/data/dataset/fuel-check"
DUMP_URL = "https://data.nsw.gov.au/data/datastore/dump/"
RESOURCE_PAGE_URL = "https://data.nsw.gov.au/data/dataset/fuel-check/resource/"

DATASET_HTML = """
<html><body>
<section id="dataset-resources">
  <div class="resource-item" data-id="aaa-111">
    <a href="http://example.com/service-station-and-price-history-jan-2024.xlsx"
       class="resource-url-analytics">Download</a>
  </div>
  <div class="resource-item" data-id="bbb-222">
    <a href="http://www.fairtrading.nsw.gov.au/faq"
       class="resource-url-analytics">FAQ</a>
  </div>
  <div class="resource-item" data-id="ccc-333">
    <a href="http://example.com/pricehistory-feb-2024.xlsx"
       class="resource-url-analytics">Download</a>
  </div>
</section>
</body></html>
"""

SAMPLE_CSV_BODY = (
    "_id,ServiceStationName,Address,Suburb,Postcode,Brand,FuelCode,PriceUpdatedDate,Price\n"
    "1,Shell,1 Main St,Sydney,2000,Shell,E10,2024-01-15T00:00:00,180.0\n"
)


def _make_xlsx(rows: list[list]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_raw_csv(rows: list[dict], tmp_path, name="test.csv") -> pathlib.Path:
    import csv
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ServiceStationName", "Address", "Suburb", "Postcode",
            "Brand", "FuelCode", "PriceUpdatedDate", "Price",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return p


# ---------------------------------------------------------------------------
# discover_price_resources
# ---------------------------------------------------------------------------

@rsps.activate
def test_discover_filters_non_price_resources():
    rsps.add(rsps.GET, DATASET_URL, body=DATASET_HTML)
    resources = discover_price_resources()
    ids = [r["id"] for r in resources]
    assert ids == ["aaa-111", "ccc-333"]


@rsps.activate
def test_discover_returns_download_urls():
    rsps.add(rsps.GET, DATASET_URL, body=DATASET_HTML)
    resources = discover_price_resources()
    assert resources[0]["download_url"] == (
        "http://example.com/service-station-and-price-history-jan-2024.xlsx"
    )


# ---------------------------------------------------------------------------
# ResourceFetcher — CSV path
# ---------------------------------------------------------------------------

@rsps.activate
def test_fetch_csv_success(tmp_path):
    rid = "test-uuid-csv"
    rsps.add(rsps.GET, DUMP_URL + rid, body=SAMPLE_CSV_BODY.encode())
    result = ResourceFetcher(rid, "http://example.com/price.xlsx", tmp_path).fetch()
    assert result.exists()
    assert result.name == f"{rid}.csv"
    assert "Shell" in result.read_text()


@rsps.activate
def test_fetch_skips_existing_file(tmp_path):
    rid = "test-uuid-existing"
    existing = tmp_path / f"{rid}.csv"
    existing.write_text("already here")
    ResourceFetcher(rid, "http://example.com/price.xlsx", tmp_path).fetch()
    assert existing.read_text() == "already here"
    assert len(rsps.calls) == 0


# ---------------------------------------------------------------------------
# ResourceFetcher — XLSX fallback
# ---------------------------------------------------------------------------

@rsps.activate
def test_fetch_falls_back_to_xlsx_on_csv_404(tmp_path):
    rid = "test-uuid-xlsx"
    xlsx_url = "http://example.com/pricehistory.xlsx"
    rsps.add(rsps.GET, DUMP_URL + rid, status=404)
    xlsx_data = _make_xlsx([
        ["ServiceStationName", "Address", "Suburb", "Postcode", "Brand", "FuelCode", "PriceUpdatedDate", "Price"],
        ["Shell", "1 Main St", "Sydney", "2000", "Shell", "E10", "2024-01-15", "180.0"],
    ])
    rsps.add(rsps.GET, xlsx_url, body=xlsx_data)
    result = ResourceFetcher(rid, xlsx_url, tmp_path).fetch()
    content = result.read_text()
    assert "Shell" in content
    assert "180.0" in content


@rsps.activate
def test_fetch_xlsx_falls_back_to_resource_page_on_bad_content(tmp_path):
    rid = "test-uuid-badzip"
    direct_url = "http://example.com/broken.xlsx"
    page_url = "http://example.com/real.xlsx"

    rsps.add(rsps.GET, DUMP_URL + rid, status=404)
    rsps.add(rsps.GET, direct_url, body=b"this is not a zip file")
    rsps.add(rsps.GET, RESOURCE_PAGE_URL + rid, body=f"""
        <html><head>
        <meta name="DCTERMS.Identifier" content="{page_url}"/>
        </head></html>
    """)
    xlsx_data = _make_xlsx([
        ["ServiceStationName", "Address", "Suburb", "Postcode", "Brand", "FuelCode", "PriceUpdatedDate", "Price"],
        ["BP", "2 High St", "Penrith", "2750", "BP", "E10", "2024-01-15", "175.0"],
    ])
    rsps.add(rsps.GET, page_url, body=xlsx_data)
    result = ResourceFetcher(rid, direct_url, tmp_path).fetch()
    assert "BP" in result.read_text()


@rsps.activate
def test_fetch_csv_falls_back_to_xlsx_on_empty_response(tmp_path):
    rid = "test-uuid-empty"
    xlsx_url = "http://example.com/price.xlsx"
    rsps.add(rsps.GET, DUMP_URL + rid, body=b"")
    xlsx_data = _make_xlsx([
        ["ServiceStationName", "Address", "Suburb", "Postcode", "Brand", "FuelCode", "PriceUpdatedDate", "Price"],
        ["Shell", "1 Main St", "Sydney", "2000", "Shell", "E10", "2024-01-15", "180.0"],
    ])
    rsps.add(rsps.GET, xlsx_url, body=xlsx_data)
    result = ResourceFetcher(rid, xlsx_url, tmp_path).fetch()
    assert result.exists()
    assert result.stat().st_size > 0


@rsps.activate
def test_fetch_direct_csv_url_when_not_in_datastore(tmp_path):
    rid = "test-uuid-directcsv"
    direct_url = "http://example.com/price_history_jul2024.csv"
    rsps.add(rsps.GET, DUMP_URL + rid, status=404)
    rsps.add(rsps.GET, direct_url, body=SAMPLE_CSV_BODY.encode())
    result = ResourceFetcher(rid, direct_url, tmp_path).fetch()
    assert "Shell" in result.read_text()


@rsps.activate
def test_fetch_xlsx_skips_duplicate_header_rows(tmp_path):
    rid = "test-uuid-dupeheader"
    xlsx_url = "http://example.com/pricehistory.xlsx"
    rsps.add(rsps.GET, DUMP_URL + rid, status=404)
    xlsx_data = _make_xlsx([
        ["ServiceStationName", "Address", "Suburb", "Postcode", "Brand", "FuelCode", "PriceUpdatedDate", "Price"],
        ["ServiceStationName", "Address", "Suburb", "Postcode", "Brand", "FuelCode", "PriceUpdatedDate", "Price"],
        ["BP", "2 High St", "Penrith", "2750", "BP", "E10", "2024-02-15", "175.0"],
    ])
    rsps.add(rsps.GET, xlsx_url, body=xlsx_data)
    result = ResourceFetcher(rid, xlsx_url, tmp_path).fetch()
    content = result.read_text()
    assert "BP" in content
    assert content.count("ServiceStationName") == 1  # header present once


@rsps.activate
def test_fetch_xlsx_skips_blank_header_rows(tmp_path):
    rid = "test-uuid-blankheader"
    xlsx_url = "http://example.com/pricehistory.xlsx"
    rsps.add(rsps.GET, DUMP_URL + rid, status=404)
    xlsx_data = _make_xlsx([
        ["ServiceStationName", "Address", "Suburb", "Postcode", "Brand", "FuelCode", "PriceUpdatedDate", "Price"],
        [None, None, None, None, None, None, None, None],
        ["Caltex", "3 Low St", "Springwood", "2777", "Caltex", "E10", "2024-03-15", "170.0"],
    ])
    rsps.add(rsps.GET, xlsx_url, body=xlsx_data)
    result = ResourceFetcher(rid, xlsx_url, tmp_path).fetch()
    content = result.read_text()
    assert "Caltex" in content
    assert content.count("ServiceStationName") == 1


# ---------------------------------------------------------------------------
# _format_cell
# ---------------------------------------------------------------------------

def test_format_cell_datetime():
    dt = datetime(2024, 1, 15, 9, 30, 0)
    assert _format_cell(dt) == "2024-01-15T09:30:00"

def test_format_cell_none():
    assert _format_cell(None) == ""

def test_format_cell_string():
    assert _format_cell("Shell") == "Shell"

def test_format_cell_number():
    assert _format_cell(180.0) == "180.0"


# ---------------------------------------------------------------------------
# download_all
# ---------------------------------------------------------------------------

@rsps.activate
def test_download_all_creates_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    rsps.add(rsps.GET, DATASET_URL, body=DATASET_HTML)
    rsps.add(rsps.GET, DUMP_URL + "aaa-111", body=SAMPLE_CSV_BODY.encode())
    rsps.add(rsps.GET, DUMP_URL + "ccc-333", body=SAMPLE_CSV_BODY.encode())
    download_all(raw_dir)
    assert raw_dir.exists()


@rsps.activate
def test_download_all_returns_one_path_per_resource(tmp_path):
    rsps.add(rsps.GET, DATASET_URL, body=DATASET_HTML)
    rsps.add(rsps.GET, DUMP_URL + "aaa-111", body=SAMPLE_CSV_BODY.encode())
    rsps.add(rsps.GET, DUMP_URL + "ccc-333", body=SAMPLE_CSV_BODY.encode())
    paths = download_all(tmp_path / "raw")
    assert len(paths) == 2


# ---------------------------------------------------------------------------
# Transformer.get_month_for_file
# ---------------------------------------------------------------------------

def test_get_month_for_file_with_day_over_12(tmp_path):
    # Date is YYYY-MM-DD where day=15 > 12 → month is unambiguous
    p = _make_raw_csv([
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15", "Price": "180.0"},
    ], tmp_path)
    assert Transformer(p, tmp_path / "out.csv").get_month_for_file() == 1


def test_get_month_for_file_fallback_when_all_days_under_13(tmp_path):
    # All days <= 12, single month — falls back to last seen month
    p = _make_raw_csv([
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-09-04", "Price": "180.0"},
    ], tmp_path)
    assert Transformer(p, tmp_path / "out.csv").get_month_for_file() == 9


def test_get_month_for_file_constant_day_varying_months(tmp_path):
    # YYYY-DD-MM files where day <= 12: e.g. Oct 2019 file has 2019-01-10,
    # 2019-02-10 ... 2019-09-10 (day=10 constant, months vary) → month=10.
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": f"2019-0{m}-10", "Price": "150.0"}
        for m in range(1, 10)
    ]
    p = _make_raw_csv(rows, tmp_path)
    assert Transformer(p, tmp_path / "out.csv").get_month_for_file() == 10


def test_get_month_for_file_constant_day_nov(tmp_path):
    # Nov 2019 pattern: 2019-01-11 ... 2019-09-11 → month=11
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": f"2019-0{m}-11", "Price": "150.0"}
        for m in range(1, 10)
    ]
    p = _make_raw_csv(rows, tmp_path)
    assert Transformer(p, tmp_path / "out.csv").get_month_for_file() == 11


def test_get_month_for_file_constant_day_all_12_months(tmp_path):
    # Feb 2019 pattern: 2019-01-02 ... 2019-12-02 (all months, day=2) → month=2
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": f"2019-{m:02d}-02", "Price": "150.0"}
        for m in range(1, 13)
    ]
    p = _make_raw_csv(rows, tmp_path)
    assert Transformer(p, tmp_path / "out.csv").get_month_for_file() == 2


def test_get_month_for_file_with_time_component(tmp_path):
    p = _make_raw_csv([
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-03-20T08:00:00", "Price": "180.0"},
    ], tmp_path)
    assert Transformer(p, tmp_path / "out.csv").get_month_for_file() == 3


def test_get_month_for_file_returns_none_for_no_dates(tmp_path):
    p = _make_raw_csv([
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "", "Price": "180.0"},
    ], tmp_path)
    assert Transformer(p, tmp_path / "out.csv").get_month_for_file() is None


def test_clean_skips_file_with_no_parseable_dates(tmp_path):
    p = _make_raw_csv([
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "", "Price": "180.0"},
    ], tmp_path)
    out = tmp_path / "cleaned.csv"
    rows_written = Transformer(p, out).clean()
    assert rows_written == 0
    assert not out.exists()


# ---------------------------------------------------------------------------
# Transformer.clean_date
# ---------------------------------------------------------------------------

def _clean_date(raw, prev=None, month=1):
    return Transformer.clean_date(raw, line_number=1, prev_date=prev, month_in_file=month)


def test_clean_date_normal():
    d = _clean_date("2024-01-15", month=1)
    assert d == datetime(2024, 1, 15)


def test_clean_date_with_time():
    d = _clean_date("2024-01-15T09:30:00", month=1)
    assert d == datetime(2024, 1, 15, 9, 30, 0)


def test_clean_date_dmy_ampm():
    d = _clean_date("1/08/2022 12:06:37 AM", month=8)
    assert d == datetime(2022, 8, 1, 0, 6, 37)


def test_clean_date_dmy_pm():
    d = _clean_date("15/03/2023 2:30:00 PM", month=3)
    assert d == datetime(2023, 3, 15, 14, 30, 0)


def test_clean_date_dmy_no_time():
    d = _clean_date("5/11/2021", month=11)
    assert d == datetime(2021, 11, 5)


def test_clean_date_space_separated_datetime():
    d = _clean_date("2026-02-01 00:05:31", month=2)
    assert d == datetime(2026, 2, 1, 0, 5, 31)


def test_clean_date_dmy_hhmm_no_seconds():
    d = _clean_date("1/06/2025 0:08", month=6)
    assert d == datetime(2025, 6, 1, 0, 8)


def test_clean_date_fixes_swapped_month_day():
    # File is for January (month=1). Date "2024-01-05" has month=1 matching
    # file month, so no swap. But "2024-05-01" has month=5 != file month 1,
    # and day=1 == file month 1, so swap → 2024-01-05.
    d = _clean_date("2024-05-01", month=1)
    assert d == datetime(2024, 1, 5)


def test_clean_date_missing_uses_prev():
    prev = datetime(2024, 1, 15)
    d = _clean_date("", prev=prev, month=1)
    assert d == prev


def test_clean_date_missing_no_prev_returns_none():
    assert _clean_date("", month=1) is None


def test_clean_date_future_returns_none():
    assert _clean_date("2099-01-15", month=1) is None


# ---------------------------------------------------------------------------
# Transformer.clean_price
# ---------------------------------------------------------------------------

def _clean_price(raw):
    return Transformer.clean_price(raw, line_number=1)


def test_clean_price_normal():
    assert _clean_price("180.0") == 180.0

def test_clean_price_caps_at_999():
    assert _clean_price("1500.0") == 999.0

def test_clean_price_missing_returns_none():
    assert _clean_price("") is None

def test_clean_price_rounds_to_1dp():
    assert _clean_price("180.15") == 180.2


# ---------------------------------------------------------------------------
# Transformer.clean_postcode
# ---------------------------------------------------------------------------

def _clean_postcode(raw, address="1 Main St 2000"):
    return Transformer.clean_postcode(raw, address, line_number=1)


def test_clean_postcode_normal():
    assert _clean_postcode("2000") == "2000"

def test_clean_postcode_applies_correction():
    assert _clean_postcode("2751") == "2750"

def test_clean_postcode_drops_act():
    assert _clean_postcode("2609") is None
    assert _clean_postcode("2914") is None

def test_clean_postcode_infers_from_address():
    assert _clean_postcode("", address="1 Main St 2777") == "2777"


# ---------------------------------------------------------------------------
# Transformer.clean_brand
# ---------------------------------------------------------------------------

def _clean_brand(raw, name="Shell"):
    return Transformer.clean_brand(raw, name, line_number=1)


def test_clean_brand_passthrough():
    assert _clean_brand("Shell") == "Shell"

def test_clean_brand_infers_from_name():
    assert _clean_brand("", name="BP Springwood") == "BP"
    assert _clean_brand("", name="Caltex Woolworths Penrith") == "Caltex Woolworths"
    assert _clean_brand("", name="7-Eleven Glenbrook") == "7-Eleven"
    assert _clean_brand("", name="Ampol Woolworths Erina") == "Ampol Woolworths"
    assert _clean_brand("", name="Ampol Lake Munmorah") == "Ampol"
    assert _clean_brand("", name="Ampol Foodary Croydon") == "Ampol"
    assert _clean_brand("", name="EG Ampol Berkshire Park") == "Ampol"
    # "ampol woolworths" must not be swallowed by the shorter "ampol" entry
    assert _clean_brand("", name="Ampol Woolworths Toronto") == "Ampol Woolworths"
    assert _clean_brand("", name="United Terrey Hills") == "United"
    assert _clean_brand("", name="Shell South Lismore") == "Shell"
    assert _clean_brand("", name="SHELL CONDELL PARK") == "Shell"
    assert _clean_brand("", name="Mobil Nowra") == "Mobil"
    assert _clean_brand("", name="7- Eleven Towradgi") == "7-Eleven"
    assert _clean_brand("", name="Ultra Manly") == "Ultra"
    assert _clean_brand("", name="Astron Yagoona") == "Astron"

def test_clean_brand_unknown_name_returns_none():
    assert _clean_brand("", name="Unknown Servo XYZ") is None


# ---------------------------------------------------------------------------
# Transformer.clean_suburb
# ---------------------------------------------------------------------------

def test_clean_suburb_title_cases():
    assert Transformer.clean_suburb("SPRINGWOOD", 1) == "Springwood"

def test_clean_suburb_missing_returns_none():
    assert Transformer.clean_suburb("", 1) is None


# ---------------------------------------------------------------------------
# Transformer.clean — integration
# ---------------------------------------------------------------------------

def _run_transformer(rows, tmp_path, name="raw.csv"):
    import csv as _csv
    raw = tmp_path / name
    out = tmp_path / "cleaned.csv"
    with open(raw, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=[
            "ServiceStationName", "Address", "Suburb", "Postcode",
            "Brand", "FuelCode", "PriceUpdatedDate", "Price",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    Transformer(raw, out).clean()
    with open(out, newline="") as f:
        return list(_csv.DictReader(f))


def _run_transformer_with_ckan_shift(normal_rows, shifted_rows, tmp_path):
    """Write a CSV mimicking the CKAN _id column injection bug and run the transformer.

    First len(normal_rows) data rows have the standard 8-column format.
    Then shifted_rows are written with a numeric _id prepended (9 columns after
    an 8-column header), exactly as the CKAN dump endpoint produces.
    """
    import csv as _csv
    raw = tmp_path / "raw_shifted.csv"
    out = tmp_path / "cleaned_shifted.csv"
    _FIELDS = ["ServiceStationName", "Address", "Suburb", "Postcode",
               "Brand", "FuelCode", "PriceUpdatedDate", "Price"]
    with open(raw, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for row in normal_rows:
            writer.writerow(row)
        # Shifted rows: prepend a numeric _id so each row has 9 values against
        # the 8-column header, replicating what the CKAN dump endpoint produces.
        w = _csv.writer(f)
        for i, row in enumerate(shifted_rows, start=len(normal_rows) + 1):
            w.writerow([str(i), row["ServiceStationName"], row["Address"],
                        row["Suburb"], row["Postcode"], row["Brand"],
                        row["FuelCode"], row["PriceUpdatedDate"], row["Price"]])
    Transformer(raw, out).clean()
    with open(out, newline="") as f:
        return list(_csv.DictReader(f))


def test_clean_deduplicates_same_key(tmp_path):
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15", "Price": "180.0"},
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15", "Price": "180.0"},
    ]
    result = _run_transformer(rows, tmp_path)
    assert len(result) == 1


def test_clean_keeps_last_intraday_update(tmp_path):
    # Station resets price high in the morning, drops in the afternoon.
    # Only the afternoon (end-of-day) price should be kept.
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15T09:00:00", "Price": "207.9"},
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15T15:00:00", "Price": "167.9"},
    ]
    result = _run_transformer(rows, tmp_path)
    assert len(result) == 1
    assert float(result[0]["Price"]) == 167.9


def test_clean_keeps_latest_timestamp_regardless_of_csv_order(tmp_path):
    # Even if the CSV rows are out of chronological order, the latest timestamp wins.
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15T15:00:00", "Price": "167.9"},  # later, listed first
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15T09:00:00", "Price": "207.9"},  # earlier, listed second
    ]
    result = _run_transformer(rows, tmp_path)
    assert len(result) == 1
    assert float(result[0]["Price"]) == 167.9


def test_clean_carries_forward_station_for_extra_fuel_code_lines(tmp_path):
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15", "Price": "180.0"},
        # Extra fuel-code line — no station details
        {"ServiceStationName": "", "Address": "", "Suburb": "",
         "Postcode": "", "Brand": "", "FuelCode": "U91",
         "PriceUpdatedDate": "2024-01-15", "Price": "185.0"},
    ]
    result = _run_transformer(rows, tmp_path)
    assert len(result) == 2
    assert all(r["ServiceStationName"] == "Shell" for r in result)


def test_clean_drops_act_postcodes(tmp_path):
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Canberra",
         "Postcode": "2609", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-15", "Price": "180.0"},
    ]
    assert _run_transformer(rows, tmp_path) == []


def test_clean_fixes_date_swap_bug(tmp_path):
    # File is for January. Date "2024-05-01" has month=5 != 1 and day=1 == 1 → swap to Jan 5.
    rows = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-05-01", "Price": "180.0"},
        # Anchor row with day > 12 to establish month = 1
        {"ServiceStationName": "BP", "Address": "2 High St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "BP", "FuelCode": "E10",
         "PriceUpdatedDate": "2024-01-20", "Price": "175.0"},
    ]
    result = _run_transformer(rows, tmp_path)
    dates = {r["PriceUpdatedDate"] for r in result}
    assert any("01-05" in d or "01-20" in d for d in dates)


def test_clean_recovers_ckan_shifted_rows(tmp_path):
    # The CKAN datastore dump injects a numeric _id column at row 32001+
    # without updating the header. Rows before the shift and after must both
    # appear in the cleaned output with correct field values.
    normal = [
        {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
         "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
         "PriceUpdatedDate": "15/06/2025 9:00", "Price": "180.0"},
    ]
    shifted = [
        {"ServiceStationName": "BP", "Address": "2 High St", "Suburb": "Penrith",
         "Postcode": "2750", "Brand": "BP", "FuelCode": "E10",
         "PriceUpdatedDate": "20/06/2025 10:00", "Price": "175.0"},
    ]
    result = _run_transformer_with_ckan_shift(normal, shifted, tmp_path)
    assert len(result) == 2
    stations = {r["ServiceStationName"] for r in result}
    assert stations == {"Shell", "BP"}
    prices = {r["ServiceStationName"]: float(r["Price"]) for r in result}
    assert prices["BP"] == 175.0


def test_clean_all_resources(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cleaned_dir = tmp_path / "cleaned"

    row = {"ServiceStationName": "Shell", "Address": "1 Main St", "Suburb": "Sydney",
           "Postcode": "2000", "Brand": "Shell", "FuelCode": "E10",
           "PriceUpdatedDate": "2024-01-15", "Price": "180.0"}
    _make_raw_csv([row], raw_dir, "file1.csv")
    _make_raw_csv([row], raw_dir, "file2.csv")

    paths = clean_all_resources(raw_dir, cleaned_dir)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
