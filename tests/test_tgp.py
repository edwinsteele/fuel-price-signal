"""Tests for fuel_signal.tgp — AIP Sydney TGP downloader and CSV maintenance."""

import datetime
import pathlib

import pandas as pd
import pytest
import requests
import responses as rsps
from click.testing import CliRunner

from fuel_signal.tgp import (
    LANDING_URL,
    TGP_TABLES_HTTPS_URL,
    TGP_TABLES_URL,
    discover_tgp_url,
    download_tgptables_html,
    download_xlsx,
    extract_tgp_href,
    main,
    merge_tgp_series,
    parse_sydney_series,
    parse_tgptables_series,
    publish_date_from_name,
    write_series_csv,
)

LANDING_HTML = """
<html><body>
  <a href="https://aip.com.au/wp-content/uploads/2026/01/AIP_Annual_TGP_Data.xlsx">Annual</a>
  <a href="https://aip.com.au/wp-content/uploads/2026/06/AIP_TGP_Data_19-Jun-2026.xlsx">Weekly</a>
</body></html>
"""

TGPTABLES_HTML = """
<html><body>
<h3>Petrol (ULP, cents per litre, inclusive of GST)</h3>
<table class="table table-striped">
  <tr>
    <th>City</th>
    <th>Monday<br/>17 August 2026</th>
    <th>Tuesday<br/>18 August 2026</th>
  </tr>
  <tr>
    <td><a href="http://api.aip.com.au/public/sydneyUlp">Sydney</a></td>
    <td>123.4</td>
    <td>123.6</td>
  </tr>
  <tr>
    <td><a href="http://api.aip.com.au/public/melbourneUlp">Melbourne</a></td>
    <td>119.1</td>
    <td>119.0</td>
  </tr>
</table>
<h3>Diesel (cents per litre, inclusive of GST)</h3>
<table class="table table-striped">
  <tr>
    <th>City</th>
    <th>Monday<br/>17 August 2026</th>
    <th>Tuesday<br/>18 August 2026</th>
  </tr>
  <tr>
    <td><a href="http://api.aip.com.au/public/sydneyDiesel">Sydney</a></td>
    <td>999.9</td>
    <td>999.9</td>
  </tr>
</table>
</body></html>
"""

# Same data as TGPTABLES_HTML but with the diesel table listed first, to
# prove table selection goes by the Sydney row's sydneyUlp href, not order.
TGPTABLES_HTML_DIESEL_FIRST = """
<html><body>
<h3>Diesel (cents per litre, inclusive of GST)</h3>
<table class="table table-striped">
  <tr>
    <th>City</th>
    <th>Monday<br/>17 August 2026</th>
  </tr>
  <tr>
    <td><a href="http://api.aip.com.au/public/sydneyDiesel">Sydney</a></td>
    <td>999.9</td>
  </tr>
</table>
<h3>Petrol (ULP, cents per litre, inclusive of GST)</h3>
<table class="table table-striped">
  <tr>
    <th>City</th>
    <th>Monday<br/>17 August 2026</th>
  </tr>
  <tr>
    <td><a href="http://api.aip.com.au/public/sydneyUlp">Sydney</a></td>
    <td>123.4</td>
  </tr>
</table>
</body></html>
"""


def _make_xlsx(path: pathlib.Path, dates: list[str], sydney: list[float]) -> None:
    """Write a minimal AIP-shaped workbook (Petrol TGP sheet, date + city columns)."""
    df = pd.DataFrame(
        {
            "AVERAGE ULP TGPS\n(inclusive of GST)": pd.to_datetime(dates),
            "Sydney": sydney,
            "Melbourne": [v - 1 for v in sydney],
        }
    )
    df.to_excel(path, sheet_name="Petrol TGP", index=False)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_extract_tgp_href_picks_weekly_not_annual():
    href = extract_tgp_href(LANDING_HTML)
    assert href == "https://aip.com.au/wp-content/uploads/2026/06/AIP_TGP_Data_19-Jun-2026.xlsx"


def test_extract_tgp_href_none_when_absent():
    assert extract_tgp_href("<html><body><a href='/x.pdf'>x</a></body></html>") is None


def test_publish_date_from_name():
    assert publish_date_from_name("AIP_TGP_Data_19-Jun-2026.xlsx") == datetime.date(2026, 6, 19)
    assert publish_date_from_name("https://x/2026-06/AIP_TGP_Data_06-Mar-2026.xlsx") == (
        datetime.date(2026, 3, 6)
    )
    assert publish_date_from_name("no-date-here.xlsx") is None


def test_parse_sydney_series_sorts_and_drops_nat(tmp_path):
    xlsx = tmp_path / "src.xlsx"
    _make_xlsx(xlsx, ["2004-01-02", "2004-01-01"], [86.3, 85.0])
    s = parse_sydney_series(xlsx)
    assert list(s.index) == [pd.Timestamp("2004-01-01"), pd.Timestamp("2004-01-02")]
    assert s.loc["2004-01-01"] == 85.0
    assert s.name == "tgp_cents"


def test_parse_sydney_series_accepts_bytes(tmp_path):
    xlsx = tmp_path / "src.xlsx"
    _make_xlsx(xlsx, ["2004-01-01"], [86.3])
    s = parse_sydney_series(xlsx.read_bytes())
    assert s.loc["2004-01-01"] == 86.3


def test_parse_sydney_series_raises_on_layout_change(tmp_path):
    wrong_sheet = tmp_path / "sheet.xlsx"
    pd.DataFrame({"date": ["2004-01-01"], "Sydney": [86.3]}).to_excel(
        wrong_sheet, sheet_name="Wrong", index=False
    )
    with pytest.raises(RuntimeError, match="sheet"):
        parse_sydney_series(wrong_sheet)

    no_city = tmp_path / "city.xlsx"
    pd.DataFrame({"date": ["2004-01-01"], "Melbourne": [85.0]}).to_excel(
        no_city, sheet_name="Petrol TGP", index=False
    )
    with pytest.raises(RuntimeError, match="Sydney"):
        parse_sydney_series(no_city)


# ---------------------------------------------------------------------------
# tgpTables (tail-lag source)
# ---------------------------------------------------------------------------

def test_parse_tgptables_series_picks_petrol_sydney_only():
    s = parse_tgptables_series(TGPTABLES_HTML)
    assert list(s.index) == [pd.Timestamp("2026-08-17"), pd.Timestamp("2026-08-18")]
    assert list(s) == [123.4, 123.6]
    assert s.name == "tgp_cents"


def test_parse_tgptables_series_finds_petrol_table_regardless_of_order():
    # Table selection must go by the Sydney row's sydneyUlp href, not by
    # which table on the page comes first (diesel here, unlike TGPTABLES_HTML).
    s = parse_tgptables_series(TGPTABLES_HTML_DIESEL_FIRST)
    assert list(s) == [123.4]


def test_parse_tgptables_series_raises_when_no_table():
    with pytest.raises(RuntimeError, match="table"):
        parse_tgptables_series("<html><body>no tables here</body></html>")


def test_parse_tgptables_series_raises_when_no_sydney_row():
    html = """
    <h3>Petrol (ULP, cents per litre, inclusive of GST)</h3>
    <table class="table table-striped">
      <tr><th>City</th><th>Monday<br/>17 August 2026</th></tr>
      <tr><td><a href="http://api.aip.com.au/public/melbourneUlp">Melbourne</a></td><td>119.1</td></tr>
    </table>
    """
    with pytest.raises(RuntimeError, match="Sydney"):
        parse_tgptables_series(html)


def test_parse_tgptables_series_raises_on_cell_count_mismatch():
    """A header row with more dated columns than the Sydney row has cells must
    raise, not silently zip-truncate and pair dates with the wrong day's price."""
    html = """
    <h3>Petrol (ULP, cents per litre, inclusive of GST)</h3>
    <table class="table table-striped">
      <tr><th>City</th><th>Monday<br/>17 August 2026</th><th>Tuesday<br/>18 August 2026</th></tr>
      <tr><td><a href="http://api.aip.com.au/public/sydneyUlp">Sydney</a></td><td>123.4</td></tr>
    </table>
    """
    with pytest.raises(RuntimeError, match="cells"):
        parse_tgptables_series(html)


def test_merge_tgp_series_overrides_overlap_and_extends_tail():
    base = pd.Series(
        [86.0, 86.1, 86.2],
        index=pd.to_datetime(["2026-08-14", "2026-08-17", "2026-08-18"]),
        name="tgp_cents",
    )
    tail = pd.Series(
        [123.4, 123.6, 124.0],
        index=pd.to_datetime(["2026-08-17", "2026-08-18", "2026-08-19"]),
        name="tgp_cents",
    )
    merged = merge_tgp_series(base, tail)
    assert list(merged.index) == [
        pd.Timestamp("2026-08-14"),
        pd.Timestamp("2026-08-17"),
        pd.Timestamp("2026-08-18"),
        pd.Timestamp("2026-08-19"),
    ]
    # Pre-tail history is untouched; overlapping and new tail dates come from tail.
    assert list(merged) == [86.0, 123.4, 123.6, 124.0]


@rsps.activate
def test_download_tgptables_html_prefers_https():
    rsps.add(rsps.GET, TGP_TABLES_HTTPS_URL, body=TGPTABLES_HTML, status=200)
    # No HTTP mock registered — an HTTP fallback attempt would raise ConnectionError.
    assert download_tgptables_html() == TGPTABLES_HTML


@rsps.activate
def test_download_tgptables_html_falls_back_to_http_on_https_failure():
    rsps.add(rsps.GET, TGP_TABLES_HTTPS_URL, body=requests.exceptions.ConnectionError("no TLS"))
    rsps.add(rsps.GET, TGP_TABLES_URL, body=TGPTABLES_HTML, status=200)
    assert download_tgptables_html() == TGPTABLES_HTML


@rsps.activate
def test_download_tgptables_html_raises_when_both_fail():
    rsps.add(rsps.GET, TGP_TABLES_HTTPS_URL, status=503)
    rsps.add(rsps.GET, TGP_TABLES_URL, status=503)
    with pytest.raises(requests.exceptions.HTTPError):
        download_tgptables_html()


# ---------------------------------------------------------------------------
# CSV maintenance
# ---------------------------------------------------------------------------

def test_write_series_csv_counts_new_and_is_idempotent(tmp_path):
    csv_path = tmp_path / "tgp" / "tgp_sydney.csv"
    s = pd.Series([86.3, 86.4], index=pd.to_datetime(["2004-01-01", "2004-01-02"]))

    assert write_series_csv(s, csv_path) == 2
    assert csv_path.exists()
    # Re-writing the identical series adds no new dates.
    assert write_series_csv(s, csv_path) == 0

    # Appending a fresh date counts exactly one new row, full series persisted.
    s2 = pd.concat([s, pd.Series([86.5], index=pd.to_datetime(["2004-01-05"]))])
    assert write_series_csv(s2, csv_path) == 1
    out = pd.read_csv(csv_path)
    assert list(out["date"]) == ["2004-01-01", "2004-01-02", "2004-01-05"]


# ---------------------------------------------------------------------------
# Network (mocked)
# ---------------------------------------------------------------------------

@rsps.activate
def test_discover_tgp_url_resolves_absolute():
    rsps.add(rsps.GET, LANDING_URL, body=LANDING_HTML, status=200)
    url = discover_tgp_url()
    assert url == "https://aip.com.au/wp-content/uploads/2026/06/AIP_TGP_Data_19-Jun-2026.xlsx"


@rsps.activate
def test_download_xlsx_returns_bytes():
    url = "https://aip.com.au/x/AIP_TGP_Data_19-Jun-2026.xlsx"
    rsps.add(rsps.GET, url, body=b"\x50\x4b\x03\x04payload", status=200)
    assert download_xlsx(url) == b"\x50\x4b\x03\x04payload"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_from_xlsx_writes_csv(tmp_path):
    xlsx = tmp_path / "AIP_TGP_Data_19-Jun-2026.xlsx"
    _make_xlsx(xlsx, ["2004-01-01", "2004-01-02"], [86.3, 86.4])
    csv_path = tmp_path / "tgp" / "tgp_sydney.csv"

    result = CliRunner().invoke(main, ["--from-xlsx", str(xlsx), "--csv-path", str(csv_path)])
    assert result.exit_code == 0, result.output
    out = pd.read_csv(csv_path)
    assert list(out["date"]) == ["2004-01-01", "2004-01-02"]
    assert list(out["tgp_cents"]) == [86.3, 86.4]


@rsps.activate
def test_main_live_merges_tgptables_tail(tmp_path):
    xlsx_url = "https://aip.com.au/wp-content/uploads/2026/08/AIP_TGP_Data_14-Aug-2026.xlsx"
    xlsx_path = tmp_path / "src.xlsx"
    _make_xlsx(xlsx_path, ["2026-08-13", "2026-08-14"], [86.0, 86.1])

    rsps.add(
        rsps.GET,
        LANDING_URL,
        body=f'<a href="{xlsx_url}">Weekly</a>',
        status=200,
    )
    rsps.add(rsps.GET, xlsx_url, body=xlsx_path.read_bytes(), status=200)
    rsps.add(rsps.GET, TGP_TABLES_URL, body=TGPTABLES_HTML, status=200)

    csv_path = tmp_path / "tgp" / "tgp_sydney.csv"
    result = CliRunner().invoke(main, ["--csv-path", str(csv_path)])
    assert result.exit_code == 0, result.output
    out = pd.read_csv(csv_path)
    # xlsx history is kept; the tgpTables tail (17/18 Aug) extends past it.
    assert list(out["date"]) == ["2026-08-13", "2026-08-14", "2026-08-17", "2026-08-18"]
    assert list(out["tgp_cents"]) == [86.0, 86.1, 123.4, 123.6]


@rsps.activate
def test_main_live_falls_back_to_xlsx_when_tgptables_fails(tmp_path):
    xlsx_url = "https://aip.com.au/wp-content/uploads/2026/08/AIP_TGP_Data_14-Aug-2026.xlsx"
    xlsx_path = tmp_path / "src.xlsx"
    _make_xlsx(xlsx_path, ["2026-08-13", "2026-08-14"], [86.0, 86.1])

    rsps.add(
        rsps.GET,
        LANDING_URL,
        body=f'<a href="{xlsx_url}">Weekly</a>',
        status=200,
    )
    rsps.add(rsps.GET, xlsx_url, body=xlsx_path.read_bytes(), status=200)
    rsps.add(rsps.GET, TGP_TABLES_URL, status=503)

    csv_path = tmp_path / "tgp" / "tgp_sydney.csv"
    result = CliRunner().invoke(main, ["--csv-path", str(csv_path)])
    assert result.exit_code == 0, result.output
    out = pd.read_csv(csv_path)
    assert list(out["date"]) == ["2026-08-13", "2026-08-14"]
    assert list(out["tgp_cents"]) == [86.0, 86.1]


def test_main_skip_tgptables_flag_avoids_fetch(tmp_path):
    xlsx = tmp_path / "AIP_TGP_Data_19-Jun-2026.xlsx"
    _make_xlsx(xlsx, ["2004-01-01"], [86.3])
    csv_path = tmp_path / "tgp" / "tgp_sydney.csv"

    # No responses registered at all — a tgpTables call would raise ConnectionError.
    result = CliRunner().invoke(
        main, ["--from-xlsx", str(xlsx), "--csv-path", str(csv_path), "--skip-tgptables"]
    )
    assert result.exit_code == 0, result.output
