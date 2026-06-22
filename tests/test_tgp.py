"""Tests for fuel_signal.tgp — AIP Sydney TGP downloader and CSV maintenance."""

import datetime
import pathlib

import pandas as pd
import responses as rsps
from click.testing import CliRunner

from fuel_signal.tgp import (
    LANDING_URL,
    discover_tgp_url,
    download_xlsx,
    extract_tgp_href,
    main,
    parse_sydney_series,
    publish_date_from_name,
    write_series_csv,
)

LANDING_HTML = """
<html><body>
  <a href="/sites/default/files/download-files/2026-01/AIP_Annual_TGP_Data.xlsx">Annual</a>
  <a href="/sites/default/files/download-files/2026-06/AIP_TGP_Data_19-Jun-2026.xlsx">Weekly</a>
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
    assert href == "/sites/default/files/download-files/2026-06/AIP_TGP_Data_19-Jun-2026.xlsx"


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
    assert url == "https://www.aip.com.au/sites/default/files/download-files/2026-06/AIP_TGP_Data_19-Jun-2026.xlsx"


@rsps.activate
def test_download_xlsx_returns_bytes():
    url = "https://www.aip.com.au/x/AIP_TGP_Data_19-Jun-2026.xlsx"
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
