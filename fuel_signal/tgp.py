"""Fetch and maintain the AIP Sydney ULP Terminal Gate Price (TGP) series.

Source: aip.com.au "Historical ULP and Diesel TGP Data" — a single xlsx holding
the full daily-weekday history (2004→present, c/L GST-inclusive, same units as
our pump data). The download URL is date-stamped in both path and filename
(``.../download-files/2026-06/AIP_TGP_Data_19-Jun-2026.xlsx``), so we scrape the
landing page for the current ``AIP_TGP_Data_*.xlsx`` href, download it, and
maintain a canonical single-column CSV of the Sydney series.

Storage rationale (#271): the source always serves the *full* history, so the
daily action downloads it, parses the Sydney column, and **overwrites**
``data/tgp/tgp_sydney.csv``. In the steady state git sees a one-line append (or
no change at all — the snapshot workflow's ``git diff --cached --quiet`` guard
suppresses no-op commits); a rare historical revision (posted gate prices are
final, so these are essentially data-entry fixes) shows up as a changed line and
is corrected automatically. The full rewrite *is* the reconcile — there is no
separate provisional/delta layer. The committed provenance xlsx
(``data/tgp/AIP_TGP.xlsx``) is refreshed separately by the #4 reconcile task.

The downloader feeds the ``tgp_delta_7d`` feature graduated by experiment
``2026-06-20_leading_indicators`` (see #271).
"""

from __future__ import annotations

import datetime
import io
import logging
import pathlib
import re
import urllib.parse

import click
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

LANDING_URL = "https://www.aip.com.au/historical-ulp-and-diesel-tgp-data"

# The weekly history file. Matches ``AIP_TGP_Data_19-Jun-2026.xlsx`` but NOT the
# separate ``AIP_Annual_TGP_Data.xlsx`` summary file on the same page.
_TGP_FILE_RE = re.compile(r"AIP_TGP_Data_[^/\"']*\.xlsx", re.IGNORECASE)
# Publish date embedded in the filename, e.g. ``19-Jun-2026``.
_PUBLISH_DATE_RE = re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4})")

SHEET = "Petrol TGP"
CITY = "Sydney"

DEFAULT_CSV_PATH = pathlib.Path("data/tgp/tgp_sydney.csv")


def extract_tgp_href(html: str) -> str | None:
    """Return the first ``AIP_TGP_Data_*.xlsx`` href on the landing page, if any."""
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        if _TGP_FILE_RE.search(anchor["href"]):
            return anchor["href"]
    return None


def discover_tgp_url() -> str:
    """Scrape the AIP landing page for the current weekly TGP xlsx download URL."""
    r = requests.get(LANDING_URL, timeout=30)
    r.raise_for_status()
    href = extract_tgp_href(r.text)
    if href is None:
        raise RuntimeError(f"No AIP_TGP_Data_*.xlsx link found at {LANDING_URL}")
    return urllib.parse.urljoin(LANDING_URL, href)


def download_xlsx(url: str) -> bytes:
    """Download the TGP xlsx and return its raw bytes."""
    logger.info("Downloading TGP xlsx %s", url)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def publish_date_from_name(name: str) -> datetime.date | None:
    """Parse the publish date from a TGP filename or URL (``...19-Jun-2026.xlsx``)."""
    m = _PUBLISH_DATE_RE.search(name)
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1), "%d-%b-%Y").date()
    except ValueError:
        return None


def parse_sydney_series(source: str | pathlib.Path | bytes) -> pd.Series:
    """Parse the daily Sydney ULP TGP series (c/L) from the AIP xlsx.

    ``source`` may be a path or the raw xlsx bytes. Returns a date-indexed,
    sorted ``pd.Series`` named ``tgp_cents`` with no NaT dates.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    raw = pd.read_excel(source, sheet_name=SHEET)
    date_col = raw.columns[0]
    df = raw[[date_col, CITY]].rename(columns={date_col: "date", CITY: "tgp_cents"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")["tgp_cents"]


def write_series_csv(series: pd.Series, path: pathlib.Path) -> int:
    """Overwrite ``path`` with the full series; return the count of new dates added."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.exists():
        existing = set(pd.read_csv(path)["date"].astype(str))
    out = series.rename("tgp_cents").to_frame()
    out.index = series.index.strftime("%Y-%m-%d")
    out.index.name = "date"
    out.to_csv(path)
    return sum(1 for d in out.index if d not in existing)


def _log_lag(publish_date: datetime.date | None, series: pd.Series) -> None:
    """Log publish-date vs latest-data-date vs today, to characterise availability lag."""
    data_max = series.index.max().date()
    today = datetime.date.today()
    logger.info(
        "TGP publish=%s data_max=%s today=%s data_lag_days=%d rows=%d",
        publish_date, data_max, today, (today - data_max).days, len(series),
    )


@click.command()
@click.option(
    "--from-xlsx",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Parse this local xlsx instead of downloading (backfill / offline use).",
)
@click.option(
    "--csv-path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=DEFAULT_CSV_PATH,
    show_default=True,
    help="Canonical Sydney TGP CSV to (over)write.",
)
def main(from_xlsx: pathlib.Path | None, csv_path: pathlib.Path) -> None:
    """Fetch the AIP Sydney TGP series and refresh the canonical CSV."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if from_xlsx is not None:
        source: str | pathlib.Path | bytes = from_xlsx
        publish_date = publish_date_from_name(from_xlsx.name)
    else:
        url = discover_tgp_url()
        source = download_xlsx(url)
        publish_date = publish_date_from_name(url)

    series = parse_sydney_series(source)
    _log_lag(publish_date, series)
    n_new = write_series_csv(series, csv_path)
    logger.info("Wrote %s (%d rows, %d new)", csv_path, len(series), n_new)


if __name__ == "__main__":
    main()
