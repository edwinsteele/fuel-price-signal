"""Fetch and maintain the AIP Sydney ULP Terminal Gate Price (TGP) series.

Source: aip.com.au "Historical ULP and Diesel TGP Data" — a single xlsx holding
the full daily-weekday history (2004→present, c/L GST-inclusive, same units as
our pump data). The download filename is date-stamped
(``.../wp-content/uploads/2026-08/AIP_TGP_Data_14-Aug-2026.xlsx``), so we scrape
the landing page for the current ``AIP_TGP_Data_*.xlsx`` href, download it, and
maintain a canonical single-column CSV of the Sydney series.

The site moved from ``www.aip.com.au/historical-ulp-and-diesel-tgp-data`` to
``aip.com.au/resources/historical-ulp-and-diesel-tgp-data/`` around 2026-08
(fps-8qc); the landing page still links the same ``AIP_TGP_Data_*.xlsx``
filename convention with an unchanged xlsx layout, just now with absolute
hrefs on a new domain/path rather than relative ones.

The xlsx itself is only refreshed **weekly**, not daily: four consecutive
fetches confirmed the publish date jumping every Friday (24/31 Jul, 7/14 Aug
2026), each adding exactly 5 new weekday rows. So despite the daily fetch
cron, the latest available row lags "today" by 0-6 days depending on where in
the week the job runs — see the cron comment in
``.github/workflows/tgp-fetch.yml``.

To reduce that tail lag, ``api.aip.com.au/public/tgpTables`` — a
server-rendered HTML page AIP also serves, refreshed more often (~1-2 day
lag) but showing only a rolling 5-weekday window per city, with no deep
history — is fetched alongside the xlsx. Its Sydney row overrides/extends the
xlsx-derived series for the tail dates where the two overlap or it has data
the xlsx doesn't yet (see ``merge_tgp_series``); the full-history xlsx stays
the sole source of truth for anything older than that window (#271 still
holds). A tgpTables fetch/parse failure degrades to xlsx-only rather than
failing the job — it's a lag-reduction extra, not a dependency.

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

LANDING_URL = "https://aip.com.au/resources/historical-ulp-and-diesel-tgp-data/"
# The only endpoint AIP has been confirmed to serve this page on is plain HTTP
# (no TLS, confirmed live 2026-08-20). HTTPS is tried first regardless, since a
# network intermediary could otherwise tamper with the page in transit and
# have this write bogus Sydney TGP values into the canonical CSV; falling
# back to the documented HTTP URL only if the HTTPS attempt fails outright.
TGP_TABLES_HTTPS_URL = "https://api.aip.com.au/public/tgpTables"
TGP_TABLES_URL = "http://api.aip.com.au/public/tgpTables"

# The weekly history file. Matches ``AIP_TGP_Data_19-Jun-2026.xlsx`` but NOT the
# separate ``AIP_Annual_TGP_Data.xlsx`` summary file on the same page.
_TGP_FILE_RE = re.compile(r"AIP_TGP_Data_[^/\"']*\.xlsx", re.IGNORECASE)
# Publish date embedded in the filename, e.g. ``19-Jun-2026``.
_PUBLISH_DATE_RE = re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4})")
# tgpTables column header, e.g. ``Wednesday 12 August 2026`` (weekday name is
# discarded — only the trailing "12 August 2026" is parsed).
_TGPTABLES_DATE_RE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})"
)

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


def download_tgptables_html() -> str:
    """Download the rolling 5-weekday-window tgpTables HTML page.

    Tries the HTTPS URL first, falling back to the documented plain-HTTP one
    only if the HTTPS attempt fails outright (connection refused, TLS error,
    HTTP error status, etc.) — see the ``TGP_TABLES_URL`` comment.
    """
    last_exc: requests.exceptions.RequestException | None = None
    for url in (TGP_TABLES_HTTPS_URL, TGP_TABLES_URL):
        try:
            logger.info("Downloading TGP tables page %s", url)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.text
        except requests.exceptions.RequestException as exc:
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def parse_tgptables_series(html: str) -> pd.Series:
    """Parse the Sydney ULP row from the tgpTables rolling-window HTML page.

    tgpTables renders one ``table-striped`` table per fuel type (petrol,
    diesel, ...); the petrol/ULP table is identified by its Sydney row's
    ``sydneyUlp`` href, not by table order, since AIP's own ordering of the
    per-fuel tables isn't guaranteed.

    Returns a date-indexed, sorted ``pd.Series`` named ``tgp_cents`` covering
    whichever of the ~5 weekday columns parsed. Raises ``RuntimeError`` if no
    Sydney ULP table/row or dated column can be found — the page layout may
    have changed.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = None
    sydney_row = None
    for candidate in soup.find_all("table", class_="table-striped"):
        for tr in candidate.find_all("tr"):
            anchor = tr.find("a", href=True)
            if anchor and "sydneyulp" in anchor["href"].lower():
                table, sydney_row = candidate, tr
                break
        if table is not None:
            break
    if table is None or sydney_row is None:
        raise RuntimeError("No Sydney ULP tgpTables table found — layout may have changed")

    header_row = table.find("tr")
    if header_row is None:
        raise RuntimeError("tgpTables table has no header row")
    dates: list[datetime.date | None] = []
    for cell in header_row.find_all(["th", "td"])[1:]:
        m = _TGPTABLES_DATE_RE.search(cell.get_text(" ", strip=True))
        dates.append(datetime.datetime.strptime(m.group(1), "%d %B %Y").date() if m else None)

    cells = sydney_row.find_all("td")[1:]
    if len(cells) != len(dates):
        raise RuntimeError(
            f"tgpTables Sydney row has {len(cells)} cells but header has "
            f"{len(dates)} dated columns — layout may have changed"
        )

    data: dict[datetime.date, float] = {}
    for date, cell in zip(dates, cells):
        if date is None:
            continue
        try:
            data[date] = float(cell.get_text(strip=True))
        except ValueError:
            continue
    if not data:
        raise RuntimeError("No parseable Sydney TGP values found in tgpTables")

    series = pd.Series(data, name="tgp_cents").sort_index()
    series.index = pd.to_datetime(series.index)
    series.index.name = "date"
    return series


def merge_tgp_series(base: pd.Series, tail: pd.Series) -> pd.Series:
    """Merge tgpTables' fresher tail into the xlsx-derived ``base`` series.

    ``tail`` values win wherever present — both correcting dates the two
    sources overlap on and extending with newer dates the xlsx doesn't have
    yet. Every date only ``base`` covers (i.e. all pre-tail history) is left
    untouched.
    """
    return tail.combine_first(base).sort_index()


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
    try:
        raw = pd.read_excel(source, sheet_name=SHEET)
    except ValueError as exc:  # pandas raises ValueError when the sheet is absent
        raise RuntimeError(
            f"AIP xlsx has no {SHEET!r} sheet — layout may have changed"
        ) from exc
    if CITY not in raw.columns:
        raise RuntimeError(
            f"AIP {SHEET!r} sheet missing {CITY!r} column — layout may have changed"
        )
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
@click.option(
    "--skip-tgptables",
    is_flag=True,
    default=False,
    help="Skip the tgpTables tail-lag fetch and use the weekly xlsx alone.",
)
def main(from_xlsx: pathlib.Path | None, csv_path: pathlib.Path, skip_tgptables: bool) -> None:
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

    # --from-xlsx is for offline/backfill use, which stays network-free even
    # without --skip-tgptables; a fetch/parse failure degrades to xlsx-only.
    if from_xlsx is None and not skip_tgptables:
        try:
            tail = parse_tgptables_series(download_tgptables_html())
            series = merge_tgp_series(series, tail)
        except Exception:
            logger.warning(
                "tgpTables fetch/parse failed; using xlsx-only series", exc_info=True
            )

    _log_lag(publish_date, series)
    n_new = write_series_csv(series, csv_path)
    logger.info("Wrote %s (%d rows, %d new)", csv_path, len(series), n_new)


if __name__ == "__main__":
    main()
