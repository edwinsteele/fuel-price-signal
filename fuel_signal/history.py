"""Download and clean bulk historical price CSVs from data.nsw.gov.au."""

import csv
import datetime
import io
import logging
import pathlib
import urllib.parse
import zipfile

import click
import openpyxl
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DATASET_URL = "https://data.nsw.gov.au/data/dataset/fuel-check"
DATASTORE_DUMP_URL = "https://data.nsw.gov.au/data/datastore/dump/"
RESOURCE_PAGE_URL = "https://data.nsw.gov.au/data/dataset/fuel-check/resource/"

XLSX_COLUMNS = (
    "ServiceStationName",
    "Address",
    "Suburb",
    "Postcode",
    "Brand",
    "FuelCode",
    "PriceUpdatedDate",
    "Price",
)

# Fields that identify a unique service station (used to carry forward
# across extra fuel-code lines where station details are blank).
_SERVO_ID_FIELDS = ("ServiceStationName", "Address", "Suburb", "Postcode", "Brand")

_MAX_PRICE_CENTS = 999.0

_NAME_TO_BRAND = [
    # Longer/more-specific prefixes must come before shorter ones
    ("ampol woolworths", "Ampol Woolworths"),
    ("ampol foodary", "Ampol"),
    ("eg ampol", "Ampol"),
    ("coles express", "Coles Express"),
    ("caltex woolworths", "Caltex Woolworths"),
    ("7-eleven", "7-Eleven"),
    ("7- eleven", "7-Eleven"),  # source data typo: space after hyphen
    ("bp", "BP"),
    ("ampol", "Ampol"),
    ("caltex", "Caltex"),
    ("shell", "Shell"),
    ("mobil", "Mobil"),
    ("ultra", "Ultra"),
    ("astron", "Astron"),
    ("metro", "Metro Fuel"),
    ("speedway", "Speedway"),
    ("united petroleum", "United"),
    ("united", "United"),
    ("werrington south", "7-Eleven"),
    ("the foodary", "Caltex"),
    ("budget", "Budget"),
    ("long jetty", "7-Eleven"),
]

_POSTCODE_CORRECTIONS = {
    "1579": "2019",  # Botany — incorrect, no clear reason why
    "2056": "2506",  # Berkeley — typo
    "2461": "2460",  # South Grafton — typo
    "2751": "2750",  # Penrith PO Box → Penrith
    "2860": "2680",  # Griffith — typo
}

# ACT postcodes that slipped into NSW data
_ACT_POSTCODES = {"2609", "2914"}

# Date formats seen across the dataset. D/MM/YYYY variants appear in direct
# downloads; ISO/space-separated variants appear in CKAN datastore dumps.
_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",    # CKAN dump with time (T separator)
    "%Y-%m-%d %H:%M:%S",    # CKAN dump with time (space separator)
    "%Y-%m-%d",              # CKAN dump date only
    "%d/%m/%Y %I:%M:%S %p", # direct download 12-hour AM/PM with seconds
    "%d/%m/%Y %H:%M",        # direct download 24-hour without seconds
    "%d/%m/%Y",              # direct download date only
]


def _parse_date(raw: str) -> datetime.datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def discover_price_resources() -> list[dict[str, str]]:
    """Scrape dataset page; return [{id, download_url}] for price-history resources."""
    r = requests.get(DATASET_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    resources = []
    for div in soup.select("#dataset-resources div.resource-item[data-id]"):
        link = div.find("a", class_="resource-url-analytics")
        if link and "price" in link["href"].lower():
            resources.append({"id": div["data-id"], "download_url": link["href"]})
    return resources


def _format_cell(value: object) -> str:
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    return "" if value is None else str(value)


class ResourceFetcher:
    def __init__(self, resource_id: str, download_url: str, raw_dir: pathlib.Path):
        self.resource_id = resource_id
        self.download_url = download_url
        self.raw_dir = raw_dir

    @property
    def _dest(self) -> pathlib.Path:
        return self.raw_dir / f"{self.resource_id}.csv"

    def _fetch_csv(self) -> bool:
        dest = self._dest
        if dest.exists():
            logger.info("%s already cached, skipping", dest.name)
            return True

        url = urllib.parse.urljoin(DATASTORE_DUMP_URL, self.resource_id)
        logger.info("Downloading CSV %s", self.resource_id)
        r = requests.get(url, stream=True, timeout=120)
        if not r.ok:
            logger.warning("CSV unavailable (%s): %s", r.status_code, self.resource_id)
            return False

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

        if dest.stat().st_size == 0:
            dest.unlink()
            logger.warning("Empty response from CKAN dump for %s, will try XLSX", self.resource_id)
            return False

        return True

    def _xlsx_url_from_resource_page(self) -> str | None:
        """Scrape the resource page for the actual file download URL."""
        r = requests.get(RESOURCE_PAGE_URL + self.resource_id, timeout=30)
        if not r.ok:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        meta = soup.head.find("meta", attrs={"name": "DCTERMS.Identifier"})
        return meta["content"] if meta else None

    def _try_xlsx_url(self, url: str) -> bool:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        try:
            wb = openpyxl.load_workbook(
                filename=io.BytesIO(r.content), read_only=True, data_only=True
            )
        except zipfile.BadZipFile:
            logger.warning("Not a valid XLSX at %s", url)
            return False

        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()

        # Skip 1–3 header rows: stop as soon as a non-header row is found.
        # Some older files have duplicate or blank header rows above the data.
        skip = 1
        for i in range(1, min(3, len(rows))):
            first = rows[i][0]
            if first is None or first == "ServiceStationName":
                skip = i + 1
            else:
                break

        data_rows = [row for row in rows[skip:] if any(c is not None for c in row)]

        with open(self._dest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(XLSX_COLUMNS)
            for row in data_rows:
                writer.writerow([_format_cell(c) for c in row])
        return True

    def _try_direct_csv_url(self, url: str) -> bool:
        """Download a direct CSV URL that isn't in the CKAN datastore."""
        r = requests.get(url, stream=True, timeout=120)
        if not r.ok:
            logger.warning("Direct CSV unavailable (%s): %s", r.status_code, url)
            return False
        with open(self._dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return True

    def _fetch_direct_url(self) -> bool:
        """Handle resources not in the CKAN datastore via their direct download URL."""
        logger.info("Trying direct URL for %s", self.resource_id)
        url = self.download_url
        if urllib.parse.urlparse(url).path.lower().endswith(".csv"):
            return self._try_direct_csv_url(url)

        if self._try_xlsx_url(url):
            return True

        # The URL scraped from the dataset page can be stale. Fall back to
        # the resource page which has the canonical download URL.
        logger.warning("Bad content from direct URL, trying resource page for %s", self.resource_id)
        page_url = self._xlsx_url_from_resource_page()
        if page_url and page_url != url:
            if urllib.parse.urlparse(page_url).path.lower().endswith(".csv"):
                return self._try_direct_csv_url(page_url)
            return self._try_xlsx_url(page_url)

        return False

    def fetch(self) -> pathlib.Path:
        if self._fetch_csv() or self._fetch_direct_url():
            return self._dest
        raise RuntimeError(f"Unable to download resource {self.resource_id}")


def download_all(raw_dir: pathlib.Path) -> list[pathlib.Path]:
    """Discover all price-history resources and download them into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    resources = discover_price_resources()
    logger.info("Found %d price history resources", len(resources))
    paths = []
    for res in resources:
        fetcher = ResourceFetcher(res["id"], res["download_url"], raw_dir)
        paths.append(fetcher.fetch())
    return paths


# ---------------------------------------------------------------------------
# Clean / transform
# ---------------------------------------------------------------------------

class Transformer:
    """Clean a single raw price-history CSV into a normalised form."""

    def __init__(self, raw_path: pathlib.Path, cleaned_path: pathlib.Path):
        self.raw_path = raw_path
        self.cleaned_path = cleaned_path

    def get_month_for_file(self) -> int | None:
        """Return the calendar month the file covers.

        Many files use YYYY-DD-MM instead of YYYY-MM-DD until the day
        exceeds 12, at which point the format switches back. We find the
        first row whose day > 12 to read the true month unambiguously.

        For files where every date has day <= 12 (e.g. Oct 1-9 only), a
        constant day value across varying months is the YYYY-DD-MM fingerprint:
        the constant is the true month.
        """
        last_date = None
        days_seen: set[int] = set()
        months_seen: set[int] = set()
        with open(self.raw_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw = row.get("PriceUpdatedDate", "")
                if not raw:
                    continue
                d = _parse_date(raw)
                if d is None:
                    continue
                last_date = d
                if d.day > 12:
                    return d.month
                days_seen.add(d.day)
                months_seen.add(d.month)

        if not last_date:
            logger.warning("No parseable dates in %s — skipping", self.raw_path.name)
            return None

        # Constant day + varying months = YYYY-DD-MM where the day field is
        # the true month (e.g. all 2019-XX-10 → October).
        if len(days_seen) == 1 and len(months_seen) > 1:
            true_month = days_seen.pop()
            logger.info(
                "Constant day=%s across %d months in %s; inferring YYYY-DD-MM, month=%s",
                true_month, len(months_seen), self.raw_path.name, true_month,
            )
            return true_month

        logger.info("Day > 12 not found in %s; assuming month %s", self.raw_path.name, last_date.month)
        return last_date.month

    @staticmethod
    def clean_date(raw: str, line_number: int, prev_date: datetime.datetime | None,
                   month_in_file: int) -> datetime.datetime | None:
        if not raw:
            if prev_date:
                return prev_date
            logger.debug("Line %s: missing PriceUpdatedDate and no previous row", line_number)
            return None

        d = _parse_date(raw)
        if d is None:
            logger.debug("Line %s: unparseable date %r", line_number, raw)
            return None

        # Fix YYYY-DD-MM bug: Python parses YYYY-DD-MM as month=day, day=month.
        # When d.month == month_in_file, the raw day happens to equal the true
        # month — swapping would produce the same date, so skipping is correct.
        if d.month != month_in_file:
            if d.day != month_in_file:
                logger.debug("Line %s: date %s inconsistent with file month %s", line_number, d, month_in_file)
                return None
            d = d.replace(month=d.day, day=d.month)

        if d > datetime.datetime.now():
            logger.debug("Line %s: future date %s", line_number, d)
            return None

        return d

    @staticmethod
    def clean_price(raw: str, line_number: int) -> float | None:
        if not raw:
            logger.debug("Line %s: missing Price", line_number)
            return None
        try:
            return round(min(_MAX_PRICE_CENTS, float(raw)), 1)
        except ValueError:
            logger.debug("Line %s: unparseable price %r", line_number, raw)
            return None

    @staticmethod
    def clean_postcode(raw: str, address: str, line_number: int) -> str | None:
        if not raw:
            parts = address.split()
            if not parts:
                return None
            raw = parts[-1]
            logger.debug("Line %s: postcode missing, inferred %s from address", line_number, raw)

        if raw in _ACT_POSTCODES:
            return None

        return _POSTCODE_CORRECTIONS.get(raw, raw)

    @staticmethod
    def clean_fuelcode(raw: str, line_number: int) -> str | None:
        if not raw:
            logger.debug("Line %s: missing FuelCode", line_number)
            return None
        return raw

    @staticmethod
    def clean_brand(raw: str, name: str, line_number: int) -> str | None:
        if not raw:
            for prefix, brand in _NAME_TO_BRAND:
                if name.lower().startswith(prefix):
                    logger.debug("Line %s: inferred brand %s from name %r", line_number, brand, name)
                    return brand
            logger.warning("Line %s: missing Brand, no inference for name %r", line_number, name)
            return None
        return raw

    @staticmethod
    def clean_suburb(raw: str, line_number: int) -> str | None:
        if not raw:
            logger.info("Line %s: missing Suburb", line_number)
            return None
        return raw.title()

    def clean(self) -> int:
        """Clean raw_path → cleaned_path. Returns number of rows written."""
        month_in_file = self.get_month_for_file()
        if month_in_file is None:
            return 0

        current_servo: dict[str, str] = {f: "" for f in _SERVO_ID_FIELDS}
        prev_date: datetime.datetime | None = None
        # key → (latest_datetime, row); keeps the last intraday update per station/fuel/day
        seen: dict[str, tuple[datetime.datetime, dict]] = {}
        line_number = 0

        with open(self.raw_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                line_number += 1

                # CKAN datastore dump injects a numeric _id column at row 32001+
                # without updating the header, shifting all subsequent fields right.
                # ServiceStationName is never a plain integer in real data.
                if row.get("ServiceStationName", "").isdigit():
                    extras = row.get(None) or []
                    row = {
                        "ServiceStationName": row["Address"],
                        "Address": row["Suburb"],
                        "Suburb": row["Postcode"],
                        "Postcode": row["Brand"],
                        "Brand": row["FuelCode"],
                        "FuelCode": row["PriceUpdatedDate"],
                        "PriceUpdatedDate": row["Price"],
                        "Price": extras[0] if extras else "",
                    }

                # Skip trailing blank/separator lines
                if not row.get("FuelCode") and not row.get("Price") and not row.get("PriceUpdatedDate"):
                    continue

                # Extra fuel-code lines omit station details — carry forward
                if not row.get("ServiceStationName"):
                    row.update(current_servo)

                date = self.clean_date(row.get("PriceUpdatedDate", ""), line_number, prev_date, month_in_file)
                if date is None:
                    continue

                price = self.clean_price(row.get("Price", ""), line_number)
                if price is None:
                    continue

                postcode = self.clean_postcode(row.get("Postcode", ""), row.get("Address", ""), line_number)
                if postcode is None:
                    continue

                fuelcode = self.clean_fuelcode(row.get("FuelCode", ""), line_number)
                if fuelcode is None:
                    continue

                brand = self.clean_brand(row.get("Brand", ""), row.get("ServiceStationName", ""), line_number)
                if brand is None:
                    continue

                suburb = self.clean_suburb(row.get("Suburb", ""), line_number)
                if suburb is None:
                    continue

                row["PriceUpdatedDate"] = date.strftime("%Y-%m-%d %H:%M:%S")
                row["Price"] = price
                row["Postcode"] = postcode
                row["FuelCode"] = fuelcode
                row["Brand"] = brand
                row["Suburb"] = suburb

                for field in _SERVO_ID_FIELDS:
                    current_servo[field] = row[field]
                prev_date = date

                # Dedup by station+fuel+date (not full timestamp); keep the latest
                # intraday update so we capture end-of-day price rather than morning reset.
                key = f"{fuelcode}{row['ServiceStationName']}{row['Address']}{brand}{date.date()}"
                if key not in seen or date > seen[key][0]:
                    seen[key] = (date, row)

        with open(self.cleaned_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=XLSX_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for _dt, row in seen.values():
                writer.writerow(row)

        return len(seen)


def clean_resource(raw_path: pathlib.Path, cleaned_dir: pathlib.Path) -> pathlib.Path:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dir / raw_path.name
    if cleaned_path.exists():
        logger.info("%s already cleaned, skipping", raw_path.name)
        return cleaned_path
    rows = Transformer(raw_path, cleaned_path).clean()
    logger.info("Cleaned %s → %d rows", raw_path.name, rows)
    return cleaned_path


def clean_all_resources(raw_dir: pathlib.Path, cleaned_dir: pathlib.Path) -> list[pathlib.Path]:
    """Clean all raw CSVs in raw_dir into cleaned_dir."""
    paths = []
    for raw_path in sorted(raw_dir.glob("*.csv")):
        paths.append(clean_resource(raw_path, cleaned_dir))
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@click.command("history")
@click.option("--raw-dir", default="data/raw", show_default=True, help="Directory for downloaded raw CSVs.")
@click.option("--cleaned-dir", default="data/cleaned", show_default=True, help="Directory for cleaned output CSVs.")
def main(raw_dir: str, cleaned_dir: str) -> None:
    """Download and clean bulk historical price CSVs from data.nsw.gov.au."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = pathlib.Path(raw_dir)
    cleaned = pathlib.Path(cleaned_dir)
    paths = download_all(raw)
    click.echo(f"Downloaded {len(paths)} files to {raw}")
    clean_all_resources(raw, cleaned)
    click.echo(f"Cleaned files written to {cleaned}")


if __name__ == "__main__":
    main()
