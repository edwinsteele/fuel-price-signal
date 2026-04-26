"""Collect a live E10 price snapshot from the FuelCheck API and save it as CSV."""

import csv
import datetime
import logging
import pathlib
import re
import uuid

import requests

from fuel_signal.config import (
    FUELAPI_API_KEY,
    FUELAPI_API_SECRET,
    FUELAPI_PRICES_URL,
    FUELAPI_TOKEN_URL,
    SYDNEY_METRO_POSTCODES,
)

logger = logging.getLogger(__name__)

_SUBURB_PC_RE = re.compile(
    r",\s*([^,]+?)\s+(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\s+(\d{4})\s*$",
    re.IGNORECASE,
)
_PC_ONLY_RE = re.compile(r"\b(\d{4})\b(?=[^0-9]*$)")
# Matches the state+postcode suffix so we can strip it and work on what's before.
_STATE_SUFFIX_RE = re.compile(
    r"^(.*?)\s+(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\s+(\d{4})\s*$",
    re.IGNORECASE,
)
# Splits an address on street-type words; suburb is the last segment after the split.
_STREET_TYPE_RE = re.compile(
    r"\b(?:STREET|ROAD|HIGHWAY|AVENUE|DRIVE|LANE|CRESCENT|BOULEVARD|COURT|"
    r"TERRACE|PLACE|PARADE|CLOSE|GROVE|CIRCUIT|PARKWAY|FREEWAY|EXPRESSWAY|"
    r"ST|RD|HWY|AVE|DR|LN|CRES|BLVD|CT|TCE|PL|PDE|CL|GR|CCT|PKWY|FWY|EXWY)\.?\b",
    re.IGNORECASE,
)

SNAPSHOT_COLUMNS = ("station_code", "name", "address", "suburb", "postcode", "brand", "price", "date")


# ---------------------------------------------------------------------------
# API auth
# ---------------------------------------------------------------------------

def get_token(api_key: str, api_secret: str) -> str:
    """Obtain an OAuth2 Bearer token via client_credentials."""
    resp = requests.get(
        FUELAPI_TOKEN_URL,
        params={"grant_type": "client_credentials"},
        auth=(api_key, api_secret),
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# API data fetch
# ---------------------------------------------------------------------------

def fetch_prices(token: str, api_key: str) -> dict:
    """Return the raw /fuel/prices JSON: {stations: [...], prices: [...]}."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    resp = requests.get(
        FUELAPI_PRICES_URL,
        headers={
            "Authorization":    f"Bearer {token}",
            "Content-Type":     "application/json; charset=utf-8",
            "apikey":           api_key,
            "transactionid":    uuid.uuid4().hex,
            "requesttimestamp": now_utc.strftime("%d/%m/%Y %I:%M:%S %p"),
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Snapshot row construction
# ---------------------------------------------------------------------------

def _station_suburb_postcode(address: str) -> tuple[str, str]:
    """Extract (suburb, postcode) from an API address string."""
    # Primary: comma-separated suburb — "123 Main St, Suburb NSW 2000"
    m = _SUBURB_PC_RE.search(address)
    if m:
        return m.group(1).strip().title(), m.group(2)

    # Fallback: no comma — "123 Main Rd Suburb NSW 2000" or corner addresses.
    # Strip the state+postcode suffix, then split on street-type words; suburb
    # is whatever remains after the last street-type word.
    m = _STATE_SUFFIX_RE.match(address.strip())
    if m:
        before_state, postcode = m.group(1), m.group(2)
        parts = _STREET_TYPE_RE.split(before_state)
        suburb = parts[-1].strip().strip(",-").strip()
        if suburb:
            return suburb.title(), postcode
        return "", postcode

    # No state abbreviation at all — just grab last 4-digit number
    m = _PC_ONLY_RE.search(address)
    if m:
        return "", m.group(1)

    return "", ""


def build_snapshot_rows(
    data: dict,
    fuel_code: str = "E10",
    postcodes: frozenset[str] = SYDNEY_METRO_POSTCODES,
    snapshot_date: datetime.date | None = None,
) -> list[dict]:
    """Filter API response to fuel_code + metro postcodes; return snapshot rows.

    Each row matches the SNAPSHOT_COLUMNS schema:
        station_code, name, address, suburb, postcode, brand, price, date
    """
    if snapshot_date is None:
        snapshot_date = datetime.date.today()
    date_str = snapshot_date.isoformat()

    station_map: dict[str, dict] = {s["code"]: s for s in data.get("stations", [])}

    # Pre-build postcode lookup from station metadata to avoid re-parsing per price row.
    station_postcode: dict[str, str] = {}
    for code, station in station_map.items():
        _, pc = _station_suburb_postcode(station.get("address", ""))
        station_postcode[code] = pc

    rows: list[dict] = []
    for price in data.get("prices", []):
        if price.get("fueltype") != fuel_code:
            continue
        code = price.get("stationcode", "")
        pc = station_postcode.get(code, "")
        if pc not in postcodes:
            continue
        station = station_map.get(code)
        if not station:
            continue
        address = station.get("address", "")
        suburb, _ = _station_suburb_postcode(address)
        rows.append({
            "station_code": int(code),
            "name":         station.get("name", ""),
            "address":      address,
            "suburb":       suburb,
            "postcode":     pc,
            "brand":        station.get("brand", ""),
            "price":        price["price"],
            "date":         date_str,
        })

    return rows


# ---------------------------------------------------------------------------
# Snapshot CSV persistence
# ---------------------------------------------------------------------------

def save_snapshot(
    rows: list[dict],
    snapshot_date: datetime.date,
    snapshots_dir: pathlib.Path,
) -> pathlib.Path:
    """Write rows to data/snapshots/YYYY/MM/YYYY-MM-DD.csv. Returns the path."""
    yyyy = snapshot_date.strftime("%Y")
    mm = snapshot_date.strftime("%m")
    dest = snapshots_dir / yyyy / mm / f"{snapshot_date.isoformat()}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows to %s", len(rows), dest)
    return dest


# ---------------------------------------------------------------------------
# Top-level collect function
# ---------------------------------------------------------------------------

def collect_snapshot(
    api_key: str = FUELAPI_API_KEY,
    api_secret: str = FUELAPI_API_SECRET,
    snapshots_dir: pathlib.Path = pathlib.Path("data/snapshots"),
    snapshot_date: datetime.date | None = None,
) -> pathlib.Path:
    """Fetch current prices, filter to E10/Sydney-metro, write snapshot CSV."""
    if not api_key or not api_secret:
        raise RuntimeError("FUELAPI_API_KEY and FUELAPI_API_SECRET must be set")

    if snapshot_date is None:
        snapshot_date = datetime.date.today()

    logger.info("Fetching FuelCheck token")
    token = get_token(api_key, api_secret)

    logger.info("Fetching all fuel prices")
    data = fetch_prices(token, api_key)
    logger.info(
        "API returned %d stations, %d prices",
        len(data.get("stations", [])),
        len(data.get("prices", [])),
    )

    rows = build_snapshot_rows(data, snapshot_date=snapshot_date)
    logger.info("Filtered to %d E10 Sydney-metro rows", len(rows))

    return save_snapshot(rows, snapshot_date, snapshots_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = collect_snapshot()
    print(f"Snapshot written to {path}")
