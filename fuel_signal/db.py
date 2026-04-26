"""SQLite schema and read/write helpers."""

import csv
import logging
import pathlib
import re
import sqlite3

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = pathlib.Path("fuel_signal.db")

# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

# Street-type abbreviations used in historical CSVs (keys) and their full forms.
# Applied only to the street portion of an address (before the first comma) to
# avoid mangling suburb names like "St Marys" or "St Ives".
_STREET_ABBREVS: dict[str, str] = {
    "ST": "STREET",
    "RD": "ROAD",
    "AVE": "AVENUE",
    "HWY": "HIGHWAY",
    "LN": "LANE",
    "DR": "DRIVE",
    "CRES": "CRESCENT",
    "BLVD": "BOULEVARD",
    "CT": "COURT",
    "TCE": "TERRACE",
    "PL": "PLACE",
    "PDE": "PARADE",
    "CL": "CLOSE",
    "GR": "GROVE",
    "CCT": "CIRCUIT",
    "PKWY": "PARKWAY",
    "FWY": "FREEWAY",
    "EXWY": "EXPRESSWAY",
}

_STATE_PC_RE = re.compile(
    r",?\s*(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\s+\d{4}\s*$", re.IGNORECASE
)
_PUNCT_RE = re.compile(r"[,./]")
_WS_RE = re.compile(r"\s+")


def normalize_address(raw: str) -> str:
    """Return a canonical lowercase address string for cross-source joining.

    Strips state/postcode suffix, expands street-type abbreviations in the
    street portion only (safe for suburbs like "St Marys"), then normalises
    punctuation and whitespace.
    """
    s = raw.upper()
    s = _STATE_PC_RE.sub("", s)

    # Expand abbreviations only before the first comma (street portion).
    # The optional trailing period handles "ST." forms that appear in some source data.
    comma = s.find(",")
    street, rest = (s[:comma], s[comma:]) if comma >= 0 else (s, "")
    for abbrev, full in _STREET_ABBREVS.items():
        street = re.sub(r"\b" + abbrev + r"\.?(?=[\s,]|$)", full, street)
    s = street + rest

    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s.lower()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS stations (
    station_code       INTEGER PRIMARY KEY,
    address_normalized TEXT NOT NULL UNIQUE,
    suburb             TEXT NOT NULL,
    postcode           TEXT NOT NULL,
    name               TEXT NOT NULL,
    brand              TEXT,
    latitude           REAL,
    longitude          REAL
);

CREATE TABLE IF NOT EXISTS prices (
    station_code  INTEGER NOT NULL REFERENCES stations(station_code),
    fuel_code     TEXT NOT NULL,
    price_date    DATE NOT NULL,
    price_cents   REAL NOT NULL,
    source        TEXT NOT NULL DEFAULT 'h',  -- 's' = snapshot, 'h' = historical CSV
    PRIMARY KEY (station_code, fuel_code, price_date)
);

CREATE INDEX IF NOT EXISTS prices_fuel_date   ON prices(fuel_code, price_date);
CREATE INDEX IF NOT EXISTS prices_station_fuel ON prices(station_code, fuel_code);
"""


def open_db(db_path: pathlib.Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def upsert_stations(conn: sqlite3.Connection, stations: list[dict]) -> int:
    """Insert or update stations.  Handles rebrands (name/brand update in place).

    Expected keys: station_code, name, address, suburb, postcode, brand,
                   latitude (optional), longitude (optional).
    Returns number of rows processed.
    """
    rows = [
        (
            s["station_code"],
            normalize_address(s["address"]),
            s.get("suburb", ""),
            s.get("postcode", ""),
            s.get("name", ""),
            s.get("brand"),
            s.get("latitude"),
            s.get("longitude"),
        )
        for s in stations
    ]
    # Insert rows that don't exist yet (IGNORE both PK and address_normalized conflicts).
    conn.executemany(
        """INSERT OR IGNORE INTO stations
           (station_code, address_normalized, suburb, postcode, name, brand, latitude, longitude)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    # Update mutable fields; preserve existing lat/lon when incoming value is NULL
    # (snapshot CSVs don't carry coordinates).
    conn.executemany(
        """UPDATE stations
           SET name      = ?,
               brand     = ?,
               latitude  = COALESCE(?, latitude),
               longitude = COALESCE(?, longitude)
           WHERE station_code = ?""",
        [
            (s.get("name", ""), s.get("brand"), s.get("latitude"), s.get("longitude"), s["station_code"])
            for s in stations
        ],
    )
    conn.commit()
    return len(rows)


def insert_prices(conn: sqlite3.Connection, rows: list[dict], source: str = "h") -> None:
    """Bulk-insert prices; silently ignores duplicates.

    Expected keys: station_code, fuel_code, price_date (YYYY-MM-DD), price_cents.
    source: 's' for snapshot, 'h' for historical CSV (default).
    """
    conn.executemany(
        "INSERT OR IGNORE INTO prices (station_code, fuel_code, price_date, price_cents, source) VALUES (?, ?, ?, ?, ?)",
        [(r["station_code"], r["fuel_code"], r["price_date"], r["price_cents"], source) for r in rows],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Load from snapshot CSVs
# ---------------------------------------------------------------------------

def load_snapshot_csv(conn: sqlite3.Connection, csv_path: pathlib.Path) -> tuple[int, int]:
    """Load a snapshot CSV into stations + prices tables.

    Snapshot CSV schema: station_code, name, address, suburb, postcode, brand, price, date
    Returns (stations_processed, prices_inserted).
    """
    stations: list[dict] = []
    prices: list[dict] = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                code = int(row["station_code"])
            except (ValueError, KeyError):
                continue
            stations.append({
                "station_code": code,
                "name":         row.get("name", ""),
                "address":      row.get("address", ""),
                "suburb":       row.get("suburb", ""),
                "postcode":     row.get("postcode", ""),
                "brand":        row.get("brand"),
                "latitude":     None,
                "longitude":    None,
            })
            try:
                price_cents = float(row["price"])
            except (ValueError, KeyError):
                continue
            prices.append({
                "station_code": code,
                "fuel_code":    "E10",
                "price_date":   row["date"],
                "price_cents":  price_cents,
            })

    n_stations = upsert_stations(conn, stations)

    # INSERT OR IGNORE handles UNIQUE violations but not FK violations. Some station_codes
    # may have been silently dropped by upsert_stations when two codes share the same
    # normalised address. Filter prices to avoid FK errors on those orphaned codes.
    batch_codes = {s["station_code"] for s in stations}
    if batch_codes:
        placeholders = ",".join("?" * len(batch_codes))
        known_codes = {
            r[0] for r in conn.execute(
                f"SELECT station_code FROM stations WHERE station_code IN ({placeholders})",
                sorted(batch_codes),
            )
        }
        dropped = len(batch_codes) - len(known_codes)
        if dropped:
            logger.warning("%d station(s) dropped (duplicate normalised address); skipping their prices", dropped)
        prices = [p for p in prices if p["station_code"] in known_codes]

    before = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    insert_prices(conn, prices, source="s")
    after = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    return n_stations, after - before


def load_all_snapshots(conn: sqlite3.Connection, snapshots_dir: pathlib.Path) -> tuple[int, int]:
    """Load every snapshot CSV found under snapshots_dir. Returns (stations, prices)."""
    total_stations = total_prices = 0
    for path in sorted(snapshots_dir.rglob("*.csv")):
        s, p = load_snapshot_csv(conn, path)
        logger.info("Snapshot %s: %d stations, %d new prices", path.name, s, p)
        total_stations += s
        total_prices += p
    return total_stations, total_prices


# ---------------------------------------------------------------------------
# Load from historical cleaned CSVs
# ---------------------------------------------------------------------------

def _address_index(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        row[0]: row[1]
        for row in conn.execute("SELECT address_normalized, station_code FROM stations")
    }


def load_cleaned_csv(
    conn: sqlite3.Connection,
    csv_path: pathlib.Path,
    addr_idx: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Load a historical cleaned CSV; matches rows to stations by normalised address.

    Historical CSV schema (from history.py Transformer):
        ServiceStationName, Address, Suburb, Postcode, Brand, FuelCode,
        PriceUpdatedDate, Price

    Returns (inserted, skipped).
    """
    if addr_idx is None:
        addr_idx = _address_index(conn)

    prices: list[dict] = []
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            addr_norm = normalize_address(row.get("Address", ""))
            station_code = addr_idx.get(addr_norm)
            if station_code is None:
                skipped += 1
                continue

            raw_date = row.get("PriceUpdatedDate", "")
            if not raw_date:
                skipped += 1
                continue
            price_date = raw_date[:10]  # truncate to YYYY-MM-DD

            try:
                price_cents = float(row["Price"])
            except (ValueError, KeyError):
                skipped += 1
                continue

            fuel_code = row.get("FuelCode", "")
            if not fuel_code:
                skipped += 1
                continue

            prices.append({
                "station_code": station_code,
                "fuel_code":    fuel_code,
                "price_date":   price_date,
                "price_cents":  price_cents,
            })

    before = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    if prices:
        insert_prices(conn, prices)
    after = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    return after - before, skipped


def load_all_cleaned(
    conn: sqlite3.Connection,
    cleaned_dir: pathlib.Path,
) -> tuple[int, int]:
    """Load all historical cleaned CSVs. Returns (total_inserted, total_skipped)."""
    addr_idx = _address_index(conn)
    logger.info("Address index: %d stations", len(addr_idx))
    total_inserted = total_skipped = 0
    for path in sorted(cleaned_dir.glob("*.csv")):
        inserted, skipped = load_cleaned_csv(conn, path, addr_idx)
        logger.debug("%s: %d inserted, %d skipped", path.name, inserted, skipped)
        total_inserted += inserted
        total_skipped += skipped
    logger.info(
        "Historical load complete: %d inserted, %d skipped (no station match)",
        total_inserted, total_skipped,
    )
    return total_inserted, total_skipped


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def daily_average_e10(
    conn: sqlite3.Connection,
    start_date: str | None = None,
) -> list[tuple[str, float]]:
    """Return [(price_date, avg_price_cents)] for E10 across all stations."""
    query = """
        SELECT price_date, AVG(price_cents)
        FROM prices
        WHERE fuel_code = 'E10'
    """
    params: list = []
    if start_date:
        query += " AND price_date >= ?"
        params.append(start_date)
    query += " GROUP BY price_date ORDER BY price_date"
    return [(r[0], r[1]) for r in conn.execute(query, params)]


def station_price_series(
    conn: sqlite3.Connection,
    station_code: int,
    fuel_code: str = "E10",
    start_date: str | None = None,
) -> list[tuple[str, float]]:
    """Return [(price_date, price_cents)] for a single station."""
    query = "SELECT price_date, price_cents FROM prices WHERE station_code=? AND fuel_code=?"
    params: list = [station_code, fuel_code]
    if start_date:
        query += " AND price_date >= ?"
        params.append(start_date)
    query += " ORDER BY price_date"
    return [(r[0], r[1]) for r in conn.execute(query, params)]


def db_summary(conn: sqlite3.Connection) -> dict:
    """Return basic stats for display in the inspection page."""
    station_count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    price_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    date_range = conn.execute(
        "SELECT MIN(price_date), MAX(price_date) FROM prices WHERE fuel_code='E10'"
    ).fetchone()
    return {
        "station_count": station_count,
        "price_count": price_count,
        "earliest_date": date_range[0] or "—",
        "latest_date": date_range[1] or "—",
    }


# ---------------------------------------------------------------------------
# Entry point: rebuild DB from snapshots + historical data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    db_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB_PATH
    conn = open_db(db_path)
    create_schema(conn)
    logger.info("Schema ready at %s", db_path)

    snapshots_dir = pathlib.Path("data/snapshots")
    if snapshots_dir.exists():
        s, p = load_all_snapshots(conn, snapshots_dir)
        logger.info("Snapshots: %d stations, %d new prices", s, p)
    else:
        logger.warning("No data/snapshots directory — run live.py first to populate stations")

    cleaned_dir = pathlib.Path("data/cleaned")
    if cleaned_dir.exists():
        inserted, skipped = load_all_cleaned(conn, cleaned_dir)
        logger.info("Historical: %d inserted, %d skipped", inserted, skipped)

    conn.close()
