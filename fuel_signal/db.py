"""SQLite schema and read/write helpers."""

import csv
import logging
import pathlib
import re
import sqlite3
from typing import Literal

import click

from fuel_signal.config import KNOWN_DUPLICATE_STATION_CODES
from fuel_signal.dates import date_from_int as _date_from_int
from fuel_signal.dates import date_to_int as _date_to_int
from fuel_signal.postcode_council import SYDNEY_METRO_POSTCODES, primary_council

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
CREATE TABLE IF NOT EXISTS fuel_types (
    id   INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE
);
INSERT OR IGNORE INTO fuel_types (code) VALUES
    ('E10'),('U91'),('P95'),('P98'),('PDL'),('DL'),('LPG'),('E85'),('CNG'),('B20');

CREATE TABLE IF NOT EXISTS price_sources (
    id   INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE
);
INSERT OR IGNORE INTO price_sources (code) VALUES ('h'),('s');

CREATE TABLE IF NOT EXISTS stations (
    station_code       INTEGER PRIMARY KEY,
    address_normalized TEXT NOT NULL UNIQUE,
    suburb             TEXT NOT NULL,
    postcode           TEXT NOT NULL,
    name               TEXT NOT NULL,
    brand              TEXT,
    council            TEXT,
    latitude           REAL,
    longitude          REAL
);

CREATE TABLE IF NOT EXISTS prices (
    station_code    INTEGER NOT NULL REFERENCES stations(station_code),
    fuel_type_id    INTEGER NOT NULL REFERENCES fuel_types(id),
    price_date      INTEGER NOT NULL,   -- YYYYMMDD e.g. 20240101
    price_decicents INTEGER NOT NULL,   -- price_cents * 10, e.g. 1619 = 161.9c
    source_id       INTEGER NOT NULL REFERENCES price_sources(id),
    PRIMARY KEY (station_code, fuel_type_id, price_date)
);

CREATE INDEX IF NOT EXISTS prices_fuel_date    ON prices(fuel_type_id, price_date);
CREATE INDEX IF NOT EXISTS prices_station_fuel ON prices(station_code, fuel_type_id);

CREATE TABLE IF NOT EXISTS daily_prices (
    station_code    INTEGER NOT NULL REFERENCES stations(station_code),
    fuel_type_id    INTEGER NOT NULL REFERENCES fuel_types(id),
    price_date      INTEGER NOT NULL,   -- YYYYMMDD
    price_decicents INTEGER NOT NULL,
    PRIMARY KEY (station_code, fuel_type_id, price_date)
);

CREATE INDEX IF NOT EXISTS daily_prices_fuel_date    ON daily_prices(fuel_type_id, price_date);
CREATE INDEX IF NOT EXISTS daily_prices_station_fuel ON daily_prices(station_code, fuel_type_id);

CREATE TABLE IF NOT EXISTS loaded_files (
    filename  TEXT PRIMARY KEY,
    loaded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);

CREATE TABLE IF NOT EXISTS station_class (
    station_code             INTEGER NOT NULL REFERENCES stations(station_code),
    snapshot_date            INTEGER NOT NULL,   -- YYYYMMDD; classification valid as of this date
    class                    TEXT    NOT NULL,   -- 'Competitive' | 'Sticky' | 'Discount'
    median_premium_decicents INTEGER NOT NULL,   -- median (station_price − cluster_ref) over 45d
    PRIMARY KEY (station_code, snapshot_date)
);

CREATE INDEX IF NOT EXISTS station_class_snapshot ON station_class(snapshot_date);

CREATE TABLE IF NOT EXISTS classification_summary (
    snapshot_date INTEGER NOT NULL,   -- YYYYMMDD
    lga           TEXT    NOT NULL,
    n_competitive INTEGER NOT NULL,
    n_sticky      INTEGER NOT NULL,
    n_discount    INTEGER NOT NULL,
    PRIMARY KEY (snapshot_date, lga)
);

CREATE TABLE IF NOT EXISTS lga_leadership (
    lga                     TEXT    NOT NULL,
    snapshot_date           INTEGER NOT NULL,   -- YYYYMMDD; scores valid as of this date
    trough_lead_median_days REAL,               -- NULL when no matched events
    trough_lead_consistency REAL,               -- 1/std; NULL when fewer than 2 matched events
    trough_match_fraction   REAL    NOT NULL,
    peak_lead_median_days   REAL,               -- v2; NULL in Phase 4
    peak_lead_consistency   REAL,               -- v2; NULL in Phase 4
    peak_match_fraction     REAL,               -- v2; NULL in Phase 4
    n_events_in_window      INTEGER NOT NULL,   -- LGA trough events detected in window
    PRIMARY KEY (lga, snapshot_date)
);
CREATE INDEX IF NOT EXISTS lga_leadership_snapshot ON lga_leadership(snapshot_date);
"""


# ---------------------------------------------------------------------------
# Storage format helpers
# ---------------------------------------------------------------------------


def _ensure_fuel_types(conn: sqlite3.Connection, codes: set[str]) -> dict[str, int]:
    """Insert any unknown fuel codes; return {code: id} for all known codes."""
    conn.executemany(
        "INSERT OR IGNORE INTO fuel_types (code) VALUES (?)", [(c,) for c in codes]
    )
    return {r[0]: r[1] for r in conn.execute("SELECT code, id FROM fuel_types")}


def _ensure_source_id(conn: sqlite3.Connection, code: str) -> int:
    """Insert source code if absent; return its id."""
    conn.execute("INSERT OR IGNORE INTO price_sources (code) VALUES (?)", (code,))
    return conn.execute(
        "SELECT id FROM price_sources WHERE code = ?", (code,)
    ).fetchone()[0]


def fuel_type_id(conn: sqlite3.Connection, code: str) -> int:
    """Return the fuel_types.id for a fuel code string, raising if not found."""
    row = conn.execute("SELECT id FROM fuel_types WHERE code = ?", (code,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown fuel type: {code!r}")
    return row[0]


def _in_clause(values) -> tuple[str, list]:
    """Return ('?,?,…', sorted_values) for a SQL ``IN (…)`` clause.

    Sorting gives deterministic placeholder ordering across call sites.
    """
    ordered = sorted(values)
    return ",".join("?" * len(ordered)), ordered


def _year_month_sql(col: str) -> str:
    """SQL fragment formatting a YYYYMMDD integer column as a 'YYYY-MM' string."""
    return f"PRINTF('%04d-%02d', {col}/10000, ({col}/100)%100)"


def open_db(db_path: pathlib.Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def is_file_loaded(conn: sqlite3.Connection, filename: str) -> bool:
    """Return True if filename has been recorded in loaded_files."""
    return conn.execute(
        "SELECT 1 FROM loaded_files WHERE filename = ?", (filename,)
    ).fetchone() is not None


def mark_file_loaded(conn: sqlite3.Connection, filename: str) -> None:
    """Record filename in loaded_files (idempotent)."""
    conn.execute(
        "INSERT OR IGNORE INTO loaded_files (filename) VALUES (?)", (filename,)
    )
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
            primary_council(s.get("postcode", "")),
            s.get("latitude"),
            s.get("longitude"),
        )
        for s in stations
    ]
    # Insert rows that don't exist yet (IGNORE both PK and address_normalized conflicts).
    conn.executemany(
        """INSERT OR IGNORE INTO stations
           (station_code, address_normalized, suburb, postcode, name, brand, council, latitude, longitude)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    # Update mutable fields; preserve existing lat/lon when incoming value is NULL
    # (snapshot CSVs don't carry coordinates); preserve existing suburb when
    # incoming value is blank (extraction failure).
    conn.executemany(
        """UPDATE stations
           SET name      = ?,
               brand     = ?,
               council   = COALESCE(?, council),
               suburb    = COALESCE(NULLIF(?, ''), suburb),
               latitude  = COALESCE(?, latitude),
               longitude = COALESCE(?, longitude)
           WHERE station_code = ?""",
        [
            (s.get("name", ""), s.get("brand"),
             primary_council(s.get("postcode", "")),
             s.get("suburb", ""),
             s.get("latitude"), s.get("longitude"), s["station_code"])
            for s in stations
        ],
    )
    conn.commit()
    return len(rows)


def insert_prices(conn: sqlite3.Connection, rows: list[dict], source: str = "h") -> int:
    """Bulk-insert prices; silently ignores duplicates. Returns rows inserted.

    Expected keys: station_code, fuel_code, price_date (YYYY-MM-DD), price_cents.
    source: 's' for snapshot, 'h' for historical CSV (default).
    """
    if not rows:
        return 0
    fuel_map = _ensure_fuel_types(conn, {r["fuel_code"] for r in rows})
    source_id = _ensure_source_id(conn, source)
    # total_changes delta counts only rows the INSERT OR IGNORE actually wrote
    # (ignored duplicates don't count) — avoids a full COUNT(*) scan per call.
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO prices"
        " (station_code, fuel_type_id, price_date, price_decicents, source_id)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            (r["station_code"], fuel_map[r["fuel_code"]],
             _date_to_int(r["price_date"]), round(r["price_cents"] * 10), source_id)
            for r in rows
        ],
    )
    conn.commit()
    return conn.total_changes - before


# ---------------------------------------------------------------------------
# Load from snapshot CSVs
# ---------------------------------------------------------------------------

def load_snapshot_csv(
    conn: sqlite3.Connection,
    csv_path: pathlib.Path,
    postcodes: frozenset[str] | None = None,
    fuel_codes: set[str] | None = None,
) -> tuple[int, int]:
    """Load a snapshot CSV into stations + prices tables.

    Snapshot CSV schema: station_code, name, address, suburb, postcode, brand, fuel_code, price, date

    postcodes:  if set, only load stations whose postcode is in this set.
    fuel_codes: if set, only load prices for fuel types in this set.
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
            postcode = row.get("postcode", "")
            if postcodes is not None and postcode not in postcodes:
                continue
            stations.append({
                "station_code": code,
                "name":         row.get("name", ""),
                "address":      row.get("address", ""),
                "suburb":       row.get("suburb", ""),
                "postcode":     postcode,
                "brand":        row.get("brand"),
                "latitude":     None,
                "longitude":    None,
            })
            fuel_code = row.get("fuel_code", "")
            if not fuel_code:
                continue
            if fuel_codes is not None and fuel_code not in fuel_codes:
                continue
            try:
                price_cents = float(row["price"])
            except (ValueError, KeyError):
                continue
            prices.append({
                "station_code": code,
                "fuel_code":    fuel_code,
                "price_date":   row["date"],
                "price_cents":  price_cents,
            })

    # Only upsert stations that have at least one price row after filtering.
    # This naturally excludes EV chargers and other non-fuel venues that share
    # addresses with petrol stations (they have no E10 prices and would otherwise
    # cause duplicate normalised-address collisions).
    stations_with_prices = {p["station_code"] for p in prices}
    stations = [s for s in stations if s["station_code"] in stations_with_prices]

    n_stations = upsert_stations(conn, stations)

    # INSERT OR IGNORE handles UNIQUE violations but not FK violations. Some station_codes
    # may have been silently dropped by upsert_stations when two codes share the same
    # normalised address. Filter prices to avoid FK errors on those orphaned codes.
    batch_codes = {s["station_code"] for s in stations}
    if batch_codes:
        placeholders, ordered_codes = _in_clause(batch_codes)
        known_codes = {
            r[0] for r in conn.execute(
                f"SELECT station_code FROM stations WHERE station_code IN ({placeholders})",
                ordered_codes,
            )
        }
        dropped_codes = batch_codes - known_codes
        if dropped_codes:
            station_lookup = {s["station_code"]: s["name"] for s in stations}
            unexpected = dropped_codes - KNOWN_DUPLICATE_STATION_CODES
            if unexpected:
                detail = ", ".join(
                    f"{code} ({station_lookup.get(code, '?')})" for code in sorted(unexpected)
                )
                logger.warning(
                    "%d station(s) dropped (duplicate normalised address); skipping their prices: %s",
                    len(unexpected), detail,
                )
        prices = [p for p in prices if p["station_code"] in known_codes]

    return n_stations, insert_prices(conn, prices, source="s")


def load_all_snapshots(
    conn: sqlite3.Connection,
    snapshots_dir: pathlib.Path,
    postcodes: frozenset[str] | None = None,
    fuel_codes: set[str] | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Load every snapshot CSV found under snapshots_dir. Returns (stations, prices).

    Already-loaded files (tracked in loaded_files table) are skipped unless force=True.
    postcodes and fuel_codes are passed through to load_snapshot_csv for load-time filtering.
    """
    total_stations = total_prices = 0
    for path in sorted(snapshots_dir.rglob("*.csv")):
        if not force and is_file_loaded(conn, path.name):
            logger.debug("Snapshot %s: already loaded, skipping", path.name)
            continue
        s, p = load_snapshot_csv(conn, path, postcodes=postcodes, fuel_codes=fuel_codes)
        mark_file_loaded(conn, path.name)
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
    suburb_backfill: dict[int, str] | None = None,
) -> tuple[int, int]:
    """Load a historical cleaned CSV; matches rows to stations by normalised address.

    Historical CSV schema (from history.py Transformer):
        ServiceStationName, Address, Suburb, Postcode, Brand, FuelCode,
        PriceUpdatedDate, Price

    If suburb_backfill is provided, it is updated in-place with the first
    non-blank suburb seen per station code (for use by backfill_station_suburbs).

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

            if suburb_backfill is not None and station_code not in suburb_backfill:
                suburb = row.get("Suburb", "").strip()
                if suburb:
                    suburb_backfill[station_code] = suburb

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

    return insert_prices(conn, prices), skipped


def backfill_station_suburbs(conn: sqlite3.Connection, suburb_backfill: dict[int, str]) -> int:
    """Update stations that have a blank suburb using data collected from historical CSVs.

    This is the correct place to do suburb backfill (rather than during the history.py
    clean phase) because station_code — the primary key — only comes from the FuelCheck
    API live snapshot. The stations table doesn't exist until after a snapshot is loaded,
    so historical CSVs can't populate it directly. The clean phase produces suburb data
    but has no station_code to key it on; the backfill bridges that gap after both
    sources have been loaded.

    Only updates rows where suburb is currently blank; never overwrites an existing value.
    Returns the number of rows updated.
    """
    if not suburb_backfill:
        return 0
    blank_before = conn.execute(
        "SELECT COUNT(*) FROM stations WHERE suburb = ''"
    ).fetchone()[0]
    conn.executemany(
        "UPDATE stations SET suburb = ? WHERE station_code = ? AND suburb = ''",
        [(suburb, code) for code, suburb in suburb_backfill.items()],
    )
    conn.commit()
    blank_after = conn.execute(
        "SELECT COUNT(*) FROM stations WHERE suburb = ''"
    ).fetchone()[0]
    return blank_before - blank_after


def load_all_cleaned(
    conn: sqlite3.Connection,
    cleaned_dir: pathlib.Path,
    force: bool = False,
) -> tuple[int, int]:
    """Load all historical cleaned CSVs. Returns (total_inserted, total_skipped).

    Already-loaded files (tracked in loaded_files table) are skipped unless force=True.
    """
    addr_idx = _address_index(conn)
    logger.info("Address index: %d stations", len(addr_idx))
    suburb_backfill: dict[int, str] = {}
    total_inserted = total_skipped = 0
    for path in sorted(cleaned_dir.glob("*.csv")):
        if not force and is_file_loaded(conn, path.name):
            logger.debug("%s: already loaded, skipping", path.name)
            continue
        inserted, skipped = load_cleaned_csv(conn, path, addr_idx, suburb_backfill)
        mark_file_loaded(conn, path.name)
        logger.debug("%s: %d inserted, %d skipped", path.name, inserted, skipped)
        total_inserted += inserted
        total_skipped += skipped
    n_backfilled = backfill_station_suburbs(conn, suburb_backfill)
    logger.info(
        "Historical load complete: %d inserted, %d skipped (no station match), "
        "%d station suburb(s) backfilled from historical data",
        total_inserted, total_skipped, n_backfilled,
    )
    return total_inserted, total_skipped


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def daily_average_e10(
    conn: sqlite3.Connection,
    start_date: str | None = None,
) -> list[tuple[str, float]]:
    """Return [(price_date, avg_price_cents)] for E10 across all stations (raw prices)."""
    fid = fuel_type_id(conn, "E10")
    query = "SELECT price_date, AVG(price_decicents) FROM prices WHERE fuel_type_id = ?"
    params: list = [fid]
    if start_date:
        query += " AND price_date >= ?"
        params.append(_date_to_int(start_date))
    query += " GROUP BY price_date ORDER BY price_date"
    return [(_date_from_int(r[0]), r[1] / 10) for r in conn.execute(query, params)]


def station_price_series(
    conn: sqlite3.Connection,
    station_code: int,
    fuel_code: str = "E10",
    start_date: str | None = None,
) -> list[tuple[str, float]]:
    """Return [(price_date, price_cents)] for a single station."""
    fid = fuel_type_id(conn, fuel_code)
    query = (
        "SELECT price_date, price_decicents FROM prices"
        " WHERE station_code = ? AND fuel_type_id = ?"
    )
    params: list = [station_code, fid]
    if start_date:
        query += " AND price_date >= ?"
        params.append(_date_to_int(start_date))
    query += " ORDER BY price_date"
    return [(_date_from_int(r[0]), r[1] / 10) for r in conn.execute(query, params)]


def _average_price_series_by(
    conn: sqlite3.Connection,
    fuel_code: str,
    column: str,
    values: frozenset[str] | None,
) -> list[tuple[str, float]]:
    """Return [(price_date, avg_price_cents)] from daily_prices (gap-filled).

    When ``values`` is given, average only stations whose ``s.<column>`` is in
    the set. ``column`` is an internal literal ('council' or 'brand'), never
    user input, so interpolating it into the SQL is safe.
    """
    fid = fuel_type_id(conn, fuel_code)
    if values:
        placeholders, ordered = _in_clause(values)
        query = (
            "SELECT dp.price_date, AVG(dp.price_decicents)"
            " FROM daily_prices dp"
            " JOIN stations s USING(station_code)"
            f" WHERE dp.fuel_type_id = ? AND s.{column} IN ({placeholders})"  # noqa: S608
            " GROUP BY dp.price_date ORDER BY dp.price_date"
        )
        params: list = [fid, *ordered]
    else:
        query = (
            "SELECT price_date, AVG(price_decicents) FROM daily_prices"
            " WHERE fuel_type_id = ? GROUP BY price_date ORDER BY price_date"
        )
        params = [fid]
    return [(_date_from_int(r[0]), r[1] / 10) for r in conn.execute(query, params)]


def average_price_series(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    councils: frozenset[str] | None = None,
) -> list[tuple[str, float]]:
    """Return [(price_date, avg_price_cents)] from daily_prices (gap-filled).

    councils: if provided, average only stations whose council is in this set.
              Defaults to all stations in daily_prices (which are already
              metro-filtered at DB load time).

    Requires fill.fill_all() to have been run first.
    """
    return _average_price_series_by(conn, fuel_code, "council", councils)


def upsert_daily_prices(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, str, float]],
) -> None:
    """Insert rows into daily_prices.

    rows: list of (station_code, fuel_code, date_str YYYY-MM-DD, price_cents).
    """
    if not rows:
        return
    fuel_map = _ensure_fuel_types(conn, {r[1] for r in rows})
    conn.executemany(
        "INSERT INTO daily_prices (station_code, fuel_type_id, price_date, price_decicents)"
        " VALUES (?, ?, ?, ?)",
        [(r[0], fuel_map[r[1]], _date_to_int(r[2]), round(r[3] * 10)) for r in rows],
    )


def get_daily_prices(
    conn: sqlite3.Connection,
    station_code: int,
    fuel_code: str = "E10",
) -> list[tuple[str, float]]:
    """Return [(price_date, price_cents)] from daily_prices for a single station."""
    fid = fuel_type_id(conn, fuel_code)
    return [
        (_date_from_int(r[0]), r[1] / 10)
        for r in conn.execute(
            "SELECT price_date, price_decicents FROM daily_prices"
            " WHERE station_code = ? AND fuel_type_id = ? ORDER BY price_date",
            (station_code, fid),
        )
    ]


def coverage_by_month(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    months: int = 30,
) -> list[tuple[str, int]]:
    """Return [(YYYY-MM, station_count)] for the most recent months."""
    fid = fuel_type_id(conn, fuel_code)
    rows = conn.execute(
        f"SELECT {_year_month_sql('price_date')} AS ym,"  # noqa: S608
        "       COUNT(DISTINCT station_code)"
        " FROM prices WHERE fuel_type_id = ?"
        " GROUP BY ym ORDER BY ym DESC LIMIT ?",
        (fid, months),
    ).fetchall()
    return list(rows)


def recent_prices(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    days: int = 14,
) -> list[tuple[str, str, str, float]]:
    """Return [(date_str, station_name, suburb, price_cents)] for recent prices."""
    import datetime
    cutoff = _date_to_int(
        (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    )
    fid = fuel_type_id(conn, fuel_code)
    rows = conn.execute(
        "SELECT p.price_date, s.name, s.suburb, p.price_decicents"
        " FROM prices p JOIN stations s USING(station_code)"
        " WHERE p.fuel_type_id = ? AND p.price_date >= ?"
        " ORDER BY p.price_date DESC, p.price_decicents",
        (fid, cutoff),
    ).fetchall()
    return [(_date_from_int(r[0]), r[1], r[2], r[3] / 10) for r in rows]


def db_summary(conn: sqlite3.Connection) -> dict:
    """Return basic stats for display in the inspection page."""
    station_count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    price_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    fid = fuel_type_id(conn, "E10")
    date_range = conn.execute(
        "SELECT MIN(price_date), MAX(price_date) FROM prices WHERE fuel_type_id = ?",
        (fid,),
    ).fetchone()
    return {
        "station_count": station_count,
        "price_count": price_count,
        "earliest_date": _date_from_int(date_range[0]) if date_range[0] else "—",
        "latest_date": _date_from_int(date_range[1]) if date_range[1] else "—",
    }


def average_price_series_by_brand(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    brands: frozenset[str] | None = None,
) -> list[tuple[str, float]]:
    """Return [(price_date, avg_price_cents)] from daily_prices grouped by brand.

    brands: if provided, average only stations whose brand is in this set.
    Same gap-fill semantics as average_price_series (uses daily_prices).
    """
    return _average_price_series_by(conn, fuel_code, "brand", brands)


def distinct_brands(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    min_stations: int = 3,
) -> list[str]:
    """Return brands represented by at least min_stations stations in daily_prices.

    min_stations cuts off the long tail of one-off brand labels.
    Brands are counted by distinct station_code, not price rows.
    """
    fid = fuel_type_id(conn, fuel_code)
    rows = conn.execute(
        "SELECT s.brand, COUNT(DISTINCT s.station_code) AS cnt"
        " FROM daily_prices dp JOIN stations s USING(station_code)"
        " WHERE dp.fuel_type_id = ? AND s.brand IS NOT NULL AND s.brand != ''"
        " GROUP BY s.brand HAVING cnt >= ?"
        " ORDER BY s.brand",
        (fid, min_stations),
    ).fetchall()
    return [r[0] for r in rows]


def station_search(
    conn: sqlite3.Connection,
    q: str,
    limit: int = 20,
) -> list[tuple[int, str, str, str]]:
    """Search stations by name or suburb.

    Returns [(station_code, name, suburb, brand)] ordered by name.
    Used by the /api/stations/search typeahead endpoint.
    """
    pattern = f"%{q}%"
    rows = conn.execute(
        "SELECT station_code, name, suburb, brand FROM stations"
        " WHERE name LIKE ? OR suburb LIKE ?"
        " ORDER BY name LIMIT ?",
        (pattern, pattern, limit),
    ).fetchall()
    return [(r[0], r[1], r[2] or "", r[3] or "") for r in rows]


def coverage_matrix(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    months: int = 24,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[tuple[int, str, str, int]]:
    """Return [(station_code, name, ym, n_observations)] for coverage heatmap.

    ym is a 'YYYY-MM' string.  When start_date/end_date (YYYY-MM-DD) are
    given they filter at the price_date level before monthly aggregation,
    giving exact-date semantics.  Otherwise covers the most recent *months*.
    """
    import datetime as _dt
    fid = fuel_type_id(conn, fuel_code)

    params: list = [fid]
    where_clauses = ["p.fuel_type_id = ?"]

    ym_expr = _year_month_sql("p.price_date")
    if start_date:
        where_clauses.append("p.price_date >= ?")
        params.append(_date_to_int(start_date))
    else:
        today = _dt.date.today()
        months = max(1, months)
        # Step back (months - 1) whole months from the current month.
        total_months = today.year * 12 + (today.month - 1) - (months - 1)
        cutoff_ym = f"{total_months // 12:04d}-{total_months % 12 + 1:02d}"
        where_clauses.append(f"{ym_expr} >= ?")
        params.append(cutoff_ym)

    if end_date:
        where_clauses.append("p.price_date <= ?")
        params.append(_date_to_int(end_date))

    sql = (
        f"SELECT p.station_code, s.name, {ym_expr} AS ym, COUNT(*) AS n"  # noqa: S608
        " FROM prices p JOIN stations s USING(station_code)"
        f" WHERE {' AND '.join(where_clauses)}"
        " GROUP BY p.station_code, ym ORDER BY ym, s.name"
    )
    return list(conn.execute(sql, params).fetchall())


def gradient_by_lga(
    conn: sqlite3.Connection,
    fuel_code: str = "E10",
    window_days: int = 1,
    councils: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Return [(council, date, slope_cents_per_day)] for gradient heatmap.

    When window_days=1 (default) each row is one calendar day and slope is
    the numpy.gradient at that day.  When window_days>1 rows are bucketed into
    ISO-week windows and slope is the mean gradient across the bucket.

    Pass councils to restrict results to a subset of LGAs (SQL-level filter).
    """
    import datetime as _dt
    from itertools import groupby

    import numpy as np

    fid = fuel_type_id(conn, fuel_code)
    cond = "dp.fuel_type_id = ? AND s.council IS NOT NULL"
    params: list = [fid]
    if councils:
        placeholders, ordered = _in_clause(councils)
        cond += f" AND s.council IN ({placeholders})"
        params.extend(ordered)

    all_rows = conn.execute(
        f"SELECT s.council, dp.price_date, AVG(dp.price_decicents)"  # noqa: S608
        f" FROM daily_prices dp JOIN stations s USING(station_code)"
        f" WHERE {cond}"
        f" GROUP BY s.council, dp.price_date ORDER BY s.council, dp.price_date",
        params,
    ).fetchall()
    if not all_rows:
        return []

    results: list[tuple[str, str, float]] = []
    for council, council_rows_iter in groupby(all_rows, key=lambda r: r[0]):
        council_rows = list(council_rows_iter)
        if len(council_rows) < max(window_days, 1):
            continue
        dates = [_date_from_int(r[1]) for r in council_rows]
        prices = np.array([r[2] / 10 for r in council_rows])
        gradients = np.gradient(prices)

        if window_days == 1:
            for date_str, grad in zip(dates, gradients):
                results.append((council, date_str, float(grad)))
        else:
            # Bucket by calendar week (ISO Monday)
            buckets: dict[str, list[float]] = {}
            for date_str, grad in zip(dates, gradients):
                d = _dt.date.fromisoformat(date_str)
                week_start = (d - _dt.timedelta(days=d.weekday())).isoformat()
                buckets.setdefault(week_start, []).append(float(grad))
            for week_start, grads in sorted(buckets.items()):
                results.append((council, week_start, float(np.mean(grads))))

    return results


# ---------------------------------------------------------------------------
# Snapshot-table helpers (shared by station_class / classification_summary /
# lga_leadership — all keyed by an integer snapshot_date)
# ---------------------------------------------------------------------------

_SnapshotTable = Literal["station_class", "classification_summary", "lga_leadership"]


def _delete_for_snapshot_date(conn: sqlite3.Connection, table: _SnapshotTable, snapshot_date: str) -> int:
    """Remove all rows in ``table`` for snapshot_date. Returns rows deleted.

    ``table`` is an internal literal, never user input.
    """
    cur = conn.execute(
        f"DELETE FROM {table} WHERE snapshot_date = ?",  # noqa: S608
        (_date_to_int(snapshot_date),),
    )
    return cur.rowcount or 0


def _latest_snapshot_date(conn: sqlite3.Connection, table: _SnapshotTable) -> str | None:
    """Return the most recent snapshot_date (YYYY-MM-DD) in ``table``, or None."""
    row = conn.execute(f"SELECT MAX(snapshot_date) FROM {table}").fetchone()  # noqa: S608
    return _date_from_int(row[0]) if row and row[0] else None


# ---------------------------------------------------------------------------
# Station classification helpers
# ---------------------------------------------------------------------------

def daily_prices_in_window(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    fuel_code: str = "E10",
) -> list[tuple[int, str, str, float]]:
    """Return [(station_code, council, price_date, price_cents)] from daily_prices.

    Rows are restricted to price_date in [start_date, end_date] inclusive,
    and to stations with a non-null council (LGA classification key).

    Used by the station classifier to compute per-(LGA, day) cluster medians
    and per-station premium series in a single pass.
    """
    fid = fuel_type_id(conn, fuel_code)
    rows = conn.execute(
        "SELECT dp.station_code, s.council, dp.price_date, dp.price_decicents"
        " FROM daily_prices dp JOIN stations s USING(station_code)"
        " WHERE dp.fuel_type_id = ? AND s.council IS NOT NULL"
        "   AND dp.price_date >= ? AND dp.price_date <= ?",
        (fid, _date_to_int(start_date), _date_to_int(end_date)),
    ).fetchall()
    return [(r[0], r[1], _date_from_int(r[2]), r[3] / 10) for r in rows]


def upsert_station_class_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, str, int]],
) -> None:
    """Insert/replace station_class rows.

    rows: list of (station_code, snapshot_date YYYY-MM-DD, class, median_premium_decicents).
    """
    if not rows:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO station_class"
        " (station_code, snapshot_date, class, median_premium_decicents)"
        " VALUES (?, ?, ?, ?)",
        [(r[0], _date_to_int(r[1]), r[2], r[3]) for r in rows],
    )


def delete_station_class_for_date(conn: sqlite3.Connection, snapshot_date: str) -> int:
    """Remove all station_class rows for snapshot_date. Returns rows deleted."""
    return _delete_for_snapshot_date(conn, "station_class", snapshot_date)


def upsert_classification_summary_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, int, int, int]],
) -> None:
    """Insert/replace classification_summary rows.

    rows: list of (snapshot_date YYYY-MM-DD, lga, n_competitive, n_sticky, n_discount).
    """
    if not rows:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO classification_summary"
        " (snapshot_date, lga, n_competitive, n_sticky, n_discount)"
        " VALUES (?, ?, ?, ?, ?)",
        [(_date_to_int(r[0]), r[1], r[2], r[3], r[4]) for r in rows],
    )


def delete_classification_summary_for_date(
    conn: sqlite3.Connection, snapshot_date: str
) -> int:
    """Remove all classification_summary rows for snapshot_date. Returns rows deleted."""
    return _delete_for_snapshot_date(conn, "classification_summary", snapshot_date)


def get_station_class(
    conn: sqlite3.Connection,
    station_code: int,
    snapshot_date: str,
) -> tuple[str, int] | None:
    """Return (class, median_premium_decicents) for a station on snapshot_date, or None."""
    row = conn.execute(
        "SELECT class, median_premium_decicents FROM station_class"
        " WHERE station_code = ? AND snapshot_date = ?",
        (station_code, _date_to_int(snapshot_date)),
    ).fetchone()
    return (row[0], row[1]) if row else None


def latest_station_class_date(conn: sqlite3.Connection) -> str | None:
    """Return the most recent snapshot_date (YYYY-MM-DD) in station_class, or None."""
    return _latest_snapshot_date(conn, "station_class")


# ---------------------------------------------------------------------------
# LGA leadership helpers
# ---------------------------------------------------------------------------

def upsert_lga_leadership_rows(
    conn: sqlite3.Connection,
    rows: list[tuple],
) -> None:
    """Insert/replace lga_leadership rows.

    rows: list of 9-tuples (lga, snapshot_date YYYY-MM-DD,
          trough_lead_median_days, trough_lead_consistency, trough_match_fraction,
          peak_lead_median_days, peak_lead_consistency, peak_match_fraction,
          n_events_in_window).
    """
    if not rows:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO lga_leadership"
        " (lga, snapshot_date, trough_lead_median_days, trough_lead_consistency,"
        "  trough_match_fraction, peak_lead_median_days, peak_lead_consistency,"
        "  peak_match_fraction, n_events_in_window)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r[0], _date_to_int(r[1]), r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in rows],
    )


def delete_lga_leadership_for_date(conn: sqlite3.Connection, snapshot_date: str) -> int:
    """Remove all lga_leadership rows for snapshot_date. Returns rows deleted."""
    return _delete_for_snapshot_date(conn, "lga_leadership", snapshot_date)


def latest_lga_leadership_date(conn: sqlite3.Connection) -> str | None:
    """Return the most recent snapshot_date (YYYY-MM-DD) in lga_leadership, or None."""
    return _latest_snapshot_date(conn, "lga_leadership")


def get_lga_leadership_board(
    conn: sqlite3.Connection,
    snapshot_date: str,
) -> list[tuple]:
    """Return leadership rows for snapshot_date, sorted by trough_lead_consistency desc.

    Returns list of (lga, trough_lead_median_days, trough_lead_consistency,
    trough_match_fraction, n_events_in_window).
    """
    rows = conn.execute(
        "SELECT lga, trough_lead_median_days, trough_lead_consistency,"
        "       trough_match_fraction, n_events_in_window"
        " FROM lga_leadership WHERE snapshot_date = ?"
        " ORDER BY (trough_lead_consistency IS NULL), trough_lead_consistency DESC",
        (_date_to_int(snapshot_date),),
    ).fetchall()
    return list(rows)


# ---------------------------------------------------------------------------
# Entry point: rebuild DB from snapshots + historical data
# ---------------------------------------------------------------------------

@click.command("db")
@click.option(
    "--db",
    "db_path",
    default=str(DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option("--force", is_flag=True, help="Re-ingest all files, ignoring loaded_files tracking.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(db_path: str, force: bool, verbose: bool) -> None:
    """Rebuild the SQLite database from snapshots + historical data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    conn = open_db(pathlib.Path(db_path))
    create_schema(conn)
    logger.info("Schema ready at %s", db_path)

    snapshots_dir = pathlib.Path("data/snapshots")
    if snapshots_dir.exists():
        s, p = load_all_snapshots(
            conn, snapshots_dir,
            postcodes=SYDNEY_METRO_POSTCODES,
            fuel_codes={"E10"},
            force=force,
        )
        logger.info("Snapshots: %d stations, %d new prices", s, p)
    else:
        logger.warning("No data/snapshots directory — run live.py first to populate stations")

    cleaned_dir = pathlib.Path("data/cleaned")
    if cleaned_dir.exists():
        inserted, skipped = load_all_cleaned(conn, cleaned_dir, force=force)
        logger.info("Historical: %d inserted, %d skipped", inserted, skipped)

    conn.close()


if __name__ == "__main__":
    main()
