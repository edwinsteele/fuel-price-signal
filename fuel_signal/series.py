"""Series resolver: map spec strings to ResolvedSeries objects.

Accepted spec forms
-------------------
  "sydney"           → Sydney metro E10 average
  "lga:<name>"       → LGA/council average (partial, case-insensitive)
  "council:<name>"   → same as lga:
  "brand:<name>"     → brand average
  "station:<code>"   → station by numeric code
  "<text>"           → station name search (must uniquely match one station)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from fuel_signal import db as _db
from fuel_signal.postcode_council import SYDNEY_METRO_COUNCILS


class SeriesError(ValueError):
    """Raised when a series spec cannot be resolved."""


@dataclass
class ResolvedSeries:
    spec: str                          # canonical, e.g. "lga:Penrith"
    label: str                         # human-readable, e.g. "Penrith LGA E10 mean"
    points: list[tuple[str, float]]    # (YYYY-MM-DD, cents/litre)
    kind: str                          # "station" | "lga" | "brand" | "sydney"


# Full series cache: (id(conn), canonical_spec, fuel) → [(date, price), ...]
# Keyed by conn identity so each DB connection (server run or test) has its own
# namespace.  No eviction; cache lives for the process lifetime.
_SERIES_CACHE: dict[tuple[int, str, str], list[tuple[str, float]]] = {}

# Groups cache: id(conn) → {"lgas": [...], "brands": [...]}
_GROUPS_CACHE: dict[int, dict] = {}

# Brand map cache: (id(conn), fuel) → {brand_lower: brand_canonical}
_BRAND_MAP_CACHE: dict[tuple[int, str], dict[str, str]] = {}


def _cached_series(
    conn: sqlite3.Connection,
    canonical_spec: str,
    fuel: str,
    fetch,
) -> list[tuple[str, float]]:
    """Return series from cache, calling *fetch()* on miss."""
    key = (id(conn), canonical_spec, fuel)
    if key not in _SERIES_CACHE:
        _SERIES_CACHE[key] = fetch()
    return _SERIES_CACHE[key]


def resolve(conn: sqlite3.Connection, spec: str, fuel: str = "E10") -> ResolvedSeries:
    """Return a ResolvedSeries for *spec*, raising SeriesError if unresolvable."""
    s = spec.strip()
    sl = s.lower()

    if sl == "sydney":
        points = _cached_series(
            conn, "sydney", fuel,
            lambda: _db.average_price_series(conn, fuel_code=fuel),
        )
        return ResolvedSeries(spec="sydney", label=f"Sydney {fuel} mean",
                              points=points, kind="sydney")

    for prefix in ("lga:", "council:"):
        if sl.startswith(prefix):
            return _resolve_lga(conn, sl[len(prefix):].strip(), fuel)

    if sl.startswith("brand:"):
        return _resolve_brand(conn, s[len("brand:"):].strip(), fuel)

    if sl.startswith("station:"):
        return _resolve_station(conn, s[len("station:"):].strip(), fuel)

    # Bare text → station name search
    return _resolve_station(conn, s, fuel)


def resolve_members(
    conn: sqlite3.Connection,
    spec: str,
    fuel: str = "E10",
) -> list[ResolvedSeries]:
    """Expand an lga: or brand: spec into per-station ResolvedSeries.

    Returns an empty list for specs that don't support expansion
    (sydney, station:) or when no stations match.
    """
    sl = spec.strip().lower()

    for prefix in ("lga:", "council:"):
        if sl.startswith(prefix):
            query = sl[len(prefix):].strip()
            lower_query = query.lower()
            exact = [c for c in SYDNEY_METRO_COUNCILS if c.lower() == lower_query]
            if exact:
                matches = exact
            else:
                matches = [c for c in SYDNEY_METRO_COUNCILS if lower_query in c.lower()]
            if not matches:
                return []
            council = matches[0]
            stations = conn.execute(
                "SELECT station_code, name, suburb FROM stations"
                " WHERE council = ? ORDER BY name",
                (council,),
            ).fetchall()
            return _rows_to_series(conn, stations, fuel)

    if sl.startswith("brand:"):
        brand_query = spec.strip()[len("brand:"):].strip()
        try:
            brand = _lookup_brand(conn, brand_query, fuel)
        except SeriesError:
            return []
        stations = conn.execute(
            "SELECT station_code, name, suburb FROM stations"
            " WHERE brand = ? ORDER BY name",
            (brand,),
        ).fetchall()
        return _rows_to_series(conn, stations, fuel)

    return []


def enumerate_groups(conn: sqlite3.Connection) -> dict:
    """Return {lgas: [(name, count), ...], brands: [(name, count), ...]} for the controls-form picker.

    Counts reflect distinct stations per LGA/brand in the DB.
    """
    key = id(conn)
    if key in _GROUPS_CACHE:
        return _GROUPS_CACHE[key]
    council_counts = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT council, COUNT(*) FROM stations"
            " WHERE council IS NOT NULL GROUP BY council"
        ).fetchall()
    }
    lgas = [(c, council_counts[c]) for c in sorted(SYDNEY_METRO_COUNCILS) if c in council_counts]

    fid = _db.fuel_type_id(conn, "E10")
    brand_rows = conn.execute(
        "SELECT s.brand, COUNT(DISTINCT s.station_code) AS cnt"
        " FROM daily_prices dp JOIN stations s USING(station_code)"
        " WHERE dp.fuel_type_id = ? AND s.brand IS NOT NULL AND s.brand != ''"
        " GROUP BY s.brand HAVING cnt >= 3"
        " ORDER BY s.brand",
        (fid,),
    ).fetchall()
    brands = [(brand, cnt) for brand, cnt in brand_rows]

    _GROUPS_CACHE[key] = {"lgas": lgas, "brands": brands}
    return _GROUPS_CACHE[key]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_lga(conn: sqlite3.Connection, query: str, fuel: str) -> ResolvedSeries:
    # Prefer exact case-insensitive match; fall back to substring.
    lower_query = query.lower()
    exact = [c for c in SYDNEY_METRO_COUNCILS if c.lower() == lower_query]
    if exact:
        matches = exact
    else:
        matches = [c for c in SYDNEY_METRO_COUNCILS if lower_query in c.lower()]
    if not matches:
        known = ", ".join(sorted(SYDNEY_METRO_COUNCILS))
        raise SeriesError(f"No LGA matching {query!r}. Known LGAs: {known}")
    if len(matches) > 1:
        raise SeriesError(
            f"Ambiguous LGA {query!r}, matches: {', '.join(sorted(matches))}. Be more specific."
        )
    council = matches[0]
    canonical = f"lga:{council}"
    points = _cached_series(
        conn, canonical, fuel,
        lambda: _db.average_price_series(conn, fuel_code=fuel, councils=frozenset({council})),
    )
    return ResolvedSeries(spec=canonical, label=f"{council} LGA {fuel} mean",
                          points=points, kind="lga")


def _resolve_brand(conn: sqlite3.Connection, brand_query: str, fuel: str) -> ResolvedSeries:
    brand = _lookup_brand(conn, brand_query, fuel)
    canonical = f"brand:{brand}"
    points = _cached_series(
        conn, canonical, fuel,
        lambda: _db.average_price_series_by_brand(conn, fuel_code=fuel, brands=frozenset({brand})),
    )
    return ResolvedSeries(spec=canonical, label=f"{brand} {fuel} mean",
                          points=points, kind="brand")


def _lookup_brand(conn: sqlite3.Connection, brand_query: str, fuel: str) -> str:
    """Return the exact brand name matching *brand_query* (case-insensitive)."""
    cache_key = (id(conn), fuel)
    if cache_key not in _BRAND_MAP_CACHE:
        brands = _db.distinct_brands(conn, fuel_code=fuel, min_stations=1)
        _BRAND_MAP_CACHE[cache_key] = {b.lower(): b for b in brands}
    brand_map = _BRAND_MAP_CACHE[cache_key]
    key = brand_query.lower()
    if key not in brand_map:
        raise SeriesError(
            f"No brand matching {brand_query!r}. Known brands: {', '.join(sorted(brand_map.values()))}"
        )
    return brand_map[key]


def _resolve_station(conn: sqlite3.Connection, spec: str, fuel: str) -> ResolvedSeries:
    """Resolve a station by numeric code or station-name search."""
    if spec.isdigit():
        rows = conn.execute(
            "SELECT station_code, name, suburb FROM stations WHERE station_code = ?",
            (int(spec),),
        ).fetchall()
    else:
        # Name-only search avoids matching every station in a suburb
        rows = conn.execute(
            "SELECT station_code, name, suburb FROM stations"
            " WHERE name LIKE ?"
            " ORDER BY suburb, name",
            (f"%{spec}%",),
        ).fetchall()

    if not rows:
        raise SeriesError(
            f"No station found matching {spec!r}. "
            f"Use 'uv run python -m fuel_signal.stations {spec!r}' to search."
        )
    if len(rows) > 1:
        lines = "\n".join(
            f"  station:{code:<8}  {suburb or '':<22}  {sname}"
            for code, sname, suburb in rows
        )
        raise SeriesError(
            f"Multiple stations match {spec!r} — use station:CODE to disambiguate:\n{lines}"
        )

    code, sname, suburb = rows[0]
    label = f"{sname} ({suburb})" if suburb else sname
    canonical = f"station:{code}"
    points = _cached_series(
        conn, canonical, fuel,
        lambda: _db.get_daily_prices(conn, station_code=code, fuel_code=fuel),
    )
    return ResolvedSeries(spec=canonical, label=label, points=points, kind="station")


def _rows_to_series(
    conn: sqlite3.Connection,
    stations: list[tuple[int, str, str]],
    fuel: str,
) -> list[ResolvedSeries]:
    results = []
    for code, name, suburb in stations:
        canonical = f"station:{code}"
        points = _cached_series(
            conn, canonical, fuel,
            lambda c=code: _db.get_daily_prices(conn, station_code=c, fuel_code=fuel),
        )
        if points:
            label = f"{name} ({suburb})" if suburb else name
            results.append(ResolvedSeries(spec=canonical, label=label, points=points, kind="station"))
    return results
