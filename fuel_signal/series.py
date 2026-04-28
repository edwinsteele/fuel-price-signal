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


def resolve(conn: sqlite3.Connection, spec: str, fuel: str = "E10") -> ResolvedSeries:
    """Return a ResolvedSeries for *spec*, raising SeriesError if unresolvable."""
    s = spec.strip()
    sl = s.lower()

    if sl == "sydney":
        return ResolvedSeries(
            spec="sydney",
            label=f"Sydney {fuel} mean",
            points=_db.average_price_series(conn, fuel_code=fuel),
            kind="sydney",
        )

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
            matches = [c for c in SYDNEY_METRO_COUNCILS if query in c.lower()]
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
    """Return {lgas: [...], brands: [...]} for the controls-form picker."""
    lgas = [
        c for c in sorted(SYDNEY_METRO_COUNCILS)
        if conn.execute("SELECT 1 FROM stations WHERE council = ? LIMIT 1", (c,)).fetchone()
    ]
    brands = sorted(_db.distinct_brands(conn, fuel_code="E10"))
    return {"lgas": lgas, "brands": brands}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_lga(conn: sqlite3.Connection, query: str, fuel: str) -> ResolvedSeries:
    matches = [c for c in SYDNEY_METRO_COUNCILS if query in c.lower()]
    if not matches:
        known = ", ".join(sorted(SYDNEY_METRO_COUNCILS))
        raise SeriesError(f"No LGA matching {query!r}. Known LGAs: {known}")
    if len(matches) > 1:
        raise SeriesError(
            f"Ambiguous LGA {query!r}, matches: {', '.join(sorted(matches))}. Be more specific."
        )
    council = matches[0]
    return ResolvedSeries(
        spec=f"lga:{council}",
        label=f"{council} LGA {fuel} mean",
        points=_db.average_price_series(conn, fuel_code=fuel, councils=frozenset({council})),
        kind="lga",
    )


def _resolve_brand(conn: sqlite3.Connection, brand_query: str, fuel: str) -> ResolvedSeries:
    brand = _lookup_brand(conn, brand_query, fuel)
    return ResolvedSeries(
        spec=f"brand:{brand}",
        label=f"{brand} {fuel} mean",
        points=_db.average_price_series_by_brand(conn, fuel_code=fuel, brands=frozenset({brand})),
        kind="brand",
    )


def _lookup_brand(conn: sqlite3.Connection, brand_query: str, fuel: str) -> str:
    """Return the exact brand name matching *brand_query* (case-insensitive)."""
    brands = _db.distinct_brands(conn, fuel_code=fuel, min_stations=1)
    brand_map = {b.lower(): b for b in brands}
    key = brand_query.lower()
    if key not in brand_map:
        raise SeriesError(
            f"No brand matching {brand_query!r}. Known brands: {', '.join(sorted(brands))}"
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
    return ResolvedSeries(
        spec=f"station:{code}",
        label=label,
        points=_db.get_daily_prices(conn, station_code=code, fuel_code=fuel),
        kind="station",
    )


def _rows_to_series(
    conn: sqlite3.Connection,
    stations: list[tuple[int, str, str]],
    fuel: str,
) -> list[ResolvedSeries]:
    results = []
    for code, name, suburb in stations:
        points = _db.get_daily_prices(conn, station_code=code, fuel_code=fuel)
        if points:
            label = f"{name} ({suburb})" if suburb else name
            results.append(
                ResolvedSeries(spec=f"station:{code}", label=label, points=points, kind="station")
            )
    return results
