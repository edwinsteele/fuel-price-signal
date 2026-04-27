"""Compare two E10 price series — how often one is cheaper than the other."""

import pathlib

import click

from fuel_signal import db as _db
from fuel_signal.postcode_council import SYDNEY_METRO_COUNCILS


def _resolve_series(
    conn,
    spec: str,
    fuel_code: str,
) -> tuple[str, list[tuple[str, float]]]:
    """Return (label, [(date, price_cents)]) for a series specifier.

    Accepted forms:
      "sydney"           → Sydney metro average (all stations in daily_prices)
      "lga:<name>"       → LGA/council average; partial case-insensitive match
      "council:<name>"   → same as lga:
      anything else      → station name/suburb search (must match exactly one station)
    """
    spec_stripped = spec.strip()
    spec_lower = spec_stripped.lower()

    if spec_lower == "sydney":
        series = _db.average_price_series(conn, fuel_code=fuel_code)
        return (f"Sydney {fuel_code} mean", series)

    for prefix in ("lga:", "council:"):
        if spec_lower.startswith(prefix):
            query = spec_lower[len(prefix):].strip()
            matches = [c for c in SYDNEY_METRO_COUNCILS if query in c.lower()]
            if not matches:
                known = ", ".join(sorted(SYDNEY_METRO_COUNCILS))
                raise click.ClickException(
                    f"No LGA matching {query!r}. Known LGAs: {known}"
                )
            if len(matches) > 1:
                raise click.ClickException(
                    f"Ambiguous LGA {query!r}, matches: {', '.join(sorted(matches))}. Be more specific."
                )
            council = matches[0]
            series = _db.average_price_series(
                conn, fuel_code=fuel_code, councils=frozenset({council})
            )
            return (f"{council} LGA {fuel_code} mean", series)

    rows = conn.execute(
        "SELECT station_code, name, suburb FROM stations"
        " WHERE name LIKE ? OR suburb LIKE ?"
        " ORDER BY suburb, name",
        (f"%{spec_stripped}%", f"%{spec_stripped}%"),
    ).fetchall()

    if not rows:
        raise click.ClickException(
            f"No station found matching {spec!r}. "
            f"Use 'uv run python -m fuel_signal.cli stations {spec!r}' to search."
        )
    if len(rows) > 1:
        lines = "\n".join(
            f"  {code:<8}  {suburb or '':<22}  {sname}"
            for code, sname, suburb in rows
        )
        raise click.ClickException(
            f"Multiple stations match {spec!r} — be more specific:\n{lines}"
        )

    code, sname, suburb = rows[0]
    label = f"{sname} ({suburb})" if suburb else sname
    series = _db.get_daily_prices(conn, station_code=code, fuel_code=fuel_code)
    return (label, series)


@click.command("compare")
@click.argument("series_a")
@click.argument("series_b")
@click.option("--fuel", default="E10", show_default=True, help="Fuel type.")
@click.option(
    "--within",
    default=0.5,
    show_default=True,
    help="Treat prices as equal when difference is ≤ this many cents.",
)
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite DB.",
)
def main(
    series_a: str,
    series_b: str,
    fuel: str,
    within: float,
    db_path: str,
) -> None:
    """Compare how often one price series is cheaper than another.

    SERIES_A and SERIES_B can each be:\n
      - A station name or suburb (partial match; must be unique)\n
      - 'sydney' for the Sydney metro average\n
      - 'lga:<name>' or 'council:<name>' for an LGA average\n

    Examples:\n
        uv run python -m fuel_signal.compare "BP Valley Heights" sydney\n
        uv run python -m fuel_signal.compare "Valley Heights" "lga:penrith"\n
        uv run python -m fuel_signal.compare "Ampol Springwood" "Caltex Springwood"
    """
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )

    conn = _db.open_db(path)
    try:
        label_a, a_series = _resolve_series(conn, series_a, fuel)
        label_b, b_series = _resolve_series(conn, series_b, fuel)
    finally:
        conn.close()

    if not a_series:
        raise click.ClickException(f"No data for {series_a!r}.")
    if not b_series:
        raise click.ClickException(f"No data for {series_b!r}.")

    a_map = dict(a_series)
    b_map = dict(b_series)
    common = sorted(set(a_map) & set(b_map))

    if not common:
        raise click.ClickException("No overlapping dates between the two series.")

    a_cheaper_savings: list[float] = []
    b_cheaper_savings: list[float] = []
    equal_count = 0
    all_diffs: list[float] = []

    for d in common:
        diff = a_map[d] - b_map[d]  # negative when A is cheaper
        all_diffs.append(diff)
        if abs(diff) <= within:
            equal_count += 1
        elif diff < 0:
            a_cheaper_savings.append(-diff)
        else:
            b_cheaper_savings.append(diff)

    n = len(common)
    a_count = len(a_cheaper_savings)
    b_count = len(b_cheaper_savings)
    mean_a = sum(a_map[d] for d in common) / n
    mean_b = sum(b_map[d] for d in common) / n
    overall_diff = mean_a - mean_b  # negative = A cheaper overall

    w = max(len(label_a), len(label_b))

    def _pct(k: int) -> str:
        return f"{100 * k / n:.1f}%"

    click.echo(f"\nComparing {fuel}: {label_a} vs {label_b}")
    click.echo(f"Period: {common[0]} to {common[-1]} ({n:,} overlapping days)\n")

    a_avg_str = f"  avg {sum(a_cheaper_savings)/a_count:.1f}c cheaper" if a_count else ""
    b_avg_str = f"  avg {sum(b_cheaper_savings)/b_count:.1f}c cheaper" if b_count else ""
    eq_label = f"Equal (within {within:g}c)"

    click.echo(f"  {label_a + ' cheaper':<{w+8}}  {a_count:>6,} days  ({_pct(a_count):>6}){a_avg_str}")
    click.echo(f"  {label_b + ' cheaper':<{w+8}}  {b_count:>6,} days  ({_pct(b_count):>6}){b_avg_str}")
    click.echo(f"  {eq_label:<{w+8}}  {equal_count:>6,} days  ({_pct(equal_count):>6})")

    click.echo(f"\nMean {fuel}  {label_a}: {mean_a:.1f}c   {label_b}: {mean_b:.1f}c")

    if abs(overall_diff) <= within:
        click.echo("Overall: negligible mean difference")
    elif overall_diff < 0:
        click.echo(f"Overall: {-overall_diff:.1f}c cheaper at {label_a} on average")
    else:
        click.echo(f"Overall: {overall_diff:.1f}c cheaper at {label_b} on average")


if __name__ == "__main__":
    main()
