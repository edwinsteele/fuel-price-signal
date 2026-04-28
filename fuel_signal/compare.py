"""Compare two E10 price series — how often one is cheaper than the other."""

import pathlib

import click

from fuel_signal import db as _db
from fuel_signal.series import SeriesError, resolve


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
      - A station name (partial match; must be unique) or station:CODE\n
      - 'sydney' for the Sydney metro average\n
      - 'lga:<name>' or 'council:<name>' for an LGA average\n
      - 'brand:<name>' for a brand average\n

    Examples:\n
        uv run python -m fuel_signal.compare "BP Springwood" sydney\n
        uv run python -m fuel_signal.compare station:182 "lga:penrith"\n
        uv run python -m fuel_signal.compare "Ampol Springwood" "Shell Blaxland"
    """
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )

    conn = _db.open_db(path)
    try:
        try:
            ra = resolve(conn, series_a, fuel)
        except SeriesError as e:
            raise click.ClickException(str(e))
        try:
            rb = resolve(conn, series_b, fuel)
        except SeriesError as e:
            raise click.ClickException(str(e))
    finally:
        conn.close()

    if not ra.points:
        raise click.ClickException(f"No data for {series_a!r}.")
    if not rb.points:
        raise click.ClickException(f"No data for {series_b!r}.")

    a_map = dict(ra.points)
    b_map = dict(rb.points)
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

    w = max(len(ra.label), len(rb.label))

    def _pct(k: int) -> str:
        return f"{100 * k / n:.1f}%"

    click.echo(f"\nComparing {fuel}: {ra.label} vs {rb.label}")
    click.echo(f"Period: {common[0]} to {common[-1]} ({n:,} overlapping days)\n")

    a_avg_str = f"  avg {sum(a_cheaper_savings)/a_count:.1f}c cheaper" if a_count else ""
    b_avg_str = f"  avg {sum(b_cheaper_savings)/b_count:.1f}c cheaper" if b_count else ""
    eq_label = f"Equal (within {within:g}c)"

    click.echo(f"  {ra.label + ' cheaper':<{w+8}}  {a_count:>6,} days  ({_pct(a_count):>6}){a_avg_str}")
    click.echo(f"  {rb.label + ' cheaper':<{w+8}}  {b_count:>6,} days  ({_pct(b_count):>6}){b_avg_str}")
    click.echo(f"  {eq_label:<{w+8}}  {equal_count:>6,} days  ({_pct(equal_count):>6})")

    click.echo(f"\nMean {fuel}  {ra.label}: {mean_a:.1f}c   {rb.label}: {mean_b:.1f}c")

    if abs(overall_diff) <= within:
        click.echo("Overall: negligible mean difference")
    elif overall_diff < 0:
        click.echo(f"Overall: {-overall_diff:.1f}c cheaper at {ra.label} on average")
    else:
        click.echo(f"Overall: {overall_diff:.1f}c cheaper at {rb.label} on average")


if __name__ == "__main__":
    main()
