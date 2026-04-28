"""Command-line interface for fuel-price-signal."""

import pathlib

import click

from fuel_signal import db as _db


@click.group()
def cli() -> None:
    """Fuel price signal tools."""


@cli.command()
@click.argument("query", required=False)
@click.option("--suburb", "-s", help="Filter by suburb (partial match).")
@click.option("--name", "-n", help="Filter by station name (partial match).")
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite DB.",
)
def stations(query: str | None, suburb: str | None, name: str | None, db_path: str) -> None:
    """Search stations by suburb or name.

    QUERY searches both suburb and name simultaneously.
    Use --suburb or --name for field-specific filtering.

    Examples:\n
        fuel-signal stations blaxland\n
        fuel-signal stations --suburb "emu plains"\n
        fuel-signal stations --name ampol
    """
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.live' first."
        )

    conn = _db.open_db(path)
    conditions: list[str] = []
    params: list[str] = []

    if query:
        if query.isdigit():
            conditions.append("station_code = ?")
            params.append(query)
        else:
            conditions.append("(suburb LIKE ? OR name LIKE ?)")
            params += [f"%{query}%", f"%{query}%"]
    if suburb:
        conditions.append("suburb LIKE ?")
        params.append(f"%{suburb}%")
    if name:
        conditions.append("name LIKE ?")
        params.append(f"%{name}%")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = conn.execute(
        f"SELECT station_code, name, suburb, postcode, brand FROM stations {where} ORDER BY suburb, name",
        params,
    ).fetchall()
    conn.close()

    if not rows:
        click.echo("No stations found.")
        return

    click.echo(f"{'Code':<8}  {'Suburb':<22}  {'Name':<40}  {'Brand'}")
    click.echo("-" * 90)
    for code, sname, ssuburb, postcode, brand in rows:
        click.echo(f"{code:<8}  {(ssuburb or ''):<22}  {sname:<40}  {brand or ''}")
    click.echo(f"\n{len(rows)} station(s) found.")


if __name__ == "__main__":
    cli()
