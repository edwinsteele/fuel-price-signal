"""Retire committed daily snapshots once bulk historical CSVs cover the same period.

See AGENTS.md § Snapshot retirement for the policy this implements. This is a
manually-run report/apply tool, not a scheduled job — a human reviews the
agreement numbers before anything gets deleted.
"""

import bisect
import calendar
import csv
import logging
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

import click

from fuel_signal.db import normalize_address
from fuel_signal.history import ResourceFetcher, clean_resource, discover_price_resources

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.05      # cents; two floats within this are "the same price"
DEFAULT_MIN_AGREEMENT = 0.95  # fraction of comparable rows that must agree to be eligible

_MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTH_NAMES.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTH_NAMES["sept"] = 9
_MONTH_RE = re.compile(
    "(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")(\d{2,4})"
)


def parse_resource_month(filename: str) -> tuple[int, int] | None:
    """Best-effort (year, month) from a bulk-CSV resource filename.

    Bulk resource filenames aren't consistently formatted (e.g.
    "fuelcheck_pricehistory_apr2026.xlsx", "price_history_checks_mar2026.csv"),
    so this takes the last month+year token found rather than assuming a fixed
    position. Returns None if no such token is found.
    """
    stem = pathlib.Path(filename).stem.lower()
    matches = list(_MONTH_RE.finditer(stem))
    if not matches:
        return None
    month_str, year_str = matches[-1].groups()
    month = _MONTH_NAMES[month_str]
    year = int(year_str)
    if year < 100:
        year += 2000
    return year, month


def committed_snapshot_months(snapshots_dir: pathlib.Path) -> dict[tuple[int, int], pathlib.Path]:
    """(year, month) -> directory, for every snapshot month dir with at least one file."""
    months = {}
    for year_dir in sorted(snapshots_dir.glob("[0-9][0-9][0-9][0-9]")):
        for month_dir in sorted(year_dir.glob("[0-9][0-9]")):
            if any(month_dir.glob("*.csv")):
                months[(int(year_dir.name), int(month_dir.name))] = month_dir
    return months


def build_historical_index(cleaned_path: pathlib.Path) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """(normalized_address, fuel_code) -> sorted [(date_str, price)] for as-of lookups."""
    idx: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    with open(cleaned_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            addr = normalize_address(row["Address"])
            date = row["PriceUpdatedDate"][:10]
            idx[(addr, row["FuelCode"])].append((date, float(row["Price"])))
    for series in idx.values():
        series.sort()
    return idx


def asof_price(idx: dict, addr: str, fuel: str, date: str) -> float | None:
    """Most recent historical price at or before `date`, or None if there isn't one."""
    series = idx.get((addr, fuel))
    if not series:
        return None
    dates = [d for d, _ in series]
    pos = bisect.bisect_right(dates, date) - 1
    if pos < 0:
        return None
    return series[pos][1]


@dataclass
class MonthReport:
    year: int
    month: int
    snapshot_dir: pathlib.Path
    rows_total: int = 0
    rows_comparable: int = 0
    rows_agree: int = 0
    diverging: list[tuple] = field(default_factory=list)

    @property
    def agreement_ratio(self) -> float:
        return self.rows_agree / self.rows_comparable if self.rows_comparable else 0.0

    def eligible(self, min_agreement: float) -> bool:
        return self.rows_comparable > 0 and self.agreement_ratio >= min_agreement


def compare_month(
    year: int, month: int, snapshot_dir: pathlib.Path, cleaned_path: pathlib.Path,
    tolerance: float,
) -> MonthReport:
    idx = build_historical_index(cleaned_path)
    report = MonthReport(year=year, month=month, snapshot_dir=snapshot_dir)

    for snap_file in sorted(snapshot_dir.glob("*.csv")):
        date = snap_file.stem
        with open(snap_file, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fuel = row.get("fuel_code", "")
                if not fuel:
                    continue
                addr = normalize_address(row.get("address", ""))
                report.rows_total += 1
                hist_price = asof_price(idx, addr, fuel, date)
                if hist_price is None:
                    continue
                report.rows_comparable += 1
                snap_price = float(row["price"])
                if abs(snap_price - hist_price) < tolerance:
                    report.rows_agree += 1
                else:
                    report.diverging.append(
                        (date, row.get("name", ""), fuel, snap_price, hist_price)
                    )
    return report


def find_candidate_months(
    raw_dir: pathlib.Path, cleaned_dir: pathlib.Path, snapshots_dir: pathlib.Path,
) -> list[tuple[int, int, pathlib.Path, pathlib.Path]]:
    """Bulk resources whose (year, month) overlap a committed snapshot month.

    Downloads + cleans the overlapping resources (cached — re-runs are cheap).
    Returns (year, month, snapshot_dir, cleaned_csv_path) tuples.
    """
    committed = committed_snapshot_months(snapshots_dir)
    if not committed:
        return []

    candidates = []
    for resource in discover_price_resources():
        parsed = parse_resource_month(resource["download_url"])
        if parsed is None:
            continue
        year, month = parsed
        snapshot_dir = committed.get((year, month))
        if snapshot_dir is None:
            continue
        logger.info("Fetching %s-%02d resource %s", year, month, resource["id"])
        fetcher = ResourceFetcher(resource["id"], resource["download_url"], raw_dir)
        raw_path = fetcher.fetch()
        cleaned_path = clean_resource(raw_path, cleaned_dir)
        candidates.append((year, month, snapshot_dir, cleaned_path))
    return candidates


@click.command("snapshot-retire")
@click.option("--raw-dir", default="data/raw", show_default=True)
@click.option("--cleaned-dir", default="data/cleaned", show_default=True)
@click.option("--snapshots-dir", default="data/snapshots", show_default=True)
@click.option("--tolerance", default=DEFAULT_TOLERANCE, show_default=True,
              help="Cents; snapshot and historical prices within this are 'the same'.")
@click.option("--min-agreement", default=DEFAULT_MIN_AGREEMENT, show_default=True,
              help="Fraction of comparable rows that must agree for a month to be eligible.")
@click.option("--show-diverging", default=5, show_default=True,
              help="Number of diverging example rows to print per month.")
@click.option("--apply", "apply_", is_flag=True, default=False,
              help="Delete eligible snapshot month directories. Without this flag, report only.")
def main(raw_dir: str, cleaned_dir: str, snapshots_dir: str, tolerance: float,
         min_agreement: float, show_diverging: int, apply_: bool) -> None:
    """Report (and optionally delete) snapshot months now covered by bulk historical CSVs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    candidates = find_candidate_months(
        pathlib.Path(raw_dir), pathlib.Path(cleaned_dir), pathlib.Path(snapshots_dir),
    )
    if not candidates:
        click.echo("No committed snapshot months overlap a published bulk historical CSV.")
        return

    to_delete = []
    for year, month, snapshot_dir, cleaned_path in candidates:
        report = compare_month(year, month, snapshot_dir, cleaned_path, tolerance)
        status = "ELIGIBLE" if report.eligible(min_agreement) else "keep"
        click.echo(
            f"{year}-{month:02d}: {report.rows_total} snapshot rows, "
            f"{report.rows_comparable} comparable, "
            f"{report.agreement_ratio:.1%} agree within {tolerance}c -> {status}"
        )
        for d in report.diverging[:show_diverging]:
            click.echo(f"    diverges: {d}")
        if report.eligible(min_agreement):
            to_delete.append((year, month, snapshot_dir))

    if not to_delete:
        click.echo("No months met the agreement threshold; nothing to retire.")
        return

    if not apply_:
        click.echo(
            f"\n{len(to_delete)} month(s) eligible for retirement. "
            "Re-run with --apply to delete them, then commit the deletion via a PR."
        )
        return

    for year, month, snapshot_dir in to_delete:
        n = 0
        for csv_path in snapshot_dir.glob("*.csv"):
            csv_path.unlink()
            n += 1
        click.echo(f"Deleted {n} file(s) from {snapshot_dir}")
    click.echo("\nReview with `git status`, then `git add` the deletions and open a PR.")


if __name__ == "__main__":
    main()
