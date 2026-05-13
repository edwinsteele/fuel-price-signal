"""Phase 2 τ re-validation on realised spend (Issue #64).

Sweeps τ ∈ [0.30, 0.55] on the test window via the backtest engine
using data/models/logreg_calibrated.joblib, then patches
experiments/results.csv with realised-spend columns for the Phase 2
(τ=0.40) and always-buy baseline rows.

Usage:
    uv run python -m fuel_signal.backtest_phase2 \\
        [--model-path data/models/logreg_calibrated.joblib] [--db fuel_signal.db]
    uv run python -m fuel_signal.backtest_phase2 --no-patch   # dry-run; print only
"""

from __future__ import annotations

import csv
import math
import pathlib
import tempfile
from typing import Sequence

import click

import fuel_signal.db as db
from fuel_signal.backtest import (
    AlwaysBuyStrategy,
    ModelStrategy,
    PriceHistory,
    TankParams,
    load_history,
    run_backtest,
)
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.evaluate import _RESULTS_CSV, TEST_END, TEST_START

DEFAULT_MODEL_PATH = pathlib.Path("data/models/logreg_calibrated.joblib")
TAU_SWEEP: list[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def aggregate_backtest(
    history: PriceHistory,
    strategy: object,
    station_codes: Sequence[int],
    start_date: str,
    end_date: str,
    tank: TankParams,
) -> dict:
    """Run strategy on all stations; pool spend + litres → single aggregate CPL.

    Stations with no price data are silently skipped. Returns NaN CPL only
    when every station has no data or no fill events occur.
    """
    total_spend = 0.0
    total_litres = 0.0
    fill_events = 0
    for code in station_codes:
        result = run_backtest(history, strategy, code, start_date, end_date, tank)
        if not math.isnan(result.realised_cpl):
            total_spend += result.total_spend_cents
            total_litres += result.total_litres
            fill_events += result.fill_events
    cpl = total_spend / total_litres if total_litres > 0 else float("nan")
    return {
        "total_spend_cents": total_spend,
        "total_litres": total_litres,
        "fill_events": fill_events,
        "cpl": cpl,
    }


def run_tau_sweep(
    history: PriceHistory,
    station_codes: Sequence[int],
    model_path: pathlib.Path,
    taus: list[float],
    start_date: str,
    end_date: str,
    tank: TankParams,
) -> list[dict]:
    """Sweep τ values on the test window; return metric dicts in τ order."""
    rows = []
    for tau in taus:
        strategy = ModelStrategy(model_path=model_path, threshold=tau)
        agg = aggregate_backtest(history, strategy, station_codes, start_date, end_date, tank)
        rows.append({"tau": tau, **agg})
    return rows


def pick_spend_optimal_tau(sweep_rows: list[dict]) -> float:
    """Return the τ with the lowest aggregate CPL across all sweep rows."""
    valid = [r for r in sweep_rows if not math.isnan(r["cpl"])]
    if not valid:
        raise ValueError("No valid CPL values in sweep rows.")
    return min(valid, key=lambda r: r["cpl"])["tau"]


def patch_results_csv(
    csv_path: pathlib.Path,
    always_buy_cpl: float,
    phase2_cpl: float,
    phase2_savings_pct: float,
) -> tuple[bool, bool]:
    """Patch realised-spend columns for the baseline and Phase 2 rows.

    Matches:
    - baseline:  name == 'marginal_rate_baseline' (first occurrence)
    - Phase 2:   name == 'logreg_cycle_features' AND 'tau=0.40;' in notes (first match)

    Write is atomic: written to a temp file, then renamed over csv_path.
    Returns (patched_baseline, patched_phase2).
    """
    rows = []
    patched_baseline = False
    patched_phase2 = False

    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row["name"] == "marginal_rate_baseline" and not patched_baseline:
                row["realised_spend_cpl"] = f"{always_buy_cpl:.2f}"
                row["realised_savings_vs_always_buy_pct"] = "0.00"
                patched_baseline = True
            elif (
                row["name"] == "logreg_cycle_features"
                and "tau=0.40;" in row.get("notes", "")
                and not patched_phase2
            ):
                row["realised_spend_cpl"] = f"{phase2_cpl:.2f}"
                row["realised_savings_vs_always_buy_pct"] = f"{phase2_savings_pct:.2f}"
                patched_phase2 = True
            rows.append(row)

    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=csv_path.parent, suffix=".tmp", delete=False
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = pathlib.Path(fh.name)

    tmp_path.replace(csv_path)
    return patched_baseline, patched_phase2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("backtest_phase2")
@click.option(
    "--model-path", "model_path",
    type=pathlib.Path,
    default=str(DEFAULT_MODEL_PATH),
    show_default=True,
    help="Path to calibrated logreg joblib pipeline.",
)
@click.option(
    "--db", "db_path",
    default=str(db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option(
    "--no-patch", "no_patch",
    is_flag=True,
    default=False,
    help="Print sweep results only; do not patch experiments/results.csv.",
)
def main(model_path: pathlib.Path, db_path: str, no_patch: bool) -> None:
    """Sweep τ ∈ [0.30, 0.55] on the test window and re-validate Phase 2 τ=0.40.

    Runs the calibrated logreg model through the backtest engine at six
    thresholds, compares realised CPL against always-buy, and identifies
    the spend-optimal τ. Unless --no-patch is set, patches
    experiments/results.csv with realised-spend columns for the
    marginal_rate_baseline row and the Phase 2 (τ=0.40) row.
    """
    model_path = pathlib.Path(model_path)
    if not model_path.exists():
        raise click.ClickException(
            f"Model not found: {model_path}. "
            "Run 'uv run python -m fuel_signal.calibrate' first."
        )
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. "
            "Run 'uv run python -m fuel_signal.db' first."
        )

    conn = db.open_db(path)
    try:
        station_codes = list(PREFERRED_STATIONS.keys())
        history = load_history(conn, station_codes)
    finally:
        conn.close()

    tank = TankParams()
    start, end = TEST_START, TEST_END

    click.echo(f"Test window: {start} → {end}")
    click.echo(f"Stations:    {list(PREFERRED_STATIONS.values())}")
    click.echo()

    # Always-buy baseline
    baseline_agg = aggregate_backtest(
        history, AlwaysBuyStrategy(), station_codes, start, end, tank
    )
    always_buy_cpl = baseline_agg["cpl"]
    click.echo(
        f"Always-buy  CPL: {always_buy_cpl:.2f} c/L"
        f"  ({baseline_agg['fill_events']} fills, {baseline_agg['total_litres']:.0f} L)"
    )
    click.echo()

    # τ sweep
    click.echo("τ sweep on test window:")
    click.echo(
        f"  {'τ':>5}  {'CPL (c/L)':>10}  {'vs always-buy':>14}  {'Fills':>6}  {'Litres':>8}"
    )
    click.echo("  " + "-" * 54)

    sweep_rows = run_tau_sweep(
        history, station_codes, model_path, TAU_SWEEP, start, end, tank
    )

    for row in sweep_rows:
        cpl = row["cpl"]
        marker = " ← phase 2" if row["tau"] == 0.40 else ""
        if math.isnan(cpl):
            click.echo(
                f"  {row['tau']:5.2f}  {'n/a':>10}  {'n/a':>14}"
                f"  {row['fill_events']:>6}  {'n/a':>8}{marker}"
            )
        else:
            savings_pct = (
                (always_buy_cpl - cpl) / always_buy_cpl * 100
                if always_buy_cpl > 0 else float("nan")
            )
            click.echo(
                f"  {row['tau']:5.2f}  {cpl:10.2f}  {savings_pct:+13.2f}%"
                f"  {row['fill_events']:>6}  {row['total_litres']:>8.0f} L{marker}"
            )

    click.echo()

    spend_opt_tau = pick_spend_optimal_tau(sweep_rows)
    phase2_row = next((r for r in sweep_rows if r["tau"] == 0.40), None)

    click.echo(f"Spend-optimal τ: {spend_opt_tau:.2f}")
    click.echo("Phase 2 τ      : 0.40")

    if spend_opt_tau == 0.40:
        click.echo("\nDecision: τ=0.40 confirmed — synthetic proxy and realised spend agree.")
    else:
        best_cpl = min(r["cpl"] for r in sweep_rows if not math.isnan(r["cpl"]))
        phase2_cpl_val = (
            phase2_row["cpl"] if phase2_row and not math.isnan(phase2_row["cpl"])
            else float("nan")
        )
        gap_str = (
            f"{phase2_cpl_val - best_cpl:.2f} c/L"
            if not math.isnan(phase2_cpl_val) else "n/a"
        )
        click.echo(
            f"\nDecision: spend-optimal τ={spend_opt_tau:.2f} differs from τ=0.40 "
            f"by {gap_str}. Review score_phase2.py if gap is material."
        )

    if no_patch:
        click.echo("\n--no-patch: experiments/results.csv not modified.")
        return

    if phase2_row is None or math.isnan(phase2_row["cpl"]):
        click.echo("\nWARNING: No valid Phase 2 (τ=0.40) result — results.csv not patched.")
        return
    if math.isnan(always_buy_cpl):
        click.echo("\nWARNING: Always-buy CPL is NaN — results.csv not patched.")
        return

    phase2_cpl = phase2_row["cpl"]
    phase2_savings_pct = (
        (always_buy_cpl - phase2_cpl) / always_buy_cpl * 100
    ) if always_buy_cpl > 0 else float("nan")

    if not _RESULTS_CSV.exists():
        click.echo(f"\nWARNING: {_RESULTS_CSV} not found — skipping patch.")
        return

    patched_baseline, patched_phase2 = patch_results_csv(
        _RESULTS_CSV, always_buy_cpl, phase2_cpl, phase2_savings_pct
    )
    if patched_baseline and patched_phase2:
        click.echo("\nPatched experiments/results.csv:")
        click.echo(
            f"  marginal_rate_baseline  → CPL {always_buy_cpl:.2f} c/L (0.00% vs always-buy)"
        )
        click.echo(
            f"  logreg_cycle_features   → CPL {phase2_cpl:.2f} c/L"
            f" ({phase2_savings_pct:+.2f}% vs always-buy)"
        )
    else:
        if not patched_baseline:
            click.echo("\nWARNING: 'marginal_rate_baseline' row not found — not patched.")
        if not patched_phase2:
            click.echo("\nWARNING: Phase 2 logreg row (tau=0.40) not found — not patched.")


if __name__ == "__main__":
    main()
