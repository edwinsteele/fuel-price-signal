"""Diagnostic: empirical FP cost distribution from the features dataset.

For each label=0 row, the actual cost of a false-positive BUY decision is:

    damage = today_price_cents - future_min_cents

Positive damage: price fell within the horizon — you overpaid by that many cents.
Negative/near-zero damage: price rose or was stable — buying wasn't actually harmful.

The label=0 population is structurally bimodal:

  Cluster A (only the percentile gate failed):
    future_min >= today_price - threshold_cents
    The price was "too expensive" relative to history but no meaningfully cheaper
    price followed. Damage <= threshold_cents — if you'd bought here, the actual
    cost was small or zero.

  Cluster B (a better deal was coming):
    future_min < today_price - threshold_cents
    A cheaper price arrived within the horizon. Damage > threshold_cents.

The FP penalty in score_phase2.py (currently 1.5c) sits between the two clusters.
This diagnostic shows whether that number is grounded in the actual damage data.

Usage
-----
    uv run python -m fuel_signal.fp_cost
    uv run python -m fuel_signal.fp_cost --features-csv data/features.csv --plot data/fp.png
"""

from __future__ import annotations

import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_LABEL_THRESHOLD_CENTS: float = 3.0
_CURRENT_FP_PENALTY: float = 1.5
CLUSTER_A_LABEL: str = "A: gate only (no drop)"
CLUSTER_B_LABEL: str = "B: drop came (true FP)"

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_PLOT_PATH = pathlib.Path("data/fp_cost_distribution.png")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_damage(
    df: pd.DataFrame,
    threshold_cents: float = _LABEL_THRESHOLD_CENTS,
) -> pd.DataFrame:
    """Filter to label=0 rows; add damage and cluster columns.

    damage = today_price_cents - future_min_cents
    cluster: A if damage <= threshold_cents (condition 1 passed, gate failed only)
             B if damage >  threshold_cents (condition 1 failed: a better deal came)
    """
    fp = df[df["label"] == 0].copy()
    fp["damage"] = fp["today_price_cents"] - fp["future_min_cents"]
    fp["cluster"] = np.where(
        fp["damage"] > threshold_cents,
        CLUSTER_B_LABEL,
        CLUSTER_A_LABEL,
    )
    return fp


def _stats(series: pd.Series) -> dict:
    if series.empty:
        return {k: float("nan") for k in ("n", "mean", "median", "p25", "p75", "p90")}
    return {
        "n": len(series),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_summary(
    fp: pd.DataFrame,
    threshold_cents: float = _LABEL_THRESHOLD_CENTS,
) -> str:
    """Return a formatted summary table as a string."""
    ca = fp.loc[fp["cluster"] == CLUSTER_A_LABEL, "damage"]
    cb = fp.loc[fp["cluster"] == CLUSTER_B_LABEL, "damage"]
    all_s = _stats(fp["damage"])
    a_s = _stats(ca)
    b_s = _stats(cb)

    w = 13

    def _fmt_n(d: dict) -> str:
        return f"{int(d['n']):>{w},}" if not np.isnan(d["n"]) else f"{'—':>{w}}"

    def _fmt_c(d: dict, key: str) -> str:
        v = d[key]
        return f"{v:>{w - 1}.2f}c" if not np.isnan(v) else f"{'—':>{w}}"

    rows = [
        f"FP cost analysis — {len(fp):,} label=0 rows",
        f"  Label threshold: {threshold_cents:.1f}c  |  Current FP penalty: {_CURRENT_FP_PENALTY:.1f}c",
        "",
        f"  {'':22s}{'All label=0':>{w}}{'Cluster A':>{w}}{'Cluster B':>{w}}",
        f"  {'':22s}{'':>{w}}{'(gate only)':>{w}}{'(drop came)':>{w}}",
        "  " + "-" * (22 + w * 3),
        f"  {'rows':<22s}{_fmt_n(all_s)}{_fmt_n(a_s)}{_fmt_n(b_s)}",
        f"  {'mean damage':<22s}{_fmt_c(all_s, 'mean')}{_fmt_c(a_s, 'mean')}{_fmt_c(b_s, 'mean')}",
        f"  {'median damage':<22s}{_fmt_c(all_s, 'median')}{_fmt_c(a_s, 'median')}{_fmt_c(b_s, 'median')}",
        f"  {'p25 damage':<22s}{_fmt_c(all_s, 'p25')}{_fmt_c(a_s, 'p25')}{_fmt_c(b_s, 'p25')}",
        f"  {'p75 damage':<22s}{_fmt_c(all_s, 'p75')}{_fmt_c(a_s, 'p75')}{_fmt_c(b_s, 'p75')}",
        f"  {'p90 damage':<22s}{_fmt_c(all_s, 'p90')}{_fmt_c(a_s, 'p90')}{_fmt_c(b_s, 'p90')}",
        "  " + "-" * (22 + w * 3),
    ]

    med_b = b_s["median"]
    if not np.isnan(med_b):
        differs = abs(med_b - _CURRENT_FP_PENALTY) > 0.5
        note = f"  → Suggested FP penalty: {med_b:.2f}c (median cluster B)"
        if differs:
            note += f"  ** differs from current {_CURRENT_FP_PENALTY:.1f}c by >{0.5:.1f}c **"
        rows.append(note)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_fp_distribution(
    fp: pd.DataFrame,
    threshold_cents: float = _LABEL_THRESHOLD_CENTS,
    out_path: pathlib.Path = DEFAULT_PLOT_PATH,
) -> None:
    """Save histogram of damage stratified by cluster to out_path."""
    if fp.empty:
        return

    ca = fp.loc[fp["cluster"] == CLUSTER_A_LABEL, "damage"]
    cb = fp.loc[fp["cluster"] == CLUSTER_B_LABEL, "damage"]

    lo = float(fp["damage"].quantile(0.005))
    hi = float(fp["damage"].quantile(0.995))
    if np.isnan(lo) or np.isnan(hi) or lo >= hi:
        lo = float(fp["damage"].min()) - 0.5
        hi = float(fp["damage"].max()) + 0.5
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(ca, bins=bins, alpha=0.65, color="steelblue",
            label=f"Cluster A — gate only ({len(ca):,} rows)")
    ax.hist(cb, bins=bins, alpha=0.65, color="coral",
            label=f"Cluster B — drop came ({len(cb):,} rows)")

    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.2, label="0c (break-even)")
    ax.axvline(_CURRENT_FP_PENALTY, color="red", linestyle="--", linewidth=1.5,
               label=f"Current FP penalty ({_CURRENT_FP_PENALTY:.1f}c)")
    ax.axvline(threshold_cents, color="purple", linestyle="--", linewidth=1.2,
               label=f"Label threshold ({threshold_cents:.1f}c)")

    if not cb.empty:
        med_b = float(cb.median())
        ax.axvline(med_b, color="darkorange", linestyle="-", linewidth=1.8,
                   label=f"Median cluster B ({med_b:.2f}c)")

    ax.set_xlabel("today_price − future_min  (cents)")
    ax.set_ylabel("Row count")
    ax.set_title(
        "FP damage distribution — label=0 rows\n"
        "(cost of a wrong BUY decision; positive = price fell, negative = price rose)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("fp-cost")
@click.option(
    "--features-csv", "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Features CSV produced by 'python -m fuel_signal.features'.",
)
@click.option(
    "--plot", "plot_path",
    default=str(DEFAULT_PLOT_PATH),
    show_default=True,
    help="Output PNG path for the damage distribution plot.",
)
@click.option(
    "--threshold", "threshold_cents",
    default=_LABEL_THRESHOLD_CENTS,
    show_default=True,
    type=float,
    help="Label threshold in cents — match the value used in labels.py (default 3.0).",
)
def main(features_csv: str, plot_path: str, threshold_cents: float) -> None:
    """Empirical FP cost diagnostic: distribution of wrong-BUY damage from label=0 rows.

    Shows how costly a false-positive BUY actually is, stratified into rows where
    no better deal came (cluster A, low damage) and rows where one did (cluster B,
    high damage). Use the cluster B median to ground the FP penalty in score_phase2.py.
    """
    src = pathlib.Path(features_csv)
    if not src.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )

    df = pd.read_csv(src)
    required = {"today_price_cents", "future_min_cents", "label"}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(
            f"Features CSV is missing columns: {sorted(missing)}. "
            "Re-run 'uv run python -m fuel_signal.features' to regenerate."
        )

    fp = compute_damage(df, threshold_cents)
    click.echo(format_summary(fp, threshold_cents))
    click.echo()

    out = pathlib.Path(plot_path)
    plot_fp_distribution(fp, threshold_cents, out)
    click.echo(f"Wrote plot to {out}")


if __name__ == "__main__":
    main()
