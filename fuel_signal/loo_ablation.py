"""Leave-one-out (LOO) ablation for LightGBM feature contribution checks.

Fits LightGBM at multiple seeds with one or more features dropped, then compares
mean ± std val logloss against the full-feature baseline on the same features.csv.

Usage::

    uv run python -m fuel_signal.loo_ablation \\
        --features-csv data/features.csv \\
        --drop station_minus_lga_mean_cents \\
        --seeds 1,7,42,99,2024
"""

from __future__ import annotations

import pathlib

import click
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")


def _fit_and_score(df: pd.DataFrame, feature_columns: list[str], seed: int) -> float:
    """Fit LightGBM on train split, return val log-loss."""
    train, val, _test = _ev.split(df)
    model = LGBMClassifier(random_state=seed, verbose=-1)
    model.fit(
        train[feature_columns].to_numpy(dtype=float),
        train["label"].to_numpy(dtype=int),
    )
    p_val = model.predict_proba(val[feature_columns].to_numpy(dtype=float))[:, 1]
    return float(_ev.log_loss(val["label"].to_numpy(dtype=int), p_val))


def run_loo(
    df: pd.DataFrame,
    drop_columns: list[str],
    seeds: list[int],
) -> dict:
    """Run LOO ablation over multiple seeds.

    Returns a dict with baseline stats, LOO stats, Δ, and a one-line verdict.
    Δ = loo_mean − baseline_mean; positive means removing the feature(s) hurt
    the model (i.e. they contributed); negative means removing them helped.
    """
    loo_columns = [c for c in FEATURE_COLUMNS if c not in drop_columns]

    baseline_scores = [_fit_and_score(df, FEATURE_COLUMNS, s) for s in seeds]
    loo_scores = [_fit_and_score(df, loo_columns, s) for s in seeds]

    baseline_mean = float(np.mean(baseline_scores))
    baseline_std = float(np.std(baseline_scores, ddof=1))
    loo_mean = float(np.mean(loo_scores))
    loo_std = float(np.std(loo_scores, ddof=1))
    delta = loo_mean - baseline_mean

    if abs(delta) < baseline_std:
        verdict = "within noise / redundant"
    elif delta > 0:
        verdict = "feature contributes (starved)"
    else:
        verdict = "feature harmful (unexpected)"

    return {
        "drop_columns": drop_columns,
        "seeds": seeds,
        "baseline_scores": baseline_scores,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "loo_scores": loo_scores,
        "loo_mean": loo_mean,
        "loo_std": loo_std,
        "delta": delta,
        "verdict": verdict,
    }


def _format_report(report: dict) -> str:
    dropped = ", ".join(report["drop_columns"])
    seeds_str = ", ".join(str(s) for s in report["seeds"])
    lines = [
        f"LOO ablation — dropping: {dropped}",
        f"Seeds: {seeds_str}  (n={len(report['seeds'])})",
        "",
        f"  baseline  mean ± std : {report['baseline_mean']:.4f} ± {report['baseline_std']:.4f}",
        f"  LOO       mean ± std : {report['loo_mean']:.4f} ± {report['loo_std']:.4f}",
        f"  Δ (LOO − baseline)   : {report['delta']:+.4f}",
        "",
        f"  Verdict: {report['verdict']}",
    ]
    return "\n".join(lines)


@click.command("loo_ablation")
@click.option(
    "--features-csv",
    "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Path to features CSV produced by `python -m fuel_signal.features`.",
)
@click.option(
    "--drop",
    "drop_columns",
    multiple=True,
    help="Feature column to drop. Repeatable (ablate a group by passing multiple times). "
    "Must appear in FEATURE_COLUMNS.",
)
@click.option(
    "--seeds",
    "seeds_str",
    default="1,7,42,99,2024",
    show_default=True,
    help="Comma-separated random seeds.",
)
def main(features_csv: str, drop_columns: tuple[str, ...], seeds_str: str) -> None:
    """Leave-one-out ablation: compare val logloss with/without feature(s) across seeds.

    Fits LightGBM at each seed with all features, then again with --drop column(s)
    removed. Reports mean ± std val logloss for both, the delta, and a verdict.
    """
    if not drop_columns:
        raise click.UsageError("nothing to ablate — supply at least one --drop COL")

    unknown = [c for c in drop_columns if c not in FEATURE_COLUMNS]
    if unknown:
        raise click.ClickException(
            f"Unknown feature column(s): {unknown}. "
            f"Valid columns: {FEATURE_COLUMNS}"
        )

    try:
        seeds = [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
    except ValueError as e:
        raise click.ClickException(f"Invalid --seeds value: {e}") from e

    if not seeds:
        raise click.ClickException("--seeds must contain at least one seed.")

    features_path = pathlib.Path(features_csv)
    if not features_path.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )

    df = pd.read_csv(features_path)
    required = FEATURE_COLUMNS + ["label", "price_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"Features CSV is missing required columns: {missing}. "
            "Re-run 'uv run python -m fuel_signal.features' to regenerate."
        )

    report = run_loo(df, list(drop_columns), seeds)
    click.echo(_format_report(report))


if __name__ == "__main__":
    main()
