"""Walk-forward cross-validation report for the ML price-movement model.

Trains the logreg baseline on each fold produced by walk_forward_folds() and
prints per-fold val logloss + BUY rate, then a mean ± std summary. Use this
to assess whether any single val window (e.g. the canonical Phase 2 window) is
an outlier before running Optuna.

Usage::

    uv run python -m fuel_signal.cv_report
    uv run python -m fuel_signal.cv_report --train-min-days 1825 --val-days 90 --step-days 90
"""

from __future__ import annotations

import pathlib

import click
import numpy as np
import pandas as pd

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.train_logreg import build_pipeline

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")


def run_cv(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    train_min_days: int = 1825,
    val_days: int = 90,
    step_days: int = 90,
) -> list[dict]:
    """Run walk-forward CV; return one result dict per fold with data.

    Folds whose val window falls entirely in a data gap (val_df is empty)
    are skipped silently — they carry no signal and would raise in log_loss.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    results = []
    for i, (train_df, val_df) in enumerate(
        _ev.walk_forward_folds(
            df,
            train_min_days=train_min_days,
            val_days=val_days,
            step_days=step_days,
        )
    ):
        if val_df.empty:
            continue

        X_train = train_df[feature_columns].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy(dtype=int)
        X_val = val_df[feature_columns].to_numpy(dtype=float)
        y_val = val_df["label"].to_numpy(dtype=int)

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        p_val = pipeline.predict_proba(X_val)[:, 1]

        prior = _ev.baseline_prior(train_df)
        val_logloss = _ev.log_loss(y_val, p_val)
        baseline_logloss = _ev.log_loss(y_val, np.full(len(y_val), prior))

        train_dates = pd.to_datetime(train_df["price_date"])
        val_dates = pd.to_datetime(val_df["price_date"])
        results.append({
            "fold": i + 1,
            "train_start": train_dates.min().strftime("%Y-%m-%d"),
            "train_end": train_dates.max().strftime("%Y-%m-%d"),
            "val_start": val_dates.min().strftime("%Y-%m-%d"),
            "val_end": val_dates.max().strftime("%Y-%m-%d"),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "val_buy_rate": float(y_val.mean()),
            "val_logloss": val_logloss,
            "baseline_logloss": baseline_logloss,
        })

    return results


def _format_fold(r: dict) -> str:
    delta = r["val_logloss"] - r["baseline_logloss"]
    return (
        f"fold {r['fold']:>3}  "
        f"train {r['train_start']}→{r['train_end']}  "
        f"val {r['val_start']}→{r['val_end']}  "
        f"train={r['train_rows']:>7,}  "
        f"val={r['val_rows']:>5,}  "
        f"buy_rate={r['val_buy_rate']:.3f}  "
        f"logloss={r['val_logloss']:.4f}  "
        f"baseline={r['baseline_logloss']:.4f}  "
        f"Δ={delta:+.4f}"
    )


def _format_summary(results: list[dict]) -> str:
    losses = np.array([r["val_logloss"] for r in results])
    baselines = np.array([r["baseline_logloss"] for r in results])
    return (
        f"{'─' * 72}\n"
        f"folds: {len(results)}  "
        f"logloss {losses.mean():.4f} ± {losses.std():.4f}  "
        f"(baseline {baselines.mean():.4f} ± {baselines.std():.4f})"
    )


@click.command("cv_report")
@click.option(
    "--features-csv",
    "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Path to feature rows CSV produced by `python -m fuel_signal.features`.",
)
@click.option(
    "--train-min-days",
    "train_min_days",
    type=click.IntRange(min=1),
    default=1825,
    show_default=True,
    help="Minimum training window size in days (default: 5 years).",
)
@click.option(
    "--val-days",
    "val_days",
    type=click.IntRange(min=1),
    default=90,
    show_default=True,
    help="Validation window length in days per fold.",
)
@click.option(
    "--step-days",
    "step_days",
    type=click.IntRange(min=1),
    default=90,
    show_default=True,
    help="Step size in days between consecutive folds.",
)
def main(features_csv: str, train_min_days: int, val_days: int, step_days: int) -> None:
    """Walk-forward CV report: per-fold logloss + BUY rate over the pre-test window."""
    features_path = pathlib.Path(features_csv)
    if not features_path.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )

    df = pd.read_csv(features_path)
    missing = [c for c in FEATURE_COLUMNS + ["label", "price_date"] if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"Features CSV is missing required columns: {missing}. "
            "Re-run 'uv run python -m fuel_signal.features' to regenerate."
        )

    results = run_cv(
        df,
        train_min_days=train_min_days,
        val_days=val_days,
        step_days=step_days,
    )

    if not results:
        raise click.ClickException(
            "No folds produced. Try reducing --train-min-days or extending the date range."
        )

    for r in results:
        click.echo(_format_fold(r))

    click.echo(_format_summary(results))


if __name__ == "__main__":
    main()
