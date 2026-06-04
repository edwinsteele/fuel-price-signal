"""Walk-forward cross-validation report.

**Paired mode (CLI):** loads two joblib model artifacts, re-trains both on every
walk-forward fold, and compares per-fold val logloss.  This is the promoted form
of the one-off ``experiments/cv_compare_*/run_cv.py`` scripts.

**Drop-feature mode (CLI):** pass ``--drop-feature <col>`` (repeatable) instead
of ``--baseline``.  The baseline becomes ``--model`` itself; the "model" becomes
a clone of ``--model``'s pipeline with the named column(s) removed before fitting
each fold.  No separate baseline artifact is needed.

**Library mode:** ``run_cv()`` runs a single-model logreg walk-forward CV and is
kept for programmatic use by tests and notebooks.

Usage::

    # Two-artifact mode
    uv run python -m fuel_signal.cv_report \\
      --model data/models/lgbm.joblib \\
      --baseline data/models/lgbm_phase3c.joblib \\
      --features data/features.csv \\
      --seed 42 \\
      --output experiments/cv_phase4/results.csv

    # Drop-feature mode
    uv run python -m fuel_signal.cv_report \\
      --model data/models/lgbm.joblib \\
      --drop-feature station_minus_last_max_cents \\
      --features data/features.csv \\
      --seed 42 \\
      --output experiments/<dir>/results.csv
"""

from __future__ import annotations

import pathlib

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from fuel_signal import evaluate as _ev
from fuel_signal.features import DEFAULT_FEATURES_CSV, FEATURE_COLUMNS, load_features
from fuel_signal.train_logreg import build_pipeline as _build_logreg


def run_cv(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    train_min_days: int = 1825,
    val_days: int = 90,
    step_days: int = 90,
) -> list[dict]:
    """Single-model walk-forward CV using the logreg baseline.

    Kept for programmatic use; the module CLI runs the paired comparison.
    Folds whose val window falls in a data gap are skipped silently.
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

        pipeline = _build_logreg()
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


def _set_random_state(estimator: object, seed: int) -> None:
    """Set random_state on any parameter named random_state or *__random_state.

    Works for bare estimators (e.g. LGBMClassifier) and sklearn Pipelines where
    the param is step-qualified (e.g. logreg__random_state).
    """
    updates = {
        name: seed
        for name in estimator.get_params(deep=True)  # type: ignore[union-attr]
        if name == "random_state" or name.endswith("__random_state")
    }
    if updates:
        estimator.set_params(**updates)  # type: ignore[union-attr]


def run_paired_cv(
    df: pd.DataFrame,
    model_path: pathlib.Path,
    baseline_path: pathlib.Path | None = None,
    *,
    drop_features: list[str] | None = None,
    seed: int = 42,
    train_min_days: int = 1825,
    val_days: int = 90,
    step_days: int = 90,
) -> list[dict]:
    """Paired walk-forward CV: re-train both model and baseline on each fold.

    Two modes:

    *Two-artifact mode* (``baseline_path`` supplied, ``drop_features`` omitted):
    Each joblib artifact must be a dict with keys ``pipeline`` (sklearn-compatible
    estimator) and ``feature_columns`` (list[str]).

    *Drop-feature mode* (``drop_features`` supplied, ``baseline_path`` omitted):
    The baseline is ``model_path`` with its full feature set; the "model" is a
    clone of the same pipeline with ``drop_features`` removed from its
    ``feature_columns`` before fitting each fold.

    Both are cloned per fold so only hyperparameters carry over — no
    training-set contamination across splits.

    Returns one result dict per non-empty fold with keys:
        fold_idx, train_start, train_end, val_start, val_end,
        n_val, baseline_logloss, model_logloss, delta
    where ``delta = model_logloss − baseline_logloss`` (negative means model wins).
    """
    if baseline_path is not None and drop_features is not None:
        raise ValueError("Provide baseline_path or drop_features, not both.")

    model_obj = joblib.load(model_path)

    if drop_features is not None:
        drop_set = set(drop_features)
        baseline_features: list[str] = model_obj["feature_columns"]
        model_features: list[str] = [f for f in baseline_features if f not in drop_set]
        baseline_pipeline = model_obj["pipeline"]
        model_pipeline = model_obj["pipeline"]
    else:
        baseline_obj = joblib.load(baseline_path)  # type: ignore[arg-type]
        model_features = model_obj["feature_columns"]
        baseline_features = baseline_obj["feature_columns"]
        baseline_pipeline = baseline_obj["pipeline"]
        model_pipeline = model_obj["pipeline"]

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

        y_val = val_df["label"].to_numpy(dtype=int)

        m = clone(model_pipeline)
        _set_random_state(m, seed)
        m.fit(
            train_df[model_features].to_numpy(dtype=float),
            train_df["label"].to_numpy(dtype=int),
        )
        p_model = m.predict_proba(val_df[model_features].to_numpy(dtype=float))[:, 1]
        model_logloss = _ev.log_loss(y_val, p_model)

        b = clone(baseline_pipeline)
        _set_random_state(b, seed)
        b.fit(
            train_df[baseline_features].to_numpy(dtype=float),
            train_df["label"].to_numpy(dtype=int),
        )
        p_baseline = b.predict_proba(val_df[baseline_features].to_numpy(dtype=float))[:, 1]
        baseline_logloss = _ev.log_loss(y_val, p_baseline)

        val_dates = pd.to_datetime(val_df["price_date"])
        train_dates = pd.to_datetime(train_df["price_date"])
        results.append({
            "fold_idx": i + 1,
            "train_start": train_dates.min().strftime("%Y-%m-%d"),
            "train_end": train_dates.max().strftime("%Y-%m-%d"),
            "val_start": val_dates.min().strftime("%Y-%m-%d"),
            "val_end": val_dates.max().strftime("%Y-%m-%d"),
            "n_val": len(val_df),
            "baseline_logloss": baseline_logloss,
            "model_logloss": model_logloss,
            "delta": model_logloss - baseline_logloss,
        })

    return results


def _format_paired_fold(r: dict) -> str:
    return (
        f"fold {r['fold_idx']:>3}  "
        f"val {r['val_start']}→{r['val_end']}  "
        f"n={r['n_val']:>5,}  "
        f"baseline={r['baseline_logloss']:.4f}  "
        f"model={r['model_logloss']:.4f}  "
        f"Δ={r['delta']:+.4f}"
    )


def _format_paired_summary(results: list[dict]) -> str:
    deltas = np.array([r["delta"] for r in results])
    n_wins = int((deltas < 0).sum())
    n_folds = len(results)
    regressions = [r for r in results if r["delta"] > 0.05]
    lines = [
        "─" * 72,
        (
            f"folds: {n_folds}  wins: {n_wins}/{n_folds}  "
            f"median Δ={np.median(deltas):+.4f}  mean Δ={deltas.mean():+.4f}"
        ),
    ]
    if regressions:
        names = ", ".join(
            f"fold {r['fold_idx']} ({r['val_start']}→{r['val_end']}, Δ={r['delta']:+.4f})"
            for r in regressions
        )
        lines.append(f"regressions (Δ>+0.05): {names}")
    return "\n".join(lines)


@click.command("cv_report")
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Joblib artifact for the model to evaluate.",
)
@click.option(
    "--baseline",
    "baseline_path",
    default=None,
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Joblib artifact for the baseline. Mutually exclusive with --drop-feature.",
)
@click.option(
    "--drop-feature",
    "drop_features",
    multiple=True,
    metavar="COL",
    help=(
        "Drop this feature column from --model before each fold fit. "
        "Repeatable. Mutually exclusive with --baseline."
    ),
)
@click.option(
    "--features",
    "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Feature rows CSV from `python -m fuel_signal.features`.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed applied when re-training each fold.",
)
@click.option(
    "--output",
    "output_csv",
    default=None,
    help="Path to write per-fold results CSV (optional).",
)
@click.option(
    "--train-min-days",
    "train_min_days",
    type=click.IntRange(min=1),
    default=1825,
    show_default=True,
    help="Minimum training window size in days.",
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
def main(
    model_path: pathlib.Path,
    baseline_path: pathlib.Path | None,
    drop_features: tuple[str, ...],
    features_csv: str,
    seed: int,
    output_csv: str | None,
    train_min_days: int,
    val_days: int,
    step_days: int,
) -> None:
    """Paired walk-forward CV: compare --model vs --baseline (or --drop-feature) across pre-test folds."""
    if baseline_path is not None and drop_features:
        raise click.UsageError("--baseline and --drop-feature are mutually exclusive.")
    if baseline_path is None and not drop_features:
        raise click.UsageError("Provide either --baseline or at least one --drop-feature.")

    features_path = pathlib.Path(features_csv)
    if not features_path.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )

    df = load_features(features_path)
    missing = [c for c in ("label", "price_date") if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"Features CSV is missing required columns: {missing}. "
            "Re-run 'uv run python -m fuel_signal.features' to regenerate."
        )

    if drop_features:
        model_obj = joblib.load(model_path)
        valid = set(model_obj["feature_columns"])
        unknown = [f for f in drop_features if f not in valid]
        if unknown:
            raise click.ClickException(
                f"Unknown --drop-feature column(s): {unknown}. "
                f"Valid columns: {sorted(valid)}"
            )
        results = run_paired_cv(
            df,
            model_path,
            drop_features=list(drop_features),
            seed=seed,
            train_min_days=train_min_days,
            val_days=val_days,
            step_days=step_days,
        )
    else:
        results = run_paired_cv(
            df,
            model_path,
            baseline_path,
            seed=seed,
            train_min_days=train_min_days,
            val_days=val_days,
            step_days=step_days,
        )

    if not results:
        raise click.ClickException(
            "No folds produced. Try reducing --train-min-days or extending the date range."
        )

    for r in results:
        click.echo(_format_paired_fold(r))
    click.echo(_format_paired_summary(results))

    if output_csv:
        out = pathlib.Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out, index=False)
        click.echo(f"\nSaved {out}")


if __name__ == "__main__":
    main()
