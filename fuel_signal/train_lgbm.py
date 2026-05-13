"""LightGBM baseline for the ML price-movement model.

Phase 3a.1 — apples-to-apples comparison vs the Phase 2 logreg. Uses the
exact same 10 features (FEATURE_COLUMNS) with vanilla LightGBM defaults.
No tuning, no early stopping, no new features. Issue #73.

## Pipeline

    LGBMClassifier(random_state=42, verbose=-1)

No StandardScaler: gradient-boosted trees are scale-invariant, so it would
be a no-op. Keeping it out reduces noise when comparing coefficients.

## What this module evaluates

Train on the canonical train split, score on **val only**. Test is reserved
for score_phase2.py once calibration + threshold is locked. We do not write
to ``experiments/results.csv`` from here — that file is for test scores.

## Reliability plot

Same quantile-bin format as train_logreg.py — see that module for semantics.
"""

from __future__ import annotations

import pathlib

import click
import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe; required for CLI use without a display
import matplotlib.pyplot as plt  # noqa: E402, F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402
from fuel_signal.features import FEATURE_COLUMNS  # noqa: E402
from fuel_signal.train_logreg import save_reliability_plot  # noqa: E402

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_MODEL_OUT = pathlib.Path("data/models/lgbm.joblib")
DEFAULT_RELIABILITY_PNG = pathlib.Path("experiments/reliability_lgbm_val.png")


def build_pipeline() -> LGBMClassifier:
    """Vanilla LightGBM with deterministic seed and silenced console output."""
    return LGBMClassifier(random_state=42, verbose=-1)


def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict:
    """Fit LightGBM on train; score on val.

    Returns a dict with the fitted model, train/val sizes, class balances,
    val predictions+labels, val log-loss, val brier, and baseline comparisons.
    Does not touch the test split.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    train, val, _test = _ev.split(df)
    if train.empty:
        raise ValueError("train_and_evaluate(): train split is empty.")
    if val.empty:
        raise ValueError("train_and_evaluate(): val split is empty.")

    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    X_val = val[feature_columns].to_numpy(dtype=float)
    y_val = val["label"].to_numpy(dtype=int)

    model = build_pipeline()
    model.fit(X_train, y_train)

    # predict_proba returns shape (n, 2); column 1 is P(label=1).
    p_val = model.predict_proba(X_val)[:, 1]

    val_logloss = _ev.log_loss(y_val, p_val)
    val_brier = _ev.brier(y_val, p_val)

    # Baseline = constant predictor at train marginal rate.
    p_baseline = _ev.baseline_prior(train)
    baseline_pred_val = np.full(len(y_val), p_baseline)
    baseline_logloss = _ev.log_loss(y_val, baseline_pred_val)
    baseline_brier = _ev.brier(y_val, baseline_pred_val)

    return {
        "pipeline": model,
        "feature_columns": list(feature_columns),
        "train_size": int(len(train)),
        "val_size": int(len(val)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "val_logloss": float(val_logloss),
        "val_brier": float(val_brier),
        "baseline_prior": float(p_baseline),
        "baseline_val_logloss": float(baseline_logloss),
        "baseline_val_brier": float(baseline_brier),
        "y_val": y_val,
        "p_val": p_val,
    }


def _format_results(result: dict) -> str:
    delta_ll = result["val_logloss"] - result["baseline_val_logloss"]
    delta_br = result["val_brier"] - result["baseline_val_brier"]
    lines = [
        "LightGBM — val results",
        f"  train rows            : {result['train_size']:>8,}  (pos rate {result['train_positive_rate']:.3f})",
        f"  val rows              : {result['val_size']:>8,}  (pos rate {result['val_positive_rate']:.3f})",
        f"  baseline prior (train): {result['baseline_prior']:.4f}",
        "",
        f"  val log-loss          : {result['val_logloss']:.4f}  (baseline {result['baseline_val_logloss']:.4f},"
        f"  Δ {delta_ll:+.4f})",
        f"  val brier             : {result['val_brier']:.4f}  (baseline {result['baseline_val_brier']:.4f},"
        f"  Δ {delta_br:+.4f})",
    ]
    return "\n".join(lines)


@click.command("train_lgbm")
@click.option(
    "--features-csv",
    "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Path to feature rows CSV produced by `python -m fuel_signal.features`.",
)
@click.option(
    "--model-out",
    "model_out",
    default=str(DEFAULT_MODEL_OUT),
    show_default=True,
    help="Where to save the fitted LightGBM model (joblib).",
)
@click.option(
    "--reliability-out",
    "reliability_out",
    default=str(DEFAULT_RELIABILITY_PNG),
    show_default=True,
    help="Where to save the val reliability plot.",
)
def main(features_csv: str, model_out: str, reliability_out: str) -> None:
    """Train a LightGBM baseline and score it on the val split.

    Same feature set as Phase 2 logreg — no new features, no hyperparameter
    tuning, random_state=42. Test is intentionally left untouched. This command
    does not append to experiments/results.csv.
    """
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

    result = train_and_evaluate(df)

    click.echo(_format_results(result))

    model_path = pathlib.Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": result["pipeline"], "feature_columns": result["feature_columns"]},
        model_path,
    )
    click.echo(f"\nSaved fitted model to {model_path}")

    reliability_path = pathlib.Path(reliability_out)
    save_reliability_plot(
        result["y_val"],
        result["p_val"],
        reliability_path,
        title="Reliability — LightGBM, val split",
        model_label="lgbm",
    )
    click.echo(f"Saved reliability plot to  {reliability_path}")

    if result["val_logloss"] >= result["baseline_val_logloss"]:
        click.echo(
            "\nWARNING: val log-loss did not beat the baseline. "
            "Investigate before merging — see issue #73 acceptance criteria.",
            err=True,
        )


if __name__ == "__main__":
    main()
