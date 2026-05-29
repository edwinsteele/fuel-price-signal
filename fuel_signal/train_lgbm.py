"""LightGBM classifier — numeric-only.

Default feature set is FEATURE_COLUMNS (15 cents/cycle columns, Phase 3c).
Pass --include-lga-features to also train on LGA_FEATURE_COLUMNS
(days_since_trough_entry_<lga>, Phase 4). random_state=42, no hyperparameter
tuning. Test split is intentionally left untouched — reserved for
score_phase2.py once calibration + threshold is locked.

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
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS  # noqa: E402
from fuel_signal.train_logreg import save_reliability_plot  # noqa: E402

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_MODEL_OUT = pathlib.Path("data/models/lgbm.joblib")
DEFAULT_RELIABILITY_PNG = pathlib.Path("experiments/reliability_lgbm_val.png")


def build_pipeline(random_state: int = 42) -> LGBMClassifier:
    """Vanilla LightGBM with deterministic seed and silenced console output."""
    return LGBMClassifier(random_state=random_state, verbose=-1)


def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
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

    model = build_pipeline(random_state=random_state)
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
        "feature_columns": feature_columns,
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
@click.option(
    "--include-lga-features",
    "include_lga_features",
    is_flag=True,
    default=False,
    help="Append Phase 4 LGA_FEATURE_COLUMNS (days_since_trough_entry_<lga>) to the training feature set.",
)
def main(features_csv: str, model_out: str, reliability_out: str, include_lga_features: bool) -> None:
    """Train LightGBM on numeric features.

    No hyperparameter tuning, random_state=42. Test is intentionally left
    untouched. This command does not append to experiments/results.csv.
    """
    features_path = pathlib.Path(features_csv)
    if not features_path.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )

    feature_columns = (
        FEATURE_COLUMNS + LGA_FEATURE_COLUMNS if include_lga_features else FEATURE_COLUMNS
    )

    df = pd.read_csv(features_path)
    required = feature_columns + ["label", "price_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"Features CSV is missing required columns: {missing}. "
            "Re-run 'uv run python -m fuel_signal.features' to regenerate."
        )

    # If the CSV carries LGA columns but the flag wasn't passed, the user is
    # probably running Phase 3c by accident on a Phase 4 features.csv. Warn
    # loudly — this exact mismatch produced byte-identical val log-loss in
    # the Phase 4 retrain and cost a debugging round-trip.
    if not include_lga_features:
        present_lga = [c for c in LGA_FEATURE_COLUMNS if c in df.columns]
        if present_lga:
            click.echo(
                f"WARNING: features CSV contains {len(present_lga)} LGA columns "
                "but --include-lga-features was not passed. Training on the "
                "15-feat Phase 3c schema; LGA columns will be ignored.",
                err=True,
            )

    schema_label = "Phase 4" if include_lga_features else "Phase 3c"
    click.echo(f"Training on {len(feature_columns)} features ({schema_label} schema).")
    result = train_and_evaluate(df, feature_columns=feature_columns)

    click.echo(_format_results(result))

    model_path = pathlib.Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": result["pipeline"],
            "feature_columns": result["feature_columns"],
        },
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
            "\nWARNING: val log-loss did not beat the baseline. Investigate before proceeding.",
            err=True,
        )


if __name__ == "__main__":
    main()
