"""Logistic regression baseline for the ML price-movement model.

First real model in the ML plan — a vanilla logreg trained on the cycle features
produced by ``fuel_signal.features``. Issue #35.

## Pipeline

    StandardScaler → LogisticRegression(max_iter=1000)

StandardScaler matters here: logreg's L2 regulariser penalises large coefficients,
and without scaling the features with the largest raw magnitudes (e.g. cents)
would be regularised more heavily than small-magnitude ones (e.g. pct_through).
Scaling puts all features on equal footing so the regularisation is fair.

## What this module evaluates

Train on the canonical train split, score on **val only**. Test is reserved for
issue 2.3 once the model + threshold is locked. We do not write to
``experiments/results.csv`` from here — that file is for test scores.

## Reliability plot

Predicted-probability vs actual-rate, 10 quantile bins. The y=x reference line
is what a perfectly calibrated model produces: among rows it predicted with prob
~0.3, ~30% should actually be positive. Curves below the diagonal mean the model
is overconfident; curves above mean it is under-confident.
"""

from __future__ import annotations

import pathlib

import click
import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe; required for CLI use without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402
from fuel_signal.features import FEATURE_COLUMNS  # noqa: E402

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_MODEL_OUT = pathlib.Path("data/models/logreg.joblib")
DEFAULT_RELIABILITY_PNG = pathlib.Path("experiments/reliability_logreg_val.png")


def build_pipeline() -> Pipeline:
    """Standard logreg pipeline: StandardScaler → LogisticRegression(max_iter=1000)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])


def reliability_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_mean_pred, bin_actual_rate, bin_count) for `n_bins` quantile bins.

    Quantile binning (equal row counts per bin) is more robust than equal-width
    when probabilities are clustered, which is typical for an under-confident model.
    Bins with zero rows are dropped.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"reliability_bins(): shape mismatch — y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("reliability_bins() requires non-empty inputs.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_pred, quantiles))
    if edges.size < 2:
        # Degenerate — all predictions identical. Single-point reliability.
        return (
            np.array([y_pred.mean()]),
            np.array([y_true.mean()]),
            np.array([y_true.size]),
        )

    # np.digitize with right=False puts each value into [edge_i, edge_{i+1}); clip
    # bin index to valid range so the max edge value lands in the last bin.
    bin_idx = np.clip(np.digitize(y_pred, edges[1:-1], right=False), 0, len(edges) - 2)

    means_pred, means_actual, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        means_pred.append(float(y_pred[mask].mean()))
        means_actual.append(float(y_true[mask].mean()))
        counts.append(n)
    return np.array(means_pred), np.array(means_actual), np.array(counts)


def save_reliability_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: pathlib.Path,
    n_bins: int = 10,
    title: str = "Reliability — logreg, val split",
    model_label: str = "logreg",
) -> None:
    """Save predicted-vs-actual quantile-bin reliability plot with y=x reference."""
    bin_pred, bin_actual, bin_count = reliability_bins(y_true, y_pred, n_bins=n_bins)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration (y=x)")
    ax.plot(bin_pred, bin_actual, marker="o", color="C0", label=model_label)
    for x, y, n in zip(bin_pred, bin_actual, bin_count):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Actual positive rate (per bin)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict:
    """Fit logreg pipeline on train; score on val.

    Returns a dict with the fitted pipeline, train/val sizes, class balances,
    val predictions+labels, val log-loss, val brier, and baseline comparisons.
    Does not touch the test split — caller can split themselves if needed.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    train, val, _test = _ev.split(df)
    if train.empty:
        raise ValueError("train_and_evaluate(): train split is empty.")
    if val.empty:
        raise ValueError("train_and_evaluate(): val split is empty.")

    # Logreg cannot handle NaN. Nullable features (lga_mean_cents,
    # brand_mean_cents and their deltas) drop ~3% of rows. LightGBM handles
    # NaN natively and does not need this filter.
    n_train_before, n_val_before = len(train), len(val)
    train = train.dropna(subset=list(feature_columns))
    val = val.dropna(subset=list(feature_columns))
    n_train_dropped = n_train_before - len(train)
    n_val_dropped = n_val_before - len(val)
    if train.empty:
        raise ValueError(
            f"train_and_evaluate(): train split is empty after dropping NaN rows "
            f"(dropped {n_train_dropped:,}/{n_train_before:,})."
        )
    if val.empty:
        raise ValueError(
            f"train_and_evaluate(): val split is empty after dropping NaN rows "
            f"(dropped {n_val_dropped:,}/{n_val_before:,})."
        )

    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    X_val = val[feature_columns].to_numpy(dtype=float)
    y_val = val["label"].to_numpy(dtype=int)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # predict_proba returns shape (n, 2); column 1 is P(label=1).
    p_val = pipeline.predict_proba(X_val)[:, 1]

    val_logloss = _ev.log_loss(y_val, p_val)
    val_brier = _ev.brier(y_val, p_val)

    # Baseline = constant predictor at train marginal rate.
    p_baseline = _ev.baseline_prior(train)
    baseline_pred_val = np.full(len(y_val), p_baseline)
    baseline_logloss = _ev.log_loss(y_val, baseline_pred_val)
    baseline_brier = _ev.brier(y_val, baseline_pred_val)

    return {
        "pipeline": pipeline,
        "feature_columns": list(feature_columns),
        "train_size": int(len(train)),
        "val_size": int(len(val)),
        "train_size_before_dropna": int(n_train_before),
        "val_size_before_dropna": int(n_val_before),
        "train_dropna_count": int(n_train_dropped),
        "val_dropna_count": int(n_val_dropped),
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
        "Logistic regression — val results",
        f"  train rows           : {result['train_size']:>8,}  (pos rate {result['train_positive_rate']:.3f})",
        f"  val rows             : {result['val_size']:>8,}  (pos rate {result['val_positive_rate']:.3f})",
        f"  baseline prior (train): {result['baseline_prior']:.4f}",
        "",
        f"  val log-loss         : {result['val_logloss']:.4f}  (baseline {result['baseline_val_logloss']:.4f},"
        f"  Δ {delta_ll:+.4f})",
        f"  val brier            : {result['val_brier']:.4f}  (baseline {result['baseline_val_brier']:.4f},"
        f"  Δ {delta_br:+.4f})",
    ]
    return "\n".join(lines)


@click.command("train_logreg")
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
    help="Where to save the fitted joblib pipeline.",
)
@click.option(
    "--reliability-out",
    "reliability_out",
    default=str(DEFAULT_RELIABILITY_PNG),
    show_default=True,
    help="Where to save the val reliability plot.",
)
def main(features_csv: str, model_out: str, reliability_out: str) -> None:
    """Train a logreg baseline and score it on the val split.

    Test is intentionally left untouched — that split is reserved for the locked
    final-model evaluation in issue 2.3. This command does not append to
    experiments/results.csv.
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

    if result["train_dropna_count"] or result["val_dropna_count"]:
        click.echo(
            f"[train_logreg] dropped NaN rows: "
            f"train {result['train_dropna_count']:,}/{result['train_size_before_dropna']:,} "
            f"({100*result['train_dropna_count']/result['train_size_before_dropna']:.2f}%); "
            f"val {result['val_dropna_count']:,}/{result['val_size_before_dropna']:,} "
            f"({100*result['val_dropna_count']/result['val_size_before_dropna']:.2f}%)"
        )

    click.echo(_format_results(result))

    model_path = pathlib.Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": result["pipeline"], "feature_columns": result["feature_columns"]},
        model_path,
    )
    click.echo(f"\nSaved fitted pipeline to {model_path}")

    reliability_path = pathlib.Path(reliability_out)
    save_reliability_plot(result["y_val"], result["p_val"], reliability_path)
    click.echo(f"Saved reliability plot to  {reliability_path}")

    if result["val_logloss"] >= result["baseline_val_logloss"]:
        click.echo(
            "\nWARNING: val log-loss did not beat the baseline. "
            "Investigate before merging — see issue #35 acceptance criteria.",
            err=True,
        )


if __name__ == "__main__":
    main()
