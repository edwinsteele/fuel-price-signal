"""Feature diagnostics CLI for the fitted LightGBM model.

Prints three sections against the canonical val split:
  1. Feature importance — gain % and split count per feature, sorted by gain desc.
  2. FN/FP delta table — (FN−TP) and (FP−TN) mean delta per feature, sorted by
     absolute FN−TP delta. Highlights where the model mis-ranks BUY vs WAIT rows.
  3. Error summary — TP/FP/TN/FN counts and label-BUY rate per group.

Usage::

    uv run python -m fuel_signal.feature_diagnostics
    uv run python -m fuel_signal.feature_diagnostics --model-path data/models/lgbm_calibrated.joblib
    uv run python -m fuel_signal.feature_diagnostics --threshold 0.35
"""

from __future__ import annotations

import pathlib

import click
import joblib
import numpy as np
import pandas as pd

from fuel_signal import evaluate as _ev

DEFAULT_MODEL_PATH = pathlib.Path("data/models/lgbm_calibrated.joblib")
DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_THRESHOLD = 0.40

_KNOWN_ID_COLUMNS: frozenset[str] = frozenset({"station_code"})
_ID_CARDINALITY_THRESHOLD: int = 20


def _is_id_column(col: str, series: pd.Series) -> bool:
    """Return True if col is an identifier rather than a numeric feature."""
    if col in _KNOWN_ID_COLUMNS:
        return True
    return pd.api.types.is_integer_dtype(series) and series.nunique() > _ID_CARDINALITY_THRESHOLD


def _load_artifact(model_path: pathlib.Path) -> dict:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise click.ClickException(
            f"Model artifact at {model_path} has invalid format. "
            "Expected a calibrated LightGBM artifact produced by calibrate.py."
        )
    for key in ("base_pipeline", "calibrator", "calibration_method", "feature_columns"):
        if key not in artifact:
            raise click.ClickException(
                f"Model artifact at {model_path} is missing key '{key}'. "
                "Expected a calibrated LightGBM artifact produced by calibrate.py."
            )
    return artifact


def _predict_proba(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Return calibrated probabilities for the positive class."""
    raw_p = artifact["base_pipeline"].predict_proba(X)[:, 1]
    method = artifact["calibration_method"]
    calibrator = artifact["calibrator"]
    if method == "sigmoid":
        cal_p = calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]
    else:
        cal_p = np.clip(calibrator.predict(raw_p), 0.0, 1.0)
    return cal_p


def feature_importance_section(artifact: dict) -> str:
    model = artifact["base_pipeline"]
    feature_columns = artifact["feature_columns"]
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    total_gain = gain.sum() or 1.0
    gain_pct = gain / total_gain * 100.0

    order = np.argsort(gain_pct)[::-1]
    lines = ["Feature importance (val — gain %)"]
    lines.append(f"  {'feature':<40} {'gain%':>7} {'splits':>7}")
    lines.append("  " + "-" * 58)
    for i in order:
        lines.append(f"  {feature_columns[i]:<40} {gain_pct[i]:>7.1f} {split[i]:>7d}")
    return "\n".join(lines)


def fn_fp_delta_section(
    val: pd.DataFrame,
    feature_columns: list[str],
    pred: np.ndarray,
) -> str:
    y = val["label"].to_numpy(dtype=int)
    tp_mask = (pred == 1) & (y == 1)
    fp_mask = (pred == 1) & (y == 0)
    tn_mask = (pred == 0) & (y == 0)
    fn_mask = (pred == 0) & (y == 1)

    rows = []
    id_cols: list[str] = []
    for col in feature_columns:
        if _is_id_column(col, val[col]):
            id_cols.append(col)
            continue
        values = val[col].to_numpy(dtype=float)
        mean_tp = values[tp_mask].mean() if tp_mask.any() else float("nan")
        mean_fn = values[fn_mask].mean() if fn_mask.any() else float("nan")
        mean_fp = values[fp_mask].mean() if fp_mask.any() else float("nan")
        mean_tn = values[tn_mask].mean() if tn_mask.any() else float("nan")
        rows.append({
            "feature": col,
            "fn_tp_delta": mean_fn - mean_tp,
            "fp_tn_delta": mean_fp - mean_tn,
        })

    df = pd.DataFrame(rows)
    df["abs_fn_tp"] = df["fn_tp_delta"].abs()
    df = df.sort_values("abs_fn_tp", ascending=False).drop(columns="abs_fn_tp")

    lines = ["FN−TP / FP−TN mean delta (val split, sorted by |FN−TP|)"]
    lines.append(f"  {'feature':<40} {'FN−TP':>10} {'FP−TN':>10}")
    lines.append("  " + "-" * 62)
    for _, row in df.iterrows():
        lines.append(
            f"  {row['feature']:<40} {row['fn_tp_delta']:>+10.3f} {row['fp_tn_delta']:>+10.3f}"
        )
    if id_cols:
        lines.append(f"  Excluded (ID columns — delta not meaningful): {', '.join(id_cols)}")
    return "\n".join(lines)


def error_summary_section(val: pd.DataFrame, pred: np.ndarray) -> str:
    y = val["label"].to_numpy(dtype=int)
    n = len(y)
    counts = {
        "TP": int(((pred == 1) & (y == 1)).sum()),
        "FP": int(((pred == 1) & (y == 0)).sum()),
        "TN": int(((pred == 0) & (y == 0)).sum()),
        "FN": int(((pred == 0) & (y == 1)).sum()),
    }
    buy_rates = {"TP": 1.0, "FP": 0.0, "TN": 0.0, "FN": 1.0}

    lines = ["Error summary (val split)"]
    lines.append(f"  {'group':<6} {'count':>7} {'%total':>7} {'buy_rate':>9}")
    lines.append("  " + "-" * 34)
    for group in ("TP", "FP", "TN", "FN"):
        cnt = counts[group]
        pct = cnt / n * 100.0 if n > 0 else 0.0
        lines.append(f"  {group:<6} {cnt:>7d} {pct:>7.1f}% {buy_rates[group]:>9.0%}")
    predicted_buy_rate = (counts["TP"] + counts["FP"]) / n * 100.0 if n > 0 else 0.0
    lines.append("")
    lines.append(f"  val rows: {n:,}   predicted-BUY rate: {predicted_buy_rate:.1f}%")
    return "\n".join(lines)


def run_diagnostics(
    model_path: pathlib.Path,
    features_csv: pathlib.Path,
    threshold: float,
) -> str:
    if not 0.0 <= threshold <= 1.0:
        raise click.ClickException(
            f"Invalid threshold {threshold}. Expected a value in [0, 1]."
        )
    artifact = _load_artifact(model_path)
    feature_columns: list[str] = artifact["feature_columns"]

    df = pd.read_csv(features_csv)
    missing = [c for c in feature_columns + ["label", "price_date"] if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"Features CSV is missing columns: {missing}. "
            "Re-run 'uv run python -m fuel_signal.features' to regenerate."
        )

    _train, val, _test = _ev.split(df)
    if val.empty:
        raise click.ClickException("Val split is empty — check features CSV date range.")

    X_val = val[feature_columns]
    cal_p = _predict_proba(artifact, X_val)
    pred = (cal_p >= threshold).astype(int)

    sections = [
        feature_importance_section(artifact),
        fn_fp_delta_section(val, feature_columns, pred),
        error_summary_section(val, pred),
    ]
    return "\n\n".join(sections)


@click.command("feature_diagnostics")
@click.option(
    "--model-path",
    "model_path",
    default=str(DEFAULT_MODEL_PATH),
    show_default=True,
    help="Calibrated LightGBM artifact (joblib dict with base_pipeline + calibrator).",
)
@click.option(
    "--features-csv",
    "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Feature rows CSV produced by `python -m fuel_signal.features`.",
)
@click.option(
    "--threshold",
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Decision threshold for binary BUY/WAIT prediction.",
)
def main(model_path: str, features_csv: str, threshold: float) -> None:
    """Feature importance, FN/FP deltas, and error summary for the calibrated LightGBM model."""
    mp = pathlib.Path(model_path)
    if not mp.exists():
        raise click.ClickException(
            f"Model artifact not found: {model_path}. "
            "Run calibration first."
        )
    fc = pathlib.Path(features_csv)
    if not fc.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )
    output = run_diagnostics(mp, fc, threshold)
    click.echo(output)


if __name__ == "__main__":
    main()
