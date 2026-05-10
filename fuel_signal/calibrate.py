"""Calibration diagnostics and calibrated-model artifact for the logreg baseline.

Issue #36: Phase 2.2 — Calibration check + handle val class imbalance.

## What this does

1. Reports class balance (BUY rate) for train / val / test.
2. Loads the fitted logreg pipeline from issue #35 and prints a 10-bin
   reliability table on val.  Flags if max |gap| > 0.05.
3. If miscalibrated: refits the base pipeline on the first 80% of train,
   then wraps it with ``CalibratedClassifierCV(cv='prefit')`` and fits the
   calibration layer on the remaining 20% — once for 'sigmoid' (Platt) and
   once for 'isotonic'.  Picks whichever reduces val logloss without
   increasing val Brier by more than 0.005.
4. Saves the chosen model (raw pipeline or calibrated wrapper) to
   ``data/models/logreg_calibrated.joblib``.
5. Appends a row to ``experiments/results.csv`` with val metrics and notes
   on the calibration decision and the val/train BUY-rate gap.

## Calibration concepts

Sigmoid (Platt) calibration is a parametric monotone transformation P → a·P + b
applied after predict_proba.  It is fast and works well when the raw model is
*close* to calibrated (logreg usually is).

Isotonic calibration is a non-parametric step function — more expressive but
requires more calibration data and can overfit on small sets.  It is the right
choice when the sigmoid fit is still visibly off the diagonal.

Both use ``cv='prefit'`` which means the base estimator is already fitted and
the calibration data is a held-out slice that the base model never saw.
"""

from __future__ import annotations

import pathlib
from typing import Any

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.train_logreg import build_pipeline

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_MODEL_IN = pathlib.Path("data/models/logreg.joblib")
DEFAULT_MODEL_OUT = pathlib.Path("data/models/logreg_calibrated.joblib")

_MISCAL_THRESHOLD = 0.05
_BRIER_REGRESSION_LIMIT = 0.005
_CALIB_HOLDOUT_FRAC = 0.20


# ---------------------------------------------------------------------------
# Calibrated-pipeline wrapper
# ---------------------------------------------------------------------------

class _CalibratedPipeline:
    """Thin wrapper combining a base sklearn Pipeline with a manual calibration layer.

    Exposes the same ``predict_proba`` interface as the raw pipeline so that
    downstream code can load from joblib and call ``predict_proba(X)``
    regardless of whether calibration was applied.
    """

    def __init__(self, base_pipeline: Any, calibrator: Any, method: str) -> None:
        self.base_pipeline = base_pipeline
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.base_pipeline.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            p_cal = self.calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
        else:  # isotonic
            p_cal = np.clip(self.calibrator.predict(p), 0.0, 1.0)
        return np.column_stack([1.0 - p_cal, p_cal])


# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------

def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 3-row DataFrame (split / n_rows / buy_rate) for train/val/test."""
    train, val, test = _ev.split(df)
    rows = []
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        rows.append({
            "split": name,
            "n_rows": len(subset),
            "buy_rate": float(subset["label"].mean()) if not subset.empty else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration comparison
# ---------------------------------------------------------------------------

def compare_calibrations(
    df: pd.DataFrame,
    raw_pipeline_path: pathlib.Path,
    feature_columns: list[str] | None = None,
) -> dict:
    """Compare raw logreg, sigmoid, and isotonic calibration on val.

    Re-fits the base pipeline on the first 80% of train (temporal order) and
    fits both calibration wrappers on the remaining 20%.  Evaluates all three
    on val.

    Returns a dict with keys:
        "raw"      → {"val_logloss", "val_brier", "p_val"}
        "sigmoid"  → {"val_logloss", "val_brier", "p_val", "model"}
        "isotonic" → {"val_logloss", "val_brier", "p_val", "model"}
        "y_val"    → np.ndarray
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    train, val, _test = _ev.split(df)

    # --- raw model predictions on val ---
    loaded = joblib.load(raw_pipeline_path)
    raw_pipe = loaded["pipeline"]
    X_val = val[feature_columns].to_numpy(dtype=float)
    y_val = val["label"].to_numpy(dtype=int)
    p_raw = raw_pipe.predict_proba(X_val)[:, 1]

    # --- split train temporally: refit base on first 80%, calibrate on last 20% ---
    train_sorted = train.sort_values("price_date")
    n_calib = max(1, int(len(train_sorted) * _CALIB_HOLDOUT_FRAC))
    df_fit = train_sorted.iloc[:-n_calib]
    df_calib = train_sorted.iloc[-n_calib:]

    if df_fit.empty:
        raise ValueError(
            "compare_calibrations(): fit slice is empty after temporal split — "
            "train split is too small for calibration."
        )

    X_fit = df_fit[feature_columns].to_numpy(dtype=float)
    y_fit = df_fit["label"].to_numpy(dtype=int)
    X_calib = df_calib[feature_columns].to_numpy(dtype=float)
    y_calib = df_calib["label"].to_numpy(dtype=int)

    if len(np.unique(y_fit)) < 2:
        raise ValueError(
            "compare_calibrations(): fit slice has only one class — "
            "cannot fit the base logistic regression."
        )
    if len(np.unique(y_calib)) < 2:
        raise ValueError(
            "compare_calibrations(): calibration slice has only one class — "
            "cannot fit sigmoid or isotonic calibrators."
        )

    base = build_pipeline()
    base.fit(X_fit, y_fit)
    p_calib_raw = base.predict_proba(X_calib)[:, 1]
    p_val_from_base = base.predict_proba(X_val)[:, 1]

    # Sigmoid (Platt): fit a LogisticRegression on (p_calib, y_calib).
    sigmoid_cal = LogisticRegression()
    sigmoid_cal.fit(p_calib_raw.reshape(-1, 1), y_calib)
    p_sig = sigmoid_cal.predict_proba(p_val_from_base.reshape(-1, 1))[:, 1]

    # Isotonic: fit IsotonicRegression(out_of_bounds='clip') on (p_calib, y_calib).
    isotonic_cal = IsotonicRegression(out_of_bounds="clip")
    isotonic_cal.fit(p_calib_raw, y_calib)
    p_iso = np.clip(isotonic_cal.predict(p_val_from_base), 0.0, 1.0)

    calibrated = {
        "sigmoid": {
            "model": _CalibratedPipeline(base, sigmoid_cal, "sigmoid"),
            "val_logloss": _ev.log_loss(y_val, p_sig),
            "val_brier": _ev.brier(y_val, p_sig),
            "p_val": p_sig,
        },
        "isotonic": {
            "model": _CalibratedPipeline(base, isotonic_cal, "isotonic"),
            "val_logloss": _ev.log_loss(y_val, p_iso),
            "val_brier": _ev.brier(y_val, p_iso),
            "p_val": p_iso,
        },
    }

    return {
        "raw": {
            "val_logloss": _ev.log_loss(y_val, p_raw),
            "val_brier": _ev.brier(y_val, p_raw),
            "p_val": p_raw,
        },
        "sigmoid": calibrated["sigmoid"],
        "isotonic": calibrated["isotonic"],
        "y_val": y_val,
    }


def pick_best(compare: dict, max_gap: float) -> tuple[str, object]:
    """Pick the best model variant; return (name, model_or_None).

    'model_or_None' is None for 'raw' (caller should load the raw pipeline).
    Decision rule:
      1. If raw is well-calibrated (max |gap| ≤ _MISCAL_THRESHOLD) → use raw.
      2. Otherwise pick the calibration method that has lower val logloss
         provided it does not regress Brier by more than _BRIER_REGRESSION_LIMIT.
      3. If neither calibration method beats raw, fall back to raw.
    """
    if max_gap <= _MISCAL_THRESHOLD:
        return "raw", None

    raw_ll = compare["raw"]["val_logloss"]
    raw_br = compare["raw"]["val_brier"]

    candidates = []
    for method in ("sigmoid", "isotonic"):
        c = compare[method]
        if c["val_logloss"] < raw_ll and (c["val_brier"] - raw_br) <= _BRIER_REGRESSION_LIMIT:
            candidates.append((method, c["val_logloss"], c["model"]))

    if not candidates:
        return "raw", None

    # lowest logloss among valid candidates
    best_method, _, best_model = min(candidates, key=lambda t: t[1])
    return best_method, best_model


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_class_balance(cb: pd.DataFrame) -> str:
    lines = ["Class balance (BUY rate):"]
    train_rate = cb.loc[cb["split"] == "train", "buy_rate"].iloc[0]
    for _, row in cb.iterrows():
        pct = row["buy_rate"] * 100
        ratio = f"  (×{row['buy_rate'] / train_rate:.2f} vs train)" if row["split"] != "train" else ""
        lines.append(f"  {row['split']:<6}: {pct:5.1f}%  ({int(row['n_rows']):>9,} rows){ratio}")
    return "\n".join(lines)


def _fmt_reliability(tbl: pd.DataFrame, max_gap: float, flagged: bool) -> str:
    lines = [
        "Reliability table — val (10 bins):",
        f"  {'bin':>3}  {'mean_pred':>9}  {'actual_rate':>11}  {'count':>7}  {'gap':>7}",
    ]
    for i, row in tbl.iterrows():
        marker = " ◄" if abs(row["gap"]) == max_gap else ""
        lines.append(
            f"  {i + 1:>3}  {row['bin_mean_pred']:9.4f}  {row['actual_rate']:11.4f}"
            f"  {int(row['count']):>7,}  {row['gap']:+7.4f}{marker}"
        )
    flag = "FLAG: max |gap| > 0.05 — inspect reliability plot" if flagged else "OK: max |gap| ≤ 0.05"
    lines.append(f"\n  max |gap| = {max_gap:.4f}  →  {flag}")
    return "\n".join(lines)


def _fmt_comparison(compare: dict, best_name: str) -> str:
    lines = ["Calibration comparison (val):"]
    for name in ("raw", "sigmoid", "isotonic"):
        c = compare[name]
        marker = " ◄ chosen" if name == best_name else ""
        lines.append(
            f"  {name:<9}: logloss {c['val_logloss']:.4f}  brier {c['val_brier']:.4f}{marker}"
        )
    return "\n".join(lines)


def _build_notes(cb: pd.DataFrame, max_gap: float, best_name: str, compare: dict) -> str:
    train_rate = cb.loc[cb["split"] == "train", "buy_rate"].iloc[0]
    val_rate = cb.loc[cb["split"] == "val", "buy_rate"].iloc[0]
    test_rate = cb.loc[cb["split"] == "test", "buy_rate"].iloc[0]
    ratio = val_rate / train_rate if train_rate > 0 else float("nan")
    calibrated = best_name != "raw"
    cal_note = (
        f"Calibration applied ({best_name}); raw logloss {compare['raw']['val_logloss']:.4f} "
        f"→ {compare[best_name]['val_logloss']:.4f}."
        if calibrated
        else f"Raw logreg well-calibrated (max |gap| {max_gap:.4f}); no wrapper applied."
    )
    rate_summary = (
        f"Val/train BUY-rate ratio: {ratio:.2f} "
        f"(val {val_rate:.3f} vs train {train_rate:.3f} vs test {test_rate:.3f})."
    )
    threshold_note = (
        "Issue 2.3 must not pick a threshold on val without correcting for this "
        "elevated positive rate — test mirrors train (ratio ≈1); any val-tuned "
        "threshold will be pessimistic on precision at test time."
    )
    return f"{cal_note} {rate_summary} {threshold_note}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("calibrate")
@click.option(
    "--features-csv",
    "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Path to feature rows CSV produced by `python -m fuel_signal.features`.",
)
@click.option(
    "--model-in",
    "model_in",
    default=str(DEFAULT_MODEL_IN),
    show_default=True,
    help="Fitted logreg pipeline from issue #35 (joblib).",
)
@click.option(
    "--model-out",
    "model_out",
    default=str(DEFAULT_MODEL_OUT),
    show_default=True,
    help="Where to save the chosen model artifact (raw or calibrated).",
)
@click.option(
    "--skip-results-csv",
    "skip_results_csv",
    is_flag=True,
    default=False,
    help="Do not append a row to experiments/results.csv (useful in tests).",
)
def main(features_csv: str, model_in: str, model_out: str, skip_results_csv: bool) -> None:
    """Calibration check and artifact for the logreg baseline (issue #36).

    Prints class balance, reliability table, and calibration comparison on val.
    Saves the best model (raw or calibrated) to --model-out.
    Appends a result row to experiments/results.csv unless --skip-results-csv.
    """
    features_path = pathlib.Path(features_csv)
    if not features_path.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )
    model_path = pathlib.Path(model_in)
    if not model_path.exists():
        raise click.ClickException(
            f"Model not found: {model_in}. "
            "Run 'uv run python -m fuel_signal.train_logreg' first."
        )

    df = pd.read_csv(features_path)

    # --- class balance ---
    cb = class_balance(df)
    click.echo(_fmt_class_balance(cb))
    click.echo()

    # --- reliability table on val ---
    _, val, _ = _ev.split(df)
    loaded = joblib.load(model_path)
    raw_pipe = loaded["pipeline"]
    feature_columns = loaded.get("feature_columns", FEATURE_COLUMNS)
    X_val = val[feature_columns].to_numpy(dtype=float)
    y_val = val["label"].to_numpy(dtype=int)
    p_raw = raw_pipe.predict_proba(X_val)[:, 1]

    tbl = _ev.reliability_table(y_val, p_raw)
    max_gap = float(tbl["gap"].abs().max())
    flagged = max_gap > _MISCAL_THRESHOLD
    click.echo(_fmt_reliability(tbl, max_gap, flagged))
    click.echo()

    # --- calibration comparison (always run so we can report numbers) ---
    compare = compare_calibrations(df, model_path, feature_columns)
    best_name, best_model = pick_best(compare, max_gap)
    click.echo(_fmt_comparison(compare, best_name))
    click.echo()

    # --- save artifact ---
    out_path = pathlib.Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_name == "raw":
        joblib.dump({"pipeline": raw_pipe, "feature_columns": feature_columns, "calibrated": False}, out_path)
        click.echo(f"Calibration decision: raw pipeline is sufficient — saved as-is to {out_path}")
    else:
        # Store sklearn primitives rather than the _CalibratedPipeline instance so that
        # the artifact can be loaded from any entry point (the custom class has a
        # __module__='__main__' when calibrate.py is run with -m, which breaks joblib).
        joblib.dump(
            {
                "base_pipeline": best_model.base_pipeline,
                "calibrator": best_model.calibrator,
                "calibration_method": best_name,
                "feature_columns": feature_columns,
                "calibrated": True,
            },
            out_path,
        )
        click.echo(f"Calibration decision: {best_name} applied — saved calibrated model to {out_path}")

    # --- results.csv ---
    if not skip_results_csv:
        notes = _build_notes(cb, max_gap, best_name, compare)
        chosen = compare[best_name]
        _ev.log_experiment(
            name=f"logreg_calibration_check_{best_name}",
            features=feature_columns,
            holdout_logloss=chosen["val_logloss"],
            holdout_brier=chosen["val_brier"],
            notes=notes,
        )
        click.echo("Appended calibration row to experiments/results.csv")

    # --- val/train BUY-rate summary ---
    train_rate = cb.loc[cb["split"] == "train", "buy_rate"].iloc[0]
    val_rate = cb.loc[cb["split"] == "val", "buy_rate"].iloc[0]
    test_rate = cb.loc[cb["split"] == "test", "buy_rate"].iloc[0]
    click.echo()
    click.echo(
        f"Val/train BUY-rate ratio: {val_rate / train_rate:.2f}  "
        f"(train {train_rate:.3f} / val {val_rate:.3f} / test {test_rate:.3f})\n"
        f"Threshold note for issue 2.3: val has a higher BUY rate than train/test.\n"
        f"Any threshold chosen purely on val will be tuned to an elevated positive rate.\n"
        f"Test mirrors train — validate chosen threshold on test before trusting it."
    )


if __name__ == "__main__":
    main()
