"""Calibration diagnostics and calibrated-model artifact for the logreg baseline.

Issue #36: Phase 2.2 — Calibration check + handle val class imbalance.
Issue #236: Fix calibration selection bias from unrepresentative val window.

## What this does

1. Reports class balance (BUY rate) for train / val / test.
2. Runs walk-forward CV over the training split to pool OOF predictions at
   training base rate (~0.24), not val rate (~0.32).
3. Fits sigmoid and isotonic calibrators on OOF predictions; selects the best
   method by OOF logloss (representative base rate).
4. Saves the chosen model to ``data/models/lgbm_calibrated.joblib``.
   Calibrated artifacts ship the 100%-trained raw base — no 80%-train handicap.
5. Appends a row to ``experiments/results.csv`` with OOF + val metrics and a
   note on the calibration decision.

## Calibration concepts

Sigmoid (Platt) calibration is a parametric monotone transformation P → a·P + b
applied after predict_proba.  It is fast and works well when the raw model is
*close* to calibrated (logreg usually is).

Isotonic calibration is a non-parametric step function — more expressive but
requires more calibration data and can overfit on small sets.  It is the right
choice when the sigmoid fit is still visibly off the diagonal.

Both calibrators are fit on OOF predictions pooled from walk-forward CV over the
training split.  Selection also uses OOF logloss (not val), so the 0.32 val BUY
rate does not bias the decision.  The shipped base is the 100%-trained raw
pipeline, not the 80%-refit sub-model used in the pre-#236 design.
"""

from __future__ import annotations

import pathlib
from typing import Any

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")
DEFAULT_MODEL_IN = pathlib.Path("data/models/lgbm.joblib")
DEFAULT_MODEL_OUT = pathlib.Path("data/models/lgbm_calibrated.joblib")

_MISCAL_THRESHOLD = 0.05
_BRIER_REGRESSION_LIMIT = 0.005


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
# Shared OOF helper
# ---------------------------------------------------------------------------

def pool_oof_predictions(
    raw_pipe: Any,
    train: pd.DataFrame,
    feature_columns: list[str],
    fold_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Run walk-forward CV over train and return (p_oof, y_oof).

    For each fold: clone raw_pipe, fit on fold_train, predict uncalibrated
    probabilities on fold_val.  Returns empty arrays when no folds are generated.
    Used by both compare_calibrations (calibration fitting/selection) and
    oof_threshold_predictions in score_phase2 (threshold sweep).
    """
    oof_ps: list[np.ndarray] = []
    oof_ys: list[np.ndarray] = []
    for fold_train, fold_val in _ev.walk_forward_folds(train, **fold_params):
        if len(np.unique(fold_train["label"])) < 2:
            continue
        fold_base = clone(raw_pipe)
        fold_base.fit(fold_train[feature_columns], fold_train["label"].to_numpy(dtype=int))
        p = fold_base.predict_proba(fold_val[feature_columns])[:, 1]
        oof_ps.append(p)
        oof_ys.append(fold_val["label"].to_numpy(dtype=int))
    if not oof_ps:
        return np.array([]), np.array([])
    return np.concatenate(oof_ps), np.concatenate(oof_ys)


# ---------------------------------------------------------------------------
# Calibration comparison
# ---------------------------------------------------------------------------

def compare_calibrations(
    df: pd.DataFrame,
    raw_pipeline_path: pathlib.Path,
    feature_columns: list[str] | None = None,
    fold_params: dict | None = None,
) -> dict:
    """Compare raw, sigmoid, and isotonic calibration using train-CV OOF predictions.

    Runs walk-forward CV over the training split to pool OOF predictions at
    training base rate (~0.24).  Both calibration fitting and method selection
    use OOF metrics — not val metrics at the unrepresentative 0.32 BUY rate.

    Shipped calibrated models use the 100%-trained raw pipeline as their base
    (no 80%-train handicap from the pre-#236 implementation).

    fold_params (optional): kwargs forwarded to walk_forward_folds, e.g.
        {"train_min_days": 200, "val_days": 30, "step_days": 30}.
        Useful in tests to generate folds on small synthetic datasets.

    Returns a dict with keys:
        "raw"      → {"oof_logloss", "oof_brier", "val_logloss", "val_brier", "p_val", "p_oof"}
        "sigmoid"  → {"oof_logloss", "oof_brier", "val_logloss", "val_brier", "p_val", "p_oof", "model"}
        "isotonic" → {"oof_logloss", "oof_brier", "val_logloss", "val_brier", "p_val", "p_oof", "model"}
        "y_val"    → np.ndarray
        "y_oof"    → np.ndarray
        "oof_buy_rate" → float
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    train, val, _ = _ev.split(df)

    loaded = joblib.load(raw_pipeline_path)
    raw_pipe = loaded["pipeline"]

    # --- walk-forward OOF over train to pool predictions at training base rate ---
    _folds = fold_params or {}
    p_oof_all, y_oof_all = pool_oof_predictions(raw_pipe, train, feature_columns, _folds)

    if p_oof_all.size == 0:
        raise ValueError(
            "compare_calibrations(): no CV folds generated over train — "
            "train split may be too small for walk_forward_folds defaults. "
            "Pass fold_params={'train_min_days': N, ...} to override."
        )

    if len(np.unique(y_oof_all)) < 2:
        raise ValueError(
            "compare_calibrations(): OOF labels contain only one class — "
            "cannot fit sigmoid or isotonic calibrators."
        )

    # --- fit calibrators on OOF predictions ---
    sigmoid_cal = LogisticRegression()
    sigmoid_cal.fit(p_oof_all.reshape(-1, 1), y_oof_all)

    isotonic_cal = IsotonicRegression(out_of_bounds="clip")
    isotonic_cal.fit(p_oof_all, y_oof_all)

    p_oof_sig = sigmoid_cal.predict_proba(p_oof_all.reshape(-1, 1))[:, 1]
    p_oof_iso = np.clip(isotonic_cal.predict(p_oof_all), 0.0, 1.0)

    # --- val metrics (secondary — reported but not used for selection) ---
    X_val = val[feature_columns]
    y_val = val["label"].to_numpy(dtype=int)
    p_raw_val = raw_pipe.predict_proba(X_val)[:, 1]

    # Shipped models use raw_pipe (100%-trained) as base — no 80%-train handicap.
    sig_model = _CalibratedPipeline(raw_pipe, sigmoid_cal, "sigmoid")
    iso_model = _CalibratedPipeline(raw_pipe, isotonic_cal, "isotonic")
    p_sig_val = sig_model.predict_proba(X_val)[:, 1]
    p_iso_val = iso_model.predict_proba(X_val)[:, 1]

    return {
        "raw": {
            "oof_logloss": _ev.log_loss(y_oof_all, p_oof_all),
            "oof_brier": _ev.brier(y_oof_all, p_oof_all),
            "val_logloss": _ev.log_loss(y_val, p_raw_val),
            "val_brier": _ev.brier(y_val, p_raw_val),
            "p_val": p_raw_val,
            "p_oof": p_oof_all,
        },
        "sigmoid": {
            "oof_logloss": _ev.log_loss(y_oof_all, p_oof_sig),
            "oof_brier": _ev.brier(y_oof_all, p_oof_sig),
            "val_logloss": _ev.log_loss(y_val, p_sig_val),
            "val_brier": _ev.brier(y_val, p_sig_val),
            "p_val": p_sig_val,
            "p_oof": p_oof_sig,
            "model": sig_model,
        },
        "isotonic": {
            "oof_logloss": _ev.log_loss(y_oof_all, p_oof_iso),
            "oof_brier": _ev.brier(y_oof_all, p_oof_iso),
            "val_logloss": _ev.log_loss(y_val, p_iso_val),
            "val_brier": _ev.brier(y_val, p_iso_val),
            "p_val": p_iso_val,
            "p_oof": p_oof_iso,
            "model": iso_model,
        },
        "y_val": y_val,
        "y_oof": y_oof_all,
        "oof_buy_rate": float(y_oof_all.mean()),
    }


def pick_best(compare: dict, max_gap: float) -> tuple[str, object]:
    """Pick the best model variant; return (name, model_or_None).

    'model_or_None' is None for 'raw' (caller should load the raw pipeline).
    Selection uses OOF logloss/brier (train-CV base rate) not val metrics.
    Decision rule:
      1. If raw is well-calibrated (max |gap| ≤ _MISCAL_THRESHOLD) → use raw.
      2. Otherwise pick the calibration method that has lower OOF logloss
         provided it does not regress OOF Brier by more than _BRIER_REGRESSION_LIMIT.
      3. If neither calibration method beats raw, fall back to raw.
    """
    if max_gap <= _MISCAL_THRESHOLD:
        return "raw", None

    raw_ll = compare["raw"]["oof_logloss"]
    raw_br = compare["raw"]["oof_brier"]

    candidates = []
    for method in ("sigmoid", "isotonic"):
        c = compare[method]
        if c["oof_logloss"] < raw_ll and (c["oof_brier"] - raw_br) <= _BRIER_REGRESSION_LIMIT:
            candidates.append((method, c["oof_logloss"], c["model"]))

    if not candidates:
        return "raw", None

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


def _fmt_reliability(tbl: pd.DataFrame, max_gap: float, flagged: bool, source: str = "OOF") -> str:
    lines = [
        f"Reliability table — {source} (10 bins):",
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
    lines = ["Calibration comparison (OOF primary / val secondary):"]
    for name in ("raw", "sigmoid", "isotonic"):
        c = compare[name]
        marker = " ◄ chosen" if name == best_name else ""
        lines.append(
            f"  {name:<9}: OOF logloss {c['oof_logloss']:.4f}  brier {c['oof_brier']:.4f}"
            f"  |  val logloss {c['val_logloss']:.4f}  brier {c['val_brier']:.4f}{marker}"
        )
    return "\n".join(lines)


def _build_notes(
    cb: pd.DataFrame,
    max_gap: float,
    best_name: str,
    compare: dict,
) -> str:
    train_rate = cb.loc[cb["split"] == "train", "buy_rate"].iloc[0]
    val_rate = cb.loc[cb["split"] == "val", "buy_rate"].iloc[0]
    test_rate = cb.loc[cb["split"] == "test", "buy_rate"].iloc[0]
    oof_rate = compare.get("oof_buy_rate", float("nan"))
    ratio = val_rate / train_rate if train_rate > 0 else float("nan")
    calibrated = best_name != "raw"
    cal_note = (
        f"Calibration applied ({best_name}); raw OOF logloss {compare['raw']['oof_logloss']:.4f} "
        f"→ {compare[best_name]['oof_logloss']:.4f}."
        if calibrated
        else f"Raw model well-calibrated on OOF (max |gap| {max_gap:.4f}); no wrapper applied."
    )
    rate_summary = (
        f"OOF BUY rate {oof_rate:.3f} (train {train_rate:.3f} / val {val_rate:.3f} / test {test_rate:.3f}). "
        f"Val/train ratio: {ratio:.2f} — selection based on OOF, not val."
    )
    return f"{cal_note} {rate_summary}"


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
    help="Fitted model artifact (joblib) produced by train_lgbm.py or train_logreg.py.",
)
@click.option(
    "--model-out",
    "model_out",
    default=str(DEFAULT_MODEL_OUT),
    show_default=True,
    help="Where to save the chosen model artifact (raw or calibrated).",
)
@click.option(
    "--model-name",
    "model_name",
    default="lgbm",
    show_default=True,
    help="Prefix used for the experiment name in results.csv (e.g. 'lgbm' or 'logreg').",
)
@click.option(
    "--skip-results-csv",
    "skip_results_csv",
    is_flag=True,
    default=False,
    help="Do not append a row to experiments/results.csv (useful in tests).",
)
def main(features_csv: str, model_in: str, model_out: str, model_name: str, skip_results_csv: bool) -> None:
    """Calibration check and artifact for a fitted model.

    Default: reads data/models/lgbm.joblib, writes data/models/lgbm_calibrated.joblib.
    Runs walk-forward CV over train to select and fit calibration at training base
    rate.  Saves the best model (raw or calibrated) to --model-out.
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

    # --- load model metadata (raw_pipe used if best_name == "raw") ---
    loaded = joblib.load(model_path)
    raw_pipe = loaded["pipeline"]
    feature_columns = loaded.get("feature_columns", FEATURE_COLUMNS)

    # --- CV-based calibration comparison ---
    click.echo("Running walk-forward CV over train to pool OOF predictions …")
    compare = compare_calibrations(df, model_path, feature_columns)

    # --- reliability table on OOF ---
    p_oof = compare["raw"]["p_oof"]
    y_oof = compare["y_oof"]
    tbl = _ev.reliability_table(y_oof, p_oof)
    max_gap = float(tbl["gap"].abs().max())
    flagged = max_gap > _MISCAL_THRESHOLD
    click.echo(_fmt_reliability(
        tbl, max_gap, flagged,
        source=f"train OOF ({len(y_oof):,} rows, BUY {compare['oof_buy_rate']:.3f})",
    ))
    click.echo()

    # --- pick best and show comparison ---
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
            name=f"{model_name}_calibration_check_{best_name}",
            features=feature_columns,
            holdout_logloss=chosen["oof_logloss"],
            holdout_brier=chosen["oof_brier"],
            notes=notes,
        )
        click.echo("Appended calibration row to experiments/results.csv")

    # --- BUY-rate summary ---
    train_rate = cb.loc[cb["split"] == "train", "buy_rate"].iloc[0]
    val_rate = cb.loc[cb["split"] == "val", "buy_rate"].iloc[0]
    test_rate = cb.loc[cb["split"] == "test", "buy_rate"].iloc[0]
    click.echo()
    click.echo(
        f"BUY-rate summary:\n"
        f"  OOF   (selection basis) : {compare['oof_buy_rate']:.3f}\n"
        f"  Train                   : {train_rate:.3f}\n"
        f"  Val   (not used for sel): {val_rate:.3f}  (×{val_rate / train_rate:.2f} vs train)\n"
        f"  Test                    : {test_rate:.3f}"
    )


if __name__ == "__main__":
    main()
