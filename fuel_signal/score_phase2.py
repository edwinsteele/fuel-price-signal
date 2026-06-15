"""Phase 2 final evaluation: threshold sweep on OOF/val, one-time test scoring.

Issues #37 / #34 / #236.

## Threshold-selection criterion (updated #236)

When --model-path is provided, threshold selection uses walk-forward CV OOF
predictions over the training split (base rate ~0.24), not the single val window
(BUY rate ~0.32).  OOF predictions are at training base rate so no τ adjustment
is needed.  The old +0.05 adjustment was a workaround for the val BUY-rate bias;
it is no longer applied on the model-path code path.

When --model-path is absent (logreg retraining path), the legacy val-based sweep
with tau_adjustment is preserved.

## Cost model (documented here; used consistently throughout)

  TP (BUY, label=1):  +tp_reward_cents saved  (e.g. +6.37c, 95th-pct trimmed mean from tp_benefit.py)
  FP (BUY, label=0):  −fp_cost_cents penalty  (e.g. −5.80c, population median from fp_cost.py)
  FN (WAIT, label=1):  −fn_cost_cents penalty  (e.g. −11.14c, 95th-pct trimmed mean from fn_cost.py)
  TN (WAIT, label=0):  0

Expected cents per row = (TP × tp_reward_cents − FP × fp_cost_cents − FN × fn_cost_cents) / n_rows

Note: tp_reward_cents (6.37c) and threshold_cents (3.0c) are distinct. threshold_cents is
the label definition (a 3c drop = "cheap enough"); tp_reward_cents is the empirical average
saving achieved on a correct BUY — derived from mean 7-day forward price minus today on
label=1 rows (95th-pct trimmed mean from tp_benefit.py; raw mean 8.01c, trimmed to match
FN methodology by excluding the same supply-shock extremes).

FP penalty (5.80c): population median damage across all label=0 rows — frequency-weighted
across Cluster A (gate only failed, ~0c real cost) and Cluster B (drop came, ~9c median).
FN penalty (11.14c): 95th-percentile trimmed mean of price_7d_later − today_price on
label=1 rows. Trimming top 5% removes supply-shock extremes (~1 per 20 cycles / 2 years)
without discarding normal high-damage cases.

## Cardinal rule

Run this command once to lock Phase 2. Do not re-run to tune τ after seeing
test scores. If test numbers disappoint, that is a Phase 3 problem.

## Realised-spend re-validation (Issue #64)

backtest_phase2.py swept τ ∈ [0.30, 0.55] on the test window via the backtest
engine (preferred stations, 2025-07-01 → 2025-12-31, isotonic-calibrated logreg).

| τ    | CPL (c/L) | vs always-buy |
|------|-----------|---------------|
| 0.30 | 189.35    | +1.27%        |
| 0.35 | 189.61    | +1.13%        |
| 0.40 | 190.35    | +0.74%  ← Phase 2 |
| 0.45 | 190.35    | +0.74%        |
| 0.50 | 190.72    | +0.55%        |
| 0.55 | 191.42    | +0.19%        |

Always-buy baseline: 191.78 c/L.

Spend-optimal τ = 0.30; Phase 2 locked at τ = 0.40.
Gap: 1.01 c/L (≈0.5%). Small but real — the synthetic proxy slightly over-valued
precision, pushing τ higher than the spend-optimal value.

Phase 3 must beat τ=0.40 (190.35 c/L) to show improvement over the locked baseline.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import click
import joblib
import numpy as np
import pandas as pd

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.train_logreg import train_and_evaluate

DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")

# τ grid: [0.05, 0.10, ..., 0.95] — 19 values
_TAU_STEP: float = 0.05
_TAUS: np.ndarray = np.round(np.arange(_TAU_STEP, 1.0, _TAU_STEP), 2)

# Cost model constants — see module docstring.
_THRESHOLD_CENTS: float = 3.0   # label definition: 3c drop = "cheap enough"
_TP_REWARD_CENTS: float = 6.37  # empirical avg saving on correct BUY (95th-pct trimmed mean, tp_benefit.py)
_FP_COST_CENTS: float = 5.80    # population median from fp_cost.py diagnostic
_FN_COST_CENTS: float = 11.14   # 95th-pct trimmed mean from fn_cost.py diagnostic

# Known BUY-rate gap between val and test (from issue #34, real DB 2026-05-07).
# Used only to inform the τ adjustment direction, not to look at test labels.
_VAL_LABEL_RATE: float = 0.361
_TEST_LABEL_RATE: float = 0.269


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _precision_recall_f1(
    y_true: np.ndarray,
    y_hat: np.ndarray,
) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for binary predictions y_hat."""
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def threshold_sweep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    taus: np.ndarray | None = None,
    tp_reward_cents: float = _TP_REWARD_CENTS,
    fp_cost_cents: float = _FP_COST_CENTS,
    fn_cost_cents: float = _FN_COST_CENTS,
) -> list[dict]:
    """Sweep decision thresholds on (y_true, y_pred); return sorted list of metric dicts.

    Each dict contains: tau, buy_rate, precision, recall, f1, expected_cents_per_row,
    tp, fp, fn, tn.

    tp_reward_cents controls TP valuation in expected_cents_per_row. It is distinct from
    the label-definition threshold (3.0c) used in labels.py — see module docstring.

    Monotone invariants (always hold):
    - buy_rate is non-increasing as tau increases
    - recall is non-increasing as tau increases
    """
    if taus is None:
        taus = _TAUS
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"threshold_sweep(): shape mismatch — y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("threshold_sweep() requires non-empty inputs.")
    n = len(y_true)

    rows = []
    for tau in sorted(taus):
        y_hat = (y_pred >= tau).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        buy_rate = float(y_hat.mean())
        precision, recall, f1 = _precision_recall_f1(y_true, y_hat)
        expected_cents = (tp * tp_reward_cents - fp * fp_cost_cents - fn * fn_cost_cents) / n
        rows.append({
            "tau": round(float(tau), 4),
            "buy_rate": round(buy_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "expected_cents_per_row": round(expected_cents, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })
    return rows


def _resolve_tau_adjustment(
    calibration_method: str | None,
    tau_adjustment: float | None,
) -> float:
    """Return the effective τ adjustment given model calibration and any explicit override.

    Isotonic calibration produces a piecewise-constant probability surface (~162
    unique values on a 124k-row test set). A fixed +0.05 step can cross a plateau
    boundary and drop thousands of correctly-classified BUYs into WAIT in a single
    step — the Phase 3b incident documented in memory project_threshold_policy_lesson.md.
    For isotonic models the correct default is 0.0 (use val argmax directly).
    For sigmoid or raw models, the original +0.05 bump remains appropriate.
    An explicit tau_adjustment always takes precedence.
    """
    if tau_adjustment is not None:
        return tau_adjustment
    _KNOWN = {"isotonic", "sigmoid", "raw", None}
    if calibration_method not in _KNOWN:
        raise ValueError(
            f"Unknown calibration_method {calibration_method!r}. "
            f"Expected one of {sorted(str(v) for v in _KNOWN if v is not None) + [None]}."
        )
    return 0.0 if calibration_method == "isotonic" else _TAU_STEP


def pick_tau(
    sweep_rows: list[dict],
    *,
    calibration_method: str | None = None,
    tau_adjustment: float | None = None,
) -> float:
    """Return the chosen τ: argmax(expected_cents_per_row), optionally adjusted.

    When called from the --model-path OOF path (main() in score_phase2), the
    caller already passes tau_adjustment=0.0 (OOF base rate matches deployment,
    no correction needed) or the user's explicit override.  The model-aware
    defaults below are only active for the legacy val-sweep path (no --model-path):
      - calibration_method == 'isotonic': default tau_adjustment = 0.0
        Isotonic calibration is piecewise constant; a fixed step can jump a plateau
        and discard thousands of correct BUYs.  Use val argmax directly instead.
        See memory project_threshold_policy_lesson.md for the Phase 3b diagnosis.
      - sigmoid or raw (None): default tau_adjustment = +_TAU_STEP (0.05)
        Original val-BUY-rate correction — smooth probability surface, fixed step safe.

    An explicit tau_adjustment argument always overrides the model-aware default.
    Result is clamped to [_TAU_STEP, 1.0 - _TAU_STEP].
    """
    if not sweep_rows:
        raise ValueError("pick_tau() requires at least one sweep row.")
    effective_adj = _resolve_tau_adjustment(calibration_method, tau_adjustment)
    best = max(sweep_rows, key=lambda r: r["expected_cents_per_row"])
    adjusted = round(best["tau"] + effective_adj, 4)
    lo, hi = _TAU_STEP, 1.0 - _TAU_STEP
    return float(np.clip(adjusted, lo, hi))


def load_model_artifact(path: pathlib.Path) -> tuple[Any, list[str], str | None]:
    """Load any saved model artifact and return (model, feature_columns, calibration_method).

    calibration_method is the string stored in the artifact (e.g. 'isotonic', 'sigmoid',
    'raw') for calibrated artifacts, or None for raw pipeline artifacts.
    Handles both raw pipeline artifacts and calibrated artifacts produced by calibrate.py.
    Raises ValueError with an actionable message on unexpected artifact shapes.
    """
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and loaded.get("calibrated"):
        required = {"base_pipeline", "calibrator", "calibration_method"}
        missing = sorted(required - set(loaded.keys()))
        if missing:
            raise ValueError(f"Calibrated artifact missing required keys: {missing}")
        from fuel_signal.calibrate import _CalibratedPipeline
        cal_method: str | None = loaded["calibration_method"]
        feature_columns = loaded.get("feature_columns", FEATURE_COLUMNS)
        model = _CalibratedPipeline(
            loaded["base_pipeline"], loaded["calibrator"], cal_method, list(feature_columns)
        )
    elif isinstance(loaded, dict):
        if "pipeline" not in loaded:
            raise ValueError(
                "Unsupported model artifact: dict format with no 'pipeline' key. "
                "Expected artifact saved by train_logreg.py or train_lgbm.py."
            )
        model = loaded["pipeline"]
        feature_columns = loaded.get("feature_columns", FEATURE_COLUMNS)
        cal_method = None
    else:
        model = loaded
        feature_columns = FEATURE_COLUMNS
        cal_method = None

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Loaded artifact ({type(model).__name__}) does not provide predict_proba(). "
            "Pass a fitted sklearn-compatible classifier or a calibrated pipeline artifact."
        )
    return model, list(feature_columns), cal_method


def oof_threshold_predictions(
    artifact_path: pathlib.Path,
    df: pd.DataFrame,
    feature_columns: list[str],
    fold_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_oof, y_oof) for threshold selection via walk-forward CV over train.

    For each fold: clone the base pipeline from the artifact, fit on fold_train,
    predict uncalibrated OOF on fold_val, then apply the calibrator (if any).
    Returned predictions are at training base rate (~0.24) — no τ adjustment needed.

    fold_params (optional): kwargs forwarded to walk_forward_folds, useful in tests.
    """
    from fuel_signal.calibrate import pool_oof_predictions

    loaded = joblib.load(artifact_path)
    calibrated = loaded.get("calibrated", False)
    if calibrated:
        raw_pipe = loaded["base_pipeline"]
        calibrator = loaded["calibrator"]
        cal_method: str | None = loaded["calibration_method"]
    else:
        raw_pipe = loaded["pipeline"]
        calibrator = None
        cal_method = None

    train, _, _ = _ev.split(df)
    p_uncal, y_oof = pool_oof_predictions(raw_pipe, train, feature_columns, fold_params or {})

    if p_uncal.size == 0:
        raise ValueError(
            "oof_threshold_predictions(): no CV folds generated — "
            "train split may be too small for walk_forward_folds defaults. "
            "Pass fold_params={'train_min_days': N, ...} to override."
        )

    if calibrator is not None and cal_method == "sigmoid":
        p_oof = calibrator.predict_proba(p_uncal.reshape(-1, 1))[:, 1]
    elif calibrator is not None:  # isotonic
        p_oof = np.clip(calibrator.predict(p_uncal), 0.0, 1.0)
    else:
        p_oof = p_uncal

    return p_oof, y_oof


def score_test(
    pipeline: Any,
    df: pd.DataFrame,
    tau: float,
    feature_columns: list[str] | None = None,
) -> dict:
    """Score the fitted pipeline on the test split at threshold tau.

    Returns dict with: test_size, test_positive_rate, test_logloss, test_brier,
    test_precision, test_recall, test_f1, test_buy_rate, y_test, p_test.

    This function reads test labels — only call it once, after tau is locked.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    _, _, test = _ev.split(df)
    if test.empty:
        raise ValueError("score_test(): test split is empty.")

    X_test = test[feature_columns]
    y_test = test["label"].to_numpy(dtype=int)
    p_test = pipeline.predict_proba(X_test)[:, 1]

    test_logloss = _ev.log_loss(y_test, p_test)
    test_brier = _ev.brier(y_test, p_test)

    y_hat = (p_test >= tau).astype(int)
    precision, recall, f1 = _precision_recall_f1(y_test, y_hat)

    return {
        "test_size": int(len(test)),
        "test_positive_rate": float(y_test.mean()),
        "test_logloss": float(test_logloss),
        "test_brier": float(test_brier),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_buy_rate": float(y_hat.mean()),
        "y_test": y_test,
        "p_test": p_test,
    }


def multi_seed_raw_logloss(
    df: pd.DataFrame,
    feature_columns: list[str],
    seeds: list[int],
) -> dict:
    """Retrain a raw LightGBM at each seed; return per-seed test-logloss vector + stats.

    Metric: raw (uncalibrated) test logloss — avoids the calibration confound where
    the calibrator is fit on the higher-BUY-rate val split, degrading test-set scores.
    Policy: call only at lock time with a standard seed set (e.g. {1,7,42,99,2024}).
    Do not multi-seed every experiment — that defeats the 3×std comparison gate.

    Returns dict with:
      logloss_vector: list[float]
      logloss_mean: float
      logloss_std: float (population std, ddof=0)
    """
    from fuel_signal.train_lgbm import train_and_evaluate as _lgbm_train

    _, _, test = _ev.split(df)
    if test.empty:
        raise ValueError("multi_seed_raw_logloss(): test split is empty.")

    X_test = test[feature_columns]
    y_test = test["label"].to_numpy(dtype=int)

    logloss_vector: list[float] = []
    for seed in seeds:
        result = _lgbm_train(df, feature_columns=feature_columns, random_state=seed)
        p_test = result["pipeline"].predict_proba(X_test)[:, 1]
        logloss_vector.append(float(_ev.log_loss(y_test, p_test)))

    vec = np.array(logloss_vector)
    return {
        "logloss_vector": logloss_vector,
        "logloss_mean": float(vec.mean()),
        "logloss_std": float(vec.std()),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_sweep_table(sweep_rows: list[dict]) -> str:
    header = f"{'τ':>6}  {'BUY%':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'c/row':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in sweep_rows:
        lines.append(
            f"{r['tau']:6.2f}  {r['buy_rate']:6.3f}  {r['precision']:6.3f}"
            f"  {r['recall']:6.3f}  {r['f1']:6.3f}  {r['expected_cents_per_row']:8.4f}"
        )
    return "\n".join(lines)


def _format_comparison(
    val_logloss: float,
    val_positive_rate: float,
    test_result: dict,
    baseline_test_logloss: float,
    baseline_test_brier: float,
    tau: float,
    model_label: str = "Logreg",
) -> str:
    delta_ll = test_result["test_logloss"] - baseline_test_logloss
    delta_br = test_result["test_brier"] - baseline_test_brier
    lines = [
        "",
        f"{model_label} — test split results",
        f"  Chosen τ               : {tau:.2f}",
        f"  Test rows              : {test_result['test_size']:>8,}"
        f"  (pos rate {test_result['test_positive_rate']:.3f})",
        f"  Val  logloss           : {val_logloss:.4f}  (val pos rate {val_positive_rate:.3f})",
        "",
        f"  Baseline test logloss  : {baseline_test_logloss:.4f}",
        f"  {model_label:<8} test logloss   : {test_result['test_logloss']:.4f}  (Δ {delta_ll:+.4f})",
        "",
        f"  Baseline test brier    : {baseline_test_brier:.4f}",
        f"  {model_label:<8} test brier     : {test_result['test_brier']:.4f}  (Δ {delta_br:+.4f})",
        "",
        f"  At τ={tau:.2f}: P={test_result['test_precision']:.3f}  R={test_result['test_recall']:.3f}"
        f"  F1={test_result['test_f1']:.3f}  BUY%={test_result['test_buy_rate']:.3f}",
    ]
    if delta_ll >= 0:
        lines.append("\nWARNING: test log-loss did not beat the baseline — do not retune τ.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Realised-spend backtest (Phase 2 logreg + Phase 4 LGBM share this path)
# ---------------------------------------------------------------------------


def run_realised_spend_backtest(
    *,
    db_path: str,
    model_path: str | None,
    chosen_tau: float,
    no_backtest: bool,
) -> tuple[float | None, float | None]:
    """Return (realised_cpl, realised_savings_pct) for the chosen τ, or (None, None).

    Backtest is skipped — without raising — when:
      - --no-backtest is set
      - --model-path is absent (Phase 2 retraining-from-CSV path)
      - the DB file does not exist (CI, dev sniff-tests)
      - either CPL aggregate comes back NaN (no price data in the test window)
    """
    if no_backtest:
        click.echo("\nSkipping realised-spend backtest (--no-backtest).")
        return None, None
    if model_path is None:
        click.echo("\nSkipping realised-spend backtest (no --model-path).")
        return None, None
    db_file = pathlib.Path(db_path)
    if not db_file.exists():
        click.echo(f"\nSkipping realised-spend backtest (DB not found: {db_path}).")
        return None, None

    import fuel_signal.db as _db
    from fuel_signal.backtest import (
        AlwaysBuyStrategy,
        ModelStrategy,
        TankParams,
        _evaluation_dates,
        load_history,
    )
    from fuel_signal.backtest_phase2 import aggregate_backtest
    from fuel_signal.config import PREFERRED_STATIONS

    tank = TankParams()
    bt_start, bt_end = _ev.TEST_START, _ev.TEST_END
    eval_dates = _evaluation_dates(bt_start, bt_end, tank.evaluation_interval_days)

    conn = _db.open_db(db_file)
    try:
        station_codes = list(PREFERRED_STATIONS.keys())
        history = load_history(conn, station_codes, eval_dates=eval_dates)
    finally:
        conn.close()
    always_agg = aggregate_backtest(
        history, AlwaysBuyStrategy(), station_codes, bt_start, bt_end, tank
    )
    model_agg = aggregate_backtest(
        history,
        ModelStrategy(model_path=pathlib.Path(model_path), threshold=chosen_tau),
        station_codes, bt_start, bt_end, tank,
    )
    always_cpl = always_agg["cpl"]
    model_cpl = model_agg["cpl"]
    if math.isnan(model_cpl) or math.isnan(always_cpl):
        click.echo(
            "\nWARNING: Backtest CPL is NaN — realised-spend columns not populated."
        )
        return None, None

    savings_pct = (
        (always_cpl - model_cpl) / always_cpl * 100
    ) if always_cpl > 0 else float("nan")
    click.echo(f"\nBacktest (τ={chosen_tau:.2f}, {bt_start} → {bt_end}):")
    click.echo(f"  Always-buy CPL : {always_cpl:.2f} c/L")
    click.echo(
        f"  Model      CPL : {model_cpl:.2f} c/L"
        f"  ({savings_pct:+.2f}% vs always-buy)"
    )
    return model_cpl, savings_pct


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("score_phase2")
@click.option(
    "--features-csv",
    "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Path to feature rows CSV produced by `python -m fuel_signal.features`.",
)
@click.option(
    "--model-path",
    "model_path",
    default="data/models/lgbm_calibrated.joblib",
    show_default=True,
    help=(
        "Path to a pre-trained (optionally calibrated) model joblib artifact. "
        "Default: data/models/lgbm_calibrated.joblib (Phase 4 LGBM)."
    ),
)
@click.option(
    "--model-name",
    "model_name",
    default="lgbm_cycle_features",
    show_default=True,
    help="Experiment name written to results.csv (e.g. 'lgbm_cycle_features').",
)
@click.option(
    "--tau-adjustment",
    "tau_adjustment",
    default=None,
    type=float,
    help=(
        "Override the τ adjustment applied to the sweep argmax. "
        "With --model-path (OOF sweep): default 0.0 — OOF base rate already matches "
        "deployment, so no correction is needed; an explicit value still applies. "
        "Without --model-path (val sweep, legacy logreg path): model-aware default "
        f"(0.0 for isotonic, +{_TAU_STEP} for sigmoid/raw)."
    ),
)
@click.option(
    "--seeds",
    "seeds_str",
    default=None,
    help=(
        "Comma-separated list of random seeds for multi-seed raw test-logloss banking "
        "(e.g. '1,7,42,99,2024'). Requires --model-path. "
        "Policy: use only at lock time — not for development sniff-tests."
    ),
)
@click.option(
    "--db", "db_path",
    default="fuel_signal.db",
    show_default=True,
    help=(
        "Path to SQLite DB used by the realised-spend backtest. Combined with --model-path, "
        "runs the backtest at the chosen τ over the test window and populates "
        "realised_spend_cpl / realised_savings_vs_always_buy_pct in results.csv. "
        "Skipped silently when the DB file or --model-path are absent (e.g. CI, dev sniff-tests)."
    ),
)
@click.option(
    "--no-backtest",
    is_flag=True,
    default=False,
    help="Skip the realised-spend backtest even when the DB and model are available.",
)
@click.option(
    "--skip-results-csv",
    "skip_results_csv",
    is_flag=True,
    default=False,
    help=(
        "Do not append a row to experiments/results.csv. Use for measurement / "
        "decision runs (e.g. the #254 realised-backtest) where the stdout backtest "
        "figures are read directly and no lock row is wanted. Mirrors calibrate.py."
    ),
)
def main(
    features_csv: str,
    model_path: str | None,
    model_name: str,
    tau_adjustment: float | None,
    seeds_str: str | None,
    db_path: str,
    no_backtest: bool,
    skip_results_csv: bool,
) -> None:
    """Threshold sweep on val, one-time test scoring, append to results.csv.

    Without --model-path: re-trains the logreg pipeline (Phase 2 mode).
    With --model-path: loads any pre-trained/calibrated artifact and scores it,
    enabling Phase 3+ models (e.g. LightGBM) to reuse this evaluation harness.

    Do not re-run to tune τ after seeing test results — cardinal rule.
    """
    features_path = pathlib.Path(features_csv)
    if not features_path.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )

    # Parse --seeds before any heavy work so errors surface immediately.
    seeds: list[int] | None = None
    if seeds_str is not None:
        try:
            seeds = [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
        except ValueError:
            raise click.ClickException(
                f"--seeds must be a comma-separated list of integers "
                f"(e.g. '1,7,42,99,2024'). Got: {seeds_str!r}"
            )
        if not seeds:
            raise click.ClickException("--seeds must contain at least one seed.")

    df = pd.read_csv(features_path)
    missing = [c for c in FEATURE_COLUMNS + ["label", "price_date"] if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"Features CSV is missing required columns: {missing}. "
            "Re-run 'uv run python -m fuel_signal.features' to regenerate."
        )

    # Step 1: obtain model + threshold-selection predictions.
    if model_path is not None:
        artifact_path = pathlib.Path(model_path)
        if not artifact_path.exists():
            raise click.ClickException(
                f"Model artifact not found: {model_path}. "
                "Run calibrate.py (or train_lgbm.py) first."
            )
        click.echo(f"Loading pre-trained model from {model_path} …")
        pipeline, feature_columns, calibration_method = load_model_artifact(artifact_path)

        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            raise click.ClickException(
                f"Features CSV is missing columns required by the model artifact: {missing_cols}. "
                "Re-run 'uv run python -m fuel_signal.features' to regenerate."
            )

        train, val, _ = _ev.split(df)
        p_baseline = _ev.baseline_prior(train)
        train_positive_rate = float(train["label"].mean())
        train_size = len(train)

        # OOF-based threshold selection: base rate ~0.24, no adjustment needed.
        click.echo("Running walk-forward CV over train for OOF threshold selection …")
        p_thresh, y_thresh = oof_threshold_predictions(artifact_path, df, feature_columns)
        oof_positive_rate = float(y_thresh.mean())

        # Val metrics reported for reference only — not used for τ selection.
        X_val = val[feature_columns]
        y_val = val["label"].to_numpy(dtype=int)
        p_val = pipeline.predict_proba(X_val)[:, 1]
        val_logloss = _ev.log_loss(y_val, p_val)
        val_positive_rate = float(y_val.mean())
        val_size = len(val)
        baseline_val_logloss = _ev.log_loss(y_val, np.full(len(y_val), p_baseline))
    else:
        click.echo("Training logreg on train split …")
        result = train_and_evaluate(df)
        pipeline = result["pipeline"]
        feature_columns = result["feature_columns"]
        y_val = result["y_val"]
        p_val = result["p_val"]
        val_logloss = result["val_logloss"]
        val_positive_rate = result["val_positive_rate"]
        train_positive_rate = result["train_positive_rate"]
        baseline_val_logloss = result["baseline_val_logloss"]
        train_size = result["train_size"]
        val_size = result["val_size"]
        p_baseline = result["baseline_prior"]
        calibration_method = None

    click.echo(f"  Train: {train_size:,} rows  (pos rate {train_positive_rate:.3f})")
    click.echo(f"  Val:   {val_size:,} rows  (pos rate {val_positive_rate:.3f})")
    click.echo(f"  Val logloss: {val_logloss:.4f}  (baseline {baseline_val_logloss:.4f})")

    # Step 2: threshold sweep.
    if model_path is not None:
        # OOF sweep — base rate matches deployment, default adj=0.0.
        # An explicit --tau-adjustment always overrides the default.
        click.echo(f"\nThreshold sweep on OOF ({len(y_thresh):,} rows, pos rate {oof_positive_rate:.3f}):")
        sweep = threshold_sweep(y_thresh, p_thresh)
        click.echo(_format_sweep_table(sweep))
        effective_adj = tau_adjustment if tau_adjustment is not None else 0.0
        sweep_source = f"OOF (BUY {oof_positive_rate:.3f})"
        sweep_key = "OOF_cv"  # compact token for results.csv criterion field
    else:
        # Legacy val sweep for the no-model-path logreg retraining path.
        click.echo("\nThreshold sweep on val:")
        sweep = threshold_sweep(y_val, p_val)
        click.echo(_format_sweep_table(sweep))
        effective_adj = _resolve_tau_adjustment(calibration_method, tau_adjustment)
        sweep_source = f"val (BUY {val_positive_rate:.3f})"
        sweep_key = "val"

    # Step 3: pick τ.
    _, _, test_df = _ev.split(df)
    test_label_rate = float(test_df["label"].mean())
    chosen_tau = pick_tau(sweep, calibration_method=calibration_method, tau_adjustment=effective_adj)
    best_row = max(sweep, key=lambda r: r["expected_cents_per_row"])
    click.echo(f"\nChosen τ = {chosen_tau:.2f}")
    click.echo(
        f"  Basis: argmax(expected_cents_per_row) on {sweep_source} → τ={best_row['tau']:.2f}"
        f" ({best_row['expected_cents_per_row']:.4f} c/row)"
    )
    if effective_adj != 0.0:
        click.echo(
            f"  Adjusted {effective_adj:+.2f} for BUY-rate gap "
            f"({val_positive_rate:.3f} vs {test_label_rate:.3f})"
            + (f"  [calibration={calibration_method}]" if calibration_method else "")
        )

    # Step 4: score test once at chosen τ.
    click.echo("\nScoring test split …")
    test_result = score_test(pipeline, df, chosen_tau, feature_columns)

    # Baseline constant predictor on test.
    n_test = test_result["test_size"]
    baseline_test_logloss = _ev.log_loss(
        test_result["y_test"], np.full(n_test, p_baseline)
    )
    baseline_test_brier = _ev.brier(
        test_result["y_test"], np.full(n_test, p_baseline)
    )

    model_label = model_name.split("_")[0].capitalize()
    click.echo(
        _format_comparison(
            val_logloss, val_positive_rate,
            test_result, baseline_test_logloss, baseline_test_brier,
            chosen_tau, model_label=model_label,
        )
    )

    # Step 5: val metrics at chosen τ (for notes).
    y_hat_val = (p_val >= chosen_tau).astype(int)
    val_p, val_r, val_f1 = _precision_recall_f1(y_val, y_hat_val)

    # Step 6 (optional): multi-seed raw test-logloss banking.
    seed_result: dict | None = None
    if seeds is not None:
        click.echo(f"\nComputing multi-seed raw test-logloss for seeds={seeds} …")
        seed_result = multi_seed_raw_logloss(df, feature_columns, seeds)
        click.echo(
            f"  Per-seed vector: {[f'{v:.4f}' for v in seed_result['logloss_vector']]}"
        )
        click.echo(
            f"  Mean: {seed_result['logloss_mean']:.4f}  "
            f"Std: {seed_result['logloss_std']:.4f}  "
            f"(3σ = {3 * seed_result['logloss_std']:.4f})"
        )

    # Step 7: run backtest for realised-spend columns (default on; gated by DB + model presence).
    realised_cpl, realised_savings_pct = run_realised_spend_backtest(
        db_path=db_path,
        model_path=model_path,
        chosen_tau=chosen_tau,
        no_backtest=no_backtest,
    )

    # Step 8: log to results.csv.
    sweep_rate = oof_positive_rate if model_path is not None else val_positive_rate
    notes = (
        f"tau={chosen_tau:.2f}; "
        f"criterion=max_expected_cents_{sweep_key}_adj{effective_adj:+.2f}; "
        f"cost_model=TP+{_TP_REWARD_CENTS}c_FP-{_FP_COST_CENTS}c_FN-{_FN_COST_CENTS}c; "
        f"val_logloss={val_logloss:.4f}; "
        f"test_logloss={test_result['test_logloss']:.4f}; "
        f"sweep_BUY_rate={sweep_rate:.3f}; "
        f"test_BUY_rate={test_label_rate:.3f}; "
        f"val_P={val_p:.3f}/R={val_r:.3f}/F1={val_f1:.3f}; "
        f"test_P={test_result['test_precision']:.3f}"
        f"/R={test_result['test_recall']:.3f}"
        f"/F1={test_result['test_f1']:.3f}"
    )
    if skip_results_csv:
        click.echo(
            "\nSkipping experiments/results.csv append (--skip-results-csv). "
            "Read the realised-spend backtest figures from the output above."
        )
    else:
        _ev.log_experiment(
            name=model_name,
            features=feature_columns,
            holdout_logloss=test_result["test_logloss"],
            holdout_brier=test_result["test_brier"],
            notes=notes,
            realised_spend_cpl=realised_cpl,
            realised_savings_vs_always_buy_pct=realised_savings_pct,
            seed_test_logloss_vector=seed_result["logloss_vector"] if seed_result else None,
            seed_test_logloss_mean=seed_result["logloss_mean"] if seed_result else None,
            seed_test_logloss_std=seed_result["logloss_std"] if seed_result else None,
        )
        click.echo(f"\nAppended result to experiments/results.csv  (name={model_name})")
    click.echo(
        "\nNext step: update SHAP artifacts to match this model run:\n"
        "  uv run python -m fuel_signal.shap_report \\\n"
        "      --model data/models/lgbm.joblib \\\n"
        "      --features data/features.csv \\\n"
        "      --output experiments/shap_phase4"
    )


if __name__ == "__main__":
    main()
