"""Phase 2 final evaluation: threshold sweep on val, one-time test scoring.

Issues #37 / #34. Train the canonical logreg pipeline on the train split, sweep
decision thresholds on val to pick τ, then score test exactly once and append
the result to experiments/results.csv.

## Threshold-selection criterion

Criterion: highest expected-cents-per-row on val, adjusted upward by one τ step
(+0.05) to account for val's elevated BUY rate (36.1%) vs test (26.9%). Without
the adjustment, the cost-optimal τ on val would be too aggressive (too many BUYs)
when applied to the test distribution, because val's 90-day lookback happens to
anchor against a high-price Dec 2024–Feb 2025 reference period, making March 2025
trough days look definitively cheap. See issue #34 for the full diagnosis.

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
    """Return the chosen τ: argmax(expected_cents_per_row) on val, model-aware adjusted.

    The adjustment accounts for val's elevated BUY rate vs test (#34). The default
    adjustment depends on the model's calibration scheme:
      - calibration_method == 'isotonic': default tau_adjustment = 0.0
        Isotonic calibration is piecewise constant; a fixed step can jump a plateau
        and discard thousands of correct BUYs. Use val argmax directly instead.
        See memory project_threshold_policy_lesson.md for the Phase 3b diagnosis.
      - sigmoid or raw (None): default tau_adjustment = +_TAU_STEP (0.05)
        Original behaviour — smooth probability surface makes a fixed step safe.

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
        model = _CalibratedPipeline(
            loaded["base_pipeline"], loaded["calibrator"], cal_method
        )
        feature_columns = loaded.get("feature_columns", FEATURE_COLUMNS)
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

    X_test = test[feature_columns].to_numpy(dtype=float)
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

    X_test = test[feature_columns].to_numpy(dtype=float)
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
        "Default: data/models/lgbm_calibrated.joblib (Phase 4 LGBM). "
        "Pass None explicitly to re-train logreg from scratch (Phase 2 mode)."
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
        "Override the τ adjustment applied to the val argmax. "
        "Default is model-aware: 0.0 for isotonic-calibrated models, "
        f"+{_TAU_STEP} for sigmoid or raw models."
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
def main(
    features_csv: str,
    model_path: str | None,
    model_name: str,
    tau_adjustment: float | None,
    seeds_str: str | None,
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

    # Step 1: obtain model + val predictions.
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
        X_val = val[feature_columns].to_numpy(dtype=float)
        y_val = val["label"].to_numpy(dtype=int)
        p_val = pipeline.predict_proba(X_val)[:, 1]

        p_baseline = _ev.baseline_prior(train)
        baseline_val_logloss = _ev.log_loss(y_val, np.full(len(y_val), p_baseline))
        val_logloss = _ev.log_loss(y_val, p_val)
        val_positive_rate = float(y_val.mean())
        train_positive_rate = float(train["label"].mean())
        train_size = len(train)
        val_size = len(val)
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

    # Step 2: threshold sweep on val.
    click.echo("\nThreshold sweep on val:")
    sweep = threshold_sweep(y_val, p_val)
    click.echo(_format_sweep_table(sweep))

    # Step 3: pick τ.
    _, _, test_df = _ev.split(df)
    test_label_rate = float(test_df["label"].mean())
    chosen_tau = pick_tau(sweep, calibration_method=calibration_method, tau_adjustment=tau_adjustment)
    effective_adj = _resolve_tau_adjustment(calibration_method, tau_adjustment)
    best_row = max(sweep, key=lambda r: r["expected_cents_per_row"])
    click.echo(f"\nChosen τ = {chosen_tau:.2f}")
    click.echo(
        f"  Basis: argmax(expected_cents_per_row) on val → τ={best_row['tau']:.2f}"
        f" ({best_row['expected_cents_per_row']:.4f} c/row)"
    )
    click.echo(
        f"  Adjusted {effective_adj:+.2f} for val/test BUY-rate gap "
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

    # Step 7: log to results.csv.
    notes = (
        f"tau={chosen_tau:.2f}; "
        f"criterion=max_expected_cents_val_adj{effective_adj:+.2f}; "
        f"cost_model=TP+{_TP_REWARD_CENTS}c_FP-{_FP_COST_CENTS}c_FN-{_FN_COST_CENTS}c; "
        f"val_logloss={val_logloss:.4f}; "
        f"test_logloss={test_result['test_logloss']:.4f}; "
        f"val_BUY_rate={val_positive_rate:.3f}; "
        f"test_BUY_rate={test_label_rate:.3f}; "
        f"val_P={val_p:.3f}/R={val_r:.3f}/F1={val_f1:.3f}; "
        f"test_P={test_result['test_precision']:.3f}"
        f"/R={test_result['test_recall']:.3f}"
        f"/F1={test_result['test_f1']:.3f}"
    )
    _ev.log_experiment(
        name=model_name,
        features=feature_columns,
        holdout_logloss=test_result["test_logloss"],
        holdout_brier=test_result["test_brier"],
        notes=notes,
        seed_test_logloss_vector=seed_result["logloss_vector"] if seed_result else None,
        seed_test_logloss_mean=seed_result["logloss_mean"] if seed_result else None,
        seed_test_logloss_std=seed_result["logloss_std"] if seed_result else None,
    )
    click.echo(f"\nAppended result to experiments/results.csv  (name={model_name})")


if __name__ == "__main__":
    main()
