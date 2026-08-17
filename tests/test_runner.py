"""Tests for experiments/pipeline/runner.py — the one-shot candidate runner.

The DB-backed realised backtest (run_paired_realised_backtest) is exercised by a
manual smoke against the real DB, same as experiments/lib/realised.py itself
(see tests/test_realised.py's docstring) — not reproduced here. Everything else
(validation/environment short-circuits, the WFCV screen, the lookup provider,
effect/zone grading, artifact writing) is unit-tested directly, since none of it
needs a real database.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
import pandas as pd
import pytest

from experiments.pipeline.batch_freeze import resolve_baseline_columns
from experiments.pipeline.runner import (
    STATUS_ABORTED_CANDIDATE,
    STATUS_ABORTED_ENVIRONMENT,
    STATUS_DISQUALIFIED,
    RunResult,
    _build_axis_lookup,
    _check_provider_health,
    _finish,
    _grade_run,
    _make_lookup_provider,
    _read_pass_criterion,
    _resolve_effect,
    _resolve_zone,
    _run_wfcv_screen,
    _summarise_for_comment,
    post_bd_comment,
    record_pass_criterion,
    run_candidate,
)
from experiments.pipeline.validate import load_candidate_module, validate_candidate
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, NETWORK_FEATURE_COLUMNS

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


# ── shared fixtures ──────────────────────────────────────────────────────────

def _baseline_features_df(n: int = 5) -> pd.DataFrame:
    dates = [20260810, 20260811, 20260812, 20260813, 20260814][:n]
    cols = {"price_date": dates}
    for c in FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS:
        cols[c] = [0.0] * n
    return pd.DataFrame(cols)


def _write_batch_dir(tmp_path: pathlib.Path, df: pd.DataFrame) -> pathlib.Path:
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()
    df.to_parquet(batch_dir / "features.parquet")
    (batch_dir / "baseline_columns.json").write_text(json.dumps(resolve_baseline_columns(df)))
    (batch_dir / "fuel_signal.db").write_bytes(b"not a real sqlite file")
    return batch_dir


def _write_candidate(tmp_path: pathlib.Path, body: str, name: str = "cand.py") -> pathlib.Path:
    path = tmp_path / name
    path.write_text(body)
    return path


PIT_SAFE_CANDIDATE = '''
NAME = "cand"
INPUTS = ["price_date"]
COLUMNS = ["cand_col"]
def add_columns(df):
    out = df.copy()
    out["cand_col"] = df["price_date"] % 2
    return out
'''

LEAKY_CANDIDATE = '''
import pandas as pd
NAME = "cand"
INPUTS = ["price_date"]
COLUMNS = ["cand_col"]
def add_columns(df):
    out = df.copy()
    out["cand_col"] = df["price_date"].mean()
    return out
'''

ALL_NAN_CANDIDATE = '''
NAME = "cand"
INPUTS = ["price_date"]
COLUMNS = ["cand_col"]
def add_columns(df):
    out = df.copy()
    out["cand_col"] = float("nan")
    return out
'''

OVERLAP_CANDIDATE = f'''
NAME = "cand"
INPUTS = ["price_date"]
COLUMNS = {FEATURE_COLUMNS[:1]!r}
def add_columns(df):
    out = df.copy()
    out[{FEATURE_COLUMNS[0]!r}] = df["price_date"] % 2
    return out
'''

RAISING_ADD_COLUMNS_CANDIDATE = '''
NAME = "cand"
INPUTS = ["price_date"]
COLUMNS = ["cand_col"]
def add_columns(df):
    raise ValueError("candidate bug: deliberately broken add_columns")
'''

RAISING_ADD_AXIS_CANDIDATE = '''
NAME = "cand"
INPUTS = ["price_date"]
COLUMNS = ["cand_col"]
def add_columns(df):
    out = df.copy()
    out["cand_col"] = df["price_date"] % 2
    return out
def add_axis(df):
    raise ValueError("candidate bug: deliberately broken add_axis")
'''


# ── run_candidate short-circuits (no DB access reached) ─────────────────────

def test_run_candidate_aborted_candidate_on_import_error(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, "raise RuntimeError('boom')\n")

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_ABORTED_CANDIDATE
    assert result.results_path.exists()
    assert json.loads(result.results_path.read_text())["status"] == STATUS_ABORTED_CANDIDATE


def test_run_candidate_aborted_environment_on_baseline_drift(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    # Drift the frozen frame's baseline columns after freezing.
    drifted = df.copy()
    drifted["days_since_trough_entry_new_brand"] = 0.0
    drifted.to_parquet(batch_dir / "features.parquet")
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_CANDIDATE)

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_ABORTED_ENVIRONMENT
    assert "baseline" in result.error.lower() or "contract" in result.error.lower()


def test_run_candidate_disqualified_on_pit_leak(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, LEAKY_CANDIDATE)

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_DISQUALIFIED


def test_run_candidate_aborted_candidate_on_all_nan_output(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, ALL_NAN_CANDIDATE)

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_ABORTED_CANDIDATE


def test_run_candidate_aborted_candidate_on_columns_overlap_baseline(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, OVERLAP_CANDIDATE)

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_ABORTED_CANDIDATE
    assert "overlaps" in result.error


def test_run_candidate_aborted_candidate_on_add_columns_non_pit_exception(tmp_path):
    """A candidate raising ValueError (not KeyError/PitLeakError) must still
    resolve to a status code, not an uncaught traceback with no results.json
    (fps-hvi finding #2)."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, RAISING_ADD_COLUMNS_CANDIDATE)

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_ABORTED_CANDIDATE
    assert result.results_path.exists()
    assert "candidate bug" in result.error


def test_run_candidate_aborted_candidate_on_add_axis_non_pit_exception(tmp_path):
    """Same as above but for add_axis (fps-hvi finding #2)."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, RAISING_ADD_AXIS_CANDIDATE)

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_ABORTED_CANDIDATE
    assert result.results_path.exists()
    assert "candidate bug" in result.error


# ── axis lookup (fps-hvi finding #1: must align by index label, not position) ─

def test_build_axis_lookup_aligns_by_index_when_axis_series_is_reordered():
    frame = pd.DataFrame({
        "station_code": [10, 20, 30, 40],
        "price_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
    })
    # A PIT-safe candidate is allowed to return add_axis's Series in a
    # different row order than frame (pit_test.py's docstring blesses
    # sort_values-style transforms) as long as the index labels are preserved.
    axis_series = pd.Series(
        ["D", "C", "B", "A"], index=[3, 2, 1, 0], name="axis",
    )  # reverse order, same index labels

    lookup = _build_axis_lookup(frame, axis_series)

    # Row 0 (station 10, 2026-01-01) must get axis_series's label at index 0
    # ("A"), not the label at its own row position 0 in axis_series ("D").
    by_station = lookup.set_index("station_code")["axis"]
    assert by_station.loc[10] == "A"
    assert by_station.loc[20] == "B"
    assert by_station.loc[30] == "C"
    assert by_station.loc[40] == "D"


def test_build_axis_lookup_matches_unreordered_axis_series():
    frame = pd.DataFrame({
        "station_code": [10, 20],
        "price_date": ["2026-01-01", "2026-01-02"],
    })
    axis_series = pd.Series(["A", "B"], index=[0, 1], name="axis")

    lookup = _build_axis_lookup(frame, axis_series)

    assert lookup["axis"].tolist() == ["A", "B"]


# ── provider health check (fps-hvi finding #6) ────────────────────────────────

def test_check_provider_health_flags_total_miss():
    frame = pd.DataFrame({
        "station_code": [1], "price_date": ["2026-01-01"], "cand_col": [1.0],
    })
    provider = _make_lookup_provider(frame, ["cand_col"])
    provider("2026-02-01", 1, 100.0)  # miss: wrong date
    provider("2026-03-01", 1, 100.0)  # miss

    error = _check_provider_health(provider)
    assert error is not None
    assert "2" in error  # miss count


def test_check_provider_health_passes_with_any_hit():
    frame = pd.DataFrame({
        "station_code": [1], "price_date": ["2026-01-01"], "cand_col": [1.0],
    })
    provider = _make_lookup_provider(frame, ["cand_col"])
    provider("2026-01-01", 1, 100.0)  # hit
    provider("2026-02-01", 1, 100.0)  # miss

    assert _check_provider_health(provider) is None


def test_check_provider_health_passes_with_no_calls():
    frame = pd.DataFrame({
        "station_code": [1], "price_date": ["2026-01-01"], "cand_col": [1.0],
    })
    provider = _make_lookup_provider(frame, ["cand_col"])
    assert _check_provider_health(provider) is None


# ── grading isolation (fps-hvi finding #7) ────────────────────────────────────

class _FakeRealised:
    def __init__(self, aggregate, fills):
        self.aggregate = aggregate
        self.fills = fills


def test_grade_run_succeeds_normally():
    aggregate = pd.DataFrame([
        {"arm": "R0", "cpl_held": 200.0},
        {"arm": "candidate", "cpl_held": 195.0},
    ])
    realised = _FakeRealised(aggregate, pd.DataFrame())

    effect_resolved, effect_delta, zone, grading_error = _grade_run(
        realised, target=None, axis_lookup=None, min_row_cell_n=30
    )

    assert effect_resolved is True
    assert effect_delta == pytest.approx(-5.0)
    assert zone == {"resolved": None, "reason": "candidate declared no TARGET"}
    assert grading_error is None


def test_grade_run_catches_exception_and_reports_it_instead_of_raising():
    """A grading bug (e.g. a malformed aggregate frame) must not raise past
    already-written artifacts — it should be recorded, not propagated
    (fps-hvi finding #7)."""
    aggregate = pd.DataFrame([{"arm": "R0"}])  # missing 'cpl_held' -> KeyError inside _resolve_effect
    realised = _FakeRealised(aggregate, pd.DataFrame())

    effect_resolved, effect_delta, zone, grading_error = _grade_run(
        realised, target=None, axis_lookup=None, min_row_cell_n=30
    )

    assert effect_resolved is None
    assert effect_delta is None
    assert zone["resolved"] is None
    assert grading_error is not None
    assert "grading raised" in zone["reason"]


# ── the WFCV screen (real small LightGBM fits, no DB) ────────────────────────

def _synth_panel(n_days: int = 200, n_stations: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n_days, freq="D").strftime("%Y-%m-%d")
    rows = []
    for station in range(n_stations):
        f1 = rng.normal(size=n_days)
        f2 = rng.normal(size=n_days)
        cand = rng.normal(size=n_days)
        p = 1.0 / (1.0 + np.exp(-(0.9 * f1 - 0.5 * f2)))
        label = (rng.uniform(size=n_days) < p).astype(int)
        for i, d in enumerate(dates):
            rows.append({
                "price_date": d, "station_code": station,
                "label": int(label[i]), "f1": f1[i], "f2": f2[i], "cand": cand[i],
            })
    return pd.DataFrame(rows)


def test_run_wfcv_screen_produces_rows_for_both_runs_and_seeds_and_axis(tmp_path):
    frame = _synth_panel(n_days=150, n_stations=2)
    axis_series = (frame["station_code"] == 0).rename("axis").replace({True: "A", False: "B"})
    outer = {"train_min_days": 60, "val_days": 30, "step_days": 30}

    df_rows, collector = _run_wfcv_screen(
        frame, ["f1", "f2"], ["f1", "f2", "cand"],
        seeds=(1, 2), axis_series=axis_series, outer_fold_params=outer, verbose=False,
    )

    assert set(df_rows["run"]) == {"R0", "candidate"}
    assert set(df_rows["seed"]) == {1, 2}
    assert len(df_rows) > 0

    combined = collector.to_parquet(tmp_path / "rowpreds.parquet")
    assert "axis" in combined.columns
    assert set(combined["axis"].unique()) <= {"A", "B"}


# ── run_candidate reaching the realised backtest (fps-hvi findings #3, #4) ────

def _full_baseline_df(n_days: int = 90, n_stations: int = 2, seed: int = 3) -> pd.DataFrame:
    """A baseline-contract-compliant frame (every FEATURE_COLUMNS/LGA/NETWORK
    column present) big enough for real (small) WFCV folds — for tests where
    run_candidate must reach _run_wfcv_screen / run_paired_realised_backtest.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n_days, freq="D").strftime("%Y-%m-%d")
    rows = []
    for station in range(n_stations):
        for i, d in enumerate(dates):
            row = {"price_date": d, "station_code": station, "label": int(rng.uniform() < 0.5)}
            for c in FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS:
                row[c] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


PIT_SAFE_STRING_DATE_CANDIDATE = '''
NAME = "cand"
INPUTS = ["price_date"]
COLUMNS = ["cand_col"]
def add_columns(df):
    out = df.copy()
    out["cand_col"] = df["price_date"].str[-2:].astype(int) % 2
    return out
'''


def test_run_candidate_realised_config_error_maps_to_aborted_candidate(tmp_path):
    """fold_subset excluding every fold makes run_paired_realised_backtest
    raise ValueError('no folds planned') BEFORE touching the DB — a
    deterministic config error, not a DB/environment failure (fps-hvi finding
    #3: these must not both map to aborted_environment). Also exercises a
    single-seed run (fps-hvi finding #4): with seeds=(1,), the seed-variance
    gate must be skipped rather than raising its own aborted_candidate first
    — the assertion on the error message proves it got past that stage.
    """
    df = _full_baseline_df(n_days=90, n_stations=2)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out",
        seeds=(1,), outer_fold_params={"train_min_days": 30, "val_days": 15, "step_days": 15},
        fold_subset={999},  # no real batch has fold 999 -> realised sees zero plans
        verbose=False,
    )

    assert result.status == STATUS_ABORTED_CANDIDATE
    assert "config error" in result.error
    assert "seed-variance" not in result.error


def test_run_candidate_aborted_environment_when_db_is_unreadable(tmp_path):
    """Without the fold_subset trick, run_paired_realised_backtest gets past
    its pre-DB checks and hits the batch's (deliberately fake) DB file — a
    genuine DB-shaped failure, correctly mapped to aborted_environment. Guards
    against the ValueError/Exception split (finding #3's fix) over-classifying
    real DB failures as candidate config errors.
    """
    df = _full_baseline_df(n_days=90, n_stations=2)
    batch_dir = _write_batch_dir(tmp_path, df)  # fuel_signal.db is fake bytes
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out",
        seeds=(1, 2), outer_fold_params={"train_min_days": 30, "val_days": 15, "step_days": 15},
        verbose=False,
    )

    assert result.status == STATUS_ABORTED_ENVIRONMENT


# ── lookup provider ───────────────────────────────────────────────────────────

def test_make_lookup_provider_hits_and_misses():
    frame = pd.DataFrame({
        "station_code": [1, 1, 2],
        "price_date": ["2026-01-01", "2026-01-02", "2026-01-01"],
        "cand_col": [10.0, 20.0, float("nan")],
    })
    provider = _make_lookup_provider(frame, ["cand_col"])

    assert provider("2026-01-01", 1, 100.0) == {"cand_col": 10.0}
    assert provider("2026-01-02", 1, 100.0) == {"cand_col": 20.0}
    assert provider("2026-01-01", 2, 100.0) == {"cand_col": None}  # NaN -> None
    assert provider("2026-01-03", 1, 100.0) == {"cand_col": None}  # missing date -> None
    assert provider("2026-01-01", 99, 100.0) == {"cand_col": None}  # missing station -> None


# ── effect / zone grading ─────────────────────────────────────────────────────

def test_resolve_effect_true_when_candidate_cheaper():
    aggregate = pd.DataFrame([
        {"arm": "R0", "cpl_held": 200.0},
        {"arm": "candidate", "cpl_held": 195.0},
    ])
    resolved, delta = _resolve_effect(aggregate, "R0", "candidate")
    assert resolved is True
    assert delta == pytest.approx(-5.0)


def test_resolve_effect_false_when_candidate_more_expensive():
    aggregate = pd.DataFrame([
        {"arm": "R0", "cpl_held": 200.0},
        {"arm": "candidate", "cpl_held": 205.0},
    ])
    resolved, delta = _resolve_effect(aggregate, "R0", "candidate")
    assert resolved is False
    assert delta == pytest.approx(5.0)


def _fills(n_target_each: int, n_other_each: int, *, cheaper_in_target: bool) -> pd.DataFrame:
    rows = []
    for _ in range(n_target_each):
        rows.append({
            "fold": 4, "arm": "R0", "station_code": 1, "date": "2026-01-01",
            "litres": 50, "spend_cents": 50 * 200,
        })
        rows.append({
            "fold": 4, "arm": "candidate", "station_code": 1, "date": "2026-01-01",
            "litres": 50, "spend_cents": 50 * (190 if cheaper_in_target else 210),
        })
    for _ in range(n_other_each):
        rows.append({
            "fold": 2, "arm": "R0", "station_code": 1, "date": "2026-02-01",
            "litres": 50, "spend_cents": 50 * 200,
        })
        rows.append({
            "fold": 2, "arm": "candidate", "station_code": 1, "date": "2026-02-01",
            "litres": 50, "spend_cents": 50 * 200,
        })
    return pd.DataFrame(rows)


def test_resolve_zone_by_folds_resolves_true_when_target_cell_is_cheaper():
    fills = _fills(35, 35, cheaper_in_target=True)
    target = {"folds": [4], "axis": None, "expect_concentration_in": []}
    result = _resolve_zone(target, fills, "R0", "candidate", axis_lookup=None, min_row_cell_n=30)
    assert result["resolved"] is True


def test_resolve_zone_by_folds_resolves_false_when_target_cell_is_not_cheaper():
    fills = _fills(35, 35, cheaper_in_target=False)
    target = {"folds": [4]}
    result = _resolve_zone(target, fills, "R0", "candidate", axis_lookup=None, min_row_cell_n=30)
    assert result["resolved"] is False


def test_resolve_zone_regime_axis_uses_shock_folds():
    # fold 4 is in SHOCK_FOLDS; fold 2 is not.
    fills = _fills(35, 35, cheaper_in_target=True)
    target = {"axis": "regime", "expect_concentration_in": ["shock"]}
    result = _resolve_zone(target, fills, "R0", "candidate", axis_lookup=None, min_row_cell_n=30)
    assert result["resolved"] is True


def test_resolve_zone_inconclusive_below_min_row_cell_n():
    fills = _fills(5, 5, cheaper_in_target=True)  # below default guard
    target = {"folds": [4]}
    result = _resolve_zone(target, fills, "R0", "candidate", axis_lookup=None, min_row_cell_n=30)
    assert result["resolved"] is None
    assert "min_row_cell_n" in result["reason"]


def test_resolve_zone_custom_axis_via_lookup():
    fills = _fills(35, 35, cheaper_in_target=True)
    # Custom axis: tag the target-fold's station/date as "weekday_X", other as "weekday_Y".
    axis_lookup = pd.DataFrame([
        {"station_code": 1, "date": "2026-01-01", "axis": "weekday_X"},
        {"station_code": 1, "date": "2026-02-01", "axis": "weekday_Y"},
    ])
    target = {"axis": "weekday", "expect_concentration_in": ["weekday_X"]}
    result = _resolve_zone(target, fills, "R0", "candidate", axis_lookup=axis_lookup, min_row_cell_n=30)
    assert result["resolved"] is True


def test_resolve_zone_custom_axis_without_add_axis_is_inconclusive():
    fills = _fills(35, 35, cheaper_in_target=True)
    target = {"axis": "weekday", "expect_concentration_in": ["weekday_X"]}
    result = _resolve_zone(target, fills, "R0", "candidate", axis_lookup=None, min_row_cell_n=30)
    assert result["resolved"] is None
    assert "add_axis" in result["reason"]


def test_resolve_zone_no_target_fields_is_inconclusive():
    fills = _fills(35, 35, cheaper_in_target=True)
    result = _resolve_zone({}, fills, "R0", "candidate", axis_lookup=None)
    assert result["resolved"] is None


# ── artifact writing ──────────────────────────────────────────────────────────

def test_finish_writes_status_and_error(tmp_path):
    result = _finish(STATUS_ABORTED_CANDIDATE, "cand", 0.0, tmp_path, error="boom")
    assert isinstance(result, RunResult)
    assert result.status == STATUS_ABORTED_CANDIDATE
    written = json.loads((tmp_path / "results.json").read_text())
    assert written["status"] == STATUS_ABORTED_CANDIDATE
    assert written["error"] == "boom"


# ── bd comment ─────────────────────────────────────────────────────────────

def test_summarise_for_comment_reports_effect_and_zone():
    results = {
        "candidate": {"name": "cand"},
        "status": "rejected",
        "effect_resolved": True,
        "effect_delta_cpl_held": -0.5,
        "zone": {"resolved": True, "target_delta_cpl_own": -1.0, "other_delta_cpl_own": 0.1},
        "meta": {"wall_seconds": 12.3, "n_windows": 4, "seeds": [1, 2]},
    }
    text = _summarise_for_comment(results)
    assert "moved the arbiter" in text
    assert "concentrated where predicted" in text
    assert "wall=12.3s" in text


def test_summarise_for_comment_handles_grade_run_failure_shape():
    """_grade_run's exception path returns effect_resolved/effect_delta_cpl_held
    as None (not False/0.0) — _summarise_for_comment must not crash formatting
    that, and must not misreport a grading failure as "did not move the
    arbiter" (a real, negative finding). Regression test for a bug the fps-hvi
    review caught: this exact shape raised TypeError on `f"{None:+.4f}"`,
    which escaped run_candidate uncaught since it's evaluated as an argument
    to post_bd_comment, outside that function's own try/except.
    """
    results = {
        "candidate": {"name": "cand"},
        "status": "rejected",
        "effect_resolved": None,
        "effect_delta_cpl_held": None,
        "grading_error": "KeyError('cpl_held')",
        "zone": {"resolved": None, "reason": "grading raised: KeyError('cpl_held')"},
        "meta": {"wall_seconds": 12.3, "n_windows": 4, "seeds": [1, 2]},
    }
    text = _summarise_for_comment(results)  # must not raise
    assert "grading failed" in text
    assert "n/a" in text
    assert "did not move the arbiter" not in text


def test_run_candidate_posts_bd_comment_even_when_grading_failed(monkeypatch):
    """End-to-end version of the regression above: post_bd_comment must still
    fire (and run_candidate must still return a RunResult) when a run
    completes but _grade_run's except branch fired, with bead_id set —
    exactly the path the TypeError escaped through.
    """
    captured = {}

    def fake_run(cmd, input, text, check):  # noqa: A002 - matches subprocess.run's kwarg name
        captured["cmd"] = cmd
        captured["input"] = input

    monkeypatch.setattr(subprocess, "run", fake_run)

    aggregate = pd.DataFrame([{"arm": "R0"}])  # missing cpl_held -> _resolve_effect raises
    realised = _FakeRealised(aggregate, pd.DataFrame())
    effect_resolved, effect_delta, zone, grading_error = _grade_run(
        realised, target=None, axis_lookup=None, min_row_cell_n=30
    )
    results = {
        "candidate": {"name": "cand"},
        "status": "rejected",
        "effect_resolved": effect_resolved,
        "effect_delta_cpl_held": effect_delta,
        "grading_error": grading_error,
        "zone": zone,
        "meta": {"wall_seconds": 1.0, "n_windows": 1, "seeds": [1]},
    }

    post_bd_comment("fps-3jj.4", _summarise_for_comment(results))  # must not raise

    assert captured["cmd"] == ["bd", "comment", "fps-3jj.4", "--stdin"]
    assert "grading failed" in captured["input"]


def test_post_bd_comment_invokes_bd_with_stdin(monkeypatch):
    captured = {}

    def fake_run(cmd, input, text, check):  # noqa: A002 - matches subprocess.run's kwarg name
        captured["cmd"] = cmd
        captured["input"] = input

    monkeypatch.setattr(subprocess, "run", fake_run)
    post_bd_comment("fps-3jj.4", "hello")

    assert captured["cmd"] == ["bd", "comment", "fps-3jj.4", "--stdin"]
    assert captured["input"] == "hello"


def test_post_bd_comment_does_not_raise_when_bd_cli_fails(monkeypatch):
    def fake_run(cmd, input, text, check):  # noqa: A002 - matches subprocess.run's kwarg name
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    post_bd_comment("fps-3jj.4", "hello")  # must not raise — best-effort reporting


def test_post_bd_comment_does_not_raise_when_bd_missing(monkeypatch):
    def fake_run(cmd, input, text, check):  # noqa: A002 - matches subprocess.run's kwarg name
        raise FileNotFoundError("bd not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    post_bd_comment("fps-3jj.4", "hello")  # must not raise — best-effort reporting


# ── batch-1 pass criterion ────────────────────────────────────────────────────

def test_record_and_read_pass_criterion_roundtrip(tmp_path):
    record_pass_criterion(tmp_path, "sign plus concentration", magnitude_gated=False)
    read_back = _read_pass_criterion(tmp_path)
    assert read_back == {"criterion": "sign plus concentration", "magnitude_gated": False}


def test_read_pass_criterion_returns_none_when_absent(tmp_path):
    assert _read_pass_criterion(tmp_path) is None


# ── worked example (module-format documentation) ─────────────────────────────

def test_template_candidate_passes_validation():
    candidate = load_candidate_module(REPO_ROOT / "experiments" / "candidates" / "TEMPLATE.py")
    dates = pd.date_range("2026-01-01", periods=30, freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame({
        "price_date": list(dates) * 2,
        "station_price_cents": np.linspace(150, 180, 30 * 2),
    })

    report = validate_candidate(candidate, frame)

    assert report.columns == candidate.COLUMNS
    assert candidate.CONFIDENCE_EFFECT is not None
    assert candidate.CONFIDENCE_ZONE is not None
    assert candidate.TARGET["axis"] == "day_of_week"
