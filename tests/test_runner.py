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

import experiments.pipeline.runner as runner_module
from experiments.lib.constants import ROW_AXIS_ECONOMICS_CAVEAT
from experiments.lib.realised import BaselineCache, BaselineCacheMismatch
from experiments.pipeline import batch_freeze
from experiments.pipeline.batch_freeze import resolve_baseline_columns
from experiments.pipeline.runner import (
    ABORT_REASON_LEAK_BY_DECLARATION,
    BASELINE_CACHE_FILENAME,
    CANDIDATE_ARM,
    DEFAULT_INNER_FOLD_PARAMS,
    RETRYABLE_STATUSES,
    STATUS_ABORTED_CANDIDATE,
    STATUS_ABORTED_ENVIRONMENT,
    STATUS_ABORTED_PIPELINE,
    STATUS_DISQUALIFIED,
    STATUS_GRADED,
    FoldGeometryError,
    RunResult,
    _build_axis_lookup,
    _check_fold_geometry,
    _check_provider_health,
    _finish,
    _grade_run,
    _load_baseline_cache,
    _make_lookup_provider,
    _read_pass_criterion,
    _resolve_effect,
    _resolve_zone,
    _run_wfcv_screen,
    _save_baseline_cache,
    _summarise_for_comment,
    default_out_dir,
    post_bd_comment,
    read_run_status,
    record_pass_criterion,
    run_candidate,
)
from experiments.pipeline.validate import load_candidate_module, validate_candidate
from fuel_signal import evaluate as _ev
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    LOCKED_FEATURE_COLUMNS,
    LOCKED_FEATURE_FINGERPRINT,
    NETWORK_FEATURE_COLUMNS,
    baseline_fingerprint,
)

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

TARGET_LEAK_CANDIDATE = '''
NAME = "cand"
INPUTS = ["price_date", "future_min_cents"]
COLUMNS = ["cand_col"]
def add_columns(df):
    out = df.copy()
    out["cand_col"] = df["future_min_cents"]
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


def test_run_candidate_default_out_dir_keeps_two_batch_siblings_separate(tmp_path):
    """fps-icv: two candidates filed against the same batch must not share out_dir.

    Before the fix, run_candidate's default out_dir was candidate_path.parent —
    the whole batch directory — so candidate B's results.json here would
    overwrite candidate A's in place. Both must now land in their own
    subdirectory, keyed off each candidate's own module name.
    """
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_a_path = _write_candidate(batch_dir, "raise RuntimeError('boom a')\n", name="cand_a.py")
    candidate_b_path = _write_candidate(batch_dir, "raise RuntimeError('boom b')\n", name="cand_b.py")

    result_a = run_candidate(batch_dir, candidate_a_path)
    result_b = run_candidate(batch_dir, candidate_b_path)

    assert result_a.results_path == default_out_dir(candidate_a_path) / "results.json"
    assert result_b.results_path == default_out_dir(candidate_b_path) / "results.json"
    assert result_a.results_path != result_b.results_path
    assert result_a.results_path.exists()
    assert result_b.results_path.exists()
    assert "boom a" in json.loads(result_a.results_path.read_text())["error"]
    assert "boom b" in json.loads(result_b.results_path.read_text())["error"]


def test_run_candidate_aborted_environment_on_baseline_drift(tmp_path, monkeypatch):
    """Drift is code-side: a locked-column bump landing mid-batch (fps-sa1).

    It used to be simulated by adding a brand-trough column to the frame, back when
    resolve_baseline_columns() discovered that group. It no longer does — the lock
    never contained brand troughs — so the drift has to come from the column
    constants themselves, which is the only thing that can legitimately move now.
    """
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    widened = df.copy()
    widened["new_locked_col"] = 0.0
    widened.to_parquet(batch_dir / "features.parquet")
    monkeypatch.setattr(
        batch_freeze, "LOCKED_FEATURE_COLUMNS", LOCKED_FEATURE_COLUMNS + ["new_locked_col"]
    )
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


def test_run_candidate_tags_declared_leak_with_abort_reason(tmp_path):
    """fps-3jj.13: a candidate that names a label column in INPUTS is a declared leak,
    distinct from an ordinary bug — results.json's abort_reason must say so, so the
    retrospective can count it separately from every other aborted_candidate cause."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, TARGET_LEAK_CANDIDATE)

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_ABORTED_CANDIDATE
    written = json.loads(result.results_path.read_text())
    assert written["abort_reason"] == ABORT_REASON_LEAK_BY_DECLARATION


def test_run_candidate_aborted_candidate_on_columns_overlap_baseline(tmp_path):
    """A baseline-column collision is self-defeating but not a declared leak (fps-3jj.13):
    abort_reason must stay null, unlike TARGET_LEAK_CANDIDATE above."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, OVERLAP_CANDIDATE)

    result = run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out")

    assert result.status == STATUS_ABORTED_CANDIDATE
    assert "overlaps" in result.error
    assert json.loads(result.results_path.read_text())["abort_reason"] is None


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
        realised, target=None, axis_lookup=None, shock_folds=frozenset(), min_row_cell_n=30
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
        realised, target=None, axis_lookup=None, shock_folds=frozenset(), min_row_cell_n=30
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


def test_run_wfcv_screen_honours_a_shock_folds_override(tmp_path):
    """fps-3tu follow-up: run_candidate passes a batch-level cached shock-fold set
    (experiments.pipeline.shock_folds.load_cached_shock_folds) straight through to
    _run_wfcv_screen instead of letting it derive one live — this is the wiring that
    makes that cache actually take effect rather than being computed and ignored.
    """
    frame = _synth_panel(n_days=150, n_stations=2)
    outer = {"train_min_days": 60, "val_days": 30, "step_days": 30}

    df_rows, _collector = _run_wfcv_screen(
        frame, ["f1", "f2"], ["f1", "f2", "cand"],
        seeds=(1,), axis_series=None, outer_fold_params=outer, verbose=False,
    )
    live_folds = sorted(df_rows["fold"].unique())
    assert len(live_folds) >= 2  # otherwise the override below can't be distinguishable

    # Deliberately the OPPOSITE tagging of whatever live derivation would pick: everything
    # live called "normal" is forced "shock" here, and vice versa. If the override weren't
    # actually honoured, this would just reproduce the live result.
    live_shock = frozenset(df_rows.loc[df_rows["regime"] == "shock", "fold"].unique())
    override = frozenset(live_folds) - live_shock

    overridden_rows, _collector2 = _run_wfcv_screen(
        frame, ["f1", "f2"], ["f1", "f2", "cand"],
        seeds=(1,), axis_series=None, outer_fold_params=outer, verbose=False,
        shock_folds=override,
    )
    got_shock = frozenset(overridden_rows.loc[overridden_rows["regime"] == "shock", "fold"].unique())
    assert got_shock == override
    assert got_shock != live_shock


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

# Outer/inner pair scaled to the 90-day fixtures. The real defaults (outer 1825,
# inner 1095) can't fit in a 90-day frame, and _check_fold_geometry now enforces
# that relationship, so a test using a toy outer grid has to supply a toy inner
# grid too: 15 + 7 buffer + 5 val = 27 <= 30.
SMALL_OUTER_FOLDS = {"train_min_days": 30, "val_days": 15, "step_days": 15}
SMALL_INNER_FOLDS = {"train_min_days": 15, "val_days": 5, "step_days": 5}


def test_run_candidate_realised_config_error_maps_to_aborted_pipeline(tmp_path):
    """fold_subset excluding every fold makes run_paired_realised_backtest
    raise ValueError('no folds planned') BEFORE touching the DB — a
    deterministic config error, not a DB/environment failure (fps-hvi finding
    #3: these must not both map to aborted_environment). Also exercises a
    single-seed run (fps-hvi finding #4): with seeds=(1,), the seed-variance
    gate must be skipped rather than raising its own aborted_candidate first
    — the assertion on the error message proves it got past that stage.

    fps-g31 moved this off aborted_candidate: the run config, not the
    candidate, is what's wrong, and every candidate against the batch would
    fail identically. It must land in RETRYABLE_STATUSES so the claim goes back
    on the queue instead of being consumed by a verdict the run never reached.
    """
    df = _full_baseline_df(n_days=90, n_stations=2)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out",
        seeds=(1,), outer_fold_params=SMALL_OUTER_FOLDS, inner_fold_params=SMALL_INNER_FOLDS,
        fold_subset={999},  # no real batch has fold 999 -> realised sees zero plans
        verbose=False,
    )

    assert result.status == STATUS_ABORTED_PIPELINE
    assert result.status in RETRYABLE_STATUSES
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
        seeds=(1, 2), outer_fold_params=SMALL_OUTER_FOLDS, inner_fold_params=SMALL_INNER_FOLDS,
        verbose=False,
    )

    assert result.status == STATUS_ABORTED_ENVIRONMENT


def test_run_candidate_threads_the_cached_shock_folds_into_the_wfcv_screen(tmp_path, monkeypatch):
    """fps-3tu follow-up: run_candidate must load this batch's cached shock-fold set
    (experiments.pipeline.shock_folds.load_cached_shock_folds), keyed on THIS run's own
    baseline fingerprint, and pass it straight through to _run_wfcv_screen — not silently
    drop it or let the screen re-derive one live. _run_wfcv_screen is faked to raise before
    doing any real fitting, so this only exercises the wiring, not the (already separately
    tested) derivation itself.
    """
    df = _full_baseline_df(n_days=90, n_stations=2)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)
    expected_fp = baseline_fingerprint(resolve_baseline_columns(df))
    cached_set = frozenset({2})
    captured = {}

    def fake_load_cached(batch_dir_arg, *, baseline_fingerprint, seed, outer_fold_params):
        captured["batch_dir"] = batch_dir_arg
        captured["fingerprint"] = baseline_fingerprint
        captured["seed"] = seed
        captured["outer_fold_params"] = outer_fold_params
        return cached_set

    def fake_wfcv_screen(*args, **kwargs):
        captured["shock_folds_kwarg"] = kwargs.get("shock_folds")
        raise RuntimeError("stop here — only the wiring into this call is under test")

    monkeypatch.setattr(runner_module, "load_cached_shock_folds", fake_load_cached)
    monkeypatch.setattr(runner_module, "_run_wfcv_screen", fake_wfcv_screen)

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out",
        seeds=(1,), outer_fold_params=SMALL_OUTER_FOLDS, inner_fold_params=SMALL_INNER_FOLDS,
        verbose=False,
    )

    assert captured["batch_dir"] == pathlib.Path(batch_dir)
    assert captured["fingerprint"] == expected_fp
    assert captured["seed"] == 1  # seeds[0]
    assert captured["outer_fold_params"] == SMALL_OUTER_FOLDS
    assert captured["shock_folds_kwarg"] == cached_set
    assert result.status == STATUS_ABORTED_CANDIDATE  # from fake_wfcv_screen's raise


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
    result = _resolve_zone(
        target, fills, "R0", "candidate", axis_lookup=None, shock_folds=frozenset({4}), min_row_cell_n=30
    )
    assert result["resolved"] is True
    # A fold cut is a sum of complete independent simulations — identified, no caveat.
    assert "identification_caveat" not in result


def test_resolve_zone_by_folds_resolves_false_when_target_cell_is_not_cheaper():
    fills = _fills(35, 35, cheaper_in_target=False)
    target = {"folds": [4]}
    result = _resolve_zone(
        target, fills, "R0", "candidate", axis_lookup=None, shock_folds=frozenset({4}), min_row_cell_n=30
    )
    assert result["resolved"] is False


def test_resolve_zone_regime_axis_uses_shock_folds():
    # fold 4 is in the passed-in shock_folds set; fold 2 is not.
    fills = _fills(35, 35, cheaper_in_target=True)
    target = {"axis": "regime", "expect_concentration_in": ["shock"]}
    result = _resolve_zone(
        target, fills, "R0", "candidate", axis_lookup=None, shock_folds=frozenset({4}), min_row_cell_n=30
    )
    assert result["resolved"] is True
    # A regime cut groups WHOLE folds, so this is identified too.
    assert "identification_caveat" not in result


def test_resolve_zone_inconclusive_below_min_row_cell_n():
    fills = _fills(5, 5, cheaper_in_target=True)  # below default guard
    target = {"folds": [4]}
    result = _resolve_zone(
        target, fills, "R0", "candidate", axis_lookup=None, shock_folds=frozenset({4}), min_row_cell_n=30
    )
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
    result = _resolve_zone(
        target, fills, "R0", "candidate", axis_lookup=axis_lookup, shock_folds=frozenset({4}), min_row_cell_n=30
    )
    assert result["resolved"] is True
    # fps-grp: a row-level axis slices THROUGH a window, so the graded delta allocates a
    # path-coupled cost to a sub-period. It still grades, but must say it is unidentified.
    assert result["identification_caveat"] == ROW_AXIS_ECONOMICS_CAVEAT


def test_resolve_zone_flags_a_row_level_axis_even_when_folds_also_match():
    """A mixed TARGET (folds OR axis) is unidentified as soon as the axis contributes.

    The runner ORs the two masks together, so a row-level label can pull extra fills
    into the target cell even when the fold list alone would have graded cleanly —
    which means the caveat has to key off the axis being *used*, not off the fold list
    being empty.
    """
    fills = _fills(35, 35, cheaper_in_target=True)
    axis_lookup = pd.DataFrame([
        {"station_code": 1, "date": "2026-01-01", "axis": "weekday_X"},
        {"station_code": 1, "date": "2026-02-01", "axis": "weekday_Y"},
    ])
    target = {"folds": [4], "axis": "weekday", "expect_concentration_in": ["weekday_X"]}
    result = _resolve_zone(
        target, fills, "R0", "candidate", axis_lookup=axis_lookup, shock_folds=frozenset({4}), min_row_cell_n=30
    )
    assert result["resolved"] is True
    assert result["identification_caveat"] == ROW_AXIS_ECONOMICS_CAVEAT


def test_resolve_zone_custom_axis_without_add_axis_is_inconclusive():
    fills = _fills(35, 35, cheaper_in_target=True)
    target = {"axis": "weekday", "expect_concentration_in": ["weekday_X"]}
    result = _resolve_zone(
        target, fills, "R0", "candidate", axis_lookup=None, shock_folds=frozenset({4}), min_row_cell_n=30
    )
    assert result["resolved"] is None
    assert "add_axis" in result["reason"]


def test_resolve_zone_no_target_fields_is_inconclusive():
    fills = _fills(35, 35, cheaper_in_target=True)
    result = _resolve_zone({}, fills, "R0", "candidate", axis_lookup=None, shock_folds=frozenset({4}))
    assert result["resolved"] is None


# ── artifact writing ──────────────────────────────────────────────────────────

def test_finish_writes_status_and_error(tmp_path):
    result = _finish(STATUS_ABORTED_CANDIDATE, "cand", 0.0, tmp_path, error="boom")
    assert isinstance(result, RunResult)
    assert result.status == STATUS_ABORTED_CANDIDATE
    written = json.loads((tmp_path / "results.json").read_text())
    assert written["status"] == STATUS_ABORTED_CANDIDATE
    assert written["error"] == "boom"


def test_finish_posts_bd_comment_on_abort(tmp_path, monkeypatch):
    """fps-g31: an abort that says nothing on the bead is invisible.

    fps-32p aborted and the only comment on it was the launch routine's
    "launched detached" line, so the failure looked exactly like a run that
    had never finished.
    """
    posted = []
    monkeypatch.setattr(
        runner_module, "post_bd_comment", lambda issue_id, text: posted.append((issue_id, text))
    )

    _finish(STATUS_ABORTED_CANDIDATE, "cand", 0.0, tmp_path, error="boom", bead_id="fps-xyz")

    assert len(posted) == 1
    issue_id, text = posted[0]
    assert issue_id == "fps-xyz"
    assert STATUS_ABORTED_CANDIDATE in text
    assert "boom" in text
    assert "Terminal for this candidate" in text


def test_finish_marks_retryable_status_as_not_the_candidates_fault(tmp_path, monkeypatch):
    posted = []
    monkeypatch.setattr(
        runner_module, "post_bd_comment", lambda issue_id, text: posted.append((issue_id, text))
    )

    _finish(STATUS_ABORTED_PIPELINE, "cand", 0.0, tmp_path, error="bad config", bead_id="fps-xyz")

    assert "not a verdict on the candidate" in posted[0][1]


def test_finish_without_bead_id_posts_nothing(tmp_path, monkeypatch):
    posted = []
    monkeypatch.setattr(
        runner_module, "post_bd_comment", lambda issue_id, text: posted.append((issue_id, text))
    )

    _finish(STATUS_ABORTED_CANDIDATE, "cand", 0.0, tmp_path, error="boom")

    assert posted == []


def test_finish_stamps_provider_stats_when_given(tmp_path):
    """fps-b5p: the _check_provider_health abort site has live provider.stats;
    _finish must persist them so a 100%-miss abort is distinguishable from a
    run that never recorded provider stats at all.
    """
    _finish(
        STATUS_ABORTED_CANDIDATE, "cand", 0.0, tmp_path, error="all lookups missed",
        provider_stats={"hits": 0, "misses": 42},
    )
    written = json.loads((tmp_path / "results.json").read_text())
    assert written["meta"]["extra_feature_provider_hits"] == 0
    assert written["meta"]["extra_feature_provider_misses"] == 42


def test_finish_omits_provider_stats_when_not_given(tmp_path):
    """Every other abort site has no provider stats to hand — the keys must
    stay absent (not a fabricated 0/0), so dossier_tables' `.get()` correctly
    reads them as "not recorded" rather than "0% miss rate".
    """
    _finish(STATUS_ABORTED_PIPELINE, "cand", 0.0, tmp_path, error="bad config")
    written = json.loads((tmp_path / "results.json").read_text())
    assert "extra_feature_provider_hits" not in written["meta"]
    assert "extra_feature_provider_misses" not in written["meta"]


# ── read_run_status (fps-g31) ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "written,expected",
    [
        ('{"status": "graded"}', "graded"),
        ('{"status": "aborted_pipeline"}', "aborted_pipeline"),
        ('{"error": "no status key"}', None),
        ('{ truncated mid-write', None),          # JSONDecodeError
        ('[]', None),                             # valid JSON, not an object
        ('null', None),
        ('"a bare string"', None),
        ('{"status": 42}', None),                 # present but not a string
    ],
)
def test_read_run_status_is_total(tmp_path, written, expected):
    """Both nightly routines call this unattended — it must never raise.

    A results.json that is valid JSON but not an object (a list, null, a bare
    string) would make a bare .get() raise AttributeError and take down the
    whole sweep on the strength of one malformed file.
    """
    (tmp_path / "results.json").write_text(written)
    assert read_run_status(tmp_path) == expected


def test_read_run_status_returns_none_when_file_absent(tmp_path):
    assert read_run_status(tmp_path) is None


def test_read_run_status_survives_invalid_utf8(tmp_path):
    """A write interrupted mid-multi-byte-sequence must not crash the nightly sweep.

    read_text() raises UnicodeDecodeError on invalid UTF-8. It's a ValueError
    subclass and NOT an OSError, so a narrower `except (JSONDecodeError,
    OSError)` lets it escape into find_pending_runs / find_stale_claims and take
    down the whole scan.
    """
    (tmp_path / "results.json").write_bytes(b'{"status": "reje\xff\xfecte')
    assert read_run_status(tmp_path) is None


# ── fold geometry guard (fps-g31) ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "outer_train,fits",
    [
        (1825, True),   # the real default
        (1192, True),   # exact boundary: 1095 + 7 buffer + 90 val
        (1191, False),  # one day under
        (1100, False),  # would pass a naive "inner < outer" check, still yields 0 folds
        (1000, False),
    ],
)
def test_check_fold_geometry_matches_real_fold_arithmetic(outer_train, fits):
    """The guard must agree with walk_forward_folds, not approximate it.

    Boundary verified empirically against a real date grid: outer=1192 yields 1
    inner fold at the defaults, outer=1191 yields 0. A naive `inner < outer`
    check would wave 1100 through.
    """
    outer = {"train_min_days": outer_train}
    if fits:
        _check_fold_geometry(outer, DEFAULT_INNER_FOLD_PARAMS)
    else:
        with pytest.raises(FoldGeometryError, match="don't fit inside outer fold 1"):
            _check_fold_geometry(outer, DEFAULT_INNER_FOLD_PARAMS)

    # and confirm the guard's verdict is the truth, not just self-consistent
    dates = pd.date_range("2016-01-01", periods=outer_train + 400, freq="D")
    df = pd.DataFrame({
        "price_date": dates.strftime("%Y-%m-%d"),
        "label": ([0, 1] * len(dates))[: len(dates)],
    })
    fold_one_train = next(iter(_ev.walk_forward_folds(df, train_min_days=outer_train)))[0]
    n_inner = len(list(_ev.walk_forward_folds(fold_one_train, **DEFAULT_INNER_FOLD_PARAMS)))
    assert (n_inner >= 1) is fits


def test_check_fold_geometry_uses_library_defaults_when_outer_unset():
    """Outer params left empty means walk_forward_folds' own default, not zero."""
    _check_fold_geometry({}, DEFAULT_INNER_FOLD_PARAMS)
    with pytest.raises(FoldGeometryError):
        _check_fold_geometry({}, {"train_min_days": 5000, "val_days": 90, "step_days": 90})


def test_run_candidate_rejects_impossible_fold_geometry(tmp_path):
    """Raised, not _finish()'d — a run this misconfigured must not leave a results.json,
    because that file is the queue key for both launch and the dossier scan."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_CANDIDATE)
    out_dir = tmp_path / "out"

    with pytest.raises(FoldGeometryError):
        run_candidate(
            batch_dir, candidate_path, out_dir=out_dir,
            outer_fold_params={"train_min_days": 1000}, verbose=False,
        )

    assert not (out_dir / "results.json").exists()


# ── stale results.json (fps-g31) ──────────────────────────────────────────────

def test_run_candidate_clears_stale_results_json_before_running(tmp_path):
    """A re-run reuses the run dir, and nothing else ever clears this file.

    Left in place, launch.find_stale_claims (no age gate) reads the PREVIOUS
    attempt's retryable status off disk mid-run, releases the in-flight run's
    claim, and launches a second runner into the same out_dir.
    """
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "results.json").write_text(json.dumps({"status": STATUS_ABORTED_PIPELINE}))

    # Fails immediately on import, but only AFTER the stale file is cleared.
    candidate_path = _write_candidate(tmp_path, "raise RuntimeError('boom')\n")
    result = run_candidate(batch_dir, candidate_path, out_dir=out_dir)

    assert result.status == STATUS_ABORTED_CANDIDATE
    assert json.loads((out_dir / "results.json").read_text())["status"] == STATUS_ABORTED_CANDIDATE


# ── inner fold params (fps-g31) ───────────────────────────────────────────────

def test_default_inner_fold_params_fit_inside_outer_fold_one():
    """The bug that aborted every candidate until fps-g31.

    walk_forward_folds sizes outer fold 1's train window to exactly
    train_min_days. An inner walk-forward at the SAME train_min_days therefore
    cannot fit a single fold inside it, pool_oof_predictions returns empty, and
    _train_calibrate_select_tau raises "no OOF folds over fold-train" on the
    first fold of the arbiter — for every candidate, identically.

    Asserted against a real date grid rather than by comparing the two numbers,
    so this fails if walk_forward_folds' fold geometry changes too.
    """
    outer_train_min_days = 1825
    dates = pd.date_range("2016-01-01", periods=outer_train_min_days + 400, freq="D")
    df = pd.DataFrame({
        "price_date": dates.strftime("%Y-%m-%d"),
        "label": ([0, 1] * len(dates))[: len(dates)],
    })

    outer_folds = list(_ev.walk_forward_folds(df, train_min_days=outer_train_min_days))
    assert outer_folds, "fixture must produce at least one outer fold"
    fold_one_train = outer_folds[0][0]

    naive = list(_ev.walk_forward_folds(fold_one_train, train_min_days=outer_train_min_days))
    assert naive == [], "fixture no longer reproduces the fps-g31 geometry"

    inner = list(_ev.walk_forward_folds(fold_one_train, **DEFAULT_INNER_FOLD_PARAMS))
    assert len(inner) >= 1


def _capture_realised_kwargs(monkeypatch) -> dict:
    """Intercept the realised-backtest call and stop the run there.

    The arbiter needs a real DB; these tests only care what the runner decided
    to pass it, so raise once the kwargs are captured.
    """
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here — only the kwargs matter")

    monkeypatch.setattr(runner_module, "run_paired_realised_backtest", _capture)
    return seen


def test_run_candidate_defaults_inner_fold_params(tmp_path, monkeypatch):
    """run_candidate must not pass an empty inner_fold_params through to the arbiter.

    Empty is what launch.py's CLI produced before fps-g31, and empty means the
    library default (1825) that cannot fit inside outer fold 1.
    """
    seen = _capture_realised_kwargs(monkeypatch)
    # Big enough for the REAL outer default (1825 + 7 buffer + 90 val), so this
    # exercises the production geometry rather than a toy grid — the whole point
    # is what the defaults resolve to.
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)

    run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2), verbose=False,
    )

    assert seen["inner_fold_params"] == DEFAULT_INNER_FOLD_PARAMS


def test_run_candidate_merges_partial_inner_fold_params(tmp_path, monkeypatch):
    """A caller overriding one key still gets a usable train_min_days.

    Replacing rather than merging would silently reintroduce the fps-g31 bug for
    anyone who passed only val_days.
    """
    seen = _capture_realised_kwargs(monkeypatch)
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)

    run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2),
        inner_fold_params={"val_days": 30}, verbose=False,
    )

    assert seen["inner_fold_params"]["val_days"] == 30
    assert seen["inner_fold_params"]["train_min_days"] == DEFAULT_INNER_FOLD_PARAMS["train_min_days"]


# ── R0 baseline cache (fps-e2l) ───────────────────────────────────────────────

def test_load_baseline_cache_round_trip(tmp_path):
    cache = BaselineCache(fingerprint={"seed": 42}, per_fold={1: {"own_tau": 0.3}})
    _save_baseline_cache(tmp_path, cache, verbose=False)

    loaded = _load_baseline_cache(tmp_path, verbose=False)

    assert loaded.fingerprint == cache.fingerprint
    assert loaded.per_fold == cache.per_fold
    assert not (tmp_path / (BASELINE_CACHE_FILENAME + ".tmp")).exists()


def test_load_baseline_cache_returns_none_when_absent(tmp_path):
    assert _load_baseline_cache(tmp_path, verbose=False) is None


def test_load_baseline_cache_returns_none_on_corrupt_file(tmp_path):
    """A corrupt/unreadable cache is never a reason to abort — just refit R0."""
    (tmp_path / BASELINE_CACHE_FILENAME).write_bytes(b"not a joblib file")
    assert _load_baseline_cache(tmp_path, verbose=False) is None


def test_load_baseline_cache_returns_none_when_file_is_not_a_baseline_cache(tmp_path):
    """A file that deserializes fine but isn't a BaselineCache (a foreign
    .joblib, or a corrupted write that still happens to unpickle) must not
    reach run_paired_realised_backtest — it would fail there with a confusing
    AttributeError instead of this function's clear "refitting R0" message."""
    runner_module.joblib.dump({"not": "a BaselineCache"}, tmp_path / BASELINE_CACHE_FILENAME)
    assert _load_baseline_cache(tmp_path, verbose=False) is None


def test_save_baseline_cache_failure_cleans_up_tmp_file(tmp_path, monkeypatch):
    """A dump failure must not leave a half-written .tmp file behind for the
    next run to trip over."""
    def raising_dump(obj, path):
        pathlib.Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(runner_module.joblib, "dump", raising_dump)
    cache = BaselineCache(fingerprint={}, per_fold={})

    _save_baseline_cache(tmp_path, cache, verbose=False)  # must not raise

    assert not (tmp_path / BASELINE_CACHE_FILENAME).exists()
    assert not (tmp_path / (BASELINE_CACHE_FILENAME + ".tmp")).exists()


def test_run_candidate_passes_loaded_baseline_cache_into_realised_backtest(tmp_path, monkeypatch):
    """A cache already sitting in batch_dir must be loaded and forwarded —
    the whole point of persisting it (fps-e2l)."""
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)
    cache = BaselineCache(fingerprint={"seed": 42}, per_fold={1: {"own_tau": 0.3}})
    _save_baseline_cache(batch_dir, cache, verbose=False)

    seen = _capture_realised_kwargs(monkeypatch)
    run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2), verbose=False,
    )

    assert seen["baseline_cache"].fingerprint == cache.fingerprint
    assert seen["baseline_cache"].per_fold == cache.per_fold


def test_run_candidate_retries_uncached_on_baseline_cache_mismatch(tmp_path, monkeypatch):
    """A stale/mismatched cache must not abort the run — only cost the retry.

    First call is made with the loaded cache; on BaselineCacheMismatch, the
    runner must retry once with baseline_cache=None rather than surfacing the
    mismatch as aborted_pipeline (that would misreport "every candidate
    against this batch fails identically", which isn't true here — the batch
    is fine, only the stale cache is stale).
    """
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)
    _save_baseline_cache(batch_dir, BaselineCache(fingerprint={"seed": 999}, per_fold={}), verbose=False)

    calls: list[dict] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        if kwargs["baseline_cache"] is not None:
            raise BaselineCacheMismatch("fingerprint mismatch (test)")
        raise RuntimeError("stop here — only the retry behaviour matters")

    monkeypatch.setattr(runner_module, "run_paired_realised_backtest", fake_run)

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2), verbose=False,
    )

    assert len(calls) == 2
    assert calls[0]["baseline_cache"] is not None
    assert calls[1]["baseline_cache"] is None
    # The retry's RuntimeError is a genuine unexpected failure, not the mismatch
    # itself — proves the mismatch was swallowed, not propagated as a status.
    assert result.status == STATUS_ABORTED_ENVIRONMENT


class _FakeRealisedFull:
    """A RealisedResult-shaped stand-in for the (mocked, DB-free) success path."""

    def __init__(self, baseline_cache):
        self.aggregate = pd.DataFrame([
            {"arm": "R0", "cpl_held": 200.0},
            {"arm": "candidate", "cpl_held": 195.0},
        ])
        self.fills = pd.DataFrame(columns=["fold", "arm", "station_code", "date"])
        # fps-4je: results.json's "realised_deltas" key reads this verbatim off the real
        # RealisedResult — the fake must carry the same shape (realised.py's per-(fold, arm)
        # tau_diverges flag) or run_candidate's results-dict build fails before grading. A real
        # (non-empty) row, not just the right columns, so a column-name typo in runner.py's
        # `.to_dict(orient="records")` call would show up as a wrong VALUE in a test's
        # assertion rather than passing silently against an empty frame (PR #354 review
        # finding #4 — this was previously empty, so no value ever flowed through here).
        self.deltas = pd.DataFrame([
            {"fold": 1, "arm": "candidate", "delta_cpl_held": -5.0, "delta_cpl_own": -6.0,
             "tau_diverges": True},
        ])
        self.meta = {
            "n_windows": 1, "total_wall_seconds": 1.0,
            "baseline_cache_used": baseline_cache is not None, "baseline_cache_hit_folds": [],
            "tank_params": "50/3.571/7d/10%",
            "tank_params_fields": {
                "tank_size_litres": 50.0, "daily_consumption_litres": 50.0 / 14,
                "evaluation_interval_days": 7, "floor_fraction": 0.10,
            },
        }
        self.baseline_cache = baseline_cache


def test_run_candidate_persists_baseline_cache_after_successful_run(tmp_path, monkeypatch):
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)

    new_cache = BaselineCache(fingerprint={"seed": 42}, per_fold={1: {"own_tau": 0.31}})
    monkeypatch.setattr(
        runner_module, "run_paired_realised_backtest", lambda *a, **k: _FakeRealisedFull(new_cache)
    )

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2), verbose=False,
    )

    assert result.status == STATUS_GRADED
    persisted = _load_baseline_cache(batch_dir, verbose=False)
    assert persisted.fingerprint == new_cache.fingerprint
    assert persisted.per_fold == new_cache.per_fold


def test_run_candidate_provider_health_abort_survives_into_facts_json(tmp_path, monkeypatch):
    """fps-b5p end-to-end: a run aborted by _check_provider_health must persist
    its 100%-miss provider stats, and build_facts must report them as a real
    miss_rate of 1.0 — not None, which would be indistinguishable from "not
    recorded" (a legacy run, or a candidate with no provider at all)."""
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)

    def fake_run(*args, **kwargs):
        arms = args[0]
        candidate_arm = next(a for a in arms if a.name == CANDIDATE_ARM)
        for _ in range(5):
            # A date guaranteed absent from the candidate frame -> every call misses.
            candidate_arm.extra_feature_provider("2099-01-01", 999, 100.0)
        return _FakeRealisedFull(baseline_cache=None)

    monkeypatch.setattr(runner_module, "run_paired_realised_backtest", fake_run)

    out_dir = tmp_path / "out"
    result = run_candidate(
        batch_dir, candidate_path, out_dir=out_dir, seeds=(1, 2), verbose=False,
    )

    assert result.status == STATUS_ABORTED_CANDIDATE
    written = json.loads(result.results_path.read_text())
    assert written["meta"]["extra_feature_provider_hits"] == 0
    assert written["meta"]["extra_feature_provider_misses"] == 5

    from experiments.pipeline.dossier_tables import build_facts

    facts = build_facts(out_dir)
    assert facts["validation"]["extra_feature_provider_hits"] == 0
    assert facts["validation"]["extra_feature_provider_misses"] == 5
    assert facts["validation"]["extra_feature_provider_miss_rate"] == 1.0


def test_run_candidate_records_the_baseline_fingerprint_it_was_graded_against(
    tmp_path, monkeypatch
):
    """results.json is self-describing about its R0 (fps-zci item 5).

    The fingerprint covers what the run USED — the batch's frozen
    baseline_columns.json — so a results.json stays interpretable no matter how the
    constants move afterwards. Both contract defects found so far were compared
    head-to-head for two months because this field did not exist.
    """
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)
    monkeypatch.setattr(
        runner_module, "run_paired_realised_backtest", lambda *a, **k: _FakeRealisedFull(None)
    )

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2), verbose=False,
    )

    assert result.status == STATUS_GRADED
    frozen = json.loads((batch_dir / "baseline_columns.json").read_text())
    meta = result.results["meta"]
    assert meta["n_baseline_columns"] == len(frozen)
    assert meta["baseline_fingerprint"] == baseline_fingerprint(frozen)
    assert meta["baseline_fingerprint"] == LOCKED_FEATURE_FINGERPRINT


def test_run_candidate_records_the_tank_params_it_was_graded_at(tmp_path, monkeypatch):
    """results.json's meta must carry the cadence realised.aggregate's cpl_own/
    cpl_held values were produced at (fps-15c) — read straight off
    RealisedResult.meta, the single place run_paired_realised_backtest stamps it,
    rather than reformatted here."""
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)
    monkeypatch.setattr(
        runner_module, "run_paired_realised_backtest", lambda *a, **k: _FakeRealisedFull(None)
    )

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2), verbose=False,
    )

    assert result.status == STATUS_GRADED
    assert result.results["meta"]["tank_params"] == "50/3.571/7d/10%"


def test_run_candidate_persists_realised_deltas_into_results_json(tmp_path, monkeypatch):
    """PR #354 review finding #4: the realised.py -> results.json column contract
    (results["realised_deltas"] == realised.deltas.to_dict(orient="records")) is hard-indexed
    on both ends (dossier_tables.py reads "fold"/"arm"/"tau_diverges" by name) but was untested
    end to end — every other fps-4je test hand-writes results.json fixtures directly, so a
    rename in either runner.py or realised.py could pass the whole suite while breaking every
    real run. This pins the runner half: results.json's "realised_deltas" must be exactly
    RealisedResult.deltas, verbatim."""
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)
    fake = _FakeRealisedFull(None)
    monkeypatch.setattr(runner_module, "run_paired_realised_backtest", lambda *a, **k: fake)

    result = run_candidate(
        batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2), verbose=False,
    )

    assert result.status == STATUS_GRADED
    assert result.results["realised_deltas"] == fake.deltas.to_dict(orient="records")
    written = json.loads(result.results_path.read_text())
    assert written["realised_deltas"] == [
        {"fold": 1, "arm": "candidate", "delta_cpl_held": -5.0, "delta_cpl_own": -6.0,
         "tau_diverges": True},
    ]


def test_run_candidate_does_not_persist_when_baseline_cache_is_none(tmp_path, monkeypatch):
    df = _full_baseline_df(n_days=1930, n_stations=1)
    batch_dir = _write_batch_dir(tmp_path, df)
    candidate_path = _write_candidate(tmp_path, PIT_SAFE_STRING_DATE_CANDIDATE)

    monkeypatch.setattr(
        runner_module, "run_paired_realised_backtest", lambda *a, **k: _FakeRealisedFull(None)
    )

    run_candidate(batch_dir, candidate_path, out_dir=tmp_path / "out", seeds=(1, 2), verbose=False)

    assert not (batch_dir / BASELINE_CACHE_FILENAME).exists()


# ── bd comment ─────────────────────────────────────────────────────────────

def test_summarise_for_comment_reports_effect_and_zone():
    results = {
        "candidate": {"name": "cand"},
        "status": STATUS_GRADED,
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
        "status": STATUS_GRADED,
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
        realised, target=None, axis_lookup=None, shock_folds=frozenset(), min_row_cell_n=30
    )
    results = {
        "candidate": {"name": "cand"},
        "status": STATUS_GRADED,
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
