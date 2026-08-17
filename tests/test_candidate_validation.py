"""Tests for experiments/pipeline/validate.py — the candidate validation harness."""
from __future__ import annotations

import pathlib
import types

import pandas as pd
import pytest

from experiments.pipeline.validate import (
    AllNaNColumnError,
    CandidateImportError,
    MissingInputColumnsError,
    MissingOutputColumnError,
    RestrictedFrameViolation,
    load_candidate_module,
    validate_candidate,
)


def _panel_frame() -> pd.DataFrame:
    dates = [20260801, 20260802, 20260803, 20260804, 20260805, 20260806, 20260807, 20260808]
    rows = []
    for station in ("A", "B"):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "station_code": station,
                    "price_date": d,
                    "price_cents": 100.0 + i + (0 if station == "A" else 10),
                    "tgp_delta_7d": float(i),
                }
            )
    return pd.DataFrame(rows)


def _make_candidate(add_columns, add_axis=None, *, inputs, columns=("candidate_col",)):
    return types.SimpleNamespace(
        NAME="test_candidate",
        INPUTS=list(inputs),
        COLUMNS=list(columns),
        add_columns=add_columns,
        add_axis=add_axis,
    )


def _pit_safe_add_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["station_code", "price_date"]).copy()
    out["candidate_col"] = (
        out.groupby("station_code")["tgp_delta_7d"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    return out


def test_validate_candidate_passes_for_pit_safe_columns_only():
    frame = _panel_frame()
    candidate = _make_candidate(_pit_safe_add_columns, inputs=["station_code", "price_date", "tgp_delta_7d"])

    report = validate_candidate(candidate, frame)

    assert report.columns == ["candidate_col"]
    assert report.axis_present is False
    assert 0.0 <= report.nan_rates["candidate_col"] < 1.0


def test_validate_candidate_passes_with_add_axis():
    frame = _panel_frame()

    def add_axis(df: pd.DataFrame) -> pd.Series:
        return (df["price_date"] % 2 == 0).rename("axis")

    candidate = _make_candidate(
        _pit_safe_add_columns, add_axis, inputs=["station_code", "price_date", "tgp_delta_7d"]
    )

    report = validate_candidate(candidate, frame)
    assert report.axis_present is True


def test_undeclared_read_in_add_columns_raises_restricted_frame_violation():
    frame = _panel_frame()

    def add_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["candidate_col"] = df["price_cents"]  # not declared in INPUTS
        return out

    candidate = _make_candidate(add_columns, inputs=["station_code", "price_date", "tgp_delta_7d"])

    with pytest.raises(RestrictedFrameViolation):
        validate_candidate(candidate, frame)


def test_undeclared_read_in_add_axis_raises_restricted_frame_violation():
    frame = _panel_frame()

    def add_axis(df: pd.DataFrame) -> pd.Series:
        return (df["price_cents"] > 100).rename("axis")  # not declared in INPUTS

    candidate = _make_candidate(
        _pit_safe_add_columns, add_axis, inputs=["station_code", "price_date", "tgp_delta_7d"]
    )

    with pytest.raises(RestrictedFrameViolation):
        validate_candidate(candidate, frame)


def test_leaky_add_columns_raises_via_pit_test():
    frame = _panel_frame()

    def leaky_add_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["candidate_col"] = out.groupby("station_code")["tgp_delta_7d"].transform("mean")
        return out

    candidate = _make_candidate(leaky_add_columns, inputs=["station_code", "price_date", "tgp_delta_7d"])

    from experiments.lib.pit_test import PitLeakError

    with pytest.raises(PitLeakError):
        validate_candidate(candidate, frame)


def test_leaky_add_axis_raises_via_pit_test():
    frame = _panel_frame()

    def add_axis(df: pd.DataFrame) -> pd.Series:
        whole_series_mean = df.groupby("station_code")["tgp_delta_7d"].transform("mean")
        return (df["tgp_delta_7d"] > whole_series_mean).rename("axis")

    candidate = _make_candidate(
        _pit_safe_add_columns, add_axis, inputs=["station_code", "price_date", "tgp_delta_7d"]
    )

    from experiments.lib.pit_test import PitLeakError

    with pytest.raises(PitLeakError):
        validate_candidate(candidate, frame)


def test_all_nan_column_raises():
    frame = _panel_frame()

    def add_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["candidate_col"] = float("nan")
        return out

    candidate = _make_candidate(add_columns, inputs=["station_code", "price_date", "tgp_delta_7d"])

    with pytest.raises(AllNaNColumnError):
        validate_candidate(candidate, frame)


def test_partial_nan_column_is_reported_not_rejected():
    frame = _panel_frame()

    def add_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["station_code", "price_date"]).copy()
        out["candidate_col"] = (
            out.groupby("station_code")["tgp_delta_7d"]
            .transform(lambda s: s.rolling(window=3, min_periods=3).mean())
        )
        return out

    candidate = _make_candidate(add_columns, inputs=["station_code", "price_date", "tgp_delta_7d"])

    report = validate_candidate(candidate, frame)
    assert report.nan_rates["candidate_col"] > 0.0


def test_load_candidate_module_imports_by_path(tmp_path: pathlib.Path):
    module_path = tmp_path / "my_candidate.py"
    module_path.write_text(
        "NAME = 'my_candidate'\n"
        "INPUTS = ['price_date']\n"
        "COLUMNS = ['candidate_col']\n"
        "def add_columns(df):\n"
        "    out = df.copy()\n"
        "    out['candidate_col'] = df['price_date']\n"
        "    return out\n"
    )
    candidate = load_candidate_module(module_path)
    assert candidate.NAME == "my_candidate"
    assert candidate.INPUTS == ["price_date"]


def test_load_candidate_module_wraps_import_errors(tmp_path: pathlib.Path):
    module_path = tmp_path / "broken_candidate.py"
    module_path.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(CandidateImportError):
        load_candidate_module(module_path)


def test_declared_input_missing_from_frame_raises_clear_error():
    frame = _panel_frame()
    candidate = _make_candidate(
        _pit_safe_add_columns, inputs=["station_code", "price_date", "tgp_delta_7d", "no_such_column"]
    )

    with pytest.raises(MissingInputColumnsError) as exc_info:
        validate_candidate(candidate, frame)
    assert "no_such_column" in str(exc_info.value)


def test_declared_output_column_not_produced_raises_clear_error():
    frame = _panel_frame()

    def add_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()  # never adds 'candidate_col'

    candidate = _make_candidate(add_columns, inputs=["station_code", "price_date", "tgp_delta_7d"])

    with pytest.raises(MissingOutputColumnError):
        validate_candidate(candidate, frame)
