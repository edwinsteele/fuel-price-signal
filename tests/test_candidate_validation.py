"""Tests for experiments/pipeline/validate.py — the candidate validation harness."""
from __future__ import annotations

import pathlib
import types

import pandas as pd
import pytest

from experiments.lib.pit_test import differential_pit_test
from experiments.pipeline.validate import (
    AllNaNColumnError,
    CandidateImportError,
    MissingInputColumnsError,
    MissingOutputColumnError,
    RestrictedFrameViolation,
    TargetColumnInInputs,
    load_candidate_module,
    validate_candidate,
)
from fuel_signal.features import TARGET_COLUMNS


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


# --- Target columns (features.TARGET_COLUMNS) ------------------------------------


def _oracle_frame() -> pd.DataFrame:
    """A panel frame carrying the label columns, as the real features frame does."""
    frame = _panel_frame()
    # future_min_cents as the real frame has it: the forward-looking minimum, already
    # computed and STAMPED on each row rather than derived at read time.
    frame = frame.sort_values(["station_code", "price_date"]).copy()
    frame["future_min_cents"] = (
        frame.groupby("station_code")["price_cents"]
        .transform(lambda s: s[::-1].cummin()[::-1])
    )
    frame["label"] = (frame["future_min_cents"] >= frame["price_cents"] - 3.0).astype(int)
    return frame


def _oracle_add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The leak: today's price minus the known future minimum."""
    out = df.copy()
    out["candidate_col"] = out["price_cents"] - out["future_min_cents"]
    return out


def test_differential_pit_test_does_NOT_catch_a_stamped_target_column():
    """The reason TargetColumnInInputs has to exist — documents the hole, not a wish.

    The PIT test truncates the frame by date and re-runs. `future_min_cents` was computed
    upstream and written into the row, so truncation leaves it untouched, the recomputed
    values match, and a blatant oracle feature sails through. If this test ever starts
    raising, the PIT test got stronger and this guard's rationale needs revisiting.
    """
    frame = _oracle_frame()

    differential_pit_test(_oracle_add_columns, frame)  # no PitLeakError — that is the point


def test_validate_candidate_rejects_target_column_in_inputs():
    frame = _oracle_frame()
    candidate = _make_candidate(
        _oracle_add_columns, inputs=["price_cents", "future_min_cents"]
    )

    with pytest.raises(TargetColumnInInputs) as excinfo:
        validate_candidate(candidate, frame)

    assert "future_min_cents" in str(excinfo.value)


def test_validate_candidate_rejects_label_in_inputs():
    frame = _oracle_frame()
    candidate = _make_candidate(
        lambda df: df.assign(candidate_col=df["label"].astype(float)),
        inputs=["label"],
    )

    with pytest.raises(TargetColumnInInputs):
        validate_candidate(candidate, frame)


def test_target_check_runs_before_the_restricted_frame_check():
    """Ordering matters: a candidate reading a target column AND an undeclared one should
    report the target, which is the disqualifying fault, not the provenance slip."""
    frame = _oracle_frame()
    candidate = _make_candidate(
        lambda df: df.assign(candidate_col=df["future_min_cents"] - df["tgp_delta_7d"]),
        inputs=["future_min_cents"],  # tgp_delta_7d read but undeclared
    )

    with pytest.raises(TargetColumnInInputs):
        validate_candidate(candidate, frame)


def test_target_columns_are_absent_from_the_locked_baseline():
    """A target column in the lock would mean the model trains on its own answer."""
    from fuel_signal.features import LOCKED_FEATURE_COLUMNS

    assert not set(TARGET_COLUMNS) & set(LOCKED_FEATURE_COLUMNS)


def test_a_candidate_reading_only_safe_columns_is_unaffected():
    """The guard is keyed on the target columns alone — it must not fire on a frame that
    merely CONTAINS them, which every real features frame does."""
    frame = _oracle_frame()
    candidate = _make_candidate(
        _pit_safe_add_columns, inputs=["station_code", "price_date", "tgp_delta_7d"]
    )

    report = validate_candidate(candidate, frame)

    assert report.columns == ["candidate_col"]


def test_undeclared_get_of_a_target_column_sees_None_not_the_oracle():
    """The declaration blocklist is the error message; this is the actual barrier.

    `df["future_min_cents"]` on the restricted frame raises KeyError, but
    `df.get("future_min_cents")` returns None silently — so a candidate that never names
    the column in INPUTS slips past step 0 AND the provenance check, and (per the test
    above) the PIT test cannot see the leak either. Narrowing the frame the candidate is
    handed is what closes that: the column is not there to be fetched, in validation and
    in the runner alike.
    """
    frame = _oracle_frame()

    def sneaky(df):
        out = df.copy()
        future_min = df.get("future_min_cents")
        out["candidate_col"] = 0.0 if future_min is None else out["price_cents"] - future_min
        return out

    candidate = _make_candidate(sneaky, inputs=["price_cents"])

    report = validate_candidate(candidate, frame)

    # Validation passes -- it is not a leak once the column is unreachable -- and the
    # column it computed is the None branch, not the oracle.
    assert report.nan_rates == {"candidate_col": 0.0}
    assert (sneaky(frame.drop(columns=["future_min_cents", "label"]))["candidate_col"] == 0.0).all()
