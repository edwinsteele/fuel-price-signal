"""Tests for experiments/pipeline/redundancy.py — the pre-filing batch screen."""
from __future__ import annotations

import json
import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest

from experiments.pipeline.redundancy import (
    BLOCK_R2_FLAG,
    PAIRWISE_RHO_THRESHOLD,
    DuplicateCandidateColumn,
    block_r2,
    compute_candidate_columns,
    existing_column_set,
    pairwise_rho,
    screen_batch,
    usable_predictors,
    write_batch_record,
)
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, NETWORK_FEATURE_COLUMNS


def _frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {
        "price_date": np.arange(20200101, 20200101 + n),
        "station_code": ["A"] * n,
    }
    for c in FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS:
        cols[c] = rng.normal(size=n)
    cols["tgp_delta_7d"] = rng.normal(size=n)
    return pd.DataFrame(cols)


def _write_module(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body))
    return path


_SIMPLE = '''
    import pandas as pd
    NAME = "{name}"
    MECHANISM_FAMILY = "{family}"
    COLUMNS = ["{col}"]
    INPUTS = ["{src}"]
    def add_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["{col}"] = out["{src}"] * {mult}
        return out
'''


def test_cross_candidate_duplicate_fails_the_gate(tmp_path):
    """Two candidates that are the same column scaled — the reparameterisation case."""
    frame = _frame()
    a = _write_module(tmp_path, "a", _SIMPLE.format(
        name="a", family="f1", col="a_col", src="tgp_delta_7d", mult=1.0))
    b = _write_module(tmp_path, "b", _SIMPLE.format(
        name="b", family="f2", col="b_col", src="tgp_delta_7d", mult=3.0))

    cands = compute_candidate_columns([a, b], frame, validate=False)
    rho = pairwise_rho(cands)

    row = rho.iloc[0]
    assert row["cross_candidate"]
    assert row["abs_rho"] > PAIRWISE_RHO_THRESHOLD


def test_within_candidate_correlation_is_not_gated(tmp_path):
    """A group whose members are related is the normal case for one mechanism."""
    frame = _frame()
    mod = _write_module(tmp_path, "grp", '''
        import pandas as pd
        NAME = "grp"
        MECHANISM_FAMILY = "f1"
        COLUMNS = ["g1", "g2"]
        INPUTS = ["tgp_delta_7d"]
        def add_columns(df):
            out = df.copy()
            out["g1"] = out["tgp_delta_7d"]
            out["g2"] = out["tgp_delta_7d"] * 2.0
            return out
    ''')

    cands = compute_candidate_columns([mod], frame, validate=False)
    rho = pairwise_rho(cands)

    assert len(rho) == 1
    assert rho.iloc[0]["abs_rho"] > 0.99      # perfectly correlated
    assert not rho.iloc[0]["cross_candidate"]  # ... and not a gate violation


def test_block_r2_is_high_for_a_reconstructible_candidate(tmp_path):
    frame = _frame()
    mod = _write_module(tmp_path, "dup", _SIMPLE.format(
        name="dup", family="f1", col="dup_col", src="cycle_pct_through", mult=2.0))

    cands = compute_candidate_columns([mod], frame, validate=False)
    r2 = block_r2(cands, frame)

    assert r2.iloc[0]["block_r2"] > BLOCK_R2_FLAG


def test_block_r2_averages_so_one_reconstructible_member_does_not_condemn_a_group(tmp_path):
    """generator.md's rule, arithmetically: one redundant member among several is not
    disqualifying if the group as a whole isn't."""
    frame = _frame()
    mod = _write_module(tmp_path, "mixed", '''
        import numpy as np
        import pandas as pd
        NAME = "mixed"
        MECHANISM_FAMILY = "f1"
        COLUMNS = ["copy_col", "noise_a", "noise_b"]
        INPUTS = ["cycle_pct_through"]
        def add_columns(df):
            out = df.copy()
            rng = np.random.default_rng(7)
            out["copy_col"] = out["cycle_pct_through"]        # fully reconstructible
            out["noise_a"] = rng.normal(size=len(out))        # not
            out["noise_b"] = rng.normal(size=len(out))        # not
            return out
    ''')

    cands = compute_candidate_columns([mod], frame, validate=False)
    row = block_r2(cands, frame).iloc[0]

    assert row["max_column_r2"] > 0.99      # the member IS reconstructible
    assert row["block_r2"] < BLOCK_R2_FLAG  # the group is not condemned for it


def test_screen_batch_passes_for_genuinely_different_candidates(tmp_path, monkeypatch):
    frame = _frame()
    monkeypatch.setattr("experiments.pipeline.redundancy.load_features", lambda p: frame)
    a = _write_module(tmp_path, "a", _SIMPLE.format(
        name="a", family="f1", col="a_col", src="tgp_delta_7d", mult=1.0))
    b = _write_module(tmp_path, "b", _SIMPLE.format(
        name="b", family="f2", col="b_col", src="stickiness_score", mult=1.0))

    result = screen_batch([a, b], sample_rows=0)

    assert result["passed"]
    assert result["gate_violations"].empty
    assert result["mechanism_families"] == {"a": "f1", "b": "f2"}


def test_screen_batch_fails_and_names_the_offending_pair(tmp_path, monkeypatch):
    frame = _frame()
    monkeypatch.setattr("experiments.pipeline.redundancy.load_features", lambda p: frame)
    a = _write_module(tmp_path, "a", _SIMPLE.format(
        name="a", family="f1", col="a_col", src="tgp_delta_7d", mult=1.0))
    b = _write_module(tmp_path, "b", _SIMPLE.format(
        name="b", family="f2", col="b_col", src="tgp_delta_7d", mult=-5.0))

    result = screen_batch([a, b], sample_rows=0)

    assert not result["passed"]
    assert set(result["gate_violations"].iloc[0][["candidate_a", "candidate_b"]]) == {"a", "b"}


def test_gate_is_sign_blind(tmp_path):
    """A feature and its negative are the same idea; |rho| must catch that."""
    frame = _frame()
    a = _write_module(tmp_path, "a", _SIMPLE.format(
        name="a", family="f1", col="a_col", src="tgp_delta_7d", mult=1.0))
    b = _write_module(tmp_path, "b", _SIMPLE.format(
        name="b", family="f2", col="b_col", src="tgp_delta_7d", mult=-1.0))

    rho = pairwise_rho(compute_candidate_columns([a, b], frame, validate=False))

    assert rho.iloc[0]["abs_rho"] > 0.99


def test_existing_column_set_excludes_rejected_brand_troughs():
    """Reconstructibility from an evaluated-and-REJECTED group is not evidence a
    candidate is redundant with the model."""
    existing = existing_column_set()

    assert "tgp_delta_7d" in existing
    assert not [c for c in existing if c.startswith("days_since_trough_entry_")
                and c not in LGA_FEATURE_COLUMNS]


def test_validation_runs_before_columns_are_correlated(tmp_path):
    """A candidate reaching for the label is caught here, not five nights later."""
    frame = _frame()
    frame["future_min_cents"] = 1.0
    frame["label"] = 0
    mod = _write_module(tmp_path, "leaky", '''
        import pandas as pd
        NAME = "leaky"
        MECHANISM_FAMILY = "f1"
        COLUMNS = ["leak_col"]
        INPUTS = ["future_min_cents"]
        def add_columns(df):
            out = df.copy()
            out["leak_col"] = out["future_min_cents"]
            return out
    ''')

    with pytest.raises(Exception, match="future_min_cents"):
        compute_candidate_columns([mod], frame, validate=True)


def test_single_column_single_candidate_batch_yields_an_empty_but_usable_frame(tmp_path):
    """batch0's exact shape: one candidate, one column, therefore zero pairs.

    Found by smoke-testing the CLI against real data — pd.DataFrame([]) has no columns,
    so sorting or filtering on abs_rho raised KeyError rather than returning nothing.
    """
    frame = _frame()
    mod = _write_module(tmp_path, "solo", _SIMPLE.format(
        name="solo", family="f1", col="solo_col", src="tgp_delta_7d", mult=1.0))

    rho = pairwise_rho(compute_candidate_columns([mod], frame, validate=False))

    assert rho.empty
    assert list(rho.columns) == [
        "column_a", "column_b", "candidate_a", "candidate_b", "cross_candidate", "abs_rho",
    ]
    assert rho[rho["cross_candidate"]].empty  # the filter screen_batch actually does


def test_all_nan_predictor_columns_are_dropped_not_fatal(tmp_path):
    """Five LGA trough columns are 100% NaN in the real frame.

    Requiring complete cases across every predictor therefore selected ZERO rows and
    returned NaN for every candidate — a silently useless screen, found by smoke-testing
    the CLI against live data rather than by any unit test here.
    """
    frame = _frame()
    for dead in LGA_FEATURE_COLUMNS[:5]:
        frame[dead] = np.nan
    mod = _write_module(tmp_path, "dup", _SIMPLE.format(
        name="dup", family="f1", col="dup_col", src="cycle_pct_through", mult=2.0))

    row = block_r2(compute_candidate_columns([mod], frame, validate=False), frame).iloc[0]

    assert np.isfinite(row["block_r2"]), "all-NaN predictors must not poison the fit"
    assert row["block_r2"] > BLOCK_R2_FLAG
    assert row["n_predictors_used"] == len(existing_column_set()) - 5
    assert len(row["predictors_dropped"]) == 5


def test_a_candidate_that_is_an_existing_column_scores_block_r2_one(tmp_path):
    """The screen's own sanity check: proposing a column that already exists must read
    as perfectly reconstructible. This is what the batch0 smoke test asserts by hand."""
    frame = _frame()
    mod = _write_module(tmp_path, "same", _SIMPLE.format(
        name="same", family="f1", col="tgp_copy", src="tgp_delta_7d", mult=1.0))

    row = block_r2(compute_candidate_columns([mod], frame, validate=False), frame).iloc[0]

    assert row["block_r2"] == pytest.approx(1.0, abs=1e-6)


def test_usable_predictors_drops_constant_columns_too():
    frame = _frame()
    frame["cycle_peak_count"] = 3.0          # constant
    frame[LGA_FEATURE_COLUMNS[0]] = np.nan   # all-NaN

    kept = usable_predictors(frame[existing_column_set()])

    assert "cycle_peak_count" not in kept.columns
    assert LGA_FEATURE_COLUMNS[0] not in kept.columns
    assert "cycle_pct_through" in kept.columns


def test_two_candidates_declaring_the_same_column_name_is_refused(tmp_path):
    """A name collision makes the gate unenforceable, silently.

    `pairwise_rho` attributes columns to candidates by name, so the second candidate
    overwrites the first; the colliding pair then resolves to one owner on both sides and
    is classified within-candidate, skipping the gate even at |rho| = 1.0. pandas accepts
    duplicate labels through both concat and corr without complaint, and nothing
    downstream catches it (the runner checks one candidate at a time, on separate
    nights). Refusing here is the only option that works.
    """
    frame = _frame()
    a = _write_module(tmp_path, "a", _SIMPLE.format(
        name="a", family="f1", col="shared_name", src="tgp_delta_7d", mult=1.0))
    b = _write_module(tmp_path, "b", _SIMPLE.format(
        name="b", family="f2", col="shared_name", src="stickiness_score", mult=1.0))

    with pytest.raises(DuplicateCandidateColumn, match="shared_name"):
        compute_candidate_columns([a, b], frame, validate=False)


def test_a_constant_candidate_column_is_not_a_silent_pass(tmp_path, monkeypatch):
    """corr() is NaN against a zero-variance column, and `NaN >= threshold` is False.

    Without surfacing it, a broken add_columns producing a constant column would sail
    through the gate as though it had been checked. Mirrors the silent-NaN treatment
    usable_predictors gives the R^2 side.
    """
    frame = _frame()
    monkeypatch.setattr("experiments.pipeline.redundancy.load_features", lambda p: frame)
    a = _write_module(tmp_path, "a", '''
        import pandas as pd
        NAME = "a"
        MECHANISM_FAMILY = "f1"
        COLUMNS = ["const_col"]
        INPUTS = ["tgp_delta_7d"]
        def add_columns(df):
            out = df.copy()
            out["const_col"] = 1.0
            return out
    ''')
    b = _write_module(tmp_path, "b", _SIMPLE.format(
        name="b", family="f2", col="b_col", src="stickiness_score", mult=1.0))

    result = screen_batch([a, b], sample_rows=0)

    assert not result["passed"], "an uncomputable gate check must not report PASS"
    assert result["gate_violations"].empty          # not a threshold breach...
    assert len(result["uncomputable_pairs"]) == 1   # ...but flagged for a human


# --- Batch record (generator.md § Batch record) -----------------------------------


def _screen_two(tmp_path, monkeypatch, family_a="f1", family_b="f2"):
    frame = _frame()
    monkeypatch.setattr("experiments.pipeline.redundancy.load_features", lambda p: frame)
    a = _write_module(tmp_path, "a", _SIMPLE.format(
        name="a", family=family_a, col="a_col", src="tgp_delta_7d", mult=1.0))
    b = _write_module(tmp_path, "b", _SIMPLE.format(
        name="b", family=family_b, col="b_col", src="stickiness_score", mult=1.0))
    return screen_batch([a, b], sample_rows=0)


def test_batch_record_writes_both_formats(tmp_path, monkeypatch):
    result = _screen_two(tmp_path, monkeypatch)

    md_path, json_path = write_batch_record(result, tmp_path / "batch9")

    payload = json.loads(json_path.read_text())
    assert payload["mechanism_families"] == {"a": "f1", "b": "f2"}
    assert payload["n_distinct_families"] == 2
    assert payload["rho_threshold"] == PAIRWISE_RHO_THRESHOLD
    assert len(payload["block_r2"]) == 2
    assert md_path.read_text().startswith("# Batch record — batch9")


def test_batch_record_calls_out_a_single_family_batch(tmp_path, monkeypatch):
    """The disclosure's entire purpose: five uncorrelated candidates telling one story
    reads as diversity and isn't. It is not gated, so it has to be VISIBLE."""
    result = _screen_two(tmp_path, monkeypatch, family_a="same", family_b="same")

    md_path, json_path = write_batch_record(result, tmp_path / "batch9")

    assert json.loads(json_path.read_text())["n_distinct_families"] == 1
    assert "Every candidate in this batch shares one family label" in md_path.read_text()


def test_batch_record_flags_an_undeclared_family(tmp_path, monkeypatch):
    frame = _frame()
    monkeypatch.setattr("experiments.pipeline.redundancy.load_features", lambda p: frame)
    a = _write_module(tmp_path, "a", '''
        import pandas as pd
        NAME = "a"
        COLUMNS = ["a_col"]
        INPUTS = ["tgp_delta_7d"]
        def add_columns(df):
            out = df.copy()
            out["a_col"] = out["tgp_delta_7d"]
            return out
    ''')
    result = screen_batch([a], sample_rows=0)

    md_path, json_path = write_batch_record(result, tmp_path / "batch9")

    assert json.loads(json_path.read_text())["undeclared_families"] == ["a"]
    assert "**NOT DECLARED**" in md_path.read_text()


def test_batch_record_separates_gated_from_disclosed_pairs(tmp_path, monkeypatch):
    """The record has to preserve the distinction the gate makes, or it misrepresents
    why the batch passed."""
    frame = _frame()
    monkeypatch.setattr("experiments.pipeline.redundancy.load_features", lambda p: frame)
    grp = _write_module(tmp_path, "grp", '''
        import pandas as pd
        NAME = "grp"
        MECHANISM_FAMILY = "f1"
        COLUMNS = ["g1", "g2"]
        INPUTS = ["tgp_delta_7d"]
        def add_columns(df):
            out = df.copy()
            out["g1"] = out["tgp_delta_7d"]
            out["g2"] = out["tgp_delta_7d"] * 2.0
            return out
    ''')
    other = _write_module(tmp_path, "oth", _SIMPLE.format(
        name="oth", family="f2", col="o_col", src="stickiness_score", mult=1.0))
    result = screen_batch([grp, other], sample_rows=0)

    md_path, json_path = write_batch_record(result, tmp_path / "batch9")
    text = md_path.read_text()
    payload = json.loads(json_path.read_text())

    assert result["passed"], "a within-candidate 1.0 pair must not fail the batch"
    assert "### Within-candidate (disclosure only)" in text
    assert "### Cross-candidate (gated)" in text
    within = [r for r in payload["pairwise_rho"] if not r["cross_candidate"]]
    assert len(within) == 1 and within[0]["candidate_a"] == "grp"
