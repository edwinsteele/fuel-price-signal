"""Tests for experiments/pipeline/noise_floor.py — the R0-vs-R0 noise-floor producer (fps-3jj.9).

Like test_runner.py, the real DB-backed realised backtest isn't reproduced here —
run_paired_realised_backtest is monkeypatched so these tests exercise the seed-pairing
and persistence logic in isolation.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest
from click.testing import CliRunner

import experiments.pipeline.noise_floor as noise_floor_module
from experiments.pipeline.batch_freeze import BaselineContractMismatch, resolve_baseline_columns
from experiments.pipeline.dossier_tables import NOISE_FLOOR_FILENAME
from experiments.pipeline.noise_floor import compute_noise_floor, main
from experiments.pipeline.runner import DEFAULT_INNER_FOLD_PARAMS
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    baseline_fingerprint,
)


def _baseline_features_df(n: int = 5) -> pd.DataFrame:
    dates = [20260810, 20260811, 20260812, 20260813, 20260814][:n]
    cols = {"price_date": dates}
    for c in FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS:
        cols[c] = [0.0] * n
    return pd.DataFrame(cols)


def _write_batch_dir(tmp_path: pathlib.Path, df: pd.DataFrame, name: str = "batch1") -> pathlib.Path:
    batch_dir = tmp_path / name
    batch_dir.mkdir()
    df.to_parquet(batch_dir / "features.parquet")
    (batch_dir / "baseline_columns.json").write_text(json.dumps(resolve_baseline_columns(df)))
    (batch_dir / "fuel_signal.db").write_bytes(b"not a real sqlite file")
    return batch_dir


def _fake_result(cpl_held: float, n_windows: int = 5):
    class _Fake:
        aggregate = pd.DataFrame([{"arm": "baseline", "cpl_own": cpl_held, "cpl_held": cpl_held}])
        meta = {"n_windows": n_windows}

    return _Fake()


def _stub_realised_by_seed(
    monkeypatch, cpl_by_seed: dict[int, float], n_windows_by_seed: dict[int, int] | None = None
) -> list[dict]:
    """Route each call to its seed's canned cpl_held (+ optional n_windows); record
    every call's args/kwargs."""
    calls: list[dict] = []
    n_windows_by_seed = n_windows_by_seed or {}

    def _fake(arms, feature_columns, **kwargs):
        calls.append({"arms": arms, "feature_columns": feature_columns, **kwargs})
        seed = kwargs["seed"]
        return _fake_result(cpl_by_seed[seed], n_windows_by_seed.get(seed, 5))

    monkeypatch.setattr(noise_floor_module, "run_paired_realised_backtest", _fake)
    return calls


def test_compute_noise_floor_pairs_seeds_and_writes_expected_deltas(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    # seeds_a=(1, 2), seeds_b=(10, 20) -> deltas [10-1, 20-2] = [9, 18]
    cpl_by_seed = {1: 100.0, 2: 200.0, 10: 109.0, 20: 218.0}
    _stub_realised_by_seed(monkeypatch, cpl_by_seed)

    payload = compute_noise_floor(
        batch_dir, seeds_a=(1, 2), seeds_b=(10, 20), verbose=False,
    )

    assert payload["deltas_cpl_held"] == [9.0, 18.0]
    assert payload["seeds_a"] == [1, 2]
    assert payload["seeds_b"] == [10, 20]
    assert payload["fold_subset"] is None
    assert payload["partial"] is False
    on_disk = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    assert on_disk == payload


def test_compute_noise_floor_mismatched_seed_lengths_raises(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    with pytest.raises(ValueError, match="pair 1:1"):
        compute_noise_floor(batch_dir, seeds_a=(1, 2), seeds_b=(10,))


def test_compute_noise_floor_repeated_seed_within_a_group_raises(tmp_path):
    """A repeated seed in one group pairs a fit against a duplicate of itself,
    which is harmless on its own but signals a caller error worth catching early."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    with pytest.raises(ValueError, match="free of repeats"):
        compute_noise_floor(batch_dir, seeds_a=(1, 1), seeds_b=(10, 20))


def test_compute_noise_floor_overlapping_seed_groups_raises(tmp_path):
    """A seed shared between seeds_a and seeds_b pairs a fit against itself for that
    draw — a guaranteed zero delta that silently narrows the floor."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    with pytest.raises(ValueError, match="disjoint"):
        compute_noise_floor(batch_dir, seeds_a=(1, 2), seeds_b=(2, 3))


def test_compute_noise_floor_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    with pytest.raises(FileExistsError, match="already exists"):
        compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)


def test_compute_noise_floor_overwrites_existing_file_with_force(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    payload = compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), force=True, verbose=False)

    assert payload["deltas_cpl_held"] == [1.0]


def test_compute_noise_floor_raises_on_baseline_contract_drift(tmp_path, monkeypatch):
    """The frozen baseline_columns.json is checked against the current code's own
    resolution (same call every candidate run makes), catching a mid-batch
    FEATURE_COLUMNS change rather than silently fitting the stale list."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    stale = resolve_baseline_columns(df)[:-1]  # drop one locked column -> drift
    (batch_dir / "baseline_columns.json").write_text(json.dumps(stale))
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    with pytest.raises(BaselineContractMismatch):
        compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)


def test_compute_noise_floor_calls_one_baseline_only_arm_per_seed(tmp_path, monkeypatch):
    """Every call is a single baseline-only arm with the frozen baseline columns —
    no candidate arm is involved, since this measures pure fit noise, not a
    feature's effect."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    calls = _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)

    assert len(calls) == 2
    assert {c["seed"] for c in calls} == {1, 2}
    for c in calls:
        assert len(c["arms"]) == 1
        assert c["arms"][0].name == noise_floor_module.BASELINE_ARM_NAME
        assert c["arms"][0].feature_columns == baseline_columns
        assert c["feature_columns"] == baseline_columns
        assert c["held_tau"] is None
        assert c["collect_fills"] is False


def test_compute_noise_floor_defaults_inner_fold_params(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)

    for c in calls:
        assert c["inner_fold_params"] == DEFAULT_INNER_FOLD_PARAMS


def test_compute_noise_floor_merges_partial_inner_fold_params(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    compute_noise_floor(
        batch_dir, seeds_a=(1,), seeds_b=(2,),
        inner_fold_params={"val_days": 30}, verbose=False,
    )

    for c in calls:
        assert c["inner_fold_params"]["val_days"] == 30
        assert c["inner_fold_params"]["train_min_days"] == DEFAULT_INNER_FOLD_PARAMS["train_min_days"]


def test_compute_noise_floor_stamps_provenance(tmp_path, monkeypatch):
    """fps-cf8 acceptance: baseline_fingerprint, fold geometry, and wall_seconds must
    be in the written payload, not just deltas/seeds."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0}, {1: 7, 2: 7})

    payload = compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)

    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    assert payload["baseline_fingerprint"] == baseline_fingerprint(baseline_columns)
    assert payload["n_windows"] == 7
    assert payload["outer_fold_params"] == {}
    assert payload["inner_fold_params"] == DEFAULT_INNER_FOLD_PARAMS
    assert payload["wall_seconds"] >= 0.0


def test_compute_noise_floor_raises_on_n_windows_mismatch_within_pair(tmp_path, monkeypatch):
    """Both halves of a pair walk the same frozen data + fold geometry; if they
    disagree on n_windows, something (e.g. a mid-run config change) broke that
    invariant and the resulting delta is not trustworthy."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0}, {1: 7, 2: 8})

    with pytest.raises(ValueError, match="fold geometry must match"):
        compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)


def test_compute_noise_floor_raises_on_n_windows_mismatch_across_pairs(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}, {1: 7, 2: 7, 3: 8, 4: 8})

    with pytest.raises(ValueError, match="fold geometry must be identical"):
        compute_noise_floor(batch_dir, seeds_a=(1, 3), seeds_b=(2, 4), verbose=False)


def test_compute_noise_floor_default_seeds_pair_five_and_five(tmp_path, monkeypatch):
    """SEEDS_A/SEEDS_B (the module defaults) are the fps-3jj.9 'starting idea':
    production SEEDS (42..46) paired index-wise against a disjoint 47..51."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    all_seeds = set(noise_floor_module.SEEDS_A) | set(noise_floor_module.SEEDS_B)
    calls = _stub_realised_by_seed(monkeypatch, dict.fromkeys(all_seeds, 0.0))

    payload = compute_noise_floor(batch_dir, verbose=False)

    assert noise_floor_module.SEEDS_A == (42, 43, 44, 45, 46)
    assert len(noise_floor_module.SEEDS_A) == len(noise_floor_module.SEEDS_B) == 5
    assert set(noise_floor_module.SEEDS_A).isdisjoint(noise_floor_module.SEEDS_B)
    assert len(calls) == 10
    assert len(payload["deltas_cpl_held"]) == 5


def test_compute_noise_floor_marks_fold_subset_runs_partial(tmp_path, monkeypatch):
    """--fold-subset is an iteration/smoke speed-up; the written file must say so
    rather than looking like a full calibration (see the dossier-consumption test)."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    payload = compute_noise_floor(
        batch_dir, seeds_a=(1,), seeds_b=(2,), fold_subset=[1, 3], verbose=False,
    )

    assert payload["partial"] is True
    assert payload["fold_subset"] == [1, 3]


def _matching_fingerprint(batch_dir: pathlib.Path) -> str:
    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    return baseline_fingerprint(baseline_columns)


def test_dossier_tables_consumes_the_written_file(tmp_path, monkeypatch):
    """End-to-end schema check: what this module writes is exactly what
    dossier_tables._noise_band() reads (the fps-3jj.9 contract)."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 100.0, 2: 200.0, 10: 109.0, 20: 218.0})
    compute_noise_floor(batch_dir, seeds_a=(1, 2), seeds_b=(10, 20), verbose=False)
    fp = _matching_fingerprint(batch_dir)

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": fp}}, batch_dir
    )

    assert band["available"] is True
    assert band["n_draws"] == 2


def test_dossier_tables_refuses_a_partial_fold_subset_file(tmp_path, monkeypatch):
    """A --fold-subset run's output must not be silently graded as a full ruler."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})
    compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), fold_subset=[1], verbose=False)
    fp = _matching_fingerprint(batch_dir)

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": fp}}, batch_dir
    )

    assert band["available"] is False
    assert "partial" in band["reason"] or "fold-subset" in band["reason"]


def test_dossier_tables_refuses_a_mismatched_fingerprint(tmp_path, monkeypatch):
    """fps-cf8: a floor computed against one baseline must not silently grade a
    candidate run stamped with a different baseline_fingerprint — this is the exact
    failure class (fps-sa1, fps-zci) the noise floor exists to guard other candidates
    against, and until now the floor itself was ungated."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})
    compute_noise_floor(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": "54:deadbeefcafe"}},
        batch_dir,
    )

    assert band["available"] is False
    assert "fingerprint" in band["reason"]


def test_dossier_tables_refuses_a_floor_with_no_fingerprint(tmp_path, monkeypatch):
    """A pre-fps-cf8 floor (no baseline_fingerprint at all) cannot be shown to match,
    so it must not be treated as a pass by default."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": "54:whatever"}},
        batch_dir,
    )

    assert band["available"] is False
    assert "fingerprint" in band["reason"]


def test_cli_writes_noise_floor_for_named_batch(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    _stub_realised_by_seed(
        monkeypatch,
        dict.fromkeys(set(noise_floor_module.SEEDS_A) | set(noise_floor_module.SEEDS_B), 0.0),
    )

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir)])

    assert result.exit_code == 0, result.output
    assert (batch_dir / NOISE_FLOOR_FILENAME).exists()


def test_cli_parses_fold_subset(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    _write_batch_dir(batches_dir, df)
    calls = _stub_realised_by_seed(
        monkeypatch,
        dict.fromkeys(set(noise_floor_module.SEEDS_A) | set(noise_floor_module.SEEDS_B), 0.0),
    )

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--fold-subset", "1,3"]
    )

    assert result.exit_code == 0, result.output
    for c in calls:
        assert c["fold_subset"] == [1, 3]


def test_cli_rejects_non_numeric_fold_subset(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    _write_batch_dir(batches_dir, df)
    _stub_realised_by_seed(monkeypatch, {})  # must never be called

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--fold-subset", "1,abc"]
    )

    assert result.exit_code != 0
    assert "not an integer" in result.output


def test_cli_rejects_non_positive_fold_subset(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    _write_batch_dir(batches_dir, df)
    _stub_realised_by_seed(monkeypatch, {})  # must never be called

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--fold-subset", "0,2"]
    )

    assert result.exit_code != 0
    assert "1-indexed" in result.output


def test_cli_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_seed(monkeypatch, {})  # must never be called

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir)])

    assert result.exit_code != 0


def test_cli_force_overwrites_existing_file(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_seed(
        monkeypatch,
        dict.fromkeys(set(noise_floor_module.SEEDS_A) | set(noise_floor_module.SEEDS_B), 0.0),
    )

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir), "--force"])

    assert result.exit_code == 0, result.output
    on_disk = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    assert on_disk["deltas_cpl_held"] == [0.0] * 5
