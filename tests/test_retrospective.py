"""Tests for experiments/pipeline/retrospective.py — the batch retrospective (fps-3jj.8)."""
from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from experiments.pipeline.retrospective import (
    RETROSPECTIVE_FILENAME,
    build_confidence_calibration,
    build_leaderboard,
    build_outcome_tally,
    compute_retrospective,
    family_wise_percentile_threshold,
    find_batch_candidates,
    main,
)


def _facts(
    name: str = "cand",
    batch: str = "batch1",
    status: str = "rejected",
    delta: float = 0.01,
    effect_resolved: bool | None = False,
    zone_resolved: bool | None = None,
    confidence_effect: float = 0.5,
    confidence_zone: float = 0.5,
    noise_band: dict | None = None,
) -> dict:
    return {
        "candidate": {"name": name, "confidence_effect": confidence_effect, "confidence_zone": confidence_zone},
        "provenance": {"batch": batch, "status": status},
        "headline": {
            "realised": {"delta_cpl_held": delta, "effect_resolved": effect_resolved},
            "zone": {"resolved": zone_resolved},
        },
        "noise_band": noise_band or {"available": False},
    }


def _write_candidate_module(candidates_root: pathlib.Path, batch: str, name: str) -> pathlib.Path:
    batch_dir = candidates_root / batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    module_path = batch_dir / f"{name}.py"
    module_path.write_text("NAME = %r\n" % name)
    return module_path


def _write_dossier(candidates_root: pathlib.Path, batch: str, name: str, facts: dict) -> pathlib.Path:
    """Candidate module + its run dir (default_out_dir strips .py) + facts.json."""
    module_path = _write_candidate_module(candidates_root, batch, name)
    out_dir = module_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "facts.json").write_text(json.dumps(facts))
    return module_path


# ── find_batch_candidates / _candidate_entry states ────────────────────────

def test_find_batch_candidates_lists_py_files_excludes_init(tmp_path):
    candidates_root = tmp_path / "candidates"
    _write_candidate_module(candidates_root, "batch1", "alpha")
    _write_candidate_module(candidates_root, "batch1", "beta")
    (candidates_root / "batch1" / "__init__.py").write_text("")

    found = find_batch_candidates("batch1", candidates_root)

    assert [p.stem for p in found] == ["alpha", "beta"]


def test_find_batch_candidates_missing_batch_dir_returns_empty(tmp_path):
    assert find_batch_candidates("nope", tmp_path / "candidates") == []


def test_compute_retrospective_never_run_candidate_has_no_leaderboard_row(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_candidate_module(candidates_root, "batch1", "never_ran")
    (batches_dir / "batch1").mkdir(parents=True)

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    assert payload["leaderboard"] == []
    assert payload["outcome_tally"]["never_run"] == 1
    assert payload["outcome_tally"]["total_candidates_filed"] == 1


def test_compute_retrospective_retryable_status_not_folded_into_rejected(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    module_path = _write_candidate_module(candidates_root, "batch1", "flaky")
    out_dir = module_path.with_suffix("")
    out_dir.mkdir()
    (out_dir / "results.json").write_text(json.dumps({"status": "aborted_pipeline"}))
    (batches_dir / "batch1").mkdir(parents=True)

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    assert payload["outcome_tally"]["retryable_incomplete"] == 1
    assert payload["outcome_tally"]["dossiered_by_status"] == {}


def test_compute_retrospective_terminal_status_no_facts_yet_is_pending_dossier(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    module_path = _write_candidate_module(candidates_root, "batch1", "just_finished")
    out_dir = module_path.with_suffix("")
    out_dir.mkdir()
    (out_dir / "results.json").write_text(json.dumps({"status": "rejected"}))
    (batches_dir / "batch1").mkdir(parents=True)

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    assert payload["outcome_tally"]["pending_dossier"] == 1
    assert payload["leaderboard"] == []


# ── family_wise_percentile_threshold ────────────────────────────────────────

def test_family_wise_threshold_n1_is_plain_95th_percentile():
    assert family_wise_percentile_threshold(1) == 95.0


def test_family_wise_threshold_rises_with_n():
    assert family_wise_percentile_threshold(5) == pytest.approx(99.0)
    assert family_wise_percentile_threshold(1) < family_wise_percentile_threshold(5)


def test_family_wise_threshold_rejects_non_positive_n():
    with pytest.raises(ValueError, match=">= 1"):
        family_wise_percentile_threshold(0)


# ── build_leaderboard ────────────────────────────────────────────────────

def test_build_leaderboard_ranks_by_noise_percentile_descending():
    low_band = {"available": True, "candidate_percentile_better_than_noise": 20.0, "candidate_z_vs_band": 1.0}
    high_band = {"available": True, "candidate_percentile_better_than_noise": 90.0, "candidate_z_vs_band": -1.5}
    entries = [
        {"candidate": "low", "state": "dossiered", "facts": _facts("low", delta=0.02, noise_band=low_band)},
        {"candidate": "high", "state": "dossiered", "facts": _facts("high", delta=-0.05, noise_band=high_band)},
    ]

    rows = build_leaderboard(entries, family_wise_threshold=95.0)

    assert [r["candidate"] for r in rows] == ["high", "low"]


def test_build_leaderboard_falls_back_to_raw_delta_when_no_noise_band():
    entries = [
        {"candidate": "worse", "state": "dossiered", "facts": _facts("worse", delta=0.05)},
        {"candidate": "better", "state": "dossiered", "facts": _facts("better", delta=-0.05)},
    ]

    rows = build_leaderboard(entries, family_wise_threshold=95.0)

    # delta_cpl_held is a COST: more negative is better, so ascending sort.
    assert [r["candidate"] for r in rows] == ["better", "worse"]


def test_build_leaderboard_clears_family_wise_threshold_flag():
    band = {"available": True, "candidate_percentile_better_than_noise": 96.0, "candidate_z_vs_band": -2.0}
    entries = [
        {"candidate": "borderline", "state": "dossiered", "facts": _facts("borderline", noise_band=band)},
    ]

    rows = build_leaderboard(entries, family_wise_threshold=95.0)

    assert rows[0]["clears_family_wise_threshold"] is True


def test_build_leaderboard_skips_non_dossiered_entries():
    entries = [
        {"candidate": "ran", "state": "dossiered", "facts": _facts("ran")},
        {"candidate": "not_yet", "state": "never_run", "facts": None},
    ]

    rows = build_leaderboard(entries, family_wise_threshold=95.0)

    assert [r["candidate"] for r in rows] == ["ran"]


# ── build_outcome_tally ────────────────────────────────────────────────────

def test_build_outcome_tally_counts_every_state():
    entries = [
        {"candidate": "a", "state": "dossiered", "facts": _facts("a", status="rejected")},
        {"candidate": "b", "state": "dossiered", "facts": _facts("b", status="disqualified")},
        {"candidate": "c", "state": "never_run", "facts": None},
        {"candidate": "d", "state": "retryable_incomplete", "facts": None},
        {"candidate": "e", "state": "pending_dossier", "facts": None},
    ]

    tally = build_outcome_tally(entries)

    assert tally == {
        "total_candidates_filed": 5,
        "dossiered_by_status": {"rejected": 1, "disqualified": 1},
        "never_run": 1,
        "retryable_incomplete": 1,
        "pending_dossier": 1,
    }


# ── build_confidence_calibration ────────────────────────────────────────────

def test_confidence_calibration_reports_insufficient_data_below_min_n(tmp_path):
    candidates_root = tmp_path / "candidates"
    _write_dossier(candidates_root, "batch1", "only_one", _facts("only_one"))

    result = build_confidence_calibration(candidates_root)

    assert result["n_dossiered_with_resolved_effect"] == 1
    assert result["insufficient_data"] is True
    assert result["mean_confidence_effect_when_resolved_true"] is None
    assert len(result["pairs"]) == 1


def test_confidence_calibration_scans_every_batch_not_just_one(tmp_path):
    candidates_root = tmp_path / "candidates"
    _write_dossier(candidates_root, "batch1", "a", _facts("a", batch="batch1"))
    _write_dossier(candidates_root, "batch2", "b", _facts("b", batch="batch2"))

    result = build_confidence_calibration(candidates_root)

    assert {p["batch"] for p in result["pairs"]} == {"batch1", "batch2"}


def test_confidence_calibration_computes_means_once_min_n_reached(tmp_path, monkeypatch):
    import experiments.pipeline.retrospective as retro_module

    monkeypatch.setattr(retro_module, "MIN_CALIBRATION_N", 2)
    candidates_root = tmp_path / "candidates"
    _write_dossier(candidates_root, "batch1", "won", _facts("won", confidence_effect=0.8, effect_resolved=True))
    _write_dossier(candidates_root, "batch1", "lost", _facts("lost", confidence_effect=0.2, effect_resolved=False))

    result = retro_module.build_confidence_calibration(candidates_root)

    assert result["insufficient_data"] is False
    assert result["mean_confidence_effect_when_resolved_true"] == pytest.approx(0.8)
    assert result["mean_confidence_effect_when_resolved_false"] == pytest.approx(0.2)


# ── compute_retrospective / CLI ─────────────────────────────────────────────

def test_compute_retrospective_writes_expected_file(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1"))
    (batches_dir / "batch1").mkdir(parents=True)

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    on_disk = json.loads((batches_dir / "batch1" / RETROSPECTIVE_FILENAME).read_text())
    assert on_disk == payload
    assert payload["batch"] == "batch1"
    assert len(payload["leaderboard"]) == 1


def test_compute_retrospective_refuses_to_overwrite_without_force(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    (batches_dir / "batch1").mkdir(parents=True)
    (batches_dir / "batch1" / RETROSPECTIVE_FILENAME).write_text("{}")

    with pytest.raises(FileExistsError, match="already exists"):
        compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)


def test_compute_retrospective_force_overwrites(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1"))
    (batches_dir / "batch1").mkdir(parents=True)
    (batches_dir / "batch1" / RETROSPECTIVE_FILENAME).write_text("{}")

    payload = compute_retrospective(
        "batch1", batches_dir=batches_dir, candidates_root=candidates_root, force=True
    )

    assert payload["batch"] == "batch1"


def test_cli_writes_retrospective_for_named_batch(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1"))
    (batches_dir / "batch1").mkdir(parents=True)

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--candidates-dir", str(candidates_root)]
    )

    assert result.exit_code == 0, result.output
    assert (batches_dir / "batch1" / RETROSPECTIVE_FILENAME).exists()


def test_cli_refuses_to_overwrite_without_force(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    (batches_dir / "batch1").mkdir(parents=True)
    (batches_dir / "batch1" / RETROSPECTIVE_FILENAME).write_text("{}")

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--candidates-dir", str(candidates_root)]
    )

    assert result.exit_code != 0
