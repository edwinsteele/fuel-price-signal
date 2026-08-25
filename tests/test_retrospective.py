"""Tests for experiments/pipeline/retrospective.py — the batch retrospective (fps-3jj.8)."""
from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from experiments.lib.io import artifact_has_unstamped_cpl
from experiments.pipeline.retrospective import (
    RETROSPECTIVE_FILENAME,
    _candidate_entry,
    build_confidence_calibration,
    build_leaderboard,
    build_outcome_tally,
    compute_retrospective,
    family_wise_percentile_threshold,
    family_wise_z_threshold,
    find_batch_candidates,
    main,
)

# The cadence stamp dossier_tables.build_facts writes into every graded facts.json
# (fps-15c) — format_tank_params(TankParams()), the repo's canonical 1d tank. Literal
# rather than computed: these fixtures are asserting on the SHAPE a retrospective row
# propagates, and a fixture that re-derived the string from TankParams could not catch a
# propagation bug that dropped it.
TANK_1D = "50/3.571/1d/10%"


def _facts(
    name: str = "cand",
    batch: str = "batch1",
    status: str = "graded",
    delta: float = 0.01,
    effect_resolved: bool | None = False,
    zone_resolved: bool | None = None,
    confidence_effect: float = 0.5,
    confidence_zone: float = 0.5,
    noise_band: dict | None = None,
    mechanism_family: str | None = "test-family",
    abort_reason: str | None = None,
    tank_params: str | None = TANK_1D,
) -> dict:
    return {
        "candidate": {"name": name, "confidence_effect": confidence_effect,
                      "confidence_zone": confidence_zone, "mechanism_family": mechanism_family},
        "provenance": {"batch": batch, "status": status, "abort_reason": abort_reason,
                       "tank_params": tank_params},
        "headline": {
            "realised": {"delta_cpl_held": delta, "effect_resolved": effect_resolved},
            "zone": {"resolved": zone_resolved},
        },
        "noise_band": noise_band or {"available": False},
    }


def _disqualified_facts(name: str = "cand", batch: str = "batch1") -> dict:
    """The real on-disk shape dossier_tables.build_facts writes for a terminal status
    other than 'graded' (disqualified / aborted_candidate never reach the scoring
    stages): headline and breakdowns are None, not missing keys."""
    return {
        "candidate": {"name": name, "confidence_effect": 0.4, "confidence_zone": None},
        "provenance": {"batch": batch, "status": "disqualified"},
        "headline": None,
        "breakdowns": None,
        "status_note": "status='disqualified' — run did not reach the scoring stages",
        "noise_band": {"available": False, "reason": "empty noise-floor sample or unresolved effect_delta_cpl_held"},
        "grading": {"predicted_signature": "...", "verdict": None, "explanation": None, "pending": True},
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


def test_compute_retrospective_retryable_status_not_folded_into_graded(tmp_path):
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
    (out_dir / "results.json").write_text(json.dumps({"status": "graded"}))
    (batches_dir / "batch1").mkdir(parents=True)

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    assert payload["outcome_tally"]["pending_dossier"] == 1
    assert payload["leaderboard"] == []


def test_compute_retrospective_missing_batch_dir_raises_clear_error(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"  # batch1 subdir deliberately never created

    with pytest.raises(ValueError, match="does not exist"):
        compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)


def test_compute_retrospective_no_candidate_modules_raises_clear_error(tmp_path):
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    (batches_dir / "batch1").mkdir(parents=True)  # batch exists, but no candidates filed

    with pytest.raises(ValueError, match="No candidate modules found"):
        compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)


# ── _candidate_entry ────────────────────────────────────────────────────────

def test_candidate_entry_retryable_status_wins_over_stale_facts_json(tmp_path):
    """A candidate manually re-queued after being dossiered once: run_candidate deletes
    results.json up front on every (re-)run but never touches a previous facts.json
    (runner.py's own comment on why). If the re-run goes retryable again, the retrospective
    must report retryable_incomplete, not resurrect the stale dossiered facts."""
    candidates_root = tmp_path / "candidates"
    module_path = _write_dossier(candidates_root, "batch1", "requeued", _facts("requeued"))
    out_dir = module_path.with_suffix("")
    (out_dir / "results.json").write_text(json.dumps({"status": "aborted_pipeline"}))

    entry = _candidate_entry(module_path)

    assert entry["state"] == "retryable_incomplete"
    assert entry["facts"] is None


def test_candidate_entry_corrupted_facts_json_raises_with_candidate_name(tmp_path):
    candidates_root = tmp_path / "candidates"
    module_path = _write_candidate_module(candidates_root, "batch1", "broken")
    out_dir = module_path.with_suffix("")
    out_dir.mkdir()
    (out_dir / "facts.json").write_text("{not valid json")

    with pytest.raises(ValueError, match="broken"):
        _candidate_entry(module_path)


# ── family_wise_percentile_threshold ────────────────────────────────────────

def test_family_wise_threshold_n1_is_plain_95th_percentile():
    assert family_wise_percentile_threshold(1) == 95.0


def test_family_wise_threshold_rises_with_n():
    assert family_wise_percentile_threshold(5) == pytest.approx(99.0)
    assert family_wise_percentile_threshold(1) < family_wise_percentile_threshold(5)


def test_family_wise_threshold_rejects_non_positive_n():
    with pytest.raises(ValueError, match=">= 1"):
        family_wise_percentile_threshold(0)


# ── family_wise_z_threshold ─────────────────────────────────────────────────

def test_family_wise_z_threshold_n1_matches_the_prediction_interval_formula():
    """n_candidates=1 -> alpha unmodified; the returned threshold is the raw one-tailed
    t-critical value inflated by sqrt(1 + 1/n_draws) — the prediction-interval correction
    for comparing a NEW observation against a mean/std estimated from n_draws samples."""
    import math

    from scipy.stats import t as t_dist

    expected = t_dist.ppf(0.95, df=19) * math.sqrt(1 + 1 / 20)
    assert family_wise_z_threshold(1, 20) == pytest.approx(expected)


def test_family_wise_z_threshold_exceeds_the_uncorrected_t_critical_value():
    """The sqrt(1 + 1/n_draws) factor must make the threshold STRICTER (larger), not just
    different — omitting it would make the gate too lax (review finding: ~2.5% lax at
    n_draws=20, ~10% at n_draws=5)."""
    from scipy.stats import t as t_dist

    assert family_wise_z_threshold(1, 20) > t_dist.ppf(0.95, df=19)
    assert family_wise_z_threshold(1, 5) > t_dist.ppf(0.95, df=4)


def test_family_wise_z_threshold_rises_with_n_candidates():
    assert family_wise_z_threshold(1, 20) < family_wise_z_threshold(5, 20)


def test_family_wise_z_threshold_falls_with_more_draws():
    """More draws -> more degrees of freedom -> a less extreme t-critical value (converges
    toward the normal critical value as n_draws grows)."""
    assert family_wise_z_threshold(5, 40) < family_wise_z_threshold(5, 5)


def test_family_wise_z_threshold_rejects_non_positive_n_candidates():
    with pytest.raises(ValueError, match=">= 1"):
        family_wise_z_threshold(0, 20)


def test_family_wise_z_threshold_rejects_too_few_draws():
    with pytest.raises(ValueError, match=">= 2"):
        family_wise_z_threshold(5, 1)


# ── build_leaderboard ────────────────────────────────────────────────────

def test_build_leaderboard_ranks_by_noise_band_z_ascending():
    """fps-awz: ranking is z-based (more negative = better, delta_cpl_held is a cost), not
    percentile-based — percentile only has n_draws+1 distinct values and could tie/disagree
    with the gate's own ordering at the ~20-draw default."""
    worse_z = {"available": True, "candidate_percentile_better_than_noise": 20.0, "candidate_z_vs_band": 1.0}
    better_z = {"available": True, "candidate_percentile_better_than_noise": 90.0, "candidate_z_vs_band": -1.5}
    entries = [
        {"candidate": "worse", "state": "dossiered", "facts": _facts("worse", delta=0.02, noise_band=worse_z)},
        {"candidate": "better", "state": "dossiered", "facts": _facts("better", delta=-0.05, noise_band=better_z)},
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=2.0)

    assert [r["candidate"] for r in rows] == ["better", "worse"]


def test_build_leaderboard_falls_back_to_percentile_when_z_unavailable_batch_wide():
    """When z can't be computed for the whole batch (e.g. noise_band_z is None on every
    row) but the band itself is available, percentile is still a real ordering signal —
    fall back to it rather than raw delta."""
    band = {"available": True, "candidate_percentile_better_than_noise": 20.0, "candidate_z_vs_band": None}
    band_high = {"available": True, "candidate_percentile_better_than_noise": 90.0, "candidate_z_vs_band": None}
    entries = [
        {"candidate": "worse", "state": "dossiered", "facts": _facts("worse", delta=0.02, noise_band=band)},
        {"candidate": "better", "state": "dossiered", "facts": _facts("better", delta=-0.05, noise_band=band_high)},
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=None)

    assert [r["candidate"] for r in rows] == ["better", "worse"]


def test_build_leaderboard_falls_back_to_raw_delta_when_no_noise_band():
    entries = [
        {"candidate": "worse", "state": "dossiered", "facts": _facts("worse", delta=0.05)},
        {"candidate": "better", "state": "dossiered", "facts": _facts("better", delta=-0.05)},
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=2.0)

    # delta_cpl_held is a COST: more negative is better, so ascending sort.
    assert [r["candidate"] for r in rows] == ["better", "worse"]


def test_build_leaderboard_clears_family_wise_threshold_flag_reads_z_not_percentile():
    """fps-awz: the gate is z <= -threshold, not percentile >= threshold. A 96th-percentile
    candidate whose z doesn't clear the bar must NOT pass — this is exactly the failure mode
    the rework fixes (percentile alone couldn't resolve past 'beat every draw')."""
    clears = {"available": True, "candidate_percentile_better_than_noise": 96.0, "candidate_z_vs_band": -2.5}
    percentile_only = {"available": True, "candidate_percentile_better_than_noise": 96.0, "candidate_z_vs_band": -1.0}
    entries = [
        {"candidate": "clears", "state": "dossiered", "facts": _facts("clears", noise_band=clears)},
        {
            "candidate": "percentile_only", "state": "dossiered",
            "facts": _facts("percentile_only", noise_band=percentile_only),
        },
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=2.0)

    clears_row = next(r for r in rows if r["candidate"] == "clears")
    percentile_only_row = next(r for r in rows if r["candidate"] == "percentile_only")
    assert clears_row["clears_family_wise_threshold"] is True
    assert percentile_only_row["clears_family_wise_threshold"] is False
    # Percentile is still reported, as descriptive colour, for both.
    assert clears_row["noise_band_percentile"] == 96.0
    assert percentile_only_row["noise_band_percentile"] == 96.0


def test_build_leaderboard_gate_false_when_z_gate_unavailable():
    """When the batch's noise band can't support a z estimate (e.g. < 2 draws),
    family_wise_z_gate is None and every row's gate must be False, not computed against a
    bar that doesn't exist."""
    band = {"available": True, "candidate_percentile_better_than_noise": 100.0, "candidate_z_vs_band": -5.0}
    entries = [
        {"candidate": "cand", "state": "dossiered", "facts": _facts("cand", noise_band=band)},
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=None)

    assert rows[0]["clears_family_wise_threshold"] is False


def test_build_leaderboard_skips_non_dossiered_entries():
    entries = [
        {"candidate": "ran", "state": "dossiered", "facts": _facts("ran")},
        {"candidate": "not_yet", "state": "never_run", "facts": None},
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=2.0)

    assert [r["candidate"] for r in rows] == ["ran"]


def test_build_leaderboard_handles_disqualified_candidate_with_null_headline():
    """dossier_tables writes headline: None (not a missing key) for any terminal status
    other than 'graded'. This must not crash, and the candidate still appears (it IS
    dossiered) with null metrics rather than being silently dropped."""
    entries = [
        {"candidate": "dq", "state": "dossiered", "facts": _disqualified_facts("dq")},
        {"candidate": "ok", "state": "dossiered", "facts": _facts("ok", delta=-0.02)},
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=2.0)

    assert {r["candidate"] for r in rows} == {"dq", "ok"}
    dq_row = next(r for r in rows if r["candidate"] == "dq")
    assert dq_row["status"] == "disqualified"
    assert dq_row["delta_cpl_held"] is None
    assert dq_row["effect_resolved"] is None
    assert dq_row["zone_resolved"] is None
    # No resolved metric sorts last regardless of ranking mode.
    assert rows[-1]["candidate"] == "dq"


# ── build_outcome_tally ────────────────────────────────────────────────────

def test_build_outcome_tally_counts_every_state():
    entries = [
        {"candidate": "a", "state": "dossiered", "facts": _facts("a", status="graded")},
        {"candidate": "b", "state": "dossiered", "facts": _facts("b", status="disqualified")},
        {"candidate": "c", "state": "never_run", "facts": None},
        {"candidate": "d", "state": "retryable_incomplete", "facts": None},
        {"candidate": "e", "state": "pending_dossier", "facts": None},
    ]

    tally = build_outcome_tally(entries)

    assert tally == {
        "total_candidates_filed": 5,
        "dossiered_by_status": {"graded": 1, "disqualified": 1},
        "never_run": 1,
        "retryable_incomplete": 1,
        "pending_dossier": 1,
    }


def test_build_outcome_tally_separates_declared_leaks_from_ordinary_aborts():
    """fps-3jj.13: a candidate that declared a label column as INPUTS/COLUMNS (abort_reason
    'leak_by_declaration') must tally under its own key, not merge into plain
    'aborted_candidate' with an ordinary bug in the candidate's arithmetic."""
    entries = [
        {"candidate": "leaky", "state": "dossiered",
         "facts": _facts("leaky", status="aborted_candidate", abort_reason="leak_by_declaration")},
        {"candidate": "buggy", "state": "dossiered",
         "facts": _facts("buggy", status="aborted_candidate")},
    ]

    tally = build_outcome_tally(entries)

    assert tally["dossiered_by_status"] == {
        "aborted_candidate:leak_by_declaration": 1,
        "aborted_candidate": 1,
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


def test_confidence_calibration_gates_on_usable_pairs_not_just_resolved_effect(tmp_path, monkeypatch):
    """A resolved-effect candidate with no recorded confidence_effect (predates the
    two-CONFIDENCE-field convention) contributes nothing to the means — gating
    insufficient_data on resolved-effect count alone could report `false` while every
    mean still comes out None, exactly what this flag exists to prevent."""
    import experiments.pipeline.retrospective as retro_module

    monkeypatch.setattr(retro_module, "MIN_CALIBRATION_N", 2)
    candidates_root = tmp_path / "candidates"
    _write_dossier(
        candidates_root, "batch1", "resolved_no_confidence",
        _facts("resolved_no_confidence", confidence_effect=None, effect_resolved=True),
    )
    _write_dossier(
        candidates_root, "batch1", "resolved_with_confidence",
        _facts("resolved_with_confidence", confidence_effect=0.6, effect_resolved=True),
    )

    result = retro_module.build_confidence_calibration(candidates_root)

    assert result["n_dossiered_with_resolved_effect"] == 2
    assert result["n_usable_for_calibration"] == 1
    assert result["insufficient_data"] is True
    assert result["mean_confidence_effect_when_resolved_true"] is None


def test_confidence_calibration_scans_every_batch_not_just_one(tmp_path):
    candidates_root = tmp_path / "candidates"
    _write_dossier(candidates_root, "batch1", "a", _facts("a", batch="batch1"))
    _write_dossier(candidates_root, "batch2", "b", _facts("b", batch="batch2"))

    result = build_confidence_calibration(candidates_root)

    assert {p["batch"] for p in result["pairs"]} == {"batch1", "batch2"}


def test_confidence_calibration_handles_disqualified_candidate_with_null_headline(tmp_path):
    """Same headline: None shape as the leaderboard test — this scans EVERY batch
    (module docstring point 4), so one disqualified dossier anywhere must not poison
    every other batch's calibration read."""
    candidates_root = tmp_path / "candidates"
    _write_dossier(candidates_root, "batch1", "dq", _disqualified_facts("dq", batch="batch1"))
    _write_dossier(candidates_root, "batch2", "ok", _facts("ok", batch="batch2"))

    result = build_confidence_calibration(candidates_root)

    assert len(result["pairs"]) == 2
    dq_pair = next(p for p in result["pairs"] if p["candidate"] == "dq")
    assert dq_pair["effect_resolved"] is None
    assert dq_pair["zone_resolved"] is None
    # A None effect_resolved doesn't count toward n_dossiered_with_resolved_effect.
    assert result["n_dossiered_with_resolved_effect"] == 1


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


def test_compute_retrospective_returned_payload_matches_written_file_with_nan_std(tmp_path):
    """_noise_band writes band_std_delta_cpl_held as float('nan') when a batch's noise floor
    has only one draw (ddof=1 needs 2+ samples). to_jsonable maps that to None for the file
    on disk; the RETURNED payload must go through the same conversion, or a caller reading
    the return value would see a NaN the persisted record stores as null."""
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1"))
    batch_dir = batches_dir / "batch1"
    batch_dir.mkdir(parents=True)
    (batch_dir / "noise_floor.json").write_text(
        json.dumps({
            "deltas_cpl_held": [0.01], "partial": False,
            "null_method": "placebo_column", "tank_params": TANK_1D,
        })
    )

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    on_disk = json.loads((batch_dir / RETROSPECTIVE_FILENAME).read_text())
    assert payload == on_disk
    assert payload["noise_floor"]["band_std_delta_cpl_held"] is None
    # A single-draw floor can't support a t-critical value (needs >= 1 degree of freedom) —
    # the z-gate must be None, not computed against an undefined std.
    assert payload["family_wise_z_threshold"] is None
    assert payload["leaderboard"][0]["clears_family_wise_threshold"] is False


def test_compute_retrospective_z_gate_end_to_end(tmp_path):
    """fps-awz end-to-end: a real (fake) multi-draw noise_floor.json feeds through
    _batch_noise_summary -> family_wise_z_threshold -> build_leaderboard, and the payload's
    top-level family_wise_z_threshold matches what the function itself returns."""
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    # A candidate whose delta sits far below the band (very negative z) so it clears
    # whatever a single-candidate Bonferroni-corrected gate resolves to.
    band = {"available": True, "candidate_percentile_better_than_noise": 100.0, "candidate_z_vs_band": -3.0}
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1", noise_band=band))
    batch_dir = batches_dir / "batch1"
    batch_dir.mkdir(parents=True)
    (batch_dir / "noise_floor.json").write_text(
        json.dumps({
            "deltas_cpl_held": [0.0, 0.01, -0.01, 0.02, -0.02], "partial": False,
            "null_method": "placebo_column", "tank_params": TANK_1D,
        })
    )

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    assert payload["family_wise_z_threshold"] == pytest.approx(family_wise_z_threshold(1, 5))
    assert payload["leaderboard"][0]["clears_family_wise_threshold"] is True


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


def test_leaderboard_rows_carry_mechanism_family():
    """A batch retrospective must be able to see family CONCENTRATION without reopening
    five candidate modules — that is the whole point of the disclosure being required
    (generator.md § Diversity: five uncorrelated candidates all telling one story reads
    as diversity and isn't). Nothing carried it past results.json until fps-3jj.11.
    """
    entries = [
        {"candidate": "a", "state": "dossiered",
         "facts": _facts("a", delta=-0.05, mechanism_family="wholesale-lead")},
        {"candidate": "b", "state": "dossiered",
         "facts": _facts("b", delta=0.02, mechanism_family="wholesale-lead")},
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=None)

    assert {r["mechanism_family"] for r in rows} == {"wholesale-lead"}


# ── cadence stamp propagation (fps-aam) ─────────────────────────────────────

def test_leaderboard_rows_carry_tank_params_from_the_dossiers_provenance():
    """fps-15c stamped every CPL-writing production artifact; retrospective_facts.json is
    the derived view built from them and was left out of that bead's six named sites.
    `delta_cpl_held` moves materially with cadence (fps-fii: 189.67/187.85/187.82 c/L at
    7/2/1-day on an otherwise identical run), so a leaderboard row without the stamp is a
    number a reader cannot place — and the leaderboard is where max-of-N promotion happens.
    """
    entries = [
        {"candidate": "a", "state": "dossiered", "facts": _facts("a", delta=-0.05)},
        {"candidate": "b", "state": "dossiered",
         "facts": _facts("b", delta=0.02, tank_params="50/3.571/7d/10%")},
    ]

    rows = build_leaderboard(entries, family_wise_z_gate=None)

    assert {r["candidate"]: r["tank_params"] for r in rows} == {"a": TANK_1D, "b": "50/3.571/7d/10%"}


def test_build_leaderboard_refuses_a_delta_cpl_with_no_tank_params():
    """The pairing, not just the key: a row carrying a real delta_cpl_held beside a null
    stamp would still satisfy artifact_has_unstamped_cpl (a deliberately coarse whole-
    document check — one stamped sibling row masks it), which is exactly the gap that would
    make this bead's backstop nominal rather than real."""
    entries = [
        {"candidate": "stamped", "state": "dossiered", "facts": _facts("stamped", delta=-0.05)},
        {"candidate": "unstamped", "state": "dossiered",
         "facts": _facts("unstamped", delta=-0.09, tank_params=None)},
    ]

    with pytest.raises(ValueError, match="unstamped.*delta_cpl_held"):
        build_leaderboard(entries, family_wise_z_gate=None)


def test_leaderboard_row_for_a_non_graded_candidate_has_no_stamp_and_no_cpl():
    """A disqualified/aborted candidate never reaches the scoring stages, so build_facts
    doesn't require a stamp for it either (its own fps-15c guard is gated on
    status=='graded'). No CPL, nothing to label — the row must not be refused."""
    entries = [{"candidate": "dq", "state": "dossiered", "facts": _disqualified_facts("dq")}]

    rows = build_leaderboard(entries, family_wise_z_gate=None)

    assert rows[0]["delta_cpl_held"] is None
    assert rows[0]["tank_params"] is None


def test_noise_floor_summary_carries_the_floors_own_tank_params(tmp_path):
    """The band's mean/std are CPL deltas in whatever cadence the floor was computed at —
    read verbatim off noise_floor.json, not re-derived from a TankParams here."""
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1"))
    batch_dir = batches_dir / "batch1"
    batch_dir.mkdir(parents=True)
    (batch_dir / "noise_floor.json").write_text(
        json.dumps({
            "deltas_cpl_held": [0.0, 0.01, -0.01], "partial": False,
            "null_method": "placebo_column", "tank_params": TANK_1D,
        })
    )

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    assert payload["noise_floor"]["tank_params"] == TANK_1D
    assert payload["noise_floor"]["band_mean_delta_cpl_held"] == pytest.approx(0.0)


def test_noise_floor_summary_refuses_a_floor_with_no_tank_params(tmp_path):
    """`_batch_noise_summary` calls _noise_band with check_fingerprint=False, which skips
    COMPARING the floor's stamp against a run's — it must not be read as excusing the stamp
    being absent altogether."""
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1"))
    batch_dir = batches_dir / "batch1"
    batch_dir.mkdir(parents=True)
    (batch_dir / "noise_floor.json").write_text(
        json.dumps({
            "deltas_cpl_held": [0.0, 0.01, -0.01], "partial": False,
            "null_method": "placebo_column",
        })
    )

    with pytest.raises(ValueError, match="no tank_params"):
        compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)


def test_unavailable_noise_floor_summary_is_returned_verbatim_without_a_stamp(tmp_path):
    """A refusal dict carries no CPL, so there is nothing to stamp — and its `reason` (the
    actionable half for a reader) must survive untouched."""
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1"))
    (batches_dir / "batch1").mkdir(parents=True)  # no noise_floor.json at all

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)

    assert payload["noise_floor"]["available"] is False
    assert "tank_params" not in payload["noise_floor"]
    assert "noise-floor calibration" in payload["noise_floor"]["reason"]


def test_retrospective_facts_passes_the_unstamped_cpl_backstop(tmp_path):
    """fps-aam's acceptance criterion, end to end: the generic backstop
    experiments.lib.io.artifact_has_unstamped_cpl must not flag the written artifact. The
    stripped-copy control keeps the assertion from being vacuous — it proves the payload
    really does carry CPL-shaped keys the checker would otherwise fire on."""
    candidates_root = tmp_path / "candidates"
    batches_dir = tmp_path / "batches"
    _write_dossier(candidates_root, "batch1", "cand", _facts("cand", batch="batch1"))
    batch_dir = batches_dir / "batch1"
    batch_dir.mkdir(parents=True)
    (batch_dir / "noise_floor.json").write_text(
        json.dumps({
            "deltas_cpl_held": [0.0, 0.01, -0.01], "partial": False,
            "null_method": "placebo_column", "tank_params": TANK_1D,
        })
    )

    payload = compute_retrospective("batch1", batches_dir=batches_dir, candidates_root=candidates_root)
    on_disk = json.loads((batch_dir / RETROSPECTIVE_FILENAME).read_text())

    assert artifact_has_unstamped_cpl(payload) is False
    assert artifact_has_unstamped_cpl(on_disk) is False
    assert artifact_has_unstamped_cpl(_strip_keys(on_disk, "tank_params")) is True


def _strip_keys(obj: object, key: str) -> object:
    """Deep copy with every occurrence of `key` removed — the negative control for the
    backstop assertion above."""
    if isinstance(obj, dict):
        return {k: _strip_keys(v, key) for k, v in obj.items() if k != key}
    if isinstance(obj, list):
        return [_strip_keys(v, key) for v in obj]
    return obj
