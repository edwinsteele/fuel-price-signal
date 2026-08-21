"""Tests for experiments/lib/io.py — current_git_sha (shared by runner.py and
dossier_tables.py, fps-3jj.6, to avoid two copies of the same subprocess wrapper),
write_meta's baseline stamping (fps-zci), and the cadence-stamp invariant (fps-15c)."""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from experiments.lib.constants import BASELINE_COLUMNS, BASELINE_FINGERPRINT
from experiments.lib.io import artifact_has_unstamped_cpl, current_git_sha, write_meta
from fuel_signal.backtest import TankParams


def test_current_git_sha_returns_a_real_sha_in_this_checkout():
    sha = current_git_sha()
    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_current_git_sha_returns_none_when_git_missing(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert current_git_sha() is None


def test_current_git_sha_returns_none_when_git_fails(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert current_git_sha() is None


# ── write_meta baseline stamping (fps-zci item 5) ──────────────────────────────

def test_write_meta_stamps_the_declared_baseline_by_default(tmp_path):
    """Every experiment meta.json carries the identity of the R0 it measured against.

    Neither fps-sa1 (a 64-column R0) nor fps-zci (a sorted permutation of the right
    54) left any trace in the artifacts that recorded the runs, which is why they were
    compared head-to-head for two months.
    """
    write_meta(tmp_path, {"seeds": [42]})
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["seeds"] == [42]
    assert meta["baseline"]["n_columns"] == len(BASELINE_COLUMNS)
    assert meta["baseline"]["fingerprint"] == BASELINE_FINGERPRINT
    assert meta["baseline"]["columns"] == BASELINE_COLUMNS
    assert meta["baseline"]["declared_by_caller"] is False


def test_write_meta_records_a_caller_supplied_baseline_as_declared(tmp_path):
    cols = list(BASELINE_COLUMNS[:3])
    write_meta(tmp_path, {}, baseline_columns=cols)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["baseline"]["columns"] == cols
    assert meta["baseline"]["n_columns"] == 3
    assert meta["baseline"]["declared_by_caller"] is True


def test_write_meta_fingerprints_order_not_just_membership(tmp_path):
    """A permutation is a different baseline, so it must not fingerprint alike."""
    write_meta(tmp_path, {}, baseline_columns=sorted(BASELINE_COLUMNS))
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["baseline"]["n_columns"] == len(BASELINE_COLUMNS)
    assert meta["baseline"]["fingerprint"] != BASELINE_FINGERPRINT


def test_write_meta_does_not_mutate_the_caller_dict(tmp_path):
    meta_in: dict = {"seeds": [42]}
    write_meta(tmp_path, meta_in)
    assert meta_in == {"seeds": [42]}


# ── write_meta tank_params stamping (fps-15c) ──────────────────────────────────

def test_write_meta_omits_tank_params_when_no_tank_passed(tmp_path):
    """Most write_meta callers (a WFCV log-loss-only script) carry no realised
    CPL at all — the field must not appear as though a cadence were declared."""
    write_meta(tmp_path, {"seeds": [42]})
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert "tank_params" not in meta


def test_write_meta_stamps_the_tank_it_ran_with(tmp_path):
    write_meta(tmp_path, {}, tank=TankParams(evaluation_interval_days=7))
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["tank_params"] == "50/3.571/7d/10%"


# ── artifact_has_unstamped_cpl (fps-15c) — the generic structural backstop ────

def test_artifact_has_unstamped_cpl_flags_a_bare_cpl_value():
    assert artifact_has_unstamped_cpl({"cpl_held": 190.0}) is True


def test_artifact_has_unstamped_cpl_passes_when_stamp_is_a_sibling_top_level_key():
    assert artifact_has_unstamped_cpl({"tank_params": "50/3.571/7d/10%", "cpl_held": 190.0}) is False


def test_artifact_has_unstamped_cpl_finds_the_stamp_anywhere_in_the_tree():
    """Real artifacts nest CPL values in per-fold/per-arm rows while the stamp
    lives once, under meta/provenance — a same-dict requirement would
    false-positive on every one of them."""
    artifact = {
        "aggregate": [{"arm": "R0", "cpl_own": 189.67, "cpl_held": 189.67}],
        "meta": {"tank_params": "50/3.571/7d/10%"},
    }
    assert artifact_has_unstamped_cpl(artifact) is False


def test_artifact_has_unstamped_cpl_ignores_artifacts_with_no_cpl_at_all():
    assert artifact_has_unstamped_cpl({"n_windows": 4, "seeds": [1, 2]}) is False


def test_artifact_has_unstamped_cpl_recurses_into_lists():
    artifact = [{"delta_cpl_held": 0.5}]
    assert artifact_has_unstamped_cpl(artifact) is True


@pytest.mark.parametrize("path", [
    "experiments/batches/batch0/freeze.json",
    "experiments/batches/batch0/noise_floor.json",
    "experiments/candidates/batch0/tgp_delta_7d/results.json",
    "experiments/candidates/batch0/tgp_delta_7d/facts.json",
])
def test_batch0_artifacts_carry_their_cadence_stamp(path):
    """Regression coverage for the fps-15c backfill: every one of today's six
    CPL-writing sites must carry tank_params on disk, not just in the code
    that produced it. (The generic "catches the next mechanism" backstop is
    test_every_committed_experiment_json_artifact_carries_its_cadence_stamp
    below, which scans every committed artifact rather than this hand-picked
    four — freeze.json has no CPL-shaped key at all, so this one line alone
    is vacuous for it; see the value assertion in test_batch_freeze.py's
    test_freeze_manifest_records_the_default_tank_params for that.)"""
    obj = json.loads(pathlib.Path(path).read_text())
    assert artifact_has_unstamped_cpl(obj) is False, f"{path} has a CPL-shaped key with no cadence stamp"


def test_batch0_freeze_json_stamps_the_exact_historical_value():
    """freeze.json carries no CPL-shaped key itself (it's a manifest, not a
    result), so artifact_has_unstamped_cpl never exercises its stamp — assert
    the backfilled value directly instead of relying on a vacuous scan."""
    obj = json.loads(pathlib.Path("experiments/batches/batch0/freeze.json").read_text())
    assert "cpl" not in json.dumps(obj).lower()  # confirms the scan above is indeed vacuous here
    assert obj["tank_params"] == "50/3.571/7d/10%"


# A CPL-shaped key with no cadence stamp anywhere in the SAME committed JSON file
# — real, deliberate exceptions to the fps-15c contract, not oversights:
_UNSTAMPED_CPL_ALLOWLIST = frozenset({
    # A cadence-SWEEP experiment (experiments/2026-08-20_cadence_ceiling/): cadence
    # is the deliberately-varied independent variable, recorded per row as
    # "cadence_days" rather than "tank_params"/"tank" — self-documenting by
    # construction, just under a different key name than this scan looks for.
    "experiments/2026-08-20_cadence_ceiling/stage2_model/meta.json",
    # Derived/aggregate artifact built entirely from each candidate's facts.json
    # (which DOES carry the stamp, per the parametrized test above) plus the
    # batch's noise_floor.json (ditto) — propagating the stamp into the
    # leaderboard/summary itself is fps-aam, filed and deliberately out of
    # fps-15c's scope.
    "experiments/batches/batch0/retrospective_facts.json",
})


def test_every_committed_experiment_json_artifact_carries_its_cadence_stamp():
    """The generic "catches the next mechanism" backstop fps-15c calls for only
    works if it actually scans every artifact, not a hand-picked handful — a
    future writer that's never added to a parametrize list is caught by
    nothing. Scans every committed experiments/**/*.json (git-tracked, so a
    developer's local uncommitted/in-progress batch artifacts can't fail this
    on someone else's machine) against artifact_has_unstamped_cpl, with an
    explicit, commented allowlist for the two known, deliberate exceptions."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "experiments"], cwd=repo_root, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    json_paths = [p for p in tracked if p.endswith(".json")]
    assert len(json_paths) >= 4, "sanity check: git ls-files must actually find the batch0 artifacts"

    failures = []
    for rel_path in json_paths:
        if rel_path in _UNSTAMPED_CPL_ALLOWLIST:
            continue
        obj = json.loads((repo_root / rel_path).read_text())
        if artifact_has_unstamped_cpl(obj):
            failures.append(rel_path)
    assert failures == [], f"CPL-shaped key(s) with no cadence stamp anywhere in the file: {failures}"
