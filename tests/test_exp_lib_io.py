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
    write_meta(tmp_path, {}, tank=TankParams(evaluation_interval_days=1))
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["tank_params"] == "50/3.571/1d/10%"


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
    that produced it. Doubles as the generic "catches mechanism number five"
    backstop the bead calls for — a future artifact with a CPL-shaped key and
    no stamp anywhere in it fails this same check."""
    obj = json.loads(pathlib.Path(path).read_text())
    assert artifact_has_unstamped_cpl(obj) is False, f"{path} has a CPL-shaped key with no cadence stamp"
