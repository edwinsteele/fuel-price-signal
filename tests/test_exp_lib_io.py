"""Tests for experiments/lib/io.py — current_git_sha (shared by runner.py and
dossier_tables.py, fps-3jj.6, to avoid two copies of the same subprocess wrapper) and
write_meta's baseline stamping (fps-zci)."""
from __future__ import annotations

import json
import subprocess

from experiments.lib.constants import BASELINE_COLUMNS, BASELINE_FINGERPRINT
from experiments.lib.io import current_git_sha, write_meta


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
