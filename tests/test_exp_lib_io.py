"""Tests for experiments/lib/io.py — current_git_sha (shared by runner.py and
dossier_tables.py, fps-3jj.6, to avoid two copies of the same subprocess wrapper)."""
from __future__ import annotations

import subprocess

from experiments.lib.io import current_git_sha


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
