from __future__ import annotations

import json
import numbers
import pathlib
import subprocess

import numpy as np


def current_git_sha() -> str | None:
    """The current checkout's HEAD SHA, or None if git isn't available/this isn't a repo.

    Shared by runner.py (run-time provenance) and dossier_tables.py (dossier-build-time
    fallback when a run predates that field) — one implementation, not two copies drifting.
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def to_jsonable(o: object) -> object:
    """Recursively convert non-finite real numbers to None for JSON serialisation."""
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, numbers.Real) and not np.isfinite(o):
        return None
    return o


def write_meta(out_dir: pathlib.Path, meta: dict) -> None:
    (out_dir / "meta.json").write_text(json.dumps(to_jsonable(meta), indent=2, default=str))
    print(f"\nMeta: {out_dir / 'meta.json'}", flush=True)
