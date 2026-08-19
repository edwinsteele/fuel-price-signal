from __future__ import annotations

import json
import numbers
import pathlib
import subprocess
from collections.abc import Sequence

import numpy as np

from experiments.lib.constants import BASELINE_COLUMNS
from fuel_signal.features import baseline_fingerprint


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


def write_meta(
    out_dir: pathlib.Path,
    meta: dict,
    *,
    baseline_columns: Sequence[str] | None = None,
) -> None:
    """Serialise `meta` to out_dir/meta.json, stamping the baseline's identity into it.

    Every experiment result carries a `baseline` block — `n_columns`, an ordered
    `fingerprint`, and the column list itself — so that two runs' commensurability is
    a mechanical check (fps-zci item 5). Both contract defects found so far (fps-sa1's
    64-column R0, fps-zci's sorted permutation) were completely invisible in the
    artifacts that recorded the runs; a fingerprint mismatch would have shown either
    of them the first time two runs were put side by side.

    `baseline_columns` defaults to the declared lock (constants.BASELINE_COLUMNS),
    which is what R0 is in a standard paired-WFCV script. Pass it explicitly — in the
    order the model was fit in — when the script's R0 is anything else. A caller that
    passes its own list is recorded with `declared_by_caller: true`, so a stamped
    default is never mistaken for a verified one.
    """
    stamped = dict(meta)
    columns = list(BASELINE_COLUMNS if baseline_columns is None else baseline_columns)
    stamped["baseline"] = {
        "n_columns": len(columns),
        "fingerprint": baseline_fingerprint(columns),
        "declared_by_caller": baseline_columns is not None,
        "columns": columns,
    }
    (out_dir / "meta.json").write_text(json.dumps(to_jsonable(stamped), indent=2, default=str))
    print(f"\nMeta: {out_dir / 'meta.json'}", flush=True)
