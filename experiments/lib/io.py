from __future__ import annotations

import json
import numbers
import pathlib

import numpy as np


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
