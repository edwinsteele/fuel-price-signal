from __future__ import annotations

import json
import numbers
import pathlib
import subprocess
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fuel_signal.backtest import TankParams


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


_CADENCE_STAMP_KEYS = frozenset({"tank_params", "tank"})


def _all_keys(o: object):
    """Yield every dict key anywhere in a JSON-shaped dict/list/scalar tree."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(o, (list, tuple)):
        for item in o:
            yield from _all_keys(item)


def artifact_has_unstamped_cpl(artifact: object) -> bool:
    """True if a JSON-shaped artifact (dict/list) carries a CPL-shaped key
    anywhere but no cadence stamp (`tank_params` or `tank`) anywhere in it.

    The generic backstop fps-15c's per-site guards can't be: those guards live
    in the handful of writers this repo knows about today (log_experiment,
    run_paired_realised_backtest, runner.py's results.json, dossier_tables'
    facts.json, batch_freeze's freeze.json, noise_floor.json) — the whole
    point of the bead is that the NEXT mechanism (a hand-written experiment
    script, a new pipeline stage) won't have been written yet when this landed.
    This function doesn't care how an artifact was produced — it just checks
    the shape of the finished JSON, so it catches a future writer that built
    its own dict from scratch and skipped every shared helper.

    Deliberately coarse rather than requiring the cadence key in the SAME dict
    as the CPL key: this codebase's real artifacts nest CPL values in per-fold
    /per-arm rows (results.json's `aggregate`, `fold_run_deltas`) while the
    stamp lives once, at the top level or under `meta`/`provenance` — a same-
    dict requirement would false-positive on every one of them. A whole-
    document "has a stamp somewhere" check is weaker but matches how these
    artifacts are actually shaped.

    "CPL-shaped" is a case-insensitive substring match on "cpl" in the key
    name (cpl_own, cpl_held, always_cpl, realised_spend_cpl, delta_cpl_held,
    band_mean_delta_cpl_held, ...) — every CPL field in this codebase follows
    that naming convention.
    """
    keys = list(_all_keys(artifact))
    has_cpl = any("cpl" in k.lower() for k in keys)
    has_cadence = any(k in _CADENCE_STAMP_KEYS for k in keys)
    return has_cpl and not has_cadence


def write_meta(
    out_dir: pathlib.Path,
    meta: dict,
    *,
    baseline_columns: Sequence[str] | None = None,
    tank: "TankParams | None" = None,
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

    `tank` (fps-15c): pass the TankParams a script's own realised-CPL numbers (kept
    in the script's own results, not in `meta` itself — this function has no notion
    of what "a CPL" looks like) actually ran at, and it is stamped here as
    `tank_params` — the same shared path experiments/lib/realised.py and
    log_experiment use, so a hand-written script inherits the cadence-stamping
    discipline instead of reimplementing (or forgetting) it. None (default) omits
    the field entirely — most callers of write_meta don't carry a realised CPL at
    all (a WFCV log-loss-only script has no cadence to stamp).
    """
    # Imported here, not at module scope: fuel_signal.features transitively pulls in
    # scipy.signal (via lga_leadership), which took this leaf serialisation module
    # from ~27ms to ~700ms to import. Nothing else in io.py needs the feature layer.
    from experiments.lib.constants import BASELINE_COLUMNS
    from fuel_signal.features import baseline_fingerprint

    stamped = dict(meta)
    columns = list(BASELINE_COLUMNS if baseline_columns is None else baseline_columns)
    stamped["baseline"] = {
        "n_columns": len(columns),
        "fingerprint": baseline_fingerprint(columns),
        "declared_by_caller": baseline_columns is not None,
        "columns": columns,
    }
    if tank is not None:
        from fuel_signal.backtest import require_tank_stamp

        stamped["tank_params"] = require_tank_stamp(tank, what="write_meta")
    (out_dir / "meta.json").write_text(json.dumps(to_jsonable(stamped), indent=2, default=str))
    print(f"\nMeta: {out_dir / 'meta.json'}", flush=True)
