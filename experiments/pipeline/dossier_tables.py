"""Dossier tables (fps-3jj.6) — deterministic facts.json + PNGs for one completed run.

Code / Claude boundary (DECIDED 2026-08-17, fps-3jj.6): this module does no prose and no
judgement. It reads a completed candidate run's artifacts (`results.json`, `rowpreds.parquet`,
`fills.parquet` — all written by `experiments/pipeline/runner.py`, fps-3jj.4) plus the frozen
batch's own metadata, and writes one `facts.json` plus a handful of PNGs into the run directory.
Every number in `facts.json` traces back to one of those inputs; nothing here reads
`PREDICTED_SIGNATURE` and forms a verdict about it — that's the Claude step's job
(`docs/routines/dossier.md`), reading `facts.json` and writing `README.md` prose only.

Why split this way (see fps-3jj parent design): a number computed in-context by a Claude session
cannot be checked short of a re-run; if the Claude step dies mid-session the facts already exist
on disk, so the work queue (any dir with a results.json newer than its README.md, see
`find_pending_runs`) stays correct and re-runnable; and it keeps the overnight Claude burst short.

Stale-claim recovery is deliberately NOT implemented here. fps-3jj.5 (launch routine) merged
after this module's first version was written with its own claim.json-based recovery — that
turned out to be dead code (nothing ever wrote claim.json) duplicating a weaker version of what
launch.py's `find_stale_claims`/`release_stale_claim` (bd `in_progress` + a required traceback
tail in run.log, not just "no results.json after 12h") already does correctly. Removed rather
than fixed in place — one true stale-claim path, not two.

One artifact this module reads that a given batch may not have yet, a forward-compatible slot:

  noise_floor.json (`NOISE_FLOOR_FILENAME`) — written by `experiments/pipeline/noise_floor.py`
    (fps-3jj.9): `{"deltas_cpl_held": [<float>, ...], "partial": <bool>, ...}`, one R0-vs-R0
    realised-CPL delta per seed pair, into the batch dir. That module is a deliberate
    batch-setup step (heavy — ~10 single-arm fits), not run automatically, so a batch that
    hasn't had it run yet still reads facts["noise_band"]["available"] == False — that's a
    missing per-batch step, not a missing capability. `"partial": true` means it was computed
    with `--fold-subset` (an iteration/smoke speed-up, not a real calibration); `_noise_band()`
    below refuses those rather than grading a full-fold delta against a partial-fold ruler.

FIXED (fps-icv, was a KNOWN GAP): the run-directory convention this module assumes (one
subdirectory per candidate) now holds. launch.py's `launch_detached` and runner.py's
`run_candidate` both default `out_dir` to `runner.default_out_dir(candidate_path)` —
candidate_path with its .py suffix stripped — not the candidate module's parent directory
(which is the whole BATCH directory, shared by every candidate module filed against it, since
`experiments/candidates/<batch>/<name>.py` are flat siblings — see generator.md's Filing
section). Before this fix, candidate 2+ in any batch would overwrite candidate 1's
results.json/rowpreds.parquet/fills.parquet/run.log in place, and once this module's
`find_pending_runs` had written a README.md into that shared directory for candidate 1, the
directory would be permanently excluded from the queue — candidate 2+ silently never
dossiered. `find_pending_runs`'s `root.rglob(RESULTS_FILENAME)` already recurses, so no change
was needed in this module itself.

Usage:
  PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables <run_dir>       # one run
  PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables --scan <root>   # whole queue
"""
from __future__ import annotations

import bisect
import json
import math
import pathlib

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — backend must be set before this import
import numpy as np
import pandas as pd
from scipy.stats import norm as _norm_dist
from scipy.stats import t as _t_dist

from experiments.lib.constants import ROW_AXIS_ECONOMICS_CAVEAT
from experiments.lib.flips import (
    cascade_window_days,
    diff_fills,
    parse_tank_params,
    regret_horizon_days,
    summarise_flips,
    summarise_regret,
)
from experiments.lib.io import current_git_sha, to_jsonable
from experiments.lib.zones import assign_regime, pooled_cpl

# FROZEN_DB_FILENAME is batch_freeze's artifact to name (it writes the clone); runner.py and
# noise_floor.py already import it from there, and the dossier_tables -> runner -> batch_freeze
# import direction already exists, so this adds no new edge.
from experiments.pipeline.batch_freeze import FROZEN_DB_FILENAME
from experiments.pipeline.placebo import TEXTURE_ICC_BOUND, effective_n_draws
from experiments.pipeline.redundancy import BATCH_RECORD_JSON
from experiments.pipeline.runner import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    DEFAULT_MIN_ROW_CELL_N,
    RETRYABLE_STATUSES,
    STATUS_GRADED,
    read_run_status,
)
from fuel_signal import db as _db
from fuel_signal.dates import date_from_int
from fuel_signal.score_phase2 import threshold_sweep

RESULTS_FILENAME = "results.json"
FACTS_FILENAME = "facts.json"
README_FILENAME = "README.md"
ROWPREDS_FILENAME = "rowpreds.parquet"
FILLS_FILENAME = "fills.parquet"
NOISE_FLOOR_FILENAME = "noise_floor.json"
BASELINE_COLUMNS_FILENAME = "baseline_columns.json"
FREEZE_MANIFEST_FILENAME = "freeze.json"

# noise_floor.json's "null_method" value (fps-awz). Defined here, not in noise_floor.py,
# for the same reason NOISE_FLOOR_FILENAME is: this module is the contract owner for what
# the producer must write and what _noise_band() will accept — noise_floor.py imports this
# name rather than the two modules each defining their own copy of the string.
NULL_METHOD_PLACEBO_COLUMN = "placebo_column"

# The same construction, but with every draw's source column PINNED to a small deliberate set
# so draws repeat columns on purpose (bd `fps-3jj.23`, `placebo.screen_pinned_column_draws`).
# A DIFFERENT string from the one above, deliberately: this bank exists to measure the texture
# ICC, and the property that makes it able to do so — draws sharing a source column, hence
# sharing texture exactly — is precisely what makes its band too NARROW to grade anything
# against. `_noise_band`'s null_method check already refuses anything that is not
# NULL_METHOD_PLACEBO_COLUMN, so naming it separately makes the diagnostic structurally
# ungradeable rather than relying on nobody ever pointing --out-name at noise_floor.json.
NULL_METHOD_PLACEBO_PINNED_COLUMN = "placebo_column_pinned_source"

#: `_noise_band` refusal codes, machine-readable so the dossier routine can branch on the
#: KIND of unavailability instead of pattern-matching the prose `reason` (fps-3jj.17's rule:
#: the ledger outcome is mechanical, never a 05:06 judgement call).
#:
#: Only the arity case carries a code today, and that asymmetry is the point rather than an
#: oversight. `docs/routines/dossier.md` maps `available: false` to `outcome: rejected`, which
#: is defensible for the pre-existing refusals (a batch with no floor at all has no ruler and
#: never will without human action) but is WRONG for this one: the run itself completed and
#: graded fine, and building the right-arity ruler turns it into a real measurement. Writing
#: `rejected` there would stamp dead ground onto a candidate nothing has measured — the exact
#: harm fps-3jj.17 exists to prevent. See dossier.md Step 1.4's arity carve-out.
NOISE_BAND_REFUSAL_ARITY = "floor_arity_below_run"

#: RETIRED (fps-3jj.25). There was briefly a second arity refusal, `run_arity_above_cap`, for a
#: candidate wider than a `placebo.MAX_RULER_ARITY` constant. Both are gone: the cap made the
#: instrument dictate what shape of candidate the generator could propose, and reuse turned out
#: to degrade a bank gradually enough to PRICE (`placebo.effective_n_draws` feeding
#: `family_wise_z_threshold`) rather than forbid. A wide candidate now gets a wider, harder bar
#: instead of no verdict and an unbounded nightly re-pick. Named here so a stale `facts.json`
#: carrying the code is still recognisable, and so the removal is not rediscovered as a gap.
RETIRED_NOISE_BAND_REFUSAL_ARITY_CAP = "run_arity_above_cap"
FROZEN_FEATURES_FILENAME = "features.parquet"

SEED_STD_FLAG_RATIO = 5.0

# Single-sourced here (fps-3jj.17), not in retrospective.py, because _noise_band() below is the
# only place that can attach a per-run threshold to facts.json at dossier time — retrospective.py
# already imports _noise_band FROM this module, so the dependency has to run this direction, not
# the other way, to avoid a cycle. retrospective.py imports this constant and function rather than
# keeping its own copy (the same single-symbol discipline BASELINE_COLUMNS gets — docs/CONVENTIONS.md
# § "The baseline feature set is declared, never discovered").
FAMILY_WISE_ALPHA = 0.05

# `family_wise_z_threshold` in the limit where the band is KNOWN rather than estimated from a
# finite number of draws (t -> normal, and the prediction interval's sqrt(1 + 1/n) -> 1). Used
# only as the fixed reference that `bar_draw_count_shift_cpl_held` measures against, so that
# penalty is a property of one floor rather than of a pair of floors being compared.
_LARGE_SAMPLE_Z = float(_norm_dist.ppf(1.0 - FAMILY_WISE_ALPHA))


def family_wise_z_threshold(n_candidates: int, n_draws: float, alpha: float = FAMILY_WISE_ALPHA) -> float:
    """A Bonferroni-corrected, ONE-TAILED t-critical value, in band-standard-deviation units, that
    `_noise_band`'s `candidate_z_vs_band` must clear (in magnitude) to count as distinguishable
    from the noise band rather than a draw from it.

    Moved from retrospective.py (fps-3jj.17) so `_noise_band` can attach the n_candidates=1 case —
    "the honest single-candidate bar" (docs/CONVENTIONS.md) — to every run's facts.json as
    `single_candidate_z_threshold`, giving the dossier routine a precomputed number to compare
    `candidate_z_vs_band` against instead of a judgement call. retrospective.py's own
    `family_wise_z_gate` (n_candidates = the whole batch) is a separate, larger call to this same
    function, for a different question (did any candidate clear the bar at the BATCH level,
    correcting for picking the best of N) — see its docstring there.

    **`n_draws` is the EFFECTIVE draw count, which may be fractional** (`fps-3jj.25`). A bank
    whose draws reuse source columns carries less information than its nominal count, because two
    draws built on one column share that column's texture exactly; `placebo.effective_n_draws`
    prices that. Passing the nominal count would make the threshold act as though the band were
    better estimated than it is — a bar too EASY, the bias this whole mechanism exists to remove.
    Feeding the effective count instead is what lets a candidate of ANY arity be graded rather
    than refused: a wider candidate reuses more, so its band is worth fewer draws, so its bar is
    automatically wider. `scipy.stats.t.ppf` accepts a non-integer `df`, so no rounding is needed
    and none should be added — rounding down would be a silent extra penalty.

    Uses the t distribution, not the normal, because `band_std_delta_cpl_held` is ESTIMATED from
    `n_draws` placebo draws, not known — `df = n_draws - 1`. The `sqrt(1 + 1/n_draws)` factor is a
    PREDICTION interval, not a one-sample t-test on the band's own mean: a candidate's `delta` is a
    NEW observation being compared against a band whose mean AND std are both estimated from the
    same `n_draws` draws, so the standard error of (new observation − estimated mean) needs the
    `1` (the new observation's own within-population variance) as well as the `1/n_draws`
    (uncertainty in the estimated mean itself).

    Raises ValueError if `n_draws < 2` (a t-critical value needs at least 1 degree of freedom) —
    callers should check `n_draws >= 2` first, same as `_noise_band` already does for
    `candidate_z_vs_band` itself.
    """
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if n_draws < 2:
        raise ValueError(f"n_draws must be >= 2 to estimate a band std, got {n_draws}")
    alpha_corrected = alpha / n_candidates
    t_critical = _t_dist.ppf(1.0 - alpha_corrected, df=n_draws - 1)
    return float(t_critical * math.sqrt(1.0 + 1.0 / n_draws))

# Row-level cycle-phase columns (experiments/lib/zones.py assign_regime() operates on the first).
# "candidate touches cycle phase" (Plots spec, fps-3jj.6) triggers off a candidate's declared
# INPUTS/COLUMNS intersecting this set, not any cycle_* column (cycle_mean_length etc. are
# magnitude, not phase).
CYCLE_PHASE_COLUMNS = {"cycle_pct_through", "cycle_days_since_peak"}


# ── work queue ────────────────────────────────────────────────────────────────

def find_pending_runs(root: pathlib.Path) -> list[pathlib.Path]:
    """Any directory under `root` with a results.json newer than its README.md (or no
    README.md at all), and a non-retryable status.

    Matches the parent design's "work queue is self-describing" rule. A run still in progress
    has neither file and is silently skipped.

    The RETRYABLE_STATUSES exclusion is load-bearing, not tidiness (fps-g31). A run dir is
    keyed on the candidate, so a re-run REUSES it. Without it:

      night 1  candidate aborts (aborted_pipeline); results.json written
               this scan sees results.json + no README -> writes facts.json, session writes README
      night 2  launch releases the claim, candidate re-runs and SUCCEEDS, overwriting results.json
      night 2  this scan sees the README from night 1 -> run is not pending -> never dossiered

    The mtime comparison (fps-0yd) covers the mirror image: a candidate DELIBERATELY re-run
    after a contract fix, whose prior dossiering already left a README behind. Requiring "no
    README.md" alone made that re-run permanently invisible even though its results.json was
    overwritten with a fresh verdict — the same "silently invisible forever" failure the
    RETRYABLE_STATUSES guard above exists to prevent, just triggered by a human re-queuing
    rather than an automatic retry. Comparing mtimes rather than adding a second provenance
    field to facts.json: results.json and README.md are already the two files this function
    reads, and runner.py always writes results.json fresh on every run (never touches an
    existing README), so a newer results.json is exactly "dossiered content is stale".

    Ties (equal mtimes — a real possibility on filesystems with coarse timestamp resolution)
    are treated as PENDING, not dossiered: re-dossiering a run that was actually already
    written up is a harmless no-op (facts.json is deterministic from the same run artifacts),
    while the reverse — silently dropping a genuine re-run because its mtime happened to
    collide with the stale README's — is exactly the failure this function exists to close.

    A results.json or README.md whose `stat()` fails here — vanished mid-race (a concurrent
    launch/rerun deleting it, e.g. via `rm` as part of a re-queue), a permissions error, an NFS
    staleness error — is excluded for this pass rather than raised (fps-0yd review round 2). A
    narrower `except FileNotFoundError` was tried first and rejected: this function builds its
    whole return list eagerly (it is not a generator), and `main()`'s `--scan` loop only wraps
    `process_run` in a per-run try/except, not this call — so ANY exception escaping here
    aborts the entire scan before a single run_dir is even offered to `process_run`, including
    every unrelated, healthy run_dir. That is a wider blast radius than the original "no
    README.md" bug this function exists to fix, and it also contradicts the totality
    `runner.read_run_status` (called two lines below) already commits to for the exact same
    unattended-overnight reason. So: catch broadly, like `read_run_status` does, but — unlike a
    bare `continue` — print a line first (the same `print(..., flush=True)` warning pattern
    `_plot_tau_sweep` below already uses for a different non-fatal anomaly in this module) so a
    run silently dropped from the queue by a real I/O fault is still visible in `--scan`'s
    output rather than swallowed with no trace.
    """
    root = pathlib.Path(root)
    pending = set()
    for p in root.rglob(RESULTS_FILENAME):
        run_dir = p.parent
        readme = run_dir / README_FILENAME
        try:
            if readme.exists() and readme.stat().st_mtime > p.stat().st_mtime:
                continue
        except OSError as exc:
            print(
                f"[dossier_tables] {run_dir}: could not compare results.json/README.md "
                f"timestamps ({exc!r}) — excluding from this scan.",
                flush=True,
            )
            continue
        if read_run_status(run_dir) in RETRYABLE_STATUSES:
            continue
        pending.add(run_dir)
    return sorted(pending)


# ── facts.json ────────────────────────────────────────────────────────────────

def build_facts(run_dir: pathlib.Path) -> dict:
    """Read one run's artifacts and return the facts.json dict (does not write it)."""
    run_dir = pathlib.Path(run_dir)
    results = json.loads((run_dir / RESULTS_FILENAME).read_text())
    status = results["status"]
    meta = results.get("meta", {})
    candidate = results.get("candidate", {})
    batch_dir = pathlib.Path(meta["batch_dir"]) if meta.get("batch_dir") else None

    rowpreds, fills = None, None
    if status == STATUS_GRADED:
        for filename in (ROWPREDS_FILENAME, FILLS_FILENAME):
            if not (run_dir / filename).exists():
                # status=="graded" means runner.py's success path ran, which writes both
                # parquet artifacts BEFORE results.json (see runner.py's "write artifacts before
                # grading" comment) — so this means something removed them after the fact (a
                # manual `git clean`, disk issue, etc.), not a normal in-progress race. Fail with
                # a message that says so, rather than a bare pyarrow FileNotFoundError the --scan
                # loop just logs.
                raise FileNotFoundError(
                    f"{run_dir}: status=='graded' but {filename} is missing — "
                    "expected artifact was removed after the run completed."
                )
        rowpreds = pd.read_parquet(run_dir / ROWPREDS_FILENAME)
        fills = pd.read_parquet(run_dir / FILLS_FILENAME)

    facts: dict = {
        "candidate": {
            "name": candidate.get("name"),
            "hypothesis": candidate.get("hypothesis"),
            "predicted_signature": candidate.get("predicted_signature"),
            "confidence_effect": candidate.get("confidence_effect"),
            "confidence_zone": candidate.get("confidence_zone"),
            "columns": candidate.get("columns"),
            "inputs": candidate.get("inputs"),
            "prior_art": candidate.get("prior_art"),
            "mechanism_family": candidate.get("mechanism_family"),
            "target": candidate.get("target"),
        },
        "provenance": _provenance(results, batch_dir),
        "headline": None,
        "breakdowns": None,
        "decision_flips": None,
        "seed_flags": results.get("seed_variance", {}).get("flags", []),
        "validation": _validation(results, rowpreds),
        "grading": {
            "predicted_signature": candidate.get("predicted_signature"),
            "verdict": None,
            "explanation": None,
            "pending": True,
        },
        "noise_band": _noise_band(results, batch_dir),
        # NOT `batch_dir` (fps-1l1 review of fps-qbv): `batch.json` is a
        # `experiments/candidates/<batch>/` artifact, written by
        # `redundancy.write_batch_record` against `batch_dir_for(module_paths)` — the
        # candidate MODULES' parent directory. `batch_dir` above is `meta["batch_dir"]`,
        # the runner's frozen-artifact dir `experiments/batches/<batch>/` (freeze.json,
        # noise_floor.json) — a different directory that never holds a batch.json. Per
        # this module's own fps-icv fix (see module docstring), `run_dir` is keyed
        # per-candidate directly under the candidates dir, so `run_dir.parent` is
        # exactly what `batch_dir_for` resolves to.
        "redundancy": _redundancy(run_dir.parent, candidate.get("name")),
    }

    if status != STATUS_GRADED:
        facts["status_note"] = (
            f"status={status!r} — run did not reach the scoring stages; headline/breakdowns are "
            f"unavailable. error: {results.get('error')}"
        )
        return facts

    if not facts["provenance"]["tank_params"]:
        # fps-15c: status=="graded" means headline/breakdowns are about to carry
        # realised cpl_own/cpl_held values, and results.json's meta has no
        # tank_params to stamp them with — refuse rather than write a facts.json a
        # reader can't tell apart from a 7d one. Most likely an old results.json
        # that predates this field (backfill it) or a new run-producing mechanism
        # that built its own results.json by hand and skipped the shared stamping
        # path (experiments/lib/realised.py's meta, propagated by runner.py).
        raise ValueError(
            f"{run_dir}: results.json has status=='graded' (a realised CPL is about to "
            "be written into facts.json) but its meta carries no tank_params — refusing "
            "(fps-15c). Backfill the run's results.json with the tank it actually ran at, "
            "or re-run it against current code."
        )

    facts["headline"] = _headline(results)
    facts["breakdowns"] = _breakdowns(results, rowpreds, fills)
    facts["decision_flips"] = _decision_flips(facts, fills)
    return facts


def _decision_flips(facts: dict, fills: pd.DataFrame) -> dict:
    """Per-fold buy/wait divergence between R0 and candidate (fps-gez).

    Computed only when it's likely to matter, not "when a session feels like
    it" (fps-3jj.17's rule: the ledger outcome is mechanical, not a per-run
    judgement call, and this trigger follows the same discipline so dossiers
    stay comparable): 2+ candidate columns (arity 1 has nothing to
    disaggregate) AND the noise-band call isn't a clean reject (z >= t — the
    candidate is clearly the wrong sign vs. this batch's own noise; a flip
    breakdown adds little once the aggregate already says no). Both the
    `inconclusive` case (inside the band) and the z <= -t case (clears noise
    on the good side) get computed, since both are cases a human is likely to
    look closer at.

    Always present with an explicit `computed` flag, never just absent —
    same pattern `noise_band["available"]` already uses — so no downstream
    reader has to guess whether this was skipped or forgotten.

    Guardrail for how this gets written up (docs/routines/dossier.md): flip
    detail is colour illustrating a mechanism, and must never be presented
    as a second arbiter overriding the noise-band call — same rule already
    in place for per_axis.
    """
    columns = facts["candidate"].get("columns") or []
    noise_band = facts["noise_band"]
    if len(columns) < 2:
        return {
            "computed": False,
            "reason": f"candidate has {len(columns)} column(s) — nothing to disaggregate below arity 2.",
        }
    if not noise_band.get("available"):
        return {
            "computed": False,
            "reason": "noise band unavailable for this run — see facts['noise_band']['reason'].",
        }
    z = noise_band.get("candidate_z_vs_band")
    t = noise_band.get("single_candidate_z_threshold")
    if z is None or t is None:
        return {
            "computed": False,
            "reason": "noise band has a null z / single_candidate_z_threshold (band_std not "
            "estimable from this batch's draws) — nothing to gate the trigger on.",
        }
    if z >= t:
        return {
            "computed": False,
            "reason": f"clean reject (z={z:.3f} >= threshold={t:.3f}) — candidate is clearly the "
            "wrong sign vs. this batch's noise band; flip detail is unlikely to change that call.",
        }
    raw_shock_folds = facts["provenance"].get("shock_folds")
    if raw_shock_folds is None:
        return {
            "computed": False,
            "reason": "this run's results.json predates fps-3tu's empirical shock-fold set "
            "(meta.shock_folds) — flip/regret detail is graded per (fold, regime) and there is "
            "no trustworthy regime tag to use; re-run the candidate to backfill it.",
        }
    shock_folds = frozenset(int(f) for f in raw_shock_folds)
    tank_params = facts["provenance"]["tank_params"]
    # None for results.json predating fps-o0h — cascade_window_days/regret_horizon_days both
    # fall back to re-parsing the (display-rounded, tie-prone) stamp in that case.
    exact_fields = facts["provenance"].get("tank_params_fields")
    window = cascade_window_days(tank_params, exact_fields=exact_fields)
    summary = summarise_flips(fills, BASELINE_ARM, CANDIDATE_ARM, shock_folds, window_days=window)
    summary["computed"] = True
    _attach_run_contributions(
        summary, fills, facts["breakdowns"]["per_fold"], facts["headline"]["realised"]["delta_cpl_held"],
    )
    # facts["provenance"]["batch_dir"] rather than a new parameter: _decision_flips is already
    # handed the whole facts dict, and provenance is where this module's own batch_dir lands.
    provenance_batch_dir = facts["provenance"].get("batch_dir")
    summary["regret"] = _attach_regret(
        fills, tank_params, window, shock_folds,
        pathlib.Path(provenance_batch_dir) if provenance_batch_dir else None,
        exact_fields=exact_fields,
    )
    return summary


def _attach_regret(
    fills: pd.DataFrame, tank_params: str, window_days: int, shock_folds: frozenset[int],
    batch_dir: pathlib.Path | None, *, exact_fields: dict | None = None,
) -> dict:
    """The timing-regret table (fps-2js) — `experiments/lib/flips.summarise_regret` over this
    run's flipped fills, or an explicit `computed: false` + reason when the price DB this run
    was graded against isn't reachable from where the dossier is being built.

    Optional-with-a-reason rather than fatal, deliberately: every other input to facts.json is
    a run artifact sitting beside results.json, but regret needs the ~500 MB gitignored price
    DB, so a reader who pulled the repo without it must still be able to rebuild every other
    table. Same `computed`/`reason` contract as `decision_flips` itself and `noise_band`, so no
    downstream reader has to tell "skipped" from "forgotten".

    `exact_fields` (fps-o0h) is threaded straight through to `regret_horizon_days` — see that
    function's docstring for why a caller with the run's `tank_params_fields` should always
    pass it.
    """
    db_path = _resolve_source_db(batch_dir)
    if db_path is None:
        return {
            "computed": False,
            "reason": "no batch freeze manifest — cannot tell which price DB this run was "
            "graded against, and regret must be scored on that exact series.",
        }
    if not db_path.exists():
        return {
            "computed": False,
            "reason": f"the batch's frozen price DB {str(db_path)!r} is not present — it is "
            "gitignored, so a checkout without it can build every other table but not this "
            "one.",
        }
    flips = diff_fills(fills, BASELINE_ARM, CANDIDATE_ARM)
    if flips.empty:
        return {
            "computed": False,
            "reason": "the two arms made no differing fills — no flipped fill to score regret on.",
        }
    _size, _daily, cadence_days, _floor = parse_tank_params(tank_params)
    conn = _db.open_db(db_path)
    try:
        prices = _DbStationPrices(conn, sorted(int(c) for c in flips["station_code"].unique()))
    finally:
        conn.close()
    summary = summarise_regret(
        flips, prices, shock_folds,
        horizon_days=regret_horizon_days(tank_params, exact_fields=exact_fields),
        cadence_days=cadence_days,
        window_days=window_days,
    )
    summary["computed"] = True
    # Resolved, not the raw relative "fuel_signal.db" from freeze.json — facts.json is read
    # long after and from elsewhere, so a CWD-relative path is not provenance. Named
    # "graded_db" (fps-o0h), not "source_db": freeze.json also has a field named "source_db"
    # but meaning a DIFFERENT path (the freeze-time read location, not this frozen clone) —
    # see `_resolve_source_db`'s docstring. Same name, different meaning, was the trap.
    summary["graded_db"] = str(db_path.resolve())
    return summary


def _resolve_source_db(batch_dir: pathlib.Path | None) -> pathlib.Path | None:
    """The batch's own FROZEN price DB — `batch_dir/fuel_signal.db`, the clone
    `batch_freeze._clone_db` made and the exact file `runner.py` graded the run against.

    **Not `freeze.json`'s `source_db`** (PR #348 review finding #1). That field records where
    the freeze *read* from at freeze time — a repo-root-relative `fuel_signal.db` — which
    resolves to the LIVE database, a different and continuously-updated file: `batch0`'s clone
    is 524,197,888 bytes (17 Aug) against the live DB's 524,689,408 (22 Aug). Scoring regret
    against later, possibly revised prices is exactly the drift `summarise_regret` forbids,
    since the whole measure rests on replaying the price path the arms actually faced. Two
    further harms made it concrete rather than theoretical: `_db.open_db` issues
    `PRAGMA journal_mode=WAL`, so every dossier build was a WRITE against the live production
    DB; and no worktree has a root `fuel_signal.db`, so regret reported `computed: false`
    everywhere except the primary checkout while the correct frozen DB sat in the batch dir.

    `runner.py` and `noise_floor.py` both already resolve the DB this way
    (`batch_dir / FROZEN_DB_FILENAME`); this is the third reader, not a new convention.
    `freeze.json`'s presence is still the gate — a batch dir without a manifest isn't a frozen
    batch, and its contents are not the graded artifacts.
    """
    if batch_dir is None:
        return None
    if not (batch_dir / FREEZE_MANIFEST_FILENAME).exists():
        return None
    return batch_dir / FROZEN_DB_FILENAME


class _DbStationPrices:
    """`flips.StationPriceSource` over the run's own sqlite price DB.

    Loads each station's full `daily_prices` series once and answers from memory — regret
    walks ~`horizon_days` dates per flip, so a query per lookup would be thousands of
    round-trips for no benefit. `price_at` reproduces `fuel_signal.backtest.PriceHistory.
    station_price_at` exactly (same `bisect_right(...) - 1` forward-fill over the same
    `db.get_daily_prices` series) — the point of regret is to score each arm against the price
    path it actually faced, so this must not drift from the accessor the simulator used.
    """

    def __init__(self, conn, station_codes: list[int]) -> None:
        self._dates: dict[int, list[str]] = {}
        self._prices: dict[int, list[float]] = {}
        for code in station_codes:
            series = _db.get_daily_prices(conn, code)
            self._dates[code] = [d for d, _ in series]
            self._prices[code] = [p for _, p in series]

    def price_at(self, station_code: int, as_of: str) -> float | None:
        dates = self._dates.get(int(station_code))
        if not dates:
            return None
        idx = bisect.bisect_right(dates, as_of) - 1
        return self._prices[int(station_code)][idx] if idx >= 0 else None

    def is_observed(self, station_code: int, as_of: str) -> bool:
        dates = self._dates.get(int(station_code)) or []
        idx = bisect.bisect_left(dates, as_of)
        return idx < len(dates) and dates[idx] == as_of


def _attach_run_contributions(
    summary: dict, fills: pd.DataFrame, breakdown_per_fold: list[dict], delta_cpl_held: float,
) -> None:
    """The "→ run Δ c/L" column (fps-e1w change 1): what each fold's flip divergence is
    actually worth at the scale of the whole run, litres-weighted so it reconciles EXACTLY to
    `delta_cpl_held` (mutates `summary` in place).

    This is NOT `w_f * flip_cpl_delta` — a flip-only shift-share does not reconcile (a fills-
    weighted version was tried and lands on -0.0008 against a -0.0735 headline; see design
    note). It is the standard shift-share of the run's OWN headline ratio: `w_f`, the fold's
    share of TOTAL run litres (every fill, not just flipped ones — the same population
    `delta_cpl_held` itself pools), times `delta_cpl_own` (facts["breakdowns"]["per_fold"]'s
    whole-fold pooled delta, matching and flipped fills alike — NOT `flip_cpl_delta`, which
    only covers the flipped subset). `flip_cpl_delta` stays useful as a DESCRIPTIVE figure
    beside its own SE; it is not the input to this reconciliation.

    `composition_residual_cpl` is the leftover between `sum_f w_f * delta_cpl_own_f` and the
    true `delta_cpl_held` — real, not a rounding artefact, because a litres-weighted average
    of per-fold RATIOS is not exactly the ratio of pooled totals whenever the two arms' litres
    mix differs fold to fold (the composition of WHERE volume falls, not just how much). fps-
    6yi's own residual was -0.0129 against a -0.0735 headline, ~18% of it — material enough
    to show as its own line rather than absorb silently.

    A fold whose breakdown row is suppressed (`delta_cpl_own` is `None`, below
    `min_row_cell_n`) contributes `None` here too — its true contribution is folded into the
    residual rather than guessed at, same as any other unmeasured fold.

    **The residual also absorbs any tau divergence between arms, not just composition drift**
    (PR #347 review finding #3). `delta_cpl_own` is an OWN-tau quantity (`experiments/lib/
    realised.py`'s per-window `cpl_own`, each arm thresholded at its own selected tau) while
    `delta_cpl_held` is HELD-tau (both arms thresholded at the same tau — the whole point of
    "held" being a clean, operating-point-free comparison). When the two arms' own tau
    actually agree, own and held CPL move together and this distinction is invisible; when
    they diverge, part of `composition_residual_cpl` is the tau gap, not composition drift.

    fps-4je threads `run_paired_realised_backtest`'s own `tau_diverges` flag (`experiments/
    lib/realised.py`'s `deltas` return value, per fold) through `results["realised_deltas"]`
    into `breakdown_per_fold`'s `tau_diverges` column, so this can now be told apart rather
    than caveated unconditionally. `summary["tau_diverges_any"]` set here is the reader's
    single gate: `True` — at least one fold that actually entered `run_contribution_total_cpl`
    (i.e. NOT suppressed — see below) had its own-tau genuinely differ between arms, so
    `composition_residual_cpl` is composition drift AND tau drift, not composition drift
    alone; `False` — own-tau agreed in every such fold, so the residual is composition drift
    only; `None` — either this results.json predates fps-4je (no `realised_deltas` key at
    all — same fallback if the key is simply absent from an older facts.json, PR #354 review
    finding #2), or every graded fold was suppressed, leaving nothing to vote either way. Fall
    back to the old unconditional "composition drift, and possibly tau drift" caveat
    (docs/routines/dossier.md) whenever this is `None`.

    **A suppressed fold's `tau_diverges` must not vote here** (PR #354 review finding #1): a
    suppressed fold (`delta_cpl_own` is `None`, below `min_row_cell_n`) contributes `None` to
    `run_contribution_total_cpl` above — its own-tau was never actually used in the sum this
    flag is meant to explain, so a divergence there says nothing about whether the RESIDUAL
    reflects tau drift. Only counting suppressed-fold flags would let a single thin, unmeasured
    fold force `composition_residual_cpl` to be mis-captioned as "tau drift" when the residual
    is 100% unmeasured composition, as the review's own reproduction against this module's test
    fixture showed. `breakdown_per_fold`'s own `tau_diverges` column is untouched by this —
    only the run-level roll-up excludes suppressed folds.
    """
    arms = [a for a in fills["arm"].unique() if a in (BASELINE_ARM, CANDIDATE_ARM)]
    run_litres = fills[fills["arm"].isin(arms)]["litres"].sum()
    delta_own_by_fold = {row["fold"]: row.get("delta_cpl_own") for row in breakdown_per_fold}
    tau_diverges_by_fold = {row["fold"]: row.get("tau_diverges") for row in breakdown_per_fold}

    contributions: dict[int, float | None] = {}
    for row in summary["per_fold"]:
        fold = row["fold"]
        fold_litres = fills.loc[
            (fills["fold"] == fold) & (fills["arm"].isin(arms)), "litres"
        ].sum()
        weight = fold_litres / run_litres if run_litres > 0 else 0.0
        delta_own = delta_own_by_fold.get(fold)
        contribution = weight * delta_own if delta_own is not None else None
        row["run_contribution_cpl"] = contribution
        contributions[fold] = contribution

    total = sum(v for v in contributions.values() if v is not None)
    summary["run_contribution_total_cpl"] = total
    summary["composition_residual_cpl"] = delta_cpl_held - total
    # Only folds that actually entered `total` above (delta_own_by_fold is not None — not
    # suppressed) get a say in whether the RESIDUAL reflects tau drift.
    tau_flags = [
        tau_diverges_by_fold.get(fold) for fold in contributions if delta_own_by_fold.get(fold) is not None
    ]
    summary["tau_diverges_any"] = (
        any(tau_flags) if tau_flags and all(f is not None for f in tau_flags) else None
    )


def _provenance(results: dict, batch_dir: pathlib.Path | None) -> dict:
    meta = results.get("meta", {})
    batch_name, snapshot_date = None, None
    if batch_dir is not None:
        batch_name = batch_dir.name
        freeze_path = batch_dir / FREEZE_MANIFEST_FILENAME
        if freeze_path.exists():
            snapshot_date = json.loads(freeze_path.read_text()).get("snapshot_date")
    git_sha = meta.get("git_sha")
    git_sha_is_run_time = git_sha is not None
    if git_sha is None:
        # Older results.json (pre fps-3jj.6) didn't persist this — fall back to the current
        # checkout's SHA, which is the *dossier build's* SHA, not necessarily the run's.
        git_sha = current_git_sha()
    return {
        "candidate": results.get("candidate", {}).get("name"),
        "batch": batch_name,
        "batch_dir": str(batch_dir) if batch_dir is not None else None,
        "snapshot_date": snapshot_date,
        "git_sha": git_sha,
        "git_sha_is_run_time": git_sha_is_run_time,
        "seeds": meta.get("seeds"),
        # The realised arbiter's OWN seed — a single draw, distinct from the WFCV
        # screen's `seeds` list above (fps-qbv). A dossier reader who only sees `seeds`
        # can misread the headline realised CPL as averaged over five seeds when it is
        # one; None for results.json predating this field (pre fps-3jj.4).
        "realised_seed": meta.get("realised_seed"),
        "wall_seconds": meta.get("wall_seconds"),
        "status": results.get("status"),
        # Structured reason a status=="aborted_candidate" run aborted (fps-3jj.13) — null
        # for every other status and for aborts that predate the convention. Carried here
        # (not just left in results.json) because retrospective.py reads facts.json, not
        # results.json, for its outcome tally.
        "abort_reason": results.get("abort_reason"),
        "bead_id": meta.get("bead_id"),
        # Baseline identity (fps-zci item 5). None for runs that predate the field —
        # which is exactly the population where a wrong R0 could be hiding, so the
        # absence is itself the signal, not a formatting inconvenience.
        "n_baseline_columns": meta.get("n_baseline_columns"),
        "baseline_fingerprint": meta.get("baseline_fingerprint"),
        # This run's own empirical shock-fold set (fps-3tu, experiments.lib.folds.
        # compute_shock_folds) — a run-level fact, not a shared constant, since it's
        # derived from this run's own baseline fit. None for results.json that predate
        # this field, which graded the "regime" axis (per_regime, decision_flips,
        # summarise_regret) against the old fixed-index SHOCK_FOLDS instead — every
        # regime-dependent table below refuses rather than silently reusing that
        # constant post-hoc.
        "shock_folds": meta.get("shock_folds"),
        # The cadence every realised CPL in this dossier (headline, breakdowns) was
        # produced at (fps-15c). build_facts() below refuses to proceed past this
        # point without it whenever status=="graded" — a graded run is exactly
        # the case where headline/breakdowns are about to carry a CPL.
        "tank_params": meta.get("tank_params"),
        # Exact numeric fields behind the stamp above (fps-o0h) — None for results.json
        # that predate this field, in which case decision-flip/regret windows fall back to
        # re-parsing the (display-rounded, tie-prone) stamp string instead.
        "tank_params_fields": meta.get("tank_params_fields"),
    }


def _validation(results: dict, rowpreds: pd.DataFrame | None) -> dict:
    status = results["status"]
    meta = results.get("meta", {})
    if status == STATUS_GRADED:
        pit_test = "passed — reached the WFCV/realised stages, so the differential PIT truncation test found no leak"
        inputs_check = "passed — INPUTS declaration matched every column add_columns/add_axis actually read"
    elif status == "disqualified":
        pit_test = f"FAILED — {results.get('error')}"
        inputs_check = "not reached"
    else:
        pit_test = f"not reached (status={status})"
        inputs_check = results.get("error") if status == "aborted_candidate" else f"not reached (status={status})"

    nan_rates = None
    if rowpreds is not None:
        # The candidate's own columns are results.json's own declaration (candidate.COLUMNS,
        # runner.py's authoritative record of what add_columns actually produced) — not
        # reverse-engineered by diffing rowpreds' schema against a hardcoded ident-column set.
        # The latter would silently misclassify the next ident column the runner starts
        # persisting (exactly what this PR did by adding cycle_pct_through) as a "candidate
        # feature" here, with no test catching it.
        declared_cols = results.get("candidate", {}).get("columns") or []
        candidate_cols = [c for c in declared_cols if c in rowpreds.columns]
        if candidate_cols:
            nan_rates = {c: float(rowpreds[c].isna().mean()) for c in candidate_cols}

    # The REPLAY's own coverage, not the offline frame's (fps-qbv). `candidate_column_nan_rate`
    # above describes the OFFLINE features frame and says nothing about how much of the
    # realised backtest actually ran with the candidate's extra-feature columns present —
    # `_check_provider_health` only errors at 100% misses, so batch1's 607/6300 (9.6%) partial
    # miss rate passed silently and no dossier could report it. None for results.json that
    # predate the provider-stats stamp (pre fps-3jj / candidates with no EXTRA_FEATURE_PROVIDER).
    provider_hits = meta.get("extra_feature_provider_hits")
    provider_misses = meta.get("extra_feature_provider_misses")
    provider_miss_rate = None
    if provider_hits is not None and provider_misses is not None and (provider_hits + provider_misses) > 0:
        provider_miss_rate = float(provider_misses / (provider_hits + provider_misses))
    return {
        "pit_test": pit_test,
        "inputs_check": inputs_check,
        "candidate_column_nan_rate": nan_rates,
        "extra_feature_provider_hits": provider_hits,
        "extra_feature_provider_misses": provider_misses,
        "extra_feature_provider_miss_rate": provider_miss_rate,
    }


def _redundancy(candidates_batch_dir: pathlib.Path | None, candidate_name: str | None) -> dict:
    """Per-column R^2 against the lock, from `batch.json` (`redundancy.write_batch_record`,
    fps-qbv).

    The block figure already in `results.json`/facts elsewhere is the MEAN of a
    candidate's member columns' individual R^2 against the lock — by construction it
    dilutes a single lock-redundant column out of view (generator.md's own "a member
    that is individually reconstructible is not disqualifying if the group as a whole
    is not" rule). The per-column numbers already exist in `batch.json`, written at
    screen time before filing, and answer the "is this just <X> in different clothes?"
    falsification test several candidates declare — this just carries them into
    facts.json rather than leaving them stranded in a batch-level file no dossier
    session reads.

    `candidates_batch_dir` (fps-1l1 review) is deliberately NOT `meta["batch_dir"]` —
    that is the runner's frozen-artifact dir (`experiments/batches/<batch>/`), and
    `batch.json` never lives there. It is `redundancy.batch_dir_for(module_paths)`,
    the candidate MODULES' parent directory (`experiments/candidates/<batch>/`), which
    the caller derives from `run_dir.parent`. Passing `meta["batch_dir"]` here would
    always miss the file and always claim "the record was never written" — true for
    no batch that has actually been screened.
    """
    if candidates_batch_dir is None:
        return {"available": False, "reason": "no batch_dir recorded in results.json meta"}
    path = candidates_batch_dir / BATCH_RECORD_JSON
    if not path.exists():
        return {
            "available": False,
            "reason": f"no {BATCH_RECORD_JSON} in {candidates_batch_dir} — this batch predates "
            "the redundancy screen, or the record was never written.",
        }
    batch_record = json.loads(path.read_text())
    members = {row["candidate"]: row for row in batch_record.get("block_r2", [])}
    row = members.get(candidate_name)
    if row is None:
        return {
            "available": False,
            "reason": f"{candidate_name!r} not found among {BATCH_RECORD_JSON}'s block_r2 "
            f"candidates ({sorted(members)}) — filed after the batch record was written, or "
            "the candidate name changed since.",
        }
    return {
        "available": True,
        "block_r2": row.get("block_r2"),
        "max_column_r2": row.get("max_column_r2"),
        "per_column_r2": row.get("per_column_r2"),
        "n_predictors_used": row.get("n_predictors_used"),
        "predictors_dropped": row.get("predictors_dropped"),
    }


def _resolve_effective_n_draws(noise_floor: dict, *, nominal: int) -> float:
    """The band's EFFECTIVE draw count — stamped if present, RECONSTRUCTED if not.

    `compute_noise_floor` stamps `effective_n_draws` from `fps-3jj.25` onward. Older floors do
    not carry it, and the first version of this fallback simply used the nominal count on the
    reasoning that such floors "predate column reuse entirely, so nominal IS effective".

    **That reasoning was wrong for a real window** (review of PR #336). `fps-3jj.21` landed the
    unbounded, reuse-capable `candidate_sequence` one PR BEFORE the stamp existed, so any floor
    built from main in between can contain genuine reuse and no stamp — e.g. 20 draws at arity 3
    is 60 picks from ~49 usable columns, worth ~17.5 effective draws, and would have been graded
    at 20. Silent, and in the too-easy direction: exactly the bias this machinery exists to
    remove.

    So reconstruct rather than assume. A floor records `placebo_draws` (one entry per placebo
    COLUMN, with its `source_column`) and `n_placebo_columns`, which is enough to rebuild the
    bank's overlap structure exactly and re-price it. Only a floor missing `placebo_draws`
    altogether falls back to nominal, and that is genuinely safe: no such floor could have been
    written by any construction that reuses columns.
    """
    stamped = noise_floor.get("effective_n_draws")
    if stamped:
        return float(stamped)

    members = noise_floor.get("placebo_draws") or []
    arity = int(noise_floor.get("n_placebo_columns", 1) or 1)
    if not members or arity < 1 or len(members) % arity:
        return float(nominal)
    # Group by the `draw` key when every member carries one (noise_floor.py has stamped it
    # since fps-3jj.14), falling back to positional chunking otherwise — pre-fps-3jj.14 floors
    # have no such key and are arity 1, where the two agree. Positional chunking alone is
    # correct under every writer that exists, so this is belt-and-braces against a future one
    # that emits members out of order rather than a fix for an observed defect.
    if all("draw" in m for m in members):
        grouped: dict[int, list[tuple[str, int, float]]] = {}
        for m in members:
            grouped.setdefault(int(m["draw"]), []).append(
                (str(m.get("source_column")), int(m.get("block_seed", 0)), 0.0)
            )
        draws = [grouped[k] for k in sorted(grouped)]
    else:
        draws = [
            [
                (str(m.get("source_column")), int(m.get("block_seed", 0)), 0.0)
                for m in members[i:i + arity]
            ]
            for i in range(0, len(members), arity)
        ]
    if not draws:
        return float(nominal)
    return effective_n_draws(draws)


#: Glob for every persisted noise-floor bank in a batch dir. `noise_floor.json` is the one
#: `_noise_band` grades against; the siblings are earlier/alternative calibrations kept beside
#: it (the arity-refusal message above tells operators to `mv` rather than `--force`, so they
#: accumulate by design). fps-30p: read them all.
NOISE_FLOOR_GLOB = "noise_floor*.json"


def _band_std_usable(deltas: np.ndarray, band_std: float) -> bool:
    """True iff `band_std` (an `np.std(deltas, ddof=1)`) is a real, positive ruler.

    `band_std > 0` alone is NOT enough to catch "every draw landed on the same value" —
    the case noise_floor.py's docstring names. np.std of twenty identical 0.01s is
    1.78e-18, not 0.0, because 0.01 has no exact binary representation; that sails past
    a `> 0` test and yields a z of order 1e18 (fps-tnz). np.ptp is exact and zero iff the
    draws really are identical, so the two together cover both the degenerate and the
    non-finite case. Shared by the canonical (`_noise_band`) and sibling
    (`_comparable_noise_banks`, fps-30p/PR #345) paths so the two checks cannot drift
    apart the way they did before fps-tnz.
    """
    return bool(np.isfinite(band_std) and band_std > 0 and float(np.ptp(deltas)) > 0)


def _bank_admissibility(
    bank: dict,
    *,
    run_meta: dict,
    run_arity: int,
    refuse_identity_mismatch: bool,
    refuse_null_method_mismatch: bool,
    refuse_narrow_arity: bool,
) -> dict:
    """Everything that decides "is this bank usable" BEFORE its band statistics are computed:
    identity (`baseline_fingerprint`, `tank_params`), `null_method`, arity, and `partial`
    (fps-1x5 — the one place `_noise_band` and `_comparable_noise_banks` both decide this,
    replacing two independently-written copies that had already diverged once, fps-tnz).

    Strictness is a parameter per axis because the two callers deliberately disagree on it —
    see their own docstrings for why. `refuse_identity_mismatch` mirrors `_noise_band`'s own
    `check_fingerprint` (retrospective.py's dummy-results path sets it False; a floor cannot
    be shown too narrow for a run with no real arity to compare against either, so
    `_noise_band` gates `refuse_narrow_arity` the same way). `refuse_null_method_mismatch` is
    unconditional strictness on the canonical path and always off on the sibling path — a
    pinned-source bank measures a genuinely different null and is worth reporting, never worth
    promoting to canonical (fps-awz vs fps-30p). `refuse_narrow_arity` only ever refuses a
    floor NARROWER than the run; a floor at or above the run's arity is always admissible and
    always disclosed via `arity_exceeds_run`/`arity_excess` on both paths, because a wider
    ruler can only raise the bar (fps-3jj.14/fps-3jj.25).

    Returns `{"ok": True, "floor_arity", "arity_exceeds_run", "arity_excess",
    "narrower_than_run"}` when admissible. Returns `{"ok": False, "axis", "floor_value",
    "run_value"}` — one of "baseline_fingerprint" / "null_method" / "tank_params" / "arity" /
    "partial" — naming which check failed, for the caller to build its OWN (differently
    worded — canonical is actionable-and-verbose, siblings are terse) refusal message from.
    Message text is deliberately NOT unified here: the two callers' readers are different
    (an unattended dossier session pasting canonical's reason into a README vs. a batch-level
    corroboration listing), and forcing one wording onto both was never asked for by fps-1x5.
    """
    null_method = bank.get("null_method")
    # NOT `... or 1` (review of PR #349) — that would silently promote a bank explicitly
    # claiming 0 placebo columns into an arity-1 ruler, exactly the bias fps-3jj.14 exists to
    # catch. Only a MISSING key defaults to 1 (a floor predating fps-3jj.14 entirely WAS
    # arity 1); an explicit 0 stays 0 and fails the arity guard like it always has.
    floor_arity = int(bank.get("n_placebo_columns", 1))

    if refuse_identity_mismatch:
        floor_fingerprint = bank.get("baseline_fingerprint")
        run_fingerprint = run_meta.get("baseline_fingerprint")
        if floor_fingerprint is None or floor_fingerprint != run_fingerprint:
            return {
                "ok": False, "axis": "baseline_fingerprint",
                "floor_value": floor_fingerprint, "run_value": run_fingerprint,
            }

    if refuse_null_method_mismatch and null_method != NULL_METHOD_PLACEBO_COLUMN:
        return {"ok": False, "axis": "null_method", "floor_value": null_method, "run_value": NULL_METHOD_PLACEBO_COLUMN}

    if refuse_identity_mismatch:
        floor_tank_params = bank.get("tank_params")
        run_tank_params = run_meta.get("tank_params")
        if floor_tank_params is None or floor_tank_params != run_tank_params:
            return {"ok": False, "axis": "tank_params", "floor_value": floor_tank_params, "run_value": run_tank_params}

    if refuse_narrow_arity and run_arity > floor_arity:
        return {"ok": False, "axis": "arity", "floor_value": floor_arity, "run_value": run_arity}

    if bank.get("partial"):
        return {"ok": False, "axis": "partial", "floor_value": True, "run_value": False}

    return {
        "ok": True,
        "floor_arity": floor_arity,
        "arity_exceeds_run": bool(run_arity and floor_arity > run_arity),
        "arity_excess": max(0, floor_arity - run_arity) if run_arity else 0,
        "narrower_than_run": bool(run_arity and floor_arity < run_arity),
    }


def _score_bank(deltas: np.ndarray, delta: float | None, bank: dict) -> dict:
    """Band statistics + score: mean, std, effective draw count, estimability, and — when
    `delta` is given — the z-score and single-candidate threshold (fps-1x5's other shared
    half: "one place computes mean/std/n_eff/z/threshold"). Pure computation, identical on
    both paths.

    The ESTIMABILITY GATE'S CONSEQUENCE is deliberately left to the caller, not decided here:
    `_noise_band` still reports a bank with an unestimable std as `available: True` (with
    `candidate_z_vs_band`/threshold null — docs/routines/dossier.md's "present-but-null"
    rule), while `_comparable_noise_banks` refuses it outright (`comparable: False`) — a
    pre-existing, tested divergence (see each caller) that this function does not paper over.
    """
    band_mean = float(np.mean(deltas)) if deltas.size else float("nan")
    band_std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan")
    n_eff = _resolve_effective_n_draws(bank, nominal=int(deltas.size))
    band_std_usable = _band_std_usable(deltas, band_std)
    band_estimable = bool(band_std_usable and n_eff >= 2)
    single_z = family_wise_z_threshold(n_candidates=1, n_draws=n_eff) if band_estimable else None
    return {
        "n_draws": int(deltas.size),
        "effective_n_draws": n_eff,
        "band_mean": band_mean,
        "band_std": band_std,
        "band_std_usable": band_std_usable,
        "band_estimable": band_estimable,
        "single_candidate_z_threshold": single_z,
        "candidate_z_vs_band": (
            (delta - band_mean) / band_std if band_estimable and delta is not None else None
        ),
    }


def _comparable_noise_banks(
    results: dict, batch_dir: pathlib.Path, *, canonical: pathlib.Path, delta: float | None
) -> list[dict]:
    """Every OTHER noise-floor bank in the batch dir, vetted and scored against this run.

    Why (fps-30p): `_noise_band` reads one file by fixed name, so a `not_tested` line of the
    form "a differently-shaped floor might move this call" had no way to be answered from
    inside the dossier — and answering it by hand costs hours of placebo fits that, in the
    case that motivated this, had ALREADY been paid. `stickiness_phase_saddle` (fps-6yi,
    batch1) shipped exactly that line while `noise_floor_k1.json` (20 draws, arity 1, same
    fingerprint / tank_params / null_method, computed five days earlier) sat in the same
    directory and answered it: z=-0.81 there against -0.99 on the graded bank, and an arity-2
    band lies between the two by construction. See
    `feedback_committed_artifacts_before_new_compute`.

    These are CORROBORATION and nothing else. The canonical bank alone drives
    `candidate_z_vs_band` and therefore the ledger outcome (`docs/routines/dossier.md` step
    3's mechanical `-t < z < t` split); a sibling that disagrees is a fact to report, never a
    grade to substitute. Comparability is the same identity contract the canonical bank must
    satisfy — `baseline_fingerprint` (fps-cf8) and `tank_params` (fps-v8o) must match the run,
    and a `--fold-subset` bank is refused — with ONE deliberate relaxation: a differing
    `null_method` is reported rather than refused, because a pinned-source bank
    (`placebo_column_pinned_source`) measures a genuinely different null and its z is still
    worth seeing, provided the reader is told which null it came from. Non-comparable banks
    are listed with their reason rather than dropped, so "no sibling answered this" and "no
    sibling was looked at" cannot be confused.
    """
    run_meta = results.get("meta", {})
    # Local, not borrowed from _noise_band's scope — this helper is also called from the
    # arity-refusal path before that function has finished computing its own.
    run_arity = len(results.get("candidate", {}).get("columns") or [])
    banks: list[dict] = []
    for path in sorted(batch_dir.glob(NOISE_FLOOR_GLOB)):
        if path.resolve() == canonical.resolve():
            continue
        entry: dict = {"name": path.name}
        # One try around the whole per-bank read, INCLUDING the field coercions (review of
        # PR #345). A sibling that parses as JSON but carries a non-numeric
        # `n_placebo_columns` or `deltas_cpl_held` — a stray or half-written file matching the
        # glob — would otherwise raise out of here and stop `build_facts` producing a
        # facts.json at all, letting one junk side-file block a run whose canonical floor is
        # perfectly good. A bank this function cannot read is a SKIPPED bank, never a fatal.
        try:
            bank = json.loads(path.read_text())
            entry["null_method"] = bank.get("null_method")
            entry["n_placebo_columns"] = int(bank.get("n_placebo_columns", 1))
            deltas = np.asarray(bank.get("deltas_cpl_held", []), dtype=float)
            entry["n_draws"] = int(deltas.size)
            # `_score_bank` calls `_resolve_effective_n_draws`, which coerces
            # `effective_n_draws`/`placebo_draws` fields the same way the lines above coerce
            # `n_placebo_columns`/`deltas_cpl_held` — a malformed value there (e.g. a
            # non-numeric `effective_n_draws`) raises exactly the same way, so it has to stay
            # inside this try (review of PR #349): a bank this function cannot read is a
            # SKIPPED bank, never a fatal, for EVERY field it reads, not just the first two.
            score = _score_bank(deltas, delta, bank)
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
            banks.append({**entry, "comparable": False, "reason": f"unreadable: {exc}"})
            continue
        # Permissive on both axes canonical is strict on (fps-1x5): a differing null_method
        # or a narrower arity is DISCLOSED below, never a refusal reason here.
        admiss = _bank_admissibility(
            bank, run_meta=run_meta, run_arity=run_arity,
            refuse_identity_mismatch=True,
            refuse_null_method_mismatch=False,
            refuse_narrow_arity=False,
        )
        if not admiss["ok"]:
            if admiss["axis"] == "partial":
                reason = "computed with --fold-subset (partial fold coverage)"
            else:
                reason = f"{admiss['axis']} {admiss['floor_value']!r} does not match this run's {admiss['run_value']!r}"
            entry.update(comparable=False, reason=reason)
            banks.append(entry)
            continue
        # A bank whose band cannot be ESTIMATED is not comparable, even when its identity
        # matches (review of PR #345). The canonical path already treats this state as
        # ungradeable — `docs/routines/dossier.md`'s "present-but-null z/t" rule — so a
        # sibling in it corroborates nothing, and, worse, the arity refusal below filters
        # on `comparable` alone: marking it True would let the refusal advise promoting a
        # ruler that still cannot grade anything once promoted.
        if score["n_draws"] < 2:
            entry.update(comparable=False, reason=f"only {score['n_draws']} draw(s) — no band to estimate")
        elif not score["band_std_usable"]:
            entry.update(
                comparable=False,
                reason=f"band_std {score['band_std']!r} is not estimable — every draw landed on the "
                "same value, so there is no band to compare against",
            )
        elif score["effective_n_draws"] < 2:
            entry.update(
                comparable=False,
                reason=f"effective_n_draws {score['effective_n_draws']:.2f} < 2 once source-column reuse is "
                f"priced in (nominal {score['n_draws']}) — not enough independent draws to grade",
            )
        else:
            entry.update(
                comparable=True,
                # docs/CONVENTIONS.md groups arity with fingerprint/tank_params as the
                # same failure class, and the canonical path REFUSES a floor narrower
                # than the run (biased in the candidate's favour). Siblings are not
                # refused on it — a narrower bank is exactly what answers "would a
                # different arity move this call", and excluding it would delete the use
                # case this whole helper exists for (fps-6yi was resolved by an arity-1
                # bank against a 2-column run). So disclose rather than drop, the mirror
                # of `floor_arity_exceeds_run` on the canonical side: a True here means
                # this bank's z is biased in the CANDIDATE's favour and is a lower bound
                # on the bar, not a verdict. Never read it as one.
                narrower_than_run=admiss["narrower_than_run"],
                effective_n_draws=score["effective_n_draws"],
                band_mean_delta_cpl_held=score["band_mean"],
                band_std_delta_cpl_held=score["band_std"],
                candidate_z_vs_band=score["candidate_z_vs_band"],
                single_candidate_z_threshold=score["single_candidate_z_threshold"],
            )
        banks.append(entry)
    return banks


def _noise_band(results: dict, batch_dir: pathlib.Path | None, *, check_fingerprint: bool = True) -> dict:
    """`check_fingerprint=False` is for retrospective.py's `_batch_noise_summary`,
    which reuses this function's math to summarise a batch's floor on its own
    (mean/std/n_draws) rather than to grade any one candidate run against it — it
    calls with a dummy `results` that has no real `meta.baseline_fingerprint` or
    `meta.tank_params` to compare, so the per-run identity checks below (fingerprint
    AND tank_params/cadence, fps-v8o) would always (mis)fire as a mismatch.
    `build_facts()`, which DOES grade a real run, always uses the default.

    The admissibility decision and band statistics are shared with the sibling path
    (`_comparable_noise_banks`, fps-30p) via `_bank_admissibility`/`_score_bank` (fps-1x5) —
    this function is the STRICT caller of both: it refuses a `null_method` mismatch and a
    narrower-than-run arity unconditionally (well, `refuse_narrow_arity` still follows
    `check_fingerprint`, for the reason above), where the sibling path discloses instead.
    """
    if batch_dir is None:
        return {"available": False, "reason": "no batch_dir recorded in results.json meta"}
    path = batch_dir / NOISE_FLOOR_FILENAME
    if not path.exists():
        return {
            "available": False,
            "reason": "no noise-floor calibration for this batch — run `PYTHONPATH=. uv run "
            "python -m experiments.pipeline.noise_floor <batch>` (fps-3jj.9)",
        }
    noise_floor = json.loads(path.read_text())
    run_meta = results.get("meta", {})
    run_columns = results.get("candidate", {}).get("columns") or []
    run_arity = len(run_columns)
    # fps-1x5: the identity (baseline_fingerprint, null_method, tank_params), arity and
    # partial checks are the CANONICAL half of `_bank_admissibility` — strict on every axis
    # `_comparable_noise_banks` deliberately relaxes. See that function's docstring for why
    # each strictness choice is what it is; this call only supplies the choices.
    admiss = _bank_admissibility(
        noise_floor, run_meta=run_meta, run_arity=run_arity,
        refuse_identity_mismatch=check_fingerprint,
        refuse_null_method_mismatch=True,
        refuse_narrow_arity=check_fingerprint,
    )
    if not admiss["ok"]:
        axis, floor_value, run_value = admiss["axis"], admiss["floor_value"], admiss["run_value"]
        if axis == "baseline_fingerprint":
            # fps-cf8: a floor computed against one baseline could silently grade a
            # candidate run against a different one (fps-sa1, fps-zci — exactly the
            # failure class feedback_baseline_declared_not_discovered warns about). Refuse
            # rather than trust "file exists" the way this function used to. A floor with
            # no fingerprint at all (pre-fps-cf8) is treated as a mismatch, not a pass —
            # it cannot be shown to match, so it cannot be trusted either. `_bank_admissibility`
            # applies that rule symmetrically: a run with NO baseline_fingerprint of its own
            # can't show a match either, even against a floor that also lacks the field
            # (fps-74x, same both-missing case fps-1x5's review round already fixed on the
            # tank_params axis below — there is no build_facts-level hard refusal for a
            # missing meta.baseline_fingerprint the way fps-15c added for tank_params, so this
            # branch IS reachable via build_facts, unlike the tank_params one).
            if floor_value is None and run_value is None:
                return {
                    "available": False,
                    "reason": "noise_floor.json has no baseline_fingerprint at all (predates "
                    "fps-cf8), and this run's own meta.baseline_fingerprint is also missing — "
                    "neither side can be shown to match the other, so the floor does not grade "
                    "this run. Recompute with `PYTHONPATH=. uv run python -m experiments.pipeline."
                    "noise_floor <batch> --force`.",
                }
            return {
                "available": False,
                "reason": f"noise_floor.json's baseline_fingerprint ({floor_value!r}) does not "
                f"match this run's ({run_value!r}) — the floor was computed against a "
                "different baseline (or predates baseline_fingerprint entirely) and does not grade "
                "this run. Recompute with `PYTHONPATH=. uv run python -m experiments.pipeline."
                "noise_floor <batch> --force`.",
            }
        if axis == "null_method":
            # fps-awz: this PR itself changed what noise_floor.json's deltas_cpl_held MEASURE
            # (a placebo-column null instead of a seed-swap null) without changing
            # baseline_fingerprint (same 54 columns either way) — so the fingerprint check above
            # would happily accept a pre-fps-awz batch's noise_floor.json (e.g. batch0's, already
            # committed) forever, silently grading every future candidate against the OLD, wider,
            # differently-shaped band this rework exists to replace. A floor with no null_method
            # at all (pre-fps-awz) is treated as a mismatch, not a pass — same "cannot be shown to
            # match, so cannot be trusted" rule the fingerprint check already applies.
            return {
                "available": False,
                "reason": f"noise_floor.json's null_method ({floor_value!r}) is not "
                f"{NULL_METHOD_PLACEBO_COLUMN!r} — this floor predates the fps-awz placebo-column "
                "rework (or was computed some other way) and does not grade this run. Recompute "
                "with `PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> "
                "--force`.",
            }
        if axis == "tank_params":
            # fps-v8o: the consuming-side half of fps-15c's stamping work. A floor computed
            # at one cadence (e.g. 7d) grades every draw's delta_cpl_held in that cadence's
            # units — comparing a candidate run at a different cadence against it would
            # silently misjudge the delta the same way a baseline_fingerprint mismatch would
            # (fps-cf8), just along the cadence axis instead of the feature-set axis. A floor
            # with no tank_params at all (pre-fps-v8o) is treated as a mismatch, not a pass —
            # same "cannot be shown to match, so cannot be trusted" rule as the checks above.
            # `_bank_admissibility` applies that rule symmetrically (review of PR #349): a run
            # with NO tank_params of its own can't show a match either, even against a floor
            # that also lacks the field — build_facts() itself already refuses a graded run
            # missing meta.tank_params before ever reaching here (fps-15c), so this branch is
            # unreachable via build_facts and only guards a caller that skips that gate.
            if floor_value is None and run_value is None:
                return {
                    "available": False,
                    "reason": "noise_floor.json has no tank_params at all (predates fps-v8o), and "
                    "this run's own meta.tank_params is also missing — neither side can be shown "
                    "to match the other, so the floor does not grade this run. Recompute with "
                    "`PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> "
                    "--force`.",
                }
            return {
                "available": False,
                "reason": f"noise_floor.json's tank_params ({floor_value!r}) does not match "
                f"this run's ({run_value!r}) — the floor was computed at a different cadence "
                "(or predates tank_params entirely) and does not grade this run. Recompute with "
                "`PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> --force`.",
            }
        if axis == "arity":
            # fps-3jj.14: same failure class as baseline_fingerprint, null_method and
            # tank_params above — a floor that measures a DIFFERENT operation from the one the
            # run performed — but with an asymmetry the others do not have, and the guard is
            # one-sided because of it. A k-column candidate arm has more chances for the fit
            # to find something than a j-column placebo arm does when k > j, so a floor BELOW
            # the run's arity is narrow in the favourable direction and is refused. A floor at
            # or ABOVE the run's arity can only be as wide or wider, i.e. a HARDER bar, so it
            # is allowed and disclosed (`floor_arity_exceeds_run` below) rather than refused:
            # refusing it would force a separate calibration run per distinct arity in a
            # batch, which for batch1 alone (arities 3,3,3,2,2) means two ~2h runs to grade
            # five candidates, buying nothing but a tighter bar the batch does not need.
            floor_arity = floor_value
            # The draw count is no longer forced by arity (fps-3jj.21), so the wider floor is
            # computed at the SAME n_draws as the ruler it replaces — which is also what keeps
            # the two comparable, the thing fps-3jj.14's arity comparison needed and could not
            # have.
            n_draws_hint = int(noise_floor.get("n_draws") or len(noise_floor.get("deltas_cpl_held", [])))
            # fps-30p: before telling anyone to spend ~2h computing a wider ruler, look for one.
            # These banks accumulate precisely because the promote step below says `mv`, not
            # `--force` — so the ruler this run needs may already be sitting beside the
            # canonical one, unread. Only comparable banks count (same fingerprint and
            # cadence). Every condition `_noise_band` itself would apply to the promoted file,
            # not just arity — this hint recommends a `mv`, so a bank that would fail the very
            # next run's checks is worse than no hint at all (review of PR #345).
            # `_comparable_noise_banks` deliberately RELAXES null_method so a pinned-source
            # bank's z can still be reported as corroboration; that relaxation must not leak
            # into a promote recommendation, because the canonical null_method check above is
            # unconditional.
            adequate = [
                b for b in _comparable_noise_banks(results, batch_dir, canonical=path, delta=None)
                if b.get("comparable")
                and b.get("n_placebo_columns", 1) >= run_arity
                and b.get("null_method") == NULL_METHOD_PLACEBO_COLUMN
            ]
            already = (
                "NOTE: a wide-enough ruler may already exist in this batch dir — "
                + ", ".join(
                    f"{b['name']} ({b['n_placebo_columns']} columns, {b['n_draws']} draws)"
                    for b in adequate
                )
                + ". Check it before recomputing; promoting an existing bank is step (2) alone. "
                if adequate
                else ""
            )
            return {
                "available": False,
                "reason_code": NOISE_BAND_REFUSAL_ARITY,
                "reason": already + f"noise_floor.json is a {floor_arity}-column null but this candidate adds "
                f"{run_arity} columns ({', '.join(map(str, run_columns))}) — a wider arm graded "
                "against a narrower ruler, biased in the candidate's favour by an unmeasured "
                f"amount (fps-3jj.14). This batch needs a ruler of at least {run_arity} columns, and "
                "grading reads ONLY noise_floor.json, so producing one is two steps. (1) Compute it "
                "beside the current ruler: `PYTHONPATH=. uv run python -m "
                f"experiments.pipeline.noise_floor <batch> --arity {run_arity} --n-draws "
                f"{n_draws_hint} --out-name noise_floor_k{run_arity}.json` (same draw count as the "
                "current floor, so the two are comparable). (2) Promote it, keeping the "
                f"old one: `mv noise_floor.json noise_floor_k{floor_arity}.json && mv "
                f"noise_floor_k{run_arity}.json noise_floor.json`. Do NOT `--force` over "
                "noise_floor.json instead: that destroys the ruler this batch's existing dossiers "
                "were graded against, and the baseline any arity comparison needs. Full procedure: "
                "docs/CONVENTIONS.md § 'The band's ARITY must be at least the candidate's'. This run is "
                "NOT rejected and must not be written up as such — it completed and graded fine, and "
                "becomes a real measurement once a wide-enough ruler exists. Leave it in the dossier "
                "queue (write no README.md, no ledger entry) until then.",
            }
        if axis == "partial":
            # noise_floor.py's --fold-subset is an iteration/smoke speed-up — the deltas only
            # cover some outer folds, but effect_delta_cpl_held always pools every fold.
            # Grading a candidate against a partial-fold floor as though it were the real one
            # would silently misjudge the delta, not just weaken the estimate — refuse rather
            # than mislabel it.
            return {
                "available": False,
                "reason": "noise_floor.json was computed with --fold-subset (partial fold "
                "coverage) — not a valid ruler against a delta pooled over every fold. Recompute "
                "with `PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> "
                "--force` and no --fold-subset.",
            }
        # Every axis `_bank_admissibility` can name is handled above (review of PR #349) —
        # falling through here would silently mislabel a future axis with the --fold-subset
        # message instead of failing loudly.
        raise AssertionError(f"_bank_admissibility returned an unhandled refusal axis: {axis!r}")
    floor_arity = admiss["floor_arity"]
    deltas = np.asarray(noise_floor.get("deltas_cpl_held", []), dtype=float)
    delta = results.get("effect_delta_cpl_held")
    if delta is None or deltas.size == 0:
        return {"available": False, "reason": "empty noise-floor sample or unresolved effect_delta_cpl_held"}
    score = _score_bank(deltas, delta, noise_floor)
    band_mean, band_std = score["band_mean"], score["band_std"]
    n_eff, band_std_usable = score["effective_n_draws"], score["band_std_usable"]
    single_z = score["single_candidate_z_threshold"]
    # Gated on the NOMINAL count, deliberately NOT on `band_estimable` (review of PR #336).
    # `bar_draw_count_shift_cpl_held` below is a function of the nominal draw count alone and is
    # documented as "a property of this floor alone" — routing it through `band_estimable` would
    # let a REUSE condition (`n_eff >= 2`) null it out, re-coupling the two quantities this
    # commit just separated. Latent rather than live today (rho_bar caps at TEXTURE_ICC_BOUND,
    # so n_eff cannot fall under 2 at 20 draws), but a 5-draw fully-overlapping bank reaches
    # n_eff 1.95 and would lose a thinness value its nominal 5 supports perfectly well.
    single_z_nominal = (
        family_wise_z_threshold(n_candidates=1, n_draws=int(deltas.size))
        if band_std_usable and deltas.size >= 2
        else None
    )
    return {
        "available": True,
        "n_draws": int(deltas.size),
        # What the bank is WORTH once source-column reuse is priced in — the number the
        # thresholds above actually use. Equal to n_draws when no two draws share a column.
        "effective_n_draws": n_eff,
        "texture_icc_assumed": TEXTURE_ICC_BOUND,
        "n_placebo_columns": floor_arity,
        "candidate_n_columns": run_arity,
        # True when the ruler is WIDER-armed than the run it grades — allowed (it can only
        # make the bar harder) but worth surfacing, because a candidate that fails against
        # such a floor has not been shown to fail against its own arity's band.
        "floor_arity_exceeds_run": admiss["arity_exceeds_run"],
        "band_mean_delta_cpl_held": band_mean,
        "band_std_delta_cpl_held": band_std,
        "candidate_delta_cpl_held": delta,
        # delta_cpl_held is a COST: more negative = better. "Better than N% of noise" therefore
        # means N% of the noise-floor draws were WORSE (higher/less-negative) than this candidate
        # — (deltas > delta), not (deltas < delta). A candidate with a strong negative delta
        # against a noise band centred near zero must read as a HIGH percentile here, not near 0.
        "candidate_percentile_better_than_noise": float((deltas > delta).mean() * 100),
        "candidate_z_vs_band": score["candidate_z_vs_band"],
        # "The honest single-candidate bar" (docs/CONVENTIONS.md), n_candidates=1 — the magnitude
        # candidate_z_vs_band must clear, in EITHER direction, to be distinguishable from a draw
        # off this batch's own noise floor. Drives the ledger rejected/inconclusive split
        # (docs/routines/dossier.md § Step 1.4, fps-3jj.17); None when band_std isn't estimable
        # (deltas.size < 2), same guard as candidate_z_vs_band itself.
        "single_candidate_z_threshold": single_z,
        # The same threshold computed on the NOMINAL draw count — i.e. before the reuse
        # correction. Emitted so the two components of the bar can be told apart below.
        "single_candidate_z_threshold_nominal": single_z_nominal,
        # fps-3jj.21: the same bar in c/L rather than in band-std units — the form a reader can
        # actually compare against `experiments/results.csv`'s 0.03-0.26 c/L history and against
        # another candidate's. `delta_cpl_held` is a COST, so clearing the bar means coming in
        # AT OR BELOW this number.
        #
        # This is the FAVOURABLE side only. `single_candidate_z_threshold` is two-sided ("in
        # EITHER direction", and dossier.md's three-region rule is `-t < z < t`), so a candidate
        # landing ABOVE `band_mean + z * band_std` is also distinguishable from noise — just
        # surprisingly BAD — and this single number cannot express that. Judge two-sidedness on
        # `abs(candidate_z_vs_band)` against the z threshold; use this one to read how hard the
        # favourable bar is in units the rest of the project quotes.
        "single_candidate_bar_cpl_held": (
            band_mean - single_z * band_std if single_z is not None else None
        ),
        # How much of that bar is band THINNESS rather than band WIDTH. SIGNED, and negative
        # whenever the band is estimated from finite draws — `_LARGE_SAMPLE_Z` is strictly below
        # any finite-n threshold, so a LARGER penalty reads as a MORE NEGATIVE number. Named
        # "shift" rather than "penalty" for exactly that reason (review of PR #335): it is the
        # signed move of a cost, on the same scale and in the same direction as the bar itself,
        # not a magnitude.
        #
        # Computed against the NOMINAL draw count on purpose (review of PR #336). Once
        # `family_wise_z_threshold` took the EFFECTIVE count, deriving this from the effective
        # threshold silently folded an arity-driven reuse effect into a number whose entire
        # job is to separate draw thinness from everything else —
        # `experiments/2026-08-23_placebo_arity/` used exactly this quantity to attribute two
        # thirds of batch1's k=1 -> k=3 bar move to thinness rather than arity, and the
        # conflated version would have credited reuse to "draw count". The reuse charge is
        # emitted separately as `bar_reuse_shift_cpl_held`; the two sum to the full distance
        # between the large-sample bar and the one actually applied. The t-critical carries
        # `df = n_draws - 1`, so a floor estimated from few draws sets a harder bar than the
        # same band estimated from many — and `experiments/2026-08-23_placebo_arity/` measured
        # two thirds of batch1's k=1 -> k=3 bar move as exactly this, not as the arity effect it
        # was being read as. Quoted against the large-sample limit (the band known rather than
        # estimated), so it is a property of this floor alone and comparable across floors.
        "bar_draw_count_shift_cpl_held": (
            (_LARGE_SAMPLE_Z - single_z_nominal) * band_std
            if single_z_nominal is not None
            else None
        ),
        # The rest of the distance: how much harder the bar is because the bank's draws REUSE
        # source columns and so are worth less than their count (fps-3jj.25). Zero for a
        # reuse-free bank. Kept separate from the thinness shift above so neither can absorb
        # the other; together they span large-sample bar -> applied bar.
        "bar_reuse_shift_cpl_held": (
            (single_z_nominal - single_z) * band_std
            if single_z is not None and single_z_nominal is not None
            else None
        ),
        # Columns by which the ruler is WIDER-armed than the run it grades. Allowed (a wider
        # ruler can only raise the bar) but it means a candidate failing here has not been shown
        # to fail against its OWN arity's band — the comparison fps-3jj.21 asked the dossier to
        # render rather than merely capture.
        "floor_arity_excess": admiss["arity_excess"],
        # fps-30p: every OTHER bank in this batch dir, vetted and scored against the same
        # delta. Corroboration only — `candidate_z_vs_band` above (the canonical bank) is
        # what the ledger outcome keys on, and nothing here may substitute for it. Empty
        # under `check_fingerprint=False`, where `results` is retrospective.py's dummy and
        # has no real fingerprint/tank_params for a sibling to be compared against.
        "comparable_banks": (
            _comparable_noise_banks(results, batch_dir, canonical=path, delta=delta)
            if check_fingerprint
            else []
        ),
    }


def _headline(results: dict) -> dict:
    agg = {row["arm"]: row for row in results["aggregate"]}
    r0, cand = agg.get(BASELINE_ARM, {}), agg.get(CANDIDATE_ARM, {})
    fold_deltas = pd.DataFrame(results["fold_run_deltas"])
    cand_rows = fold_deltas[fold_deltas["run"] == CANDIDATE_ARM] if len(fold_deltas) else fold_deltas
    return {
        "realised": {
            "arbiter": "realised CPL (cost per litre), held tau — clean attribution, isolated from the operating point",
            "r0_cpl_held": r0.get("cpl_held"),
            "candidate_cpl_held": cand.get("cpl_held"),
            "delta_cpl_held": results.get("effect_delta_cpl_held"),
            "effect_resolved": results.get("effect_resolved"),
            "r0_cpl_own": r0.get("cpl_own"),
            "candidate_cpl_own": cand.get("cpl_own"),
        },
        "wfcv_log_loss": {
            "label": (
                "descriptive colour, NOT the arbiter (docs/CONVENTIONS.md — #250/#254 both "
                "showed flat WFCV log-loss with opposite realised outcomes)"
            ),
            "delta_ll_all_mean_across_folds": (
                float(cand_rows["delta_ll_all_median"].mean()) if len(cand_rows) else None
            ),
            "delta_ll_hard25_mean_across_folds": (
                float(cand_rows["delta_ll_hard25_median"].mean()) if len(cand_rows) else None
            ),
        },
        "zone": results.get("zone"),
        "pass_criterion": results.get("meta", {}).get("pass_criterion"),
    }


def _cell_delta(fills: pd.DataFrame, mask: pd.Series, *, min_row_cell_n: int) -> tuple[float | None, int, bool]:
    cand_fills = fills[mask & (fills["arm"] == CANDIDATE_ARM)]
    base_fills = fills[mask & (fills["arm"] == BASELINE_ARM)]
    n = min(len(cand_fills), len(base_fills))
    suppressed = n < min_row_cell_n
    delta = None if suppressed else pooled_cpl(cand_fills) - pooled_cpl(base_fills)
    return delta, n, suppressed


def _breakdowns(
    results: dict, rowpreds: pd.DataFrame, fills: pd.DataFrame,
    *, min_row_cell_n: int = DEFAULT_MIN_ROW_CELL_N,
) -> dict:
    fold_deltas = pd.DataFrame(results["fold_run_deltas"])
    cand_ll = (
        fold_deltas[fold_deltas["run"] == CANDIDATE_ARM].set_index("fold")
        if len(fold_deltas)
        else pd.DataFrame()
    )

    # tau_diverges per fold (fps-4je): sourced from realised.py's own per-fold comparison of
    # the candidate's vs. baseline's own-tau, threaded through results["realised_deltas"] —
    # not re-derived here. None (not False) for a results.json predating fps-4je, which never
    # wrote this key at all, so a legacy run can't be misread as "tau agreed everywhere".
    raw_realised_deltas = results.get("realised_deltas")
    tau_diverges_by_fold: dict[int, bool] | None = None
    if raw_realised_deltas is not None:
        realised_deltas = pd.DataFrame(raw_realised_deltas)
        cand_deltas = (
            realised_deltas[realised_deltas["arm"] == CANDIDATE_ARM]
            if len(realised_deltas)
            else realised_deltas
        )
        # PR #354 review finding #5: don't coerce blindly — a missing/NaN cell (heterogeneous
        # records feeding a NaN into this column) must stay None, not silently become False
        # (missing) or True (NaN is truthy under bool()). Same None-vs-False discipline as the
        # results["realised_deltas"] is not None check above, one level down.
        tau_diverges_by_fold = {
            int(row["fold"]): (None if pd.isna(row["tau_diverges"]) else bool(row["tau_diverges"]))
            for _, row in cand_deltas.iterrows()
        }

    # This run's own empirical shock-fold set (fps-3tu) — None for a results.json that
    # predates it, which graded "regime" against the old fixed-index SHOCK_FOLDS. Rather
    # than silently reuse that superseded constant post-hoc, per_fold's "regime" and the
    # whole per_regime table degrade to an explicit refusal; nothing else here depends on
    # shock/normal labelling.
    raw_shock_folds = results.get("meta", {}).get("shock_folds")
    shock_folds = frozenset(int(f) for f in raw_shock_folds) if raw_shock_folds is not None else None
    # Shared with per_regime's own {"computed": False, "reason": ...} below rather than
    # restated — one string, two symptoms. per_fold[]'s bare "regime": None otherwise
    # carries no explanation of its own (review finding on PR #350): a reader hitting a
    # blank regime column with no accompanying note has no way to tell "not computable"
    # from "no regime effect", which is exactly the ambiguity this module's computed/
    # reason contract exists to remove everywhere else.
    regime_unavailable_reason = (
        None if shock_folds is not None else
        "this run's results.json predates fps-3tu's empirical shock-fold set "
        "(meta.shock_folds) — re-run the candidate to backfill it."
    )

    per_fold = []
    for fold in sorted(fills["fold"].unique()):
        delta, n, suppressed = _cell_delta(fills, fills["fold"] == fold, min_row_cell_n=min_row_cell_n)
        row = {
            "fold": int(fold),
            "regime": (
                ("shock" if fold in shock_folds else "normal") if shock_folds is not None else None
            ),
            "n_fills": n,
            "suppressed": suppressed,
            "delta_cpl_own": delta,
            "tau_diverges": (
                tau_diverges_by_fold.get(int(fold)) if tau_diverges_by_fold is not None else None
            ),
        }
        if fold in cand_ll.index:
            row["delta_ll_all_median"] = float(cand_ll.loc[fold, "delta_ll_all_median"])
            row["delta_ll_hard25_median"] = float(cand_ll.loc[fold, "delta_ll_hard25_median"])
        per_fold.append(row)

    if shock_folds is None:
        per_regime = {"computed": False, "reason": regime_unavailable_reason}
    else:
        all_folds = set(fills["fold"].unique())
        per_regime = []
        for regime_name, folds in (("shock", shock_folds & all_folds), ("normal", all_folds - shock_folds)):
            delta, n, suppressed = _cell_delta(fills, fills["fold"].isin(folds), min_row_cell_n=min_row_cell_n)
            per_regime.append(
                {"regime": regime_name, "n_fills": n, "suppressed": suppressed, "delta_cpl_own": delta}
            )

    per_axis = None
    per_axis_coverage_note = None
    if "axis" in rowpreds.columns:
        axis_lookup = (
            rowpreds[["station_code", "price_date", "axis"]]
            .drop_duplicates()
            .assign(date=lambda d: pd.to_datetime(d["price_date"]).dt.strftime("%Y-%m-%d"))
            [["station_code", "date", "axis"]]
        )
        # many_to_one, not a bare left join: axis is supposed to be a function of (station_code,
        # date) alone (same invariant the tau-sweep label check relies on) — if add_axis ever
        # produces two different labels for the same key, this must raise loudly (mirroring
        # runner.py's _resolve_zone, which does the same merge with the same guard) rather than
        # silently fan out and inflate n_fills / skew pooled_cpl.
        tagged = fills.merge(axis_lookup, on=["station_code", "date"], how="left", validate="many_to_one")
        n_unmatched = int(tagged["axis"].isna().sum())
        per_axis_coverage_note = (
            f"axis_lookup is built from rowpreds.parquet, which only carries WFCV "
            f"validation-window rows — not the full candidate frame results.json's own `zone` "
            f"grading uses. {n_unmatched} of {len(tagged)} fills had no axis match and are "
            f"excluded from every per_axis cell below; per_axis deltas may not fully agree with "
            f"`headline.zone`."
        )
        per_axis = []
        for axis_value in sorted(tagged["axis"].dropna().unique(), key=str):
            delta, n, suppressed = _cell_delta(tagged, tagged["axis"] == axis_value, min_row_cell_n=min_row_cell_n)
            per_axis.append({"axis": axis_value, "n_fills": n, "suppressed": suppressed, "delta_cpl_own": delta})

    return {
        "per_fold": per_fold, "per_regime": per_regime, "per_axis": per_axis,
        # None whenever per_fold[]'s "regime" values are real (shock/normal); explains
        # BOTH per_fold[].regime being bare None and per_regime being computed:false
        # otherwise, so a reader hitting either symptom finds the reason beside it
        # rather than having to infer it from the other's shape.
        "per_fold_regime_unavailable_reason": regime_unavailable_reason,
        "per_axis_coverage_note": per_axis_coverage_note,
        # per_fold and per_regime cut on WHOLE folds, each of which is an independent
        # simulation with its own tank, so they are sums of complete windows. per_axis cuts
        # on a per-row label, which slices through a window and allocates a path-coupled
        # cost — not identified (fps-grp). The caveat ships WITH the number so the dossier
        # reader cannot meet the delta without it; None when no axis was declared.
        "per_axis_identification_note": ROW_AXIS_ECONOMICS_CAVEAT if per_axis else None,
        "min_row_cell_n": min_row_cell_n,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def make_plots(run_dir: pathlib.Path, facts: dict, batch_dir: pathlib.Path | None) -> list[str]:
    """Write every applicable plot into run_dir; return the filenames written.

    Four always (per the Plots decision): per-fold delta bars, mean-vs-median across seeds,
    realised CPL by fold, tau-sweep. Plus the candidate-over-time/NaN-band plot (unconditional —
    every candidate has at least one COLUMNS entry). Conditional plots are routed off metadata the
    candidate already declares (INPUTS / TARGET / add_axis) — see the trigger checks below.
    """
    run_dir = pathlib.Path(run_dir)
    if facts.get("headline") is None:
        return []  # run didn't reach the scoring stages — nothing to plot

    rowpreds = pd.read_parquet(run_dir / ROWPREDS_FILENAME)
    fills = pd.read_parquet(run_dir / FILLS_FILENAME)
    name = facts["candidate"]["name"]

    candidate_cols = [c for c in (facts["candidate"].get("columns") or []) if c in rowpreds.columns]

    written = []
    for fn in (
        lambda: _plot_per_fold_delta_bars(run_dir, facts, name),
        lambda: _plot_seed_mean_vs_median(run_dir, rowpreds, name),
        lambda: _plot_realised_cpl_by_fold(run_dir, fills, name, facts["provenance"].get("shock_folds")),
        lambda: _plot_tau_sweep(run_dir, rowpreds, name),
        lambda: _plot_candidate_over_time(run_dir, rowpreds, candidate_cols, name),
        lambda: _plot_axis_breakdown(run_dir, facts, name),
        lambda: _plot_cycle_phase(run_dir, rowpreds, facts, candidate_cols, name),
        lambda: _plot_external_overlay(run_dir, facts, batch_dir, name),
    ):
        path = fn()
        if path is not None:
            written.append(path)
    return written


def _plot_per_fold_delta_bars(run_dir: pathlib.Path, facts: dict, name: str) -> str | None:
    per_fold = facts["breakdowns"]["per_fold"] if facts.get("breakdowns") else None
    if not per_fold:
        return None
    df = pd.DataFrame(per_fold)
    if "delta_ll_all_median" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    # pd.notna guard: an NaN here (aggregate_with_deltas' median over an unexpectedly empty/all-NaN
    # cohort) must render as visibly unresolved, not fall through "not > 0" into the "candidate
    # better" blue — a silent misread in an unattended pipeline.
    colours = [
        "#bdc3c7" if pd.isna(v) else ("#c0392b" if v > 0 else "#2980b9")
        for v in df["delta_ll_all_median"]
    ]
    heights = df["delta_ll_all_median"].fillna(0.0)
    edgecolours = ["#f1c40f" if r == "shock" else "black" for r in df["regime"]]
    ax.bar(df["fold"], heights, color=colours, edgecolor=edgecolours, linewidth=2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("fold")
    ax.set_ylabel("Δ WFCV log-loss, all rows, median across seeds\n(positive = candidate worse)")
    ax.set_title(f"{name}: per-fold Δ log-loss (yellow edge = shock fold)")
    ax.set_xticks(df["fold"])
    fig.tight_layout()
    path = run_dir / "per_fold_delta_bars.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_seed_mean_vs_median(run_dir: pathlib.Path, rowpreds: pd.DataFrame, name: str) -> str | None:
    df = rowpreds[rowpreds["run"] == CANDIDATE_ARM].copy()
    if df.empty:
        return None
    y = df["label"].to_numpy(dtype=float)
    p = np.clip(df["proba"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    df["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    per_seed = df.groupby(["fold", "seed"], observed=True)["ll"].mean().reset_index()
    if per_seed["seed"].nunique() < 2:
        return None  # seed-variance gate itself is skipped below 2 seeds — mirror that here
    stats = per_seed.groupby("fold")["ll"].agg(["mean", "median"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for fold in sorted(per_seed["fold"].unique()):
        sub = per_seed[per_seed["fold"] == fold]
        ax.scatter([fold] * len(sub), sub["ll"], color="#888", s=25, zorder=2)
    ax.plot(stats["fold"], stats["mean"], "o-", color="#2980b9", label="mean across seeds")
    ax.plot(stats["fold"], stats["median"], "s--", color="#c0392b", label="median across seeds")
    ax.set_xlabel("fold")
    ax.set_ylabel("candidate mean log-loss (all rows)")
    ax.set_title(f"{name}: seed spread per fold, candidate arm (dots = individual seeds)")
    ax.legend()
    fig.tight_layout()
    path = run_dir / "seed_mean_vs_median.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_realised_cpl_by_fold(
    run_dir: pathlib.Path, fills: pd.DataFrame, name: str, shock_folds: list[int] | None,
) -> str | None:
    folds = sorted(fills["fold"].unique())
    if not folds:
        return None
    r0_cpl = [pooled_cpl(fills[(fills["fold"] == f) & (fills["arm"] == BASELINE_ARM)]) for f in folds]
    cand_cpl = [pooled_cpl(fills[(fills["fold"] == f) & (fills["arm"] == CANDIDATE_ARM)]) for f in folds]
    shock_set = frozenset(int(f) for f in shock_folds) if shock_folds is not None else frozenset()

    fig, ax = plt.subplots(figsize=(10, 5))
    for f in folds:
        if f in shock_set:
            ax.axvspan(f - 0.4, f + 0.4, color="#f1c40f", alpha=0.15)
    ax.plot(folds, r0_cpl, "o-", color="#888", label=BASELINE_ARM)
    ax.plot(folds, cand_cpl, "o-", color="#c0392b", label=CANDIDATE_ARM)
    ax.set_xlabel("fold")
    ax.set_ylabel("pooled realised CPL (own τ)")
    title_suffix = "shaded = shock fold" if shock_folds is not None else "shock fold unknown — legacy run"
    ax.set_title(f"{name}: realised CPL by fold ({title_suffix})")
    ax.set_xticks(folds)
    ax.legend()
    fig.tight_layout()
    path = run_dir / "realised_cpl_by_fold.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_tau_sweep(run_dir: pathlib.Path, rowpreds: pd.DataFrame, name: str) -> str | None:
    fig, ax = plt.subplots(figsize=(8, 5))
    any_curve = False
    for run_name, color in ((BASELINE_ARM, "#888"), (CANDIDATE_ARM, "#c0392b")):
        sub = rowpreds[rowpreds["run"] == run_name]
        if sub.empty:
            continue
        label_nunique = sub.groupby(["fold", "station_code", "price_date"], observed=True)["label"].nunique()
        if (label_nunique > 1).any():
            # label is a property of (station_code, price_date), fixed at labelling time — not of
            # (run, seed). More than one distinct label per key means the ident_base rows for this
            # (fold, station_code, price_date) disagree, which "label=first" would silently paper
            # over and skew the tau curve. Warn loudly rather than plot a number nobody can trust.
            n_bad = int((label_nunique > 1).sum())
            print(
                f"[dossier_tables] {name}/{run_name}: {n_bad} (fold, station_code, price_date) "
                "keys have inconsistent labels across rows — tau-sweep plot uses an arbitrary one "
                "per key; treat this plot as suspect and investigate rowpreds.parquet.",
                flush=True,
            )
        g = sub.groupby(["fold", "station_code", "price_date"], observed=True).agg(
            proba=("proba", "mean"), label=("label", "first")
        )
        curve = pd.DataFrame(threshold_sweep(g["label"].to_numpy(), g["proba"].to_numpy()))
        ax.plot(curve["tau"], curve["expected_cents_per_row"], color=color, label=run_name)
        star = curve.loc[curve["expected_cents_per_row"].idxmax()]
        ax.scatter([star["tau"]], [star["expected_cents_per_row"]], color=color, zorder=5)
        ax.annotate(
            f"τ*={star['tau']:.2f}", (star["tau"], star["expected_cents_per_row"]),
            textcoords="offset points", xytext=(5, -10), color=color,
        )
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.set_xlabel("τ")
    ax.set_ylabel("proxy expected c/row (seed-pooled probas)")
    ax.set_title(f"{name}: τ-sweep over saved rowpreds")
    ax.legend()
    fig.tight_layout()
    path = run_dir / "tau_sweep.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_candidate_over_time(
    run_dir: pathlib.Path, rowpreds: pd.DataFrame, cand_cols: list[str], name: str,
) -> str | None:
    if not cand_cols:
        return None
    df = rowpreds[rowpreds["run"] == CANDIDATE_ARM].drop_duplicates(subset=["station_code", "price_date"])
    if df.empty:
        return None
    fig, axes = plt.subplots(len(cand_cols), 1, figsize=(11, 3.5 * len(cand_cols)), squeeze=False)
    for i, col in enumerate(cand_cols):
        ax = axes[i][0]
        daily = df.groupby("price_date").agg(
            mean_val=(col, "mean"), nan_rate=(col, lambda s: s.isna().mean())
        ).reset_index()
        ax.plot(daily["price_date"], daily["mean_val"], color="#2980b9")
        ax2 = ax.twinx()
        ax2.fill_between(daily["price_date"], daily["nan_rate"], color="#c0392b", alpha=0.25, step="mid")
        ax2.set_ylim(0, 1)
        ax.set_ylabel(f"{col}\n(daily mean)", color="#2980b9")
        ax2.set_ylabel("NaN rate", color="#c0392b")
        ax.set_title(f"{col} over time — red band is the NaN rate")
    axes[-1][0].set_xlabel("price_date")
    fig.tight_layout()
    path = run_dir / "candidate_over_time.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_axis_breakdown(run_dir: pathlib.Path, facts: dict, name: str) -> str | None:
    """Trigger: candidate declared add_axis (facts["breakdowns"]["per_axis"] is non-empty)."""
    per_axis = facts["breakdowns"]["per_axis"] if facts.get("breakdowns") else None
    if not per_axis:
        return None
    df = pd.DataFrame(per_axis)
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(df)), 5))
    # `d` is None for a suppressed cell (by _cell_delta's construction) but can still be a real
    # NaN for a non-suppressed one (pooled_cpl on a zero-litre arm) — `(d or 0) > 0` treats NaN as
    # falsy-via-or here (NaN is truthy in Python, so `NaN or 0` returns NaN, and `NaN > 0` is
    # False), silently rendering a degenerate cell as a confident "candidate better" blue bar.
    # pd.isna covers both None and NaN explicitly, rendered the same grey as a suppressed cell.
    colours = [
        "#bdc3c7" if s or pd.isna(d) else ("#c0392b" if d > 0 else "#2980b9")
        for s, d in zip(df["suppressed"], df["delta_cpl_own"])
    ]
    heights = [0.0 if pd.isna(v) else v for v in df["delta_cpl_own"]]
    ax.bar(df["axis"].astype(str), heights, color=colours, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6)
    for i, (n, suppressed, d) in enumerate(zip(df["n_fills"], df["suppressed"], df["delta_cpl_own"])):
        if suppressed:
            tag = " (suppressed)"
        elif pd.isna(d):
            tag = " (degenerate: zero-litre cell)"
        else:
            tag = ""
        ax.annotate(
            f"n={n}{tag}", (i, 0), textcoords="offset points", xytext=(0, 4), ha="center",
            fontsize=8, rotation=90 if len(df) > 8 else 0,
        )
    ax.set_ylabel("Δ realised CPL, own τ (positive = candidate worse)")
    ax.set_title(f"{name}: per-axis delta (grey = below min-cell-n)")
    fig.tight_layout()
    path = run_dir / "axis_breakdown.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_cycle_phase(
    run_dir: pathlib.Path, rowpreds: pd.DataFrame, facts: dict, cand_cols: list[str], name: str,
) -> str | None:
    """Trigger: candidate's INPUTS/COLUMNS touch cycle phase (CYCLE_PHASE_COLUMNS)."""
    declared = set(facts["candidate"].get("inputs") or []) | set(facts["candidate"].get("columns") or [])
    if not (declared & CYCLE_PHASE_COLUMNS):
        return None
    if "cycle_pct_through" not in rowpreds.columns or not cand_cols:
        return None
    df = rowpreds[rowpreds["run"] == CANDIDATE_ARM].drop_duplicates(subset=["station_code", "price_date"]).copy()
    if df.empty:
        return None
    df["regime"] = df["cycle_pct_through"].map(assign_regime)
    colours = {"normal": "#2980b9", "late_descent": "#e67e22", "overdue": "#c0392b", "unmatched": "#888"}
    # One subplot per declared candidate column — a multi-column candidate must not silently lose
    # every column past the first (candidate_over_time.png already does this correctly; this plot
    # used to just pick cand_cols[0]).
    fig, axes = plt.subplots(len(cand_cols), 1, figsize=(9, 4.5 * len(cand_cols)), squeeze=False)
    for i, col in enumerate(cand_cols):
        ax = axes[i][0]
        for regime, sub in df.groupby("regime", observed=True):
            ax.scatter(
                sub["cycle_pct_through"], sub[col], s=6, alpha=0.4,
                color=colours.get(regime, "#888"), label=regime,
            )
        ax.set_ylabel(col)
        ax.set_title(f"{col} vs cycle_pct_through, by regime")
        ax.legend(markerscale=3)
    axes[-1][0].set_xlabel("cycle_pct_through")
    fig.suptitle(f"{name}: candidate columns vs cycle phase")
    fig.tight_layout()
    path = run_dir / "cycle_phase_breakdown.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _parse_price_date(s: pd.Series) -> pd.Series:
    """Normalise a raw features-frame price_date column to datetime64, handling both on-disk
    forms this codebase produces (batch_freeze.py's `_snapshot_date` documents the same split):
    INTEGER YYYYMMDD, or an ISO date string already parseable by pd.to_datetime.
    """
    if pd.api.types.is_integer_dtype(s):
        return pd.to_datetime(s.map(date_from_int))
    return pd.to_datetime(s)


def _plot_external_overlay(run_dir: pathlib.Path, facts: dict, batch_dir: pathlib.Path | None, name: str) -> str | None:
    """Trigger: an INPUTS entry isn't part of the batch's trained-model baseline contract —
    a proxy for "external/not-yet-in-the-model series" (matches the TGP precedent: tgp_delta_7d's
    own inputs read a TGP-derived column that's deliberately held out of baseline_columns.json).

    Plots the raw input series over time from the batch's frozen features frame — not the
    candidate's own derived column (recomputing add_columns needs the candidate module, which
    isn't an artifact this module reads).
    """
    if batch_dir is None:
        return None
    baseline_path = batch_dir / BASELINE_COLUMNS_FILENAME
    features_path = batch_dir / FROZEN_FEATURES_FILENAME
    if not baseline_path.exists() or not features_path.exists():
        return None
    baseline_cols = set(json.loads(baseline_path.read_text()))
    inputs = facts["candidate"].get("inputs") or []
    external = [c for c in inputs if c != "price_date" and c not in baseline_cols]
    if not external:
        return None
    frame = pd.read_parquet(features_path)
    external = [c for c in external if c in frame.columns]
    if not external:
        return None
    # Unlike rowpreds.parquet's price_date (normalised to datetime64 by the runner before
    # writing), this is the batch's raw frozen features.parquet, read the same way
    # fuel_signal.features.load_features() reads its parquet path — no date parsing applied.
    # price_date there is INTEGER YYYYMMDD or an ISO string depending on how it was produced
    # (batch_freeze.py's own _snapshot_date documents both forms) — plotted directly, either
    # renders as an unreadable smear (a string axis is categorical: one x-tick per unique value).
    frame = frame.assign(price_date=_parse_price_date(frame["price_date"]))
    daily = frame.groupby("price_date")[external].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in external:
        ax.plot(daily["price_date"], daily[col], label=col)
    ax.set_xlabel("price_date")
    ax.set_title(f"{name}: declared external input(s) over time (frozen batch frame)")
    ax.legend()
    fig.tight_layout()
    path = run_dir / "external_series_overlay.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


# ── orchestration / CLI ─────────────────────────────────────────────────────

def process_run(run_dir: pathlib.Path) -> pathlib.Path:
    """Build facts.json + plots for one run directory. Returns the facts.json path."""
    run_dir = pathlib.Path(run_dir)
    facts = build_facts(run_dir)
    batch_dir_str = facts["provenance"].get("batch_dir")
    batch_dir = pathlib.Path(batch_dir_str) if batch_dir_str else None
    facts["plots"] = make_plots(run_dir, facts, batch_dir)
    facts_path = run_dir / FACTS_FILENAME
    # to_jsonable, not raw json.dumps: a noise-floor draw of n=1 (band_std=nan) or a zero-litre
    # fill cell (pooled_cpl=nan) both reach a facts.json field with no other guard, and Python's
    # json.dumps happily emits the literal (invalid, non-RFC-8259) `NaN` token by default. This
    # file is the sole input to the downstream Claude session and to any external reader (jq
    # included) — the same reason runner.py's results.json already goes through this helper.
    facts_path.write_text(json.dumps(to_jsonable(facts), indent=2, default=str))
    return facts_path


@click.command("dossier_tables")
@click.argument("target", required=False)
@click.option(
    "--scan", "scan_root", default=None,
    help="Process every pending run under this root instead of one run_dir.",
)
def main(target: str | None, scan_root: str | None) -> None:
    """Build facts.json + PNGs for a completed run, or every pending run under --scan.

    Stale-claim recovery is NOT this module's job — launch.py's own `recover_stale_claims`
    (bd `in_progress` + a required traceback in run.log) already runs at the start of every
    nightly launch cycle. See the module docstring.
    """
    if scan_root:
        root = pathlib.Path(scan_root)
        for run_dir in find_pending_runs(root):
            click.echo(f"[dossier_tables] processing {run_dir}")
            try:
                process_run(run_dir)
            except Exception as exc:  # noqa: BLE001 — one malformed/partial run must not stop the queue
                # A run killed mid-write (e.g. between results.json and rowpreds.parquet landing)
                # is exactly the unattended-pipeline case this scan runs unsupervised overnight —
                # every other pending run in the batch must still get its dossier.
                click.echo(f"[dossier_tables] {run_dir}: failed, skipping — {exc!r}", err=True)
        return
    if not target:
        raise click.UsageError("pass a run_dir, or --scan <root>")
    facts_path = process_run(pathlib.Path(target))
    click.echo(f"Wrote {facts_path}")


if __name__ == "__main__":
    main()
