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
on disk, so the work queue (any dir with results.json and no README.md) stays correct and
re-runnable; and it keeps the overnight Claude burst short.

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

import json
import pathlib

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — backend must be set before this import
import numpy as np
import pandas as pd

from experiments.lib.constants import ROW_AXIS_ECONOMICS_CAVEAT, SHOCK_FOLDS
from experiments.lib.io import current_git_sha, to_jsonable
from experiments.lib.zones import assign_regime, pooled_cpl
from experiments.pipeline.runner import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    DEFAULT_MIN_ROW_CELL_N,
    RETRYABLE_STATUSES,
    read_run_status,
)
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
FROZEN_FEATURES_FILENAME = "features.parquet"

SEED_STD_FLAG_RATIO = 5.0

# Row-level cycle-phase columns (experiments/lib/zones.py assign_regime() operates on the first).
# "candidate touches cycle phase" (Plots spec, fps-3jj.6) triggers off a candidate's declared
# INPUTS/COLUMNS intersecting this set, not any cycle_* column (cycle_mean_length etc. are
# magnitude, not phase).
CYCLE_PHASE_COLUMNS = {"cycle_pct_through", "cycle_days_since_peak"}


# ── work queue ────────────────────────────────────────────────────────────────

def find_pending_runs(root: pathlib.Path) -> list[pathlib.Path]:
    """Any directory under `root` with results.json, no README.md, and a non-retryable status.

    Matches the parent design's "work queue is self-describing" rule. A run still in progress
    has neither file and is silently skipped.

    The RETRYABLE_STATUSES exclusion is load-bearing, not tidiness (fps-g31). A run dir is
    keyed on the candidate, so a re-run REUSES it. Without the exclusion:

      night 1  candidate aborts (aborted_pipeline); results.json written
               this scan sees results.json + no README -> writes facts.json, session writes README
      night 2  launch releases the claim, candidate re-runs and SUCCEEDS, overwriting results.json
      night 2  this scan sees the README from night 1 -> run is not pending -> never dossiered

    The successful run would be silently invisible forever. That failure mode only became
    reachable when aborts started going back on the queue, which is why the guard lives here
    rather than in the caller: anything that reuses a run dir has to agree on what "finished"
    means, so both sides read runner.read_run_status.
    """
    root = pathlib.Path(root)
    return sorted({
        p.parent
        for p in root.rglob(RESULTS_FILENAME)
        if not (p.parent / README_FILENAME).exists()
        and read_run_status(p.parent) not in RETRYABLE_STATUSES
    })


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
    if status == "rejected":
        for filename in (ROWPREDS_FILENAME, FILLS_FILENAME):
            if not (run_dir / filename).exists():
                # status=="rejected" means runner.py's success path ran, which writes both
                # parquet artifacts BEFORE results.json (see runner.py's "write artifacts before
                # grading" comment) — so this means something removed them after the fact (a
                # manual `git clean`, disk issue, etc.), not a normal in-progress race. Fail with
                # a message that says so, rather than a bare pyarrow FileNotFoundError the --scan
                # loop just logs.
                raise FileNotFoundError(
                    f"{run_dir}: status=='rejected' but {filename} is missing — "
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
            "target": candidate.get("target"),
        },
        "provenance": _provenance(results, batch_dir),
        "headline": None,
        "breakdowns": None,
        "seed_flags": results.get("seed_variance", {}).get("flags", []),
        "validation": _validation(results, rowpreds),
        "grading": {
            "predicted_signature": candidate.get("predicted_signature"),
            "verdict": None,
            "explanation": None,
            "pending": True,
        },
        "noise_band": _noise_band(results, batch_dir),
    }

    if status != "rejected":
        facts["status_note"] = (
            f"status={status!r} — run did not reach the scoring stages; headline/breakdowns are "
            f"unavailable. error: {results.get('error')}"
        )
        return facts

    facts["headline"] = _headline(results)
    facts["breakdowns"] = _breakdowns(results, rowpreds, fills)
    return facts


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
        "wall_seconds": meta.get("wall_seconds"),
        "status": results.get("status"),
        "bead_id": meta.get("bead_id"),
        # Baseline identity (fps-zci item 5). None for runs that predate the field —
        # which is exactly the population where a wrong R0 could be hiding, so the
        # absence is itself the signal, not a formatting inconvenience.
        "n_baseline_columns": meta.get("n_baseline_columns"),
        "baseline_fingerprint": meta.get("baseline_fingerprint"),
    }


def _validation(results: dict, rowpreds: pd.DataFrame | None) -> dict:
    status = results["status"]
    if status == "rejected":
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

    return {"pit_test": pit_test, "inputs_check": inputs_check, "candidate_column_nan_rate": nan_rates}


def _noise_band(results: dict, batch_dir: pathlib.Path | None, *, check_fingerprint: bool = True) -> dict:
    """`check_fingerprint=False` is for retrospective.py's `_batch_noise_summary`,
    which reuses this function's math to summarise a batch's floor on its own
    (mean/std/n_draws) rather than to grade any one candidate run against it — it
    calls with a dummy `results` that has no real `meta.baseline_fingerprint` to
    compare, so the per-run identity check below would always (mis)fire as a
    mismatch. `build_facts()`, which DOES grade a real run, always uses the default.
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
    floor_fingerprint = noise_floor.get("baseline_fingerprint")
    run_fingerprint = results.get("meta", {}).get("baseline_fingerprint")
    if check_fingerprint and (floor_fingerprint is None or floor_fingerprint != run_fingerprint):
        # fps-cf8: a floor computed against one baseline could silently grade a
        # candidate run against a different one (fps-sa1, fps-zci — exactly the
        # failure class feedback_baseline_declared_not_discovered warns about). Refuse
        # rather than trust "file exists" the way this function used to. A floor with
        # no fingerprint at all (pre-fps-cf8) is treated as a mismatch, not a pass —
        # it cannot be shown to match, so it cannot be trusted either.
        return {
            "available": False,
            "reason": f"noise_floor.json's baseline_fingerprint ({floor_fingerprint!r}) does not "
            f"match this run's ({run_fingerprint!r}) — the floor was computed against a "
            "different baseline (or predates baseline_fingerprint entirely) and does not grade "
            "this run. Recompute with `PYTHONPATH=. uv run python -m experiments.pipeline."
            "noise_floor <batch> --force`.",
        }
    if noise_floor.get("partial"):
        # noise_floor.py's --fold-subset is an iteration/smoke speed-up: the deltas only cover
        # some outer folds, but effect_delta_cpl_held always pools every fold. Grading a
        # candidate against a partial-fold floor as though it were the real one would silently
        # misjudge the delta, not just weaken the estimate — refuse rather than mislabel it.
        return {
            "available": False,
            "reason": "noise_floor.json was computed with --fold-subset (partial fold "
            "coverage) — not a valid ruler against a delta pooled over every fold. Recompute "
            "with `PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> "
            "--force` and no --fold-subset.",
        }
    deltas = np.asarray(noise_floor.get("deltas_cpl_held", []), dtype=float)
    delta = results.get("effect_delta_cpl_held")
    if delta is None or deltas.size == 0:
        return {"available": False, "reason": "empty noise-floor sample or unresolved effect_delta_cpl_held"}
    band_mean = float(np.mean(deltas))
    band_std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan")
    return {
        "available": True,
        "n_draws": int(deltas.size),
        "band_mean_delta_cpl_held": band_mean,
        "band_std_delta_cpl_held": band_std,
        "candidate_delta_cpl_held": delta,
        # delta_cpl_held is a COST: more negative = better. "Better than N% of noise" therefore
        # means N% of the noise-floor draws were WORSE (higher/less-negative) than this candidate
        # — (deltas > delta), not (deltas < delta). A candidate with a strong negative delta
        # against a noise band centred near zero must read as a HIGH percentile here, not near 0.
        "candidate_percentile_better_than_noise": float((deltas > delta).mean() * 100),
        "candidate_z_vs_band": (delta - band_mean) / band_std if np.isfinite(band_std) and band_std > 0 else None,
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

    per_fold = []
    for fold in sorted(fills["fold"].unique()):
        delta, n, suppressed = _cell_delta(fills, fills["fold"] == fold, min_row_cell_n=min_row_cell_n)
        row = {
            "fold": int(fold),
            "regime": "shock" if fold in SHOCK_FOLDS else "normal",
            "n_fills": n,
            "suppressed": suppressed,
            "delta_cpl_own": delta,
        }
        if fold in cand_ll.index:
            row["delta_ll_all_median"] = float(cand_ll.loc[fold, "delta_ll_all_median"])
            row["delta_ll_hard25_median"] = float(cand_ll.loc[fold, "delta_ll_hard25_median"])
        per_fold.append(row)

    all_folds = set(fills["fold"].unique())
    per_regime = []
    for regime_name, folds in (("shock", SHOCK_FOLDS & all_folds), ("normal", all_folds - SHOCK_FOLDS)):
        delta, n, suppressed = _cell_delta(fills, fills["fold"].isin(folds), min_row_cell_n=min_row_cell_n)
        per_regime.append({"regime": regime_name, "n_fills": n, "suppressed": suppressed, "delta_cpl_own": delta})

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
        lambda: _plot_realised_cpl_by_fold(run_dir, fills, name),
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


def _plot_realised_cpl_by_fold(run_dir: pathlib.Path, fills: pd.DataFrame, name: str) -> str | None:
    folds = sorted(fills["fold"].unique())
    if not folds:
        return None
    r0_cpl = [pooled_cpl(fills[(fills["fold"] == f) & (fills["arm"] == BASELINE_ARM)]) for f in folds]
    cand_cpl = [pooled_cpl(fills[(fills["fold"] == f) & (fills["arm"] == CANDIDATE_ARM)]) for f in folds]

    fig, ax = plt.subplots(figsize=(10, 5))
    for f in folds:
        if f in SHOCK_FOLDS:
            ax.axvspan(f - 0.4, f + 0.4, color="#f1c40f", alpha=0.15)
    ax.plot(folds, r0_cpl, "o-", color="#888", label=BASELINE_ARM)
    ax.plot(folds, cand_cpl, "o-", color="#c0392b", label=CANDIDATE_ARM)
    ax.set_xlabel("fold")
    ax.set_ylabel("pooled realised CPL (own τ)")
    ax.set_title(f"{name}: realised CPL by fold (shaded = shock fold)")
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
