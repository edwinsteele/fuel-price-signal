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

Two artifacts this module reads that don't exist yet, both intentional forward-compatible slots:

  claim.json (`CLAIM_FILENAME`) — contract the launch routine (fps-3jj.5, in progress) must
    satisfy: write `{"bead_id": ..., "candidate": ..., "started_at": <ISO8601 UTC>}` into the run
    directory the moment it claims a candidate bead, before launching the runner detached. This
    module never writes it, only reads it, to detect and recover a crashed launch (parent design's
    "dir exists, run.log ends in a traceback, no results.json, started >12h ago" case).

  noise_floor.json (`NOISE_FLOOR_FILENAME`) — contract fps-3jj.9 (P3, not yet built) must satisfy:
    write `{"deltas_cpl_held": [<float>, ...]}` into the batch dir at batch-setup time (10-20
    uninformative draws through the same fit + realised-backtest path). Until that file exists,
    facts["noise_band"]["available"] is False — the slot exists now so the schema doesn't change
    once fps-3jj.9 lands.

Usage:
  PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables <run_dir>       # one run
  PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables --scan <root>   # whole queue
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from datetime import datetime, timedelta, timezone

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — backend must be set before this import
import numpy as np
import pandas as pd

from experiments.lib.constants import SHOCK_FOLDS
from experiments.lib.zones import assign_regime, pooled_cpl
from experiments.pipeline.runner import BASELINE_ARM, CANDIDATE_ARM, DEFAULT_MIN_ROW_CELL_N
from fuel_signal.score_phase2 import threshold_sweep

RESULTS_FILENAME = "results.json"
FACTS_FILENAME = "facts.json"
README_FILENAME = "README.md"
ROWPREDS_FILENAME = "rowpreds.parquet"
FILLS_FILENAME = "fills.parquet"
RUN_LOG_FILENAME = "run.log"
CLAIM_FILENAME = "claim.json"
NOISE_FLOOR_FILENAME = "noise_floor.json"
BASELINE_COLUMNS_FILENAME = "baseline_columns.json"
FREEZE_MANIFEST_FILENAME = "freeze.json"
FROZEN_FEATURES_FILENAME = "features.parquet"

STALE_AFTER_HOURS = 12
SEED_STD_FLAG_RATIO = 5.0

# rowpreds columns that are never a persisted candidate/context column — everything else on a
# rowpreds row is either an identity column baked in by the runner or a value fps-3jj.6 asked the
# runner to persist (candidate.COLUMNS, cycle_pct_through). See runner.py's
# `_run_wfcv_screen(..., persist_columns=...)`.
_IDENT_COLUMNS = {"fold", "station_code", "price_date", "label", "is_hard25", "axis", "run", "seed", "proba"}

# Row-level cycle-phase columns (experiments/lib/zones.py assign_regime() operates on the first).
# "candidate touches cycle phase" (Plots spec, fps-3jj.6) triggers off a candidate's declared
# INPUTS/COLUMNS intersecting this set, not any cycle_* column (cycle_mean_length etc. are
# magnitude, not phase).
CYCLE_PHASE_COLUMNS = {"cycle_pct_through", "cycle_days_since_peak"}


# ── work queue ────────────────────────────────────────────────────────────────

def find_pending_runs(root: pathlib.Path) -> list[pathlib.Path]:
    """Any directory under `root` with results.json and no README.md yet.

    Matches the parent design's "work queue is self-describing" rule exactly. A run still in
    progress has neither file and is silently skipped.
    """
    root = pathlib.Path(root)
    return sorted(
        {p.parent for p in root.rglob(RESULTS_FILENAME) if not (p.parent / README_FILENAME).exists()}
    )


def find_stale_claims(
    root: pathlib.Path, *, stale_after_hours: float = STALE_AFTER_HOURS, now: datetime | None = None
) -> list[dict]:
    """Directories with a claim.json but no results.json, past the crash-recovery window.

    See the module docstring's `claim.json` contract. A run still legitimately in flight (started
    recently) is not stale; this only flags a claim old enough that a normal claim-to-results cycle
    should have finished.
    """
    root = pathlib.Path(root)
    now = now or datetime.now(timezone.utc)
    stale: list[dict] = []
    for claim_path in root.rglob(CLAIM_FILENAME):
        run_dir = claim_path.parent
        if (run_dir / RESULTS_FILENAME).exists():
            continue
        try:
            claim = json.loads(claim_path.read_text())
            started_at = datetime.fromisoformat(claim["started_at"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        age = now - started_at
        if age < timedelta(hours=stale_after_hours):
            continue
        run_log = run_dir / RUN_LOG_FILENAME
        traceback_tail = None
        if run_log.exists():
            lines = run_log.read_text(errors="replace").strip().splitlines()
            traceback_tail = "\n".join(lines[-40:]) if lines else None
        stale.append({
            "run_dir": run_dir,
            "bead_id": claim.get("bead_id"),
            "candidate": claim.get("candidate"),
            "started_at": claim["started_at"],
            "age_hours": age.total_seconds() / 3600,
            "traceback_tail": traceback_tail,
        })
    return stale


def release_stale_claim(entry: dict, *, dry_run: bool = False) -> None:
    """bd-side half of stale-claim recovery: post the traceback tail, release the claim.

    Deterministic bookkeeping (no judgement about *why* it failed), so it belongs here rather
    than in the Claude step, same reasoning as facts.json itself.
    """
    bead_id = entry.get("bead_id")
    if not bead_id:
        print(f"[dossier_tables] {entry['run_dir']}: claim.json has no bead_id, skipping bd release", flush=True)
        return
    body = (
        f"Tail of {RUN_LOG_FILENAME}:\n```\n{entry['traceback_tail']}\n```"
        if entry.get("traceback_tail")
        else f"{RUN_LOG_FILENAME} missing or empty."
    )
    message = (
        f"[dossier] stale claim released — no {RESULTS_FILENAME} after "
        f"{entry['age_hours']:.1f}h (started {entry['started_at']}).\n\n{body}"
    )
    if dry_run:
        print(f"[dossier_tables] DRY RUN — would release {bead_id}:\n{message}", flush=True)
        return
    try:
        subprocess.run(["bd", "comment", bead_id, "--stdin"], input=message, text=True, check=True)
        subprocess.run(["bd", "assign", bead_id, ""], check=True)
        subprocess.run(["bd", "update", bead_id, "--status", "open"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[dossier_tables] releasing {bead_id} failed, continuing: {exc}", flush=True)


# ── facts.json ────────────────────────────────────────────────────────────────

def build_facts(run_dir: pathlib.Path) -> dict:
    """Read one run's artifacts and return the facts.json dict (does not write it)."""
    run_dir = pathlib.Path(run_dir)
    results = json.loads((run_dir / RESULTS_FILENAME).read_text())
    status = results["status"]
    meta = results.get("meta", {})
    candidate = results.get("candidate", {})
    batch_dir = pathlib.Path(meta["batch_dir"]) if meta.get("batch_dir") else None

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
        "validation": _validation(results, run_dir),
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

    rowpreds = pd.read_parquet(run_dir / ROWPREDS_FILENAME)
    fills = pd.read_parquet(run_dir / FILLS_FILENAME)
    facts["headline"] = _headline(results)
    facts["breakdowns"] = _breakdowns(results, rowpreds, fills)
    return facts


def _current_git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


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
        git_sha = _current_git_sha()
    return {
        "candidate": results.get("candidate", {}).get("name"),
        "batch": batch_name,
        "snapshot_date": snapshot_date,
        "git_sha": git_sha,
        "git_sha_is_run_time": git_sha_is_run_time,
        "seeds": meta.get("seeds"),
        "wall_seconds": meta.get("wall_seconds"),
        "status": results.get("status"),
        "bead_id": meta.get("bead_id"),
    }


def _validation(results: dict, run_dir: pathlib.Path) -> dict:
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
    rowpreds_path = run_dir / ROWPREDS_FILENAME
    if status == "rejected" and rowpreds_path.exists():
        rp = pd.read_parquet(rowpreds_path)
        candidate_cols = [c for c in rp.columns if c not in _IDENT_COLUMNS and c != "cycle_pct_through"]
        if candidate_cols:
            nan_rates = {c: float(rp[c].isna().mean()) for c in candidate_cols}

    return {"pit_test": pit_test, "inputs_check": inputs_check, "candidate_column_nan_rate": nan_rates}


def _noise_band(results: dict, batch_dir: pathlib.Path | None) -> dict:
    if batch_dir is None:
        return {"available": False, "reason": "no batch_dir recorded in results.json meta"}
    path = batch_dir / NOISE_FLOOR_FILENAME
    if not path.exists():
        return {"available": False, "reason": "fps-3jj.9 not yet landed — no noise-floor calibration for this batch"}
    deltas = np.asarray(json.loads(path.read_text()).get("deltas_cpl_held", []), dtype=float)
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
        "candidate_percentile_in_band": float((deltas < delta).mean() * 100),
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
    if "axis" in rowpreds.columns:
        axis_lookup = (
            rowpreds[["station_code", "price_date", "axis"]]
            .drop_duplicates()
            .assign(date=lambda d: pd.to_datetime(d["price_date"]).dt.strftime("%Y-%m-%d"))
            [["station_code", "date", "axis"]]
        )
        tagged = fills.merge(axis_lookup, on=["station_code", "date"], how="left")
        per_axis = []
        for axis_value in sorted(tagged["axis"].dropna().unique(), key=str):
            delta, n, suppressed = _cell_delta(tagged, tagged["axis"] == axis_value, min_row_cell_n=min_row_cell_n)
            per_axis.append({"axis": axis_value, "n_fills": n, "suppressed": suppressed, "delta_cpl_own": delta})

    return {"per_fold": per_fold, "per_regime": per_regime, "per_axis": per_axis, "min_row_cell_n": min_row_cell_n}


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

    written = []
    for fn in (
        lambda: _plot_per_fold_delta_bars(run_dir, facts, name),
        lambda: _plot_seed_mean_vs_median(run_dir, rowpreds, name),
        lambda: _plot_realised_cpl_by_fold(run_dir, fills, name),
        lambda: _plot_tau_sweep(run_dir, rowpreds, name),
        lambda: _plot_candidate_over_time(run_dir, rowpreds, name),
        lambda: _plot_axis_breakdown(run_dir, facts, name),
        lambda: _plot_cycle_phase(run_dir, rowpreds, facts, name),
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
    colours = ["#c0392b" if v > 0 else "#2980b9" for v in df["delta_ll_all_median"]]
    edgecolours = ["#f1c40f" if r == "shock" else "black" for r in df["regime"]]
    ax.bar(df["fold"], df["delta_ll_all_median"], color=colours, edgecolor=edgecolours, linewidth=2)
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


def _plot_candidate_over_time(run_dir: pathlib.Path, rowpreds: pd.DataFrame, name: str) -> str | None:
    cand_cols = [c for c in rowpreds.columns if c not in _IDENT_COLUMNS and c != "cycle_pct_through"]
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
    colours = [
        "#bdc3c7" if s else ("#c0392b" if (d or 0) > 0 else "#2980b9")
        for s, d in zip(df["suppressed"], df["delta_cpl_own"])
    ]
    heights = [0.0 if v is None else v for v in df["delta_cpl_own"]]
    ax.bar(df["axis"].astype(str), heights, color=colours, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6)
    for i, (n, suppressed) in enumerate(zip(df["n_fills"], df["suppressed"])):
        label = f"n={n}" + (" (suppressed)" if suppressed else "")
        ax.annotate(
            label, (i, 0), textcoords="offset points", xytext=(0, 4), ha="center",
            fontsize=8, rotation=90 if len(df) > 8 else 0,
        )
    ax.set_ylabel("Δ realised CPL, own τ (positive = candidate worse)")
    ax.set_title(f"{name}: per-axis delta (grey = below min-cell-n)")
    fig.tight_layout()
    path = run_dir / "axis_breakdown.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_cycle_phase(run_dir: pathlib.Path, rowpreds: pd.DataFrame, facts: dict, name: str) -> str | None:
    """Trigger: candidate's INPUTS/COLUMNS touch cycle phase (CYCLE_PHASE_COLUMNS)."""
    declared = set(facts["candidate"].get("inputs") or []) | set(facts["candidate"].get("columns") or [])
    if not (declared & CYCLE_PHASE_COLUMNS):
        return None
    if "cycle_pct_through" not in rowpreds.columns:
        return None
    cand_cols = [c for c in rowpreds.columns if c not in _IDENT_COLUMNS and c != "cycle_pct_through"]
    if not cand_cols:
        return None
    df = rowpreds[rowpreds["run"] == CANDIDATE_ARM].drop_duplicates(subset=["station_code", "price_date"]).copy()
    if df.empty:
        return None
    df["regime"] = df["cycle_pct_through"].map(assign_regime)
    col = cand_cols[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {"normal": "#2980b9", "late_descent": "#e67e22", "overdue": "#c0392b", "unmatched": "#888"}
    for regime, sub in df.groupby("regime", observed=True):
        ax.scatter(sub["cycle_pct_through"], sub[col], s=6, alpha=0.4, color=colours.get(regime, "#888"), label=regime)
    ax.set_xlabel("cycle_pct_through")
    ax.set_ylabel(col)
    ax.set_title(f"{name}: {col} vs cycle_pct_through, by regime")
    ax.legend(markerscale=3)
    fig.tight_layout()
    path = run_dir / "cycle_phase_breakdown.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


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
    results = json.loads((run_dir / RESULTS_FILENAME).read_text())
    batch_dir = pathlib.Path(results["meta"]["batch_dir"]) if results.get("meta", {}).get("batch_dir") else None
    facts["plots"] = make_plots(run_dir, facts, batch_dir)
    facts_path = run_dir / FACTS_FILENAME
    facts_path.write_text(json.dumps(facts, indent=2, default=str))
    return facts_path


@click.command("dossier_tables")
@click.argument("target", required=False)
@click.option(
    "--scan", "scan_root", default=None,
    help="Process every pending run under this root instead of one run_dir.",
)
@click.option("--release-stale/--no-release-stale", default=True, help="Also recover stale claims found under --scan.")
def main(target: str | None, scan_root: str | None, release_stale: bool) -> None:
    """Build facts.json + PNGs for a completed run, or every pending run under --scan."""
    if scan_root:
        root = pathlib.Path(scan_root)
        if release_stale:
            for entry in find_stale_claims(root):
                release_stale_claim(entry)
        for run_dir in find_pending_runs(root):
            click.echo(f"[dossier_tables] processing {run_dir}")
            process_run(run_dir)
        return
    if not target:
        raise click.UsageError("pass a run_dir, or --scan <root>")
    facts_path = process_run(pathlib.Path(target))
    click.echo(f"Wrote {facts_path}")


if __name__ == "__main__":
    main()
