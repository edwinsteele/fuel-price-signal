"""Batch retrospective (fps-3jj.8) — the payoff artifact for aim (b) (gaining experience with
an AI-sourced pipeline, not turnaround time).

Same code/Claude split as `dossier_tables.py` (fps-3jj.6): this module does no prose and no
judgement. It reads every candidate's own `facts.json` (already written by dossier_tables.py —
nothing here recomputes a noise-band percentile or a headline delta, it reads what's already on
disk) plus the batch's `noise_floor.json`, and writes one deterministic `retrospective_facts.json`
into the batch dir. The Claude step (`docs/routines/retrospective.md`) reads that file and writes
`RETROSPECTIVE.md` prose, same division of labour as the dossier routine.

Four things this module produces, per the fps-3jj.8 bead body:

  1. Leaderboard — every candidate filed against the batch, ranked by how it fared. Only
     dossiered candidates (those with a facts.json) get a rank; everything else falls into the
     outcome tally instead (see below) rather than being silently dropped from the batch.

  2. Noise-band comparison, with the multiple-comparisons correction the bead explicitly calls
     out: "the ranking step is exactly where a noise delta gets promoted to a finding — max-of-N
     is a multiple-comparisons operation the old one-at-a-time workflow never performed." A
     single candidate's raw percentile against the noise band (already computed per-candidate by
     dossier_tables._noise_band) is a fair one-shot test; picking the BEST of N candidates against
     that same band is not, and needs a higher bar. `family_wise_percentile_threshold` reports it
     (a plain Bonferroni correction on the single-draw 95th-percentile threshold), and
     `clears_family_wise_threshold` on each leaderboard row uses it, not the raw 95th percentile.

  3. Outcome-code tally — every candidate MODULE filed against the batch
     (`experiments/candidates/<batch>/*.py`) is the universe, not just the ones that reached a
     dossier. A module with no results.json at all is `never_run` — missing data, not a rejection.
     A module with a results.json in a RETRYABLE status has not had a fair hearing yet either
     (still cycling through launch.py's retry budget, or exhausted and sitting `blocked` in bd —
     this module reads disk only, so it cannot tell those two apart; `docs/routines/retrospective.md`
     flags that as a known gap rather than guessing). Folding either into `rejected` would bias the
     calibration read below.

  4. Confidence calibration — did CONFIDENCE_EFFECT/CONFIDENCE_ZONE predict what actually
     happened? Per `docs/routines/generator.md`: "Five candidates cannot calibrate anything on
     their own... reads calibration across batches, not within one." So unlike the leaderboard and
     outcome tally (both scoped to the one batch this module was invoked on), calibration scans
     EVERY facts.json under `experiments/candidates/` — every batch ever dossiered — because the
     (confidence, outcome) pairs already live there; there is no separate append-only record to
     keep in sync (the single-sourcing lesson from fps-zci applies here too).

Usage (after a batch's candidates have all reached a terminal dossier — see
docs/routines/retrospective.md for when to invoke this):
  PYTHONPATH=. uv run python -m experiments.pipeline.retrospective <batch-name>
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import click
import numpy as np

from experiments.lib.io import current_git_sha, to_jsonable
from experiments.pipeline.batch_freeze import DEFAULT_BATCHES_DIR
from experiments.pipeline.dossier_tables import FACTS_FILENAME, _noise_band
from experiments.pipeline.runner import RETRYABLE_STATUSES, default_out_dir, read_run_status

DEFAULT_CANDIDATES_DIR = DEFAULT_BATCHES_DIR.parent / "candidates"
RETROSPECTIVE_FILENAME = "retrospective_facts.json"

# "Five candidates cannot calibrate anything on their own" (generator.md) — batch 2 alone is 5.
# Set above that so a single batch's confidence pairs never masquerade as a calibration read;
# the field only turns on once pairs have accumulated across multiple batches.
MIN_CALIBRATION_N = 10

FAMILY_WISE_ALPHA = 0.05


def find_batch_candidates(
    batch_name: str, candidates_root: pathlib.Path = DEFAULT_CANDIDATES_DIR
) -> list[pathlib.Path]:
    """Every candidate module filed against this batch — the universe the retrospective grades,
    not just the ones that reached a dossier. Mirrors generator.md's filing convention
    (`experiments/candidates/<batch>/<NAME>.py`, flat siblings)."""
    batch_dir = pathlib.Path(candidates_root) / batch_name
    if not batch_dir.is_dir():
        return []
    return sorted(p for p in batch_dir.glob("*.py") if p.stem != "__init__")


def _candidate_entry(candidate_path: pathlib.Path) -> dict:
    """One candidate's disk state: a dossiered entry (has facts.json), or one of the
    not-yet-dossiered states the outcome tally counts separately from a real verdict.

    Checks results.json's status BEFORE consulting facts.json, not after: run_candidate
    deletes results.json up front on every (re-)run but nothing ever deletes a previous
    facts.json (runner.py's own comment on why — an in-flight run must not look like a
    stale old verdict to a concurrent reader). A manually re-queued candidate that goes
    retryable again would otherwise have its stale facts.json read as "dossiered" for as
    long as the retry sits unresolved — checking status first means a retryable result
    always wins over a leftover facts.json from a previous attempt.
    """
    name = candidate_path.stem
    out_dir = default_out_dir(candidate_path)
    status = read_run_status(out_dir)
    if status in RETRYABLE_STATUSES:
        return {"candidate": name, "state": "retryable_incomplete", "facts": None}

    facts_path = out_dir / FACTS_FILENAME
    if facts_path.exists():
        try:
            facts = json.loads(facts_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            # Deliberately loud, unlike read_run_status's total read: this routine is
            # interactively invoked (docs/routines/retrospective.md), not an unattended
            # nightly sweep, so a corrupted facts.json is worth stopping for rather than
            # silently miscounting — but the raw exception doesn't name the candidate.
            raise ValueError(f"candidate {name!r}: could not read {facts_path}: {exc}") from exc
        return {"candidate": name, "state": "dossiered", "facts": facts}

    if status is None:
        return {"candidate": name, "state": "never_run", "facts": None}
    # Terminal status but no facts.json yet — the narrow race window between a run finishing
    # and the next dossier scan. Real, but should be rare and self-resolving.
    return {"candidate": name, "state": "pending_dossier", "facts": None}


def family_wise_percentile_threshold(n_candidates: int, alpha: float = FAMILY_WISE_ALPHA) -> float:
    """Bonferroni-corrected percentile a candidate's noise-band position must clear to be
    read as surprising at the BATCH level, not just individually.

    A single candidate's raw percentile against the noise band answers "how often would pure
    fit noise beat this candidate?" — a fair one-shot test. Asking the same question of the BEST
    of N candidates is a different question (order statistics): even under a null where every
    candidate is truly zero-effect, the max of N noisy draws is expected to look more extreme
    than any single draw. Bonferroni divides the family-wise error rate alpha by N candidates and
    reports the corresponding single-candidate percentile threshold — standard, conservative, and
    exact for n_candidates=1 (reduces to the ordinary 1-alpha percentile).
    """
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    return 100.0 * (1.0 - alpha / n_candidates)


def _batch_noise_summary(batch_dir: pathlib.Path) -> dict:
    """Batch-level noise-band summary (mean/std/n_draws), reusing dossier_tables._noise_band's
    own math rather than re-deriving it — single-sourced the same way BASELINE_COLUMNS is
    (fps-zci). The dummy delta below only feeds the per-candidate percentile/z fields, which this
    function discards; mean/std/n_draws/available/reason don't depend on it.
    """
    band = _noise_band({"effect_delta_cpl_held": 0.0}, batch_dir)
    if not band.get("available"):
        return band
    return {
        "available": True,
        "n_draws": band["n_draws"],
        "band_mean_delta_cpl_held": band["band_mean_delta_cpl_held"],
        "band_std_delta_cpl_held": band["band_std_delta_cpl_held"],
    }


def build_leaderboard(entries: list[dict], *, family_wise_threshold: float) -> list[dict]:
    """Rank dossiered candidates by noise-band percentile when available (higher = better,
    already oriented that way by dossier_tables._noise_band), falling back to raw delta_cpl_held
    ascending (a COST — more negative is better) when the batch has no noise floor yet.
    """
    rows = []
    for entry in entries:
        if entry["state"] != "dossiered":
            continue
        facts = entry["facts"]
        # dossier_tables.build_facts writes headline: None (not a missing key) for every
        # terminal status other than "rejected" (disqualified / aborted_candidate never
        # reach the scoring stages) — `.get("headline", {})` doesn't catch that, since the
        # key IS present, just null. `or {}` does.
        headline = facts.get("headline") or {}
        realised = headline.get("realised") or {}
        zone = headline.get("zone") or {}
        noise_band = facts.get("noise_band", {"available": False})
        candidate_conf = facts.get("candidate", {})
        percentile = noise_band.get("candidate_percentile_better_than_noise") if noise_band.get("available") else None
        rows.append(
            {
                "candidate": entry["candidate"],
                "status": facts.get("provenance", {}).get("status"),
                "delta_cpl_held": realised.get("delta_cpl_held"),
                "effect_resolved": realised.get("effect_resolved"),
                "zone_resolved": zone.get("resolved"),
                "confidence_effect": candidate_conf.get("confidence_effect"),
                "confidence_zone": candidate_conf.get("confidence_zone"),
                "noise_band_available": noise_band.get("available", False),
                "noise_band_percentile": percentile,
                "noise_band_z": noise_band.get("candidate_z_vs_band") if noise_band.get("available") else None,
                "clears_family_wise_threshold": (
                    percentile is not None and percentile >= family_wise_threshold
                ),
            }
        )
    has_noise_band = any(r["noise_band_available"] for r in rows)
    if has_noise_band:
        # Higher percentile = better; candidates without a percentile (noise band unavailable
        # for that one, shouldn't happen within one batch but not assumed) sort last.
        rows.sort(key=lambda r: (r["noise_band_percentile"] is None, -(r["noise_band_percentile"] or 0.0)))
    else:
        rows.sort(
            key=lambda r: (
                r["delta_cpl_held"] is None,
                r["delta_cpl_held"] if r["delta_cpl_held"] is not None else 0.0,
            )
        )
    return rows


def build_outcome_tally(entries: list[dict]) -> dict:
    dossiered_by_status: dict[str, int] = {}
    never_run = retryable_incomplete = pending_dossier = 0
    for entry in entries:
        if entry["state"] == "dossiered":
            status = entry["facts"].get("provenance", {}).get("status") or "unknown"
            dossiered_by_status[status] = dossiered_by_status.get(status, 0) + 1
        elif entry["state"] == "never_run":
            never_run += 1
        elif entry["state"] == "retryable_incomplete":
            retryable_incomplete += 1
        elif entry["state"] == "pending_dossier":
            pending_dossier += 1
    return {
        "total_candidates_filed": len(entries),
        "dossiered_by_status": dossiered_by_status,
        "never_run": never_run,
        "retryable_incomplete": retryable_incomplete,
        "pending_dossier": pending_dossier,
    }


def build_confidence_calibration(candidates_root: pathlib.Path = DEFAULT_CANDIDATES_DIR) -> dict:
    """Cumulative across every batch's facts.json — see the module docstring's point 4 for why
    this is NOT scoped to just the batch this retrospective was invoked on."""
    pairs = []
    for facts_path in sorted(pathlib.Path(candidates_root).glob("*/*/" + FACTS_FILENAME)):
        facts = json.loads(facts_path.read_text())
        candidate = facts.get("candidate", {})
        # See build_leaderboard's comment: headline is None, not missing, for a
        # disqualified/aborted_candidate dossier.
        headline = facts.get("headline") or {}
        realised = headline.get("realised") or {}
        zone = headline.get("zone") or {}
        pairs.append(
            {
                "candidate": candidate.get("name"),
                "batch": facts.get("provenance", {}).get("batch"),
                "confidence_effect": candidate.get("confidence_effect"),
                "effect_resolved": realised.get("effect_resolved"),
                "confidence_zone": candidate.get("confidence_zone"),
                "zone_resolved": zone.get("resolved"),
            }
        )

    n_resolved = sum(1 for p in pairs if p["effect_resolved"] is not None)
    insufficient_data = n_resolved < MIN_CALIBRATION_N
    result = {
        "scope": "cumulative across every dossiered candidate under experiments/candidates/, not just this batch",
        "min_calibration_n": MIN_CALIBRATION_N,
        "n_dossiered_with_resolved_effect": n_resolved,
        "insufficient_data": insufficient_data,
        "pairs": pairs,
        "mean_confidence_effect_when_resolved_true": None,
        "mean_confidence_effect_when_resolved_false": None,
    }
    if not insufficient_data:
        true_confs = [
            p["confidence_effect"] for p in pairs
            if p["effect_resolved"] is True and p["confidence_effect"] is not None
        ]
        false_confs = [
            p["confidence_effect"] for p in pairs
            if p["effect_resolved"] is False and p["confidence_effect"] is not None
        ]
        if true_confs:
            result["mean_confidence_effect_when_resolved_true"] = float(np.mean(true_confs))
        if false_confs:
            result["mean_confidence_effect_when_resolved_false"] = float(np.mean(false_confs))
    return result


def compute_retrospective(
    batch_name: str,
    *,
    batches_dir: pathlib.Path = DEFAULT_BATCHES_DIR,
    candidates_root: pathlib.Path = DEFAULT_CANDIDATES_DIR,
    force: bool = False,
) -> dict:
    """Build and persist `<batch_dir>/retrospective_facts.json`. Returns the written payload.

    Raises FileExistsError if the file already exists and `force` is not set (same overwrite
    guard as noise_floor.py — a retrospective already written is a record other work may already
    reference)."""
    batch_dir = pathlib.Path(batches_dir) / batch_name
    if not batch_dir.is_dir():
        raise ValueError(
            f"{batch_dir} does not exist — has this batch been frozen yet "
            "(experiments.pipeline.batch_freeze)? A typo'd batch name would otherwise either "
            "crash on the final write (no parent dir) or, if the batch dir happens to exist "
            "with no matching candidates dir, silently write a zero-candidate retrospective."
        )
    out_path = batch_dir / RETROSPECTIVE_FILENAME
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Pass force=True (CLI: --force) only if you intend to "
            "replace it."
        )

    candidate_modules = find_batch_candidates(batch_name, candidates_root)
    if not candidate_modules:
        raise ValueError(
            f"No candidate modules found under {pathlib.Path(candidates_root) / batch_name} — "
            "check the batch name and --candidates-dir; a real batch always has at least one "
            "candidate filed against it by the generator."
        )
    entries = [_candidate_entry(p) for p in candidate_modules]
    n_dossiered = sum(1 for e in entries if e["state"] == "dossiered")
    threshold = family_wise_percentile_threshold(max(n_dossiered, 1))

    payload = {
        "batch": batch_name,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "noise_floor": _batch_noise_summary(batch_dir),
        "family_wise_percentile_threshold": threshold,
        "leaderboard": build_leaderboard(entries, family_wise_threshold=threshold),
        "outcome_tally": build_outcome_tally(entries),
        "confidence_calibration": build_confidence_calibration(candidates_root),
    }
    out_path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    return payload


@click.command("retrospective")
@click.argument("batch_name")
@click.option(
    "--batches-dir", default=str(DEFAULT_BATCHES_DIR), show_default=True,
    help="Parent directory for batch snapshots.",
)
@click.option(
    "--candidates-dir", default=str(DEFAULT_CANDIDATES_DIR), show_default=True,
    help="Parent directory for candidate modules (experiments/candidates).",
)
@click.option(
    "--force", is_flag=True, default=False,
    help="Overwrite an existing retrospective_facts.json for this batch.",
)
def main(batch_name: str, batches_dir: str, candidates_dir: str, force: bool) -> None:
    """Compute and persist the batch retrospective for BATCH_NAME."""
    payload = compute_retrospective(
        batch_name,
        batches_dir=pathlib.Path(batches_dir),
        candidates_root=pathlib.Path(candidates_dir),
        force=force,
    )
    out_path = pathlib.Path(batches_dir) / batch_name / RETROSPECTIVE_FILENAME
    click.echo(
        f"Wrote retrospective for batch '{batch_name}' -> {out_path} "
        f"({payload['outcome_tally']['total_candidates_filed']} candidates filed)"
    )


if __name__ == "__main__":
    main()
