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
     single candidate's raw distance from the noise band (already computed per-candidate by
     dossier_tables._noise_band as `candidate_z_vs_band`) is a fair one-shot test; picking the
     BEST of N candidates against that same band is not, and needs a higher bar.
     `family_wise_z_threshold` reports it (fps-awz: a Bonferroni-corrected, t-distributed
     critical value in BAND-STANDARD-DEVIATION space, honest about the band's std being
     estimated from a finite draw count), and `clears_family_wise_threshold` on each leaderboard
     row uses it, not the raw empirical percentile. `family_wise_percentile_threshold` and each
     row's `noise_band_percentile` are still reported — descriptive colour, not the gate — but
     `clears_family_wise_threshold` no longer reads them: with only 5 draws the empirical rank
     statistic had 6 possible values, so "gate on percentile" collapsed to "beat every draw"
     regardless of where the threshold was nominally set (fps-awz "Why 2").

  3. Outcome-code tally — every candidate MODULE filed against the batch
     (`experiments/candidates/<batch>/*.py`) is the universe, not just the ones that reached a
     dossier. A module with no results.json at all is `never_run` — missing data, not a rejection.
     A module with a results.json in a RETRYABLE status has not had a fair hearing yet either
     (still cycling through launch.py's retry budget, or exhausted and sitting `blocked` in bd —
     this module reads disk only, so it cannot tell those two apart; `docs/routines/retrospective.md`
     flags that as a known gap rather than guessing). Folding either into `graded` would bias the
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
from experiments.pipeline.dossier_tables import (
    FACTS_FILENAME,
    FAMILY_WISE_ALPHA,
    NOISE_FLOOR_FILENAME,
    _noise_band,
    family_wise_z_threshold,
)
from experiments.pipeline.runner import RETRYABLE_STATUSES, default_out_dir, read_run_status

DEFAULT_CANDIDATES_DIR = DEFAULT_BATCHES_DIR.parent / "candidates"
RETROSPECTIVE_FILENAME = "retrospective_facts.json"

# "Five candidates cannot calibrate anything on their own" (generator.md) — batch 2 alone is 5.
# Set above that so a single batch's confidence pairs never masquerade as a calibration read;
# the field only turns on once pairs have accumulated across multiple batches.
MIN_CALIBRATION_N = 10


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

    `tank_params` (fps-aam) is read verbatim off the batch's own noise_floor.json — the file
    `_noise_band` just graded this band from — never re-derived from a TankParams here. The
    band's mean/std ARE CPL deltas, in whatever cadence's units the floor was computed at
    (fps-fii measured realised CPL moving 189.67/187.85/187.82 c/L across 7/2/1-day cadence),
    so the summary is unreadable without the stamp the same way results.csv was before fps-15c.
    """
    band = _noise_band({"effect_delta_cpl_held": 0.0}, batch_dir, check_fingerprint=False)
    if not band.get("available"):
        # No CPL value in a refusal dict, so nothing to stamp — returned verbatim, including
        # its `reason` (which is the actionable half for a reader).
        return band
    # `_noise_band` already read this file; re-reading it here rather than threading the raw
    # value out through that shared return keeps fps-15c's facts.json contract (already
    # closed) untouched — this is the one CPL-carrying artifact fps-15c left unstamped.
    noise_floor = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    tank_params = noise_floor.get("tank_params")
    if not tank_params:
        # Same refusal as dossier_tables.build_facts' (fps-15c): a band mean/std is a CPL, and
        # a CPL with no cadence beside it is the ambiguity the stamp exists to remove. Note
        # `_noise_band` was called with check_fingerprint=False, which skips COMPARING the
        # floor's stamp against a run's — it does not excuse the stamp being absent, and this
        # summary is the batch-level artifact where that distinction has to be re-asserted.
        raise ValueError(
            f"{batch_dir / NOISE_FLOOR_FILENAME}: carries deltas_cpl_held (this summary is "
            "about to report their mean/std) but no tank_params — refusing to write an "
            "unstamped CPL into the retrospective (fps-aam/fps-15c). Recompute the floor with "
            "`PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor "
            f"{batch_dir.name} --force`, or backfill the cadence it actually ran at."
        )
    return {
        "available": True,
        "n_draws": band["n_draws"],
        # fps-3jj.25, forwarded because the BATCH gate below needs it. What the bank is worth
        # once source-column reuse is priced in — `family_wise_z_threshold` takes the effective
        # count, not the nominal one, and dropping it here left the batch-level bar computed at
        # df = nominal - 1 while every per-run bar in the same batch used the effective count.
        # The two artifacts then contradict each other, and the one left too easy is the
        # BATCH-level gate: the one that corrects for taking the best of N.
        "effective_n_draws": band["effective_n_draws"],
        # The arity this band measures (fps-3jj.14). Surfaced at batch level because the
        # leaderboard ranks candidates of DIFFERENT arities against this one band, and
        # whether that comparison is like-for-like is a fact about the ruler, not about any
        # single row.
        "n_placebo_columns": band["n_placebo_columns"],
        "tank_params": tank_params,
        "band_mean_delta_cpl_held": band["band_mean_delta_cpl_held"],
        "band_std_delta_cpl_held": band["band_std_delta_cpl_held"],
    }


def build_leaderboard(entries: list[dict], *, family_wise_z_gate: float | None) -> list[dict]:
    """Rank dossiered candidates by noise-band z (lower/more-negative = better — delta_cpl_held
    is a COST, oriented that way by dossier_tables._noise_band) when available, falling back to
    raw delta_cpl_held ascending when the batch has no noise floor yet.

    family_wise_z_gate: `family_wise_z_threshold(...)`'s output, a POSITIVE distance in band
    standard deviations. `clears_family_wise_threshold` fires when a row's `noise_band_z` is
    <= its negation (delta_cpl_held is a cost, so "surprisingly better" is very negative z).
    None when the batch's noise band can't support a z estimate (fewer than 2 draws, or
    unavailable) — every row's gate is then False rather than computed against a bar that
    doesn't exist.

    `noise_band_percentile` is still reported per row as descriptive colour, but no longer
    drives either the gate or the sort order — at the fps-awz draw count (~20) the empirical
    percentile only has `n_draws + 1` distinct values, so sorting by it (as this function used
    to) produces ties a continuous `noise_band_z` doesn't have, and could disagree with the
    gate's own ordering. There is no `family_wise_threshold` (percentile-space) parameter here
    any more — the caller (`compute_retrospective`) still computes and reports it at the
    payload's top level, but this function has no use for it.

    Raises ValueError (fps-aam) for a dossiered candidate whose facts.json carries a
    `delta_cpl_held` but no `provenance.tank_params`, and for a batch whose rows don't
    all share ONE cadence — ranking is a comparison, and a cross-cadence one isn't
    valid. See the two guards below.
    """
    rows = []
    for entry in entries:
        if entry["state"] != "dossiered":
            continue
        facts = entry["facts"]
        # dossier_tables.build_facts writes headline: None (not a missing key) for every
        # terminal status other than "graded" (disqualified / aborted_candidate never
        # reach the scoring stages) — `.get("headline", {})` doesn't catch that, since the
        # key IS present, just null. `or {}` does.
        headline = facts.get("headline") or {}
        realised = headline.get("realised") or {}
        zone = headline.get("zone") or {}
        noise_band = facts.get("noise_band", {"available": False})
        candidate_conf = facts.get("candidate", {})
        percentile = noise_band.get("candidate_percentile_better_than_noise") if noise_band.get("available") else None
        z = noise_band.get("candidate_z_vs_band") if noise_band.get("available") else None
        provenance = facts.get("provenance") or {}
        delta_cpl_held = realised.get("delta_cpl_held")
        tank_params = provenance.get("tank_params")
        if delta_cpl_held is not None and not tank_params:
            # fps-aam, the same refusal dossier_tables.build_facts makes one artifact
            # upstream: this row is about to carry a realised CPL delta, and the facts.json
            # it came from names no cadence to label it with. A null stamp beside a real
            # CPL is worse than none — experiments.lib.io.artifact_has_unstamped_cpl is a
            # whole-document "a stamp exists somewhere" check (deliberately coarse, see its
            # docstring), so ONE stamped row would mask an unstamped sibling and the
            # backstop would pass on an artifact this bead exists to make readable.
            raise ValueError(
                f"candidate {entry['candidate']!r}: facts.json carries "
                f"delta_cpl_held={delta_cpl_held!r} but provenance.tank_params is "
                f"{tank_params!r} — refusing to rank an unstamped CPL (fps-aam/fps-15c). "
                "Its dossier predates the stamp: re-run "
                "`PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables` for "
                "that candidate (backfilling its results.json meta first if that is where "
                "the cadence is missing)."
            )
        rows.append(
            {
                "candidate": entry["candidate"],
                # Carried per-row so a batch retrospective can see family CONCENTRATION
                # without reopening five candidate modules -- five uncorrelated
                # candidates that all tell one story is the finding about the generator
                # that generator.md's diversity rule exists to surface.
                "mechanism_family": (facts.get("candidate") or {}).get("mechanism_family"),
                "status": provenance.get("status"),
                # The cadence this row's delta_cpl_held was produced at (fps-aam), read
                # straight off the candidate's own facts.json provenance — never
                # re-derived from a TankParams here, so a row can never disagree with the
                # dossier it is a summary of. Null only for a non-graded candidate, which
                # has no delta_cpl_held to label (the guard above enforces that pairing).
                "tank_params": tank_params,
                "delta_cpl_held": delta_cpl_held,
                "effect_resolved": realised.get("effect_resolved"),
                "zone_resolved": zone.get("resolved"),
                "confidence_effect": candidate_conf.get("confidence_effect"),
                "confidence_zone": candidate_conf.get("confidence_zone"),
                "noise_band_available": noise_band.get("available", False),
                "noise_band_percentile": percentile,
                "noise_band_z": z,
                "clears_family_wise_threshold": (
                    z is not None and family_wise_z_gate is not None and z <= -family_wise_z_gate
                ),
            }
        )
    stamps: dict[str, list[str]] = {}
    for row in rows:
        if row["tank_params"] is not None:
            stamps.setdefault(row["tank_params"], []).append(row["candidate"])
    if len(stamps) > 1:
        # Carrying the stamp is not enough on its own: every sort branch below, and
        # `clears_family_wise_threshold`, ASSERT that these deltas are commensurable.
        # They are not across cadences — fps-fii measured realised CPL at
        # 189.67/187.85/187.82 c/L across 7/2/1-day on an otherwise identical run,
        # which is an order of magnitude larger than the deltas this function ranks,
        # so the ordering would be reporting the cadence and not the candidates.
        #
        # Raising rather than reporting an "unranked" leaderboard, because there is no
        # legitimate mixed-cadence batch to report: batch_freeze stamps ONE cadence
        # "for every candidate run against it" (freeze.json's tank_params), and
        # noise_floor.py enforces it across every draw. Reaching here means something
        # upstream already broke that declaration, which is a human's call to make
        # (re-run the odd candidates, or re-freeze the batch), not this module's.
        detail = "; ".join(f"{stamp} ({', '.join(sorted(names))})" for stamp, names in sorted(stamps.items()))
        raise ValueError(
            f"leaderboard spans {len(stamps)} cadences — {detail} — and delta_cpl_held is "
            "not comparable across them (fps-aam/fps-fii), so ranking them against each "
            "other would report the cadence, not the candidates. A batch declares ONE "
            "cadence for every candidate run against it (freeze.json's tank_params); "
            "re-run the candidates that disagree with it, or re-freeze the batch."
        )
    has_z = any(r["noise_band_z"] is not None for r in rows)
    if has_z:
        # Lower/more-negative z = better (delta_cpl_held is a cost). A row with no z
        # (noise band unavailable for that one specifically, or too few draws to estimate a
        # std) sorts last rather than being assumed best-or-worst.
        rows.sort(key=lambda r: (r["noise_band_z"] is None, r["noise_band_z"] if r["noise_band_z"] is not None else 0.0))
    elif any(r["noise_band_available"] for r in rows):
        # z unavailable batch-wide (e.g. < 2 draws) but the band itself is — percentile is
        # still a real, if coarser, ordering signal, so fall back to it rather than raw delta.
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
    """`dossiered_by_status` keys on `"{status}:{abort_reason}"` when a candidate's facts.json
    carries an abort_reason (fps-3jj.13), `"{status}"` otherwise — so e.g.
    `"aborted_candidate:leak_by_declaration"` and plain `"aborted_candidate"` tally separately.
    This is the leak-vs-bug breakdown aim (b) needs: without it every declared leak (a candidate
    that named a label column as INPUTS/COLUMNS) was indistinguishable from an ordinary bug in
    the candidate's arithmetic, both landing on the same "aborted_candidate" count."""
    dossiered_by_status: dict[str, int] = {}
    never_run = retryable_incomplete = pending_dossier = 0
    for entry in entries:
        if entry["state"] == "dossiered":
            provenance = entry["facts"].get("provenance", {})
            status = provenance.get("status") or "unknown"
            abort_reason = provenance.get("abort_reason")
            key = f"{status}:{abort_reason}" if abort_reason else status
            dossiered_by_status[key] = dossiered_by_status.get(key, 0) + 1
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
    # Gate on USABLE pairs, not just resolved-effect ones: a resolved-effect candidate with
    # no recorded confidence_effect (e.g. predates the two-CONFIDENCE-field convention)
    # contributes nothing to the means below. Gating on n_resolved alone could report
    # insufficient_data: false while every mean still comes out None — exactly the
    # unsupported-calibration-read this flag exists to prevent.
    n_usable = sum(1 for p in pairs if p["effect_resolved"] is not None and p["confidence_effect"] is not None)
    insufficient_data = n_usable < MIN_CALIBRATION_N
    result = {
        "scope": "cumulative across every dossiered candidate under experiments/candidates/, not just this batch",
        "min_calibration_n": MIN_CALIBRATION_N,
        "n_dossiered_with_resolved_effect": n_resolved,
        "n_usable_for_calibration": n_usable,
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
    n_candidates = max(n_dossiered, 1)
    threshold = family_wise_percentile_threshold(n_candidates)
    noise_floor_summary = _batch_noise_summary(batch_dir)
    # The EFFECTIVE draw count, not the nominal one (fps-3jj.25, fixed in review of PR #336).
    # `family_wise_z_threshold`'s contract is that it takes effective draws; passing nominal
    # here made the batch-level gate act as though the band were better estimated than it is —
    # a bar too EASY, at precisely the step that corrects for taking the best of N. Measured on
    # a 20-draw arity-35 floor (n_eff 3.23) with 5 candidates: the gate was 2.602 where the
    # priced value is 6.96, so a candidate at z = -4.0 was stamped
    # `clears_family_wise_threshold: true` here while failing its own
    # `single_candidate_z_threshold` in facts.json. Falls back to the nominal count only when
    # the summary predates the field.
    n_draws = None
    if noise_floor_summary.get("available"):
        n_draws = noise_floor_summary.get("effective_n_draws") or noise_floor_summary.get("n_draws")
    # family_wise_z_threshold needs n_draws >= 2 (a t-critical value needs >= 1 degree of
    # freedom) — the same condition dossier_tables._noise_band() already gates
    # candidate_z_vs_band on, so z_gate is None in exactly the cases where no row could have
    # a non-None noise_band_z to compare it against anyway.
    z_gate = family_wise_z_threshold(n_candidates, n_draws) if n_draws and n_draws >= 2 else None

    payload = {
        "batch": batch_name,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "noise_floor": noise_floor_summary,
        "family_wise_percentile_threshold": threshold,
        "family_wise_z_threshold": z_gate,
        "leaderboard": build_leaderboard(entries, family_wise_z_gate=z_gate),
        "outcome_tally": build_outcome_tally(entries),
        "confidence_calibration": build_confidence_calibration(candidates_root),
    }
    # Converted once and both written and returned from the SAME object: to_jsonable maps
    # non-finite floats (e.g. band_std_delta_cpl_held at n_draws=1, a real value _noise_band
    # can produce) to None, so returning the pre-conversion `payload` instead would let a
    # caller observe a NaN that the file on disk stores as null — same value, different type,
    # a real (if narrow) way for a caller and the persisted record to silently disagree.
    jsonable_payload = to_jsonable(payload)
    out_path.write_text(json.dumps(jsonable_payload, indent=2) + "\n")
    return jsonable_payload


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
