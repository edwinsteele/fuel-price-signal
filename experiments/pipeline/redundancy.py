"""Batch-level redundancy screen for candidate modules (fps-3jj.7 / fps-3jj.11).

The two mechanical checks `docs/routines/generator.md` requires **before any bead is
filed**, in one place so a generator session doesn't hand-roll them differently every
batch:

1. **Pairwise |rho| ACROSS candidates** — a HARD GATE. Two candidates correlated above
   the threshold are the same idea twice and would waste one of the batch's nights.
   Correlation *within* one multi-column candidate is reported but never gated: a group
   whose members are related is usually what makes them one mechanism rather than three,
   and gating it would ban the group shape by the back door.
2. **R^2 of each candidate against the existing column set**, computed as a BLOCK for a
   multi-column candidate. Reported, not gated — `generator.md` calls a candidate 90%
   reconstructible from existing columns something to *rework before filing*, which is a
   redesign prompt for a human, not a mechanical refusal.

Runs against **live** `data/features.parquet`, deliberately not the frozen batch: a
redundancy screen doesn't depend on which day's data it uses (a column 0.95-correlated
with `tgp_delta_7d` on Monday is 0.95-correlated on Thursday), and the freeze exists to
make cross-night *runs* comparable, which this is not.

Why a module rather than a scratch script: the checks are re-run every batch, the |rho|
threshold is a declared constant rather than a per-session guess (fps-3jj.11 — set from
the locked features' own correlation structure, see PAIRWISE_RHO_THRESHOLD), and the
output is the input to the batch record. The generator session still *runs* it; launch
validation does not duplicate it, because pairwise rho is a batch-level property that a
one-candidate-per-night launch structurally cannot compute.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import click
import numpy as np
import pandas as pd

from experiments.lib.io import to_jsonable
from experiments.pipeline.validate import load_candidate_module, validate_candidate
from fuel_signal.features import (
    DEFAULT_FEATURES_CSV,
    LOCKED_FEATURE_COLUMNS,
    load_features,
)

#: Cross-candidate |rho| at or above this fails the batch.
#:
#: Set empirically (fps-3jj.11), NOT by feel: measured against the 54 locked columns'
#: own pairwise correlations on batch1 data (300k-row sample), because the only honest
#: reference for "suspiciously similar" is what this dataset's *accepted* features
#: already look like. 25% of locked pairs sit at |rho| >= 0.7 and the highest is 0.971
#: (cycle_pct_through vs cycle_days_since_peak — the same quantity in two units, both
#: locked), so a 0.7 gate would reject candidate pairs for being no more similar than a
#: quarter of production. Only ~1% of locked pairs reach 0.85, which puts it in the
#: genuinely-unusual tail while staying below the ~0.95 reparameterisation case
#: generator.md names as the thing to catch.
#:
#: The asymmetry that settles the remaining doubt: a false REJECT costs one redesign
#: inside the generator session, a false ACCEPT costs a whole night of a five-night
#: batch. When in doubt, tighter.
PAIRWISE_RHO_THRESHOLD = 0.85

#: R^2 at or above this is *flagged for redesign*, never auto-rejected. generator.md's
#: "90% reconstructible ... should be reworked now, not filed" — a prompt to a human.
BLOCK_R2_FLAG = 0.9

#: Rows sampled for the screen. Correlation and R^2 over 65 predictors are stable long
#: before the full ~2.5M rows, and the screen is meant to be re-runnable in seconds
#: while candidates are still being edited.
DEFAULT_SAMPLE_ROWS = 300_000
SAMPLE_SEED = 42


def existing_column_set() -> list[str]:
    """The columns a candidate must not be reconstructible from: the lock, and only the lock.

    Single-sourced from ``fuel_signal.features.LOCKED_FEATURE_COLUMNS`` — the same symbol
    ``batch_freeze.resolve_baseline_columns`` writes into a batch's ``baseline_columns.json``
    and the runner's R0 arm trains on (docs/CONVENTIONS.md § "The baseline feature set is
    declared, never discovered"). Block R^2 asks whether a candidate adds anything the MODEL
    does not already have, so the predictor set has to be the set the model actually gets
    rather than a hand-rolled superset of it.

    That excludes every computed-but-non-model column: the Phase 4b brand troughs, and — since
    this function used to include it — ``tgp_delta_7d``, registered in
    ``fuel_signal.features.NON_MODEL_COLUMNS`` as ``evaluated-inconclusive`` and explicitly not
    in the lock. Being reconstructible from a column R0 never sees is not evidence a candidate
    is redundant with anything. Keeping it in biased the screen against the one lead
    ``experiments/ledger.yaml`` rates highest among still-open ground (the TGP gap against
    depth-remaining, which bd ``fps-x0f`` exists to revisit), and scored batch0's own
    ``tgp_delta_7d`` at block R^2 = 1.000 — a predictor of itself.

    PAIRWISE_RHO_THRESHOLD above was already derived from these 54 columns' own correlation
    structure, so this is what the rest of the module already assumed.
    """
    return list(LOCKED_FEATURE_COLUMNS)


class DuplicateCandidateColumn(RuntimeError):
    """Two candidates in one batch declared the same output column name.

    Unscreenable rather than merely untidy: `pairwise_rho` attributes each column to a
    candidate by NAME, so a collision makes the second candidate overwrite the first in
    that mapping. The colliding pair then resolves to the same owner on both sides, is
    classified within-candidate, and skips the hard gate entirely -- even at |rho| = 1.0
    from two genuinely different candidates. `pd.concat(axis=1)` and `.corr()` both accept
    duplicate labels silently, and nothing downstream catches it either: the runner's
    collision check is per-candidate against the baseline, and candidates run on separate
    nights, so it never sees two at once. This screen is the only place it can be caught.
    """


@dataclasses.dataclass(frozen=True)
class CandidateColumns:
    name: str
    mechanism_family: str | None
    columns: list[str]
    values: pd.DataFrame


def compute_candidate_columns(
    module_paths: list[pathlib.Path],
    frame: pd.DataFrame,
    *,
    validate: bool = True,
) -> list[CandidateColumns]:
    """Import each candidate and evaluate its `add_columns` against `frame`.

    Validates first by default — a candidate that can't pass the harness has no columns
    worth correlating, and finding that out here rather than five nights later is the
    entire point of screening before filing.
    """
    out: list[CandidateColumns] = []
    for path in module_paths:
        candidate = load_candidate_module(path)
        if validate:
            validate_candidate(candidate, frame)
        produced = candidate.add_columns(frame[list(candidate.INPUTS)].copy())
        out.append(
            CandidateColumns(
                name=candidate.NAME,
                mechanism_family=getattr(candidate, "MECHANISM_FAMILY", None),
                columns=list(candidate.COLUMNS),
                values=produced[list(candidate.COLUMNS)],
            )
        )

    seen: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for cand in out:
        for col in cand.columns:
            if col in seen:
                collisions.setdefault(col, [seen[col]]).append(cand.name)
            else:
                seen[col] = cand.name
    if collisions:
        detail = "; ".join(
            f"{col!r} declared by {owners}" for col, owners in collisions.items()
        )
        raise DuplicateCandidateColumn(
            "Candidates in this batch declare overlapping output column names: "
            f"{detail}. Rename one of each pair before screening -- a shared name cannot "
            "be attributed to a candidate, so the gate would silently classify the pair "
            "as within-candidate and never apply the threshold."
        )
    return out


def pairwise_rho(candidates: list[CandidateColumns]) -> pd.DataFrame:
    """Long-form |rho| for every column pair, tagged cross- vs within-candidate."""
    frames, owners = [], {}
    for cand in candidates:
        for col in cand.columns:
            owners[col] = cand.name
        frames.append(cand.values)
    joined = pd.concat(frames, axis=1)
    corr = joined.corr().abs()

    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            rows.append({
                "column_a": a,
                "column_b": b,
                "candidate_a": owners[a],
                "candidate_b": owners[b],
                "cross_candidate": owners[a] != owners[b],
                "abs_rho": float(corr.iat[i, j]),
            })
    # Explicit schema so a batch with no pairs at all still returns a frame the callers
    # can filter and sort. Not hypothetical: a one-candidate, one-column batch produces
    # zero pairs, which is exactly what batch0 was.
    if not rows:
        return pd.DataFrame({
            "column_a": pd.Series(dtype="object"),
            "column_b": pd.Series(dtype="object"),
            "candidate_a": pd.Series(dtype="object"),
            "candidate_b": pd.Series(dtype="object"),
            "cross_candidate": pd.Series(dtype="bool"),
            "abs_rho": pd.Series(dtype="float64"),
        })
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False, ignore_index=True)


def usable_predictors(predictors: pd.DataFrame) -> pd.DataFrame:
    """Drop predictor columns that carry no information on this frame.

    Necessary, not tidiness: five LGA trough columns (`bayside`, `waverley`,
    `hunters_hill`, `botany_bay`, `lane_cove`) are entirely NaN in the real features
    frame — they are locked columns whose LGA never produced a trough event — so a
    complete-case mask over ALL predictors selects ZERO rows and every R^2 comes back
    NaN. Found by smoke-testing the CLI against live data, where it silently produced a
    whole-batch report of `nan` rather than failing. `placebo.py` meets the same five
    (INDEX.md: "5 all-NaN skipped, 49 usable").

    Constant columns go too: they contribute nothing to a fit and only cost conditioning.
    """
    keep = [c for c in predictors.columns
            if predictors[c].notna().any() and predictors[c].nunique(dropna=True) > 1]
    return predictors[keep]


def _r2_against(target: pd.Series, predictors: pd.DataFrame) -> float:
    """OLS R^2 of one column on `predictors`, via lstsq with an intercept.

    `predictors` is expected to have been through `usable_predictors` already; the
    complete-case mask below is over what remains.
    """
    mask = target.notna() & predictors.notna().all(axis=1)
    y = target[mask].to_numpy(dtype=float)
    if y.size == 0:
        return float("nan")
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0.0:
        return float("nan")  # constant column — "reconstructible" is meaningless
    x = predictors[mask].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = float(((y - x @ coef) ** 2).sum())
    return float(1.0 - residual / total)


def block_r2(
    candidates: list[CandidateColumns], frame: pd.DataFrame, existing: list[str] | None = None
) -> pd.DataFrame:
    """Per-candidate block R^2 against the existing column set, plus per-column detail.

    The block figure is the MEAN of its columns' individual R^2 — the standard redundancy
    index for one set of variables against another. That is what makes generator.md's
    "a member that is individually reconstructible is not disqualifying if the group as a
    whole is not" true arithmetically: one member at 0.95 alongside two at 0.10 gives a
    block figure of 0.38, not a rejection.
    """
    existing = existing or existing_column_set()
    predictors = usable_predictors(frame[existing])
    dropped = sorted(set(existing) - set(predictors.columns))
    rows = []
    for cand in candidates:
        per_column = {col: _r2_against(cand.values[col], predictors) for col in cand.columns}
        finite = [v for v in per_column.values() if np.isfinite(v)]
        rows.append({
            "candidate": cand.name,
            "mechanism_family": cand.mechanism_family,
            "n_columns": len(cand.columns),
            "n_predictors_used": len(predictors.columns),
            "predictors_dropped": dropped,
            "block_r2": float(np.mean(finite)) if finite else float("nan"),
            "max_column_r2": float(np.max(finite)) if finite else float("nan"),
            "per_column_r2": per_column,
        })
    return pd.DataFrame(rows)


def screen_batch(
    module_paths: list[pathlib.Path],
    *,
    features_path: pathlib.Path | str = DEFAULT_FEATURES_CSV,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    rho_threshold: float = PAIRWISE_RHO_THRESHOLD,
) -> dict:
    """Run both checks over a whole batch. Returns the batch record's raw material."""
    frame = load_features(features_path)
    if sample_rows and len(frame) > sample_rows:
        frame = frame.sample(n=sample_rows, random_state=SAMPLE_SEED).sort_index()

    candidates = compute_candidate_columns(module_paths, frame)
    rho = pairwise_rho(candidates)
    r2 = block_r2(candidates, frame)

    cross = rho[rho["cross_candidate"]]
    violations = cross[cross["abs_rho"] >= rho_threshold]
    # A constant column correlates to NaN with everything, and `NaN >= threshold` is
    # False -- so without this an uncomputable pair reads as a clean pass. Same silent-NaN
    # failure mode `usable_predictors` exists to stop on the R^2 side; the gate deserves
    # the same treatment rather than the opposite one. Reported rather than raised: a
    # column can be constant on the SAMPLE without being constant on the full frame, so
    # this asks a human to look instead of hard-failing on a possible sampling artifact.
    uncomputable = cross[~np.isfinite(cross["abs_rho"])]
    return {
        "n_candidates": len(candidates),
        "n_rows_sampled": len(frame),
        "rho_threshold": rho_threshold,
        "pairwise_rho": rho,
        "block_r2": r2,
        "gate_violations": violations,
        "uncomputable_pairs": uncomputable,
        "passed": bool(violations.empty and uncomputable.empty),
        "mechanism_families": {c.name: c.mechanism_family for c in candidates},
    }


BATCH_RECORD_MD = "batch.md"
BATCH_RECORD_JSON = "batch.json"


def batch_dir_for(module_paths) -> pathlib.Path | None:
    """The single directory a batch record belongs to, or None if the modules span more.

    A record spanning two batch directories has no batch to belong to, and picking one
    silently would file this batch's numbers under another batch's name.
    """
    dirs = {pathlib.Path(m).parent for m in module_paths}
    return dirs.pop() if len(dirs) == 1 else None


def write_batch_record(result: dict, batch_candidates_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Persist the screen's output as the batch record generator.md § Batch record asks for.

    Two files, deliberately:

    * `batch.json` — what the retrospective reads. Prose is not a data format, and the
      one question this record exists to answer later ("did the generator produce five
      variants of one idea?") is a `groupby` over mechanism families, not a close reading.
    * `batch.md` — what a human reads at 7am. Same numbers, rendered.

    Written at SCREEN time rather than at filing time because these are the numbers that
    justified filing: a record reconstructed afterwards would be a fresh measurement of
    whatever the modules say *now*, which is exactly the residual risk generator.md
    accepts when it notes that a module hand-edited after filing is unchecked.
    """
    batch_candidates_dir = pathlib.Path(batch_candidates_dir)
    batch_candidates_dir.mkdir(parents=True, exist_ok=True)

    rho = result["pairwise_rho"]
    r2 = result["block_r2"]
    families = result["mechanism_families"]

    payload = {
        "n_candidates": result["n_candidates"],
        "n_rows_sampled": result["n_rows_sampled"],
        "rho_threshold": result["rho_threshold"],
        "passed": result["passed"],
        "mechanism_families": families,
        "n_distinct_families": len({f for f in families.values() if f}),
        "undeclared_families": sorted(n for n, f in families.items() if not f),
        "block_r2": r2.to_dict("records"),
        "pairwise_rho": rho.to_dict("records"),
        "gate_violations": result["gate_violations"].to_dict("records"),
        "uncomputable_pairs": result["uncomputable_pairs"].to_dict("records"),
    }
    # to_jsonable, not bare json.dumps: this screen treats NaN as a FIRST-CLASS outcome
    # (an uncomputable |rho| from a constant column, an unmeasurable block R^2), and
    # json.dumps emits those as the bare token `NaN`, which is not valid JSON. Python's
    # own loads() tolerates it, so it looks fine from inside the repo and breaks for
    # every strict consumer -- jq included -- in exactly the cases this module goes out
    # of its way to surface. Shared helper for the same reason results.json and
    # retrospective_facts.json use it (docs/CONVENTIONS.md).
    json_path = batch_candidates_dir / BATCH_RECORD_JSON
    json_path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str) + "\n")

    cross = rho[rho["cross_candidate"]]
    within = rho[~rho["cross_candidate"]]

    lines = [
        f"# Batch record — {batch_candidates_dir.name}",
        "",
        "Written by `experiments.pipeline.redundancy` at screen time, before any bead was",
        "filed. These are the numbers that justified filing; see that module for why the",
        "gate sits where it does.",
        "",
        f"- Candidates: **{result['n_candidates']}**",
        f"- Rows sampled: {result['n_rows_sampled']:,}",
        f"- Cross-candidate |rho| gate: **{result['rho_threshold']}**",
        f"- Verdict: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "## Mechanism families (disclosure, never gated)",
        "",
        f"{payload['n_distinct_families']} distinct family label(s) across "
        f"{result['n_candidates']} candidate(s).",
        "",
        "| candidate | mechanism family |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | {family or '**NOT DECLARED**'} |" for name, family in families.items()]
    # `n_distinct_families` counts truthy labels only, so {a: "f1", b: None} would
    # otherwise assert "every candidate shares one family" two lines under a table row
    # marking b as NOT DECLARED. This callout is the record's headline finding; a
    # self-contradicting edge case costs more than the case is worth.
    if (payload["n_distinct_families"] == 1 and result["n_candidates"] > 1
            and not payload["undeclared_families"]):
        lines += ["", "> **Every candidate in this batch shares one family label.** Not a failure "
                  "and not gated — but it is the finding about the generator that this "
                  "disclosure exists to make visible. Read the retrospective with it in mind."]

    lines += ["", "## Block R^2 against the existing column set", "",
              f"Flagged at {BLOCK_R2_FLAG}; reported, never auto-rejected. Block figure is the "
              "mean of the candidate's member columns.", "",
              "| candidate | n cols | block R^2 | max column R^2 |", "|---|---|---|---|"]
    for row in r2.to_dict("records"):
        flag = " **<- redesign**" if np.isfinite(row["block_r2"]) and row["block_r2"] >= BLOCK_R2_FLAG else ""
        block = f"{row['block_r2']:.3f}" if np.isfinite(row["block_r2"]) else "NaN — NOT MEASURED"
        mx = f"{row['max_column_r2']:.3f}" if np.isfinite(row["max_column_r2"]) else "NaN"
        lines.append(f"| `{row['candidate']}` | {row['n_columns']} | {block}{flag} | {mx} |")

    lines += ["", "## Pairwise |rho|", "",
              "**Cross-candidate pairs are the hard gate. Within-candidate pairs are disclosure "
              "only** — a group whose members are related is usually what makes them one "
              "mechanism rather than three.", ""]
    if cross.empty:
        lines.append("_No cross-candidate pairs (single-candidate batch)._")
    else:
        lines += ["### Cross-candidate (gated)", "", "| |rho| | a | b |", "|---|---|---|"]
        for row in cross.to_dict("records"):
            val = f"{row['abs_rho']:.3f}" if np.isfinite(row["abs_rho"]) else "NaN"
            mark = " **<- GATE**" if np.isfinite(row["abs_rho"]) and row["abs_rho"] >= result["rho_threshold"] else ""
            lines.append(f"| {val}{mark} | `{row['candidate_a']}.{row['column_a']}` "
                         f"| `{row['candidate_b']}.{row['column_b']}` |")
    if not within.empty:
        lines += ["", "### Within-candidate (disclosure only)", "", "| |rho| | candidate | a | b |",
                  "|---|---|---|---|"]
        for row in within.to_dict("records"):
            val = f"{row['abs_rho']:.3f}" if np.isfinite(row["abs_rho"]) else "NaN"
            lines.append(f"| {val} | `{row['candidate_a']}` | `{row['column_a']}` | `{row['column_b']}` |")

    md_path = batch_candidates_dir / BATCH_RECORD_MD
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, json_path


@click.command("redundancy")
@click.argument("module_paths", nargs=-1, required=True,
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option("--features-path", default=str(DEFAULT_FEATURES_CSV), show_default=True)
@click.option("--sample-rows", default=DEFAULT_SAMPLE_ROWS, show_default=True)
@click.option("--rho-threshold", default=PAIRWISE_RHO_THRESHOLD, show_default=True)
@click.option("--write-record/--no-write-record", default=True, show_default=True,
              help="Write batch.md + batch.json beside the candidate modules "
                   "(generator.md § Batch record). Requires all modules in one directory.")
def main(module_paths, features_path, sample_rows, rho_threshold, write_record) -> None:
    """Screen a batch of candidate modules before filing any bead."""
    result = screen_batch(list(module_paths), features_path=features_path,
                          sample_rows=sample_rows, rho_threshold=rho_threshold)

    click.echo(f"[redundancy] {result['n_candidates']} candidates, "
               f"{result['n_rows_sampled']:,} rows sampled\n")

    click.echo("Mechanism families (disclosure, not gated):")
    for name, family in result["mechanism_families"].items():
        click.echo(f"  {name:<34} {family or 'NOT DECLARED'}")

    first = result["block_r2"].iloc[0] if len(result["block_r2"]) else None
    if first is not None:
        click.echo(f"\nBlock R^2 vs existing set ({first['n_predictors_used']} usable "
                   f"predictors, flag at {BLOCK_R2_FLAG}):")
        if first["predictors_dropped"]:
            click.echo(f"  (dropped {len(first['predictors_dropped'])} all-NaN/constant: "
                       f"{', '.join(first['predictors_dropped'])})")
    for row in result["block_r2"].to_dict("records"):
        if not np.isfinite(row["block_r2"]):
            click.echo(f"  {row['candidate']:<34} block=NaN  <-- NOT MEASURED, investigate")
            continue
        flag = "  <-- REDESIGN" if row["block_r2"] >= BLOCK_R2_FLAG else ""
        click.echo(f"  {row['candidate']:<34} block={row['block_r2']:.3f} "
                   f"max_col={row['max_column_r2']:.3f} n={row['n_columns']}{flag}")

    within = result["pairwise_rho"][~result["pairwise_rho"]["cross_candidate"]]
    if not within.empty:
        click.echo("\nWithin-candidate |rho| (disclosure only, never gated):")
        for row in within.head(10).to_dict("records"):
            click.echo(f"  {row['abs_rho']:.3f}  {row['column_a']} <-> {row['column_b']}")

    cross = result["pairwise_rho"][result["pairwise_rho"]["cross_candidate"]]
    # Finite only: an uncomputable pair gets its own section below, with the explanation
    # attached. A bare `nan` in the GATE listing reads as a display bug to anyone who
    # hasn't scrolled far enough to find out it isn't one.
    click.echo(f"\nCross-candidate |rho| (HARD GATE at {rho_threshold}):")
    for row in cross[np.isfinite(cross["abs_rho"])].head(10).to_dict("records"):
        flag = "  <-- GATE" if row["abs_rho"] >= rho_threshold else ""
        click.echo(f"  {row['abs_rho']:.3f}  {row['candidate_a']}.{row['column_a']} "
                   f"<-> {row['candidate_b']}.{row['column_b']}{flag}")

    if not result["uncomputable_pairs"].empty:
        click.echo("\nUNCOMPUTABLE cross-candidate |rho| (constant column?) -- investigate:")
        for row in result["uncomputable_pairs"].to_dict("records"):
            click.echo(f"  {row['candidate_a']}.{row['column_a']} <-> "
                       f"{row['candidate_b']}.{row['column_b']}")

    if write_record:
        batch_dir = batch_dir_for(module_paths)
        if batch_dir is None:
            dirs = sorted({str(pathlib.Path(m).parent) for m in module_paths})
            click.echo(f"\n[redundancy] batch record SKIPPED — modules span {len(dirs)} "
                       f"directories ({dirs}); a batch record belongs to exactly one batch.")
        else:
            md_path, json_path = write_batch_record(result, batch_dir)
            click.echo(f"\n[redundancy] batch record: {md_path}, {json_path}")

    if result["passed"]:
        click.echo("\n[redundancy] PASS — no cross-candidate pair at or above the gate.")
    else:
        click.echo(f"\n[redundancy] FAIL — {len(result['gate_violations'])} pair(s) at or "
                   f"above the gate, {len(result['uncomputable_pairs'])} uncomputable. "
                   "Redesign before filing.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
