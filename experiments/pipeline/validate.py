"""Candidate validation harness — the whole safety layer for LLM-authored candidate
modules, since nobody reviews the code (fps-3jj.2).

Run before any fitting, in this order:

0. **Target columns are removed from the frame, and named in `INPUTS` is an abort.**
   `features.TARGET_COLUMNS` (`future_min_cents`, `label`) are computed from prices
   *after* `price_date`, so step 2 structurally cannot see a candidate reading one:
   truncating the frame by date doesn't change a value already written into the cell.
   Two halves, and the order matters — the **barrier** is that the candidate is handed
   a frame with those columns dropped (matching what `runner.py` hands it), so an
   undeclared `frame.get("future_min_cents")` returns None rather than the answer; the
   **`INPUTS` check** on top of that turns a declared read into a named, immediate abort
   instead of a silent None. Neither alone is enough: the check only sees declarations,
   and the drop alone would fail confusingly.
1. **Restricted-frame INPUTS check** — call the candidate's `add_columns` /
   `add_axis` against `frame[declared_inputs]` instead of the full frame. Any
   undeclared read raises `KeyError` on its own; exact, unevadable, free. This is a
   *provenance* check (INPUTS feeds the correlation/redundancy checks and the
   ledger's `inputs` field), not a leak check.
2. **Differential PIT test** (`experiments/lib/pit_test.py`) — the leak abort for
   *computed* leaks (a forward-reaching aggregate). Step 0 covers the stamped kind it
   cannot see; the two are complementary, neither subsumes the other.
3. **NaN-rate assert** — a declared output column that is entirely NaN means
   `add_columns` computed nothing; abort. Partial NaN (warm-up rows) is normal and
   just reported for the dossier.

Columns-only smoke throughout — this module never fits a model.

Candidate protocol (see fps-3jj / fps-3jj.4 for the full module-format spec): an
object exposing `NAME`, `INPUTS` (list[str]), `COLUMNS` (list[str]),
`add_columns(df) -> pd.DataFrame`, and optionally `add_axis(df) -> pd.Series`.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import click
import pandas as pd

from experiments.lib.pit_test import differential_pit_test
from fuel_signal.features import DEFAULT_FEATURES_CSV, TARGET_COLUMNS, load_features


class RestrictedFrameViolation(RuntimeError):
    """A candidate's add_columns/add_axis read a column outside its declared INPUTS."""


class MissingInputColumnsError(RuntimeError):
    """A candidate declares an INPUTS column that doesn't exist in the features frame.

    Distinct from RestrictedFrameViolation: this is a bad declaration, not an
    undeclared read.
    """


class TargetColumnInInputs(RuntimeError):
    """A candidate declared one of the label columns (`features.TARGET_COLUMNS`) in INPUTS.

    The one leak the differential PIT test structurally cannot catch, so it is caught
    here instead, by declaration, before anything runs. See TARGET_COLUMNS' own comment:
    a forward-looking value stamped ON the row survives date-truncation unchanged, so
    recompute-and-compare finds nothing wrong with `future_min_cents - station_price_cents`.
    """


class MissingOutputColumnError(RuntimeError):
    """A candidate's add_columns didn't produce a column it declared in COLUMNS."""


class AllNaNColumnError(RuntimeError):
    """A candidate's declared output column is entirely NaN — it computed nothing."""


class CandidateImportError(RuntimeError):
    """A candidate module file could not be imported."""


@dataclasses.dataclass(frozen=True)
class CandidateValidationReport:
    columns: list[str]
    nan_rates: dict[str, float]
    axis_present: bool


def load_candidate_module(path: pathlib.Path):
    """Import a candidate module by file path, e.g. experiments/candidates/<batch>/<name>.py."""
    path = pathlib.Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise CandidateImportError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CandidateImportError(f"Failed to import candidate module {path}: {exc}") from exc
    return module


def validate_candidate(candidate, frame: pd.DataFrame, *, date_column: str = "price_date") -> CandidateValidationReport:
    """Run the full pre-fit validation harness for one candidate module against `frame`.

    Raises on the first failure. Never fits a model.
    """
    inputs = list(candidate.INPUTS)

    # Before the restricted frame is even built: the restricted-frame check enforces that
    # a candidate reads only what it DECLARED, and the PIT test catches reaching forward
    # across rows. Neither says a declared read is allowed to be the target.
    declared_targets = [col for col in inputs if col in TARGET_COLUMNS]
    if declared_targets:
        why = "; ".join(f"{col}: {TARGET_COLUMNS[col]}" for col in declared_targets)
        raise TargetColumnInInputs(
            f"{candidate.NAME}: INPUTS declares label column(s) {declared_targets}, which are "
            f"computed from prices after price_date and are never a valid feature input. {why}"
        )

    # Output side. Without this a candidate declaring COLUMNS=["future_min_cents"] and an
    # identity add_columns is still stopped -- the target columns are dropped below, so it
    # fails to produce what it declared -- but it fails as MissingOutputColumnError, which
    # reads as "your function is broken" rather than "you tried to make the label a
    # feature". Same refusal, named.
    #
    # Scoped to TARGET_COLUMNS deliberately, NOT to "any column already in the frame":
    # re-declaring an existing frame column is the LEGITIMATE shape for a candidate that
    # promotes a computed-but-excluded column into the model. batch0's own known-graduate
    # candidate is exactly that (COLUMNS = INPUTS = ["tgp_delta_7d"], a column present in
    # the frame and deliberately outside the lock), so a blanket frame-collision rule
    # would have rejected the run that calibrated this pipeline. The baseline contract is
    # the other half and is checked in runner.py, which is where baseline_columns lives.
    produced_targets = [col for col in candidate.COLUMNS if col in TARGET_COLUMNS]
    if produced_targets:
        raise TargetColumnInInputs(
            f"{candidate.NAME}: COLUMNS declares label column(s) {produced_targets} as candidate "
            "output, which would overwrite the model's target with a candidate-computed value."
        )

    try:
        restricted = frame[inputs].copy()
    except KeyError as exc:
        missing = [col for col in inputs if col not in frame.columns]
        raise MissingInputColumnsError(
            f"{candidate.NAME}: INPUTS declares columns not present in the features frame: {missing}"
        ) from exc

    # The full-frame calls below use the same target-stripped view the runner gives a
    # candidate (runner.py), so validation exercises exactly what will run — and so a
    # candidate that reaches for a label column WITHOUT declaring it (frame.get(...),
    # which returns None rather than raising) sees None here too, instead of the oracle.
    frame = frame.drop(columns=list(TARGET_COLUMNS), errors="ignore")

    _check_restricted_frame(candidate.add_columns, restricted, candidate_name=candidate.NAME, fn_label="add_columns")
    columns_result = differential_pit_test(candidate.add_columns, frame, date_column=date_column)

    nan_rates = {}
    for col in candidate.COLUMNS:
        if col not in columns_result.columns:
            raise MissingOutputColumnError(
                f"{candidate.NAME}: declared output column '{col}' was not produced by add_columns"
            )
        rate = float(columns_result[col].isna().mean())
        if rate >= 1.0:
            raise AllNaNColumnError(f"{candidate.NAME}: column '{col}' is entirely NaN — add_columns computed nothing")
        nan_rates[col] = rate

    add_axis = getattr(candidate, "add_axis", None)
    axis_present = add_axis is not None
    if axis_present:
        _check_restricted_frame(add_axis, restricted, candidate_name=candidate.NAME, fn_label="add_axis")
        differential_pit_test(add_axis, frame, date_column=date_column)

    return CandidateValidationReport(columns=list(candidate.COLUMNS), nan_rates=nan_rates, axis_present=axis_present)


def _check_restricted_frame(fn, restricted_frame: pd.DataFrame, *, candidate_name: str, fn_label: str) -> None:
    try:
        fn(restricted_frame)
    except KeyError as exc:
        raise RestrictedFrameViolation(
            f"{candidate_name}.{fn_label} read a column outside its declared INPUTS "
            f"{list(restricted_frame.columns)}: {exc}"
        ) from exc


@click.command("validate")
@click.argument("candidate_path", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option(
    "--features-path", default=str(DEFAULT_FEATURES_CSV), show_default=True,
    help="Features CSV to validate against (its .parquet sibling is used if a batch froze one).",
)
def main(candidate_path: pathlib.Path, features_path: str) -> None:
    """Run the validation harness against one candidate module and print the result."""
    candidate = load_candidate_module(candidate_path)
    frame = load_features(features_path)
    report = validate_candidate(candidate, frame)
    click.echo(
        f"{candidate.NAME}: OK — columns={report.columns} "
        f"nan_rates={report.nan_rates} axis_present={report.axis_present}"
    )


if __name__ == "__main__":
    main()
