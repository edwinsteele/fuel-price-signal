"""Candidate validation harness — the whole safety layer for LLM-authored candidate
modules, since nobody reviews the code (fps-3jj.2).

Run before any fitting, in this order:

0. **Target-column check** — `INPUTS` may not name a `features.TARGET_COLUMNS` column
   (`future_min_cents`, `label`). These are computed from prices *after* `price_date`
   and are stamped on the row, so step 2 cannot see them: truncating the frame by date
   doesn't change a value that was already written into the cell. Declaration-time and
   exact; the one hazard that has to be caught by name rather than by behaviour.
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

    try:
        restricted = frame[inputs].copy()
    except KeyError as exc:
        missing = [col for col in inputs if col not in frame.columns]
        raise MissingInputColumnsError(
            f"{candidate.NAME}: INPUTS declares columns not present in the features frame: {missing}"
        ) from exc

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
