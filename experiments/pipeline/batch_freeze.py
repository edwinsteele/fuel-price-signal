"""Batch freeze — pin a batch's data + feature-column contract for reproducible nightly runs.

Freezes, once per batch:
  - data/features.parquet -> <batch_dir>/features.parquet. load_features() prefers a
    .parquet sibling when the CSV is absent, so downstream code just calls
    load_features(batch_dir / "features.csv") and transparently gets the frozen snapshot.
  - the live SQLite DB -> <batch_dir>/fuel_signal.db, via an APFS clone (cp -c) so
    pinning the 500MB+ DB is near-instant and near-zero disk. run_paired_realised_backtest
    replays economics out of the live DB (experiments/lib/realised.py:210), so an
    unpinned DB would silently un-freeze the batch.
  - the resolved baseline feature-column list -> <batch_dir>/baseline_columns.json
  - a freeze manifest (snapshot date, git SHA, source paths) -> <batch_dir>/freeze.json

DB refresh (the `make update` chain: pull, db, fill, classify, lga-leadership), followed
by a features.csv/.parquet regeneration (`make features`), is a hard-gated pre-step:
refresh_db() raises loudly if either step fails rather than let freeze_batch() pin a
stale DB or a features.csv/.parquet that predates it (fps-3vo — previously the DB was
hard-gated but features.csv/.parquet was a separate, unchecked manual step, so a batch
could silently freeze months-old features against a freshly-pulled DB). It moves out of
the nightly path entirely — freeze happens once at batch setup; nights 2..N deliberately
run on day-0 data.

Usage (batch setup, once, before the launch routine claims candidate 1):
  PYTHONPATH=. uv run python experiments/pipeline/batch_freeze.py <batch-name>
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone

import click
import numpy as np
import pandas as pd

from fuel_signal.dates import date_from_int
from fuel_signal.db import DEFAULT_DB_PATH
from fuel_signal.features import (
    DEFAULT_FEATURES_CSV,
    LOCKED_FEATURE_COLUMNS,
    baseline_fingerprint,
    load_features,
    non_model_columns,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_BATCHES_DIR = REPO_ROOT / "experiments" / "batches"

BASELINE_COLUMNS_FILENAME = "baseline_columns.json"
FREEZE_MANIFEST_FILENAME = "freeze.json"
FROZEN_FEATURES_FILENAME = "features.parquet"
FROZEN_DB_FILENAME = "fuel_signal.db"


class DbRefreshError(RuntimeError):
    """The hard-gated DB refresh (`make update`) or features regen (`make features`)
    failed; batch setup must abort."""


class BaselineContractMismatch(RuntimeError):
    """A run's resolved baseline columns no longer match the frozen batch's.

    Maps to the runner's aborted_environment status, not aborted_candidate — the
    candidate is fine, the environment moved under it (fps-3jj.1: "pin the feature
    contract, not just the data").
    """

    def __init__(self, added: list[str], removed: list[str]) -> None:
        self.added = added
        self.removed = removed
        super().__init__(
            f"Baseline feature-column contract changed since freeze: +{added} -{removed}"
        )


class NonModelColumnLeak(RuntimeError):
    """A column the project deliberately excludes from the lock reached the baseline.

    Raised by resolve_baseline_columns() rather than returned quietly: this is the
    exact shape of fps-sa1 (10 rejected Phase 4b brand-trough columns in R0), and it
    went unnoticed for two months because nothing asserted the negative.
    """

    def __init__(self, leaked: dict[str, tuple[str, str]]) -> None:
        self.leaked = leaked
        detail = "; ".join(f"{c} ({code}: {why})" for c, (code, why) in sorted(leaked.items()))
        super().__init__(
            f"{len(leaked)} non-model column(s) leaked into the locked baseline: {detail}"
        )


def resolve_baseline_columns(df: pd.DataFrame) -> list[str]:
    """The locked production baseline column set, in production order. 54 columns.

    Thin wrapper over ``fuel_signal.features.LOCKED_FEATURE_COLUMNS`` — the single
    symbol for the contract (fps-zci) — plus the two frame-side assertions that only
    make sense once a frame is in hand.

    DECLARED, never discovered (fps-sa1). The features frame is deliberately a
    SUPERSET of the model contract — a column sits in features.csv for one of three
    reasons, and only the first puts it in the baseline:

      * in the lock — FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS;
      * evaluated and REJECTED — the 10 ``days_since_trough_entry_<brand>`` columns
        (Phase 4b, walked away 2026-06-02: lost 9/14 folds in paired WFCV with a
        non-shock regression at fold 11 — docs/STATUS.md § Phase 4b; AGENTS.md
        § Canonical feature set: "Brand trough columns are excluded from the locked
        baseline until a separate ablation graduates them");
      * held out pending graduation — ``tgp_delta_7d`` (#271).

    This used to append ``discover_brand_feature_columns(df)``, on the stated
    rationale that it "mirrors train_lgbm.py's default resolution". It does — but
    train_lgbm's *default* is not the lock; the lock is
    ``train_lgbm --no-brand-features`` (AGENTS.md § Canonical feature set,
    docs/STATUS.md § CLI). The result was a 64-column R0: production plus a feature
    group the project had already rejected, silently applied to every candidate in
    the batch. Discovery cannot tell "not yet in scope" from "in scope", so it is
    the wrong mechanism here whatever the frame happens to contain.

    ORDER IS PART OF THE CONTRACT, so this returns the columns in production order
    and does NOT sort them (fps-zci). LightGBM breaks equal-gain split ties by
    feature index, and with 35 near-identical LGA trough columns exact ties are
    common — so the same 54 columns in a different order fit a DIFFERENT model.
    Measured: sorted vs production order, same seed, changes 732/47,823 val rows'
    probabilities (max |dp| 0.090) and moved batch0's pooled realised delta by
    0.038 c/L, ~0.8 of a buy/wait decision flip.

    The sorting this replaces was justified as "stable regardless of column
    insertion order", which is backwards: it makes the on-disk contract diff-stable
    while making the MODEL unstable to feature additions, since one
    alphabetically-early insertion reshuffles every index after it. Production
    order is append-only, so a new group leaves existing indices untouched.

    Production order is authoritative because it is what the locked artifact was
    trained in — ``joblib.load("data/models/lgbm_calibrated.joblib")["feature_columns"]``
    equals this list exactly, and differs from its sorted permutation in 52 of 54
    positions.

    Raises ValueError if the frame is missing a locked column, and NonModelColumnLeak
    if a computed-but-excluded column (see features.non_model_columns) somehow reached
    the resolved list — the assertion whose absence let fps-sa1 run for two months.
    """
    resolved = list(LOCKED_FEATURE_COLUMNS)
    missing = [c for c in resolved if c not in df.columns]
    if missing:
        raise ValueError(
            f"Features frame is missing {len(missing)} locked baseline column(s): "
            f"{missing}. Regenerate features.csv/.parquet before freezing a batch."
        )
    excluded = non_model_columns(df)
    leaked = {c: excluded[c] for c in resolved if c in excluded}
    if leaked:
        raise NonModelColumnLeak(leaked)
    return resolved


def _snapshot_date(df: pd.DataFrame) -> str:
    """The batch's data date, as 'YYYY-MM-DD', from features.csv's max price_date.

    price_date's on-disk representation has varied across the codebase (INTEGER
    YYYYMMDD in the DB, ISO date strings in some CSV outputs) — handle both rather
    than assume one.
    """
    raw = df["price_date"].max()
    if isinstance(raw, (int, np.integer)):
        return date_from_int(int(raw))
    return str(raw)[:10]


def refresh_db(repo_root: pathlib.Path = REPO_ROOT) -> None:
    """Run the `make update` chain (pull, db, fill, classify, lga-leadership), then
    regenerate features.csv/.parquet from the refreshed DB (`make features`).

    Hard-gated: raises DbRefreshError if either step exits non-zero, rather than let a
    batch freeze on a stale DB or on a features.csv/.parquet that predates it (fps-3vo).
    """
    result = subprocess.run(
        ["make", "update"], cwd=repo_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise DbRefreshError(
            f"`make update` failed (exit {result.returncode}); aborting batch setup "
            f"rather than freeze stale data.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    result = subprocess.run(
        ["make", "features"], cwd=repo_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise DbRefreshError(
            f"`make features` failed (exit {result.returncode}) after a successful DB "
            f"refresh; aborting batch setup rather than freeze a features.csv/.parquet "
            f"that predates the refreshed DB.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _clone_db(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy the SQLite DB into the batch dir, preferring an APFS clone (cp -c).

    cp -c is near-instant and near-zero disk on APFS (copy-on-write). Falls back to a
    plain copy on filesystems that don't support cloning — correctness never depends
    on the clone, only speed/disk usage.
    """
    result = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"[batch_freeze] cp -c failed ({result.stderr.strip()}); "
            "falling back to a plain copy (slower, uses full disk)."
        )
        shutil.copy2(src, dst)


def freeze_batch(
    batch_name: str,
    *,
    features_path: pathlib.Path = DEFAULT_FEATURES_CSV,
    db_path: pathlib.Path = DEFAULT_DB_PATH,
    batches_dir: pathlib.Path = DEFAULT_BATCHES_DIR,
    skip_refresh: bool = False,
) -> pathlib.Path:
    """Freeze one batch: refresh (hard-gated) -> snapshot data + DB -> pin the column contract.

    Returns the batch directory. Raises FileExistsError if the batch dir already
    exists — a batch is frozen once; re-freezing an existing batch is very likely a
    mistake, not an update.
    """
    if not skip_refresh:
        refresh_db()

    features_path = pathlib.Path(features_path)
    parquet_src = features_path.with_suffix(".parquet")
    if not parquet_src.exists():
        raise FileNotFoundError(
            f"No features snapshot at {parquet_src}. Run "
            "'uv run python -m fuel_signal.features' first."
        )
    db_path = pathlib.Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No DB to freeze at {db_path}.")

    batch_dir = pathlib.Path(batches_dir) / batch_name
    if batch_dir.exists():
        raise FileExistsError(f"Batch dir already exists: {batch_dir}. Batches are frozen once.")
    batch_dir.mkdir(parents=True)

    shutil.copy2(parquet_src, batch_dir / FROZEN_FEATURES_FILENAME)
    _clone_db(db_path, batch_dir / FROZEN_DB_FILENAME)

    df = load_features(batch_dir / "features.csv")
    baseline_columns = resolve_baseline_columns(df)
    (batch_dir / BASELINE_COLUMNS_FILENAME).write_text(
        json.dumps(baseline_columns, indent=2) + "\n"
    )

    snapshot_date = _snapshot_date(df)
    manifest = {
        "batch": batch_name,
        "snapshot_date": snapshot_date,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_features": str(parquet_src),
        "source_db": str(db_path),
        "n_baseline_columns": len(baseline_columns),
        # Identity of the ORDERED baseline this batch pinned (fps-zci item 5). Every
        # run in the batch stamps the same value into its results.json, so two runs
        # can be checked for commensurability mechanically instead of by eye.
        "baseline_fingerprint": baseline_fingerprint(baseline_columns),
    }
    (batch_dir / FREEZE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")

    return batch_dir


def check_baseline_contract(batch_dir: pathlib.Path, df: pd.DataFrame) -> None:
    """Re-resolve the baseline columns for `df` and compare against the frozen batch's.

    Called at the start of every run (fps-3jj.4, against the batch's own frozen
    features DataFrame). The locked column set is read fresh from the current
    fuel_signal.features import (today's code), so this fires on a genuine code-side
    contract change — e.g. a FEATURE_COLUMNS bump landing mid-batch. Raises
    BaselineContractMismatch on drift; the runner maps that to aborted_environment.
    """
    batch_dir = pathlib.Path(batch_dir)
    frozen = json.loads((batch_dir / BASELINE_COLUMNS_FILENAME).read_text())
    current = resolve_baseline_columns(df)
    if current == frozen:
        return
    frozen_set, current_set = set(frozen), set(current)
    raise BaselineContractMismatch(
        added=sorted(current_set - frozen_set),
        removed=sorted(frozen_set - current_set),
    )


@click.command("batch_freeze")
@click.argument("batch_name")
@click.option(
    "--features-path", default=str(DEFAULT_FEATURES_CSV), show_default=True,
    help="Source features CSV (its .parquet sibling is what actually gets frozen).",
)
@click.option(
    "--db-path", default=str(DEFAULT_DB_PATH), show_default=True,
    help="Source SQLite DB to pin.",
)
@click.option(
    "--batches-dir", default=str(DEFAULT_BATCHES_DIR), show_default=True,
    help="Parent directory for batch snapshots.",
)
@click.option(
    "--skip-refresh", is_flag=True, default=False,
    help="Skip the hard-gated `make update` DB refresh (for local testing only).",
)
def main(batch_name: str, features_path: str, db_path: str, batches_dir: str, skip_refresh: bool) -> None:
    """Freeze a batch: refresh the DB, snapshot data/features + the DB, pin the baseline contract."""
    batch_dir = freeze_batch(
        batch_name,
        features_path=pathlib.Path(features_path),
        db_path=pathlib.Path(db_path),
        batches_dir=pathlib.Path(batches_dir),
        skip_refresh=skip_refresh,
    )
    click.echo(f"Froze batch '{batch_name}' -> {batch_dir}")


if __name__ == "__main__":
    main()
