"""Tests for experiments/pipeline/batch_freeze — batch data + column-contract pinning."""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from experiments.pipeline.batch_freeze import (
    BaselineContractMismatch,
    DbRefreshError,
    check_baseline_contract,
    freeze_batch,
    refresh_db,
    resolve_baseline_columns,
)
from fuel_signal.dates import date_from_int
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, NETWORK_FEATURE_COLUMNS


def _features_df(brand_cols: list[str] | None = None) -> pd.DataFrame:
    n = 5
    cols = {"price_date": [20260810, 20260811, 20260812, 20260813, 20260814]}
    for c in FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS:
        cols[c] = [0.0] * n
    for c in brand_cols or ["days_since_trough_entry_zzz_test_brand"]:
        cols[c] = [0.0] * n
    return pd.DataFrame(cols)


def _write_source(tmp_path: pathlib.Path, df: pd.DataFrame) -> tuple[pathlib.Path, pathlib.Path]:
    features_path = tmp_path / "data" / "features.csv"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path.with_suffix(".parquet"))
    db_path = tmp_path / "fuel_signal.db"
    db_path.write_bytes(b"not a real sqlite file, just needs to exist")
    return features_path, db_path


def test_resolve_baseline_columns_is_sorted_and_includes_brand_columns():
    df = _features_df()
    resolved = resolve_baseline_columns(df)
    assert resolved == sorted(resolved)
    assert "days_since_trough_entry_zzz_test_brand" in resolved
    assert set(FEATURE_COLUMNS) <= set(resolved)


def test_resolve_baseline_columns_excludes_tgp_and_unrelated_columns():
    df = _features_df()
    df["tgp_delta_7d"] = 0.0
    df["label"] = 0
    resolved = resolve_baseline_columns(df)
    assert "tgp_delta_7d" not in resolved
    assert "label" not in resolved


def test_freeze_batch_writes_snapshot_and_manifest(tmp_path):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batches_dir = tmp_path / "batches"

    batch_dir = freeze_batch(
        "batch1",
        features_path=features_path,
        db_path=db_path,
        batches_dir=batches_dir,
        skip_refresh=True,
    )

    assert batch_dir == batches_dir / "batch1"
    assert (batch_dir / "features.parquet").exists()
    assert (batch_dir / "fuel_signal.db").exists()

    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    assert baseline_columns == resolve_baseline_columns(df)

    manifest = json.loads((batch_dir / "freeze.json").read_text())
    assert manifest["batch"] == "batch1"
    assert manifest["snapshot_date"] == date_from_int(20260814)
    assert manifest["n_baseline_columns"] == len(baseline_columns)


def test_freeze_batch_handles_iso_string_price_date(tmp_path):
    """price_date is ISO-string in some features.csv outputs, INTEGER YYYYMMDD in the DB."""
    df = _features_df()
    df["price_date"] = ["2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14"]
    features_path, db_path = _write_source(tmp_path, df)

    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True,
    )
    manifest = json.loads((batch_dir / "freeze.json").read_text())
    assert manifest["snapshot_date"] == "2026-05-14"


def test_freeze_batch_refuses_to_refreeze_existing_batch(tmp_path):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batches_dir = tmp_path / "batches"

    freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=batches_dir, skip_refresh=True,
    )
    with pytest.raises(FileExistsError):
        freeze_batch(
            "batch1", features_path=features_path, db_path=db_path,
            batches_dir=batches_dir, skip_refresh=True,
        )


def test_freeze_batch_requires_source_features(tmp_path):
    with pytest.raises(FileNotFoundError):
        freeze_batch(
            "batch1",
            features_path=tmp_path / "data" / "features.csv",
            db_path=tmp_path / "fuel_signal.db",
            batches_dir=tmp_path / "batches",
            skip_refresh=True,
        )


def test_freeze_batch_requires_source_db(tmp_path):
    df = _features_df()
    features_path = tmp_path / "data" / "features.csv"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path.with_suffix(".parquet"))

    with pytest.raises(FileNotFoundError):
        freeze_batch(
            "batch1",
            features_path=features_path,
            db_path=tmp_path / "fuel_signal.db",
            batches_dir=tmp_path / "batches",
            skip_refresh=True,
        )


def test_check_baseline_contract_passes_when_unchanged(tmp_path):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True,
    )
    check_baseline_contract(batch_dir, df)  # must not raise


def test_check_baseline_contract_flags_added_column(tmp_path):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True,
    )
    drifted = df.copy()
    drifted["days_since_trough_entry_new_brand"] = 0.0

    with pytest.raises(BaselineContractMismatch) as exc_info:
        check_baseline_contract(batch_dir, drifted)
    assert exc_info.value.added == ["days_since_trough_entry_new_brand"]
    assert exc_info.value.removed == []


def test_check_baseline_contract_flags_removed_column(tmp_path):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True,
    )
    drifted = df.drop(columns=["days_since_trough_entry_zzz_test_brand"])

    with pytest.raises(BaselineContractMismatch) as exc_info:
        check_baseline_contract(batch_dir, drifted)
    assert exc_info.value.added == []
    assert exc_info.value.removed == ["days_since_trough_entry_zzz_test_brand"]


def _write_fake_make(
    repo_root: pathlib.Path, exit_code: int, *, features_exit_code: int = 0
) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    makefile = repo_root / "Makefile"
    makefile.write_text(
        f"update:\n\t@exit {exit_code}\n"
        f"features:\n\t@exit {features_exit_code}\n"
    )


def test_refresh_db_raises_on_failed_update(tmp_path):
    repo_root = tmp_path / "repo"
    _write_fake_make(repo_root, exit_code=1)
    with pytest.raises(DbRefreshError):
        refresh_db(repo_root=repo_root)


def test_refresh_db_raises_on_failed_features_regen(tmp_path):
    """A successful DB refresh followed by a failed features regen must still abort —
    otherwise freeze_batch() would silently pin a features.csv/.parquet that predates
    the just-refreshed DB (fps-3vo)."""
    repo_root = tmp_path / "repo"
    _write_fake_make(repo_root, exit_code=0, features_exit_code=1)
    with pytest.raises(DbRefreshError):
        refresh_db(repo_root=repo_root)


def test_refresh_db_succeeds_on_clean_update_and_features_regen(tmp_path):
    repo_root = tmp_path / "repo"
    _write_fake_make(repo_root, exit_code=0)
    refresh_db(repo_root=repo_root)  # must not raise
