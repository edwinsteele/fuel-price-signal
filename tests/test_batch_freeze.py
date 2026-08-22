"""Tests for experiments/pipeline/batch_freeze — batch data + column-contract pinning."""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from experiments.pipeline import batch_freeze
from experiments.pipeline.batch_freeze import (
    BaselineContractMismatch,
    DbRefreshError,
    NonModelColumnLeak,
    UnclassifiedFrameColumns,
    check_baseline_contract,
    freeze_batch,
    refresh_db,
    resolve_baseline_columns,
)
from fuel_signal.dates import date_from_int
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    LOCKED_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    baseline_fingerprint,
)


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


def test_resolve_baseline_columns_excludes_brand_columns():
    """Brand-trough columns are computed into features.csv but are NOT in the lock.

    Phase 4b was evaluated and walked away on 2026-06-02 (docs/STATUS.md § Phase 4b;
    AGENTS.md § Canonical feature set). Resolving them in by discovery gave batch0 a
    64-column R0 — production plus a rejected feature group — and silently graded
    every candidate in that batch against it (fps-sa1). Declared, never discovered.
    """
    df = _features_df()
    resolved = resolve_baseline_columns(df)
    assert "days_since_trough_entry_zzz_test_brand" not in resolved
    assert set(FEATURE_COLUMNS) <= set(resolved)
    assert set(resolved) == set(
        FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    )


def test_resolve_baseline_columns_is_in_production_order_not_sorted():
    """Order is part of the contract (fps-zci).

    LightGBM breaks equal-gain split ties by feature index, so the same 54 columns
    in a different order fit a different model — measured at 732/47,823 val rows'
    probabilities changing, and 0.038 c/L on batch0's pooled realised delta. The
    authoritative order is the one the locked artifact was trained in, which is
    the group concatenation below, NOT its sorted permutation (they differ in 52
    of 54 positions).
    """
    resolved = resolve_baseline_columns(_features_df())
    assert resolved == FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    assert resolved != sorted(resolved), (
        "sorted() reshuffles 52 of 54 indices away from the locked artifact's order"
    )


def test_resolve_baseline_columns_matches_the_locked_54_feat_baseline():
    assert len(resolve_baseline_columns(_features_df())) == 54


def test_resolve_baseline_columns_raises_when_a_locked_column_is_absent():
    df = _features_df().drop(columns=[NETWORK_FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing 1 locked baseline column"):
        resolve_baseline_columns(df)


def test_resolve_baseline_columns_excludes_tgp_and_unrelated_columns():
    df = _features_df()
    df["tgp_delta_7d"] = 0.0
    df["label"] = 0
    resolved = resolve_baseline_columns(df)
    assert "tgp_delta_7d" not in resolved
    assert "label" not in resolved


def test_resolve_baseline_columns_is_the_single_locked_symbol():
    """No second copy of the composition here — batch_freeze imports the contract.

    Retyping `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS` at
    every call site is how a script written before #216 silently carried a 50-column
    baseline; fps-zci binds it to one name.
    """
    assert resolve_baseline_columns(_features_df()) == LOCKED_FEATURE_COLUMNS


def test_resolve_baseline_columns_raises_when_a_non_model_column_leaks_in(monkeypatch):
    """The assertion whose absence let fps-sa1 run for two months.

    Simulated code-side: a rejected brand-trough column appearing in the locked
    symbol itself. Whatever the route in, resolving a deliberately-excluded column
    into R0 must be loud, not silent.
    """
    leaked_col = "days_since_trough_entry_zzz_test_brand"
    monkeypatch.setattr(
        batch_freeze, "LOCKED_FEATURE_COLUMNS", LOCKED_FEATURE_COLUMNS + [leaked_col]
    )
    with pytest.raises(NonModelColumnLeak) as exc_info:
        resolve_baseline_columns(_features_df())
    assert leaked_col in exc_info.value.leaked
    assert "evaluated-and-rejected" in str(exc_info.value)


def test_resolve_baseline_columns_raises_when_the_LOCK_ITSELF_is_corrupted(monkeypatch):
    """The realistic corruption: a features.py edit, so BOTH references move.

    The test above patches only batch_freeze's reference, which leaves
    fuel_signal.features' copy pristine — so the detector would flag the leak even if
    it deferred to the lock. Patching both is what an actual bad edit looks like, and
    is the case that used to slip through: non_model_columns() skipped anything the
    lock already claimed, which is precisely the thing that was wrong in fps-sa1.
    """
    import fuel_signal.features as feats

    leaked_col = "days_since_trough_entry_zzz_test_brand"
    corrupted = LOCKED_FEATURE_COLUMNS + [leaked_col]
    monkeypatch.setattr(feats, "LOCKED_FEATURE_COLUMNS", corrupted)
    monkeypatch.setattr(batch_freeze, "LOCKED_FEATURE_COLUMNS", corrupted)

    with pytest.raises(NonModelColumnLeak) as exc_info:
        resolve_baseline_columns(_features_df())
    assert leaked_col in exc_info.value.leaked


def test_freeze_manifest_records_the_baseline_fingerprint(tmp_path):
    """A batch's freeze.json says which R0 it pinned, in a form runs can be checked
    against mechanically (fps-zci item 5)."""
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )
    manifest = json.loads((batch_dir / "freeze.json").read_text())
    assert manifest["baseline_fingerprint"] == baseline_fingerprint(LOCKED_FEATURE_COLUMNS)
    assert manifest["baseline_fingerprint"].startswith(f"{manifest['n_baseline_columns']}:")


def test_freeze_manifest_records_the_default_tank_params(tmp_path):
    """fps-15c: freeze.json declares the batch's cadence contract the same way
    it declares baseline_fingerprint — the identity every run in the batch is
    expected to agree with."""
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )
    manifest = json.loads((batch_dir / "freeze.json").read_text())
    assert manifest["tank_params"] == "50/3.571/1d/10%"


def test_freeze_batch_passes_its_tank_through_to_the_noise_floor(tmp_path, monkeypatch):
    """The noise floor must be computed at the SAME tank freeze.json declares —
    two independently-constructed TankParams() only agree by luck; passing the
    one object through is what actually guarantees it."""
    import experiments.pipeline.noise_floor as noise_floor_module
    from fuel_signal.backtest import TankParams

    calls: list[TankParams] = []
    monkeypatch.setattr(
        noise_floor_module, "compute_noise_floor", lambda batch_dir, **kw: calls.append(kw.get("tank"))
    )

    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    tank = TankParams(evaluation_interval_days=7)
    freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, tank=tank,
    )

    assert calls[-1] == tank


def test_freeze_batch_writes_snapshot_and_manifest(tmp_path):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batches_dir = tmp_path / "batches"

    batch_dir = freeze_batch(
        "batch1",
        features_path=features_path,
        db_path=db_path,
        batches_dir=batches_dir,
        skip_refresh=True, skip_noise_floor=True,
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


def test_freeze_batch_refuses_with_a_recovery_hint_when_noise_floor_is_missing(tmp_path):
    """fps-cf8: the noise floor is the final freeze step (~10-25min), so a crash/
    interrupt partway through leaves everything else already pinned but no
    noise_floor.json. Re-running batch_freeze on that dir must not just say 'frozen
    once' as though this were an ordinary refreeze mistake — point at the actual
    (safe, no --force) recovery."""
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batches_dir = tmp_path / "batches"
    freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=batches_dir, skip_refresh=True, skip_noise_floor=True,
    )

    with pytest.raises(FileExistsError, match="noise_floor batch1"):
        freeze_batch(
            "batch1", features_path=features_path, db_path=db_path,
            batches_dir=batches_dir, skip_refresh=True, skip_noise_floor=True,
        )


def test_freeze_batch_refuses_without_the_hint_when_noise_floor_already_exists(tmp_path):
    """A batch that already has its noise_floor.json is a genuine full refreeze
    attempt, not a crash-recovery case — no recovery hint to show."""
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batches_dir = tmp_path / "batches"
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=batches_dir, skip_refresh=True, skip_noise_floor=True,
    )
    (batch_dir / "noise_floor.json").write_text("{}")

    with pytest.raises(FileExistsError) as exc_info:
        freeze_batch(
            "batch1", features_path=features_path, db_path=db_path,
            batches_dir=batches_dir, skip_refresh=True, skip_noise_floor=True,
        )
    assert "noise_floor batch1" not in str(exc_info.value)


def test_freeze_batch_computes_the_noise_floor_as_its_final_step(tmp_path, monkeypatch):
    """fps-cf8: batch setup is one command — the noise floor is no longer a second
    step the operator must remember to run, in order, after freeze."""
    import experiments.pipeline.noise_floor as noise_floor_module

    calls: list[pathlib.Path] = []
    monkeypatch.setattr(
        noise_floor_module, "compute_noise_floor", lambda batch_dir, **kw: calls.append(batch_dir)
    )

    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True,
    )

    assert calls == [batch_dir]


def test_freeze_batch_skip_noise_floor_leaves_batch_correctly_ungraded(tmp_path):
    """The --skip-noise-floor escape hatch must not produce a silently-ungraded
    batch: no noise_floor.json means the dossier refuses to grade, not that it
    grades against nothing."""
    import experiments.pipeline.dossier_tables as dt

    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )

    assert not (batch_dir / "noise_floor.json").exists()
    band = dt._noise_band({"effect_delta_cpl_held": 5.0, "meta": {}}, batch_dir)
    assert band["available"] is False


def test_freeze_batch_handles_iso_string_price_date(tmp_path):
    """price_date is ISO-string in some features.csv outputs, INTEGER YYYYMMDD in the DB."""
    df = _features_df()
    df["price_date"] = ["2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14"]
    features_path, db_path = _write_source(tmp_path, df)

    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )
    manifest = json.loads((batch_dir / "freeze.json").read_text())
    assert manifest["snapshot_date"] == "2026-05-14"


def test_freeze_batch_refuses_to_refreeze_existing_batch(tmp_path):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batches_dir = tmp_path / "batches"

    freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=batches_dir, skip_refresh=True, skip_noise_floor=True,
    )
    with pytest.raises(FileExistsError):
        freeze_batch(
            "batch1", features_path=features_path, db_path=db_path,
            batches_dir=batches_dir, skip_refresh=True, skip_noise_floor=True,
        )


def test_freeze_batch_requires_source_features(tmp_path):
    with pytest.raises(FileNotFoundError):
        freeze_batch(
            "batch1",
            features_path=tmp_path / "data" / "features.csv",
            db_path=tmp_path / "fuel_signal.db",
            batches_dir=tmp_path / "batches",
            skip_refresh=True, skip_noise_floor=True,
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
            skip_refresh=True, skip_noise_floor=True,
        )


def test_check_baseline_contract_passes_when_unchanged(tmp_path):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )
    check_baseline_contract(batch_dir, df)  # must not raise


def test_check_baseline_contract_ignores_a_new_brand_column_in_the_frame(tmp_path):
    """The regression this whole change exists to prevent (fps-sa1).

    A brand-trough column appearing in the features header is not a contract change
    — the lock never contained that group. Before the fix it silently WAS one, and
    the baseline grew by 10 columns without anything flagging it.
    """
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )
    widened = df.copy()
    widened["days_since_trough_entry_new_brand"] = 0.0

    check_baseline_contract(batch_dir, widened)  # must not raise


def test_check_baseline_contract_flags_added_column(tmp_path, monkeypatch):
    """Drift is now code-side only: a FEATURE_COLUMNS-family bump landing mid-batch."""
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )
    monkeypatch.setattr(
        batch_freeze, "LOCKED_FEATURE_COLUMNS", LOCKED_FEATURE_COLUMNS + ["new_locked_col"]
    )
    widened = df.copy()
    widened["new_locked_col"] = 0.0

    with pytest.raises(BaselineContractMismatch) as exc_info:
        check_baseline_contract(batch_dir, widened)
    assert exc_info.value.added == ["new_locked_col"]
    assert exc_info.value.removed == []


def test_check_baseline_contract_flags_removed_column(tmp_path, monkeypatch):
    df = _features_df()
    features_path, db_path = _write_source(tmp_path, df)
    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )
    dropped = LOCKED_FEATURE_COLUMNS[-1]
    monkeypatch.setattr(
        batch_freeze, "LOCKED_FEATURE_COLUMNS", LOCKED_FEATURE_COLUMNS[:-1]
    )

    with pytest.raises(BaselineContractMismatch) as exc_info:
        check_baseline_contract(batch_dir, df)
    assert exc_info.value.added == []
    assert exc_info.value.removed == [dropped]


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


def test_freeze_batch_refuses_an_unclassified_frame_column(tmp_path):
    """The forcing function that keeps TARGET_COLUMNS' blocklist honest.

    A blocklist of label columns only stays complete while adding a NEW column to the
    frame makes someone declare what it is. Freezing is where that gets enforced, since
    it is the one step that reads the real frame before any candidate runs.
    """
    df = _features_df()
    df["future_max_cents"] = [0.0] * len(df)
    features_path, db_path = _write_source(tmp_path, df)

    with pytest.raises(UnclassifiedFrameColumns) as excinfo:
        freeze_batch(
            "batch1", features_path=features_path, db_path=db_path,
            batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
        )

    assert "future_max_cents" in str(excinfo.value)


def test_freeze_batch_accepts_a_frame_carrying_the_label_columns(tmp_path):
    """The real frame always carries future_min_cents/label/today_price_cents — the
    freeze gate must classify them, not trip over them."""
    df = _features_df()
    for col in ("station_code", "today_price_cents", "future_min_cents", "label"):
        df[col] = [0.0] * len(df)
    features_path, db_path = _write_source(tmp_path, df)

    batch_dir = freeze_batch(
        "batch1", features_path=features_path, db_path=db_path,
        batches_dir=tmp_path / "batches", skip_refresh=True, skip_noise_floor=True,
    )

    assert (batch_dir / "freeze.json").exists()
