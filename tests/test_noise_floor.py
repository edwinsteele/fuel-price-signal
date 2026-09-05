"""Tests for experiments/pipeline/noise_floor.py — the noise-floor producer
(fps-3jj.9, reworked to a placebo-column null by fps-awz).

Like test_runner.py, the real DB-backed realised backtest isn't reproduced here —
run_paired_realised_backtest is monkeypatched so these tests exercise the draw
construction and persistence logic in isolation. Placebo CONSTRUCTION itself
(axis detection, the block permutation) is tested in test_placebo.py against a
realistic multi-station fixture; here the frame is a minimal one-row-per-date
fixture whose job is only to exercise noise_floor.py's own wiring.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest
from click.testing import CliRunner

import experiments.pipeline.noise_floor as noise_floor_module
from experiments.lib.universe import station_codes_digest
from experiments.pipeline.batch_freeze import BaselineContractMismatch, resolve_baseline_columns
from experiments.pipeline.dossier_tables import (
    NOISE_FLOOR_FILENAME,
    NULL_METHOD_PLACEBO_COLUMN,
    NULL_METHOD_PLACEBO_PINNED_COLUMN,
    _noise_band,
)
from experiments.pipeline.noise_floor import (
    FIT_STABILITY_FILENAME,
    compute_fit_stability_diagnostic,
    compute_noise_floor,
    main,
)
from experiments.pipeline.placebo import (
    MIN_BLOCKS,
    PLACEBO_BLOCK_DAYS,
    PLACEBO_COLUMN_NAME,
)
from experiments.pipeline.runner import DEFAULT_INNER_FOLD_PARAMS
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    baseline_fingerprint,
)


def _baseline_features_df(n: int = MIN_BLOCKS * PLACEBO_BLOCK_DAYS) -> pd.DataFrame:
    """One row per date, single station (placebo axis-detection logic itself is covered by
    test_placebo.py's multi-station fixture; every column here is trivially market-wide
    since there's only one row per (date, station) group). A `station_code` column is still
    required — `_make_lookup_provider` (reused from runner.py for the placebo arm's
    extra_feature_provider) keys its lookup on (station_code, date) unconditionally, even
    for a single-station fixture.

    Two constraints on the column values, both from placebo.py's block-permutation
    construction (fps-d7m). They must be non-constant: a constant column has zero variance,
    which makes the self_correlation check NaN and fails the enforced bound. And they must
    not be a pure monotone ramp: a drift-dominated column is the ADVERSARIAL case for any
    time-axis reordering (see MIN_BLOCKS's own measurement table), so a ramp fixture would
    put every draw in the screen's tail and make these tests flaky for a reason that has
    nothing to do with what they assert. Each column here is a distinct short-period sawtooth
    instead, which block permutation decorrelates reliably at every seed.

    `n` is likewise sized for the construction rather than for the assertions:
    MIN_BLOCKS * PLACEBO_BLOCK_DAYS is the SMALLEST length at which `_effective_block_days`
    returns the full block, so the fixture exercises the same code path a real batch does
    instead of the short-series fallback. Derived rather than hardcoded — a bare literal
    silently becomes wrong the moment either constant moves (an earlier 720 here claimed to
    do this and actually got a 30-position block)."""
    dates = [20260801 + i for i in range(n)]
    cols = {"price_date": dates, "station_code": [12345] * n}
    for k, c in enumerate(FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS):
        cols[c] = [float((i * (k + 3) + k) % 23) for i in range(n)]
    return pd.DataFrame(cols)


def _write_batch_dir(tmp_path: pathlib.Path, df: pd.DataFrame, name: str = "batch1") -> pathlib.Path:
    batch_dir = tmp_path / name
    batch_dir.mkdir()
    df.to_parquet(batch_dir / "features.parquet")
    (batch_dir / "baseline_columns.json").write_text(json.dumps(resolve_baseline_columns(df)))
    (batch_dir / "fuel_signal.db").write_bytes(b"not a real sqlite file")
    return batch_dir


_DEFAULT_TANK_PARAMS = "50/3.571/1d/10%"
_CACHE_SENTINEL = "fake-baseline-cache"


# ── compute_noise_floor (the placebo-draw producer) ────────────────────────────

def _fake_two_arm_result(
    baseline_cpl, placebo_cpl, n_windows=5, tank_params=_DEFAULT_TANK_PARAMS, cache=_CACHE_SENTINEL,
):
    class _Fake:
        aggregate = pd.DataFrame(
            [
                {"arm": "baseline", "cpl_own": baseline_cpl, "cpl_held": baseline_cpl},
                {"arm": "placebo", "cpl_own": placebo_cpl, "cpl_held": placebo_cpl},
            ]
        )
        meta = {"n_windows": n_windows, "tank_params": tank_params}
        baseline_cache = cache

    return _Fake()


def _stub_realised_by_draw(
    monkeypatch,
    deltas: list[float],
    n_windows: int = 5,
    tank_params: str = _DEFAULT_TANK_PARAMS,
    cache: str | None = _CACHE_SENTINEL,
) -> list[dict]:
    """Call i returns baseline cpl_held=100.0, placebo cpl_held=100.0+deltas[i] — records
    every call's args/kwargs (including the `baseline_cache` it was passed) in order."""
    calls: list[dict] = []

    def _fake(arms, feature_columns, **kwargs):
        calls.append({"arms": arms, "feature_columns": feature_columns, **kwargs})
        delta = deltas[len(calls) - 1]
        return _fake_two_arm_result(100.0, 100.0 + delta, n_windows, tank_params, cache=cache)

    monkeypatch.setattr(noise_floor_module, "run_paired_realised_backtest", _fake)
    return calls


def test_compute_noise_floor_writes_expected_deltas(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [9.0, -3.0, 5.0])

    payload = compute_noise_floor(batch_dir, n_draws=3, verbose=False)

    assert payload["deltas_cpl_held"] == [9.0, -3.0, 5.0]
    assert payload["n_draws"] == 3
    assert payload["seed"] == noise_floor_module.PLACEBO_SEED_DEFAULT
    assert payload["null_method"] == NULL_METHOD_PLACEBO_COLUMN
    assert payload["fold_subset"] is None
    assert payload["partial"] is False
    on_disk = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    assert on_disk == payload


def test_compute_noise_floor_uses_two_arms_baseline_and_placebo(tmp_path, monkeypatch):
    """Every call is baseline vs baseline+placebo — the same 'add one column at a fixed
    seed' shape a real candidate run measures, not a baseline-only fit."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    calls = _stub_realised_by_draw(monkeypatch, [1.0, 2.0, 3.0])

    compute_noise_floor(batch_dir, n_draws=3, verbose=False)

    assert len(calls) == 3
    for c in calls:
        assert len(c["arms"]) == 2
        assert c["arms"][0].name == noise_floor_module.BASELINE_ARM_NAME
        assert c["arms"][0].feature_columns == baseline_columns
        assert c["arms"][1].name == noise_floor_module.PLACEBO_ARM_NAME
        assert c["arms"][1].feature_columns == [*baseline_columns, PLACEBO_COLUMN_NAME]
        assert PLACEBO_COLUMN_NAME in c["arms"][1].df.columns
        # ModelStrategy's live-replay path never reads an arm's df directly — a placebo
        # arm without an extra_feature_provider would KeyError on PLACEBO_COLUMN_NAME the
        # first time a real (non-monkeypatched) backtest runs.
        assert c["arms"][1].extra_feature_provider is not None
        assert c["arms"][0].extra_feature_provider is None
        assert c["feature_columns"] == baseline_columns
        assert c["seed"] == noise_floor_module.PLACEBO_SEED_DEFAULT
        assert c["held_tau"] is None
        assert c["collect_fills"] is False


def test_compute_noise_floor_reuses_baseline_cache_across_draws(tmp_path, monkeypatch):
    """Draw 1 has no prior cache; every later draw is passed the previous call's
    result.baseline_cache — the fps-e2l mechanism runner.py already relies on across
    nights, reused here across draws within one process."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_draw(monkeypatch, [1.0, 2.0, 3.0])

    compute_noise_floor(batch_dir, n_draws=3, verbose=False)

    assert calls[0]["baseline_cache"] is None
    assert calls[1]["baseline_cache"] == _CACHE_SENTINEL
    assert calls[2]["baseline_cache"] == _CACHE_SENTINEL


def test_compute_noise_floor_draws_distinct_columns_and_seeds(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0, 3.0, 4.0])

    payload = compute_noise_floor(batch_dir, n_draws=4, verbose=False)

    columns = [d["source_column"] for d in payload["placebo_draws"]]
    seeds = [d["block_seed"] for d in payload["placebo_draws"]]
    assert len(set(columns)) == 4
    assert len(set(seeds)) == 4


def test_compute_noise_floor_excludes_an_all_nan_locked_column_from_the_draw_pool(tmp_path, monkeypatch):
    """A locked column that's entirely NaN in the frame (found in review against real batch0
    data) must never be drawn — its placebo would be all-NaN, correlation with itself
    unverifiable, and placebo.screen_draws (called by compute_noise_floor before any
    backtest fit) skips and substitutes it rather than including it."""
    df = _baseline_features_df()
    nan_column = resolve_baseline_columns(df)[0]  # select_draws always picks index 0 for draw 1
    df[nan_column] = float("nan")
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [0.0] * 5)

    payload = compute_noise_floor(batch_dir, n_draws=5, verbose=False)

    drawn_columns = {d["source_column"] for d in payload["placebo_draws"]}
    assert nan_column not in drawn_columns


def test_compute_noise_floor_raises_if_placebo_provider_never_hits(tmp_path, monkeypatch):
    """A real (non-monkeypatched) run_paired_realised_backtest calls each arm's
    extra_feature_provider during the live replay. If every lookup misses (a
    (station_code, date) key-format mismatch), the draw's delta was computed with the
    placebo column silently NaN throughout — this must raise, not read as a legitimate
    'the column does nothing' result. Exercises the REAL _make_lookup_provider /
    _check_provider_health (only run_paired_realised_backtest itself is faked here) by
    having the fake caller invoke the provider with a station_code it can't match."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    def _fake(arms, feature_columns, **kwargs):
        placebo_arm = arms[1]
        # The fixture's only station is 12345 — an unmatched code guarantees a miss.
        placebo_arm.extra_feature_provider("2026-08-01", 99999, 100.0)
        return _fake_two_arm_result(100.0, 101.0)

    monkeypatch.setattr(noise_floor_module, "run_paired_realised_backtest", _fake)

    with pytest.raises(RuntimeError, match="missed on all"):
        compute_noise_floor(batch_dir, n_draws=1, verbose=False)


def test_compute_noise_floor_stamps_the_default_tank_params(tmp_path, monkeypatch):
    """fps-15c: the floor's deltas_cpl_held must carry the cadence they were
    produced at, read back off RealisedResult.meta (not reformatted here)."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_draw(monkeypatch, [1.0, 2.0])

    payload = compute_noise_floor(batch_dir, n_draws=2, verbose=False)

    assert payload["tank_params"] == "50/3.571/1d/10%"
    from fuel_signal.backtest import TankParams

    assert all(call["tank"] == TankParams() for call in calls)


def test_compute_noise_floor_stamps_a_non_default_tank(tmp_path, monkeypatch):
    from fuel_signal.backtest import TankParams

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    tank = TankParams(evaluation_interval_days=7)
    calls = _stub_realised_by_draw(monkeypatch, [1.0, 2.0], tank_params="50/3.571/7d/10%")

    payload = compute_noise_floor(batch_dir, n_draws=2, tank=tank, verbose=False)

    assert payload["tank_params"] == "50/3.571/7d/10%"
    assert all(call["tank"] == tank for call in calls)


# ── check_freeze_cadence: floor-vs-freeze, the third side of the stamp triangle (fps-oqz) ──

def _write_freeze(batch_dir: pathlib.Path, tank_params: str | None) -> None:
    payload = {"snapshot_date": "2026-08-01"}
    if tank_params is not None:
        payload["tank_params"] = tank_params
    (batch_dir / "freeze.json").write_text(json.dumps(payload))


def test_compute_noise_floor_refuses_a_cadence_the_batch_does_not_declare(tmp_path, monkeypatch):
    """fps-oqz: `--force` resolves tank from the CURRENT TankParams() default, so after
    a re-lock it would overwrite an old batch's floor at the new cadence while
    freeze.json keeps declaring the old one. Nothing consumes freeze.json's stamp and
    _noise_band (fps-v8o) compares only run-vs-floor, so the batch would go
    split-brained with no reader to notice."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _write_freeze(batch_dir, "50/3.571/7d/10%")
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0])

    with pytest.raises(ValueError, match="refusing"):
        compute_noise_floor(batch_dir, n_draws=2, verbose=False)


def test_compute_noise_floor_allows_the_batchs_own_declared_cadence(tmp_path, monkeypatch):
    """The refusal is about DISAGREEMENT, not about being non-default: recomputing a
    7-day batch's floor at 7 days is exactly the legitimate case."""
    from fuel_signal.backtest import TankParams

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _write_freeze(batch_dir, "50/3.571/7d/10%")
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0], tank_params="50/3.571/7d/10%")

    payload = compute_noise_floor(
        batch_dir, n_draws=2, tank=TankParams(evaluation_interval_days=7), verbose=False
    )

    assert payload["tank_params"] == "50/3.571/7d/10%"


def test_compute_noise_floor_is_silent_when_the_freeze_predates_the_stamp(tmp_path, monkeypatch):
    """A pre-fps-15c freeze.json carries no tank_params. An absent declaration is not
    a disagreement — refusing there would block every legacy batch."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _write_freeze(batch_dir, None)
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0])

    payload = compute_noise_floor(batch_dir, n_draws=2, verbose=False)

    assert payload["tank_params"] == _DEFAULT_TANK_PARAMS


def test_batch0s_real_freeze_manifest_refuses_the_current_default_cadence():
    """Regression on the actual scenario, against the committed artifact rather than a
    fixture: batch0 is a 7-day batch, the canonical cadence is now 1d, so
    `noise_floor.py batch0 --force` must refuse. Its own module docstring recommends
    --force 'after a re-lock', which is precisely the dangerous path here."""
    from experiments.pipeline.noise_floor import check_freeze_cadence
    from fuel_signal.backtest import TankParams

    with pytest.raises(ValueError, match="refusing"):
        check_freeze_cadence(pathlib.Path("experiments/batches/batch0"), TankParams())


def test_compute_noise_floor_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0])

    with pytest.raises(FileExistsError, match="already exists"):
        compute_noise_floor(batch_dir, n_draws=2, verbose=False)


def test_compute_noise_floor_overwrites_existing_file_with_force(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_draw(monkeypatch, [1.0])

    payload = compute_noise_floor(batch_dir, n_draws=1, force=True, verbose=False)

    assert payload["deltas_cpl_held"] == [1.0]


def test_compute_noise_floor_raises_on_baseline_contract_drift(tmp_path, monkeypatch):
    """The frozen baseline_columns.json is checked against the current code's own
    resolution (same call every candidate run makes), catching a mid-batch
    FEATURE_COLUMNS change rather than silently fitting the stale list."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    stale = resolve_baseline_columns(df)[:-1]  # drop one locked column -> drift
    (batch_dir / "baseline_columns.json").write_text(json.dumps(stale))
    _stub_realised_by_draw(monkeypatch, [1.0])

    with pytest.raises(BaselineContractMismatch):
        compute_noise_floor(batch_dir, n_draws=1, verbose=False)


def test_compute_noise_floor_defaults_inner_fold_params(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_draw(monkeypatch, [1.0])

    compute_noise_floor(batch_dir, n_draws=1, verbose=False)

    for c in calls:
        assert c["inner_fold_params"] == DEFAULT_INNER_FOLD_PARAMS


def test_compute_noise_floor_merges_partial_inner_fold_params(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_draw(monkeypatch, [1.0])

    compute_noise_floor(
        batch_dir, n_draws=1, inner_fold_params={"val_days": 30}, verbose=False,
    )

    for c in calls:
        assert c["inner_fold_params"]["val_days"] == 30
        assert c["inner_fold_params"]["train_min_days"] == DEFAULT_INNER_FOLD_PARAMS["train_min_days"]


def test_compute_noise_floor_stamps_provenance(tmp_path, monkeypatch):
    """fps-cf8 acceptance (still holds under the reworked producer): baseline_fingerprint,
    fold geometry, and wall_seconds must be in the written payload. fps-awz adds n_draws,
    seed, and per-draw placebo_draws provenance."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0], n_windows=7)

    payload = compute_noise_floor(batch_dir, n_draws=2, verbose=False)

    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    assert payload["baseline_fingerprint"] == baseline_fingerprint(baseline_columns)
    assert payload["n_windows"] == 7
    assert payload["outer_fold_params"] == {}
    assert payload["inner_fold_params"] == DEFAULT_INNER_FOLD_PARAMS
    assert payload["wall_seconds"] >= 0.0
    assert len(payload["placebo_draws"]) == 2
    # fps-3jj.14: one record per draw MEMBER, tagged with its 1-indexed draw. At arity 1
    # that is still one record per draw.
    assert payload["n_placebo_columns"] == 1
    for i, draw in enumerate(payload["placebo_draws"], start=1):
        assert set(draw) == {"draw", "source_column", "block_seed", "self_correlation"}
        assert draw["draw"] == i
        assert isinstance(draw["self_correlation"], float)


# ── station_population stamping (fps-916) ──────────────────────────────────────

def test_compute_noise_floor_stamps_the_five_station_default_population(tmp_path, monkeypatch):
    """station_codes was already threaded to run_paired_realised_backtest (None resolves
    to PREFERRED_STATIONS there) but never stamped into the payload — every floor
    computed so far was in fact five-station, and nothing recorded that."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0])

    payload = compute_noise_floor(batch_dir, n_draws=1, verbose=False)

    assert payload["station_population"] == station_codes_digest(PREFERRED_STATIONS)


def test_compute_noise_floor_stamps_a_digest_of_an_explicit_station_codes_list(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0])

    payload = compute_noise_floor(
        batch_dir, n_draws=1, station_codes=[101, 102, 103], verbose=False,
    )

    assert payload["station_population"] == station_codes_digest([101, 102, 103])
    assert payload["station_population"] != station_codes_digest(PREFERRED_STATIONS)


def test_compute_noise_floor_stamps_an_explicit_station_population_verbatim(tmp_path, monkeypatch):
    """The CLI's --n-stations path already has an open DB connection when it draws the
    codes, so it computes eligible_pool_digest itself and passes the result straight
    through — compute_noise_floor must not second-guess it with its own codes-only hash."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0])

    payload = compute_noise_floor(
        batch_dir, n_draws=1, station_codes=[101, 102, 103],
        station_population="410:4a9920301726", verbose=False,
    )

    assert payload["station_population"] == "410:4a9920301726"


def test_fit_stability_stamps_the_five_station_default_population(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    payload = compute_fit_stability_diagnostic(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)

    assert payload["station_population"] == station_codes_digest(PREFERRED_STATIONS)


def test_compute_noise_floor_propagates_a_screen_draws_failure(tmp_path, monkeypatch):
    """The self_correlation screen is placebo.screen_draws's job (see test_placebo.py
    for its own substitute-don't-abort behaviour and its NaN/failure-exhaustion coverage);
    compute_noise_floor must propagate a screen_draws failure rather than swallow it —
    verified here at the wiring level, not by re-deriving screen_draws's own logic."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0])

    def _always_fails(df, baseline_columns, n_draws, *, arity, max_self_correlation):
        raise ValueError("only found 0/1 placebo draws passing abs(self_correlation)")

    monkeypatch.setattr(noise_floor_module, "screen_draw_groups", _always_fails)

    with pytest.raises(ValueError, match="only found 0/1"):
        compute_noise_floor(batch_dir, n_draws=1, verbose=False)


def test_compute_noise_floor_raises_on_n_windows_mismatch_across_draws(tmp_path, monkeypatch):
    """Every draw walks the same frozen data + fold geometry; if two draws disagree on
    n_windows, something (e.g. a mid-run config change) broke that invariant and the
    resulting band is not trustworthy."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls_seen = {"n": 0}

    def _fake(arms, feature_columns, **kwargs):
        calls_seen["n"] += 1
        n_windows = 7 if calls_seen["n"] == 1 else 8
        return _fake_two_arm_result(100.0, 101.0, n_windows=n_windows)

    monkeypatch.setattr(noise_floor_module, "run_paired_realised_backtest", _fake)

    with pytest.raises(ValueError, match="fold geometry must be identical"):
        compute_noise_floor(batch_dir, n_draws=2, verbose=False)


def test_compute_noise_floor_raises_on_tank_params_mismatch_across_draws(tmp_path, monkeypatch):
    """fps-15c: every draw shares one `tank`, so disagreeing tank_params means
    run_paired_realised_backtest didn't actually run at the tank it was passed."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls_seen = {"n": 0}

    def _fake(arms, feature_columns, **kwargs):
        calls_seen["n"] += 1
        tank_params = "50/3.571/7d/10%" if calls_seen["n"] == 1 else "50/3.571/1d/10%"
        return _fake_two_arm_result(100.0, 101.0, tank_params=tank_params)

    monkeypatch.setattr(noise_floor_module, "run_paired_realised_backtest", _fake)

    with pytest.raises(ValueError, match="cadence must match"):
        compute_noise_floor(batch_dir, n_draws=2, verbose=False)


def test_compute_noise_floor_default_n_draws_is_20(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_draw(monkeypatch, [0.0] * 20)

    payload = compute_noise_floor(batch_dir, verbose=False)

    assert noise_floor_module.DEFAULT_N_DRAWS == 20
    assert len(calls) == 20
    assert len(payload["deltas_cpl_held"]) == 20


def test_compute_noise_floor_marks_fold_subset_runs_partial(tmp_path, monkeypatch):
    """--fold-subset is an iteration/smoke speed-up; the written file must say so
    rather than looking like a full calibration (see the dossier-consumption test)."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0])

    payload = compute_noise_floor(
        batch_dir, n_draws=1, fold_subset=[1, 3], verbose=False,
    )

    assert payload["partial"] is True
    assert payload["fold_subset"] == [1, 3]


def _matching_fingerprint(batch_dir: pathlib.Path) -> str:
    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    return baseline_fingerprint(baseline_columns)


def test_dossier_tables_consumes_the_written_file(tmp_path, monkeypatch):
    """End-to-end schema check: what this module writes is exactly what
    dossier_tables._noise_band() reads (the fps-3jj.9 contract, unchanged by fps-awz)."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0])
    compute_noise_floor(batch_dir, n_draws=2, verbose=False)
    fp = _matching_fingerprint(batch_dir)

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": fp, "tank_params": _DEFAULT_TANK_PARAMS}},
        batch_dir,
    )

    assert band["available"] is True
    assert band["n_draws"] == 2


def test_dossier_tables_refuses_a_partial_fold_subset_file(tmp_path, monkeypatch):
    """A --fold-subset run's output must not be silently graded as a full ruler."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0])
    compute_noise_floor(batch_dir, n_draws=1, fold_subset=[1], verbose=False)
    fp = _matching_fingerprint(batch_dir)

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": fp, "tank_params": _DEFAULT_TANK_PARAMS}},
        batch_dir,
    )

    assert band["available"] is False
    assert "partial" in band["reason"] or "fold-subset" in band["reason"]


def test_dossier_tables_refuses_a_mismatched_fingerprint(tmp_path, monkeypatch):
    """fps-cf8: a floor computed against one baseline must not silently grade a
    candidate run stamped with a different baseline_fingerprint."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0])
    compute_noise_floor(batch_dir, n_draws=1, verbose=False)

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": "54:deadbeefcafe"}},
        batch_dir,
    )

    assert band["available"] is False
    assert "fingerprint" in band["reason"]


def test_dossier_tables_refuses_a_mismatched_tank_params(tmp_path, monkeypatch):
    """fps-v8o: a floor computed at one cadence must not silently grade a candidate
    run stamped with a different tank_params — the consuming-side half of fps-15c."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0], tank_params=_DEFAULT_TANK_PARAMS)
    compute_noise_floor(batch_dir, n_draws=1, verbose=False)
    fp = _matching_fingerprint(batch_dir)

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": fp, "tank_params": "50/3.571/7d/10%"}},
        batch_dir,
    )

    assert band["available"] is False
    assert "tank_params" in band["reason"]


def test_dossier_tables_refuses_a_floor_with_no_fingerprint(tmp_path, monkeypatch):
    """A pre-fps-cf8 floor (no baseline_fingerprint at all) cannot be shown to match,
    so it must not be treated as a pass by default."""
    import experiments.pipeline.dossier_tables as dt

    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))

    band = dt._noise_band(
        {"effect_delta_cpl_held": 5.0, "meta": {"baseline_fingerprint": "54:whatever"}},
        batch_dir,
    )

    assert band["available"] is False
    assert "fingerprint" in band["reason"]


def test_cli_writes_noise_floor_for_named_batch(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    _stub_realised_by_draw(monkeypatch, [0.0] * 20)

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir)])

    assert result.exit_code == 0, result.output
    assert (batch_dir / NOISE_FLOOR_FILENAME).exists()


class _FakeFoldPlan:
    def __init__(self, val_start: str, val_end: str) -> None:
        self.val_start = val_start
        self.val_end = val_end


def test_cli_n_stations_draws_a_wide_population_and_stamps_it(tmp_path, monkeypatch):
    """--n-stations builds a UniverseSpec off the real fold windows (stubbed here — the
    fixture's price_date values aren't real dates, see _plan_folds' own tests for that
    coverage), draws station_codes through sample_station_universe, and stamps
    eligible_pool_digest's result (not a codes-only hash) as station_population."""
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    calls = _stub_realised_by_draw(monkeypatch, [0.0] * 20)
    monkeypatch.setattr(
        noise_floor_module, "_plan_folds",
        lambda frame, params, subset: [_FakeFoldPlan("2024-01-01", "2024-03-30")],
    )
    drawn = [101, 102, 103]
    spec_seen = {}

    def _fake_sample(conn, spec):
        spec_seen["spec"] = spec
        return drawn

    monkeypatch.setattr(noise_floor_module, "sample_station_universe", _fake_sample)
    monkeypatch.setattr(
        noise_floor_module, "eligible_pool_digest", lambda conn, spec: "3:cafef00dfeed"
    )

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--n-stations", "3"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    assert payload["station_population"] == "3:cafef00dfeed"
    assert all(c["station_codes"] == drawn for c in calls)
    assert spec_seen["spec"].n == 3
    assert spec_seen["spec"].seed == noise_floor_module.UNIVERSE_SEED
    assert spec_seen["spec"].start_date == "2024-01-01"
    assert spec_seen["spec"].end_date == "2024-03-30"
    assert spec_seen["spec"].windows == (("2024-01-01", "2024-03-30"),)


def test_cli_without_n_stations_leaves_station_codes_default(tmp_path, monkeypatch):
    """The default path must not touch sample_station_universe/eligible_pool_digest at
    all — no DB connection needed when nobody asked for a non-default population."""
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    calls = _stub_realised_by_draw(monkeypatch, [0.0] * 20)

    def _boom(*args, **kwargs):
        raise AssertionError("must not be called without --n-stations")

    monkeypatch.setattr(noise_floor_module, "sample_station_universe", _boom)
    monkeypatch.setattr(noise_floor_module, "eligible_pool_digest", _boom)

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir)])

    assert result.exit_code == 0, result.output
    assert all(c["station_codes"] is None for c in calls)
    payload = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    assert payload["station_population"] == station_codes_digest(PREFERRED_STATIONS)


def test_cli_n_draws_override(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    _stub_realised_by_draw(monkeypatch, [0.0] * 4)

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir), "--n-draws", "4"])

    assert result.exit_code == 0, result.output
    on_disk = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    assert on_disk["n_draws"] == 4


def test_cli_parses_fold_subset(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    _write_batch_dir(batches_dir, df)
    calls = _stub_realised_by_draw(monkeypatch, [0.0] * 20)

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--fold-subset", "1,3"]
    )

    assert result.exit_code == 0, result.output
    for c in calls:
        assert c["fold_subset"] == [1, 3]


def test_cli_rejects_non_numeric_fold_subset(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    _write_batch_dir(batches_dir, df)
    _stub_realised_by_draw(monkeypatch, [])  # must never be called

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--fold-subset", "1,abc"]
    )

    assert result.exit_code != 0
    assert "not an integer" in result.output


def test_cli_rejects_non_positive_fold_subset(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    _write_batch_dir(batches_dir, df)
    _stub_realised_by_draw(monkeypatch, [])  # must never be called

    result = CliRunner().invoke(
        main, ["batch1", "--batches-dir", str(batches_dir), "--fold-subset", "0,2"]
    )

    assert result.exit_code != 0
    assert "1-indexed" in result.output


def test_cli_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_draw(monkeypatch, [])  # must never be called

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir)])

    assert result.exit_code != 0


def test_cli_force_overwrites_existing_file(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    (batch_dir / NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_draw(monkeypatch, [0.0] * 20)

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir), "--force"])

    assert result.exit_code == 0, result.output
    on_disk = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    assert on_disk["deltas_cpl_held"] == [0.0] * 20


# ── compute_fit_stability_diagnostic (the retained seed-swap null) ─────────────
# The original fps-3jj.9 seed-pairing tests, relocated: this diagnostic is no longer the
# noise-floor gate (see noise_floor.py's module docstring), but the mechanism itself is
# unchanged, so these are the same assertions against the new function/filename.

def _fake_one_arm_result(cpl_held: float, n_windows: int = 5, tank_params: str = _DEFAULT_TANK_PARAMS):
    class _Fake:
        aggregate = pd.DataFrame([{"arm": "baseline", "cpl_own": cpl_held, "cpl_held": cpl_held}])
        meta = {"n_windows": n_windows, "tank_params": tank_params}

    return _Fake()


def _stub_realised_by_seed(
    monkeypatch,
    cpl_by_seed: dict[int, float],
    n_windows_by_seed: dict[int, int] | None = None,
    tank_params_by_seed: dict[int, str] | None = None,
) -> list[dict]:
    calls: list[dict] = []
    n_windows_by_seed = n_windows_by_seed or {}
    tank_params_by_seed = tank_params_by_seed or {}

    def _fake(arms, feature_columns, **kwargs):
        calls.append({"arms": arms, "feature_columns": feature_columns, **kwargs})
        seed = kwargs["seed"]
        return _fake_one_arm_result(
            cpl_by_seed[seed],
            n_windows_by_seed.get(seed, 5),
            tank_params_by_seed.get(seed, _DEFAULT_TANK_PARAMS),
        )

    monkeypatch.setattr(noise_floor_module, "run_paired_realised_backtest", _fake)
    return calls


def test_fit_stability_pairs_seeds_and_writes_expected_deltas(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    cpl_by_seed = {1: 100.0, 2: 200.0, 10: 109.0, 20: 218.0}
    _stub_realised_by_seed(monkeypatch, cpl_by_seed)

    payload = compute_fit_stability_diagnostic(
        batch_dir, seeds_a=(1, 2), seeds_b=(10, 20), verbose=False,
    )

    assert payload["deltas_cpl_held"] == [9.0, 18.0]
    assert payload["null_method"] == "seed_swap"
    assert payload["seeds_a"] == [1, 2]
    assert payload["seeds_b"] == [10, 20]
    on_disk = json.loads((batch_dir / FIT_STABILITY_FILENAME).read_text())
    assert on_disk == payload


def test_fit_stability_mismatched_seed_lengths_raises(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    with pytest.raises(ValueError, match="pair 1:1"):
        compute_fit_stability_diagnostic(batch_dir, seeds_a=(1, 2), seeds_b=(10,))


def test_fit_stability_repeated_seed_within_a_group_raises(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    with pytest.raises(ValueError, match="free of repeats"):
        compute_fit_stability_diagnostic(batch_dir, seeds_a=(1, 1), seeds_b=(10, 20))


def test_fit_stability_overlapping_seed_groups_raises(tmp_path):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    with pytest.raises(ValueError, match="disjoint"):
        compute_fit_stability_diagnostic(batch_dir, seeds_a=(1, 2), seeds_b=(2, 3))


def test_fit_stability_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    (batch_dir / FIT_STABILITY_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    with pytest.raises(FileExistsError, match="already exists"):
        compute_fit_stability_diagnostic(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)


def test_fit_stability_overwrites_existing_file_with_force(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    (batch_dir / FIT_STABILITY_FILENAME).write_text(json.dumps({"deltas_cpl_held": [0.01]}))
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    payload = compute_fit_stability_diagnostic(
        batch_dir, seeds_a=(1,), seeds_b=(2,), force=True, verbose=False,
    )

    assert payload["deltas_cpl_held"] == [1.0]


def test_fit_stability_calls_one_baseline_only_arm_per_seed(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0})

    compute_fit_stability_diagnostic(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)

    assert len(calls) == 2
    assert {c["seed"] for c in calls} == {1, 2}
    for c in calls:
        assert len(c["arms"]) == 1
        assert c["arms"][0].name == noise_floor_module.BASELINE_ARM_NAME
        assert c["held_tau"] is None
        assert c["collect_fills"] is False


def test_fit_stability_raises_on_n_windows_mismatch_within_pair(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(monkeypatch, {1: 1.0, 2: 2.0}, {1: 7, 2: 8})

    with pytest.raises(ValueError, match="fold geometry must match"):
        compute_fit_stability_diagnostic(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)


def test_fit_stability_raises_on_tank_params_mismatch_within_pair(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_seed(
        monkeypatch, {1: 1.0, 2: 2.0},
        tank_params_by_seed={1: "50/3.571/7d/10%", 2: "50/3.571/1d/10%"},
    )

    with pytest.raises(ValueError, match="cadence must match"):
        compute_fit_stability_diagnostic(batch_dir, seeds_a=(1,), seeds_b=(2,), verbose=False)


def test_fit_stability_default_seeds_pair_five_and_five(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    all_seeds = set(noise_floor_module.SEEDS_A) | set(noise_floor_module.SEEDS_B)
    calls = _stub_realised_by_seed(monkeypatch, dict.fromkeys(all_seeds, 0.0))

    payload = compute_fit_stability_diagnostic(batch_dir, verbose=False)

    assert noise_floor_module.SEEDS_A == (42, 43, 44, 45, 46)
    assert len(noise_floor_module.SEEDS_A) == len(noise_floor_module.SEEDS_B) == 5
    assert set(noise_floor_module.SEEDS_A).isdisjoint(noise_floor_module.SEEDS_B)
    assert len(calls) == 10
    assert len(payload["deltas_cpl_held"]) == 5


def test_fit_stability_not_called_by_compute_noise_floor(tmp_path, monkeypatch):
    """fps-awz: the retained diagnostic is opt-in, not part of the noise-floor gate's
    automatic path — a plain compute_noise_floor() call must not also write
    fit_stability.json."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0])

    compute_noise_floor(batch_dir, n_draws=1, verbose=False)

    assert not (batch_dir / FIT_STABILITY_FILENAME).exists()


def test_cli_fit_stability_flag_writes_fit_stability_file(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    _stub_realised_by_seed(
        monkeypatch,
        dict.fromkeys(set(noise_floor_module.SEEDS_A) | set(noise_floor_module.SEEDS_B), 0.0),
    )

    result = CliRunner().invoke(main, ["batch1", "--batches-dir", str(batches_dir), "--fit-stability"])

    assert result.exit_code == 0, result.output
    assert (batch_dir / FIT_STABILITY_FILENAME).exists()
    assert not (batch_dir / NOISE_FLOOR_FILENAME).exists()


# ── arity (fps-3jj.14) ────────────────────────────────────────────────────────

def test_arity_1_is_byte_identical_to_the_pre_arity_behaviour(tmp_path, monkeypatch):
    """The whole point of `placebo_column_names`' arity-1 branch: adding the parameter must
    not move an existing floor. Asserts the arm's feature list still ends in the historical
    single `__placebo__` name and the deltas are unchanged."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_draw(monkeypatch, [0.5, -0.25])

    payload = compute_noise_floor(batch_dir, n_draws=2, verbose=False)

    assert payload["n_placebo_columns"] == 1
    assert payload["deltas_cpl_held"] == pytest.approx([0.5, -0.25])
    for call in calls:
        placebo_arm = call["arms"][1]
        assert placebo_arm.feature_columns[-1:] == [PLACEBO_COLUMN_NAME]


def test_arity_3_adds_three_distinct_placebo_columns_to_the_placebo_arm(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    calls = _stub_realised_by_draw(monkeypatch, [0.1, 0.2])

    payload = compute_noise_floor(batch_dir, n_draws=2, arity=3, verbose=False)

    assert payload["n_placebo_columns"] == 3
    assert payload["n_draws"] == 2
    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    for call in calls:
        baseline_arm, placebo_arm = call["arms"]
        assert baseline_arm.feature_columns == baseline_columns
        added = placebo_arm.feature_columns[len(baseline_columns):]
        assert added == ["__placebo_1__", "__placebo_2__", "__placebo_3__"]
        # Every added column must actually be present and non-degenerate in the arm's frame,
        # not merely declared — a declared-but-absent column would KeyError at fit time in
        # the real path and silently pass a shape-only assertion here.
        for col in added:
            assert placebo_arm.df[col].notna().any()
            assert placebo_arm.df[col].nunique() > 1


def test_arity_3_uses_a_distinct_source_column_for_every_member_of_every_draw(tmp_path, monkeypatch):
    """n_draws * arity distinct columns, not n_draws columns reused three times — the whole
    bank must be distinct or the band under-samples the column space it claims to cover."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [0.1, 0.2, 0.3])

    payload = compute_noise_floor(batch_dir, n_draws=3, arity=3, verbose=False)

    records = payload["placebo_draws"]
    assert len(records) == 9
    assert [r["draw"] for r in records] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    sources = [r["source_column"] for r in records]
    assert len(set(sources)) == 9
    # Members of one draw get distinct block seeds on this fixture, where no substitution
    # fires. Not asserted as a universal invariant: `candidate_pool`'s fallback tail cycles
    # the seed pool from index 0, so a substituted draw CAN reuse a seed — pre-existing
    # behaviour visible in batch1's own committed arity-1 floor (seeds 97 and 101 twice).
    for draw_i in (1, 2, 3):
        seeds = [r["block_seed"] for r in records if r["draw"] == draw_i]
        assert len(set(seeds)) == 3


def test_arity_below_1_is_rejected(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [0.1])
    with pytest.raises(ValueError, match="arity must be >= 1"):
        compute_noise_floor(batch_dir, n_draws=1, arity=0, verbose=False)


def test_out_name_writes_beside_the_real_floor_without_forcing(tmp_path, monkeypatch):
    """An arity calibration is a MEASUREMENT of the ruler, not a replacement for it. It must
    be writable while noise_floor.json already exists, without --force (which would destroy
    the floor every dossier so far was graded against)."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [0.1, 0.2, 0.3, 0.4])

    compute_noise_floor(batch_dir, n_draws=2, verbose=False)
    assert (batch_dir / NOISE_FLOOR_FILENAME).exists()

    compute_noise_floor(
        batch_dir, n_draws=2, arity=3, out_name="noise_floor_k3.json", verbose=False
    )

    real = json.loads((batch_dir / NOISE_FLOOR_FILENAME).read_text())
    calib = json.loads((batch_dir / "noise_floor_k3.json").read_text())
    assert real["n_placebo_columns"] == 1
    assert calib["n_placebo_columns"] == 3
    assert real["deltas_cpl_held"] == pytest.approx([0.1, 0.2])


def test_cli_arity_and_out_name(tmp_path, monkeypatch):
    df = _baseline_features_df()
    (tmp_path / "batches").mkdir()
    batch_dir = _write_batch_dir(tmp_path / "batches", df, name="b1")
    _stub_realised_by_draw(monkeypatch, [0.1, 0.2])

    result = CliRunner().invoke(
        main,
        [
            "b1", "--batches-dir", str(tmp_path / "batches"),
            "--n-draws", "2", "--arity", "3", "--out-name", "noise_floor_k3.json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((batch_dir / "noise_floor_k3.json").read_text())
    assert payload["n_placebo_columns"] == 3
    assert not (batch_dir / NOISE_FLOOR_FILENAME).exists()


# ── --same-source-column: the texture-ICC diagnostic (fps-3jj.23) ──────────────

def test_same_source_columns_pins_every_draw_and_stamps_them(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    pinned = resolve_baseline_columns(df)[:2]
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0, 3.0, 4.0])

    payload = compute_noise_floor(
        batch_dir, n_draws=4, same_source_columns=pinned, verbose=False
    )

    columns = [d["source_column"] for d in payload["placebo_draws"]]
    assert set(columns) == set(pinned)
    assert columns == [pinned[0], pinned[1], pinned[0], pinned[1]]
    assert payload["same_source_columns"] == pinned
    assert len(set(d["block_seed"] for d in payload["placebo_draws"])) == 4


def test_same_source_columns_gets_its_own_null_method(tmp_path, monkeypatch):
    """The property that lets a pinned bank measure the ICC — draws repeating a source
    column, so their deltas share texture exactly — is the same property that makes its band
    too NARROW to grade against. A distinct null_method makes that structural: _noise_band
    already refuses anything that isn't NULL_METHOD_PLACEBO_COLUMN."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    pinned = resolve_baseline_columns(df)[:1]
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0])

    payload = compute_noise_floor(
        batch_dir, n_draws=2, same_source_columns=pinned, verbose=False
    )

    assert payload["null_method"] == NULL_METHOD_PLACEBO_PINNED_COLUMN
    assert payload["null_method"] != NULL_METHOD_PLACEBO_COLUMN


def test_dossier_tables_refuses_to_grade_against_a_pinned_floor(tmp_path, monkeypatch):
    """Written to the RULER's own filename on purpose — the refusal must not depend on
    whoever ran it having chosen a safe --out-name."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    pinned = resolve_baseline_columns(df)[:2]
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0, 3.0, 4.0])
    payload = compute_noise_floor(
        batch_dir, n_draws=4, same_source_columns=pinned, out_name=NOISE_FLOOR_FILENAME,
        verbose=False,
    )

    band = _noise_band(
        {"meta": {"baseline_fingerprint": payload["baseline_fingerprint"],
                  "tank_params": payload["tank_params"]}},
        batch_dir,
    )

    assert band["available"] is False
    assert "null_method" in band["reason"]


def test_same_source_columns_leaves_an_ordinary_floor_untouched(tmp_path, monkeypatch):
    """The new key must be absent-or-null exactly where it used to be absent, and the ordinary
    path's null_method must not move — every dossier so far was graded on it."""
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0, 2.0, 3.0])

    payload = compute_noise_floor(batch_dir, n_draws=3, verbose=False)

    assert payload["same_source_columns"] is None
    assert payload["null_method"] == NULL_METHOD_PLACEBO_COLUMN


def test_same_source_columns_refuses_arity_above_one(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    pinned = resolve_baseline_columns(df)[:2]
    _stub_realised_by_draw(monkeypatch, [1.0] * 4)

    with pytest.raises(ValueError, match="DISTINCT source columns"):
        compute_noise_floor(
            batch_dir, n_draws=4, arity=2, same_source_columns=pinned, verbose=False
        )


def test_same_source_columns_refuses_a_column_outside_the_baseline_contract(tmp_path, monkeypatch):
    """The ICC being measured is the one effective_n_draws charges for reuse of a BASELINE
    column, so a pinned column that merely exists in the frame is not enough."""
    df = _baseline_features_df()
    df["not_locked"] = [float(i % 17) for i in range(len(df))]
    batch_dir = _write_batch_dir(tmp_path, df)
    _stub_realised_by_draw(monkeypatch, [1.0] * 4)

    with pytest.raises(ValueError, match="baseline contract"):
        compute_noise_floor(
            batch_dir, n_draws=4, same_source_columns=["not_locked"], verbose=False
        )


def test_cli_same_source_column_is_repeatable(tmp_path, monkeypatch):
    df = _baseline_features_df()
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_dir = _write_batch_dir(batches_dir, df)
    pinned = resolve_baseline_columns(df)[:2]
    _stub_realised_by_draw(monkeypatch, [0.0] * 6)

    result = CliRunner().invoke(
        main,
        ["batch1", "--batches-dir", str(batches_dir), "--n-draws", "6",
         "--same-source-column", pinned[0], "--same-source-column", pinned[1],
         "--out-name", "noise_floor_icc.json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((batch_dir / "noise_floor_icc.json").read_text())
    assert payload["same_source_columns"] == pinned
    assert payload["null_method"] == NULL_METHOD_PLACEBO_PINNED_COLUMN
    assert "NOT a grading ruler" in result.output
