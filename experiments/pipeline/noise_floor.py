"""Noise-floor calibration (fps-3jj.9, reworked fps-awz) — a placebo-column null, persisted
once per batch.

The dossier (experiments/pipeline/dossier_tables.py's `_noise_band()`) already reads
`<batch_dir>/noise_floor.json` and grades every candidate's realised delta against it — the
required key (`"deltas_cpl_held": [<float>, ...]`) and filename (`NOISE_FLOOR_FILENAME`) are
fixed by dossier_tables.py, not redefined here. The payload also carries provenance (seed,
draw construction, a "partial" flag, timestamp, git SHA, `baseline_fingerprint`, fold
geometry, wall_seconds) — `baseline_fingerprint` IS required by dossier_tables.py's
`_noise_band()` (fps-cf8: it refuses to grade a candidate against a floor whose fingerprint
doesn't match the candidate's own), the rest is for a human debugging a batch later.

Why this shape (fps-awz, reopening fps-3jj.9's original seed-swap design — see that bead's
own "Why 1"): a real candidate's `effect_delta_cpl_held` holds seed, data, and folds fixed
and adds ONE column, so the fit's idiosyncrasy is common-mode and cancels in the difference.
The seed-swap null this replaces held columns fixed and re-rolled the SEED instead — a
different, unpaired perturbation with no fixed ratio to the one a candidate actually
measures (batch0: baseline and candidate arms agreed on 741/752 fills, a near-total
cancellation the seed-swap null can't reproduce). The replacement here fits the SAME frozen
baseline at a SINGLE fixed seed (`PLACEBO_SEED_DEFAULT`, matching `runner.py`'s own
`realised_seed` default for the realised arbiter) across every draw, adding one PLACEBO
column each time — `experiments/pipeline/placebo.py` builds it as a block permutation of a
real baseline column (contiguous date-blocks reordered) along whichever axis that column
actually varies on, so it resembles a real feature in distribution/autocorrelation while its
alignment to the target is destroyed, and — unlike a naive within-date shuffle — never
degenerates to a no-op against a market-wide
column (see placebo.py's module docstring for why that matters: 45 of the 54 locked columns
are market-wide).

`placebo.screen_draw_groups` checks every candidate's `self_correlation` against
`MAX_SELF_CORRELATION` BEFORE any backtest fit and substitutes the next candidate on a
violation, rather than committing to a fixed draw list up front (an earlier version of this
module did exactly that, via `select_draws` alone, and could not produce a floor on batch0 at
all as a result — the first unusable column drawn aborted the whole computation). Under the
original circular-shift construction that substitution fired on every batch0 run and silently
emptied the bank of an entire feature CATEGORY; fps-d7m replaced the construction with a block
permutation, under which all 49 usable batch0 columns pass and no correlation-driven
substitution fires. (Substitution still fires twice per batch0 run for the unrelated all-NaN
skip — see placebo.py.) See placebo.py's module docstring for the measurements behind both.

Cost: every draw shares the exact same baseline arm (same frame, columns, seed, tank, folds)
— only the placebo arm changes — so `run_paired_realised_backtest`'s existing
`baseline_cache` mechanism (fps-e2l) lets draws 2..N skip refitting the baseline entirely.
Draw 1 pays a full baseline + placebo fit; every later draw pays only the placebo arm. This
keeps `DEFAULT_N_DRAWS = 20` roughly "1 baseline fit + 20 placebo fits", not "40 fits".

The placebo arm needs an `extra_feature_provider` (`experiments.pipeline.runner.
_make_lookup_provider`, reused, not reimplemented). `fuel_signal.backtest.ModelStrategy
.decide()` — the LIVE realised-replay path, distinct from the DataFrame-slicing fold-training
path — never reads an arm's `df` directly; it rebuilds every canonical feature fresh from
the DB-backed `PriceHistory` and only consults `extra_feature_provider` for anything beyond
that. A placebo column has no production `PriceHistory` source (it's synthetic), so without
a provider the replay raises `KeyError` on `PLACEBO_COLUMN_NAME` the moment it runs against
real data — the same wiring runner.py already needs for a real candidate's added column (the
`tgp_delta_7d` precedent). `_check_provider_health` (also reused) raises if a draw's provider
missed on every lookup — a 100%-miss placebo silently reads as "the column does nothing"
when it's actually a (station_code, date) key-format mismatch.

The OLD seed-swap null is NOT deleted — it measures something real (pure fit-instability
from the seed alone), just not the quantity the gate needs. It's retained as a separate,
explicitly-invoked diagnostic: `compute_fit_stability_diagnostic()` below, writing its own
`fit_stability.json`. It is NOT called automatically by `compute_noise_floor()` or
`batch_freeze.freeze_batch()` — wiring it into the automatic path would roughly double batch
setup's wall-clock for a number the gate doesn't consume.

fps-cf8: `batch_freeze.py`'s `freeze_batch()` calls `compute_noise_floor()` as its own final
step, so batch setup is one command and the floor cannot be forgotten or run against a
different snapshot than the one it grades. This module's CLI remains for recomputing a floor
on its own (e.g. after `--skip-noise-floor`, or with `--force` to replace a floor computed
at the batch's OWN declared cadence). Note `--force` is NOT the way to follow a cadence
re-lock: it resolves `tank` from the current `TankParams()` default, so on a batch frozen
at the old cadence it is refused by `check_freeze_cadence` below (fps-oqz) rather than
leaving freeze.json and noise_floor.json declaring different cadences. Freeze a new batch
at the new cadence instead — see docs/CONVENTIONS.md.

Usage (recompute a floor for an already-frozen batch):
  PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch-name> --force
Usage (recompute the fit-stability diagnostic, a separate opt-in artifact):
  PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch-name> --fit-stability --force
"""
from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Iterable
from datetime import datetime, timezone

import click

from experiments.lib.constants import SEEDS
from experiments.lib.io import current_git_sha, to_jsonable
from experiments.lib.realised import ArmSpec, BaselineCache, _plan_folds, run_paired_realised_backtest
from experiments.lib.universe import (
    UNIVERSE_SEED,
    draw_batch_universe,
    station_codes_digest,
)
from experiments.pipeline.batch_freeze import (
    BASELINE_COLUMNS_FILENAME,
    DEFAULT_BATCHES_DIR,
    FREEZE_MANIFEST_FILENAME,
    FROZEN_DB_FILENAME,
    check_baseline_contract,
)
from experiments.pipeline.dossier_tables import (
    NOISE_FLOOR_FILENAME,
    NULL_METHOD_PLACEBO_COLUMN,
    NULL_METHOD_PLACEBO_PINNED_COLUMN,
)
from experiments.pipeline.placebo import (
    DEFAULT_PLACEBO_ARITY,
    TEXTURE_ICC_BOUND,
    effective_n_draws,
    make_placebo_frame,
    placebo_column_names,
    screen_draw_groups,
    screen_pinned_column_draws,
)
from experiments.pipeline.runner import (
    DEFAULT_INNER_FOLD_PARAMS,
    _check_provider_health,
    _make_lookup_provider,
)
from fuel_signal.backtest import TankParams, format_tank_params
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.features import baseline_fingerprint, load_features

BASELINE_ARM_NAME = "baseline"
PLACEBO_ARM_NAME = "placebo"

# ~20 draws (fps-awz "Why 3"): the gate now reads a t-based distance from the band, which
# needs enough points that the band's own std is worth trusting — 5 draws (the old default)
# gives only 4 degrees of freedom.
DEFAULT_N_DRAWS = 20

# The realised arbiter's own single-seed default (runner.py's `realised_seed: int =
# SEEDS[0]`) — held fixed across every draw, matching what a real candidate's
# effect_delta_cpl_held is actually measured at, rather than varying it the way the old
# seed-swap null did.
PLACEBO_SEED_DEFAULT: int = SEEDS[0]

FIT_STABILITY_FILENAME = "fit_stability.json"

# Enforced bound on each draw's self_correlation self-check (see the draw loop below): the
# largest |correlation| a placebo may retain with the column it was built from.
#
# Now measured, not guessed (fps-d7m, full screen of batch0's 54 locked columns under the
# block-permutation construction). Batch0's draws separate cleanly into two groups: 46 of the
# 48 non-all-NaN columns sit at |corr| <= 0.12, and the two whose information is mostly
# cross-sectional rather than temporal (`stickiness_score` ~0.47,
# `station_minus_sydney_avg_cents` ~0.33) sit well above the rest — no time-axis reorder can
# move those, see placebo.py's NAMED LIMITATION. 0.6 admits both of those as (weak) draws
# while leaving clear air above them; it is deliberately NOT tightened to exclude them,
# because doing so would re-introduce exactly the systematic category exclusion fps-d7m
# exists to remove, just along the station axis instead of the level axis.
MAX_SELF_CORRELATION = 0.6

# Retained diagnostic only (see module docstring) — the original fps-3jj.9 seed pairing.
SEEDS_A: tuple[int, ...] = SEEDS
SEEDS_B: tuple[int, ...] = (47, 48, 49, 50, 51)


def check_freeze_cadence(batch_dir: pathlib.Path, tank: TankParams) -> None:
    """Refuse to write a floor at a cadence the batch does not declare (fps-oqz).

    `freeze.json`'s `tank_params` is the batch's declared cadence contract — the
    same role `baseline_fingerprint` plays for column identity. `freeze_batch`
    guarantees the two agree when it computes the floor itself, but this module's
    CLI can recompute a floor in place (`--force`), and it resolves `tank` from the
    CURRENT `TankParams()` default. After a re-lock those differ, so `--force` would
    quietly leave the batch split-brained: `freeze.json` still declaring 7d beside a
    `noise_floor.json` stamped 1d, with no reader to notice — nothing consumes
    `freeze.json`'s stamp, and `dossier_tables._noise_band()` (fps-v8o) compares only
    run-vs-floor, so a 1d candidate would then grade cleanly against what every doc
    calls a 7-day batch.

    This closes that triangle: fps-15c stamps the writers, fps-v8o guards run-vs-floor,
    and this guards floor-vs-freeze. Silent on a batch with no stamp (pre-fps-15c
    freeze manifests) — an absent declaration is not a disagreement.
    """
    freeze_path = batch_dir / FREEZE_MANIFEST_FILENAME
    if not freeze_path.exists():
        return
    declared = json.loads(freeze_path.read_text()).get("tank_params")
    actual = format_tank_params(tank)
    if declared and declared != actual:
        raise ValueError(
            f"{batch_dir.name} declares tank_params {declared!r} in "
            f"{FREEZE_MANIFEST_FILENAME}, but this floor would be computed at "
            f"{actual!r} — refusing (fps-oqz). A batch's floor and its freeze manifest "
            "must agree on cadence, or every candidate graded against that floor is "
            "compared to a ruler the batch does not declare. This batch was frozen "
            "before the cadence re-lock; freeze a NEW batch at the current cadence "
            "rather than recomputing this one's floor at it (docs/CONVENTIONS.md "
            "§ The decision cadence is a lock parameter). To recompute deliberately "
            "at the batch's own declared cadence, pass that tank explicitly."
        )


def _load_batch(batch_dir: pathlib.Path):
    """Common batch-loading + contract check shared by both artifacts below."""
    frame = load_features(batch_dir / "features.csv")  # .parquet sibling, batch_freeze convention
    check_baseline_contract(batch_dir, frame)
    baseline_columns = json.loads((batch_dir / BASELINE_COLUMNS_FILENAME).read_text())
    db_path = batch_dir / FROZEN_DB_FILENAME
    return frame, baseline_columns, db_path


def _draw_label(draw: list[tuple[str, int, float]]) -> str:
    """A one-line human label for a draw: its source columns and their block seeds.

    Used in progress output and in every error raised from inside the draw loop. A draw is
    a GROUP at arity > 1, so naming only its first member would point a reader at the wrong
    column half the time.
    """
    return ", ".join(f"{col}@{seed}" for col, seed, _corr in draw)


def compute_noise_floor(
    batch_dir: pathlib.Path,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    arity: int = DEFAULT_PLACEBO_ARITY,
    seed: int = PLACEBO_SEED_DEFAULT,
    outer_fold_params: dict | None = None,
    inner_fold_params: dict | None = None,
    fold_subset: Iterable[int] | None = None,
    station_codes: list[int] | None = None,
    station_universe_spec: dict | None = None,
    station_universe_pool_digest: str | None = None,
    tank: TankParams | None = None,
    out_name: str = NOISE_FLOOR_FILENAME,
    same_source_columns: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Compute `n_draws` placebo-column realised-CPL deltas and persist them to
    `<batch_dir>/<out_name>` (default `noise_floor.json`).

    Every draw is a two-arm `run_paired_realised_backtest` call: the frozen baseline vs the
    SAME baseline plus `arity` placebo columns (`experiments.pipeline.placebo`), at the SAME
    fixed `seed` throughout, `held_tau=None` (baseline's own per-fold τ is the held τ — the
    same contract runner.py's real candidate runs use). `delta = placebo.cpl_held -
    baseline.cpl_held` is exactly the operation `effect_delta_cpl_held` measures for a real
    candidate, so the resulting band is commensurable with it.

    arity (fps-3jj.14): how many placebo columns each draw adds. Default 1, the historical
    shape. A candidate is one MECHANISM and may be 2-4 columns (docs/routines/generator.md),
    and a k-column arm has more chances for the fit to find something than a 1-column arm
    does — so a k-column candidate graded against a 1-column band is graded against a ruler
    that does not know k, in the favourable direction. Stamped into the payload as
    `n_placebo_columns`; `dossier_tables._noise_band` refuses a floor whose arity is BELOW
    the run's candidate arity.

    `n_draws` and `arity` are INDEPENDENT (fps-3jj.21). They were not: both draw pools bound
    `n_draws * arity`, so a wider candidate got a thinner band, and `family_wise_z_threshold`
    turns a thinner band into a much harder bar (`df = n_draws - 1`) for a reason unrelated to
    arity. Both pools are unbounded now — pick `n_draws` for the certainty wanted and pay the
    compute. Arity is NOT capped either (`fps-3jj.25`): source-column reuse is priced into the
    threshold by `placebo.effective_n_draws`, so a wider candidate gets a wider, harder bar
    rather than being refused. The only arity limit is physical — a draw needs `arity` distinct
    source columns that pass screening.

    out_name: the filename written under `batch_dir`. Defaults to the batch's real ruler,
    `noise_floor.json`. An arity-calibration run is a MEASUREMENT of the ruler's arity
    sensitivity, not a replacement ruler, so it is written beside the real one under its own
    name (e.g. `noise_floor_k3.json`) rather than forcing over it — `--force` on the default
    name destroys the floor every dossier so far was graded against.

    station_codes (fps-916): was already threaded to `run_paired_realised_backtest` (None
    resolves to `fuel_signal.config.PREFERRED_STATIONS` there) but never stamped into the
    payload — every floor computed so far was in fact five-station, and nothing recorded
    that. Stamped as `station_population`: `experiments.lib.universe
    .station_codes_digest(station_codes or PREFERRED_STATIONS)`, a `count:hash` identity
    over the ACTUAL codes, not just a count — a bare count can't tell a five-station
    population from a different five-station population, and `dossier_tables
    ._bank_admissibility` refuses grading a run against a floor whose `station_population`
    doesn't match (fps-916). An absent key on either side reads as the five-station default,
    so every floor committed before this axis existed keeps grading exactly as it does today.

    `station_codes_digest` is used for `station_population` on BOTH the five-station default
    and the CLI's `--n-stations`-drawn path — a single namespace is the point (review of PR
    #362 finding 1b: `eligible_pool_digest` hashes a spec's whole GATED POOL, which is not
    only a different hash construction but is also blind to `spec.n`/`spec.seed` entirely, so
    two different `--n-stations` widths drawn from an unchanged pool would stamp the IDENTICAL
    digest under it — the exact cross-population admission this axis exists to prevent).
    `station_universe_spec` / `station_universe_pool_digest` are separate, PURELY
    informational provenance fields the CLI passes through when it drew a sampled
    population (`spec.as_dict()` and `station_pool_digest`/`eligible_pool_digest` respectively)
    — neither is read by `_bank_admissibility`; both are None for the five-station default.

    tank (fps-15c): the TankParams every draw runs at, and the cadence stamped into the
    payload's `tank_params`. None (default) resolves to `TankParams()` — the canonical
    1-day cadence (docs/CONVENTIONS.md, re-locked from 7d 2026-08-22, fps-oqz) — same
    default every other call in this pipeline resolves to when not overridden.

    same_source_columns (fps-3jj.23): pin every draw's source column to this small list,
    cycled round-robin, instead of spreading picks across the whole baseline. This is a
    MEASUREMENT of the ruler, not a ruler — it deliberately builds a bank whose draws repeat
    source columns so that grouping the deltas BY COLUMN isolates the texture component
    `placebo.TEXTURE_ICC_BOUND` stands for (between-column variance is texture, within-column
    is alignment). Because those repeats make the band narrower than a real one, the payload
    is stamped `null_method: placebo_column_pinned_source`, which `dossier_tables._noise_band`
    refuses outright — the diagnostic cannot be mistaken for a grading floor even if it is
    written to `noise_floor.json`. Arity must be 1 (a draw needs `arity` DISTINCT columns), and
    the pinned columns are stamped into the payload as `same_source_columns`.

    Reads the frozen batch's own features + baseline column contract (declared, never
    discovered — same `baseline_columns.json` every candidate run reads, fps-sa1) and its
    cloned DB, so this reproduces the exact fit + realised-backtest path a candidate run
    takes, minus the candidate's own feature. Also re-checks that contract against the
    current code (`check_baseline_contract`, same call every candidate run makes) so a
    code-side FEATURE_COLUMNS drift since freeze is caught here too, not just silently fit
    against a stale list.

    Returns the written payload. Raises:
      - FileExistsError if `<batch_dir>/noise_floor.json` already exists and `force` is not
        set — a full calibration already sitting there is the ruler every dossier so far was
        graded against; overwriting it silently (e.g. from a re-run) would move that ruler
        under them after the fact.
      - BaselineContractMismatch (via check_baseline_contract) on column drift.
      - ValueError (via placebo.screen_draw_groups) if arity exceeds the baseline column
        count — a draw needs that many DISTINCT source columns. There is no longer any
        STATISTICAL arity refusal (fps-3jj.25); reuse is priced, not capped.
      - ValueError (via placebo.screen_draw_groups) if this batch's data can't support
        n_draws draws at this arity passing MAX_SELF_CORRELATION even after exhausting the
        candidate sequence — rare, but a real outcome on a real batch (unlike the config-drift
        errors above, this is about the DATA, not a caller/environment mistake).
    """
    batch_dir = pathlib.Path(batch_dir)
    if arity < 1:
        raise ValueError(f"arity must be >= 1, got {arity}")
    if same_source_columns and arity != 1:
        raise ValueError(
            f"same_source_columns is incompatible with arity {arity} — a draw needs `arity` "
            "DISTINCT source columns, so 'arity k, all from one column' does not exist. The "
            "texture ICC is a property of ONE column's shape; measure it at arity 1."
        )
    out_path = batch_dir / out_name
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Every dossier written so far was graded against it — "
            "pass force=True (CLI: --force) only if you intend to replace that ruler for the "
            "whole batch, not just for this run."
        )

    outer_fold_params = dict(outer_fold_params or {})
    inner_fold_params = {**DEFAULT_INNER_FOLD_PARAMS, **(inner_fold_params or {})}
    tank = tank or TankParams()
    check_freeze_cadence(batch_dir, tank)

    frame, baseline_columns, db_path = _load_batch(batch_dir)
    # Screened BEFORE any backtest fit, not per-draw after one: a candidate that cannot be
    # decorrelated from its own source column is rejected by a correlation check rather than
    # by a wasted fit, and screen_draw_groups substitutes the next candidate in its pool so one
    # unusable column cannot abort the whole computation. Under the previous circular-shift
    # construction this fired on every batch0 run for CORRELATION (eight columns failed at
    # every offset); under block permutation all 49 usable batch0 columns pass, so that reason
    # never fires here. The tail is still used twice per batch0 run for the unrelated all-NaN
    # skip (two primaries land on all-NaN LGA columns) — see placebo.py's module docstring.
    if same_source_columns:
        # A pinned bank NEVER substitutes a different column on a screen failure (unlike the
        # path below, which must). Substitution here would quietly convert a same-column pair
        # into a different-column pair and bias the measured ICC toward zero — the direction
        # that reads "no problem" — so `screen_pinned_column_draws` retries at fresh seeds and
        # then raises. See its docstring.
        unknown = [c for c in same_source_columns if c not in baseline_columns]
        if unknown:
            raise ValueError(
                f"same_source_columns not in this batch's baseline contract: {unknown}. The "
                "ICC being measured is the one `effective_n_draws` charges for reuse of a "
                "BASELINE column, so the pinned columns must come from that same declared list "
                f"({BASELINE_COLUMNS_FILENAME}), not from the frame at large."
            )
        draws = screen_pinned_column_draws(
            frame, same_source_columns, n_draws, max_self_correlation=MAX_SELF_CORRELATION
        )
    else:
        draws = screen_draw_groups(
            frame, baseline_columns, n_draws, arity=arity,
            max_self_correlation=MAX_SELF_CORRELATION,
        )
    placebo_columns = placebo_column_names(arity)
    # Which declared columns are entirely NaN in THIS frozen frame (fps-3jj.23). Stamped as
    # provenance because the overlap structure `effective_n_draws` prices — and therefore every
    # published bar-vs-arity table — depends on HOW MANY columns the screen can actually
    # deliver, not on which ones: at 20 draws and arity 20, n_eff is 5.35 / 5.25 / 5.16 for 50 /
    # 49 / 48 usable columns. That count was previously an environmental fact asserted in a
    # docstring and unverifiable from the repo, since the frozen frame is gitignored. The frame
    # is right here, so the floor records it and any future reader can check rather than trust.
    all_nan_columns = sorted(c for c in baseline_columns if not frame[c].notna().any())

    t0 = time.perf_counter()
    deltas: list[float] = []
    placebo_draws: list[dict] = []
    n_windows: int | None = None
    tank_params: str | None = None
    baseline_cache: BaselineCache | None = None
    for i, draw in enumerate(draws, start=1):
        placebo_cols = make_placebo_frame(frame, draw)
        placebo_frame = frame.assign(**{c: placebo_cols[c] for c in placebo_columns})
        # ModelStrategy.decide() (the LIVE realised-replay path) never reads an arm's df
        # directly — it rebuilds every canonical feature fresh from the DB-backed
        # PriceHistory and only consults extra_feature_provider for anything beyond that
        # (fuel_signal/backtest.py). A placebo column has no PriceHistory source (it's
        # synthetic, computed here), so without a provider the replay would KeyError on
        # PLACEBO_COLUMN_NAME. Same lookup-provider mechanism runner.py already uses for a
        # real candidate's added column (e.g. the tgp_delta_7d precedent) — reused, not
        # reimplemented, so both sites agree on the (station_code, date)-keyed lookup.
        provider = _make_lookup_provider(placebo_frame, placebo_columns)
        arms = [
            ArmSpec(BASELINE_ARM_NAME, frame, feature_columns=baseline_columns),
            ArmSpec(
                PLACEBO_ARM_NAME, placebo_frame,
                feature_columns=[*baseline_columns, *placebo_columns],
                extra_feature_provider=provider,
            ),
        ]
        result = run_paired_realised_backtest(
            arms, baseline_columns,
            seed=seed,
            held_tau=None,
            station_codes=station_codes,
            outer_fold_params=outer_fold_params,
            inner_fold_params=inner_fold_params,
            fold_subset=fold_subset,
            db_path=db_path,
            tank=tank,
            collect_fills=False,
            baseline_cache=baseline_cache,
            verbose=verbose,
        )
        if result.baseline_cache is not None:
            baseline_cache = result.baseline_cache

        # A 100%-miss provider means the placebo column was NaN for the entire replay —
        # that would read as a legitimate "the column does nothing" delta when it's
        # actually a (station_code, date) key-format mismatch, silently narrowing the
        # floor with a draw that measured nothing. Same check runner.py runs for a real
        # candidate's provider, reused here rather than re-derived.
        health = _check_provider_health(provider)
        if health is not None:
            raise RuntimeError(f"draw {i} ({_draw_label(draw)}): {health}")

        agg = result.aggregate.set_index("arm")
        delta = float(agg.loc[PLACEBO_ARM_NAME, "cpl_held"] - agg.loc[BASELINE_ARM_NAME, "cpl_held"])
        draw_n_windows = int(result.meta["n_windows"])
        draw_tank_params = result.meta["tank_params"]

        # Every draw walks the same frozen data + fold geometry and the same tank, so all
        # of them must agree — a mismatch means something changed mid-run (config drift),
        # which would make the deltas incomparable (fps-15c: cadence must match, same shape
        # as the n_windows check).
        if n_windows is None:
            n_windows = draw_n_windows
            tank_params = draw_tank_params
        elif draw_n_windows != n_windows:
            raise ValueError(
                f"draw {i} ({_draw_label(draw)}): walked {draw_n_windows} windows but draw 1 "
                f"walked {n_windows} — fold geometry must be identical across every draw."
            )
        elif draw_tank_params != tank_params:
            raise ValueError(
                f"draw {i} ({_draw_label(draw)}): ran at tank_params {draw_tank_params!r} but "
                f"draw 1 ran at {tank_params!r} — cadence must match across every draw."
            )

        # self_correlation (correlation between the block-permuted series and the column's
        # own un-permuted values) was already checked against MAX_SELF_CORRELATION by
        # screen_draws, BEFORE this draw's fit ran — reused here for provenance, not
        # recomputed. Persisted per draw so a human reading a floor can see which draws were
        # weak ones (placebo.py's NAMED LIMITATION: a mostly-cross-sectional column keeps an
        # irreducible correlation no time-axis reorder can remove).
        deltas.append(delta)
        # One record per MEMBER of the draw, each tagged with its draw index, so an
        # arity-k floor stays readable as "which columns made up draw 3" rather than
        # collapsing a group into whichever member happened to be first. At arity 1 this
        # is one record per draw with one extra key, so an existing reader that only
        # looks at source_column/block_seed/self_correlation is unaffected.
        for source_column, block_seed, self_correlation in draw:
            placebo_draws.append(
                {
                    "draw": i,
                    "source_column": source_column,
                    "block_seed": block_seed,
                    "self_correlation": self_correlation,
                }
            )
        if verbose:
            print(
                f"[noise_floor] draw {i}/{len(draws)} ({_draw_label(draw)}): "
                f"delta_cpl_held={delta:+.4f}",
                flush=True,
            )

    wall_seconds = time.perf_counter() - t0
    payload = {
        "deltas_cpl_held": deltas,
        # dossier_tables._noise_band() refuses (available: False) any noise_floor.json
        # whose null_method isn't this — a pre-fps-awz batch's already-committed
        # seed-swap floor (e.g. batch0's) has no such key and is refused, not silently
        # graded, until it's recomputed under this construction.
        # fps-3jj.23: a pinned bank gets its OWN method string, so `_noise_band` refuses it
        # by the check it already runs. The property that lets it measure the texture ICC —
        # draws sharing a source column, hence sharing texture exactly — is the same property
        # that makes its band too narrow to grade against, so "cannot be used as a ruler" is
        # made structural here rather than left to whoever chooses --out-name.
        "null_method": (
            NULL_METHOD_PLACEBO_PINNED_COLUMN if same_source_columns
            else NULL_METHOD_PLACEBO_COLUMN
        ),
        # The pinned columns, in the order they were cycled. None for an ordinary floor, so an
        # existing reader sees the key absent-or-null exactly where it used to be absent.
        "same_source_columns": list(same_source_columns) if same_source_columns else None,
        "n_draws": len(draws),
        # fps-3jj.25: what this bank is WORTH once source-column reuse is priced in.
        # `dossier_tables._noise_band` feeds this to `family_wise_z_threshold` in place of the
        # nominal count, so a bank whose draws overlap gets a correspondingly wider, harder bar
        # instead of the band being refused. Stamped here rather than derived at grading time
        # because it is a property of the bank that actually got BUILT — screening substitutes
        # columns, so which draws really overlap is not recoverable from n_draws and arity.
        "effective_n_draws": effective_n_draws(draws),
        "texture_icc_assumed": TEXTURE_ICC_BOUND,
        # fps-3jj.14: the arity this band measures. dossier_tables._noise_band refuses a
        # floor whose n_placebo_columns is below the run's own candidate column count —
        # a floor with no such key at all (pre-fps-3jj.14) reads as arity 1, which is what
        # it in fact was, so an existing committed floor keeps grading 1-column candidates
        # and only stops grading multi-column ones.
        "n_placebo_columns": arity,
        # fps-916: `count:hash` identity of the population every draw actually replayed —
        # see this function's docstring for why a hash rather than a bare station count.
        # A floor with no such key at all (pre-fps-916) reads as the five-station default in
        # dossier_tables._bank_admissibility, which is what every one of them in fact was.
        "station_population": station_codes_digest(station_codes or PREFERRED_STATIONS),
        # Provenance ONLY (review of PR #362) — never read by _bank_admissibility. None for
        # the five-station default; the CLI's --n-stations fills these in from the
        # UniverseSpec it drew station_codes from.
        "station_universe_spec": station_universe_spec,
        "station_universe_pool_digest": station_universe_pool_digest,
        # Declared columns that are entirely NaN in the frozen frame, so the screen can never
        # accept them. NOT the full set of screen failures (a column can also fail on
        # correlation); this is the structural subset, which is what bounds the draw pool.
        "all_nan_baseline_columns": all_nan_columns,
        "n_baseline_columns_available": len(baseline_columns) - len(all_nan_columns),
        "seed": seed,
        "placebo_draws": placebo_draws,
        "fold_subset": list(fold_subset) if fold_subset is not None else None,
        # fold_subset is documented as an iteration/smoke speed-up, not a real calibration —
        # a partial-fold floor is not comparable to effect_delta_cpl_held, which always pools
        # every fold. dossier_tables._noise_band() refuses a "partial": true floor outright.
        "partial": fold_subset is not None,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "baseline_fingerprint": baseline_fingerprint(baseline_columns),
        "tank_params": tank_params,
        "n_windows": n_windows,
        "outer_fold_params": outer_fold_params,
        "inner_fold_params": inner_fold_params,
        "wall_seconds": wall_seconds,
    }
    out_path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    if verbose:
        print(
            f"[noise_floor] wrote {len(deltas)} draws to {out_path} in {wall_seconds:.1f}s",
            flush=True,
        )
    return payload


def _baseline_cpl_held(
    frame,
    baseline_columns: list[str],
    seed: int,
    *,
    outer_fold_params: dict,
    inner_fold_params: dict,
    fold_subset: Iterable[int] | None,
    station_codes: list[int] | None,
    db_path: pathlib.Path,
    tank: TankParams,
    verbose: bool,
) -> tuple[float, int, str]:
    """Fit the baseline alone (no candidate/placebo arm) at `seed`; return (cpl_held,
    n_windows, tank_params). Used only by `compute_fit_stability_diagnostic` below — the
    retained seed-swap diagnostic, not the noise-floor gate.
    """
    result = run_paired_realised_backtest(
        [ArmSpec(BASELINE_ARM_NAME, frame, feature_columns=baseline_columns)],
        baseline_columns,
        seed=seed,
        held_tau=None,
        station_codes=station_codes,
        outer_fold_params=outer_fold_params,
        inner_fold_params=inner_fold_params,
        fold_subset=fold_subset,
        db_path=db_path,
        tank=tank,
        collect_fills=False,
        verbose=verbose,
    )
    cpl_held = float(result.aggregate.set_index("arm").loc[BASELINE_ARM_NAME, "cpl_held"])
    return cpl_held, int(result.meta["n_windows"]), result.meta["tank_params"]


def compute_fit_stability_diagnostic(
    batch_dir: pathlib.Path,
    *,
    seeds_a: tuple[int, ...] = SEEDS_A,
    seeds_b: tuple[int, ...] = SEEDS_B,
    outer_fold_params: dict | None = None,
    inner_fold_params: dict | None = None,
    fold_subset: Iterable[int] | None = None,
    station_codes: list[int] | None = None,
    station_universe_spec: dict | None = None,
    station_universe_pool_digest: str | None = None,
    tank: TankParams | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """The ORIGINAL fps-3jj.9 seed-swap null, kept as an explicitly-invoked diagnostic —
    NOT the noise-floor gate `dossier_tables._noise_band()` reads (that's
    `compute_noise_floor()` above). Writes `<batch_dir>/fit_stability.json`.

    Measures pure fit instability from the seed alone: the SAME frozen baseline (same
    columns, same data), fit at two different seed groups, realised-CPL delta. Real — this
    is a legitimate reproducibility check on the pipeline itself — but a DIFFERENT,
    unpaired perturbation from "add one column at a fixed seed", which is why fps-awz moved
    the gate off it (see this module's docstring). Not called by `compute_noise_floor()` or
    `batch_freeze.freeze_batch()`: wiring it into the automatic path would roughly double
    batch-setup wall-clock for a number nothing downstream consumes.

    Raises the same shape of errors as the old `compute_noise_floor` did: mismatched/
    repeated/overlapping seed groups, an existing file without `force`, or a baseline
    contract drift.
    """
    if len(seeds_a) != len(seeds_b):
        raise ValueError(
            f"seeds_a and seeds_b must pair 1:1; got {len(seeds_a)} vs {len(seeds_b)} seeds."
        )
    if len(set(seeds_a)) != len(seeds_a) or len(set(seeds_b)) != len(seeds_b):
        raise ValueError(f"seeds_a and seeds_b must each be free of repeats; got {seeds_a}, {seeds_b}.")
    overlap = set(seeds_a) & set(seeds_b)
    if overlap:
        raise ValueError(
            f"seeds_a and seeds_b must be disjoint — a shared seed {sorted(overlap)} pairs a fit "
            "against itself and writes a zero delta, silently narrowing the floor."
        )

    batch_dir = pathlib.Path(batch_dir)
    out_path = batch_dir / FIT_STABILITY_FILENAME
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Pass force=True (CLI: --force) only if you intend "
            "to replace it."
        )

    outer_fold_params = dict(outer_fold_params or {})
    inner_fold_params = {**DEFAULT_INNER_FOLD_PARAMS, **(inner_fold_params or {})}
    tank = tank or TankParams()
    check_freeze_cadence(batch_dir, tank)

    frame, baseline_columns, db_path = _load_batch(batch_dir)

    t0 = time.perf_counter()
    deltas: list[float] = []
    n_windows: int | None = None
    tank_params: str | None = None
    for i, (seed_a, seed_b) in enumerate(zip(seeds_a, seeds_b, strict=True), start=1):
        kwargs = dict(
            outer_fold_params=outer_fold_params,
            inner_fold_params=inner_fold_params,
            fold_subset=fold_subset,
            station_codes=station_codes,
            db_path=db_path,
            tank=tank,
            verbose=verbose,
        )
        cpl_a, n_windows_a, tank_params_a = _baseline_cpl_held(frame, baseline_columns, seed_a, **kwargs)
        cpl_b, n_windows_b, tank_params_b = _baseline_cpl_held(frame, baseline_columns, seed_b, **kwargs)
        if n_windows_a != n_windows_b:
            raise ValueError(
                f"pair {i}: seed {seed_a} walked {n_windows_a} windows but seed {seed_b} "
                f"walked {n_windows_b} — fold geometry must match within a pair."
            )
        if n_windows is None:
            n_windows = n_windows_a
        elif n_windows != n_windows_a:
            raise ValueError(
                f"pair {i}: walked {n_windows_a} windows but pair 1 walked {n_windows} — "
                "fold geometry must be identical across every draw."
            )
        if tank_params_a != tank_params_b:
            raise ValueError(
                f"pair {i}: seed {seed_a} ran at tank_params {tank_params_a!r} but seed "
                f"{seed_b} ran at {tank_params_b!r} — cadence must match within a pair."
            )
        tank_params = tank_params_a
        delta = cpl_b - cpl_a
        deltas.append(delta)
        if verbose:
            print(
                f"[fit_stability] pair {i}/{len(seeds_a)} (seed {seed_a} vs {seed_b}): "
                f"delta_cpl_held={delta:+.4f}",
                flush=True,
            )

    wall_seconds = time.perf_counter() - t0
    payload = {
        "deltas_cpl_held": deltas,
        # Informational only — nothing reads fit_stability.json programmatically the way
        # dossier_tables._noise_band() reads noise_floor.json's null_method — but self-
        # describing the construction is cheap and matches this module's other artifacts.
        "null_method": "seed_swap",
        "seeds_a": list(seeds_a),
        "seeds_b": list(seeds_b),
        # fps-916: same identity `compute_noise_floor` stamps — informational here too
        # (nothing grades this diagnostic against a run the way noise_floor.json is), but
        # matching the sibling artifact keeps a human reading both files unsurprised.
        "station_population": station_codes_digest(station_codes or PREFERRED_STATIONS),
        # Provenance ONLY (review of PR #362) — never read by _bank_admissibility. None for
        # the five-station default; the CLI's --n-stations fills these in from the
        # UniverseSpec it drew station_codes from.
        "station_universe_spec": station_universe_spec,
        "station_universe_pool_digest": station_universe_pool_digest,
        "fold_subset": list(fold_subset) if fold_subset is not None else None,
        "partial": fold_subset is not None,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "baseline_fingerprint": baseline_fingerprint(baseline_columns),
        "tank_params": tank_params,
        "n_windows": n_windows,
        "outer_fold_params": outer_fold_params,
        "inner_fold_params": inner_fold_params,
        "wall_seconds": wall_seconds,
    }
    out_path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    if verbose:
        print(
            f"[fit_stability] wrote {len(deltas)} draws to {out_path} in {wall_seconds:.1f}s",
            flush=True,
        )
    return payload


def _parse_fold_subset(value: str | None) -> list[int] | None:
    """Comma-separated 1-indexed fold numbers, e.g. '1,2'. None -> None (every fold)."""
    if not value:
        return None
    folds: list[int] = []
    for token in value.split(","):
        try:
            fold = int(token)
        except ValueError:
            raise click.BadParameter(
                f"{token!r} is not an integer — expected comma-separated 1-indexed fold "
                "numbers, e.g. '1,2'."
            ) from None
        if fold < 1:
            raise click.BadParameter(f"fold {fold} is not 1-indexed — folds start at 1.")
        folds.append(fold)
    return folds


@click.command("noise_floor")
@click.argument("batch_name")
@click.option(
    "--batches-dir", default=str(DEFAULT_BATCHES_DIR), show_default=True,
    help="Parent directory for batch snapshots.",
)
@click.option(
    "--n-draws", default=DEFAULT_N_DRAWS, show_default=True,
    help="Number of placebo draws. Ignored with --fit-stability (that diagnostic is always "
    "seed-paired at the module's SEEDS_A/SEEDS_B defaults — not a CLI-overridable option).",
)
@click.option(
    "--arity", default=DEFAULT_PLACEBO_ARITY, show_default=True,
    help="Placebo columns per draw (fps-3jj.14). 1 is the historical shape. Raise it to "
    "match a batch whose candidates are multi-column GROUPS — a k-column candidate graded "
    "against a 1-column band is graded favourably by an unmeasured amount. --n-draws is "
    "INDEPENDENT of this (fps-3jj.21): pick the draw count for the certainty you want and "
    "pay the compute. Arity is NOT capped (fps-3jj.25) — source-column reuse is priced into "
    "the bar by placebo.effective_n_draws, so a wider ruler is graded, not refused. The limit "
    "is physical: a draw needs that many distinct source columns that pass screening, so it "
    "cannot exceed the count of USABLE baseline columns (all-NaN columns do not count).",
)
@click.option(
    "--out-name", default=NOISE_FLOOR_FILENAME, show_default=True,
    help="Output filename under the batch dir. Change it to write an arity-calibration "
    "run BESIDE the batch's real ruler (e.g. noise_floor_k3.json) instead of --force'ing "
    "over the floor every dossier so far was graded against.",
)
@click.option(
    "--same-source-column", "same_source_column", multiple=True,
    help="Pin every draw's source column to this baseline column, instead of spreading picks "
    "across the whole baseline. Repeat the flag to cycle several columns round-robin. This "
    "computes the texture-ICC DIAGNOSTIC (fps-3jj.23), not a grading ruler: draws deliberately "
    "repeat source columns so that grouping deltas BY COLUMN separates texture (between-column) "
    "from alignment (within-column). The result is stamped with its own null_method and the "
    "dossier refuses to grade against it. Requires --arity 1. Pass 4-6 columns with enough "
    "draws each for the ANOVA to resolve anything — see "
    "experiments/2026-08-27_texture_icc/README.md for the power table before choosing.",
)
@click.option(
    "--fit-stability", is_flag=True, default=False,
    help="Compute the retained seed-swap diagnostic (fit_stability.json) instead of the "
    "noise-floor gate (noise_floor.json). Not part of the automatic batch-freeze path.",
)
@click.option(
    "--fold-subset", default=None,
    help="Comma-separated 1-indexed outer folds, e.g. '1,2'. Iteration/smoke speed-up ONLY — "
    "the result covers a partial fold set and is written as 'partial': true, which the "
    "dossier refuses to grade candidates against.",
)
@click.option(
    "--force", is_flag=True, default=False,
    help="Overwrite an existing output file for this batch. Every dossier written so far "
    "was graded against the noise floor — only pass this if you intend to replace that "
    "ruler.",
)
@click.option(
    "--n-stations", "n_stations", default=None, type=click.IntRange(min=1),
    help="Draw a station population of this width from the batch's eligible pool "
    "(experiments.lib.universe, gated on every outer fold's real val window) instead of "
    "grading at the five-station default (fps-916). Mirrors "
    "experiments/2026-09-05_arbiter_universe_width/homogeneity.py's own draw — same "
    "seed, same window construction — so this CLI reproduces the population that "
    "script already measured rather than an unvetted new sample of the same width. "
    "Stamped into the floor as `station_population` (a hash of the ACTUAL drawn codes, "
    "same identity the five-station default uses — not a hash of the eligible pool, which "
    "would not distinguish two different --n-stations widths drawn from it), so a run "
    "graded against a differently-populated floor is refused rather than silently "
    "mismatched.",
)
def main(
    batch_name: str, batches_dir: str, n_draws: int, arity: int, out_name: str,
    same_source_column: tuple[str, ...], fit_stability: bool, fold_subset: str | None,
    force: bool, n_stations: int | None,
) -> None:
    """Compute and persist the noise-floor calibration for an already-frozen batch."""
    batch_dir = pathlib.Path(batches_dir) / batch_name
    subset = _parse_fold_subset(fold_subset)
    if fit_stability and same_source_column:
        # The seed-swap diagnostic has no placebo columns at all, so it cannot honour a pinned
        # source. Returning early (as this used to) accepted the flag and silently ignored it,
        # which would look like a texture-ICC run and produce a fit_stability.json.
        raise click.BadParameter(
            "--same-source-column has no meaning with --fit-stability: that diagnostic swaps "
            "SEEDS on the unchanged baseline and adds no placebo column to pin. Drop one."
        )

    station_codes: list[int] | None = None
    station_universe_spec: dict | None = None
    station_universe_pool_digest: str | None = None
    if n_stations is not None:
        # Real val windows, not the span envelope — see draw_batch_universe's docstring for
        # why, and for why `windows` is derived here rather than inside the helper.
        frame = load_features(batch_dir / "features.csv")
        windows = tuple((p.val_start, p.val_end) for p in _plan_folds(frame, {}, None))
        station_codes, station_universe_spec, station_universe_pool_digest = (
            draw_batch_universe(
                batch_dir / FROZEN_DB_FILENAME, n=n_stations, windows=windows,
                seed=UNIVERSE_SEED,
            )
        )

    if fit_stability:
        compute_fit_stability_diagnostic(
            batch_dir, fold_subset=subset, force=force,
            station_codes=station_codes, station_universe_spec=station_universe_spec,
            station_universe_pool_digest=station_universe_pool_digest,
        )
        click.echo(f"Wrote fit-stability diagnostic for batch '{batch_name}' -> {batch_dir / FIT_STABILITY_FILENAME}")
        return
    compute_noise_floor(
        batch_dir, n_draws=n_draws, arity=arity, out_name=out_name,
        same_source_columns=list(same_source_column) or None,
        fold_subset=subset, force=force,
        station_codes=station_codes, station_universe_spec=station_universe_spec,
        station_universe_pool_digest=station_universe_pool_digest,
    )
    what = (
        f"texture-ICC diagnostic (pinned to {', '.join(same_source_column)}; NOT a grading ruler)"
        if same_source_column else f"noise floor (arity {arity})"
    )
    if n_stations is not None:
        what += f", {n_stations}-station population"
    click.echo(f"Wrote {what} for batch '{batch_name}' -> {batch_dir / out_name}")


if __name__ == "__main__":
    main()
