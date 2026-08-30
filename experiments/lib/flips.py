"""Decision-flip attribution between two realised-backtest arms (fps-gez).

A "flip" here means: on this (fold, station_code, date), one arm bought and the
other did not. That is the realised-backtest ground truth for "the two arms made
a different decision" — not a proba-threshold crossing on rowpreds.parquet.
rowpreds.parquet's `proba` comes from the WFCV screen's own plain `fit_score`
across several seeds (experiments/pipeline/runner.py's `_run_wfcv_screen`), a
different fit from the realised backtest's single-seed, OOF-calibrated,
per-fold-tau-selected model (`experiments/lib/realised.py`'s
`_train_calibrate_select_tau`). Thresholding rowpreds' raw proba at a realised
arm's own_tau would compare a decision from one model against a tau chosen for
a different one — not the thing that actually drove the realised CPL number.
fills.parquet, by contrast, is the tank simulator's own executed record, so
diffing it directly is exact: no re-fitting, no re-thresholding, just "did the
two arms buy on the same days."

That exactness comes with a known limitation, not a bug: a WAIT day writes no
fill row, so this only sees a flip when it changes which day a station
actually bought on. Two arms can diverge in their day-by-day probability
without ever producing a differing fill (see
`feedback_realised_arbiter_decision_flip_quantum` — flip counts are a LOWER
bound on decision divergence). And because the tank's state is path-dependent,
one real flip can cascade into a run of later fills that also differ, without
each of those later fills being an independent "flip" in the threshold-crossing
sense — that's a property of the mechanism being measured (a buy decision
changes future tank state), not noise to average away. `summarise_flips`
below collapses those cascades into `n_decisions` (fps-e1w) so a per-fold
count isn't read as independent evidence when most of it is one decision
replayed forward.

fps-e1w rework: the raw per-fold `flip_cpl_delta` this module used to emit on
its own (a difference of two DISJOINT, thin, differently-timed fill sets) was
being read as evidence it can't support — see this module's `_price_dispersion`
docstring and `docs/routines/dossier.md`'s citation rule. `summarise_flips` now
also emits each delta's own standard error (so a reader can see when a cell
can't resolve its own number), cascade-collapsed decision counts, and per-arm
flip litres. The run-level reconciliation (litres-weighted shift-share against
`delta_cpl_held`) needs the FULL fold's fills, not just the flipped ones, so it
is computed one layer up, in `experiments/pipeline/dossier_tables.py`.
"""
from __future__ import annotations

import math
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from experiments.lib.zones import pooled_cpl
from fuel_signal.backtest import TankParams, format_tank_params

FLIP_KEY_COLUMNS = ["fold", "station_code", "date"]
FLIP_ROW_COLUMNS = FLIP_KEY_COLUMNS + ["price", "litres", "spend_cents", "bought_by"]

#: A cell's flip-price standard deviation is trusted fold-local only with at least this many
#: flip fills (both arms combined) in the fold; thinner folds fall back to the run-wide
#: dispersion (fps-e1w change 2).
MIN_FOLD_LOCAL_DISPERSION_N = 4

#: How many standard errors wide the printed interval is, and the bar a delta must clear
#: (in magnitude) before it stops being "inside its own noise" — 2*SE, matching the review
#: this module's docstring above references (docs/routines/dossier.md's citation rule).
INTERVAL_SE_MULTIPLE = 2.0


def diff_fills(fills: pd.DataFrame, baseline_arm: str, candidate_arm: str) -> pd.DataFrame:
    """Fills that exist in exactly one of the two arms, tagged with which one bought.

    One row per differing fill; `bought_by` is `"baseline"` or `"candidate"`.
    Returns an empty frame (right columns, zero rows) when the arms agree on
    every fill.
    """
    base = fills[fills["arm"] == baseline_arm]
    cand = fills[fills["arm"] == candidate_arm]
    base_keys = pd.MultiIndex.from_frame(base[FLIP_KEY_COLUMNS])
    cand_keys = pd.MultiIndex.from_frame(cand[FLIP_KEY_COLUMNS])

    only_base = base.loc[~base_keys.isin(cand_keys)].copy()
    only_cand = cand.loc[~cand_keys.isin(base_keys)].copy()
    only_base["bought_by"] = "baseline"
    only_cand["bought_by"] = "candidate"
    return pd.concat(
        [only_base[FLIP_ROW_COLUMNS], only_cand[FLIP_ROW_COLUMNS]], ignore_index=True
    )


def _tank_from_exact_fields(exact_fields: Mapping[str, float], tank_params: str) -> TankParams:
    """Build the real `TankParams` from `fuel_signal.backtest.tank_params_fields`'s exact
    numbers, so a caller that has them can get windows straight from their one owner
    (`TankParams`'s own properties, fps-o0h) instead of this module's stamp-parsing path.

    Asserts the rebuilt tank re-formats to the SAME stamp the caller also passed
    (PR #355 review finding #6) — the two are supposed to be the same run's paired fields
    (`experiments/lib/realised.py`'s meta writes both off the one `tank`), so a mismatch means
    a caller mixed fields from a different run/results.json into this one, the same class of
    provenance bug `fps-cf8`/`fps-v8o` already guard against for `baseline_fingerprint`/
    `tank_params` themselves. This does NOT catch every float-precision hazard `exact_fields`
    can hit (see `cascade_window_days`'/`regret_horizon_days`' docstrings for the one it
    can't: floating-point non-invertibility already baked into a stored `daily_consumption_litres`
    value, independent of which run it came from) — it only catches the fields belonging to
    the wrong tank entirely.
    """
    tank = TankParams(
        tank_size_litres=float(exact_fields["tank_size_litres"]),
        daily_consumption_litres=float(exact_fields["daily_consumption_litres"]),
        evaluation_interval_days=int(exact_fields["evaluation_interval_days"]),
        floor_fraction=float(exact_fields["floor_fraction"]),
    )
    restamped = format_tank_params(tank)
    if restamped != tank_params:
        raise ValueError(
            f"exact_fields {exact_fields!r} re-stamp to {restamped!r}, not the tank_params "
            f"{tank_params!r} passed alongside them — these belong to two different tanks/runs, "
            "not a matched pair (fps-o0h, PR #355 review finding #6)."
        )
    return tank


def cascade_window_days(tank_params: str, *, exact_fields: Mapping[str, float] | None = None) -> int:
    """The flip-cascade collapse window, derived from the run's OWN tank cadence rather than
    a hardcoded literal (fps-e1w change 3).

    Half the tank's full-to-empty duration (`tank_size_litres / daily_consumption_litres`,
    the same ratio `TankParams`'s own default comment describes as "empties in 14 days") — the
    physical scale over which one arm buying a fill early or late can plausibly still be the
    same underlying decision rather than an independent one, before the tank's phase has had
    time to fully resync. At this run's own tank_params (`50/3.571/1d/10%`), that number is 7 —
    the value the fps-6yi post-dossier review used by hand (`docs/routines/dossier.md`'s
    worked example), reproduced here from the run's own parameters instead of copied as a
    literal.

    **Floored strictly above the run's own fill cadence** (`evaluation_interval_days`, PR
    #347 review finding #2). Fills only land on evaluation dates, so two flips at the SAME
    station are never closer together than one cadence period — a half-tank-life window at or
    below that period is a structural degenerate case, not a rare corner: this project's own
    committed `batch0/tgp_delta_7d` and the equivalent tank size/daily ratio at a 7d cadence
    put the two in EXACT collision (half-life 7, cadence 7), and below the cadence the
    collapse window would be inert for every fold (no on-grid gap can ever be <= a window
    narrower than the cadence, so `n_decisions` would silently equal `n_flips` always — the
    opposite failure from what this window exists to prevent). `max(half_life, cadence + 1)`
    guarantees the window can never degenerate into either extreme by an accident of rounding.

    A long unbroken run of on-grid divergence at one station still collapses to ONE decision
    regardless of this floor — that is the shipped design (a sustained divergence is one
    continuing decision replayed forward, `flips.py`'s module docstring), not something the
    floor is meant to prevent.

    **`n_decisions` is NOT cadence-invariant, and that asymmetry is quantifiable, not just
    philosophical** (PR #347 review, second pass). Gaps between same-station flips are
    quantised to the run's cadence, so the quiet period actually needed to start a new
    decision scales with it: at a 1-day cadence the first gap exceeding the 7-day window is 8
    days; at a 7-day cadence the first gap exceeding a (floored) 8-day window is 14 days — the
    coarser-cadence run needs proportionally MORE elapsed quiet time before a second decision
    registers, and so will systematically report a LOWER `n_decisions` than a finer-cadence run
    seeing the identical underlying divergence pattern. Do not read a lower `decns` on a
    coarser-cadence run as weaker evidence when comparing dossiers across cadences without
    accounting for this.

    `tank_params` is the formatted stamp `fuel_signal.backtest.format_tank_params` writes
    (`f"{size}/{daily}/{interval}d/{floor}%"`); parsing is safe because that function is this
    string's only writer (`fuel_signal.backtest.require_tank_stamp` is the only sanctioned way
    to obtain one).

    `exact_fields` (fps-o0h), when given, is `fuel_signal.backtest.tank_params_fields(tank)`'s
    dict — the exact floats behind the stamp, persisted structurally alongside it. Pass it
    whenever the caller's meta carries it: `format_tank_params` renders daily consumption at
    3dp, which round-trips losslessly almost always but can flip this window's half-life on an
    exact rounding tie. `tank_params` (the stamp) is still required even when `exact_fields` is
    given — it is the one contract-bound identity string (fps-15c) — but every number this
    function derives, INCLUDING the cadence floor, now comes from the rebuilt `TankParams` when
    `exact_fields` is available, not a mix of the two sources (PR #355 review finding #6): ints
    round-trip a stamp exactly, so there was never a precision reason to keep parsing
    `evaluation_interval_days` off the string once `exact_fields` is in hand, only an
    inconsistency risk.

    **`exact_fields` does not "solve" rounding ties in general — it only removes the specific
    one this window can suffer from re-parsing the display stamp** (PR #355 review finding #2,
    correcting this docstring's earlier overclaim). A 560-config sweep (mirroring
    `tests/test_exp_lib_flips.py`'s own) found **90** configs where the stamp-only and
    `exact_fields` answers disagree — not the 12 an earlier draft of this docstring claimed
    (that number belongs to `regret_horizon_days`, transplanted here in error). Every one of the
    90 has an ODD `tank_size_litres / daily_consumption_litres` in low terms (so the true
    half-life is an exact `n.5`), confirming the disagreement is real rounding-tie territory,
    not noise — but `exact_fields` does not reliably pick the mathematically-correct side of
    it. Worked counterexample: `TankParams(50.0, 50.0/11)` has a true, exact half-life of 5.5
    (round-half-to-even -> 6), but `full_to_empty_days` computes `50.0 / (50.0/11)` as
    `10.999999999999998` (an unavoidable double-rounding artifact of the intermediate
    `50.0/11` — no reordering of the division recovers exactness), so `exact_fields` rounds
    this one DOWN to 5 where the stamp-only path (`50/4.545/1d/...`, `50/4.545/2` computing
    slightly *above* 5.5) happens to land on the correct 6. So `exact_fields` trades the
    stamp's specific, well-understood failure mode (3dp display truncation) for a different,
    equally-real one (floating-point non-invertibility already latent in a `daily_consumption_
    litres` value that was itself constructed by division) — it is not strictly more accurate,
    only differently imprecise, and both paths can disagree with the "true" tie on different
    configs. Pass it anyway when available, because the two failure modes are NOT the same size
    in practice for HOW THIS RUN'S TANK is normally constructed (`tank_life` in whole days,
    `daily = size / tank_life`) — but do not describe it as removing ties, here or downstream.
    """
    size, daily, cadence_days, _floor = parse_tank_params(tank_params)
    if exact_fields is not None:
        tank = _tank_from_exact_fields(exact_fields, tank_params)
        half_life = round(tank.full_to_empty_days / 2)
        cadence_days = tank.evaluation_interval_days
    else:
        half_life = round(size / daily / 2)
    return max(half_life, cadence_days + 1)


def parse_tank_params(tank_params: str) -> tuple[float, float, int, float]:
    """`(tank_size_litres, daily_consumption_litres, evaluation_interval_days, floor_fraction)`
    from the stamp `fuel_signal.backtest.format_tank_params` writes.

    Single-sourced so `cascade_window_days` and `regret_horizon_days` — two different windows
    derived from the same four numbers — can never disagree about how the stamp is read.
    `floor` is stamped as a percentage (`10%`) and returned as a fraction (`0.10`), matching
    `TankParams.floor_fraction`'s own units rather than the string's.
    """
    size_str, daily_str, interval_str, floor_str = tank_params.split("/")
    return (
        float(size_str),
        float(daily_str),
        int(interval_str.rstrip("d")),
        float(floor_str.rstrip("%")) / 100.0,
    )


def regret_horizon_days(tank_params: str, *, exact_fields: Mapping[str, float] | None = None) -> int:
    """`summarise_regret`'s forward window: a FIXED, EXOGENOUS ceiling on how long a wait the
    tank could ever fund — `(1 - floor_fraction) * tank_size_litres / daily_consumption_litres`
    (fps-2js). At fps-6yi's tank (`50/3.571/1d/10%`), 0.9 * 50 / 3.571 = 12.6 -> **13 days**.

    **This ceiling is not attained by any real fill, and must not be described as reachable**
    (PR #348 review finding #2). A fill only happens when the tank is short, and `run_backtest`
    buys to full (`buy_litres = size - level`), so the level immediately before a fill is
    `size - litres`, and that fill's genuinely feasible wait is
    `((1 - floor) * size - litres) / daily`. Since a fill is at least one depletion's worth,
    the per-fill bound is strictly below the ceiling. Measured over fps-6yi's committed 303
    flips: **max 11.60 days, mean 7.94, litres-weighted 5.71 (baseline) / 5.07 (candidate),
    11 flips under one day, and 0 of 303 reaching 13.**

    **The fixed ceiling is kept anyway, and the reason is identification, not convenience.**
    The per-fill bound is a function of `litres`, which is itself the residue of the arm's own
    earlier buy/wait decisions — an ENDOGENOUS, path-coupled window. This project has already
    established that path-coupled quantities cut per-row have no unique answer (the 2026-08-21
    path-coupling audit; `cut economics on folds, not on row labels`), and a measurement window
    that each arm partly chooses for itself is that same defect relocated into the estimator.
    A fixed window is worse-calibrated but identified; a per-fill window is better-calibrated
    but not. Both were computed, and the choice does not move any reading:

        litres-weighted regret       baseline   candidate    delta
        fixed ceiling H=13 (shipped)   10.03       9.38      -0.6437
        per-fill feasible H_i           4.70       3.72      -0.9726

    Levels roughly halve and the delta grows ~51%, but both sit far inside the run-level 2*SE
    of +/-4.24, so neither supports a conclusion the other refuses.

    **Known residual bias, stated because it is the kind this measure exists to remove.** The
    ceiling over-charges regret to whichever arm buys in LARGER fills, because those fills have
    the least real headroom and the fixed window over-reaches furthest for them. On fps-6yi the
    baseline's litres-weighted capacity is 5.71 days against the candidate's 5.07, so the
    shipped delta is conservative toward the candidate by roughly the gap the table above
    shows. Do not read a regret delta smaller than that gap as a difference between arms.

    **Derived from the TANK, not from cycle length — a departure from fps-2js as filed.** The
    bead proposed deriving H from `cycle_mean_length` ("longer than a cycle makes every fill
    look bad"). The direction is right, but the binding constraint arrives far earlier: batch0's
    `cycle_mean_length` is mean 30.5 / max 35.3 days (`experiments/pipeline/placebo.py`) while
    this tank cannot fund even 13. Sweeping H over the real fps-6yi flips:

        H (days)   3     5     7    10    12    13    14  |   15    18    21    28    30    61
        cand-base -0.44 +0.16 -0.02 -0.19 -0.65 -0.64 -0.64| -0.98 -2.22 -3.22 -4.19 -4.71 -6.56

    Read this as a trend, not a cliff (PR #348 review): the separation is gradual, and H=12's
    -0.65 is already ~9x the run's realised `delta_cpl_held` of -0.0735 c/L. What the sweep
    shows is that the measure grows without bound as the window leaves the tank's reach — at a
    cycle-length horizon it claims ~64x the realised effect — because the two arms' fills sit at
    systematically different cycle phases, so an infeasible window tail scores PHASE, not skill.
    The argument for 13 rests on the tank derivation; the sweep corroborates it, and on its own
    would not pick a number.

    **`horizon_days` is cadence-independent but regret LEVELS ARE NOT** (PR #348 review finding
    #3). The window's length comes from the tank, but `summarise_regret` walks it on the run's
    own evaluation grid, so a 1-day-cadence run samples 14 candidate prices inside it and a
    7-day run samples 2. Measured on the same fps-6yi flips at a fixed H=13:

        cadence     1d              2d              7d
        base/cand   10.03 / 9.38    9.53 / 8.84     5.80 / 5.94
        delta       -0.644          -0.695          **+0.143**

    The level nearly halves and **the sign of the delta flips**. Regret is therefore comparable
    only within one cadence — never quote a regret figure without its `cadence_days`, and never
    compare regret across dossiers at different cadences. Same defect class as
    `cascade_window_days`' `n_decisions` quantisation caveat, reached by a different route.

    **Rounding-tie limitation (fps-o0h), and what `exact_fields` does and does not fix.**
    Without `exact_fields`, this reads the stamp, which is a DISPLAY format —
    `format_tank_params` renders daily consumption `:.3f`, so the default 3.5714285714285716
    arrives here as 3.571 and the feasible wait computes as 12.6015 rather than 12.6. Both
    round to 13, so neither committed stamp (`50/3.571/1d/10%`, `50/3.571/7d/10%`) is affected
    in this function or in `cascade_window_days`. Swept over the same 560-config grid
    `tests/test_exp_lib_flips.py` uses, **12** configs disagree with the same quantity computed
    from `TankParams` directly (e.g. `40/2.857/1d/25%` gives 11 here against 10 from the
    dataclass) — always on an exact rounding tie, never general float drift.

    **`exact_fields` does NOT remove rounding ties in general — it only removes the specific
    one caused by re-parsing the 3dp stamp** (PR #355 review finding #2, correcting this
    docstring's earlier claim that it "removes the tie entirely"; see finding #4 too — an
    earlier draft here also said "52 configs", which was `cascade_window_days`' count,
    transplanted). All 12 of THIS 560-config sweep's disagreements happen to resolve correctly
    with `exact_fields` (matching round-half-to-even on the true, exact ceiling) — but a wider
    sweep (`size` 20-120, `tank_life` 6-39, `floor` 1-50%) finds `exact_fields` landing on the
    WRONG side of the tie for roughly 11% of exact-.5 ceilings, for the same reason
    `cascade_window_days`' docstring documents: `max_feasible_wait_days` is `(1 - floor) * size
    / daily`, and when `daily` was itself constructed as `size / tank_life`, the round-trip
    `size / (size / tank_life)` is not always exactly `tank_life` in double precision — an
    unavoidable floating-point property of the STORED values, not something `TankParams` or
    `exact_fields` can correct after the fact. Passing `exact_fields` (`fuel_signal.backtest.
    tank_params_fields(tank)`'s dict, persisted alongside the stamp) still routes the
    computation through `TankParams.max_feasible_wait_days` — the single owner of this
    quantity, so at least both callers computing it agree with EACH OTHER — but "single owner"
    is not the same claim as "exact", and this docstring no longer conflates the two. Prefer
    `exact_fields` whenever the caller's meta carries it (it fixes the one well-understood,
    avoidable failure mode); fall back to the stamp only for results.json predating fps-o0h.

    **Separately, `round()` vs. `math.ceil()` (PR #355 review finding #5, pre-existing, out of
    scope here).** This function is documented as a "ceiling", but `round()` on an exact `.5`
    can round DOWN (banker's rounding), understating the ceiling it claims to be — `math.ceil`
    would resolve every tie upward instead, by definition rather than float parity. That
    question predates fps-o0h (this function used `round()` before `exact_fields` existed) and
    changing it would move MANY non-tie values too (any fractional ceiling `round()` currently
    rounds down), not just the tie cases — a real behaviour change to a locked function,
    needing its own decision and sign-off, not something to fold into a refactor whose own
    acceptance criterion is that the locked config's answer (13) must not move. Filed
    separately; not fixed here.
    """
    if exact_fields is not None:
        return round(_tank_from_exact_fields(exact_fields, tank_params).max_feasible_wait_days)
    size, daily, _cadence, floor = parse_tank_params(tank_params)
    return round((1.0 - floor) * size / daily)


def _finite_positive_spread(values: pd.Series, std: float) -> bool:
    """True iff `std` (an `np.std`/`.std(ddof=1)` over `values`) is a real, positive ruler.

    Same guard as `experiments/pipeline/dossier_tables.py`'s `_band_std_usable` (fps-tnz) and
    for the same reason: `std > 0` alone does not catch "every value is identical" — float
    noise from binary-representable rounding (e.g. `np.std` of twenty identical 0.01s is
    1.78e-18, not 0.0) sails past a bare `> 0` check. `np.ptp` is exact and zero iff the values
    really are identical. Not imported from dossier_tables.py: that module imports THIS one
    (flips.py is the lower layer), so the dependency can't run the other way, and the
    predicate is two lines — duplicating it here is cheaper than restructuring the import
    graph to share it.
    """
    return bool(np.isfinite(std) and std > 0 and float(np.ptp(values)) > 0)


def _price_dispersion(fold_rows: pd.DataFrame, all_rows: pd.DataFrame) -> tuple[float | None, str]:
    """Per-fill flip-price standard deviation to size a cell's standard error against
    (fps-e1w change 2), fold-local where the fold has enough flip fills to estimate one,
    run-wide otherwise — and run-wide again, explicitly unresolved, if even the run-wide
    spread is degenerate (fps-e1w's "SE estimator float-degeneracy trap" design note).

    Degeneracy must fall through to run-wide regardless of COUNT (a fold-local `s` from 20
    identical prices is still unusable at n=20) — `_finite_positive_spread`'s np.ptp guard
    is checked whether or not `MIN_FOLD_LOCAL_DISPERSION_N` was met, never skipped because
    the count looked adequate. A hairline `s` collapsing a cell's SE toward zero would print
    the tightest, most confident-looking interval in exactly the thinnest, least trustworthy
    cell — the opposite of what this column exists to show.

    Returns `(None, reason)` only when NEITHER fold-local nor run-wide dispersion is usable —
    the caller must print no interval at all in that case, not a hairline one.
    """
    return _dispersion(fold_rows, all_rows, "price", what="price")


def _dispersion(
    fold_rows: pd.DataFrame, all_rows: pd.DataFrame, column: str, *, what: str
) -> tuple[float | None, str]:
    """`_price_dispersion`'s column-generic body, shared with `summarise_regret`'s per-fill
    REGRET dispersion (fps-2js).

    Deliberately one function rather than two similar ones: the fold-local/run-wide fallback
    and — more importantly — the `_finite_positive_spread` float-degeneracy guard are the part
    that took a review round to get right (fps-e1w, and fps-tnz before it). A second estimator
    with its own copy of that logic is exactly how one of the two ends up missing the next fix.
    """
    if len(fold_rows) >= MIN_FOLD_LOCAL_DISPERSION_N:
        local_std = float(fold_rows[column].std(ddof=1))
        if _finite_positive_spread(fold_rows[column], local_std):
            return local_std, "fold-local"
    if len(all_rows) >= 2:
        run_std = float(all_rows[column].std(ddof=1))
        if _finite_positive_spread(all_rows[column], run_std):
            return run_std, "run-wide"
    return None, (
        f"{what} dispersion not estimable — every flip fill (fold-local and run-wide) landed "
        f"at the same {what}"
    )


def _effective_n(litres: pd.Series) -> float:
    """Kish's effective sample size for a litres-weighted arm: `(sum w)^2 / sum(w^2)`.

    Equal-weight fills recover the raw count; a litres distribution dominated by one large
    fill (a 42.86 L top-up beside a 3.57 L one) reads as far fewer independent draws than its
    row count, which is exactly the resolution question `flip_cpl_delta`'s SE needs answered
    (fps-e1w change 4 — pooled_cpl is spend-weighted, so a fill's SIZE matters, not just its
    existence). Zero when the arm has no litres (no flips on that side in this fold).
    """
    total = float(litres.sum())
    if total <= 0:
        return 0.0
    return total ** 2 / float((litres ** 2).sum())


def _cascade_group_ids(fold_flips: pd.DataFrame, window_days: int) -> pd.Series:
    """Decision-group id per row, index-aligned with `fold_flips`: flips at the SAME station
    within `window_days` of the previous one (either arm — a fold's flip stream is one shared
    timeline of divergence at that station, not two independent ones) share a group id, per
    `flips.py`'s module docstring on cascades. A gap strictly greater than the window starts a
    new decision.

    Single-sourced (PR #347 review finding #1) so `n_decisions` (`_collapse_cascades`, below)
    and each delta's decision-level standard error (`summarise_flips`) can never disagree
    about what counts as one decision — before this, the SE sized effective n on individual
    FILLS, silently contradicting `n_decisions`' own premise that a cascade isn't several
    independent draws: a fold could print `n_decisions=5` while its SE was still computed as
    though it had 30 independent observations, printing an interval systematically too narrow
    in exactly the over-confident direction the whole column exists to avoid.
    """
    if fold_flips.empty:
        return pd.Series(dtype="int64")
    dates = pd.to_datetime(fold_flips["date"])
    group_ids = pd.Series(-1, index=fold_flips.index, dtype="int64")
    next_id = 0
    for station in fold_flips["station_code"].unique():
        station_idx = fold_flips.index[fold_flips["station_code"] == station]
        prev_date = None
        current_group = -1
        for row_idx, date in dates.loc[station_idx].sort_values().items():
            if prev_date is None or (date - prev_date).days > window_days:
                current_group = next_id
                next_id += 1
            group_ids.loc[row_idx] = current_group
            prev_date = date
    return group_ids


def _collapse_cascades(fold_flips: pd.DataFrame, window_days: int) -> int:
    """Cascade-collapsed decision count for one fold's flips (fps-e1w change 3): `n_flips`'s
    cascade-collapsed sibling — see `_cascade_group_ids`."""
    return int(_cascade_group_ids(fold_flips, window_days).nunique())


def _decision_litres(side: pd.DataFrame, group_ids: pd.Series) -> pd.Series:
    """One arm's flip litres, summed per cascade-decision group rather than per fill (PR #347
    review finding #1) — the weights `_effective_n` must size that arm's standard error on.
    Several fills in the same cascade are one decision with one combined litres weight, not
    several independent draws; passing per-fill litres here would let a long cascade (many
    small fills, one continuing divergence) masquerade as many independent observations and
    understate the delta's own SE.
    """
    if side.empty:
        return side["litres"]
    return side["litres"].groupby(group_ids.loc[side.index]).sum()


def summarise_flips(
    fills: pd.DataFrame,
    baseline_arm: str,
    candidate_arm: str,
    shock_folds: set[int],
    *,
    window_days: int,
) -> dict:
    """Per-fold flip counts, flip-only pooled CPL, and each delta's own resolution (fps-e1w),
    plus the row-level detail.

    (Parameter named `window_days`, not `cascade_window_days` — PR #347 review finding #7: the
    module-level `cascade_window_days` function of that name would otherwise be shadowed
    inside this function's body, a trap for a future edit that tries to call it here.)

    `flip_cpl_baseline` / `flip_cpl_candidate` are `pooled_cpl` computed over ONLY the fills
    that differ between arms in that fold — NOT the same quantity as that fold's
    `delta_cpl_own` in facts["breakdowns"]["per_fold"] (which pools every fill, matching and
    differing alike). This is deliberately narrower: it answers "how expensive were just the
    decisions that changed," which is the aggregate delta can't answer on its own.

    **`flip_cpl_delta` cannot be read on its own** (fps-e1w). `flip_cpl_baseline` and
    `flip_cpl_candidate` pool DISJOINT fills on different days at different points in the
    price cycle — their difference has no stable denominator and explodes in thin cells. Each
    delta is emitted beside its own `flip_cpl_delta_se` / `flip_cpl_delta_interval`
    (`_price_dispersion` + `_effective_n`, sized on DECISION-level litres via
    `_decision_litres` — not per-fill litres, so a long cascade of small fills can't
    masquerade as many independent draws and understate its own SE), and
    `flip_cpl_delta_inside_own_se` marks the cells (routinely most of them) whose delta cannot
    be distinguished from zero at 2*its own SE — see `docs/routines/dossier.md`'s citation
    rule: such a cell is not evidence.

    `n_decisions` (`_collapse_cascades`) is `n_flips`'s cascade-collapsed sibling, both
    per-fold and summed at the top level: one real flip can cascade into a run of later
    differing fills at the same station, so `n_flips` overstates independent decisions — see
    this module's docstring.

    `rows` carries the full flip-level detail (`diff_fills`'s output, as records) so a
    committed facts.json is self-contained — fills.parquet itself is gitignored and never
    reaches a reader who didn't run this batch locally.
    """
    flips = diff_fills(fills, baseline_arm, candidate_arm)
    per_fold: list[dict] = []
    for fold in sorted(fills["fold"].unique()):
        fold_flips = flips[flips["fold"] == fold]
        base_side = fold_flips[fold_flips["bought_by"] == "baseline"]
        cand_side = fold_flips[fold_flips["bought_by"] == "candidate"]
        n_base, n_cand = len(base_side), len(cand_side)
        base_cpl = pooled_cpl(base_side) if n_base else None
        cand_cpl = pooled_cpl(cand_side) if n_cand else None
        litres_base = float(base_side["litres"].sum())
        litres_cand = float(cand_side["litres"].sum())
        group_ids = _cascade_group_ids(fold_flips, window_days)
        n_decisions = int(group_ids.nunique())

        delta = cand_cpl - base_cpl if base_cpl is not None and cand_cpl is not None else None
        delta_reason = None
        if delta is None:
            delta_reason = "no flips" if n_base == 0 and n_cand == 0 else "one arm only"
        elif not math.isfinite(delta):
            # base_cpl/cand_cpl are pooled_cpl's own NaN for a zero-litres side (PR #347
            # review finding #4) — not None, so the branch above never fires, and a raw NaN
            # would silently serialise to `null` (io.to_jsonable) with no reason attached,
            # the exact "blank cell with no explanation" this whole rework exists to remove.
            delta_reason = (
                "one or both arms have flip fills but zero total litres on that side — "
                "cost-per-litre is undefined"
            )
            delta = None

        se_diff, interval, inside_own_se, interval_reason = None, None, None, None
        if delta is not None:
            s, dispersion_note = _price_dispersion(fold_flips, flips)
            if s is None:
                interval_reason = dispersion_note
            else:
                n_eff_base = _effective_n(_decision_litres(base_side, group_ids))
                n_eff_cand = _effective_n(_decision_litres(cand_side, group_ids))
                se_base = s / math.sqrt(n_eff_base) if n_eff_base > 0 else None
                se_cand = s / math.sqrt(n_eff_cand) if n_eff_cand > 0 else None
                if se_base is not None and se_cand is not None:
                    se_diff = math.sqrt(se_base ** 2 + se_cand ** 2)
                    half_width = INTERVAL_SE_MULTIPLE * se_diff
                    interval = [delta - half_width, delta + half_width]
                    inside_own_se = bool(abs(delta) < half_width)
                else:
                    interval_reason = (
                        "price dispersion resolvable but one arm's decision-weighted "
                        "effective n is 0 — cannot size that arm's own standard error"
                    )

        per_fold.append({
            "fold": int(fold),
            "regime": "shock" if fold in shock_folds else "normal",
            "n_baseline_only": n_base,
            "n_candidate_only": n_cand,
            "n_flips": n_base + n_cand,
            "n_decisions": n_decisions,
            "litres_baseline": litres_base,
            "litres_candidate": litres_cand,
            "flip_cpl_baseline": base_cpl,
            "flip_cpl_candidate": cand_cpl,
            "flip_cpl_delta": delta,
            "flip_cpl_delta_reason": delta_reason,
            "flip_cpl_delta_se": se_diff,
            "flip_cpl_delta_interval": interval,
            "flip_cpl_delta_inside_own_se": inside_own_se,
            "flip_cpl_delta_interval_reason": interval_reason,
        })
    return {
        "n_flips": len(flips),
        "n_decisions": sum(row["n_decisions"] for row in per_fold),
        "cascade_window_days": window_days,
        "per_fold": per_fold,
        "rows": flips.to_dict(orient="records"),
    }


class StationPriceSource(Protocol):
    """The price accessor `summarise_regret` needs, mirroring `fuel_signal.backtest`'s
    `PriceHistory` rather than inventing a second convention.

    `price_at` MUST forward-fill ("latest price on or before as_of", `PriceHistory.
    station_price_at`'s exact contract) — that is the price path the tank simulator itself
    bought at, so anything else scores the arms against a series they never saw. `is_observed`
    is the un-filled question ("did the station actually report on this date"), used only to
    count and report the dark days, never to drop a fill.

    A Protocol rather than a concrete class so this module stays DB-free: the dossier builds
    one over sqlite, and tests build one over a dict.
    """

    def price_at(self, station_code: int, as_of: str) -> float | None: ...

    def is_observed(self, station_code: int, as_of: str) -> bool: ...


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    """Litres-weighted mean, or None when the arm carries no litres.

    Litres-weighted, not row-weighted, for the same reason `pooled_cpl` is spend-weighted: a
    3.57 L top-up and a 42.86 L fill are not two equal observations of how well the strategy
    bought, and the run-level number has to answer "per litre actually purchased".
    """
    total = float(weights.sum())
    if total <= 0:
        return None
    return float((values * weights).sum() / total)


def _regret_stats(
    side: pd.DataFrame, group_ids: pd.Series, s: float
) -> tuple[float | None, float | None]:
    """One arm's litres-weighted regret and its decision-level standard error.

    `n_eff` is Kish's effective n over CASCADE-COLLAPSED litres (`_decision_litres`), never
    per-fill litres — the same correction PR #347's review forced on `flip_cpl_delta`'s SE,
    applied here from the start rather than rediscovered. A cascade is one decision replayed
    forward; counting its fills as independent draws understates the interval in exactly the
    over-confident direction.
    """
    if side.empty:
        return None, None
    mean = _weighted_mean(side["regret"], side["litres"])
    n_eff = _effective_n(_decision_litres(side, group_ids))
    se = s / math.sqrt(n_eff) if n_eff > 0 else None
    return mean, se


def summarise_regret(
    flips: pd.DataFrame,
    prices: "StationPriceSource",
    shock_folds: set[int],
    *,
    horizon_days: int,
    cadence_days: int,
    window_days: int,
) -> dict:
    """Per-arm TIMING REGRET over the flipped fills — `price paid - the cheapest price the
    same station reached within the tank's feasible wait` (fps-2js). 0 means the arm bought
    the best price it could actually have reached; lower is better.

    **Why this exists beside `flip_cpl_delta`.** `flip_cpl_baseline`/`flip_cpl_candidate` pool
    two DISJOINT sets of fills bought on different days at different points in the price cycle,
    so their difference has no stable denominator and explodes in thin cells (fps-e1w, and
    `feedback_disjoint_basket_comparison`). Regret is defined identically on both arms and is
    cycle-normalised by construction — "how much better could this fill have done, from where
    it stood" is comparable between a 172.90 fill in Nov 2024 and a 205.90 one in May 2024,
    which their raw prices are not. Three consequences, all measured on fps-6yi:

    - **It pools.** There is a run-level ALL row (baseline 10.03 vs candidate 9.38 c/L at
      H=13), which a difference of disjoint baskets can never produce, and it agrees with the
      realised headline that the arms are a dead heat. `flip_cpl_delta` is not additive and
      has no such row.
    - **Dispersion falls by ~38%** — per-fill regret sd 8.49 c/L against the raw flip-price
      sd of 13.72 the old SE was sized on.
    - **Thin cells read truthfully.** fps-6yi's fold 13, whose `flip_cpl_delta` prints a much-
      quoted -32.73 c/L off ONE candidate fill, reads baseline 5.73 / candidate 0.00: the
      candidate bought the exact optimum. Same fill, a description that survives contact with
      its own sample size.

    **This is a WHEN measure, not a WHAT measure — same station only, deliberately.** The
    counterfactual is that same station's own forward prices, so regret scores timing and
    nothing else. A cross-station variant ("cheapest among the preferred set") was considered
    and rejected: it would fold station choice into the same number and destroy the one
    decomposition this table is for — fps-6yi's fold 4 contributes +0.0437 c/L to the headline
    (unfavourable) while its regret says the candidate timed 0.26 c/L BETTER, and reading that
    as "the difference came from WHAT was bought, not WHEN" only works while regret is purely
    a WHEN measure.

    **Dark days are FORWARD-FILLED, and that is not a policy choice** — it is the price path
    the arms actually faced. `fuel_signal.backtest.PriceHistory.station_price_at` returns "the
    latest price on or before as_of" (its own docstring: "station_price_at forward-fills, so
    None occurs only as a leading prefix in practice"), so the tank simulator bought at the
    carried price on days the station reported nothing. `prices` must implement those same
    semantics. Scoring regret against observed-only prices would compare each arm to a series
    it never saw AND drop fills asymmetrically — on fps-6yi, 56 of 303 flips fall on dark days
    (all station 414, folds 4-7; 42 candidate vs 14 baseline, 364 vs 207 litres), so dropping
    them would reintroduce the disjoint-basket defect through the back door. The count is
    reported as `dark_fill_days` because the outage is non-random (it thins and tilts specific
    folds) even though it no longer biases the estimate.

    `prices` is any object with `price_at(station_code, as_of) -> float | None` and
    `is_observed(station_code, as_of) -> bool`; the window is walked on the run's OWN
    `cadence_days` grid anchored at the fill date (a fill only ever lands on an evaluation
    date), because those are the days the strategy could actually have bought on.

    The reference price is `min(price paid, cheapest reachable)`, so regret is never negative
    even if a ledger price and its DB row disagree; `n_price_mismatch` counts any such
    disagreement rather than letting it vanish into the floor.
    """
    if cadence_days > horizon_days:
        # Only offset 0 would be evaluated, so every regret is exactly 0 and the table prints a
        # resolved-looking dead heat that is pure arithmetic (PR #348 review finding #5). No
        # live config reaches this — the horizon is a whole tank-life and the cadence is days —
        # but a silently-zero table is the worst possible failure for a measure whose entire
        # purpose is to stop unreadable numbers being read.
        raise ValueError(
            f"cadence_days={cadence_days} exceeds horizon_days={horizon_days}: the feasible "
            "wait is shorter than one evaluation step, so no alternative buy day exists and "
            "every regret would be a meaningless 0."
        )
    per_fold: list[dict] = []
    scored = flips.copy()
    regrets: list[float | None] = []
    dark_rows, mismatches = [], 0
    for _, row in flips.iterrows():
        station, paid = int(row["station_code"]), float(row["price"])
        as_of = pd.Timestamp(row["date"])
        dark_rows.append(not prices.is_observed(station, as_of.strftime("%Y-%m-%d")))
        reachable = []
        fill_day_price = None
        offset = 0
        while offset <= horizon_days:
            price = prices.price_at(station, (as_of + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"))
            if price is not None:
                if offset == 0:
                    fill_day_price = float(price)
                reachable.append(float(price))
            offset += cadence_days
        if not reachable:
            # station_price_at only returns None before a station's first ever price, so this
            # is a fill dated ahead of its own station's series — a broken ledger, not an
            # outage. Score nothing rather than guess.
            regrets.append(None)
            continue
        # Explicitly the OFFSET-0 price, not reachable[0]: `reachable` skips None lookups, so
        # on a fill dated before its station's first price reachable[0] is some later
        # evaluation date's price, and comparing the ledger against that would report a
        # data-integrity mismatch that is really just an out-of-range fill date.
        if fill_day_price is not None and abs(fill_day_price - paid) > 1e-6:
            mismatches += 1
        regrets.append(paid - min(paid, min(reachable)))
    scored["regret"] = regrets
    scored["dark"] = dark_rows
    resolvable = scored[scored["regret"].notna()]

    for fold in sorted(flips["fold"].unique()) if not flips.empty else []:
        fold_rows = resolvable[resolvable["fold"] == fold]
        base_side = fold_rows[fold_rows["bought_by"] == "baseline"]
        cand_side = fold_rows[fold_rows["bought_by"] == "candidate"]
        group_ids = _cascade_group_ids(fold_rows, window_days)
        s, dispersion_note = _dispersion(fold_rows, resolvable, "regret", what="regret")
        row = _regret_row(base_side, cand_side, group_ids, s, dispersion_note)
        row.update({
            "fold": int(fold),
            "regime": "shock" if fold in shock_folds else "normal",
            # NOTE the different basis from summarise_flips' same-named field (PR #348 review
            # finding #6): that one collapses EVERY flip, this one only the SCORED ones, so the
            # two can differ for the same fold whenever a flip could not be scored.
            # `n_unscored` beside it is what lets a reader reconcile them.
            "n_decisions": int(group_ids.nunique()),
            "n_unscored": int(
                len(scored[(scored["fold"] == fold) & scored["regret"].isna()])
            ),
        })
        per_fold.append(row)

    all_groups = _cascade_group_ids_run_wide(resolvable, window_days)
    s_all, note_all = _dispersion(resolvable, resolvable, "regret", what="regret")
    all_row = _regret_row(
        resolvable[resolvable["bought_by"] == "baseline"],
        resolvable[resolvable["bought_by"] == "candidate"],
        all_groups, s_all, note_all,
    )
    all_row["n_decisions"] = int(all_groups.nunique()) if len(all_groups) else 0
    return {
        "horizon_days": horizon_days,
        "cadence_days": cadence_days,
        "n_scored": int(len(resolvable)),
        "n_unscored": int(len(scored) - len(resolvable)),
        "dark_fill_days": int(scored["dark"].sum()),
        "dark_fill_folds": sorted(int(f) for f in scored.loc[scored["dark"], "fold"].unique()),
        "n_price_mismatch": mismatches,
        "per_fold": per_fold,
        "all": all_row,
    }


def _cascade_group_ids_run_wide(rows: pd.DataFrame, window_days: int) -> pd.Series:
    """`_cascade_group_ids` applied per fold and re-offset so ids stay unique run-wide.

    The ALL row's standard error has to be sized on the run's whole decision population, and
    `_cascade_group_ids` allocates ids fold-locally — concatenating them without an offset
    would merge unrelated folds' cascades into one group and understate `n_decisions`.
    """
    if rows.empty:
        return pd.Series(dtype="int64")
    out = pd.Series(-1, index=rows.index, dtype="int64")
    offset = 0
    for _, fold_rows in rows.groupby("fold"):
        ids = _cascade_group_ids(fold_rows, window_days)
        out.loc[fold_rows.index] = ids + offset
        offset += int(ids.max()) + 1
    return out


def _regret_row(
    base_side: pd.DataFrame,
    cand_side: pd.DataFrame,
    group_ids: pd.Series,
    s: float | None,
    dispersion_note: str,
) -> dict:
    """One cell (a fold, or the ALL row): both arms' regret, their difference, and the
    difference's own 2*SE — the same "print the interval or print nothing" contract
    `summarise_flips` follows, so the two tables cannot be read to different standards."""
    base_regret, se_base = _regret_stats(base_side, group_ids, s) if s is not None else (
        _weighted_mean(base_side["regret"], base_side["litres"]) if not base_side.empty else None, None
    )
    cand_regret, se_cand = _regret_stats(cand_side, group_ids, s) if s is not None else (
        _weighted_mean(cand_side["regret"], cand_side["litres"]) if not cand_side.empty else None, None
    )
    delta = cand_regret - base_regret if base_regret is not None and cand_regret is not None else None
    reason = None
    if delta is None:
        if base_side.empty and cand_side.empty:
            reason = "no scored flips"
        elif base_side.empty or cand_side.empty:
            reason = "one arm only"
        else:
            # BOTH arms flipped, but a litres-weighted mean is undefined on a zero-litres side
            # — "one arm only" would be a false statement about a fold that did flip on both,
            # and docs/routines/dossier.md tells authors to quote this string verbatim. Same
            # branch summarise_flips grew in PR #347's review (finding #4).
            reason = (
                "both arms have scored flips but one side carries zero total litres — a "
                "litres-weighted regret is undefined there"
            )
    se_diff, interval, inside, interval_reason = None, None, None, None
    if delta is not None:
        if s is None:
            interval_reason = dispersion_note
        elif se_base is None or se_cand is None:
            # Defensive, and unreachable as written: an arm's effective n is 0 only when its
            # litres sum to 0, which already made its _weighted_mean None and so `delta` None
            # above. Kept so the branch exists if either side's definition ever moves apart —
            # the alternative is a silent None interval with no reason at all.
            interval_reason = (
                "regret dispersion resolvable but one arm's decision-weighted effective n "
                "is 0 — cannot size that arm's own standard error"
            )
        else:
            se_diff = math.sqrt(se_base ** 2 + se_cand ** 2)
            half_width = INTERVAL_SE_MULTIPLE * se_diff
            interval = [delta - half_width, delta + half_width]
            inside = bool(abs(delta) < half_width)
    return {
        "n_baseline": int(len(base_side)),
        "n_candidate": int(len(cand_side)),
        "litres_baseline": float(base_side["litres"].sum()),
        "litres_candidate": float(cand_side["litres"].sum()),
        "regret_cpl_baseline": base_regret,
        "regret_cpl_candidate": cand_regret,
        "regret_cpl_delta": delta,
        "regret_cpl_delta_reason": reason,
        "regret_cpl_delta_se": se_diff,
        "regret_cpl_delta_interval": interval,
        "regret_cpl_delta_inside_own_se": inside,
        "regret_cpl_delta_interval_reason": interval_reason,
    }
