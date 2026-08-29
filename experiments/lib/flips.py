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

import numpy as np
import pandas as pd

from experiments.lib.zones import pooled_cpl

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


def cascade_window_days(tank_params: str) -> int:
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

    `tank_params` is the formatted stamp `fuel_signal.backtest.format_tank_params` writes
    (`f"{size}/{daily}/{interval}d/{floor}%"`) — only the first two slash-separated fields are
    used here; parsing is safe because that function is this string's only writer
    (`fuel_signal.backtest.require_tank_stamp` is the only sanctioned way to obtain one).
    """
    size_str, daily_str = tank_params.split("/")[:2]
    return round(float(size_str) / float(daily_str) / 2)


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
    if len(fold_rows) >= MIN_FOLD_LOCAL_DISPERSION_N:
        local_std = float(fold_rows["price"].std(ddof=1))
        if _finite_positive_spread(fold_rows["price"], local_std):
            return local_std, "fold-local"
    if len(all_rows) >= 2:
        run_std = float(all_rows["price"].std(ddof=1))
        if _finite_positive_spread(all_rows["price"], run_std):
            return run_std, "run-wide"
    return None, (
        "price dispersion not estimable — every flip fill (fold-local and run-wide) landed "
        "at the same price"
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


def _collapse_cascades(fold_flips: pd.DataFrame, window_days: int) -> int:
    """Cascade-collapsed decision count for one fold's flips (fps-e1w change 3): flips at the
    SAME station within `window_days` of the previous one (either arm — a fold's flip stream
    is one shared timeline of divergence at that station, not two independent ones) count as
    one decision, per `flips.py`'s module docstring on cascades. A gap strictly greater than
    the window starts a new decision.
    """
    if fold_flips.empty:
        return 0
    dates = pd.to_datetime(fold_flips["date"])
    decisions = 0
    for _, station_dates in dates.groupby(fold_flips["station_code"].to_numpy()):
        prev = None
        for date in sorted(station_dates):
            if prev is None or (date - prev).days > window_days:
                decisions += 1
            prev = date
    return decisions


def summarise_flips(
    fills: pd.DataFrame,
    baseline_arm: str,
    candidate_arm: str,
    shock_folds: set[int],
    *,
    cascade_window_days: int,
) -> dict:
    """Per-fold flip counts, flip-only pooled CPL, and each delta's own resolution (fps-e1w),
    plus the row-level detail.

    `flip_cpl_baseline` / `flip_cpl_candidate` are `pooled_cpl` computed over ONLY the fills
    that differ between arms in that fold — NOT the same quantity as that fold's
    `delta_cpl_own` in facts["breakdowns"]["per_fold"] (which pools every fill, matching and
    differing alike). This is deliberately narrower: it answers "how expensive were just the
    decisions that changed," which is the aggregate delta can't answer on its own.

    **`flip_cpl_delta` cannot be read on its own** (fps-e1w). `flip_cpl_baseline` and
    `flip_cpl_candidate` pool DISJOINT fills on different days at different points in the
    price cycle — their difference has no stable denominator and explodes in thin cells. Each
    delta is emitted beside its own `flip_cpl_delta_se` / `flip_cpl_delta_interval`
    (`_price_dispersion` + `_effective_n`), and `flip_cpl_delta_inside_own_se` marks the cells
    (routinely most of them) whose delta cannot be distinguished from zero at 2*its own SE —
    see `docs/routines/dossier.md`'s citation rule: such a cell is not evidence.

    `n_decisions` (`_collapse_cascades`) is `n_flips`'s cascade-collapsed sibling: one real
    flip can cascade into a run of later differing fills at the same station, so `n_flips`
    overstates independent decisions — see this module's docstring.

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

        delta = cand_cpl - base_cpl if base_cpl is not None and cand_cpl is not None else None
        delta_reason = None
        if delta is None:
            delta_reason = "no flips" if n_base == 0 and n_cand == 0 else "one arm only"

        se_diff, interval, inside_own_se, interval_reason = None, None, None, None
        if delta is not None:
            s, dispersion_note = _price_dispersion(fold_flips, flips)
            if s is None:
                interval_reason = dispersion_note
            else:
                n_eff_base = _effective_n(base_side["litres"])
                n_eff_cand = _effective_n(cand_side["litres"])
                se_base = s / math.sqrt(n_eff_base) if n_eff_base > 0 else None
                se_cand = s / math.sqrt(n_eff_cand) if n_eff_cand > 0 else None
                if se_base is not None and se_cand is not None:
                    se_diff = math.sqrt(se_base ** 2 + se_cand ** 2)
                    half_width = INTERVAL_SE_MULTIPLE * se_diff
                    interval = [delta - half_width, delta + half_width]
                    inside_own_se = bool(abs(delta) < half_width)
                else:
                    interval_reason = (
                        "price dispersion resolvable but one arm's litres-weighted "
                        "effective n is 0 — cannot size that arm's own standard error"
                    )

        per_fold.append({
            "fold": int(fold),
            "regime": "shock" if fold in shock_folds else "normal",
            "n_baseline_only": n_base,
            "n_candidate_only": n_cand,
            "n_flips": n_base + n_cand,
            "n_decisions": _collapse_cascades(fold_flips, cascade_window_days),
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
        "cascade_window_days": cascade_window_days,
        "per_fold": per_fold,
        "rows": flips.to_dict(orient="records"),
    }
