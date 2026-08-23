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
changes future tank state), not noise to average away.
"""
from __future__ import annotations

import pandas as pd

from experiments.lib.zones import pooled_cpl

FLIP_KEY_COLUMNS = ["fold", "station_code", "date"]
FLIP_ROW_COLUMNS = FLIP_KEY_COLUMNS + ["price", "litres", "spend_cents", "bought_by"]


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


def summarise_flips(
    fills: pd.DataFrame, baseline_arm: str, candidate_arm: str, shock_folds: set[int],
) -> dict:
    """Per-fold flip counts and flip-only pooled CPL, plus the row-level detail.

    `flip_cpl_baseline` / `flip_cpl_candidate` are `pooled_cpl` computed over ONLY
    the fills that differ between arms in that fold — NOT the same quantity as
    that fold's `delta_cpl_own` in facts["breakdowns"]["per_fold"] (which pools
    every fill, matching and differing alike). This is deliberately narrower: it
    answers "how expensive were just the decisions that changed," which is the
    question a fold-aggregate delta can't answer on its own — it says nothing
    about the (usually larger) set of fills both arms agree on.

    `rows` carries the full flip-level detail (`diff_fills`'s output, as
    records) so a committed facts.json is self-contained — fills.parquet
    itself is gitignored and never reaches a reader who didn't run this
    batch locally.
    """
    flips = diff_fills(fills, baseline_arm, candidate_arm)
    per_fold: list[dict] = []
    for fold in sorted(fills["fold"].unique()):
        fold_flips = flips[flips["fold"] == fold]
        base_side = fold_flips[fold_flips["bought_by"] == "baseline"]
        cand_side = fold_flips[fold_flips["bought_by"] == "candidate"]
        base_cpl = pooled_cpl(base_side) if len(base_side) else None
        cand_cpl = pooled_cpl(cand_side) if len(cand_side) else None
        per_fold.append({
            "fold": int(fold),
            "regime": "shock" if fold in shock_folds else "normal",
            "n_baseline_only": len(base_side),
            "n_candidate_only": len(cand_side),
            "flip_cpl_baseline": base_cpl,
            "flip_cpl_candidate": cand_cpl,
            "flip_cpl_delta": (
                cand_cpl - base_cpl if base_cpl is not None and cand_cpl is not None else None
            ),
        })
    return {
        "n_flips": len(flips),
        "per_fold": per_fold,
        "rows": flips.to_dict(orient="records"),
    }
