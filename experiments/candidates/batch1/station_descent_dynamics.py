"""batch1 candidate — the station's OWN descent trajectory (fps-3jj.11).

Why this exists: every dynamic column in the lock is NETWORK-level. Measured on the
live frame, `cycle_pct_through`, `cycle_days_since_peak`, `cycle_mean_length`,
`cycle_last_min_cents`, `cycle_last_max_cents`, `cycle_peak_count`, `network_px_std*`
and `lga_phase_std*` all take exactly ONE distinct value per date (max distinct/date =
1). The only station-varying locked columns are contemporaneous LEVEL differences
(`station_price_cents`, `station_minus_*`, `lga_mean_cents`, `brand_mean_cents`,
`stickiness_score`) — none of them is a rate. So the model currently has no idea how
fast the station in front of it is cutting, only where it sits today.
"""
from __future__ import annotations

import pandas as pd

from experiments.lib.features.diagnostics import px_change_lag_diagnostic

NAME = "station_descent_dynamics"

HYPOTHESIS = (
    "The retail descent is a sequence of discrete cuts, and the SHAPE of a station's "
    "own cut sequence says how much of the descent is left. A station still cutting "
    "1-2 c/day is mid-descent; a station whose recent pace has collapsed toward zero "
    "while its 14-day pace was steep has arrived at its floor and is now exposed to "
    "the restoration. The lock cannot express this: every dynamic column in it is "
    "network-level (one value per date), so the model sees the Sydney-average clock "
    "and each station's contemporaneous LEVEL, but never the station's own rate of "
    "change. station_minus_last_min_cents is the closest substitute and is a distance "
    "to a NETWORK trough, not a station velocity."
)

PREDICTED_SIGNATURE = (
    "Decision-timing, not accuracy: the model should stop waiting earlier on rows "
    "where station_descent_decel has gone positive (recent pace slower than the "
    "trailing pace) at moderate cycle_pct_through, and should keep waiting longer on "
    "rows where it is negative (the cuts are still accelerating). Expect a MODEST "
    "number of buy/wait flips concentrated in elongated cycles, where the "
    "network clock and the station's own pace diverge most — i.e. exactly the "
    "shallow-vs-steep-within-elongated distinction the #214 oracle diagnostic found "
    "invisible in network_px_std until phase > 1.4. If the mechanism is real, the "
    "per-fold contributions should be largest in folds 3 and 7 (the two folds the "
    "ledger names on the elongation axis) and near-zero elsewhere; a uniform "
    "improvement across all 14 folds would be evidence the columns are working as "
    "generic momentum rather than as a descent-shape reading."
)

# P(pooled realised CPL delta < 0). The named missing axis in the ledger ("the missing
# axis is descent SHAPE, not elongation") and the only one of the five that attacks it
# with data the lock structurally does not contain. Discounted because three adjacent
# attempts on this corner already failed (#214 R_raw, #231 A_x_shallow_elong,
# is_extended_shallow_descent) — but all three built the fix out of family A itself,
# which is the thing the oracle diagnostic showed is misleading in the corner.
CONFIDENCE_EFFECT = 0.45

# P(it concentrates in folds 3/7). Weaker than the effect prior: fold 3 is the ledger's
# counterexample fold (A helps extended descent strongly there) and fold 7 is the
# shallow-elongated problem fold, but "the elongation folds" is inferred from prior
# per-fold readings of a DIFFERENT feature, not measured for this one.
CONFIDENCE_ZONE = 0.30

# Fold-expressed zone claim, deliberately: a row-level cycle-phase axis cannot carry a
# COST claim (path-coupled pooled CPL, fps-grp), and folds can. 3 and 7 are the two
# folds the ledger names on the elongation/descent-shape axis. axis left as the built-in
# "regime" with an EMPTY expect_concentration_in so only the fold claim is graded --
# note "regime" here is the RESERVED FOLD-LEVEL name (shock vs non-shock), not
# zones.assign_regime()'s row-level cycle phase.
TARGET = {"axis": "regime", "expect_concentration_in": [], "folds": [3, 7]}

MECHANISM_FAMILY = "station-own-price-dynamics"

PRIOR_ART = (
    "Adjacent to three rejected attempts on the shallow-elongated corner, all of which "
    "built their fix out of family A (network_px_std): elongation_ratio + "
    "cycle_descent_slope_so_far (R_raw, #214 — rejected, fold 7 +0.0092), "
    "is_extended_shallow_descent (null, ~-0.0001), and A_x_shallow_elong/A_x_other/"
    "A_x_smooth (#231 — rejected and BACKWARDS, fold 7 went +0.018 -> +0.040). What "
    "differs mechanically: all three supplied the model with derived NETWORK quantities "
    "(dispersion, network cycle elongation, network descent slope). The ledger's own "
    "not_tested on the A_x_* entry says the open ground is 'whether any OTHER "
    "intra-series signal carries the shallow-vs-steep-within-elongated distinction — "
    "the oracle diagnostic tested A specifically and is silent about trough-proximity, "
    "cross-LGA/brand late-descent consensus...'. These columns are neither A nor a "
    "network aggregate: they are the individual station's own cut sequence, a quantity "
    "with no representative at all in the 54-column lock. Also adjacent to "
    "cycle_mean_length (rejected, accuracy != objective): note this candidate makes no "
    "claim to estimate the cycle better, only to change WHEN the model acts."
)

COLUMNS = [
    "station_px_change_3d",
    "station_px_change_14d",
    "station_descent_decel",
]

INPUTS = ["station_code", "price_date", "station_price_cents"]

#: Short leg (recent pace) and long leg (cycle-scale pace) of the deceleration term.
_SHORT_DAYS = 3
_LONG_DAYS = 14


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station calendar-aware price changes plus their curvature term.

    PIT-safe: px_change_lag_diagnostic is an exact-date backward self-merge on
    (station_code, price_date - k). It never reads a date after the row's own, and
    returns NaN rather than silently spanning a data gap, so truncating the frame
    cannot change any surviving row's value.

    station_descent_decel is the SHORT change minus the trailing pace rescaled to the
    short window. Positive = the recent cuts are smaller than the 14-day pace implies
    (the descent is flattening out); negative = the cuts are accelerating. It is an
    exact linear combination of the other two columns and is supplied anyway on
    purpose: a gradient-boosted tree splits on axis-aligned thresholds and cannot form
    a linear combination of two features, so the difference has to be handed over
    explicitly to be usable (docs/CONVENTIONS.md, tree interaction limits).
    """
    out = df.copy()

    # px_change_lag_diagnostic merges on the date column as-is, so it must be datetime;
    # the frame stores price_date as a string. Do the conversion on a scratch frame so
    # the returned frame's own price_date dtype is left exactly as handed in.
    work = df[["station_code", "station_price_cents"]].copy()
    work["price_date"] = pd.to_datetime(df["price_date"])

    short = px_change_lag_diagnostic(work, _SHORT_DAYS)
    long = px_change_lag_diagnostic(work, _LONG_DAYS)

    out["station_px_change_3d"] = short
    out["station_px_change_14d"] = long
    out["station_descent_decel"] = short - (_SHORT_DAYS / _LONG_DAYS) * long
    return out
