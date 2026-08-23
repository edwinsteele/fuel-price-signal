"""batch1 candidate — wholesale displacement measured over the RETAIL cycle (fps-3jj.11).

The ledger's highest-rated still-open lead: station_minus_tgp_cents was rejected on the
SHOCK axis, but the oracle diagnostic that fed that rejection also measured
gap -> depth-remaining at r2 = 0.63, and "how much further is there to fall" is a
DEPTH question, not a shock question. That regime-segmented test was never run (bd
fps-x0f). This candidate asks the depth question with a differently-anchored statistic.
"""
from __future__ import annotations

import pandas as pd

NAME = "tgp_cycle_displacement"

HYPOTHESIS = (
    "The retail trough is bounded below by the wholesale floor, so how far the floor "
    "has MOVED since this cycle's retail peak sets how deep this particular trough can "
    "go. A cycle that began when the floor was 6 c/L higher than it is now has room for "
    "a deeper trough than a cycle of identical length and phase whose floor never "
    "moved. Neither of the two TGP expressions tested so far asks this. "
    "station_minus_tgp_cents reads today's floor LEVEL (accurate when the floor is "
    "stable, misleading when it is moving — which is precisely how it failed). "
    "tgp_delta_7d reads the floor's velocity over a FIXED 7-day window that has no "
    "relationship to where the retail cycle is. Anchoring the displacement to the "
    "cycle's own peak makes it a depth-remaining statistic instead of a level or a "
    "momentum term."
)

PREDICTED_SIGNATURE = (
    "Decision-timing: on rows where the floor has fallen a long way since the peak, the "
    "model should WAIT past the point the lock would buy, because the trough it is "
    "approaching is deeper than the network clock implies; where the floor has been "
    "flat or has risen since the peak, it should buy earlier. Expect the effect to "
    "concentrate on NON-SHOCK folds — the mechanism is about the floor setting trough "
    "depth in an ordinary cycle, and the shock folds are exactly where the previous TGP "
    "gap candidate was measured to HURT (+0.011 log-loss). A shock-concentrated win "
    "here would therefore be evidence against the stated mechanism rather than for it, "
    "and should be read as such. Magnitude expectation is small: results.csv history "
    "puts single-column features at 0.03-0.26 c/L and both prior TGP expressions landed "
    "inside the noise band."
)

# P(pooled realised CPL delta < 0). Deliberately the LOWEST of the batch. The TGP series
# has now been measured twice and carried nothing either time (station_minus_tgp_cents
# rejected; tgp_delta_7d inconclusive at +0.0059 c/L, 40th percentile of pure fit
# noise), and the reconstruction below can only recover a 7-day-smoothed floor path, not
# the raw series. Set above 0.2 only because the axis this targets is genuinely
# untested and the r2 = 0.63 gap->depth-remaining measurement is real.
CONFIDENCE_EFFECT = 0.30

# P(it concentrates on non-shock folds), scored only if EFFECT resolves true. HIGHER
# than the effect prior on purpose: I am much more confident about WHERE this would show
# up than about WHETHER it shows up, because the shock-axis behaviour of the TGP gap is
# already measured and points the other way.
CONFIDENCE_ZONE = 0.55

# Fold-level "regime" (shock vs non-shock), the reserved name -- not the row-level
# cycle-phase axis that shares the level name "normal".
TARGET = {"axis": "regime", "expect_concentration_in": ["normal"], "folds": []}

MECHANISM_FAMILY = "wholesale-lead"

PRIOR_ART = (
    "Adjacent to two ledger entries on the same upstream series. (1) "
    "station_minus_tgp_cents — REJECTED: helps calm (-0.012) but hurts shock (+0.011), "
    "because today's TGP is accurate when the floor is stable and misleading when it is "
    "moving; its ledger not_tested says outright 'AND THE OPEN ONE: whether the gap "
    "helps on LATE-DESCENT rows specifically. It was rejected on the SHOCK axis, which "
    "is not the axis its mechanism targets — the same oracle diagnostic that fed this "
    "rejection also measured gap -> depth-remaining at r2=0.63.' (2) tgp_delta_7d — "
    "INCONCLUSIVE (+0.0059 c/L on the corrected 54-column baseline; the June 2026 "
    "graduation retracted 2026-08-19); its not_tested names 'horizons between 7 and 14 "
    "days' and 'whether ANY TGP expression helps on the LATE-DESCENT axis specifically'. "
    "What differs mechanically: this is neither a level nor a fixed-window velocity. It "
    "is a DISPLACEMENT between two dates chosen by the retail cycle itself, so it is "
    "immune to the level's stability problem (it is a difference, so the unknown "
    "reconstruction constant cancels) and its horizon is the cycle's age rather than an "
    "arbitrary 7 days. It is also NOT the gap: it never compares a retail price to a "
    "wholesale price, so it cannot reproduce the floor-proximity brake that cancelled "
    "velocity's pre-spike buy in gap_x_vel7 (also rejected). Explicit reparameterisation "
    "check: correlation against tgp_delta_7d itself is reported in the batch record; "
    "note the screen's predictor set is the LOCK ONLY, which excludes tgp_delta_7d "
    "(NON_MODEL_COLUMNS, evaluated-inconclusive) — see PR #331 for why that is the right "
    "set and what the old superset did to exactly this ground."
)

COLUMNS = [
    "tgp_cycle_displacement_cents",
    "tgp_cycle_displacement_frac_amp",
]

INPUTS = [
    "price_date",
    "tgp_delta_7d",
    "cycle_days_since_peak",
    "cycle_last_max_cents",
    "cycle_last_min_cents",
]

#: tgp_delta_7d is tgp(d) - tgp(d-7) (features.TGP_DELTA_DAYS), so dividing by 7 and
#: cumulating telescopes exactly into the 7-day trailing MEAN of the floor:
#:     sum_{u=a+1..b} [tgp(u) - tgp(u-7)] / 7  ==  MA7(b) - MA7(a).
#: The identity is exact, not an approximation -- see add_columns.
_TGP_DELTA_DAYS = 7

#: Guard for the amplitude denominator. The smallest cycle amplitude on the live frame
#: is 12.17 c/L (1st percentile 13.51), so this floor is three orders of magnitude below
#: anything observed and exists only so a degenerate frame cannot produce an infinity.
_MIN_AMPLITUDE_CENTS = 0.01


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Wholesale-floor displacement between this cycle's retail peak and today.

    The frame carries the floor's 7-day CHANGE (tgp_delta_7d), never its level, and no
    new dataset may be pulled in. The level is not needed: writing S(t) for the trailing
    7-day sum of the floor, tgp(u) - tgp(u-7) = S(u) - S(u-1), so cumulating
    tgp_delta_7d/7 from any origin telescopes to MA7(t) - MA7(origin). The origin's
    unknown constant is common to every row and cancels in the difference taken below,
    so the output is an exact 7-day-smoothed floor displacement rather than a proxy.

    PIT-safe: the cumulative sum runs strictly backwards from the frame's first date and
    the peak date is price_date - cycle_days_since_peak, always in the past. Truncating
    the frame removes later dates only, leaving every earlier prefix sum untouched, so
    surviving values are bit-identical. Note also that the CONSUMED column is itself
    PIT-strict upstream (features._tgp_delta_by_date builds it from
    tgp_series.asfreq('D').ffill().shift(1)), so today's floor never enters today's row.
    """
    out = df.copy()
    dates = pd.to_datetime(df["price_date"])

    # One row per date -- tgp_delta_7d and the cycle columns are all date-level
    # (verified on the live frame: max 1 distinct value per date).
    per_date = (
        pd.Series(df["tgp_delta_7d"].to_numpy(), index=dates.values)
        .groupby(level=0)
        .first()
        .sort_index()
    )
    # Reindex to a contiguous daily range so a missing calendar day cannot silently
    # collapse two steps into one; ffill is backward-looking and keeps the telescoping
    # well defined. The live frame has every calendar day present, so this is a no-op
    # there and a guard for a sampled/partial frame.
    full_idx = pd.date_range(per_date.index.min(), per_date.index.max(), freq="D")
    floor_path = (per_date.reindex(full_idx).ffill() / _TGP_DELTA_DAYS).cumsum()

    days_since_peak = pd.to_numeric(df["cycle_days_since_peak"], errors="coerce")
    peak_dates = dates - pd.to_timedelta(days_since_peak, unit="D")

    level_now = dates.map(floor_path)
    level_at_peak = peak_dates.map(floor_path)
    displacement = (level_now - level_at_peak).to_numpy()
    out["tgp_cycle_displacement_cents"] = displacement

    # Same displacement as a share of the previous cycle's peak-to-trough amplitude:
    # "how much of this cycle's depth is the floor accounting for?" Scale-free, so it
    # stays comparable across the 2016 and 2022 price regimes, and non-linear in the
    # locked columns it divides by.
    amplitude = (
        pd.to_numeric(df["cycle_last_max_cents"], errors="coerce")
        - pd.to_numeric(df["cycle_last_min_cents"], errors="coerce")
    )
    amplitude = amplitude.where(amplitude.abs() >= _MIN_AMPLITUDE_CENTS)
    out["tgp_cycle_displacement_frac_amp"] = displacement / amplitude.to_numpy()
    return out
