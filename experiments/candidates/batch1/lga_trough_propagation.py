"""batch1 candidate — how far the trough wave has propagated across LGAs (fps-3jj.11).

Answers the open question recorded against family C in experiments/ledger.yaml:
"Whether a leader-WEIGHTED consensus beats an unweighted LGA std -- the lead/lag roles
are known (Penrith leads falls; Blue Mountains and Randwick lead rises) but were never
used to weight this statistic."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.lib.features.deltas import calendar_aware_delta
from fuel_signal.features import LGA_FEATURE_COLUMNS

NAME = "lga_trough_propagation"

HYPOTHESIS = (
    "The Sydney trough is not a moment, it is a wave: individual LGAs bottom on "
    "different days and the wave runs from a known set of early LGAs outward. The 35 "
    "days_since_trough_entry_<lga> clocks are all in the lock, but the model reduces "
    "them through 35 axis-aligned splits, and the ONE cross-sectional summary it is "
    "handed (lga_phase_std) is a dispersion statistic — sign-blind and order-blind. "
    "Std cannot distinguish 'a third of the network has just bottomed and the rest is "
    "about to follow' from 'a third bottomed six weeks ago and the rest is mid-descent', "
    "yet those are opposite buy/wait situations. Breadth (how many have turned), its "
    "velocity (is the wave still spreading), and the leaders' head start (is there "
    "anyone left to follow) recover exactly what the std discards."
)

PREDICTED_SIGNATURE = (
    "Decision-timing: the model should keep waiting while lga_trough_breadth_delta_3d "
    "is positive and lga_leader_lead_days is large (the wave is still arriving, so "
    "cheaper prices are still spreading), and should buy once breadth velocity rolls "
    "over toward zero with leader lead already collapsed (everyone who was going to cut "
    "has cut, and the restoration is next). Expect the flips to be WAIT->BUY moves a "
    "few days earlier than the lock makes them, and to concentrate on non-shock folds: "
    "the wave is a property of the ordinary cycle, and in a shock window the wholesale "
    "move overrides the propagation order entirely. A signature that would falsify the "
    "mechanism: gains driven by lga_trough_breadth_7d's LEVEL alone with the velocity "
    "and leader-lead columns inert — that would mean this is lga_phase_std in different "
    "clothes rather than the order information it claims to add."
)

# P(pooled realised CPL delta < 0). Higher than a cold proposal because the underlying
# panel is already known to carry signal (family C graduated on it) and the ledger names
# this specific unweighted->weighted step as untested. Discounted because the columns
# are built ENTIRELY from locked columns, so the model already has the raw material and
# only the assembly is new -- and this project's history says the assembly usually is
# the whole win (35 LGA columns went in as a group; 4 network columns moved ~1 c/L).
CONFIDENCE_EFFECT = 0.45

# P(it concentrates on non-shock folds). Reasonably confident: the mechanism is
# explicitly a calm-cycle propagation story, and the shock folds are the ones where an
# exogenous wholesale move resets every LGA at once.
CONFIDENCE_ZONE = 0.45

# "regime" is the RESERVED FOLD-LEVEL axis (shock = SHOCK_FOLDS {1,4,9,13}, everything
# else normal) -- NOT zones.assign_regime()'s row-level cycle phase, which shares the
# level name "normal". The claim intended here is the fold-level one, and it is graded
# on an identified quantity because a fold is an independent simulation with its own tank.
TARGET = {"axis": "regime", "expect_concentration_in": ["normal"], "folds": []}

MECHANISM_FAMILY = "lead-lag-propagation"

PRIOR_ART = (
    "Directly adjacent to family C (lga_phase_std, lga_phase_std_delta_3d — GRADUATED, "
    "in the lock), whose ledger entry records exactly this as untested: 'Whether a "
    "leader-WEIGHTED consensus beats an unweighted LGA std -- the lead/lag roles are "
    "known ... but were never used to weight this statistic.' What differs mechanically: "
    "lga_phase_std is the sample std of the same 35 clocks, a scale statistic that is "
    "invariant to which LGAs hold which values; these columns are order/level statistics "
    "over the same panel and are invariant to nothing. Also adjacent to family A "
    "(network_px_std, graduated) and, through it, to the memory note that A and C both "
    "decode 'the network is converging on a trough' — this candidate is in that same "
    "super-family and the batch record should be read with that in mind. It is NOT a "
    "reparameterisation: a dispersion measure and a breadth count can move in opposite "
    "directions (std rises when a few LGAs run far ahead; breadth rises only when many "
    "have already turned). Caveat carried from project_lga_leadership: LGA leadership is "
    "TIME-VARYING, and the Phase 4 SHAP verdict retired Penrith (directionally right, "
    "predictively inert -- absorbed by stickiness_score + station_minus_last_min) and "
    "Sutherland Shire (flat) while confirming Blue Mountains and Randwick. The leader "
    "set below is therefore the two SHAP-confirmed rise leaders only, not the full "
    "user-observed prior."
)

COLUMNS = [
    "lga_trough_breadth_7d",
    "lga_trough_breadth_delta_3d",
    "lga_leader_lead_days",
]

INPUTS = ["price_date", *LGA_FEATURE_COLUMNS]

#: The two LGAs the Phase 4 SHAP work CONFIRMED as rise leaders. A trough entry is the
#: fall->rise turn, so "leads rises" is the role a trough-entry clock reads. Penrith
#: (leads falls, but predictively inert in Phase 4) and Sutherland Shire (flat) are
#: deliberately excluded -- see PRIOR_ART.
_LEADER_LGAS = (
    "days_since_trough_entry_blue_mountains",
    "days_since_trough_entry_randwick",
)

#: "Recently bottomed" window for the breadth count, in days.
_BREADTH_DAYS = 7

#: Propagation-velocity lag, matching the lock's own DELTA_LAG_DAYS convention.
_DELTA_DAYS = 3


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional order statistics over the 35 per-date LGA trough clocks.

    PIT-safe by construction. Every input column is already a PIT-strict quantity
    (compute_pit_strict_days_since_trough detects each LGA's troughs from prices
    restricted to <= the label date), the aggregations here are within-date only, and
    the one time-axis operation is calendar_aware_delta, which shifts backwards.
    Truncating the frame removes whole later dates and cannot change an earlier date's
    row-set, so every surviving value is bit-identical.

    Five of the 35 clocks (bayside, botany_bay, hunters_hill, lane_cove, waverley) are
    entirely NaN on the real frame; skipna handles them the same way
    features._lga_phase_std_per_date already does, by simply not counting them.
    """
    out = df.copy()
    dates = pd.to_datetime(df["price_date"])

    clocks = df[list(LGA_FEATURE_COLUMNS)].astype("float64")

    # Breadth: share of the LGAs that ARE reporting a clock which bottomed within the
    # window. Denominator is the non-null count per row, not 35, so the five dead
    # columns don't silently deflate every value by 14%.
    reporting = clocks.notna().sum(axis=1)
    recent = (clocks <= _BREADTH_DAYS).sum(axis=1)
    breadth = np.where(reporting > 0, recent / reporting.where(reporting > 0), np.nan)
    out["lga_trough_breadth_7d"] = breadth

    # Velocity of the wave. Collapse to one row per date first: the clocks are
    # date-level (verified: max 1 distinct value per date), so the per-date series is
    # well defined and the delta must be taken on it rather than per row.
    per_date = (
        pd.Series(breadth, index=dates.values)
        .groupby(level=0)
        .first()
        .sort_index()
    )
    delta = calendar_aware_delta(per_date, _DELTA_DAYS)
    out["lga_trough_breadth_delta_3d"] = dates.map(delta).to_numpy()

    # Leaders' head start over the typical LGA. Positive = the confirmed rise leaders
    # bottomed EARLIER than the median LGA, i.e. the wave is mid-propagation and there
    # are still laggards to follow. Decays toward zero as the network catches up.
    leader_mean = clocks[list(_LEADER_LGAS)].mean(axis=1, skipna=True)
    network_median = clocks.median(axis=1, skipna=True)
    out["lga_leader_lead_days"] = leader_mean - network_median
    return out
