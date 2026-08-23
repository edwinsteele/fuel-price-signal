"""batch1 candidate — directional participation in the network's move (fps-3jj.11).

Everything in the lock that reads the cross-section is SIGN-BLIND. network_px_std rises
when a few stations run ahead of the pack -- and it does that identically whether they
are running DOWN (the cuts spreading) or UP (the restoration igniting). Those are the
two opposite ends of the buy/wait decision.
"""
from __future__ import annotations

import pandas as pd

from experiments.lib.features.deltas import calendar_aware_delta
from experiments.lib.features.diagnostics import px_change_lag_diagnostic

NAME = "network_move_breadth"

HYPOTHESIS = (
    "The Sydney cycle ends with a coordinated restoration: a subset of stations jumps "
    "20-30 c/L within a day or two and the rest follow over the next few days. For a "
    "buyer the cost of being one day late to that turn dwarfs the gain from one more "
    "day of a 1-2 c/L descent, so the single most valuable thing the model could know "
    "is that peers have STARTED jumping while this station has not. It cannot currently "
    "know it. network_px_std measures the SPREAD of the cross-section and is invariant "
    "to direction: a handful of stations 20 c/L above the pack and a handful 20 c/L "
    "below produce the same reading. Counting the share of stations moving UP versus "
    "DOWN over a fixed window recovers the sign that the dispersion statistic throws "
    "away, and does it as a participation count that is robust to the size of any one "
    "station's jump."
)

PREDICTED_SIGNATURE = (
    "Decision-timing, in the most direct form in this batch: rows where "
    "network_rise_breadth_delta_2d turns sharply positive should flip WAIT -> BUY "
    "immediately, one to three days earlier than the lock manages, and the saved cents "
    "per flip should be LARGE (a restoration is worth 20-30 c/L, versus ~1-2 c/L for a "
    "day of descent). So the expected signature is FEW decision flips carrying an "
    "outsized share of the pooled delta -- if the pooled delta instead comes from many "
    "small flips spread evenly, the columns are working as generic momentum and the "
    "restoration-timing story is not what is paying. Expect concentration on the SHOCK "
    "folds, where restorations are most violent and the network clock is least "
    "reliable. Note the asymmetry: network_fall_breadth_3d should mostly cause the "
    "model to WAIT longer, so the two columns are expected to push in opposite "
    "directions rather than reinforce."
)

# P(pooled realised CPL delta < 0). The HIGHEST of the batch, and the one I would
# defend hardest: the quantity is absent from the lock in a way that is verifiable
# rather than argued (no locked column is a directional cross-sectional count), the
# mechanism is the largest single cost in the objective, and the objective is realised
# CPL rather than log-loss, which is where a rare-but-huge event actually scores. Held
# below 0.6 because the model may already infer the turn from level features fast
# enough that being told explicitly buys nothing -- the is_post_covid null result is the
# cautionary precedent for "the tree gains nothing from being told".
CONFIDENCE_EFFECT = 0.60

# P(it concentrates on shock folds), scored only if EFFECT resolves true. Moderate: the
# restoration happens every cycle, in every fold, so a shock concentration is plausible
# (violent turns, unreliable clock) but not the only credible outcome -- a uniform
# spread across all 14 folds would be entirely consistent with the mechanism too.
CONFIDENCE_ZONE = 0.35

# Fold-level "regime" -- the RESERVED name meaning SHOCK_FOLDS {1,4,9,13} vs the rest,
# not zones.assign_regime()'s row-level cycle phase.
TARGET = {"axis": "regime", "expect_concentration_in": ["shock"], "folds": []}

MECHANISM_FAMILY = "directional-network-breadth"

PRIOR_ART = (
    "Adjacent to family A (network_px_std, network_px_std_delta_3d — GRADUATED, in the "
    "lock) and, through it, to family C: all three are per-date cross-sectional "
    "summaries, and the memory note that A and C both decode 'the network is converging "
    "on a trough' applies to this candidate's super-family too. Stated plainly rather "
    "than relabelled: this batch files TWO cross-sectional-consensus candidates (this "
    "one and lga_trough_propagation) and the batch record should be read with that in "
    "mind. What differs mechanically from A: A is a scale statistic (a std) computed on "
    "price LEVELS within a cohort; these are signed participation COUNTS computed on "
    "price CHANGES. A cannot distinguish up from down and is dominated by the few "
    "largest deviations; a breadth count is direction-explicit and caps every station's "
    "contribution at one vote. A's own ledger not_tested flags its unresolved defect as "
    "'the missing axis is descent shape, not elongation' — a directional breadth reading "
    "is one candidate for that axis, though this module claims restoration timing "
    "specifically, not the elongation corner. Also adjacent to family B (discount-cohort "
    "gap — REJECTED, flat), whose not_tested asks 'whether B's signal lives in its "
    "VELOCITY rather than its level'; this is a velocity-of-participation statistic over "
    "the whole network rather than a level gap between two brand cohorts."
)

COLUMNS = [
    "network_rise_breadth_3d",
    "network_fall_breadth_3d",
    "network_rise_breadth_delta_2d",
]

INPUTS = ["station_code", "price_date", "station_price_cents"]

#: Window over which a station's own move is measured, in days.
_CHANGE_DAYS = 3

#: A station "participates" when its move over the window exceeds this, in cents. Sized
#: against the two processes being separated: an ordinary descent cuts ~1-2 c/L per day
#: (so a 3-day fall clears 2 c/L routinely) while a restoration jumps 20-30 c/L at once.
#: The threshold is therefore permissive on the fall side by design -- fall breadth is
#: meant to read "the cuts are still spreading", which is a common state -- and highly
#: specific on the rise side, where a 3-day rise of 2 c/L is not something a descending
#: station does by accident.
_MOVE_THRESHOLD_CENTS = 2.0

#: Lag for the rise-breadth velocity. Shorter than the 3-day change window on purpose:
#: the quantity of interest is the IGNITION of a restoration, and a 3-day velocity of a
#: 3-day breadth would smear the first day of it across the whole window.
_BREADTH_DELTA_DAYS = 2


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Share of the network moving up / down over a 3-day window, plus rise velocity.

    PIT-safe. px_change_lag_diagnostic is an exact-date backward self-merge on
    (station_code, price_date - 3), the per-date means aggregate only rows of that same
    date, and calendar_aware_delta shifts backwards. Truncating the frame removes whole
    later dates without changing any earlier date's row-set, so surviving values are
    bit-identical.

    Stations with no observation exactly 3 days earlier contribute NaN and are excluded
    from that date's denominator rather than counted as "did not move" -- a data gap is
    not evidence of a flat price. The breadths are therefore shares of the stations that
    could be MEASURED on the date, not of every station in the frame.
    """
    out = df.copy()
    dates = pd.to_datetime(df["price_date"])

    work = df[["station_code", "station_price_cents"]].copy()
    work["price_date"] = dates
    change = px_change_lag_diagnostic(work, _CHANGE_DAYS)

    measurable = change.notna()
    rose = (change >= _MOVE_THRESHOLD_CENTS) & measurable
    fell = (change <= -_MOVE_THRESHOLD_CENTS) & measurable

    counts = pd.DataFrame(
        {"n": measurable.to_numpy(), "up": rose.to_numpy(), "down": fell.to_numpy()},
        index=dates.values,
    ).groupby(level=0).sum().sort_index()

    denom = counts["n"].where(counts["n"] > 0)
    rise_breadth = counts["up"] / denom
    fall_breadth = counts["down"] / denom

    out["network_rise_breadth_3d"] = dates.map(rise_breadth).to_numpy()
    out["network_fall_breadth_3d"] = dates.map(fall_breadth).to_numpy()

    delta = calendar_aware_delta(rise_breadth, _BREADTH_DELTA_DAYS)
    out["network_rise_breadth_delta_2d"] = dates.map(delta).to_numpy()
    return out
