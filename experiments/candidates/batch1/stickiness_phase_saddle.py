"""batch1 candidate — the stickiness x cycle-phase saddle (fps-3jj.11).

The one item in experiments/ledger.yaml that is flagged as open and was never pursued:
"Left genuinely open: the stickiness x cycle_pct_through opposite-orientation saddle,
surfaced unexpectedly in step 1 and never pursued."
(2026-06-05_phase_lookup_nonparametric, not_tested.)
"""
from __future__ import annotations

import pandas as pd

NAME = "stickiness_phase_saddle"

HYPOTHESIS = (
    "cycle_pct_through is a NETWORK clock (one value per date), but a station's "
    "willingness to follow that clock is a per-station property. A chronically cheap, "
    "competitive station tracks the cycle closely: at phase 0.9 it really is near its "
    "own trough. A chronically dear or erratic station is only loosely coupled to it: "
    "the same phase reading carries much less information about where that station's "
    "own price is going next. So the SAME clock value should drive different buy/wait "
    "decisions at different stickiness levels. The lock hands the model both terms "
    "separately and nothing else; a gradient-boosted tree can only approximate their "
    "product through nested axis-aligned splits, which is the representation limit that "
    "made the LGA and network groups worth engineering explicitly."
)

PREDICTED_SIGNATURE = (
    "Decision-timing: for stations with near-zero stickiness the decisions should be "
    "essentially unchanged (the product is ~0 there, so the columns add nothing), and "
    "the flips should be concentrated on the tails of the stickiness distribution — "
    "buying EARLIER on high-|stickiness| stations at late phase, because the network "
    "clock is over-promising a trough those stations will not fully follow. The "
    "distinctive signature to look for is that the SIGNED and ABSOLUTE columns are both "
    "used and disagree in orientation somewhere in the SHAP surface; if only "
    "abs_sticky_x_phase is used, the real content is 'outlier stations are decoupled' "
    "rather than a saddle, which is a weaker and different claim. Expect it to spread "
    "across folds rather than concentrate, because station composition is a "
    "cross-sectional property with no particular fold affinity."
)

# P(pooled realised CPL delta < 0). Modest. On the plus side this is a genuinely
# unpursued lead that surfaced from a SHAP interaction surface rather than from
# speculation, and it costs the model nothing on the ~50% of rows near stickiness 0. On
# the minus side, the closest precedent in this repo is discouraging: the explicit
# interaction column A_x_shallow_elong made fold 7 WORSE, not flat, and
# feedback_partner_score_substitution warns that a partner-score plateau conflates
# substitution with interaction.
CONFIDENCE_EFFECT = 0.35

# P(the effect concentrates where TARGET says). LOW, and low on purpose: TARGET below
# makes NO zone claim, because a cross-sectional station property has no reason to
# prefer any fold. Stating a spread-out expectation honestly is worth more to the
# calibration record than inventing a zone to be graded on.
CONFIDENCE_ZONE = 0.15

# No zone claim. Empty axis levels AND empty folds -- only CONFIDENCE_EFFECT is graded.
TARGET = {"axis": "regime", "expect_concentration_in": [], "folds": []}

MECHANISM_FAMILY = "station-heterogeneity-interaction"

PRIOR_ART = (
    "The saddle itself comes from experiments/2026-06-04_cycle_pct_through_interaction "
    "/ 2026-06-05_phase_lookup_nonparametric, whose ledger not_tested records it as the "
    "one thing left genuinely open on that track. Adjacent to two rejected candidates "
    "from the SAME experiments, and the difference matters: both of those "
    "(station_price_cents - interp(trough, peak, cycle_pct_through), and the "
    "non-parametric E[norm_price | phase] lookup) engineered a PHASE-RESIDUAL, and both "
    "failed the same way — additive-neutral, ablation-negative, because the diagonal "
    "projection DISCARDS the absolute anchors the sibling features carry. The durable "
    "constraint recorded there is 'no phase-residual variant should be pursued without "
    "an architecture that PRESERVES the absolute anchors'. This candidate is not a "
    "residual: it adds two columns and removes nothing, so every anchor stays in place. "
    "Also adjacent to A_x_shallow_elong (#231, rejected and backwards) as another "
    "explicit interaction column — noted as a real warning, not dismissed; the "
    "difference is that #231's interaction was built on a corner the oracle diagnostic "
    "had already shown A reads MISLEADINGLY, whereas neither stickiness_score nor "
    "cycle_pct_through has been shown to misread anywhere. Also relevant: "
    "project_sticky_station_exclusion (aggregates must EXCLUDE sticky stations) is not "
    "violated here — nothing is aggregated, this is a per-row product."
)

COLUMNS = ["sticky_x_phase", "abs_sticky_x_phase"]

INPUTS = ["stickiness_score", "cycle_pct_through"]


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Signed and unsigned products of the station's chronic premium with cycle phase.

    Two columns because the saddle has two axes and the tree can use neither implicitly:
    the SIGNED product separates chronically-cheap from chronically-dear stations at the
    same phase (the saddle's orientation flip), while the ABSOLUTE product separates
    stations that are merely unusual in either direction — the 'Competitive' cohort is
    defined on |stickiness_score|, so magnitude is the axis that carries the coupling
    strength. Dropping either leaves a strictly weaker statement: sign alone cannot say
    'this station is decoupled', magnitude alone cannot say 'in which direction'.

    PIT-safe trivially: a row-wise product of two columns already on the row. Nothing is
    aggregated, shifted, rolled or grouped, so no value depends on any other row and
    truncation cannot change anything. Deliberately NOT centred on a sample statistic —
    subtracting a whole-frame median or mean would read every date's rows including
    future ones, which is exactly the leak the differential PIT test exists to catch.
    """
    out = df.copy()
    sticky = pd.to_numeric(df["stickiness_score"], errors="coerce")
    phase = pd.to_numeric(df["cycle_pct_through"], errors="coerce")
    out["sticky_x_phase"] = sticky * phase
    out["abs_sticky_x_phase"] = sticky.abs() * phase
    return out
