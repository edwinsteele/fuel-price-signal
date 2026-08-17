"""Batch 0 candidate — tgp_delta_7d, the known-graduate pipeline check (fps-3jj, batch sizing).

tgp_delta_7d is already computed into every features.csv/parquet row by
fuel_signal/features.py (TGP_FEATURE_COLUMNS, chip 3 / PR #276) but deliberately
held OUT of FEATURE_COLUMNS pending the TGP production re-lock (#271 /
fps-1785999729707-1). This candidate's only job is to add the already-computed
column to the model contract for one pipeline run, so the pipeline can be
checked against a known answer before anything genuinely AI-sourced is trusted
to it: the realised arbiter (experiments/2026-06-20_leading_indicators/, 14
walk-forward folds) already found pooled delta -0.039 c/L, concentrated in the
expensive shock-regime folds.

Batch 1's pre-registered pass criterion (docs/routines/generator.md) is sign +
concentration, not magnitude — this run uses a later data vintage and
add_columns rather than the original experiment's extra_feature_provider
closure, so the number is expected to land NEAR -0.039, not on it.
"""
from __future__ import annotations

import pandas as pd

NAME = "tgp_delta_7d"

HYPOTHESIS = (
    "7-day AIP Sydney TGP (wholesale floor) momentum predicts near-term retail "
    "descent, concentrating in shock regimes where today's wholesale floor "
    "position alone misleads (the raw gap station_minus_tgp_cents helps calm "
    "but hurts shock; tgp_delta_7d rescues exactly the shock regression). "
    "Already graduated the realised arbiter, 2026-06-22, "
    "experiments/2026-06-20_leading_indicators/."
)

PREDICTED_SIGNATURE = (
    "Pooled realised CPL delta negative, sign concentrated in shock-regime "
    "folds (the highest-baseline-CPL / highest oracle-headroom folds), not "
    "magnitude-matched to the original -0.039 c/L pooled figure — later data "
    "vintage (snapshot 2026-08-10 vs the June 2026 experiment) and add_columns "
    "passthrough rather than the experiment's extra_feature_provider closure."
)

# Own prior: P(pooled realised CPL delta < 0). This is a known graduate, not a
# fresh proposal, so confidence is high but not 1.0 (later vintage + different
# compute path than the original arbiter run).
CONFIDENCE_EFFECT = 0.9

# P(the effect concentrates where TARGET says). Scored only if CONFIDENCE_EFFECT
# resolves true.
CONFIDENCE_ZONE = 0.75

TARGET = {"axis": "regime", "expect_concentration_in": ["shock"], "folds": []}

MECHANISM_FAMILY = "external upstream wholesale signal"

PRIOR_ART = (
    "Graduated the realised arbiter 2026-06-22 (14 WF folds, pooled -0.039 "
    "c/L; see project_tgp_leading_indicator memory / experiments/"
    "2026-06-20_leading_indicators/). The raw gap station_minus_tgp_cents "
    "(same upstream series, adjacent mechanism) was rejected: it helps calm "
    "(-0.012) but hurts shock (+0.011), because today's TGP position is "
    "accurate when the floor is stable and misleading when it's moving. "
    "tgp_delta_7d reads the floor's direction instead of its level, which is "
    "the complementary, not redundant, mechanism. Production compute already "
    "landed (#272 downloader, #275 DB ingest, #276 features.py compute); this "
    "candidate is the held-out re-lock check, not a new proposal."
)

COLUMNS = ["tgp_delta_7d"]
INPUTS = ["tgp_delta_7d"]


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Passthrough. tgp_delta_7d is already computed PIT-safely by
    fuel_signal/features.py (pit = tgp_series.asfreq('D').ffill().shift(1);
    delta = pit - pit.shift(7)) into the frozen frame. This candidate's job is
    only to add it to feature_columns for one run, not to recompute it.
    """
    return df.copy()
