"""Batch 0 candidate — tgp_delta_7d, the known-graduate pipeline check (fps-3jj, batch sizing).

tgp_delta_7d is already computed into every features.csv/parquet row by
fuel_signal/features.py (TGP_FEATURE_COLUMNS, chip 3 / PR #276) but deliberately
held OUT of FEATURE_COLUMNS pending the TGP production re-lock (#271 /
fps-1785999729707-1). This candidate's only job is to add the already-computed
column to the model contract for one pipeline run, so the pipeline can be
checked against a known answer.

REWRITTEN 2026-08-19 (bd fps-nor). The known answer is no longer June's
-0.039 c/L. Two things were wrong with the first attempt: batch0's R0 carried
the 10 REJECTED Phase 4b brand-trough columns (bd fps-sa1, fixed PR #309), and
the June figure was never resolvable anyway — the arbiter's quantum is ~0.05 c/L
per buy/wait flip, larger than the effect that graduated the feature.

The reference is now +0.0059 c/L, from
tgp_delta_7d/diagnostic_baseline_composition.py: same harness, same frozen
snapshot, same corrected 54-column baseline, same folds, seed 42. runner.py's
realised stage is single-seed at SEEDS[0]=42, so the pipeline should reproduce
it EXACTLY.

Read the pass criterion accordingly: this run calibrates the PIPELINE against a
reference number, and says nothing about whether tgp_delta_7d earns a place in
FEATURE_COLUMNS. #271 stays on hold either way.
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
    "Pooled realised delta_cpl_held = +0.0059 c/L, matching "
    "experiments/candidates/batch0/tgp_delta_7d/diagnostic_baseline_composition.py "
    "to the decimal. That reference was produced by the same harness "
    "(run_paired_realised_backtest) on this same frozen snapshot, same 54-column "
    "baseline, same fold geometry, at seed 42 — and runner.py's realised stage is "
    "single-seed at SEEDS[0]=42, so this is an exact-reproduction check, not a "
    "ballpark one. THIS RUN CALIBRATES THE PIPELINE, NOT THE FEATURE: the number "
    "landing anywhere else means the pipeline is wired wrong."
)

# Own prior: P(pooled realised CPL delta < 0). Deliberately ~coin-flip. The
# effect is inert (+0.0059 c/L) and the arbiter's quantum is ~0.05 c/L per
# buy/wait flip, so SIGN CARRIES NO INFORMATION HERE — see bd fps-nor and
# experiments/candidates/batch0/tgp_delta_7d/README.md § Diagnostic 2. The
# earlier 0.9 was premised on "known graduate, expect -0.039 concentrated in
# shock"; all three premises are now known false.
CONFIDENCE_EFFECT = 0.5

# No concentration claim. The original shock concentration was an artifact of
# the wrong 64-column baseline (fps-sa1); on the corrected baseline the largest
# fold contribution is fold 8, a NORMAL fold.
CONFIDENCE_ZONE = 0.5

TARGET = {"axis": "regime", "expect_concentration_in": [], "folds": []}

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
