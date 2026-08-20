"""Candidate module template + worked example — the fps-3jj module format.

Copy this file into experiments/candidates/<batch>/<name>.py and edit the
NAME/HYPOTHESIS/... fields and add_columns/add_axis for your real candidate.
The rest of this file (imports, add_columns body) is a genuine, runnable,
PIT-safe example — not just TODO placeholders — so it can be validated as-is:

  PYTHONPATH=. uv run python experiments/pipeline/validate.py \\
    experiments/candidates/TEMPLATE.py

experiments/** is exempt from the PR rule (docs/CONVENTIONS.md), so a generator
session commits candidate files straight to main.

Every field below is read by the runner (experiments/pipeline/runner.py) and the
validation harness (experiments/pipeline/validate.py) — do not rename them.
"""
from __future__ import annotations

import pandas as pd

from experiments.lib.features.rolling import rolling_baseline

# A short, distinctive column-family name.
NAME = "example_network_avg_vs_rolling7"

# Mechanism in price-formation terms, not "the tree might like it".
HYPOTHESIS = (
    "The network-wide average price's deviation from its own recent (7d) "
    "rolling median captures short-term market-wide momentum that the "
    "production cycle-phase features (station-relative, not momentum) don't "
    "carry directly."
)

# What the numbers should look like if the mechanism is real.
PREDICTED_SIGNATURE = (
    "This is the module-format worked example, not a proposed feature — no "
    "real prediction. A genuine candidate would describe expected sign/"
    "magnitude and where it concentrates."
)

# Your own prior: P(pooled realised CPL delta < 0). Scored in the retrospective.
CONFIDENCE_EFFECT = 0.05

# P(the effect concentrates where TARGET says). Scored only if CONFIDENCE_EFFECT
# resolves true — a candidate that never moves the arbiter has no zone to be
# right about.
CONFIDENCE_ZONE = 0.10

# Where the effect should concentrate. "axis" is either the built-in "regime"
# (SHOCK_FOLDS shock-vs-normal) or the name add_axis(df) below produces;
# "folds" is a direct list of 1-indexed WFCV fold numbers. Either or both may
# be set — a fill counts as "in target" if it matches EITHER. Empty list /
# omitted axis means "not claiming a zone" (only CONFIDENCE_EFFECT is graded).
TARGET = {"axis": "day_of_week", "expect_concentration_in": ["Monday", "Tuesday"], "folds": []}

# Adjacent to something rejected? Say what differs. "None" only for a genuine
# first-of-its-kind candidate.
PRIOR_ART = "None — this is the fps-3jj module-format worked example, not a proposed feature."

# The column(s) add_columns produces.
COLUMNS = ["example_network_avg_minus_rolling7"]

# Every column add_columns (and add_axis, if present) reads. The runner and
# validation harness call add_columns(frame[INPUTS].copy()) as a provenance
# check — an undeclared read raises KeyError and the candidate is aborted.
INPUTS = ["price_date", "station_price_cents"]


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pure function of df[INPUTS]. No DB, no network, no future dates.

    Must be point-in-time: the differential PIT test recomputes this on a
    date-truncated frame and aborts the candidate if the result changes. This
    example reuses experiments/lib/features/rolling.rolling_baseline, which is
    PIT-strict by construction (closed='left' — today's value never enters
    today's aggregate).
    """
    out = df.copy()
    dates = pd.to_datetime(out["price_date"])
    network_avg = out.groupby(dates)["station_price_cents"].mean()
    baseline = rolling_baseline(network_avg, window_days=7)
    out[COLUMNS[0]] = dates.map(network_avg) - dates.map(baseline)
    return out


def add_axis(df: pd.DataFrame) -> pd.Series:
    """OPTIONAL — per-row group labels for TARGET's "axis". Delete this
    function if unused; the runner falls back to fold/regime cuts only.

    Same PIT-test requirement as add_columns: this must not use future
    information, since an axis leak doesn't corrupt the model (the model never
    sees it) but corrupts the CONCLUSION ("helps on days that turned out to be
    near the trough" is unactionable at 7am). Day-of-week is trivially
    PIT-safe: it's knowable in advance from the calendar alone.

    A ROW-LEVEL AXIS CANNOT CARRY A COST CLAIM (fps-grp). Pooled realised CPL is
    a path-coupled total -- a buy now changes what is possible later -- so
    cutting it on a per-row label allocates that total to a sub-period, which
    has no unique answer. Both the dossier's per_axis deltas and CONFIDENCE_ZONE
    graded on this axis ship an identification caveat and are colour, not
    findings. TARGET["folds"] and the built-in axis "regime" (SHOCK_FOLDS) are
    graded on identified quantities instead, because each (fold, station) is an
    independent simulation with its own tank. Prefer them for the headline zone
    claim; keep add_axis for descriptive breakdowns (feature values, NaN rates,
    log-loss), which are natively stamped at a moment and unaffected.
    """
    return pd.to_datetime(df["price_date"]).dt.day_name().rename("axis")
