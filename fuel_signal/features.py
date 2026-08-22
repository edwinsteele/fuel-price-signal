"""Feature pipeline for the ML price-movement model.

All features are computed using only data with price_date <= date_d
(point-in-time safe). The CycleDetector.detect() method slices its internal
series to date_d, so building one detector from the full series and calling
detect(date_d) per row is both correct and fast.

Usage
-----
Standalone (builds CycleDetector internally — fine for one-off calls)::

    conn = open_db(...)
    features = compute_features(conn, station_code=182, date_d="2024-06-15")

Batched (pre-build CycleDetector once for a large loop — see CLAUDE.md perf note)::

    from fuel_signal.db import average_price_series
    from fuel_signal.cycle import CycleDetector
    cd = CycleDetector(average_price_series(conn))
    for date_d in dates:
        features = compute_features(conn, station_code, date_d, cycle_detector=cd)
"""

from __future__ import annotations

import hashlib
import pathlib
import sqlite3
from collections.abc import Sequence
from datetime import date as _date
from datetime import timedelta as _timedelta

import click
import numpy as np
import pandas as pd

from fuel_signal import db as _db
from fuel_signal.brand_leadership import (
    brand_slug,
    compute_pit_strict_days_since_trough_brand,
    qualifying_brands,
)
from fuel_signal.cycle import CycleDetector, CycleState
from fuel_signal.dates import date_from_int as _date_from_int
from fuel_signal.dates import date_to_int as _date_to_int
from fuel_signal.labels import assemble_training_rows
from fuel_signal.lga_leadership import (
    LGA_FEATURE_COUNCILS,
    compute_pit_strict_days_since_trough,
    lga_feature_columns,
    lga_slug,
)

# Minimum label rows a station must have to be included in the training dataset.
# Roughly one year of daily observations. Stations below this threshold are
# too new to have survived a full price cycle and produce uninformative label patterns.
DEFAULT_FEATURES_CSV: pathlib.Path = pathlib.Path("data/features.csv")

MIN_TRAINING_ROWS_PER_STATION: int = 365

# Stations excluded from training due to confirmed data-gap distortion.
# Issue #29: both stations went offline during high-price years, so their
# rolling P33 was computed against a cheap-only price history. This causes
# both label conditions to fire almost constantly (positive rate 0.72–0.84),
# producing misleading training signal that the min-rows filter alone won't catch.
#   20528 — Speedway William Street, Granville: median 116.9c, positive rate 0.84
#   20133 — Metro Condell Park West: median 143.7c, positive rate 0.72
EXCLUDED_STATION_CODES: frozenset[int] = frozenset({20133, 20528})

# Canonical ordered list of feature column names for the current trained model.
FEATURE_COLUMNS: list[str] = [
    "cycle_pct_through",
    "cycle_days_since_peak",
    "cycle_mean_length",
    "cycle_last_min_cents",
    "cycle_last_max_cents",
    "cycle_peak_count",
    "station_price_cents",
    "station_minus_last_min_cents",
    "station_minus_last_max_cents",
    "station_minus_sydney_avg_cents",
    "lga_mean_cents",
    "station_minus_lga_mean_cents",
    "brand_mean_cents",
    "station_minus_brand_mean_cents",
    "stickiness_score",
]

# Phase 4 LGA trough features — separate from FEATURE_COLUMNS so the existing
# trained model contract is not broken until a Phase 4 retrain is complete.
# Compose with FEATURE_COLUMNS when training / evaluating the Phase 4 model.
LGA_FEATURE_COLUMNS: list[str] = lga_feature_columns()

# RAC_full network-aggregate features (issue #216, graduated from #212).
# Per-date cross-station aggregates over the canonical Competitive cohort
# (sc.class = 'Competitive') and per-date LGA-phase dispersion.
# Compose alongside LGA_FEATURE_COLUMNS for the 54-feat baseline.
DELTA_LAG_DAYS: int = 3
NETWORK_FEATURE_COLUMNS: list[str] = [
    "network_px_std",
    "network_px_std_delta_3d",
    "lga_phase_std",
    "lga_phase_std_delta_3d",
]

# TGP momentum feature (#271), graduated from experiment 2026-06-20_leading_indicators.
# Separate list (like NETWORK_FEATURE_COLUMNS) so the column lands in the features
# CSV now while the trained model contract (FEATURE_COLUMNS) only changes at the
# chip-4 re-lock retrain.
TGP_DELTA_DAYS: int = 7
TGP_FEATURE_COLUMNS: list[str] = ["tgp_delta_7d"]

_TROUGH_PREFIX = "days_since_trough_entry_"

# ---------------------------------------------------------------------------
# THE feature-scope contract
# ---------------------------------------------------------------------------
# One symbol for "the locked feature set". Everything that needs the production
# baseline — batch_freeze, the experiment TEMPLATE, experiments/lib.constants —
# imports THIS, rather than retyping the group composition (fps-zci).
#
# Two properties are load-bearing, and both have already been violated once:
#
#   SET   — the frame is a SUPERSET of the contract, so the set must be DECLARED,
#           never discovered from a features-CSV header (fps-sa1 gave batch0 a
#           64-column R0 by discovering the rejected brand-trough group).
#   ORDER — LightGBM breaks equal-gain split ties by feature index, so the same
#           columns in a different order fit a DIFFERENT model (fps-zci: sorting
#           moved 732/47,823 val rows' probabilities and 0.038 c/L of batch0's
#           pooled realised delta). Production order is the group concatenation
#           below; it differs from its own sorted permutation in 52 of 54
#           positions. NEW GROUPS APPEND — never insert, never sort.
#
# Ground truth is data/models/lgbm_calibrated.joblib's "feature_columns", which
# this list must equal element-for-element. tests/test_feature_contract.py
# asserts exactly that whenever the (gitignored) artifact is present.
LOCKED_FEATURE_COLUMNS: list[str] = (
    FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
)

# Why a column can be computed into features.csv yet sit OUTSIDE the lock.
# Machine-readable so that "not in scope" stops being indistinguishable from
# "in scope" to anything that inspects the frame (fps-zci item 3).
NON_MODEL_REASON_REJECTED = "evaluated-and-rejected"
NON_MODEL_REASON_HELD_OUT = "held-out-pending-graduation"
#: Measured, but below the arbiter's resolution — neither graduated nor dead ground.
#: Mirrors experiments/ledger.yaml's three-way vocabulary, which draws the same line:
#: a REJECTED claim is ground the feature generator should not re-propose; an
#: INCONCLUSIVE one was measured below the instrument's resolution and may still be
#: open. Excluded from the lock exactly as firmly as the other two — the distinction
#: is about what a future session should conclude, not about what the model sees.
NON_MODEL_REASON_INCONCLUSIVE = "evaluated-inconclusive"

#: Statically-named computed-but-excluded columns -> (reason_code, why).
NON_MODEL_COLUMNS: dict[str, tuple[str, str]] = {
    "tgp_delta_7d": (
        NON_MODEL_REASON_INCONCLUSIVE,
        "TGP momentum (#271). NOT graduated — the June 2026 realised-arbiter "
        "graduation (-0.039 c/L) was retracted 2026-08-19: it does not reproduce "
        "(+0.0059 c/L on identical columns, folds and seed) and was never "
        "resolvable, being smaller than the ~0.05 c/L value of one buy/wait "
        "decision flip. Against the R0-vs-R0 noise floor it sits at the 40th "
        "percentile of pure fit noise. No TGP feature is in the lock. The column "
        "stays computed because the ground may still be open on an axis nobody "
        "has tested (see experiments/ledger.yaml), NOT because a retrain is "
        "queued.",
    ),
}

#: Brand trough columns are DB-derived (qualifying brands depend on station counts),
#: so they cannot be enumerated statically — they are matched by rule instead. Note
#: the LGA trough columns share this prefix and ARE in the lock, hence the
#: LGA_FEATURE_COLUMNS exclusion in non_model_columns().
_BRAND_TROUGH_REASON = (
    NON_MODEL_REASON_REJECTED,
    "Phase 4b brand troughs, walked away 2026-06-02: lost 9/14 folds in paired WFCV "
    "with a non-shock regression at fold 11 (docs/STATUS.md § Phase 4b; AGENTS.md "
    "§ Canonical feature set). Still computed so a future ablation can graduate them.",
)


#: The label frame's own columns (``labels.assemble_training_rows``), which ride along
#: in features.csv because the frame is built by joining features onto labels. Split by
#: whether the value is knowable at decision time.
#:
#: TARGET_COLUMNS are computed from prices AFTER price_date. They are what the model
#: predicts, never an input to it — and the candidate pipeline cannot discover that on
#: its own. Its differential PIT test (experiments/lib/pit_test.py) works by deleting
#: future rows and re-running a candidate's add_columns, which catches a function that
#: reaches FORWARD ACROSS ROWS but not a forward-looking value already stamped ON the
#: row: truncating the frame leaves an existing future_min_cents cell untouched, the
#: recomputed value matches, and the leak test passes. Naming them here is what lets the
#: candidate pipeline close that hole structurally: experiments/pipeline/validate.py and
#: runner.py both DROP these columns from the frame a candidate is handed, and refuse
#: outright any candidate that names one in INPUTS or produces one in COLUMNS.
TARGET_COLUMNS: dict[str, str] = {
    "future_min_cents": (
        "Minimum price over the forward horizon (labels.assemble_training_rows). Pure "
        "oracle — it is the quantity the model exists to predict."
    ),
    "label": (
        "The training target itself: future_min_cents thresholded against today's price "
        "and a trailing percentile (labels.assemble_training_rows)."
    ),
}

#: Label-frame columns that ARE knowable at decision time: the two join keys, plus
#: today_price_cents (byte-identical to the locked station_price_cents on every row of
#: the real frame). Not model inputs and not hazards — but they have to classify as
#: SOMETHING for unclassified_columns() below to mean anything.
KEY_COLUMNS: frozenset[str] = frozenset({"station_code", "price_date", "today_price_cents"})


def non_model_columns(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Columns present in `df` that are deliberately outside the lock, with why.

    The features frame holds columns for three reasons and only the first puts a
    column in the model contract: in the lock; evaluated and REJECTED; held out
    pending graduation. This names the latter two so that a leak into the baseline
    is a machine-checkable condition rather than something a human has to notice
    two months later (fps-sa1, fps-nor).

    Deliberately does NOT consult LOCKED_FEATURE_COLUMNS. A detector that skipped
    whatever the lock already claims could not detect a column wrongly IN the lock —
    which is the entire fps-sa1 failure — so membership is decided by what a column
    IS: named in NON_MODEL_COLUMNS, or trough-prefixed without being one of the LGA
    troughs (which share the prefix and are genuinely locked).

    A consequence worth knowing: graduating a held-out column means deleting its
    NON_MODEL_COLUMNS entry in the same change that adds it to the lock. Until then
    resolve_baseline_columns() raises. That is the point — graduation is a
    declaration, not something the code should infer.
    """
    lga = set(LGA_FEATURE_COLUMNS)
    out: dict[str, tuple[str, str]] = {}
    for col in df.columns:
        if col in NON_MODEL_COLUMNS:
            out[col] = NON_MODEL_COLUMNS[col]
        elif col.startswith(_TROUGH_PREFIX) and col not in lga:
            out[col] = _BRAND_TROUGH_REASON
    return out


def unclassified_columns(df: pd.DataFrame) -> list[str]:
    """Columns in `df` that no declaration accounts for, in frame order.

    Every column in the features frame should be exactly one of: in the lock; computed
    but excluded (`non_model_columns`); the target (`TARGET_COLUMNS`); or a decision-time
    key (`KEY_COLUMNS`). Anything else is a column somebody added without saying what it
    is — and "unsaid" is the state in which a forward-looking column becomes readable by
    a candidate, because the pipeline's other guards are all keyed on declarations.

    This is the forcing function for `TARGET_COLUMNS` being a BLOCKLIST rather than an
    allowlist. A blocklist is the right shape here (the hazard is the target, a closed
    set, not an open category) but only while adding a new column to the frame forces
    someone to decide which bucket it belongs in. `freeze_batch` calls this on the real
    frame so that a batch cannot be frozen around an unclassified column.

    Deliberately DOES consult LOCKED_FEATURE_COLUMNS, unlike `non_model_columns`. The
    two guard different invariants and the difference is not an oversight: that one
    answers "is this column wrongly IN the lock?" and so must not take the lock's word
    for anything, while this one answers "has this column been categorised at all?", for
    which being in the lock is a perfectly good answer.
    """
    known = (
        set(LOCKED_FEATURE_COLUMNS)
        | set(non_model_columns(df))
        | set(TARGET_COLUMNS)
        | KEY_COLUMNS
    )
    return [col for col in df.columns if col not in known]


def baseline_fingerprint(columns: Sequence[str]) -> str:
    """Identity of one baseline column list, as '<n>:<12 hex chars of sha256>'.

    Hashes the ORDERED list — order is part of the contract, so two runs whose
    baselines differ only by a permutation must NOT fingerprint alike. Stamped into
    experiment meta.json, batch freeze.json and run results.json so cross-run
    comparisons can be checked mechanically; both contract defects found so far were
    invisible in the artifacts that recorded the runs (fps-zci item 5).

    The leading count is redundant with the digest but is what a human reads first:
    the fps-sa1 defect was "64" where "54" belonged.
    """
    digest = hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()
    return f"{len(columns)}:{digest[:12]}"


LOCKED_FEATURE_FINGERPRINT: str = baseline_fingerprint(LOCKED_FEATURE_COLUMNS)


# Brand trough feature columns are DB-derived (qualifying brands depend on
# station counts) so they cannot be a module-level constant.  Call
# brand_feature_columns(conn) within assemble_feature_rows to get the list,
# or discover_brand_feature_columns() against a features-CSV DataFrame at
# train/score time.


def discover_brand_feature_columns(df: pd.DataFrame) -> list[str]:
    """Brand trough columns present in df (excludes LGA trough columns).

    The features CSV header is the source of truth for which brands qualified
    at generation time. Returns alphabetical order so the column list is
    deterministic across runs against the same CSV.
    """
    lga = set(LGA_FEATURE_COLUMNS)
    return sorted(
        c for c in df.columns
        if c.startswith(_TROUGH_PREFIX) and c not in lga
    )



def _station_price_on_date(
    conn: sqlite3.Connection,
    station_code: int,
    date_d: str,
    fuel_type_id: int,
) -> float | None:
    row = conn.execute(
        "SELECT price_decicents FROM daily_prices"
        " WHERE station_code = ? AND fuel_type_id = ? AND price_date = ?",
        (station_code, fuel_type_id, _date_to_int(date_d)),
    ).fetchone()
    return row[0] / 10 if row else None


def _sydney_avg_on_date(
    conn: sqlite3.Connection,
    date_d: str,
    fuel_type_id: int,
) -> float | None:
    # Averages over all stations in daily_prices — intentionally unfiltered because
    # the DB contains only Sydney metro stations by design (filtered at load time).
    row = conn.execute(
        "SELECT AVG(price_decicents) FROM daily_prices"
        " WHERE fuel_type_id = ? AND price_date = ?",
        (fuel_type_id, _date_to_int(date_d)),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return row[0] / 10


def _lga_mean_on_date(
    conn: sqlite3.Connection,
    date_d: str,
    lga: str,
    fuel_type_id: int,
) -> float | None:
    """Return mean price (cents) for non-Sticky stations in lga on date_d, or None.

    Returns None when fewer than 3 non-Sticky stations contributed (NULL floor),
    or when no station_class rows exist for the LGA on that date (zero-Competitive gap).
    """
    row = conn.execute(
        "SELECT AVG(dp.price_decicents)"
        " FROM daily_prices dp"
        " JOIN stations s ON dp.station_code = s.station_code"
        " JOIN station_class sc ON dp.station_code = sc.station_code"
        "   AND dp.price_date = sc.snapshot_date"
        " WHERE s.council = ? AND dp.fuel_type_id = ? AND dp.price_date = ?"
        "   AND sc.class != 'Sticky'"
        " HAVING COUNT(*) >= 3",
        (lga, fuel_type_id, _date_to_int(date_d)),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return row[0] / 10


def _stickiness_score_on_date(
    conn: sqlite3.Connection,
    station_code: int,
    date_d: str,
) -> float | None:
    """Return stickiness_score (cents) = station_class.median_premium_decicents / 10.

    Returns None when no station_class row exists for (station_code, date_d).
    PIT-safe: queries by exact snapshot_date, so future rows are never read.
    """
    row = conn.execute(
        "SELECT median_premium_decicents FROM station_class"
        " WHERE station_code = ? AND snapshot_date = ?",
        (station_code, _date_to_int(date_d)),
    ).fetchone()
    return row[0] / 10 if row else None


def _brand_mean_on_date(
    conn: sqlite3.Connection,
    date_d: str,
    brand: str,
    fuel_type_id: int,
) -> float | None:
    """Return Sydney-wide mean price (cents) for non-Sticky stations of brand on date_d.

    Sydney-wide (not per-LGA-Brand) to avoid thin cells. Returns None when fewer
    than 3 non-Sticky stations contributed (NULL floor), or no station_class rows exist.
    """
    row = conn.execute(
        "SELECT AVG(dp.price_decicents)"
        " FROM daily_prices dp"
        " JOIN stations s ON dp.station_code = s.station_code"
        " JOIN station_class sc ON dp.station_code = sc.station_code"
        "   AND dp.price_date = sc.snapshot_date"
        " WHERE s.brand = ? AND dp.fuel_type_id = ? AND dp.price_date = ?"
        "   AND sc.class != 'Sticky'"
        " HAVING COUNT(*) >= 3",
        (brand, fuel_type_id, _date_to_int(date_d)),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return row[0] / 10


def _network_px_std_per_date(
    conn: sqlite3.Connection,
    fuel_type_id: int,
) -> dict[str, float]:
    """Per-date sample std of E10 prices (cents) over the canonical Competitive cohort.

    Cohort at date D: stations whose station_class.class = 'Competitive' on
    snapshot_date = D. PIT-safe — station_class rows are looked up by
    (station_code, snapshot_date=price_date), so the cohort at date D depends
    only on data available on or before D.

    Returns {date_iso: std_cents}. Dates with fewer than 2 contributing
    stations are absent (sample std undefined). One pass over daily_prices
    joined to station_class; aggregation is in Python because SQLite lacks a
    native STDDEV.
    """
    cur = conn.execute(
        "SELECT dp.price_date, dp.price_decicents"
        " FROM daily_prices dp"
        " JOIN station_class sc ON dp.station_code = sc.station_code"
        "   AND dp.price_date = sc.snapshot_date"
        " WHERE dp.fuel_type_id = ?"
        "   AND sc.class = 'Competitive'",
        (fuel_type_id,),
    )
    by_date: dict[int, list[float]] = {}
    for date_int, decicents in cur:
        by_date.setdefault(date_int, []).append(decicents / 10.0)

    result: dict[str, float] = {}
    for date_int, values in by_date.items():
        if len(values) >= 2:
            arr = np.asarray(values, dtype=float)
            result[_date_from_int(date_int)] = float(arr.std(ddof=1))
    return result


def _lga_phase_std_per_date(
    lga_days_since_by_key: dict[tuple[str, str], int | None],
    date_strs: list[str],
) -> dict[str, float]:
    """Per-date sample std of days_since_trough_entry across LGA_FEATURE_COUNCILS.

    Reads from the (date, lga) → days_since lookup already produced by
    compute_pit_strict_days_since_trough. Dates with fewer than 2 non-null
    LGA values are absent.
    """
    result: dict[str, float] = {}
    for d_str in date_strs:
        values = [
            lga_days_since_by_key.get((d_str, lga))
            for lga in LGA_FEATURE_COUNCILS
        ]
        non_null = [float(v) for v in values if v is not None]
        if len(non_null) >= 2:
            result[d_str] = float(np.asarray(non_null).std(ddof=1))
    return result


def _calendar_delta(
    level_by_date: dict[str, float],
    date_strs: list[str],
    lag_days: int = DELTA_LAG_DAYS,
) -> dict[str, float]:
    """Per-date level(d) − level(d − lag_days), calendar-aware.

    Absent for any date whose prior date is missing from level_by_date
    (e.g. start of the data window, or a gap day with <2 contributors).
    """
    result: dict[str, float] = {}
    for d_str in date_strs:
        prior = (_date.fromisoformat(d_str) - _timedelta(days=lag_days)).isoformat()
        level_d = level_by_date.get(d_str)
        level_prior = level_by_date.get(prior)
        if level_d is not None and level_prior is not None:
            result[d_str] = level_d - level_prior
    return result


def _tgp_delta_by_date(
    conn: sqlite3.Connection,
    n: int = TGP_DELTA_DAYS,
) -> dict[str, float]:
    """Per-date PIT Sydney ULP TGP N-day momentum, keyed by ISO date string.

    Matches the graduating experiment (2026-06-20_leading_indicators) exactly:
    the raw weekday series is resampled to a daily grid, weekend-/holiday-ffilled,
    then lagged one day (``pit = s.asfreq("D").ffill().shift(1)``) so inference on
    day D reads TGP only up to D−1. The feature is ``pit(D) − pit(D−n)``.

    Returned as {date_iso: value} for ``.get()`` onto label rows; dates with no
    valid prior (start of history) are absent. MUST stay bit-identical to the
    live provider added in chip 5 — see the realised-arbiter two-sites rule.
    """
    series = _db.tgp_series(conn)
    if not series:
        return {}
    s = pd.Series(
        [c for _, c in series],
        index=pd.to_datetime([d for d, _ in series]),
    ).sort_index()
    pit = s.asfreq("D").ffill().shift(1)
    delta = pit - pit.shift(n)
    return {
        ts.strftime("%Y-%m-%d"): float(v)
        for ts, v in delta.items()
        if pd.notna(v)
    }


def _build_feature_dict(
    state: CycleState,
    station_price: float,
    sydney_avg: float,
    lga_mean: float | None,
    brand_mean: float | None,
    stickiness_score: float | None,
) -> dict[str, float | None]:
    return {
        "cycle_pct_through": state.pct_through_cycle,
        "cycle_days_since_peak": float(state.days_since_last_peak),
        "cycle_mean_length": state.mean_cycle_length,
        "cycle_last_min_cents": state.last_cycle_min,
        "cycle_last_max_cents": state.last_cycle_max,
        "cycle_peak_count": float(state.peak_count),
        "station_price_cents": station_price,
        "station_minus_last_min_cents": station_price - state.last_cycle_min,
        "station_minus_last_max_cents": station_price - state.last_cycle_max,
        "station_minus_sydney_avg_cents": station_price - sydney_avg,
        "lga_mean_cents": lga_mean,
        "station_minus_lga_mean_cents": station_price - lga_mean if lga_mean is not None else None,
        "brand_mean_cents": brand_mean,
        "station_minus_brand_mean_cents": station_price - brand_mean if brand_mean is not None else None,
        "stickiness_score": stickiness_score,
    }


def compute_features(
    conn: sqlite3.Connection,
    station_code: int,
    date_d: str,
    cycle_detector: CycleDetector | None = None,
) -> dict[str, float | None] | None:
    """Return a feature dict for (station, date), or None if insufficient data.

    If cycle_detector is None, build one from average_price_series(conn).
    detect(date_d) slices the series to date_d internally — PIT-safe regardless
    of how much data the detector was built with.

    For batched callers: pre-build one CycleDetector and pass it in.
    Building per-row costs 3650x on a full backtest (CLAUDE.md perf note).
    For very large batches use assemble_feature_rows, which caches all three
    inputs across stations sharing a date.

    Returns None when:
    - Station has no price on date_d in daily_prices
    - CycleDetector.detect(date_d) returns None (fewer than 2 peaks)
    - Sydney average is absent on date_d (data gap)

    lga_mean_cents and brand_mean_cents may be None (NaN in DataFrame) when fewer
    than 3 non-Sticky stations are classified for that LGA/brand/date — this does
    not cause the row to be dropped.
    """
    fid = _db.fuel_type_id(conn, "E10")

    station_price = _station_price_on_date(conn, station_code, date_d, fid)
    if station_price is None:
        return None

    if cycle_detector is None:
        cycle_detector = CycleDetector(_db.average_price_series(conn))

    state = cycle_detector.detect(date_d)
    if state is None:
        return None

    sydney_avg = _sydney_avg_on_date(conn, date_d, fid)
    if sydney_avg is None:
        return None

    row = conn.execute(
        "SELECT council, brand FROM stations WHERE station_code = ?",
        (station_code,),
    ).fetchone()
    lga = row[0] if row else None
    brand = row[1] if row else None

    lga_mean = _lga_mean_on_date(conn, date_d, lga, fid) if lga else None
    brand_mean = _brand_mean_on_date(conn, date_d, brand, fid) if brand else None
    stickiness_score = _stickiness_score_on_date(conn, station_code, date_d)

    return _build_feature_dict(state, station_price, sydney_avg, lga_mean, brand_mean, stickiness_score)


def assemble_feature_rows(
    conn: sqlite3.Connection,
    horizon_days: int = 7,
    threshold_cents: float = 3.0,
    lookback_days: int = 90,
    percentile_pct: float = 33.0,
    station_codes: list[int] | None = None,
    min_rows_per_station: int = MIN_TRAINING_ROWS_PER_STATION,
) -> pd.DataFrame:
    """Build labels (via labels.assemble_training_rows) and join feature columns.

    Returns the labels DataFrame plus one column per feature in FEATURE_COLUMNS.
    Rows where compute_features returns None are dropped.
    CycleDetector is built once from the full average series — detect() slices
    per row, so PIT-safety is preserved.

    Stations in EXCLUDED_STATION_CODES are always removed (data-gap distortion).
    Stations with fewer than min_rows_per_station label rows are also removed
    (too-new stations haven't survived a full price cycle).
    """
    if min_rows_per_station < 0:
        raise ValueError("min_rows_per_station must be >= 0")

    label_df = assemble_training_rows(
        conn,
        horizon_days=horizon_days,
        threshold_cents=threshold_cents,
        lookback_days=lookback_days,
        percentile_pct=percentile_pct,
        station_codes=station_codes,
    )
    qualifying_brand_list = qualifying_brands(conn)
    brand_cols = [f"days_since_trough_entry_{brand_slug(b)}" for b in qualifying_brand_list]
    all_cols = (
        list(label_df.columns)
        + FEATURE_COLUMNS
        + LGA_FEATURE_COLUMNS
        + NETWORK_FEATURE_COLUMNS
        + TGP_FEATURE_COLUMNS
        + brand_cols
    )
    if label_df.empty:
        return pd.DataFrame(columns=all_cols)

    if EXCLUDED_STATION_CODES:
        label_df = label_df[~label_df["station_code"].isin(EXCLUDED_STATION_CODES)]

    if min_rows_per_station > 0:
        counts = label_df.groupby("station_code")["label"].count()
        eligible = counts[counts >= min_rows_per_station].index
        label_df = label_df[label_df["station_code"].isin(eligible)]

    if label_df.empty:
        return pd.DataFrame(columns=all_cols)

    fid = _db.fuel_type_id(conn, "E10")

    # Cache 1: Sydney avg by date. average_price_series IS the GROUP BY query
    # the per-row path runs, so reusing its result is bit-for-bit identical.
    sydney_series = _db.average_price_series(conn)
    sydney_avg_by_date: dict[str, float] = dict(sydney_series)

    cd = CycleDetector(sydney_series)

    # Cache 2: cycle state by date. detect() is pure in (cd._series, date),
    # and cd._series is set in __init__ and never mutated, so a single call per
    # unique date is correct.
    cycle_state_by_date: dict[str, CycleState | None] = {
        d: cd.detect(d) for d in label_df["price_date"].unique()
    }

    # Cache 3: station price by (station_code, date_iso). One bulk SELECT
    # replaces ~2M point-lookups. price_date is INTEGER YYYYMMDD in the DB but
    # ISO string in label_df; convert once at load time so the lookup key
    # matches.
    station_price_by_key: dict[tuple[int, str], float] = {
        (sc, _date_from_int(date_int)): decicents / 10
        for sc, date_int, decicents in conn.execute(
            "SELECT station_code, price_date, price_decicents FROM daily_prices"
            " WHERE fuel_type_id = ?",
            (fid,),
        )
    }

    # Derive scoping sets once so caches 4–6 only cover the label slice.
    label_station_codes = label_df["station_code"].unique().tolist()
    label_date_ints = [_date_to_int(d) for d in label_df["price_date"].unique()]
    _sc_ph = ", ".join(["?"] * len(label_station_codes))
    _dt_ph = ", ".join(["?"] * len(label_date_ints))

    # Cache 4: station LGA and brand by station_code (scoped to label stations).
    station_info_by_code: dict[int, tuple[str | None, str | None]] = {
        sc: (council, brand)
        for sc, council, brand in conn.execute(
            f"SELECT station_code, council, brand FROM stations"
            f" WHERE station_code IN ({_sc_ph})",
            label_station_codes,
        )
    }

    # Cache 5: LGA mean (non-Sticky, ≥3 stations) by (date_iso, lga).
    # Scoped to dates in label_df so the JOIN only touches the relevant slice.
    lga_mean_by_key: dict[tuple[str, str], float] = {
        (_date_from_int(date_int), lga): avg_decicents / 10
        for date_int, lga, avg_decicents in conn.execute(
            "SELECT dp.price_date, s.council, AVG(dp.price_decicents)"
            " FROM daily_prices dp"
            " JOIN stations s ON dp.station_code = s.station_code"
            " JOIN station_class sc ON dp.station_code = sc.station_code"
            "   AND dp.price_date = sc.snapshot_date"
            f" WHERE dp.fuel_type_id = ? AND sc.class != 'Sticky'"
            f"   AND s.council IS NOT NULL"
            f"   AND dp.price_date IN ({_dt_ph})"
            " GROUP BY dp.price_date, s.council"
            " HAVING COUNT(*) >= 3",
            [fid, *label_date_ints],
        )
    }

    # Cache 6: brand mean (non-Sticky, ≥3 stations) by (date_iso, brand).
    # Sydney-wide — not per-LGA-Brand, to avoid thin cells.
    # Scoped to dates in label_df.
    brand_mean_by_key: dict[tuple[str, str], float] = {
        (_date_from_int(date_int), brand): avg_decicents / 10
        for date_int, brand, avg_decicents in conn.execute(
            "SELECT dp.price_date, s.brand, AVG(dp.price_decicents)"
            " FROM daily_prices dp"
            " JOIN stations s ON dp.station_code = s.station_code"
            " JOIN station_class sc ON dp.station_code = sc.station_code"
            "   AND dp.price_date = sc.snapshot_date"
            f" WHERE dp.fuel_type_id = ? AND sc.class != 'Sticky'"
            f"   AND s.brand IS NOT NULL"
            f"   AND dp.price_date IN ({_dt_ph})"
            " GROUP BY dp.price_date, s.brand"
            " HAVING COUNT(*) >= 3",
            [fid, *label_date_ints],
        )
    }

    # Cache 7: stickiness_score (cents) by (station_code, date_iso).
    # Reads median_premium_decicents from station_class for each (station, date)
    # pair in the label set. Absent pairs → .get() returns None (NaN in DataFrame).
    stickiness_by_key: dict[tuple[int, str], float] = {
        (sc, _date_from_int(date_int)): decicents / 10
        for sc, date_int, decicents in conn.execute(
            "SELECT station_code, snapshot_date, median_premium_decicents"
            " FROM station_class"
            f" WHERE station_code IN ({_sc_ph})"
            f"   AND snapshot_date IN ({_dt_ph})",
            [*label_station_codes, *label_date_ints],
        )
    }

    # Cache 8: per-LGA days_since_trough_entry, PIT-strict.
    # For each unique label date d, detect_trough_events runs on prices[..d]
    # only — so the recorded trough date never depends on prices after d.
    # See compute_pit_strict_days_since_trough docstring for why the naive
    # full-history detect (build_lga_trough_lookups) leaks future data.
    label_date_strs: list[str] = list(label_df["price_date"].unique())
    # Extend with (d - DELTA_LAG_DAYS) dates so lga_phase_std_delta_3d has a
    # valid prior at the start of the label window. Extra dates only populate
    # the trough lookup; the row-construction loop never reads them.
    lga_lookup_dates = sorted(
        set(label_date_strs)
        | {
            (_date.fromisoformat(d) - _timedelta(days=DELTA_LAG_DAYS)).isoformat()
            for d in label_date_strs
        }
    )
    lga_days_since_by_key = compute_pit_strict_days_since_trough(conn, lga_lookup_dates)

    # Cache 9: per-brand days_since_trough_entry, PIT-strict.
    # Same contract as Cache 8 but uses brand median (non-Sticky stations).
    brand_days_since_by_key = compute_pit_strict_days_since_trough_brand(
        conn, label_date_strs, qualifying_brand_list
    )

    # Cache 10: network_px_std per date — sample std of E10 prices over the
    # canonical Competitive cohort (sc.class = 'Competitive'). Computed once
    # over the full daily_prices history; .get() returns None for sparse dates.
    network_px_std_by_date = _network_px_std_per_date(conn, fid)

    # Cache 11: network_px_std_delta_3d — calendar-aware level(d) − level(d-3).
    network_px_std_delta_by_date = _calendar_delta(
        network_px_std_by_date, label_date_strs
    )

    # Cache 12: lga_phase_std per date — sample std of the 35 LGA
    # days_since_trough_entry values from Cache 8. Computed over the same
    # extended date set so Cache 13 has its prior.
    lga_phase_std_by_date = _lga_phase_std_per_date(
        lga_days_since_by_key, lga_lookup_dates
    )

    # Cache 13: lga_phase_std_delta_3d — calendar-aware level(d) − level(d-3).
    lga_phase_std_delta_by_date = _calendar_delta(
        lga_phase_std_by_date, label_date_strs
    )

    # Cache 14: tgp_delta_7d per date — PIT Sydney ULP TGP 7-day momentum (#271).
    # Empty when the tgp table is unpopulated (e.g. a DB built before chip 2 ran);
    # the column is then all-NaN, never a row drop.
    tgp_delta_by_date = _tgp_delta_by_date(conn)

    # Every (station_code, price_date) in label_df came from daily_prices, and
    # every label date is in cd._series for the same reason — so cache misses
    # on station_price / sydney_avg are upstream bugs, not data conditions.
    # Letting dict[key] raise KeyError surfaces them rather than silently
    # dropping rows the per-row path would have kept.
    # lga_mean / brand_mean / trough features are legitimately absent → .get()
    records = []
    for row_dict in label_df.to_dict("records"):
        date_d: str = row_dict["price_date"]
        state = cycle_state_by_date[date_d]
        if state is None:
            continue
        sc = row_dict["station_code"]
        station_price = station_price_by_key[(sc, date_d)]
        sydney_avg = sydney_avg_by_date[date_d]
        lga, brand = station_info_by_code.get(sc, (None, None))
        lga_mean = lga_mean_by_key.get((date_d, lga)) if lga else None
        brand_mean = brand_mean_by_key.get((date_d, brand)) if brand else None
        stickiness_score = stickiness_by_key.get((sc, date_d))
        feature_dict = _build_feature_dict(state, station_price, sydney_avg, lga_mean, brand_mean, stickiness_score)
        for _lga in LGA_FEATURE_COUNCILS:
            feature_dict[f"days_since_trough_entry_{lga_slug(_lga)}"] = (
                lga_days_since_by_key.get((date_d, _lga))
            )
        feature_dict["network_px_std"] = network_px_std_by_date.get(date_d)
        feature_dict["network_px_std_delta_3d"] = network_px_std_delta_by_date.get(date_d)
        feature_dict["lga_phase_std"] = lga_phase_std_by_date.get(date_d)
        feature_dict["lga_phase_std_delta_3d"] = lga_phase_std_delta_by_date.get(date_d)
        feature_dict["tgp_delta_7d"] = tgp_delta_by_date.get(date_d)
        for _brand in qualifying_brand_list:
            feature_dict[f"days_since_trough_entry_{brand_slug(_brand)}"] = (
                brand_days_since_by_key.get((date_d, _brand))
            )
        records.append({**row_dict, **feature_dict})

    if not records:
        return pd.DataFrame(columns=all_cols)
    df = pd.DataFrame(records, columns=all_cols)
    # LGA/brand trough values are int|None in records; pandas assigns object dtype
    # to any column with mixed int+None or all-None values.  LightGBM rejects
    # object columns — cast to float so None → NaN and dtype becomes float64.
    _cast_trough_columns(df)
    # Same object-dtype risk for tgp_delta_7d when the tgp table is empty (all-None).
    df["tgp_delta_7d"] = pd.to_numeric(df["tgp_delta_7d"], errors="raise")
    return df


def _cast_trough_columns(df: pd.DataFrame) -> None:
    """In-place: cast days_since_trough_entry_* columns to float64.

    Raises on unexpected non-numeric values so upstream data bugs surface
    immediately rather than being silently converted to NaN.
    """
    for c in df.columns:
        if c.startswith(_TROUGH_PREFIX):
            df[c] = pd.to_numeric(df[c], errors="raise")


def load_features(path: pathlib.Path | str = DEFAULT_FEATURES_CSV) -> pd.DataFrame:
    """Load features from parquet cache when fresher than CSV, else from CSV."""
    csv_path = pathlib.Path(path)
    parquet_path = csv_path.with_suffix(".parquet")
    if parquet_path.exists() and (
        not csv_path.exists()
        or parquet_path.stat().st_mtime >= csv_path.stat().st_mtime
    ):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(csv_path)
    # Normalise trough columns on both paths: parquet preserves object dtype,
    # and CSV behaviour may change in future pandas versions.
    _cast_trough_columns(df)
    return df


@click.command("features")
@click.option(
    "--output",
    default="data/features.csv",
    show_default=True,
    help="Output CSV path.",
)
@click.option("--horizon", type=click.IntRange(min=1), default=7, show_default=True, help="Forward horizon in days.")
@click.option(
    "--threshold", type=click.FloatRange(min=0.0), default=3.0, show_default=True,
    help="Minimum price drop (cents) to label as 1.",
)
@click.option(
    "--lookback", type=click.IntRange(min=1), default=90, show_default=True,
    help="Past days for price percentile (~2 cycles).",
)
@click.option(
    "--percentile", type=click.FloatRange(min=0.0, max=100.0), default=33.0, show_default=True,
    help="Percentile gate for 'price is cheap' condition.",
)
@click.option(
    "--min-rows", "min_rows", type=click.IntRange(min=0), default=MIN_TRAINING_ROWS_PER_STATION,
    show_default=True,
    help="Minimum label rows per station to include in training set (0 = no filter).",
)
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite DB.",
)
def main(  # noqa: PLR0913
    output: str, horizon: int, threshold: float, lookback: int,
    percentile: float, min_rows: int, db_path: str,
) -> None:
    """Assemble ML training rows with cycle features joined to labels."""
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )

    conn = _db.open_db(path)
    df = assemble_feature_rows(
        conn,
        horizon_days=horizon,
        threshold_cents=threshold,
        lookback_days=lookback,
        percentile_pct=percentile,
        min_rows_per_station=min_rows,
    )
    conn.close()

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    df.to_parquet(out_path.with_suffix(".parquet"), index=False)
    click.echo(f"Wrote {len(df):,} rows ({int(df['label'].sum()):,} positive) to {out_path}")


if __name__ == "__main__":
    main()
