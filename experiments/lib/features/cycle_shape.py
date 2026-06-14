from __future__ import annotations

import pandas as pd

# Matches #214's row-wise shallow definition (experiments/2026-06-09_shallow_elongated).
SHALLOW_SLOPE_THRESHOLD = -0.9


def label_cycle_shape(
    per_date: pd.DataFrame,
    *,
    date_col: str = "price_date",
    dsp_col: str = "cycle_days_since_peak",
    trough_price_col: str = "sydney_avg",
    peak_price_col: str = "cycle_last_max_cents",
    shallow_slope_threshold: float = SHALLOW_SLOPE_THRESHOLD,
    partial_frac: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ORACLE classification of network price cycles by eventual shape.

    ⚠️ PARKED / KNOWN-FLAWED (#237, #250). This reconstructs cycle boundaries
    from resets in ``cycle_days_since_peak``, which is UNSTABLE — it whipsaws at
    cycle boundaries (#250), so this over-segments ~2.6x vs the detector's own
    ``find_peaks(distance=7, prominence=1.0)``. Do NOT trust classifications
    from this until it is rebuilt on find_peaks. Kept only as the parked #237
    record. See ``experiments/2026-06-14_corner_oracle_sweep/README.md``.

    Reconstructs cycle boundaries from resets in ``dsp_col`` (days-since-peak
    drops when a new peak is confirmed), summarises each cycle by its length
    and peak->trough descent slope, then labels it:

      - ``normal``            : length <= train-median length
      - ``elongated_steep``   : length >  median AND descent_slope <= threshold
      - ``elongated_shallow`` : length >  median AND descent_slope >  threshold

    USES FUTURE INFO (full-cycle length + trough) — this is the oracle view,
    valid only for train-only existence diagnostics. The returned
    ``cycle_type`` must never be used as a model feature.

    Extracted verbatim from
    ``experiments/2026-06-09_shallow_elongated/phase_oracle_cycles.py`` so the
    #237 corner-oracle sweep classifies cycles identically to #214 (the two
    diagnostics are only comparable if the class definitions match).

    Parameters
    ----------
    per_date
        One row per date, sorted ascending, with a clean RangeIndex. Call
        ``df.drop_duplicates(date_col).sort_values(date_col).reset_index(drop=True)``
        first. Must contain ``date_col``, ``dsp_col``, ``trough_price_col`` and
        ``peak_price_col``.
    partial_frac
        A leading/trailing cycle shorter than ``partial_frac × median_length``
        is dropped as a boundary stub (the detector takes a couple of cycles to
        confirm the first peak; the trailing cycle may be incomplete at the
        cutoff).

    Returns
    -------
    (per_date, cyc)
        ``per_date`` is a copy of the input with ``cycle_id`` and ``cycle_type``
        columns added (``cycle_type`` is NaN for rows in dropped stub cycles).
        ``cyc`` is one row per retained cycle with summary stats + ``cycle_type``.
    """
    per_date = per_date.copy()

    # --- Cycle boundary detection ---
    # dsp resets (drops) when a new peak is confirmed; each reset starts a cycle.
    per_date["_dsp_prev"] = per_date[dsp_col].shift(1)
    per_date["_is_cycle_start"] = (
        per_date[dsp_col] < per_date["_dsp_prev"] - 1
    ).fillna(False)
    per_date.loc[per_date.index[0], "_is_cycle_start"] = True
    per_date["cycle_id"] = per_date["_is_cycle_start"].cumsum()

    # --- Per-cycle summary ---
    cyc = (
        per_date.groupby("cycle_id")
        .agg(
            start_date=(date_col, "min"),
            end_date=(date_col, "max"),
            length_days=(date_col, "size"),
            peak_price=(peak_price_col, "first"),
            trough_price=(trough_price_col, "min"),
        )
        .reset_index()
    )
    # Trough day = first date the cycle hits its minimum network price.
    trough_day = (
        per_date.merge(
            per_date.groupby("cycle_id")[trough_price_col].min().reset_index()
            .rename(columns={trough_price_col: "_min"}),
            on="cycle_id",
        )
        .loc[lambda d: d[trough_price_col] == d["_min"]]
        .groupby("cycle_id")[date_col].min()
        .rename("trough_date")
        .reset_index()
    )
    cyc = cyc.merge(trough_day, on="cycle_id")
    cyc["days_to_trough"] = (cyc["trough_date"] - cyc["start_date"]).dt.days
    cyc = cyc[cyc["days_to_trough"] > 0].copy()
    cyc["descent_slope"] = (
        (cyc["trough_price"] - cyc["peak_price"]) / cyc["days_to_trough"]
    )

    # --- Trim boundary stubs ---
    median_len = float(cyc["length_days"].median())
    if cyc.iloc[-1]["length_days"] < partial_frac * median_len:
        cyc = cyc.iloc[:-1].copy()
    if cyc.iloc[0]["length_days"] < partial_frac * median_len:
        cyc = cyc.iloc[1:].copy()

    # --- Classify (train-median length cutoff; data-driven, train-only) ---
    elongation_threshold = float(cyc["length_days"].median())
    is_elongated = cyc["length_days"] > elongation_threshold
    is_shallow = cyc["descent_slope"] > shallow_slope_threshold
    cyc["cycle_type"] = "normal"
    cyc.loc[is_elongated & ~is_shallow, "cycle_type"] = "elongated_steep"
    cyc.loc[is_elongated & is_shallow, "cycle_type"] = "elongated_shallow"

    # --- Tag each row with its parent cycle's type ---
    per_date["cycle_type"] = per_date["cycle_id"].map(
        cyc.set_index("cycle_id")["cycle_type"]
    )
    per_date = per_date.drop(columns=["_dsp_prev", "_is_cycle_start"])
    return per_date, cyc
