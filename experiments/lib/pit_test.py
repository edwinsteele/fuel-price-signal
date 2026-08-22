"""Differential PIT (point-in-time) leak test — shared by hand-written experiments and
the AI-sourced candidate pipeline (fps-3jj.2).

The leak abort for COMPUTED leaks in the candidate validation harness. Computes a candidate function
on the full frame and again on the frame truncated at several checkpoint dates; values
at dates <= the checkpoint must be identical either way. Catches the likeliest
LLM-authored leak — a whole-series aggregate such as
``df.groupby('station_code')['price'].transform('mean')`` (no shift, every row sees the
station's entire future) — on the first truncation, since only a leaky computation
changes when future rows are removed.

**What this cannot catch, by construction:** a forward-looking value already STAMPED on
the row, such as `fuel_signal.features.TARGET_COLUMNS`' `future_min_cents`. Truncating
the frame at date T removes later rows but does not alter what is written in row T's own
cell, so the recomputed value matches and the test passes. That hazard is a closed,
named set and is blocked by declaration instead, in `experiments/pipeline/validate.py`
step 0.

Applies equally to a candidate's ``add_columns`` and optional ``add_axis``: an axis
computed with future information doesn't leak into the model (the model never sees it)
but leaks into the *conclusion* ("helps a lot on days that turned out to be near the
trough" is unactionable at 7am).
"""
from __future__ import annotations

import pandas as pd


class PitLeakError(RuntimeError):
    """A compute function's output changed when future rows were truncated away.

    The function is reading data it could not have seen at decision time.
    """


def differential_pit_test(
    compute_fn,
    frame: pd.DataFrame,
    *,
    date_column: str = "price_date",
    checkpoint_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> pd.DataFrame | pd.Series:
    """Raise ``PitLeakError`` if `compute_fn(frame)` depends on rows dated after present.

    For each checkpoint date T (picked from `frame[date_column]`'s unique values at the
    given quantiles), recomputes `compute_fn` on `frame` truncated to `date_column <= T`
    and compares it against the full-frame result restricted to the same rows. A
    genuinely point-in-time function is unaffected by rows it never looked forward to;
    any difference means the function saw something it shouldn't have.

    Returns the full-frame result so callers (e.g. the validation harness's NaN-rate
    check) can reuse it instead of recomputing.

    Comparison is by index label, not row position: `compute_fn`'s output only needs to
    preserve `frame`'s row index (any `df.copy()`/`sort_values()`-style transform is
    fine); a `reset_index(drop=True)` that discards it would produce false-positive
    leaks here, since aligned rows would no longer line up.
    """
    full_result = compute_fn(frame)
    dates = frame[date_column]
    unique_dates = sorted(dates.unique())
    if len(unique_dates) < 2:
        raise ValueError(
            f"differential_pit_test needs at least two distinct {date_column!r} values, "
            f"got {len(unique_dates)}"
        )

    fn_name = getattr(compute_fn, "__name__", repr(compute_fn))
    for q in checkpoint_quantiles:
        idx = int(round(q * (len(unique_dates) - 1)))
        checkpoint = unique_dates[idx]
        mask = dates <= checkpoint
        truncated_frame = frame.loc[mask].copy()
        truncated_result = compute_fn(truncated_frame)
        expected = full_result.loc[mask]
        try:
            _assert_equal(expected, truncated_result.reindex(expected.index))
        except AssertionError as exc:
            raise PitLeakError(
                f"{fn_name} leaks future information: output changed when truncating "
                f"to {date_column} <= {checkpoint!r}"
            ) from exc

    return full_result


def _assert_equal(expected, actual) -> None:
    # check_exact=True: these are deterministic recomputations of the SAME
    # underlying arithmetic, so they should be bit-identical. The default
    # tolerance-based comparison (rtol=1e-5) silently passed a real leak here
    # once — a whole-series mean of YYYYMMDD-scale price_date values differed
    # by ~1.5 out of ~20260812, comfortably inside the default relative
    # tolerance despite being a genuine, detectable leak.
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(expected, actual, check_names=False, check_exact=True)
    else:
        pd.testing.assert_frame_equal(expected, actual, check_names=False, check_exact=True)
