"""Cycle detection for E10 price series.

Usage:
    series = db.average_price_series(conn)                  # full history
    cd = CycleDetector(series)                              # converts once
    state = cd.detect("2024-06-15")                         # in-memory slice
    if state:
        print(f"{state.pct_through_cycle:.0%} through cycle")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.signal

logger = logging.getLogger(__name__)


@dataclass
class CycleState:
    """Snapshot of cycle phase as of a specific date."""

    as_of_date: str          # YYYY-MM-DD; series was truncated here
    days_since_last_peak: int
    mean_cycle_length: float  # mean inter-peak distance in days
    pct_through_cycle: float  # days_since_last_peak / mean_cycle_length; can exceed 1.0
    last_cycle_min: float     # cents; min price in the last complete cycle
    last_cycle_max: float     # cents; max price in the last complete cycle
    last_3_gradients: list[float]  # np.gradient of the (possibly smoothed) series, last 3
    peak_count: int           # number of peaks detected; low values mean low confidence


class CycleDetector:
    """Detect price cycle phase from a daily price series.

    Parameters
    ----------
    series:
        ``[(date_str, price_cents), ...]`` — any aggregated daily series.
        Dates must be in YYYY-MM-DD format and chronologically ordered.
        The caller decides which series to use (Sydney metro average,
        Blue Mountains council average, etc.).
    smooth:
        If True, apply a Savitzky-Golay filter (window=7, polyorder=2) before
        peak detection.  Reduces noise in sparse series (e.g. BM council average
        with ~20–30 stations) without introducing the lag of a rolling mean.
        Leave False for the Sydney metro average, which is already smooth.
    """

    # scipy.signal.find_peaks parameters — tuned for NSW E10 weekly price cycles
    _PEAK_DISTANCE = 7       # minimum days between peaks
    _PEAK_PROMINENCE = 1.0   # minimum cents of prominence

    def __init__(
        self,
        series: list[tuple[str, float]],
        smooth: bool = False,
    ) -> None:
        # _first_confirm maps peak index -> the first series position at which
        # that peak became scipy-confirmed. Built lazily and cached (see #250).
        self._first_confirm: dict[int, int] | None = None
        if not series:
            self._series: pd.Series = pd.Series(dtype=float)
            return
        dates, prices = zip(*series)
        index = pd.to_datetime(list(dates))
        values = np.array(prices, dtype=float)
        if smooth and len(values) >= 7:
            values = scipy.signal.savgol_filter(values, window_length=7, polyorder=2)
        self._series = pd.Series(values, index=index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, as_of_date: str) -> CycleState | None:
        """Return cycle state as of *as_of_date*, or None if not enough data.

        The full series is sliced to *as_of_date* (inclusive) before all
        calculations, so this is safe to call with historical dates for
        backtesting without re-querying the database.

        Returns None when fewer than 2 confirmed peaks exist (insufficient data
        for a cycle phase estimate).

        Peak confirmation (#250)
        ------------------------
        ``days_since_last_peak`` is driven solely by ``scipy.find_peaks``
        confirmed peaks, applied causally and *with memory*:

        - A peak is only counted once its right-side prominence has accumulated
          (i.e. price has fallen far enough after the peak for ``find_peaks`` to
          confirm it). This introduces a small confirmation lag near peaks.
        - Once confirmed, a peak is **never un-confirmed** — the confirmed set
          only grows, so the "last confirmed peak" index is non-decreasing in
          date. ``days_since_last_peak`` therefore counts up monotonically
          within a cycle and resets exactly once per genuine new peak.

        This replaces the old expanding-window scheme, whose
        ``_plateau_width_at_boundary`` heuristic guessed at an unconfirmed
        boundary peak and flip-flopped that guess day-to-day, producing the
        whipsaw documented in #250. That heuristic has been removed entirely.
        """
        if self._series.empty:
            return None
        sliced = self._series.loc[:as_of_date]
        if sliced.empty:
            return None

        boundary_pos = len(sliced) - 1
        peaks = self._confirmed_peaks_as_of(boundary_pos)
        if len(peaks) < 2:
            return None

        last_peak_idx = int(peaks[-1])
        last_peak_date = sliced.index[last_peak_idx]
        days_since_last_peak = (pd.Timestamp(as_of_date) - last_peak_date).days

        mean_cycle_length = self._mean_cycle_length(sliced, peaks)
        if mean_cycle_length <= 0 or np.isnan(mean_cycle_length):
            return None
        pct_through_cycle = days_since_last_peak / mean_cycle_length

        last_cycle = self._last_cycle_prices(sliced, peaks)
        if last_cycle.empty:
            return None
        last_cycle_min = float(last_cycle.min())
        last_cycle_max = float(last_cycle.max())

        last_3_gradients = np.gradient(sliced.values)[-3:].round(2).tolist()

        return CycleState(
            as_of_date=as_of_date,
            days_since_last_peak=days_since_last_peak,
            mean_cycle_length=mean_cycle_length,
            pct_through_cycle=pct_through_cycle,
            last_cycle_min=last_cycle_min,
            last_cycle_max=last_cycle_max,
            last_3_gradients=last_3_gradients,
            peak_count=len(peaks),
        )

    def _confirmed_peaks_as_of(self, boundary_pos: int) -> np.ndarray:
        """Return sorted peak indices confirmed at or before *boundary_pos*.

        Confirmation is computed causally and is sticky: see ``detect``. The
        returned indices are positions into ``self._series`` and are guaranteed
        ``<= boundary_pos`` (a peak's confirmation position is never earlier
        than the peak itself).
        """
        first_confirm = self._build_first_confirm()
        idxs = sorted(p for p, c in first_confirm.items() if c <= boundary_pos)
        return np.array(idxs, dtype=int)

    def _build_first_confirm(self) -> dict[int, int]:
        """Map each peak index to the first series position at which it became
        ``find_peaks``-confirmed.

        A cycle peak is confirmed (gains right-side prominence) once the price
        has fallen at least ``_PEAK_PROMINENCE`` below it — which is exactly
        when ``find_peaks`` on the growing prefix first reports it, since the
        right-side base is the running minimum of the descent. So we run
        ``find_peaks`` once over the full held series for the canonical peak set
        and, in a single forward pass, record each peak's confirmation position
        as the first day its cumulative drop crosses the prominence threshold.

        Confirmation is therefore causal (uses only data up to that position)
        and sticky: a peak, once confirmed, is in the set for all later dates,
        so ``_confirmed_peaks_as_of`` is monotone in date (#250). A trailing
        peak whose drop has not yet accumulated is simply absent until it does.

        Cached after the first call. Cost is one ``find_peaks`` plus an O(n)
        scan; the detector is built once per backtest/feature run.
        """
        if self._first_confirm is not None:
            return self._first_confirm
        first_confirm: dict[int, int] = {}
        values = self._series.values
        n = len(values)
        peaks, _ = scipy.signal.find_peaks(
            values,
            distance=self._PEAK_DISTANCE,
            prominence=self._PEAK_PROMINENCE,
        )
        for raw_p in peaks:
            p = int(raw_p)
            running_min = values[p]
            for j in range(p + 1, n):
                if values[j] < running_min:
                    running_min = values[j]
                if values[p] - running_min >= self._PEAK_PROMINENCE:
                    first_confirm[p] = j
                    break
        self._first_confirm = first_confirm
        return first_confirm

    # ------------------------------------------------------------------
    # Internal helpers (ported from ff-aws-backend PriceCycleDetector)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_peaks(series: pd.Series) -> tuple[np.ndarray, dict]:
        """Run scipy peak detection on *series*. Returns (peak_indices, properties)."""
        # TODO: implement
        # scipy.signal.find_peaks(
        #     series.values,
        #     distance=CycleDetector._PEAK_DISTANCE,
        #     prominence=CycleDetector._PEAK_PROMINENCE,
        # )
        return scipy.signal.find_peaks(
            series.values,
            distance=CycleDetector._PEAK_DISTANCE,
            prominence=CycleDetector._PEAK_PROMINENCE,
        )

    @staticmethod
    def _last_cycle_prices(
        series: pd.Series,
        peak_indices: np.ndarray,
    ) -> pd.Series:
        """Return the price sub-series spanning the last complete cycle.

        The last complete cycle runs from peak[-2] to peak[-1]. Returns an empty
        Series if fewer than 2 confirmed peaks exist.

        Ported from ff-aws-backend PriceCycleDetector._get_last_cycle_prices.
        """
        if len(peak_indices) >= 2:
            return series.iloc[peak_indices[-2] : peak_indices[-1]]
        return pd.Series(dtype=float)

    def peaks_for_plot(self, as_of_date: str | None = None) -> dict:
        """Return peak metadata for visualisation in the inspection page.

        Uses the same confirmed-peak timeline as ``detect`` (#250): peaks are
        scipy-confirmed and sticky, so the chart matches the feature.

        Returns:
            peak_dates:         YYYY-MM-DD list of confirmed peak dates
            last_cycle_start:   left edge of the last-cycle window (date str or None)
            last_cycle_end:     right edge of the last-cycle window (date str or None)
        """
        empty: dict = {
            "peak_dates": [],
            "last_cycle_start": None,
            "last_cycle_end": None,
        }
        if self._series.empty:
            return empty
        series = self._series.loc[:as_of_date] if as_of_date else self._series
        if series.empty:
            return empty

        peaks = self._confirmed_peaks_as_of(len(series) - 1)
        peak_dates = [series.index[int(i)].strftime("%Y-%m-%d") for i in peaks]

        last_cycle_start = last_cycle_end = None
        if len(peaks) >= 2:
            last_cycle_start = series.index[int(peaks[-2])].strftime("%Y-%m-%d")
            last_cycle_end = series.index[int(peaks[-1])].strftime("%Y-%m-%d")

        return {
            "peak_dates": peak_dates,
            "last_cycle_start": last_cycle_start,
            "last_cycle_end": last_cycle_end,
        }

    @staticmethod
    def _mean_cycle_length(
        series: pd.Series,
        peak_indices: np.ndarray,
    ) -> float:
        """Return mean inter-peak distance in days over the confirmed peaks.

        Ported from ff-aws-backend PriceCycleDetector.get_mean_cycle_time.
        """
        if len(peak_indices) < 2:
            return float("nan")
        cycle_times = series.index[peak_indices].astype("datetime64[ns]")
        days_between_peaks = (
            np.diff(cycle_times).astype("float64") / 1_000_000_000 / 60 / 60 / 24
        )
        return float(round(np.average(days_between_peaks), 2))
