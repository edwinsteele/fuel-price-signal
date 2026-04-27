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

        Returns None when fewer than 2 peaks are detected (insufficient data
        for a cycle phase estimate).
        """
        # TODO: implement
        #
        # Steps:
        # 1. Slice self._series up to and including as_of_date.
        #    Use: sliced = self._series.loc[:as_of_date]
        #    Guard: return None if sliced is empty.
        #
        # 2. Detect peaks on the sliced series via self._get_peaks(sliced).
        #    Return None if fewer than 2 peaks found.
        #    (plateau_width_at_boundary can provide a synthetic "current peak"
        #    on the right boundary — include it in the effective peak list
        #    when it fires.)
        #
        # 3. Compute days_since_last_peak:
        #    last_peak_date = sliced.index[effective_last_peak_idx]
        #    days_since_last_peak = (pd.Timestamp(as_of_date) - last_peak_date).days
        #
        # 4. Compute mean_cycle_length via self._mean_cycle_length(sliced, peaks,
        #    plateau_width).  See _mean_cycle_length docstring.
        #
        # 5. Compute pct_through_cycle = days_since_last_peak / mean_cycle_length.
        #
        # 6. Compute last_cycle_min / last_cycle_max via
        #    self._last_cycle_prices(sliced, peaks, plateau_width).
        #
        # 7. Compute last_3_gradients = np.gradient(sliced.values)[-3:].round(2).tolist()
        #    (use the smoothed series if smooth=True was set — self._series is
        #    already smoothed, so np.gradient on sliced is correct regardless)
        #
        # 8. Return CycleState(...)
        if self._series.empty:
            return None
        sliced = self._series.loc[:as_of_date]
        if sliced.empty:
            return None

        peaks, _ = self._get_peaks(sliced)
        plateau_width = self._plateau_width_at_boundary(
            sliced, self._PEAK_PROMINENCE
        )

        effective_peak_count = len(peaks) + (1 if plateau_width else 0)
        if effective_peak_count < 2:
            return None

        if plateau_width:
            last_peak_idx = len(sliced) - plateau_width
        else:
            last_peak_idx = int(peaks[-1])
        last_peak_date = sliced.index[last_peak_idx]
        days_since_last_peak = (pd.Timestamp(as_of_date) - last_peak_date).days

        mean_cycle_length = self._mean_cycle_length(sliced, peaks, plateau_width)
        if mean_cycle_length <= 0 or np.isnan(mean_cycle_length):
            return None
        pct_through_cycle = days_since_last_peak / mean_cycle_length

        last_cycle = self._last_cycle_prices(sliced, peaks, plateau_width)
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
            peak_count=effective_peak_count,
        )

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
    def _plateau_width_at_boundary(series: pd.Series, prominence: float) -> int:
        """Return how many trailing points are part of an unconfirmed peak at the
        right boundary of *series*.

        scipy cannot detect a peak at the boundary because it has no right-side
        drop-off to measure prominence against.  This method identifies the case
        where the series has plateaued (or barely declined) at the end and the
        upswing before the plateau was large enough to constitute a peak.

        Returns 0 if no boundary peak is detected.

        Ported from ff-aws-backend PriceCycleDetector._plateau_width_at_boundary.
        Logic summary:
        - Compute np.gradient(series.values).
        - Walk backwards through gradients, accumulating the running decline.
        - Stop when either (a) a positive gradient is encountered (we've entered
          the ramp-up of a new cycle) or (b) the accumulated decline exceeds
          -prominence (the drop is large enough that scipy would have seen the
          peak already).
        - If the stop was caused by (a) and the gradient just before the
          positive step is >= 2 * prominence, the plateau points count as a
          peak boundary.  Return 1 + len(plateau_gradients).
        - Otherwise return 0.

        The original implementation in ff-aws-backend:

            def _plateau_width_at_boundary(self) -> int:
                plateau_width_at_boundary = 0
                if len(self.daily_prices) > 1:
                    gradients = np.gradient(self.daily_prices)
                    plateau_gradients, other_gradients = (
                        self.partition_when_threshold_reached(
                            list(reversed(gradients)),
                            -self.PRICE_SERIES_MIN_PROMINENCE,
                        )
                    )
                    if (
                        other_gradients
                        and other_gradients[0] >= 2 * self.PRICE_SERIES_MIN_PROMINENCE
                    ):
                        plateau_width_at_boundary = 1 + len(plateau_gradients)
                return plateau_width_at_boundary

            @staticmethod
            def partition_when_threshold_reached(input_list, threshold):
                running_total = 0.0
                for idx, val in enumerate(input_list):
                    if val > 0:
                        return input_list[:idx], input_list[idx:]
                    if running_total + val < threshold:
                        return input_list[:idx], input_list[idx:]
                    else:
                        running_total += val
                return input_list, []
        """
        plateau_width = 0
        if len(series) > 1:
            gradients = np.gradient(series.values)
            reversed_grads = list(reversed(gradients))
            running_total = 0.0
            plateau_grads: list[float] = reversed_grads
            other_grads: list[float] = []
            for idx, val in enumerate(reversed_grads):
                if val > 0:
                    plateau_grads = reversed_grads[:idx]
                    other_grads = reversed_grads[idx:]
                    break
                if running_total + val < -prominence:
                    plateau_grads = reversed_grads[:idx]
                    other_grads = reversed_grads[idx:]
                    break
                running_total += val
            if other_grads and other_grads[0] >= 2 * prominence:
                plateau_width = 1 + len(plateau_grads)
        return plateau_width

    @staticmethod
    def _last_cycle_prices(
        series: pd.Series,
        peak_indices: np.ndarray,
        plateau_width: int,
    ) -> pd.Series:
        """Return the price sub-series spanning the last complete cycle.

        When plateau_width > 0 the current boundary counts as the latest peak,
        so the last complete cycle runs from peak[-1] to the start of the plateau.
        When plateau_width == 0 the last complete cycle runs from peak[-2] to peak[-1].

        Returns an empty Series if fewer than 2 effective peaks exist.

        Ported from ff-aws-backend PriceCycleDetector._get_last_cycle_prices.
        """
        if len(peak_indices) >= 2:
            if plateau_width:
                return series.iloc[peak_indices[-1] : -plateau_width]
            return series.iloc[peak_indices[-2] : peak_indices[-1]]
        if len(peak_indices) >= 1 and plateau_width:
            return series.iloc[peak_indices[-1] : -plateau_width]
        return pd.Series(dtype=float)

    def peaks_for_plot(self, as_of_date: str | None = None) -> dict:
        """Return peak metadata for visualisation in the inspection page.

        Returns:
            peak_dates:         YYYY-MM-DD list of scipy-confirmed peak dates
            plateau_peak_date:  YYYY-MM-DD of the synthetic boundary peak, or None
            last_cycle_start:   left edge of the last-cycle window (date str or None)
            last_cycle_end:     right edge of the last-cycle window (date str or None)
        """
        empty: dict = {
            "peak_dates": [],
            "plateau_peak_date": None,
            "last_cycle_start": None,
            "last_cycle_end": None,
        }
        if self._series.empty:
            return empty
        series = self._series.loc[:as_of_date] if as_of_date else self._series
        if series.empty:
            return empty

        peaks, _ = self._get_peaks(series)
        plateau_width = self._plateau_width_at_boundary(series, self._PEAK_PROMINENCE)

        peak_dates = [series.index[int(i)].strftime("%Y-%m-%d") for i in peaks]
        plateau_peak_date = (
            series.index[-plateau_width].strftime("%Y-%m-%d") if plateau_width else None
        )

        last_cycle_start = last_cycle_end = None
        if len(peaks) >= 2:
            if plateau_width:
                last_cycle_start = series.index[int(peaks[-1])].strftime("%Y-%m-%d")
                last_cycle_end = series.index[-plateau_width].strftime("%Y-%m-%d")
            else:
                last_cycle_start = series.index[int(peaks[-2])].strftime("%Y-%m-%d")
                last_cycle_end = series.index[int(peaks[-1])].strftime("%Y-%m-%d")
        elif len(peaks) >= 1 and plateau_width:
            last_cycle_start = series.index[int(peaks[-1])].strftime("%Y-%m-%d")
            last_cycle_end = series.index[-plateau_width].strftime("%Y-%m-%d")

        return {
            "peak_dates": peak_dates,
            "plateau_peak_date": plateau_peak_date,
            "last_cycle_start": last_cycle_start,
            "last_cycle_end": last_cycle_end,
        }

    @staticmethod
    def _mean_cycle_length(
        series: pd.Series,
        peak_indices: np.ndarray,
        plateau_width: int,
    ) -> float:
        """Return mean inter-peak distance in days.

        When plateau_width > 0, the right-boundary plateau point is appended
        to the peak timestamps before computing inter-peak differences.

        Ported from ff-aws-backend PriceCycleDetector.get_mean_cycle_time.
        """
        effective_count = len(peak_indices) + (1 if plateau_width else 0)
        if effective_count < 2:
            return float("nan")
        peak_times = series.index[peak_indices].astype("datetime64[ns]")
        if plateau_width:
            cycle_times = np.append(
                peak_times,
                series.index[[-plateau_width]].astype("datetime64[ns]"),
            )
        else:
            cycle_times = peak_times
        days_between_peaks = (
            np.diff(cycle_times).astype("float64") / 1_000_000_000 / 60 / 60 / 24
        )
        return float(round(np.average(days_between_peaks), 2))
