"""Tests for fuel_signal.cycle — CycleDetector and CycleState.

Synthetic series strategy
--------------------------
All tests use programmatically generated price series with *known* cycle
parameters so expected outputs are computed, not hand-tuned.

A canonical NSW-style E10 cycle looks like:
  - Sharp rise to a peak over ~3 days (+25c)
  - Gradual decline to a trough over ~43 days (-25c)
  - Total cycle length: ~46 days
  - Amplitude: ~25c peak-to-trough

Helper ``_sawtooth_series`` generates N full cycles of this shape starting
from a given base date, returning a list[(date_str, price_cents)] that
CycleDetector can consume directly.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import scipy.signal

from fuel_signal.cycle import CycleDetector

# ---------------------------------------------------------------------------
# Synthetic series helpers
# ---------------------------------------------------------------------------

_CYCLE_LENGTH = 46       # days per cycle
_RISE_DAYS = 3           # days of sharp rise to peak
_BASE_PRICE = 150.0      # cents at trough
_AMPLITUDE = 25.0        # cents peak-to-trough


def _sawtooth_series(
    n_cycles: float = 3.0,
    cycle_length: int = _CYCLE_LENGTH,
    base_price: float = _BASE_PRICE,
    amplitude: float = _AMPLITUDE,
    start: str = "2020-01-01",
) -> list[tuple[str, float]]:
    """Return a list[(date_str, price_cents)] with *n_cycles* full price cycles.

    Each cycle: sharp 3-day rise to peak, then linear decline back to trough.
    No noise — peaks are exactly at known positions so test assertions are exact.
    """
    total_days = int(n_cycles * cycle_length)
    start_date = datetime.date.fromisoformat(start)
    result = []
    for day in range(total_days):
        pos = day % cycle_length
        if pos < _RISE_DAYS:
            # Rising phase: linear ramp from trough to peak
            price = base_price + amplitude * (pos / _RISE_DAYS)
        else:
            # Falling phase: linear decline from peak back to trough
            price = base_price + amplitude * (1.0 - (pos - _RISE_DAYS) / (cycle_length - _RISE_DAYS))
        date_str = (start_date + datetime.timedelta(days=day)).isoformat()
        result.append((date_str, round(price, 1)))
    return result


def _partial_series(series: list[tuple[str, float]], as_of_date: str) -> list[tuple[str, float]]:
    """Truncate *series* to rows on or before *as_of_date*."""
    return [(d, p) for d, p in series if d <= as_of_date]


# ---------------------------------------------------------------------------
# CycleDetector — basic construction
# ---------------------------------------------------------------------------

class TestCycleDetectorConstruction:
    def test_empty_series_does_not_raise(self):
        cd = CycleDetector([])
        assert cd.detect("2024-01-01") is None

    def test_single_point_returns_none(self):
        cd = CycleDetector([("2024-01-01", 150.0)])
        assert cd.detect("2024-01-01") is None

    def test_as_of_date_before_series_returns_none(self):
        series = _sawtooth_series(n_cycles=3.0)
        cd = CycleDetector(series)
        assert cd.detect("2019-01-01") is None

    def test_insufficient_peaks_returns_none(self):
        # Only 30 days — not enough for 2 peaks at distance=7
        series = _sawtooth_series(n_cycles=0.5)
        cd = CycleDetector(series)
        last_date = series[-1][0]
        assert cd.detect(last_date) is None


# ---------------------------------------------------------------------------
# CycleDetector — cycle length estimation
# ---------------------------------------------------------------------------

class TestMeanCycleLength:
    def test_correct_cycle_length_recovered(self):
        """Mean cycle length should be close to _CYCLE_LENGTH for clean series."""
        series = _sawtooth_series(n_cycles=5.0)
        cd = CycleDetector(series)
        last_date = series[-1][0]
        state = cd.detect(last_date)
        assert state is not None
        assert abs(state.mean_cycle_length - _CYCLE_LENGTH) <= 2.0

    def test_peak_count_matches_expected(self):
        n = 5
        series = _sawtooth_series(n_cycles=float(n))
        cd = CycleDetector(series)
        state = cd.detect(series[-1][0])
        assert state is not None
        # Should detect n-1 to n peaks (last peak may land right at boundary)
        assert state.peak_count >= n - 1

    def test_different_cycle_length(self):
        series = _sawtooth_series(n_cycles=4.0, cycle_length=30)
        cd = CycleDetector(series)
        state = cd.detect(series[-1][0])
        assert state is not None
        assert abs(state.mean_cycle_length - 30) <= 2.0


# ---------------------------------------------------------------------------
# CycleDetector — phase / pct_through_cycle
# ---------------------------------------------------------------------------

class TestCyclePhase:
    def test_near_trough_gives_high_pct(self):
        """Day 40 of a 46-day cycle should give ~87% through."""
        series = _sawtooth_series(n_cycles=4.0)
        # Find a date that is ~40 days after a peak
        cd = CycleDetector(series)
        # Peak of cycle 4 is at day 3*46 + 3 = 141 (0-indexed)
        # 40 days later = day 181
        as_of_date = series[180][0]
        state = cd.detect(as_of_date)
        assert state is not None
        assert state.pct_through_cycle > 0.75

    def test_near_peak_gives_low_pct(self):
        """Day 5 of a 46-day cycle should give ~11% through."""
        series = _sawtooth_series(n_cycles=4.0)
        # Peak of cycle 4 is day 141; day 5 after = day 146
        cd = CycleDetector(series)
        as_of_date = series[145][0]
        state = cd.detect(as_of_date)
        assert state is not None
        assert state.pct_through_cycle < 0.25

    def test_pct_can_exceed_one_when_overdue(self):
        """If cycle is overdue (no peak for longer than mean_cycle_length), pct > 1.0."""
        # Build a series with 3 normal cycles then extend with a long flat tail
        series = _sawtooth_series(n_cycles=3.0)
        last_date_dt = datetime.date.fromisoformat(series[-1][0])
        # Add 60 days of flat prices after the last cycle
        flat_price = _BASE_PRICE
        extended = list(series)
        for i in range(1, 61):
            d = (last_date_dt + datetime.timedelta(days=i)).isoformat()
            extended.append((d, flat_price))
        cd = CycleDetector(extended)
        state = cd.detect(extended[-1][0])
        assert state is not None
        assert state.pct_through_cycle > 1.0


# ---------------------------------------------------------------------------
# CycleDetector — last cycle min / max
# ---------------------------------------------------------------------------

class TestLastCycleMinMax:
    def test_last_cycle_min_near_base_price(self):
        series = _sawtooth_series(n_cycles=4.0)
        cd = CycleDetector(series)
        state = cd.detect(series[-1][0])
        assert state is not None
        assert abs(state.last_cycle_min - _BASE_PRICE) <= 2.0

    def test_last_cycle_max_near_peak_price(self):
        series = _sawtooth_series(n_cycles=4.0)
        cd = CycleDetector(series)
        state = cd.detect(series[-1][0])
        assert state is not None
        assert abs(state.last_cycle_max - (_BASE_PRICE + _AMPLITUDE)) <= 2.0


# ---------------------------------------------------------------------------
# CycleDetector — gradients
# ---------------------------------------------------------------------------

class TestGradients:
    def test_three_gradients_returned(self):
        series = _sawtooth_series(n_cycles=4.0)
        cd = CycleDetector(series)
        state = cd.detect(series[-1][0])
        assert state is not None
        assert len(state.last_3_gradients) == 3

    def test_gradient_negative_during_decline(self):
        """Mid-decline gradients should be negative."""
        series = _sawtooth_series(n_cycles=3.0)
        # Mid-decline of cycle 3: around day 3*46/2 = ~69
        as_of_date = series[100][0]
        cd = CycleDetector(series)
        state = cd.detect(as_of_date)
        assert state is not None
        assert all(g < 0 for g in state.last_3_gradients)

    def test_gradient_near_zero_at_trough(self):
        """Gradients should be near zero at the trough (price flatline)."""
        series = _sawtooth_series(n_cycles=3.0)
        # Just before the next peak: trough region at the end of the series
        as_of_date = series[-3][0]
        cd = CycleDetector(series)
        state = cd.detect(as_of_date)
        assert state is not None
        assert all(abs(g) < 2.0 for g in state.last_3_gradients)


# ---------------------------------------------------------------------------
# CycleDetector — point-in-time (as_of_date truncation)
# ---------------------------------------------------------------------------

class TestPointInTime:
    def test_detect_uses_only_data_up_to_as_of_date(self):
        """Two calls with different as_of_dates should give different states."""
        series = _sawtooth_series(n_cycles=6.0)
        cd = CycleDetector(series)  # full series loaded once
        early_date = series[90][0]   # ~2 cycles in
        late_date = series[240][0]   # ~5 cycles in
        state_early = cd.detect(early_date)
        state_late = cd.detect(late_date)
        assert state_early is not None
        assert state_late is not None
        # Mean cycle lengths should both be ~46, but computed independently
        assert abs(state_early.mean_cycle_length - _CYCLE_LENGTH) <= 3.0
        assert abs(state_late.mean_cycle_length - _CYCLE_LENGTH) <= 2.0

    def test_future_data_does_not_affect_historical_state(self):
        """CycleState at a past date should be stable regardless of what comes after."""
        series = _sawtooth_series(n_cycles=6.0)
        cd = CycleDetector(series)
        mid_date = series[180][0]
        state_from_full = cd.detect(mid_date)

        # Build a detector with only history up to mid_date
        truncated = _partial_series(series, mid_date)
        cd_truncated = CycleDetector(truncated)
        state_from_truncated = cd_truncated.detect(mid_date)

        assert state_from_full is not None
        assert state_from_truncated is not None
        assert state_from_full.days_since_last_peak == state_from_truncated.days_since_last_peak
        assert abs(state_from_full.mean_cycle_length - state_from_truncated.mean_cycle_length) < 0.01


# ---------------------------------------------------------------------------
# CycleDetector — plateau at boundary
# ---------------------------------------------------------------------------

class TestBoundaryPeakConfirmation:
    """An unconfirmed boundary peak is NOT counted until its right-side drop
    accumulates — the #250 fix. (The old code guessed at a boundary plateau
    peak, which flip-flopped day-to-day and caused the whipsaw.)"""

    def _three_cycles_then_rise_and_plateau(self):
        series = _sawtooth_series(n_cycles=3.0)
        last_date_dt = datetime.date.fromisoformat(series[-1][0])
        extended = list(series)
        # Rise phase (3 days) up to a fresh peak
        for i in range(1, 4):
            price = _BASE_PRICE + _AMPLITUDE * (i / _RISE_DAYS)
            extended.append(((last_date_dt + datetime.timedelta(days=i)).isoformat(), round(price, 1)))
        # Hold flat at the peak for 5 days (no drop yet → not confirmable)
        plateau_start = datetime.date.fromisoformat(extended[-1][0])
        peak_price = round(_BASE_PRICE + _AMPLITUDE, 1)
        for i in range(1, 6):
            extended.append(((plateau_start + datetime.timedelta(days=i)).isoformat(), peak_price))
        return extended

    def test_unconfirmed_boundary_peak_not_counted(self):
        """A fresh peak held flat at the boundary has no right-side drop, so it
        is not yet confirmed: days_since reflects the previous confirmed peak."""
        extended = self._three_cycles_then_rise_and_plateau()
        cd = CycleDetector(extended)
        state = cd.detect(extended[-1][0])
        assert state is not None
        # ~43d (rest of last cycle's decline) + 3d rise + 5d plateau ≈ 50+ days.
        assert state.days_since_last_peak > 20

    def test_boundary_peak_confirmed_once_price_drops(self):
        """Once the price falls > prominence after the boundary peak, the peak
        confirms and days_since resets to the (small) confirmation lag."""
        extended = self._three_cycles_then_rise_and_plateau()
        # Decline a few days past the plateau so the new peak gains prominence.
        drop_start = datetime.date.fromisoformat(extended[-1][0])
        peak_price = _BASE_PRICE + _AMPLITUDE
        for i in range(1, 6):
            extended.append(
                ((drop_start + datetime.timedelta(days=i)).isoformat(), round(peak_price - 2.0 * i, 1))
            )
        cd = CycleDetector(extended)
        state = cd.detect(extended[-1][0])
        assert state is not None
        # The peak was ~5 (plateau) + ~5 (drop) days ago; well under a full cycle.
        assert state.days_since_last_peak < 15

    def test_no_false_boundary_peak_on_declining_series(self):
        """A series ending in normal decline should not trigger boundary peak detection."""
        series = _sawtooth_series(n_cycles=3.5)  # ends mid-decline
        cd = CycleDetector(series)
        state = cd.detect(series[-1][0])
        # Should still produce a valid state based on the 3 confirmed peaks
        assert state is not None
        assert state.peak_count >= 2


# ---------------------------------------------------------------------------
# CycleDetector — smoothing
# ---------------------------------------------------------------------------

class TestBoundaryStability:
    """Acceptance criteria for #250: the confirmed-peak timeline is monotone,
    never whipsaws, and reconstructs the same cycle count as raw find_peaks."""

    @staticmethod
    def _noisy_cycles(n_cycles=8.0, seed=0):
        # Light noise (well below the 1.0c prominence threshold) so find_peaks
        # tracks genuine cycle peaks, mirroring the smooth metro-average series.
        rng = np.random.default_rng(seed)
        series = _sawtooth_series(n_cycles=n_cycles)
        return [(d, round(p + float(rng.normal(0, 0.3)), 1)) for d, p in series]

    def _walk(self, series):
        """Return (as_of_dates, last_peak_dates, days_since) for every date with a state."""
        cd = CycleDetector(series)
        dates, last_peaks, dsp = [], [], []
        for d, _ in series:
            st = cd.detect(d)
            if st is None:
                continue
            dates.append(pd.Timestamp(d))
            dsp.append(st.days_since_last_peak)
            last_peaks.append(pd.Timestamp(d) - pd.Timedelta(days=st.days_since_last_peak))
        return dates, last_peaks, dsp

    def test_last_peak_date_is_non_decreasing(self):
        """AC1: the confirmed last-peak date never moves backwards in time."""
        _, last_peaks, _ = self._walk(self._noisy_cycles())
        for prev, cur in zip(last_peaks, last_peaks[1:]):
            assert cur >= prev, f"last-peak regressed from {prev} to {cur}"

    def test_days_since_monotone_between_resets(self):
        """AC1: within a cycle days_since counts up by ~1/day; any drop must
        coincide with the last-peak date moving forward (a genuine new peak)."""
        _, last_peaks, dsp = self._walk(self._noisy_cycles())
        for i in range(1, len(dsp)):
            if dsp[i] < dsp[i - 1]:
                assert last_peaks[i] > last_peaks[i - 1]

    def test_no_whipsaw_resets(self):
        """AC3: no drop-then-recover-above-prior within 5 days."""
        _, _, dsp = self._walk(self._noisy_cycles())
        for i in range(1, len(dsp)):
            if dsp[i] < dsp[i - 1] - 1:
                prior = dsp[i - 1]
                assert all(dsp[j] <= prior for j in range(i + 1, min(i + 6, len(dsp))))

    def test_reconstructed_cycle_count_matches_find_peaks(self):
        """AC2: number of resets ≈ number of find_peaks peaks (within tolerance)."""
        series = self._noisy_cycles()
        _, last_peaks, _ = self._walk(series)
        n_resets = len({lp for lp in last_peaks})
        prices = np.array([p for _, p in series])
        fp, _ = scipy.signal.find_peaks(
            prices, distance=CycleDetector._PEAK_DISTANCE, prominence=CycleDetector._PEAK_PROMINENCE
        )
        assert abs(n_resets - len(fp)) <= 2


class TestSmoothing:
    def test_smooth_true_still_detects_cycles(self):
        """Smoothing should not destroy cycle detection on a clean series."""
        series = _sawtooth_series(n_cycles=4.0)
        cd = CycleDetector(series, smooth=True)
        state = cd.detect(series[-1][0])
        assert state is not None
        assert abs(state.mean_cycle_length - _CYCLE_LENGTH) <= 3.0

    def test_smooth_reduces_gradient_noise(self):
        """Noisy series: smooth=True should give lower gradient magnitudes than smooth=False."""
        rng = np.random.default_rng(42)
        series = _sawtooth_series(n_cycles=4.0)
        noisy = [(d, p + float(rng.normal(0, 2.0))) for d, p in series]

        cd_raw = CycleDetector(noisy, smooth=False)
        cd_smooth = CycleDetector(noisy, smooth=True)
        last_date = noisy[-1][0]

        state_raw = cd_raw.detect(last_date)
        state_smooth = cd_smooth.detect(last_date)
        assert state_raw is not None
        assert state_smooth is not None
        raw_grad_mag = sum(abs(g) for g in state_raw.last_3_gradients)
        smooth_grad_mag = sum(abs(g) for g in state_smooth.last_3_gradients)
        assert smooth_grad_mag < raw_grad_mag
