"""Tests for experiments.lib.universe — reproducible station-universe sampling (fps-nas).

The properties under test are the ones the realised arbiter's economics depend on:
a universe must be a pure function of its recorded spec, must not vary with DB row
order, and must never come back quietly smaller than it was asked for.
"""

import math
from datetime import date, timedelta
from fractions import Fraction

import pytest

from experiments.lib.universe import (
    ARBITER_FUEL_CODE,
    DEFAULT_MAX_STICKY_FRACTION,
    DEFAULT_MIN_COVERAGE,
    GATE_COUNCIL,
    GATE_COVERAGE_SPAN,
    GATE_COVERAGE_WINDOW,
    GATE_STICKY,
    GATE_UNCLASSIFIED,
    EligibleStation,
    UniverseSpec,
    UniverseTooSmall,
    _min_days,
    allocate_by_stratum,
    describe_universe,
    draw_universe,
    eligible_pool_digest,
    eligible_stations,
    gate_failures,
    sample_station_universe,
)
from fuel_signal.db import (
    create_schema,
    open_db,
    upsert_daily_prices,
    upsert_station_class_rows,
    upsert_stations,
)

# Postcodes chosen so each maps to a distinct Sydney-metro council.
PC_PARRAMATTA = "2150"   # → Parramatta
PC_LIVERPOOL = "2170"    # → Liverpool
PC_SYDNEY = "2000"       # → Sydney
PC_WAGGA = "2650"        # → Wagga Wagga (NOT Sydney metro)

START = "2024-01-01"
END = "2024-03-30"       # 90-day inclusive span
SPAN_DAYS = 90


def _station(code: int, postcode: str, brand: str = "Shell") -> dict:
    return {
        "station_code": code,
        "name": f"Station {code}",
        "address": f"{code} Test Street, Town",
        "suburb": "Town",
        "postcode": postcode,
        "brand": brand,
    }


def _seed_station(
    conn,
    code: int,
    postcode: str,
    *,
    price_days: int = SPAN_DAYS,
    sticky_days: int = 0,
    classified_days: int = SPAN_DAYS,
    brand: str = "Shell",
    start: str = START,
) -> None:
    """One station with `price_days` priced days and `sticky_days` Sticky classifications."""
    upsert_stations(conn, [_station(code, postcode, brand)])
    start_d = date.fromisoformat(start)
    upsert_daily_prices(
        conn,
        [
            (code, "E10", (start_d + timedelta(days=i)).isoformat(), 180.0)
            for i in range(price_days)
        ],
    )
    upsert_station_class_rows(
        conn,
        [
            (
                code,
                (start_d + timedelta(days=i)).isoformat(),
                "Sticky" if i < sticky_days else "Competitive",
                0,
            )
            for i in range(classified_days)
        ],
    )
    conn.commit()


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "universe.db")
    create_schema(c)
    yield c
    c.close()


def _spec(**kw) -> UniverseSpec:
    base = {"n": 1, "seed": 1, "start_date": START, "end_date": END}
    return UniverseSpec(**{**base, **kw})


# ---------------------------------------------------------------------------
# UniverseSpec
# ---------------------------------------------------------------------------

class TestUniverseSpec:
    def test_span_days_is_inclusive(self):
        assert _spec().span_days == SPAN_DAYS
        assert UniverseSpec(n=1, seed=1, start_date=START, end_date=START).span_days == 1

    def test_resolved_councils_defaults_to_sydney_metro_and_is_sorted(self):
        resolved = _spec().resolved_councils
        assert "Parramatta" in resolved
        assert "Wagga Wagga" not in resolved
        assert list(resolved) == sorted(resolved)

    def test_as_dict_records_resolved_councils_not_the_none_sentinel(self):
        stamp = _spec().as_dict()
        assert stamp["councils"] == list(_spec().resolved_councils)
        assert stamp["stratify_by"] == "council"
        assert stamp["span_days"] == SPAN_DAYS

    def test_allocation_stamp_states_the_property_not_a_mechanism(self):
        """`allocate_by_stratum` has no cap step and argues one would be unreachable,
        so a stamp saying "capped at stratum size" described code that does not exist.
        The stamp is the durable record of the method.
        """
        allocation = _spec().as_dict()["allocation"]
        assert "capped" not in allocation
        assert "never exceeds" in allocation

    def test_gates_dict_omits_the_draw(self):
        """describe_universe stamps this for populations no draw produced."""
        gates = _spec(n=50, seed=7).gates_dict()
        assert "n" not in gates and "seed" not in gates
        assert gates["min_days"] == _min_days(DEFAULT_MIN_COVERAGE, SPAN_DAYS)
        assert gates["coverage_gated_per_window"] is False
        assert set(_spec().as_dict()) == set(gates) | {"n", "seed", "stratify_by", "allocation"}

    def test_fuel_is_stamped_but_is_not_a_knob(self):
        """`station_class` has no fuel dimension, so a per-spec fuel could make the
        coverage gate and the Sticky gate describe different fuels. The stamp still
        records which fuel the gates mean.
        """
        assert _spec().as_dict()["fuel"] == ARBITER_FUEL_CODE == "E10"
        with pytest.raises(TypeError):
            UniverseSpec(n=1, seed=1, start_date=START, end_date=END, fuel="U91")

    @pytest.mark.parametrize(
        "kw",
        [
            {"n": 0},
            {"min_coverage": 0.0},
            {"min_coverage": 1.5},
            {"max_sticky_fraction": -0.1},
            {"max_sticky_fraction": 1.1},
            {"councils": ()},
            {"start_date": "2024-04-01", "end_date": "2024-01-01"},
            {"windows": ()},
            {"windows": (("2024-02-01", "2024-01-01"),)},   # window ends before it starts
            {"windows": (("2023-12-01", "2024-01-15"),)},   # starts before the span
            {"windows": (("2024-03-01", "2024-04-30"),)},   # ends after the span
        ],
    )
    def test_rejects_bad_config(self, kw):
        with pytest.raises(ValueError):
            _spec(**kw)


# ---------------------------------------------------------------------------
# _min_days — the exact coverage threshold
# ---------------------------------------------------------------------------

class TestMinDays:
    def test_inclusive_boundary_holds_for_thresholds_float_math_gets_wrong(self):
        """`min_coverage * span_days` is a binary-float product and can land just
        ABOVE the exact decimal, silently turning the documented inclusive boundary
        into an exclusive one. 0.56*25 is 14.000000000000002, so a station at exactly
        0.56 coverage over 25 days used to be REJECTED. The 0.90 default is exact,
        which is why the original boundary test never saw this.
        """
        assert 0.56 * 25 > 14  # the trap this guards, stated so it can't silently go away
        assert _min_days(0.56, 25) == 14
        for coverage, span in [(0.54, 50), (0.55, 20), (0.67, 3), (0.68, 25), (0.81, 100)]:
            exact = Fraction(str(coverage)) * span
            assert _min_days(coverage, span) == math.ceil(exact)
            if exact.denominator == 1:
                assert _min_days(coverage, span) == exact

    def test_rounds_up_when_the_threshold_is_not_a_whole_number_of_days(self):
        assert _min_days(0.9, 90) == 81       # exact
        assert _min_days(0.9, 91) == 82       # 81.9 -> 82
        assert _min_days(1.0, 7) == 7

    def test_matches_the_sql_gate_at_the_boundary(self, tmp_path):
        """End-to-end through the real query, not just the helper."""
        c = open_db(tmp_path / "boundary.db")
        create_schema(c)
        _seed_station(c, 1, PC_PARRAMATTA, price_days=14, classified_days=14, start="2024-01-01")
        spec = UniverseSpec(
            n=1, seed=1, start_date="2024-01-01", end_date="2024-01-25", min_coverage=0.56
        )
        assert spec.span_days == 25
        assert [s.station_code for s in eligible_stations(c, spec)] == [1]
        c.close()


# ---------------------------------------------------------------------------
# allocate_by_stratum
# ---------------------------------------------------------------------------

class TestAllocateByStratum:
    def test_allocation_sums_to_n_and_is_proportional(self):
        alloc = allocate_by_stratum({"a": 60, "b": 30, "c": 10}, 10)
        assert sum(alloc.values()) == 10
        assert alloc == {"a": 6, "b": 3, "c": 1}

    def test_every_stratum_is_present_even_at_zero(self):
        alloc = allocate_by_stratum({"a": 100, "b": 1}, 2)
        assert set(alloc) == {"a", "b"}
        assert sum(alloc.values()) == 2

    def test_remainder_ties_break_on_name_not_dict_order(self):
        forward = allocate_by_stratum({"a": 1, "b": 1, "c": 1}, 2)
        reversed_order = allocate_by_stratum({"c": 1, "b": 1, "a": 1}, 2)
        assert forward == reversed_order == {"a": 1, "b": 1, "c": 0}

    def test_never_allocates_more_than_a_stratum_holds(self):
        """Sweep the whole domain of the docstring's largest-remainder argument.

        The function has no cap-and-redistribute pass — it asserts instead — so
        this is what stands behind that claim. Shapes chosen to include the cases
        that make over-allocation plausible: one dominant stratum, singletons,
        exact ties, and every `n` up to and including the full pool.
        """
        shapes = [
            {"a": 1, "b": 5, "c": 5},
            {"a": 1, "b": 1, "c": 1},
            {"a": 97, "b": 1, "c": 1},
            {"a": 3, "b": 3, "c": 3, "d": 3},
            {"a": 2, "b": 3, "c": 5, "d": 7, "e": 11},
            {"a": 1, "b": 2},
        ]
        for sizes in shapes:
            for n in range(sum(sizes.values()) + 1):
                alloc = allocate_by_stratum(sizes, n)
                assert sum(alloc.values()) == n, (sizes, n, alloc)
                assert all(alloc[k] <= sizes[k] for k in sizes), (sizes, n, alloc)
                assert all(v >= 0 for v in alloc.values()), (sizes, n, alloc)

    def test_n_equal_to_total_takes_everything(self):
        sizes = {"a": 3, "b": 4}
        assert allocate_by_stratum(sizes, 7) == sizes

    def test_rejects_negative_stratum_sizes(self):
        """A negative size shrinks `total`, inflating every other quota past what
        that stratum holds — which would falsify the no-over-allocation argument
        the function asserts on rather than repairs.
        """
        with pytest.raises(ValueError, match="sizes must be >= 0"):
            allocate_by_stratum({"a": 5, "b": -1}, 2)

    def test_rejects_n_larger_than_the_pool(self):
        with pytest.raises(ValueError, match="strata holding"):
            allocate_by_stratum({"a": 2}, 3)

    def test_empty_pool_allocates_nothing(self):
        assert allocate_by_stratum({"a": 0, "b": 0}, 0) == {"a": 0, "b": 0}


# ---------------------------------------------------------------------------
# eligible_stations — the gates
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_a_fully_covered_competitive_metro_station_is_eligible(self, conn):
        _seed_station(conn, 1, PC_PARRAMATTA)
        (station,) = eligible_stations(conn, _spec())
        assert station.station_code == 1
        assert station.council == "Parramatta"
        assert station.coverage == pytest.approx(1.0)
        assert station.sticky_fraction == pytest.approx(0.0)

    def test_low_coverage_is_excluded_and_the_boundary_is_inclusive(self, conn):
        just_under = int(DEFAULT_MIN_COVERAGE * SPAN_DAYS) - 1
        _seed_station(conn, 1, PC_PARRAMATTA, price_days=just_under)
        _seed_station(conn, 2, PC_PARRAMATTA, price_days=just_under + 1)
        assert [s.station_code for s in eligible_stations(conn, _spec())] == [2]

    def test_mostly_sticky_is_excluded_and_the_boundary_is_inclusive(self, conn):
        at_bar = int(DEFAULT_MAX_STICKY_FRACTION * SPAN_DAYS)
        _seed_station(conn, 1, PC_PARRAMATTA, sticky_days=at_bar + 1)
        _seed_station(conn, 2, PC_PARRAMATTA, sticky_days=at_bar)
        assert [s.station_code for s in eligible_stations(conn, _spec())] == [2]

    def test_never_classified_is_excluded(self, conn):
        _seed_station(conn, 1, PC_PARRAMATTA, classified_days=0)
        _seed_station(conn, 2, PC_PARRAMATTA)
        assert [s.station_code for s in eligible_stations(conn, _spec())] == [2]

    def test_classifications_outside_the_span_do_not_count(self, conn):
        # Sticky throughout a PRIOR year, Competitive inside the span: eligible.
        _seed_station(conn, 1, PC_PARRAMATTA)
        upsert_station_class_rows(conn, [(1, "2023-01-01", "Sticky", 0)])
        conn.commit()
        (station,) = eligible_stations(conn, _spec())
        assert station.sticky_fraction == pytest.approx(0.0)

    def test_council_allow_list_is_applied(self, conn):
        _seed_station(conn, 1, PC_PARRAMATTA)
        _seed_station(conn, 2, PC_LIVERPOOL)
        _seed_station(conn, 3, PC_WAGGA)
        # Default (Sydney metro) drops the non-metro station entirely.
        assert [s.station_code for s in eligible_stations(conn, _spec())] == [1, 2]
        narrowed = _spec(councils=("Liverpool",))
        assert [s.station_code for s in eligible_stations(conn, narrowed)] == [2]

    def test_prices_for_another_fuel_do_not_count_as_coverage(self, conn):
        upsert_stations(conn, [_station(1, PC_PARRAMATTA)])
        start_d = date.fromisoformat(START)
        upsert_daily_prices(
            conn,
            [(1, "U91", (start_d + timedelta(days=i)).isoformat(), 190.0) for i in range(SPAN_DAYS)],
        )
        upsert_station_class_rows(conn, [(1, START, "Competitive", 0)])
        conn.commit()
        assert eligible_stations(conn, _spec()) == []

    def test_classified_only_outside_the_span_is_excluded(self, conn):
        """The gate-2/gate-3 interaction: fully priced inside the span, but every
        classification falls outside it. Coverage alone would admit it; gate 3's
        "at least one row IN the span" is what must not.
        """
        _seed_station(conn, 1, PC_PARRAMATTA, classified_days=0)
        upsert_station_class_rows(conn, [(1, "2023-06-01", "Competitive", 0)])
        conn.commit()
        assert eligible_stations(conn, _spec()) == []
        assert gate_failures(conn, _spec(), [1]) == {1: [GATE_UNCLASSIFIED]}

    def test_returns_empty_rather_than_raising_on_an_empty_db(self, conn):
        assert eligible_stations(conn, _spec()) == []

    def test_restrict_to_narrows_the_scan_without_changing_the_result(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 3, PC_LIVERPOOL: 2})
        everything = [s.station_code for s in eligible_stations(conn, _spec())]
        assert everything == [1, 2, 3, 4, 5]
        assert [s.station_code for s in eligible_stations(conn, _spec(), restrict_to=[2, 4])] == [2, 4]
        assert eligible_stations(conn, _spec(), restrict_to=[]) == []


# ---------------------------------------------------------------------------
# The per-window coverage gate (PR #361 review finding 1)
# ---------------------------------------------------------------------------

class TestWindowCoverageGate:
    """A span-level threshold cannot see a dark run shorter than the span.

    Measured on batch1: the span gate alone admits 4 stations with a dark run of
    90+ days — longer than a whole val window — the worst at 113 consecutive days,
    which is exactly the class of station the gate exists to exclude.
    """

    #: Three consecutive 30-day windows inside the 90-day fixture span.
    WINDOWS = (
        ("2024-01-01", "2024-01-30"),
        ("2024-01-31", "2024-02-29"),
        ("2024-03-01", "2024-03-30"),
    )

    def _seed_dark_middle_window(self, conn, code: int) -> None:
        """Priced in windows 1 and 3, entirely dark in window 2.

        60/90 = 0.667 over the span, so it fails a 0.90 span gate — but it clears a
        0.60 one while being 100% dark for a whole window. That is the shape.
        """
        upsert_stations(conn, [_station(code, PC_PARRAMATTA)])
        start = date.fromisoformat(START)
        priced = [i for i in range(SPAN_DAYS) if not (30 <= i < 60)]
        upsert_daily_prices(
            conn,
            [(code, "E10", (start + timedelta(days=i)).isoformat(), 180.0) for i in priced],
        )
        upsert_station_class_rows(
            conn,
            [(code, (start + timedelta(days=i)).isoformat(), "Competitive", 0) for i in range(SPAN_DAYS)],
        )
        conn.commit()

    def test_span_gate_alone_admits_a_station_dark_for_a_whole_window(self, conn):
        """The defect, pinned as a test so the fix cannot silently regress."""
        self._seed_dark_middle_window(conn, 1)
        span_only = _spec(min_coverage=0.60)
        (station,) = eligible_stations(conn, span_only)
        assert station.station_code == 1
        assert station.coverage == pytest.approx(60 / SPAN_DAYS)
        assert station.worst_window_coverage is None  # nothing here can reveal the gap

    def test_window_gate_rejects_it(self, conn):
        self._seed_dark_middle_window(conn, 1)
        gated = _spec(min_coverage=0.60, windows=self.WINDOWS)
        assert eligible_stations(conn, gated) == []
        assert gate_failures(conn, gated, [1]) == {1: [GATE_COVERAGE_WINDOW]}

    def test_window_gate_keeps_a_station_that_is_evenly_covered(self, conn):
        """Same 0.667 span coverage, spread evenly instead of concentrated: admitted.

        Without this, the gate could pass its rejection test by rejecting everything.
        """
        upsert_stations(conn, [_station(1, PC_PARRAMATTA)])
        start = date.fromisoformat(START)
        priced = [i for i in range(SPAN_DAYS) if i % 3]  # 60 of 90, 20 per window
        upsert_daily_prices(
            conn, [(1, "E10", (start + timedelta(days=i)).isoformat(), 180.0) for i in priced]
        )
        upsert_station_class_rows(
            conn,
            [(1, (start + timedelta(days=i)).isoformat(), "Competitive", 0) for i in range(SPAN_DAYS)],
        )
        conn.commit()
        gated = _spec(min_coverage=0.60, windows=self.WINDOWS)
        (station,) = eligible_stations(conn, gated)
        assert station.worst_window_coverage == pytest.approx(20 / 30)

    def test_worst_window_coverage_reaches_describe_universe(self, conn):
        self._seed_dark_middle_window(conn, 1)
        gated = _spec(min_coverage=0.60, windows=self.WINDOWS)
        described = describe_universe(conn, [1], gated)
        assert described["worst_window_coverage"] == 0.0
        assert described["spec_gates"]["coverage_gated_per_window"] is True
        assert described["gate_failures"] == {"1": [GATE_COVERAGE_WINDOW]}

    def test_sampling_honours_the_window_gate(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 2})
        self._seed_dark_middle_window(conn, 99)
        gated = _spec(n=2, min_coverage=0.60, windows=self.WINDOWS)
        assert 99 not in sample_station_universe(conn, gated)


# ---------------------------------------------------------------------------
# eligible_pool_digest (PR #361 review finding 2)
# ---------------------------------------------------------------------------

class TestPoolDigest:
    """`daily_prices` is a rebuilt derived table (fill.py DELETEs and refills it), so
    a spec alone does not pin a universe. The digest is the DB-vintage half of a stamp.
    """

    def test_is_stable_for_an_unchanged_db(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 3})
        assert eligible_pool_digest(conn, _spec()) == eligible_pool_digest(conn, _spec())

    def test_moves_when_the_pool_moves_under_an_identical_spec(self, conn):
        """The whole point: same spec, different digest."""
        _seed_pool(conn, {PC_PARRAMATTA: 3})
        before = eligible_pool_digest(conn, _spec())
        _seed_station(conn, 99, PC_PARRAMATTA)
        after = eligible_pool_digest(conn, _spec())
        assert before != after
        assert after.startswith("4:") and before.startswith("3:")

    def test_moves_when_only_a_members_coverage_changes(self, conn):
        """Membership alone is not enough — a gap being back-filled changes the
        replayed price path without changing who is in the pool.
        """
        _seed_station(conn, 1, PC_PARRAMATTA, price_days=SPAN_DAYS - 3)
        before = eligible_pool_digest(conn, _spec())
        upsert_daily_prices(
            conn,
            [
                (1, "E10", (date.fromisoformat(START) + timedelta(days=i)).isoformat(), 180.0)
                for i in range(SPAN_DAYS - 3, SPAN_DAYS)
            ],
        )
        conn.commit()
        after = eligible_pool_digest(conn, _spec())
        assert before != after
        assert before.startswith("1:") and after.startswith("1:")


# ---------------------------------------------------------------------------
# sample_station_universe
# ---------------------------------------------------------------------------

def _seed_pool(conn, per_council: dict[str, int]) -> None:
    """Seed `per_council` eligible stations, keyed by postcode, codes 1..N."""
    code = 1
    for postcode, count in per_council.items():
        for _ in range(count):
            _seed_station(conn, code, postcode)
            code += 1


class TestSampling:
    def test_is_deterministic_for_a_given_seed(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 20, PC_LIVERPOOL: 20})
        spec = _spec(n=10, seed=7)
        assert sample_station_universe(conn, spec) == sample_station_universe(conn, spec)

    def test_a_different_seed_draws_a_different_universe(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 20, PC_LIVERPOOL: 20})
        a = sample_station_universe(conn, _spec(n=10, seed=7))
        b = sample_station_universe(conn, _spec(n=10, seed=8))
        assert a != b

    def test_is_independent_of_db_row_order(self, tmp_path):
        """The same pool inserted in the opposite order must sample identically.

        End-to-end companion to `TestDrawUniverse.test_is_independent_of_input_order`:
        that one pins the guarantee on the pure function (and fails if the sort is
        removed), this one confirms the DB path actually routes through it.
        """
        codes_and_postcodes = [
            (c, PC_PARRAMATTA if c % 2 else PC_LIVERPOOL) for c in range(1, 21)
        ]
        samples = []
        for order, name in ((codes_and_postcodes, "fwd"), (codes_and_postcodes[::-1], "rev")):
            c = open_db(tmp_path / f"{name}.db")
            create_schema(c)
            for code, postcode in order:
                _seed_station(c, code, postcode)
            samples.append(sample_station_universe(c, _spec(n=8, seed=3)))
            c.close()
        assert samples[0] == samples[1]

    def test_eligible_stations_comes_back_sorted_by_code(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 3, PC_LIVERPOOL: 3})
        codes = [s.station_code for s in eligible_stations(conn, _spec())]
        assert codes == sorted(codes)

    def test_returns_exactly_n_sorted_unique_codes(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 10, PC_LIVERPOOL: 10, PC_SYDNEY: 10})
        picked = sample_station_universe(conn, _spec(n=12, seed=5))
        assert len(picked) == 12
        assert len(set(picked)) == 12
        assert picked == sorted(picked)

    def test_allocation_is_proportional_to_council_size(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 60, PC_LIVERPOOL: 30, PC_SYDNEY: 10})
        picked = set(sample_station_universe(conn, _spec(n=10, seed=5)))
        by_council: dict[str, int] = {}
        for station in eligible_stations(conn, _spec()):
            if station.station_code in picked:
                by_council[station.council] = by_council.get(station.council, 0) + 1
        assert by_council == {"Parramatta": 6, "Liverpool": 3, "Sydney": 1}

    def test_ineligible_stations_are_never_drawn(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 5})
        _seed_station(conn, 99, PC_PARRAMATTA, price_days=1)  # far below min_coverage
        assert 99 not in sample_station_universe(conn, _spec(n=5, seed=1))

    def test_raises_rather_than_silently_returning_a_smaller_universe(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 3})
        with pytest.raises(UniverseTooSmall, match="only 3 pass the gates"):
            sample_station_universe(conn, _spec(n=4, seed=1))

    def test_n_equal_to_the_whole_pool_returns_the_whole_pool(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 4, PC_LIVERPOOL: 3})
        assert sample_station_universe(conn, _spec(n=7, seed=1)) == list(range(1, 8))


# ---------------------------------------------------------------------------
# describe_universe
# ---------------------------------------------------------------------------

class TestDescribeUniverse:
    def test_describes_the_list_it_is_handed_without_re_sampling(self, conn):
        _seed_pool(conn, {PC_PARRAMATTA: 2, PC_LIVERPOOL: 1})
        described = describe_universe(conn, [3, 1], _spec(n=3))
        assert described["station_codes"] == [1, 3]
        assert described["n_stations"] == 2
        assert described["n_councils"] == 2
        assert described["stations_per_council"] == {"Liverpool": 1, "Parramatta": 1}

    def test_reports_gate_failures_without_dropping_them(self, conn):
        """The five preferred stations include one below `min_coverage` (414).

        `describe_universe` must report that rather than silently excluding it —
        dropping a station would rewrite the very population being described.
        """
        _seed_station(conn, 1, PC_PARRAMATTA)
        _seed_station(conn, 2, PC_PARRAMATTA, price_days=10)
        described = describe_universe(conn, [1, 2], _spec(n=1))
        assert described["n_stations"] == 2
        assert described["gate_failures"] == {"2": [GATE_COVERAGE_SPAN]}
        assert described["n_failing_spec_gates"] == 1
        assert described["n_eligible_of_supplied"] == 1
        assert described["coverage_min"] == pytest.approx(10 / SPAN_DAYS)

    def test_gate_failures_name_the_gate_not_just_the_fact(self, conn):
        """A coverage problem must be distinguishable from a station that merely sits
        outside a narrowed `councils` list — the field's whole job is to carry station
        414's COVERAGE caveat, and a bare membership boolean cannot.
        """
        _seed_station(conn, 1, PC_PARRAMATTA)
        _seed_station(conn, 2, PC_LIVERPOOL)                      # wrong council only
        _seed_station(conn, 3, PC_PARRAMATTA, price_days=10)      # coverage only
        _seed_station(conn, 4, PC_PARRAMATTA, classified_days=0)  # unclassified only
        _seed_station(conn, 5, PC_PARRAMATTA, sticky_days=SPAN_DAYS)  # sticky only
        _seed_station(conn, 6, PC_LIVERPOOL, price_days=10)       # two gates at once
        spec = _spec(n=1, councils=("Parramatta",))
        assert gate_failures(conn, spec, [1, 2, 3, 4, 5, 6]) == {
            2: [GATE_COUNCIL],
            3: [GATE_COVERAGE_SPAN],
            4: [GATE_UNCLASSIFIED],
            5: [GATE_STICKY],
            6: [GATE_COUNCIL, GATE_COVERAGE_SPAN],
        }

    def test_stamps_the_gates_not_a_draw_that_never_happened(self, conn):
        """Describing an exogenous population with a spec built for a 50-station draw
        must not record n=50/seed=7 beside n_stations=1 — that block is what a later
        reader diffs.
        """
        _seed_station(conn, 1, PC_PARRAMATTA)
        described = describe_universe(conn, [1], _spec(n=50, seed=7))
        assert "spec" not in described
        assert "n" not in described["spec_gates"]
        assert "seed" not in described["spec_gates"]
        assert described["spec_gates"]["min_coverage"] == DEFAULT_MIN_COVERAGE

    def test_council_counts_are_reconcilable(self, conn):
        """`stations_per_council` buckets a missing council under "unknown" so its
        counts sum to n_stations; `n_councils` counts real councils only. Emitting
        `n_unknown_council` is what lets a reader reconcile the two.
        """
        _seed_station(conn, 1, PC_PARRAMATTA)
        described = describe_universe(conn, [1, 4242], _spec(n=1))
        assert sum(described["stations_per_council"].values()) == described["n_stations"]
        assert described["n_councils"] == 1
        assert described["n_unknown_council"] == 1
        assert (
            described["n_councils"] + described["n_unknown_council"]
            == len(described["stations_per_council"])
        )

    def test_rejects_duplicate_station_codes(self, conn):
        """aggregate_backtest really would replay a repeated station twice, so a
        duplicate double-weights its litres in the pooled CPL. Silently
        de-duplicating would describe a different population from the one about to
        be replayed.
        """
        _seed_station(conn, 1, PC_PARRAMATTA)
        with pytest.raises(ValueError, match=r"duplicates \[1\]"):
            describe_universe(conn, [1, 1], _spec(n=1))

    def test_flags_a_station_absent_from_the_stations_table(self, conn):
        _seed_station(conn, 1, PC_PARRAMATTA)
        described = describe_universe(conn, [1, 4242], _spec(n=1))
        assert described["unknown_station_codes"] == [4242]
        assert described["stations_per_council"]["unknown"] == 1
        assert described["coverage_min"] == 0.0


# ---------------------------------------------------------------------------
# draw_universe — the pure draw
# ---------------------------------------------------------------------------

def _pool(**per_council: int) -> list[EligibleStation]:
    """`n` eligible stations per council, codes assigned 1..N in council order."""
    out, code = [], 1
    for council, count in per_council.items():
        for _ in range(count):
            out.append(
                EligibleStation(
                    station_code=code, council=council, brand="Shell",
                    coverage=1.0, sticky_fraction=0.0,
                )
            )
            code += 1
    return out


class TestDrawUniverse:
    def test_is_independent_of_input_order(self):
        """A shuffled pool must draw the SAME universe as a sorted one.

        This is the test that stands behind `draw_universe`'s canonicalisation:
        remove the `sorted(...)` and it fails. Reversal alone is not enough — a
        symmetric permutation can survive a shuffle by luck — so several distinct
        permutations of the same pool are checked against the sorted baseline.
        """
        pool = _pool(a=9, b=6, c=3)
        expected = draw_universe(pool, n=8, seed=11)
        permutations = [
            pool[::-1],
            pool[9:] + pool[:9],
            sorted(pool, key=lambda s: (s.council, -s.station_code)),
            sorted(pool, key=lambda s: (s.station_code % 5, s.station_code)),
        ]
        for permuted in permutations:
            assert draw_universe(permuted, n=8, seed=11) == expected

    def test_is_deterministic_and_seed_sensitive(self):
        pool = _pool(a=10, b=10)
        assert draw_universe(pool, n=6, seed=1) == draw_universe(pool, n=6, seed=1)
        assert draw_universe(pool, n=6, seed=1) != draw_universe(pool, n=6, seed=2)

    def test_draws_are_proportional_and_sorted(self):
        pool = _pool(a=60, b=30, c=10)
        picked = draw_universe(pool, n=10, seed=4)
        assert picked == sorted(picked)
        by_council = {s.council: 0 for s in pool}
        for station in pool:
            if station.station_code in picked:
                by_council[station.council] += 1
        assert by_council == {"a": 6, "b": 3, "c": 1}

    def test_rejects_a_pool_smaller_than_n(self):
        with pytest.raises(UniverseTooSmall):
            draw_universe(_pool(a=2), n=3, seed=1)

    def test_rejects_duplicate_station_codes(self):
        pool = _pool(a=2)
        with pytest.raises(ValueError, match="duplicate station codes"):
            draw_universe(pool + pool[:1], n=2, seed=1)
