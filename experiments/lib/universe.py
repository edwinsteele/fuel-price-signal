"""Reproducible station-universe sampling for the realised arbiter (fps-nas).

`run_paired_realised_backtest` grades every candidate by replaying a tank walk
over `station_codes`, and every caller so far has left that argument `None` — so
`experiments/lib/realised.py` resolves it to `fuel_signal.config.PREFERRED_STATIONS`,
the owner's five commute stations. `experiments/pipeline/noise_floor.py` threads
it the same way, so the noise band each candidate is graded against is five
stations too. The whole arbiter is therefore ~92 independent decisions per run
(fps-e6i), and no feature in `experiments/results.csv` history has ever delivered
0.5 c/L on its own.

This module supplies the missing half of "grade on a broad Sydney sample, report
at the five": a **declared, reproducible** rule for which stations the broad
sample contains. It only picks station codes — it does not change how anything
is scored, and it has no opinion on which population should grade candidates.
That decision is fps-nas's to record once both CPLs have been measured.

Two properties the sampling rule has to have, because the arbiter's economics are
path-dependent:

*Reproducibility.* `aggregate_backtest` pools spend and litres over whatever list
it is handed, so a universe that is not byte-reproducible from its recorded spec
makes two runs incomparable for a reason unrelated to the feature under test.
Everything here is a pure function of `UniverseSpec` plus the frozen DB.

*Continuity.* `aggregate_backtest` SILENTLY SKIPS a station with no price data in
the window, and `run_backtest` clamps a dry tank rather than failing, so a station
with a long dark span does not announce itself — it just contributes fewer, and
differently-timed, litres. Hence the `min_coverage` gate: a broad universe must not
smuggle in a reweighting of the pool alongside its extra resolution.

*Continuity is not liveness, and the gate only buys the first.* `fuel_signal/fill.py`
forward-fills any gap of `MAX_GAP_FILL_DAYS = 28` or less into `daily_prices` and
leaves longer ones unfilled (trail-fill to `end_date` is capped the same way), so a
presence count sees ONLY dark spans longer than 28 days. Everything shorter is a
stale price the replay trades on as if live — up to 27 consecutive days, since the
cap is on the interval between observations. This is the dominant regime, not an
edge case: measured over batch1's 14-fold span, the forward-filled share of replayed
days has median **0.663** at the five and **0.679** across the 599-station pool, and
the longest filled run maxes at 26 days and 27 days respectively. Two consequences,
and they point opposite ways. The gate is not overclaiming about darkness — it is
simply silent about staleness, which is PRE-EXISTING and unchanged by this module:
the arbiter has always replayed two-thirds forward-filled days. But a wider universe
does shift the tail — 64.8% of the broad pool carries a >= 21-day filled run against
40% of the five, even though the medians differ by only 1.02x. `describe_universe`
therefore reports `observed_fraction_*` for BOTH populations, so a homogeneity read
can see the difference rather than assume it away. Measured, reported, never filtered
on — the same policy `flips.StationPriceSource.is_observed` already sets for the
regret ledger ("used only to count and report the dark days, never to drop a fill").

Deliberate non-goals:

* **Universes at different `n` are not nested.** Largest-remainder allocation is
  not monotone in `n`, so a station in the 50-station sample can be absent from
  the 200-station one. Do not read a difference between two sizes as a pure
  sample-size effect without checking the two lists.
* **No `include`/`exclude` hooks.** The homogeneity question fps-nas asks is
  answered by measuring two SEPARATE populations (the five, and a broad sample)
  and reading each delta against its own noise — never by differencing them
  (`feedback_disjoint_basket_comparison`: two disjoint baskets share no
  denominator). Forcing the five into the broad sample would blur exactly the
  comparison the bead exists to make.
"""
from __future__ import annotations

import hashlib
import math
import random
import sqlite3
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction

from fuel_signal.dates import date_to_int
from fuel_signal.postcode_council import SYDNEY_METRO_COUNCILS

#: Minimum fraction of days in the evaluation span on which a station must have a
#: `daily_prices` row to enter a sampled universe. 0.90 is not tuned — it is the
#: loosest gate that still excludes the one KNOWN long-outage station among the
#: five preferred (station 414, BP Valley Heights, 0.718 across batch1's 14-fold
#: span; see `project_preferred_station_outages`). Read that the other way round
#: and it is the caveat this constant carries: the REFERENCE population is not
#: itself homogeneous on coverage, so a broad sample gated at 0.90 differs from
#: the five on an axis unrelated to any candidate feature. Report the coverage
#: distribution of both populations (`describe_universe`) rather than assuming it
#: away.
DEFAULT_MIN_COVERAGE = 0.90

#: Maximum fraction of a station's CLASSIFIED days inside the span on which it may
#: be labelled Sticky. A Sticky station's price does not track the cycle, so there
#: is no timing for a decision-timing feature to improve there — its rows dilute a
#: delta without being able to carry one. This matches the exclusion the LGA and
#: brand aggregates already apply throughout `fuel_signal/features.py`, and it is
#: cheap: on batch1's frozen DB only 15 of 716 classified stations sit above 0.50.
#: All five preferred stations are at or below 0.017, so the gate does not open a
#: gap between the two populations.
DEFAULT_MAX_STICKY_FRACTION = 0.50

#: The one fuel this module can speak about, and NOT a parameter. Two reasons, and
#: the second is the load-bearing one. (a) The arbiter only ever replays E10:
#: `load_history` pins `db.fuel_type_id(conn, "E10")` and `db.get_daily_prices`
#: defaults to it, so a universe gated on another fuel would gate on prices no
#: backtest reads. (b) `station_class` carries NO fuel dimension — it records
#: whatever fuel `classify_range` last ran with (E10 by default) — so a per-spec
#: fuel would have made the coverage gate and the Sticky gate silently describe
#: different fuels for the same station. A knob that cannot be honoured by half the
#: query it configures is worse than no knob (fps-nas, PR #361 review).
ARBITER_FUEL_CODE = "E10"


class UniverseTooSmall(ValueError):
    """Raised when fewer than `n` stations pass the eligibility gates.

    Deliberately an error rather than a silent "return what there is": the point
    of a wider universe is a stated resolution, and quietly grading on 43 stations
    when the spec said 200 misstates it in the direction that flatters the result.
    """


@dataclass(frozen=True)
class EligibleStation:
    """One station that passed every gate, with the measurements that passed it."""

    station_code: int
    council: str
    brand: str | None
    coverage: float
    sticky_fraction: float
    #: Lowest per-window coverage across `spec.windows`, or None when the spec gates
    #: on the span only. The number the span figure hides: a station can sit at 0.900
    #: over 1150 days and still be entirely dark for one 90-day val window.
    worst_window_coverage: float | None = None


@dataclass(frozen=True)
class UniverseSpec:
    """Every KNOB that shapes a sampled universe. Stamp this into a run's meta.

    It is deliberately NOT "everything that determines the universe" (an earlier
    draft of this docstring claimed that and was wrong — PR #361 review finding 2).
    The universe is a function of this spec AND the vintage of `daily_prices`, which
    is a rebuilt derived table: `fuel_signal/fill.py::fill_all` DELETEs and rebuilds
    it, forward-filling gaps up to `max_gap_days` and trailing-filling to `end_date`.
    So the eligible pool over a FIXED historical span moves when late raw prices make
    an old gap fillable, when `--max-gap-days` changes, or simply when `fill` runs on
    a later day. Two runs can stamp byte-identical specs and have drawn different
    universes. Two mitigations, and you want both:

    * Sample against a FROZEN batch DB (`experiments/pipeline/batch_freeze.py` copies
      `fuel_signal.db` into the batch dir and stamps `source_db` / `snapshot_date` /
      `frozen_at` in `freeze.json`). This is what an fps-nas run does, and it pins the
      vintage properly.
    * Stamp `eligible_pool_digest(conn, spec)` beside the spec. It hashes the pool the
      gates actually admitted, so a moved DB shows up as a changed digest even if the
      spec is identical.

    `seed` is required and has no default on purpose: it must be a recorded choice,
    and it must not silently collide with the FIT seed a run also carries (nothing
    breaks if they are equal, but a reader comparing two runs should never have to
    wonder whether "42" meant the sampler or the model).

    `councils=None` means `SYDNEY_METRO_COUNCILS` — the "broad Sydney sample" the
    bead asks for. Pass an explicit tuple to sample within a narrower geography.

    `windows` is the list of (start, end) validation windows the universe will
    actually be replayed through — for an fps-nas run, the val window of every outer
    fold. **Pass it.** The coverage gate's whole justification is per-window
    (`aggregate_backtest` silently skips a station whose window CPL is NaN, and
    `run_backtest` clamps a dry tank), but a span-level threshold cannot see a dark
    run shorter than the span: over a 1250-day span a station can be dark for 125
    consecutive days — longer than a whole 90-day val window — and still sit at
    exactly 0.900.

    That is not a theoretical gap. Measured on batch1's real 14-fold geometry
    (2021-11-05 .. 2025-04-17), requiring `min_coverage` in EVERY window instead of
    across the span drops **189 of 599** stations — a third of the pool the span gate
    admits has at least one badly-covered fold. The sharpest case is the reference
    population itself: station 414's `worst_window_coverage` is **0.0**, i.e. it is
    entirely dark for a whole val window, which is precisely the silent-skip the
    module docstring describes. 410 stations still survive, ~82x the incumbent five,
    so the resolution argument is untouched by the stricter gate.

    Leaving `windows` as `None` keeps span-only behaviour and is only appropriate
    when no replay is involved — nothing in a span-gated result can rule out a dark
    window (PR #361 review finding 1).
    """

    n: int
    seed: int
    start_date: str
    end_date: str
    min_coverage: float = DEFAULT_MIN_COVERAGE
    max_sticky_fraction: float = DEFAULT_MAX_STICKY_FRACTION
    councils: tuple[str, ...] | None = None
    windows: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"UniverseSpec.n must be >= 1; got {self.n}.")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError(
                f"UniverseSpec.min_coverage must be in (0, 1]; got {self.min_coverage}."
            )
        if not 0.0 <= self.max_sticky_fraction <= 1.0:
            raise ValueError(
                "UniverseSpec.max_sticky_fraction must be in [0, 1]; got "
                f"{self.max_sticky_fraction}."
            )
        # date.fromisoformat also rejects a malformed string — a silently-misparsed
        # bound would move the coverage denominator without moving the numerator.
        start, end = date.fromisoformat(self.start_date), date.fromisoformat(self.end_date)
        if start > end:
            raise ValueError(
                f"UniverseSpec start_date {self.start_date} is after end_date {self.end_date}."
            )
        if self.councils is not None and not self.councils:
            raise ValueError(
                "UniverseSpec.councils=() selects nothing; pass None for all Sydney metro "
                "councils, or a non-empty tuple."
            )
        if self.windows is not None:
            if not self.windows:
                raise ValueError(
                    "UniverseSpec.windows=() gates on nothing; pass None for a span-only "
                    "gate, or a non-empty tuple of (start, end) pairs."
                )
            for w_start, w_end in self.windows:
                ws, we = date.fromisoformat(w_start), date.fromisoformat(w_end)
                if ws > we:
                    raise ValueError(f"UniverseSpec window {w_start}..{w_end} starts after it ends.")
                if ws < start or we > end:
                    # A window outside the span would be gated against dates the span
                    # query never looked at, so the two gates would disagree silently.
                    raise ValueError(
                        f"UniverseSpec window {w_start}..{w_end} falls outside the span "
                        f"{self.start_date}..{self.end_date}."
                    )

    @property
    def resolved_councils(self) -> tuple[str, ...]:
        """The council allow-list actually applied, sorted (never `None`)."""
        return tuple(sorted(self.councils if self.councils is not None else SYDNEY_METRO_COUNCILS))

    @property
    def span_days(self) -> int:
        """Inclusive calendar length of the span — the coverage denominator."""
        return (date.fromisoformat(self.end_date) - date.fromisoformat(self.start_date)).days + 1

    def gates_dict(self) -> dict:
        """JSON-safe stamp of the ELIGIBILITY half only — no `n`, no `seed`.

        `describe_universe` stamps this rather than `as_dict()`: it characterises a
        list it was handed, which may be an exogenous population (the five preferred
        stations) that no draw produced. Stamping `n`/`seed` there would record a draw
        that never happened, in exactly the block a later reader diffs (PR #361 review
        finding 4).
        """
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "span_days": self.span_days,
            "min_coverage": self.min_coverage,
            "min_days": _min_days(self.min_coverage, self.span_days),
            "max_sticky_fraction": self.max_sticky_fraction,
            "councils": list(self.resolved_councils),
            "fuel": ARBITER_FUEL_CODE,
            "windows": [list(w) for w in self.windows] if self.windows else None,
            "coverage_gated_per_window": self.windows is not None,
        }

    def as_dict(self) -> dict:
        """JSON-safe stamp of the whole spec. `councils` is RESOLVED, not `None`."""
        return {
            **self.gates_dict(),
            "n": self.n,
            "seed": self.seed,
            "stratify_by": "council",
            # Stated as the PROPERTY, not as a mechanism: `allocate_by_stratum` has no
            # cap step and argues one would be unreachable, so "capped at stratum size"
            # (the earlier wording) described code that does not exist.
            "allocation": "proportional (largest remainder); never exceeds a stratum's size",
        }


def _min_days(min_coverage: float, span_days: int) -> int:
    """Smallest priced-day count that satisfies `coverage >= min_coverage`, exactly.

    Via `Fraction(str(...))`, not float arithmetic. `min_coverage * span_days` is a
    binary-float product and can land a hair ABOVE the exact decimal, which silently
    turns the documented inclusive boundary into an exclusive one: `0.56 * 25` is
    14.000000000000002, so a station with exactly 0.56 coverage over a 25-day span was
    REJECTED. Affects 0.54, 0.55, 0.56, 0.67, 0.68 and 0.81 among two-decimal values;
    the 0.90 default happens to be exact, which is why the boundary test did not catch
    it (PR #361 review finding 5). `str()` on the float gives the shortest round-tripping
    decimal, so a literal like 0.56 parses back to exactly 56/100.
    """
    return math.ceil(Fraction(str(min_coverage)) * span_days)


def _placeholders(values: Sequence) -> tuple[str, list]:
    """('?,?,…', list) for a SQL IN clause, in the CALLER'S order.

    Not `fuel_signal.db._in_clause`, which sorts its values — deliberately named
    differently so the two can't be mistaken for each other.

    An earlier version of this docstring justified the duplication by claiming "order
    matters here: `describe_universe` reports on the list it was handed". That was
    false, and is the exact pattern this repo has scar tissue for — a docstring
    certifying behaviour the code does not have (PR #361 review finding 8).
    `describe_universe` sorts its codes BEFORE calling this, every call site feeds a
    dict comprehension keyed by station_code or an already-sorted council list, and
    nothing downstream reads the bound-parameter order. `db._in_clause` would return
    identical results at all of them. What this helper actually buys is not depending
    on another module's private for a four-line builder, and somewhere to put the
    note below; the order it happens to preserve is incidental, and no caller may
    start relying on it without saying so here.

    The returned string is `?` characters and commas ONLY — it is derived from
    `len(values)`, never from the values themselves — so f-string-ing it into a
    query interpolates no data. Every value travels as a bound parameter, the same
    shape `fuel_signal/db.py` and `fuel_signal/backtest.py` already use for IN
    clauses. Static scanners flag the f-string regardless; that is a false positive
    here, and this note is the answer to it.
    """
    return ", ".join(["?"] * len(values)), list(values)


#: Reasons a station can fail eligibility, in the order the gates are applied.
GATE_COUNCIL = "council"
GATE_COVERAGE_SPAN = "coverage_span"
GATE_COVERAGE_WINDOW = "coverage_window"
GATE_UNCLASSIFIED = "unclassified"
GATE_STICKY = "sticky"


def _window_coverage(
    conn: sqlite3.Connection, spec: UniverseSpec, codes: Sequence[int]
) -> dict[int, list[float]]:
    """Per-station coverage in each of `spec.windows`, in window order.

    One query per window rather than one clever grouped query: 14 windows is nothing,
    and a station absent from a window's result set has coverage 0 there, which a
    GROUP BY over a union would silently omit instead of reporting.
    """
    if not spec.windows or not codes:
        return {}
    code_ph, code_vals = _placeholders(list(codes))
    out: dict[int, list[float]] = {int(c): [] for c in codes}
    for w_start, w_end in spec.windows:
        w_days = (date.fromisoformat(w_end) - date.fromisoformat(w_start)).days + 1
        counts = dict(
            conn.execute(
                f"""SELECT dp.station_code, COUNT(DISTINCT dp.price_date)
                      FROM daily_prices dp
                      JOIN fuel_types f ON f.id = dp.fuel_type_id
                     WHERE f.code = ?
                       AND dp.price_date BETWEEN ? AND ?
                       AND dp.station_code IN ({code_ph})
                     GROUP BY dp.station_code""",
                [ARBITER_FUEL_CODE, date_to_int(w_start), date_to_int(w_end), *code_vals],
            )
        )
        for code in out:
            out[code].append(counts.get(code, 0) / w_days)
    return out


def gate_failures(
    conn: sqlite3.Connection, spec: UniverseSpec, station_codes: Sequence[int]
) -> dict[int, list[str]]:
    """Which of `spec`'s gates each supplied station fails, by name. {} entries omitted.

    Named reasons, not a set-difference against the eligible pool: the field exists to
    carry a specific caveat (station 414's 0.718 coverage) into a run's meta.json, and
    a bare "failed the gates" cannot be told apart from a station that merely sits
    outside a narrowed `councils` list (PR #361 review finding 3).
    """
    codes = [int(c) for c in station_codes]
    if not codes:
        return {}
    code_ph, code_vals = _placeholders(codes)
    start_int, end_int = date_to_int(spec.start_date), date_to_int(spec.end_date)
    min_days = _min_days(spec.min_coverage, spec.span_days)
    allowed = set(spec.resolved_councils)

    councils = {
        int(code): council
        for code, council in conn.execute(
            f"SELECT station_code, council FROM stations WHERE station_code IN ({code_ph})",
            code_vals,
        )
    }
    priced_days = dict(
        conn.execute(
            f"""SELECT dp.station_code, COUNT(DISTINCT dp.price_date)
                  FROM daily_prices dp
                  JOIN fuel_types f ON f.id = dp.fuel_type_id
                 WHERE f.code = ? AND dp.price_date BETWEEN ? AND ?
                   AND dp.station_code IN ({code_ph})
                 GROUP BY dp.station_code""",
            [ARBITER_FUEL_CODE, start_int, end_int, *code_vals],
        )
    )
    sticky = dict(
        conn.execute(
            f"""SELECT station_code, AVG(CASE WHEN class = 'Sticky' THEN 1.0 ELSE 0.0 END)
                  FROM station_class
                 WHERE snapshot_date BETWEEN ? AND ? AND station_code IN ({code_ph})
                 GROUP BY station_code""",
            [start_int, end_int, *code_vals],
        )
    )
    per_window = _window_coverage(conn, spec, codes)

    failures: dict[int, list[str]] = {}
    for code in codes:
        reasons: list[str] = []
        if councils.get(code) not in allowed:
            reasons.append(GATE_COUNCIL)
        if priced_days.get(code, 0) < min_days:
            reasons.append(GATE_COVERAGE_SPAN)
        if spec.windows and any(c < spec.min_coverage for c in per_window.get(code, [])):
            reasons.append(GATE_COVERAGE_WINDOW)
        if code not in sticky:
            reasons.append(GATE_UNCLASSIFIED)
        elif sticky[code] > spec.max_sticky_fraction:
            reasons.append(GATE_STICKY)
        if reasons:
            failures[code] = reasons
    return failures


def eligible_stations(
    conn: sqlite3.Connection,
    spec: UniverseSpec,
    *,
    restrict_to: Sequence[int] | None = None,
) -> list[EligibleStation]:
    """Every station passing `spec`'s gates, sorted by station_code.

    Gates, in the order a rejected station fails them:

    1. `stations.council` is non-NULL and in `spec.resolved_councils`.
    2. Distinct `daily_prices` dates for `ARBITER_FUEL_CODE` inside the span >=
       `_min_days(min_coverage, span_days)` — an exact integer, not a float product.
    3. When `spec.windows` is set, that same `min_coverage` in EVERY window. This is
       the gate that matches the stated correctness property; see `UniverseSpec` on
       why the span gate alone does not (it admits stations dark for longer than a
       whole val window).
    4. At least one `station_class` row inside the span, and the Sticky share of
       those rows <= `max_sticky_fraction`.

    Gate 4's "at least one row" half is a real exclusion, not a formality: a station
    the classifier never saw cannot be shown to be non-Sticky, and admitting it on
    the grounds that no evidence exists would put exactly the unmeasurable stations
    into a population whose whole purpose is to be measurable. On batch1's frozen DB
    this costs ~35 of 751 stations.

    `restrict_to` narrows the scan to those station codes. Pure optimisation — the
    result is exactly the eligible subset of that list — for callers describing a
    known population instead of building a pool from the whole network.
    """
    council_ph, council_vals = _placeholders(spec.resolved_councils)
    start_int, end_int = date_to_int(spec.start_date), date_to_int(spec.end_date)
    min_days = _min_days(spec.min_coverage, spec.span_days)

    restrict_clause, restrict_vals = "", []
    if restrict_to is not None:
        if not restrict_to:
            return []
        restrict_ph, restrict_vals = _placeholders([int(c) for c in restrict_to])
        restrict_clause = f" AND dp.station_code IN ({restrict_ph})"

    coverage_rows = conn.execute(
        f"""SELECT s.station_code, s.council, s.brand, COUNT(DISTINCT dp.price_date)
              FROM daily_prices dp
              JOIN stations s ON s.station_code = dp.station_code
              JOIN fuel_types f ON f.id = dp.fuel_type_id
             WHERE f.code = ?
               AND dp.price_date BETWEEN ? AND ?
               AND s.council IN ({council_ph}){restrict_clause}
             GROUP BY s.station_code
            HAVING COUNT(DISTINCT dp.price_date) >= ?""",
        [ARBITER_FUEL_CODE, start_int, end_int, *council_vals, *restrict_vals, min_days],
    ).fetchall()
    if not coverage_rows:
        return []

    codes = [int(r[0]) for r in coverage_rows]
    code_ph, code_vals = _placeholders(codes)
    sticky_by_code: dict[int, float] = {
        int(code): sticky
        for code, sticky in conn.execute(
            f"""SELECT station_code,
                       AVG(CASE WHEN class = 'Sticky' THEN 1.0 ELSE 0.0 END)
                  FROM station_class
                 WHERE snapshot_date BETWEEN ? AND ?
                   AND station_code IN ({code_ph})
                 GROUP BY station_code""",
            [start_int, end_int, *code_vals],
        )
    }
    per_window = _window_coverage(conn, spec, codes)

    out: list[EligibleStation] = []
    for code, council, brand, n_days in coverage_rows:
        code = int(code)
        sticky = sticky_by_code.get(code)
        if sticky is None or sticky > spec.max_sticky_fraction:
            continue
        windows = per_window.get(code, [])
        if spec.windows and any(c < spec.min_coverage for c in windows):
            continue
        out.append(
            EligibleStation(
                station_code=code,
                council=council,
                brand=brand,
                coverage=n_days / spec.span_days,
                sticky_fraction=float(sticky),
                worst_window_coverage=min(windows) if windows else None,
            )
        )
    # Sorted by station_code as a documented postcondition. `draw_universe` does
    # NOT rely on it (it re-sorts its own input) — this is here so a caller reading
    # or persisting the eligible pool gets a stable, diffable order.
    out.sort(key=lambda s: s.station_code)
    return out


def eligible_pool_digest(conn: sqlite3.Connection, spec: UniverseSpec) -> str:
    """A short hash of the pool `spec` admits — the DB-vintage half of a run's stamp.

    `UniverseSpec` records the knobs; this records what those knobs actually selected
    out of THIS database. `daily_prices` is rebuilt by `fill.py`, so a spec alone does
    not pin a universe (see `UniverseSpec`'s docstring). Stamp both, and two runs that
    drew different pools cannot look identical in meta.json.
    """
    pool = eligible_stations(conn, spec)
    payload = ";".join(f"{s.station_code}:{s.coverage:.6f}" for s in pool)
    return f"{len(pool)}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def allocate_by_stratum(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Split `n` across strata proportionally to `sizes` (largest remainder).

    Returns a dict over every key of `sizes` (zeros included) summing to exactly
    `n`, with no stratum allocated more than it holds.

    Ties on the fractional remainder break on stratum NAME, ascending — arbitrary,
    but arbitrary-and-fixed is the requirement: a tie broken by dict order would
    make the whole universe depend on the order rows came back from SQLite.

    No cap-and-redistribute pass is needed, and one written "defensively" would be
    unreachable: with ``n <= total``, every quota satisfies ``n*c_k/total <= c_k``,
    so ``floor(quota_k) <= c_k``; the largest-remainder pass can push a stratum to
    ``c_k + 1`` only when ``floor(quota_k) == c_k``, which needs ``n >= total``, and
    at ``n == total`` the floors already sum to ``n`` so no remainder is handed out
    at all. The final ``max(alloc) <= sizes`` check below is therefore a live
    assertion about that argument, not a fallback that silently repairs a bad
    allocation — if it ever fires, the reasoning is wrong and the caller should see
    it rather than receive a quietly reshaped universe. `TestAllocateByStratum
    .test_never_allocates_more_than_a_stratum_holds` sweeps the argument's whole
    domain.
    """
    if n < 0:
        raise ValueError(f"allocate_by_stratum(): n must be >= 0; got {n}.")
    negative = {k: v for k, v in sizes.items() if v < 0}
    if negative:
        # A negative size is not a stratum that "holds nothing" — it silently
        # shrinks `total`, inflating every other stratum's quota past what it
        # actually holds, which is exactly the over-allocation the docstring
        # argues cannot happen. Reject it rather than let it falsify the proof.
        raise ValueError(
            f"allocate_by_stratum(): stratum sizes must be >= 0; got {negative!r}."
        )
    total = sum(sizes.values())
    if n > total:
        raise ValueError(
            f"allocate_by_stratum(): asked for {n} from strata holding {total}."
        )
    if total == 0:
        return {k: 0 for k in sizes}

    quotas = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remainder = n - sum(alloc.values())
    by_remainder = sorted(sizes, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in by_remainder[:remainder]:
        alloc[k] += 1

    over = {k: alloc[k] for k in sizes if alloc[k] > sizes[k]}
    if over:  # pragma: no cover — unreachable; see the docstring's argument.
        raise AssertionError(
            f"allocate_by_stratum({sizes!r}, {n}) over-allocated {over!r}; the "
            "largest-remainder bound in this function's docstring is wrong."
        )
    return alloc


def sample_station_universe(conn: sqlite3.Connection, spec: UniverseSpec) -> list[int]:
    """Return `spec.n` station codes, sorted ascending.

    Stratified by **council**, proportional to each council's eligible-station count.
    Council is the stratification axis, rather than brand or price level, because:

    * The cycle propagates geographically — `lga_leadership` / `lga_phase_std` /
      `days_since_trough_entry_<lga>` are a whole feature family built on LGAs
      leading and lagging each other, so LGA is the axis along which two stations
      are most likely to make DIFFERENT decisions on the same day. Balancing it is
      what buys independent decisions rather than more copies of the same one.
    * It is the axis on which the reference population is most obviously narrow:
      the five preferred stations sit in just two councils (Blue Mountains,
      Penrith), both on the network's lagging edge (`project_lga_leadership`).
      An LGA-balanced sample is the population the homogeneity question is about.
    * Brand is largely nested inside geography and already enters the model as a
      feature; price LEVEL is downstream of both, and stratifying on something so
      close to the measured outcome (CPL) risks conditioning the estimate rather
      than describing the population.

    Raises `UniverseTooSmall` when the gates admit fewer than `spec.n` stations.
    """
    eligible = eligible_stations(conn, spec)
    if len(eligible) < spec.n:
        raise UniverseTooSmall(
            f"{spec.n} stations requested but only {len(eligible)} pass the gates "
            f"(coverage >= {spec.min_coverage}, sticky <= {spec.max_sticky_fraction}, "
            f"{len(spec.resolved_councils)} councils, {spec.start_date}..{spec.end_date})."
        )
    return draw_universe(eligible, n=spec.n, seed=spec.seed)


def draw_universe(
    eligible: Sequence[EligibleStation], *, n: int, seed: int
) -> list[int]:
    """The pure half of `sample_station_universe`: stratify, allocate, draw.

    Separate from the DB query so the draw's determinism is testable without a
    fixture, and so the ordering guarantee below is a property of THIS function
    rather than of whatever plan SQLite happened to pick for the query upstream.

    `eligible` is canonicalised (sorted by station_code) before the shuffle, so
    the result is a function of the SET of eligible stations and nothing else —
    callers do not have to supply an order, and an unordered one cannot leak in.
    Raises `UniverseTooSmall` if `eligible` holds fewer than `n` stations.
    """
    pool = sorted(eligible, key=lambda s: s.station_code)
    if len(pool) < n:
        raise UniverseTooSmall(
            f"{n} stations requested but only {len(pool)} pass the gates."
        )
    if len({s.station_code for s in pool}) != len(pool):
        # A duplicated code would be drawn twice and quietly shrink the universe.
        raise ValueError("draw_universe(): eligible contains duplicate station codes.")

    sizes: dict[str, int] = {}
    for station in pool:
        sizes[station.council] = sizes.get(station.council, 0) + 1
    quota = allocate_by_stratum(sizes, n)

    # One RNG over the whole canonical pool, then fill each council's quota in that
    # single shuffled order. Deliberately NOT a per-council RNG seeded from the
    # council name: seeding on `hash(str)` varies with PYTHONHASHSEED between
    # processes, which would make the "reproducible" universe reproducible only
    # within one interpreter.
    random.Random(seed).shuffle(pool)

    remaining = dict(quota)
    picked: list[int] = []
    for station in pool:
        if remaining.get(station.council, 0) > 0:
            picked.append(station.station_code)
            remaining[station.council] -= 1
    return sorted(picked)


def describe_universe(
    conn: sqlite3.Connection,
    station_codes: Sequence[int],
    spec: UniverseSpec,
) -> dict:
    """A JSON-safe description of an ACTUAL station list, for a run's meta.json.

    Takes the list rather than re-sampling, so it can describe the five preferred
    stations on exactly the same axes as a sampled universe — which is the whole
    point: the homogeneity read in fps-nas is only interpretable if both populations
    are characterised the same way. Stations absent from `stations` are reported under
    `unknown_station_codes` rather than dropped silently.

    Stamps `spec.gates_dict()`, not `spec.as_dict()`: this function characterises a
    supplied list, which may be a population no draw produced, so recording `n`/`seed`
    here would stamp a draw that never happened. The caller that actually drew a
    universe stamps the full spec itself.

    The gates are REPORTED here, never enforced — the five preferred stations are an
    exogenous population and station 414 fails `min_coverage`. Dropping it would
    rewrite the very population being reported on. Failures are named per station
    (`gate_failures`) so a coverage problem cannot be confused with a station that
    merely sits outside a narrowed `councils` list.

    Both council fields count the same way and can disagree only in what they are
    counting: `stations_per_council` buckets a missing council under the literal key
    `"unknown"` so the counts always sum to `n_stations`, while `n_councils` counts
    real councils and excludes it. `n_unknown_council` is emitted so the two can be
    reconciled without inferring the rule.
    """
    codes = [int(c) for c in station_codes]
    duplicates = sorted({c for c in codes if codes.count(c) > 1})
    if duplicates:
        # Silently de-duplicating would make `n_stations` and every per-council
        # count describe a DIFFERENT population from the one the caller is about
        # to replay — and `aggregate_backtest` really would replay a repeated
        # station twice, double-weighting its litres in the pooled CPL.
        raise ValueError(
            f"describe_universe(): station_codes contains duplicates {duplicates}; "
            "a repeated station is double-weighted by aggregate_backtest, so this "
            "is a caller bug, not something to normalise away."
        )
    codes = sorted(codes)
    # restrict_to keeps this to the supplied stations instead of rebuilding the whole
    # network's pool for two summary fields (PR #361 review finding 6).
    eligible = {s.station_code: s for s in eligible_stations(conn, spec, restrict_to=codes)}
    failures = gate_failures(conn, spec, codes)

    code_ph, code_vals = _placeholders(codes)
    start_int, end_int = date_to_int(spec.start_date), date_to_int(spec.end_date)
    meta = {
        int(code): (council, brand)
        for code, council, brand in conn.execute(
            f"SELECT station_code, council, brand FROM stations WHERE station_code IN ({code_ph})",
            code_vals,
        )
    }
    coverage = {
        int(code): n / spec.span_days
        for code, n in conn.execute(
            f"""SELECT dp.station_code, COUNT(DISTINCT dp.price_date)
                  FROM daily_prices dp
                  JOIN fuel_types f ON f.id = dp.fuel_type_id
                 WHERE f.code = ? AND dp.price_date BETWEEN ? AND ?
                   AND dp.station_code IN ({code_ph})
                 GROUP BY dp.station_code""",
            [ARBITER_FUEL_CODE, start_int, end_int, *code_vals],
        )
    }
    # `prices` is the RAW observation table; `daily_prices` is it plus fill.py's
    # forward-filled gaps. The ratio is the share of replayed days on which the
    # station actually reported — the staleness axis `coverage` cannot see. Counted
    # and reported, never filtered on (see the module docstring).
    observed = {
        int(code): n
        for code, n in conn.execute(
            f"""SELECT p.station_code, COUNT(DISTINCT p.price_date)
                  FROM prices p
                  JOIN fuel_types f ON f.id = p.fuel_type_id
                 WHERE f.code = ? AND p.price_date BETWEEN ? AND ?
                   AND p.station_code IN ({code_ph})
                 GROUP BY p.station_code""",
            [ARBITER_FUEL_CODE, start_int, end_int, *code_vals],
        )
    }
    per_window = _window_coverage(conn, spec, codes)

    councils: dict[str, int] = {}
    brands: dict[str, int] = {}
    for code in codes:
        council, brand = meta.get(code, (None, None))
        councils[council or "unknown"] = councils.get(council or "unknown", 0) + 1
        brands[brand or "unknown"] = brands.get(brand or "unknown", 0) + 1

    coverages = [coverage.get(code, 0.0) for code in codes]
    worst_window = [min(per_window[c]) for c in codes if per_window.get(c)]
    # Denominator is the station's own replayed days, not the span: a station that is
    # dark for part of the span is not thereby "stale" on the days it does not appear.
    observed_fractions = [
        observed.get(code, 0) / days
        for code in codes
        if (days := round(coverage.get(code, 0.0) * spec.span_days))
    ]
    return {
        "spec_gates": spec.gates_dict(),
        "n_stations": len(codes),
        "station_codes": codes,
        "unknown_station_codes": [c for c in codes if c not in meta],
        "n_councils": len({meta.get(c, (None, None))[0] for c in codes} - {None}),
        "n_unknown_council": sum(1 for c in codes if meta.get(c, (None, None))[0] is None),
        "stations_per_council": dict(sorted(councils.items())),
        "stations_per_brand": dict(sorted(brands.items())),
        "coverage_min": min(coverages) if coverages else None,
        "coverage_mean": sum(coverages) / len(coverages) if coverages else None,
        # The span figure's blind spot, made visible: None when the spec gates on the
        # span only, in which case nothing here can rule out a dark val window.
        "worst_window_coverage": min(worst_window) if worst_window else None,
        # Staleness, the axis `coverage` is blind to. ~2/3 of replayed days are
        # forward-filled for EVERY population, so a low number here is normal and is
        # not a defect in the station — it is there so a homogeneity read can compare
        # two populations on it instead of assuming they match.
        "observed_fraction_min": min(observed_fractions) if observed_fractions else None,
        "observed_fraction_median": (
            statistics.median(observed_fractions) if observed_fractions else None
        ),
        "n_eligible_of_supplied": len(eligible),
        "n_failing_spec_gates": len(failures),
        "gate_failures": {str(c): failures[c] for c in sorted(failures)},
    }
