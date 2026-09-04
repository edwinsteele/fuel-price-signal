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

import random
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

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


@dataclass(frozen=True)
class UniverseSpec:
    """Everything that determines a sampled universe. Stamp this into a run's meta.

    `seed` is required and has no default on purpose: it must be a recorded choice,
    and it must not silently collide with the FIT seed a run also carries (nothing
    breaks if they are equal, but a reader comparing two runs should never have to
    wonder whether "42" meant the sampler or the model).

    `councils=None` means `SYDNEY_METRO_COUNCILS` — the "broad Sydney sample" the
    bead asks for. Pass an explicit tuple to sample within a narrower geography.
    """

    n: int
    seed: int
    start_date: str
    end_date: str
    min_coverage: float = DEFAULT_MIN_COVERAGE
    max_sticky_fraction: float = DEFAULT_MAX_STICKY_FRACTION
    councils: tuple[str, ...] | None = None
    fuel: str = "E10"

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

    @property
    def resolved_councils(self) -> tuple[str, ...]:
        """The council allow-list actually applied, sorted (never `None`)."""
        return tuple(sorted(self.councils if self.councils is not None else SYDNEY_METRO_COUNCILS))

    @property
    def span_days(self) -> int:
        """Inclusive calendar length of the span — the coverage denominator."""
        return (date.fromisoformat(self.end_date) - date.fromisoformat(self.start_date)).days + 1

    def as_dict(self) -> dict:
        """JSON-safe stamp. `councils` is the RESOLVED list, not the `None` sentinel."""
        return {
            "n": self.n,
            "seed": self.seed,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "span_days": self.span_days,
            "min_coverage": self.min_coverage,
            "max_sticky_fraction": self.max_sticky_fraction,
            "councils": list(self.resolved_councils),
            "fuel": self.fuel,
            "stratify_by": "council",
            "allocation": "proportional (largest remainder), capped at stratum size",
        }


def _placeholders(values: Sequence) -> tuple[str, list]:
    """('?,?,…', list) for a SQL IN clause, in the CALLER'S order.

    Not `fuel_signal.db._in_clause`, which sorts its values — deliberately named
    differently so the two can't be mistaken for each other. Order matters here:
    `describe_universe` reports on the list it was handed.
    """
    return ", ".join(["?"] * len(values)), list(values)


def eligible_stations(conn: sqlite3.Connection, spec: UniverseSpec) -> list[EligibleStation]:
    """Every station passing `spec`'s gates, sorted by station_code.

    Gates, in the order a rejected station fails them:

    1. `stations.council` is non-NULL and in `spec.resolved_councils`.
    2. Distinct `daily_prices` dates for `spec.fuel` inside the span >=
       `min_coverage * span_days`.
    3. At least one `station_class` row inside the span, and the Sticky share of
       those rows <= `max_sticky_fraction`.

    Gate 3's "at least one row" half is a real exclusion, not a formality: a station
    the classifier never saw cannot be shown to be non-Sticky, and admitting it on
    the grounds that no evidence exists would put exactly the unmeasurable stations
    into a population whose whole purpose is to be measurable. On batch1's frozen DB
    this costs ~35 of 751 stations.
    """
    council_ph, council_vals = _placeholders(spec.resolved_councils)
    start_int, end_int = date_to_int(spec.start_date), date_to_int(spec.end_date)
    min_days = spec.min_coverage * spec.span_days

    coverage_rows = conn.execute(
        f"""SELECT s.station_code, s.council, s.brand, COUNT(DISTINCT dp.price_date)
              FROM daily_prices dp
              JOIN stations s ON s.station_code = dp.station_code
              JOIN fuel_types f ON f.id = dp.fuel_type_id
             WHERE f.code = ?
               AND dp.price_date BETWEEN ? AND ?
               AND s.council IN ({council_ph})
             GROUP BY s.station_code
            HAVING COUNT(DISTINCT dp.price_date) >= ?""",
        [spec.fuel, start_int, end_int, *council_vals, min_days],
    ).fetchall()
    if not coverage_rows:
        return []

    code_ph, code_vals = _placeholders([r[0] for r in coverage_rows])
    sticky_by_code: dict[int, float] = {
        code: sticky
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

    out: list[EligibleStation] = []
    for code, council, brand, n_days in coverage_rows:
        sticky = sticky_by_code.get(code)
        if sticky is None or sticky > spec.max_sticky_fraction:
            continue
        out.append(
            EligibleStation(
                station_code=int(code),
                council=council,
                brand=brand,
                coverage=n_days / spec.span_days,
                sticky_fraction=float(sticky),
            )
        )
    # Sorted by station_code as a documented postcondition. `draw_universe` does
    # NOT rely on it (it re-sorts its own input) — this is here so a caller reading
    # or persisting the eligible pool gets a stable, diffable order.
    out.sort(key=lambda s: s.station_code)
    return out


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
    point: the homogeneity read in fps-nas is only interpretable if both
    populations are characterised the same way. Stations absent from `stations`
    are reported under `unknown_station_codes` rather than dropped silently.
    """
    codes = sorted(set(int(c) for c in station_codes))
    by_code = {s.station_code: s for s in eligible_stations(conn, spec)}

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
            [spec.fuel, start_int, end_int, *code_vals],
        )
    }

    councils: dict[str, int] = {}
    brands: dict[str, int] = {}
    for code in codes:
        council, brand = meta.get(code, (None, None))
        councils[council or "unknown"] = councils.get(council or "unknown", 0) + 1
        brands[brand or "unknown"] = brands.get(brand or "unknown", 0) + 1

    coverages = [coverage.get(code, 0.0) for code in codes]
    return {
        "spec": spec.as_dict(),
        "n_stations": len(codes),
        "station_codes": codes,
        "unknown_station_codes": [c for c in codes if c not in meta],
        "n_councils": len(set(meta.get(c, (None, None))[0] for c in codes) - {None}),
        "stations_per_council": dict(sorted(councils.items())),
        "stations_per_brand": dict(sorted(brands.items())),
        "coverage_min": min(coverages) if coverages else None,
        "coverage_mean": sum(coverages) / len(coverages) if coverages else None,
        # The gates are reported, never enforced, here — the five preferred stations
        # are an exogenous population and one of them (414) fails `min_coverage`.
        # Silently dropping it would rewrite the very population being reported on.
        "n_failing_spec_gates": sum(1 for c in codes if c not in by_code),
        "station_codes_failing_spec_gates": [c for c in codes if c not in by_code],
    }
