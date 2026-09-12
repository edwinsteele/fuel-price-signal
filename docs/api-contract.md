# fuel-price-signal HTTP API — v1 contract (rev 3)

Status: **implemented server-side** (#416) — `fuel_signal/api_v1.py`,
`fuel_signal/generate_signal_cache.py`, and the blueprint in `fuel_signal/inspect.py`.
The canonical copy lives in `fuel-price-signal`; `fuel-price-signal-app` vendors
it byte-for-byte. App-side implementation status is tracked in
`fuel-price-signal-app`, not in this document.

The app makes one batch of calls each morning on the home LAN and renders the
result. Both endpoints are projections of a single nightly-precomputed blob, so
a pair of calls made seconds apart cannot disagree.

## A note on the examples

Every value in the worked examples below is **derived from the server's code
paths**, not written to look plausible. Four fixtures in earlier revisions were
wrong in ways that read fine — an impossible next-reachable date, a `fill_size`
contradicting its own prose, an invented signal description, a `days_stale`
that disagreed with its own timestamps, and a `signals` array of a length the
server cannot produce.

Nothing is elided: every example is complete, so no reader has to reconstruct
an abbreviated block.

One note for calibration — the illustrative station prices sit *below* the
network average, whereas viking's real data frequently has them above it. No
relationship in this contract depends on that sign and the example is
internally consistent either way, so it is left alone: these fixtures have a
poor track record under editing, and churning a correct example to improve its
realism is a bad trade.

If a future editor does want to realign them, the blast radius is six
mechanical values and two strings — `delta_vs_network` on each station,
`diversion.delta_vs_cheapest_on_route` and its `worth_diverting` flag,
`target_station.price`, and the two `_fill_advice` strings that embed the
price. **No verdict moves**, and in particular the rule-signal table does not:
`AverageNearPreviousMinMaxSignal` is evaluated against the *network average*
(`evaluate_all_signals` passes `avg_current_price`, from
`db.average_price_series`), not against any station's price, so station prices
cannot move it. `verdict` and `fill_size` depend on `probability_buy` against
the model threshold and on drift, neither of which is a function of price
level. The prose in this document has been
through several rounds of review; the numbers had been through none, and they
are the part an implementer copies fastest. If an example looks written rather
than computed, treat it as suspect and check it against `signal.py`.

## Transport

- Base: `http://fuel.home.wordspeak.org:5000/api/v1` (a CNAME for viking's LAN
  address, the existing `fuelsignal-workbench.service` waitress process, one
  extra Flask blueprint).
- No auth. LAN-only is enforced by bind address, as it already is for the
  workbench. A `before_request` hook on the blueprint is the seam if a token is
  ever wanted; nothing speculative is built now.
- No TLS. Household LAN, public fuel prices.

## Fuel type

**E10 is implicit.** There is no `fuel` field. The entire signal path — cycle
detection, the trained model, `PREFERRED_STATIONS` — is E10-only. Adding another
fuel type is a breaking change requiring `/api/v2`, and that is a deliberate,
accepted tradeoff, recorded here so the assumption is written down.

## Dates and times — four distinct values

Conflating any two of these has already caused one real bug (#410). All four
appear in both endpoints.

| Field | Meaning |
|---|---|
| `as_of` | The date whose **prices** were evaluated. |
| `generated_at` | When the nightly precompute ran (ISO 8601, Sydney offset). |
| `routing_day` | The date whose **weekday** decides station reachability. Computed at request time from `_today_in_sydney()`, never cached. |
| `served_at` | When this response was produced. Lets the app detect clock skew and a midnight rollover between its two calls. |

`routing_day` differs from `as_of` routinely: a payload generated 22:00 Tuesday
has `as_of` = Monday or Tuesday, but is read Wednesday morning and must answer
routing for Wednesday.

## Numbers and rounding

Prices are **cents**, as a JSON number. Decicents are a storage detail that never
escapes `db.py` — its read helpers already divide by 10 and return cents — so
cents on the wire is the repo convention, not a departure from it.

**Every cents-valued field is rounded to 1 decimal at serialization**
(`price`, `average_price`, `delta_vs_network`, `drift_cents_per_day`,
`last_cycle_min`, `last_cycle_max`, and the prices inside `target_station` /
`nearest_off_route`). Only `price` is naturally clean; the others are means,
subtractions and divisions that otherwise put `-16.499999999999997` on the wire.
This matches the CLI, which renders all of them `%.1f`.

**Consequence the app must respect:** after rounding,
`price - average_price` will not always equal `delta_vs_network`. These fields
are for **display, not recomputation or cross-checking**. The CLI has exactly the
same property today.

`probability_buy` is not a cents value: 3 decimals (the CLI displays 2).

## Freshness — staleness and fabrication are different things

```json
"freshness": {
  "latest_observation_date": "2026-09-11",
  "days_stale": 1,
  "expected_lag_days": 1,
  "fill_gap": null
}
```

**`days_stale` is computed per request, not cached.** It is the gap between
`latest_observation_date` and the current Sydney date — so computing it at
generation time under-reports by a day every single morning, and that one day
is exactly the difference between the staleness banner firing and staying
silent. With viking's real numbers (last real reading 09-10, generated 22:00
on 09-11, read 07:00 on 09-12) generation-time gives 1 and request-time gives
2, against an `expected_lag_days` of 1. `EXPECTED_LAG_DAYS` was tuned to 1 so
the banner would not cry wolf over the normal overnight lag; a generation-time
implementation would instead make it permanently silent. `latest_observation_date`
is the cached half; the subtraction is the per-request half — the same split as
`routing_day`.

- `latest_observation_date` / `days_stale` — measured against the `prices`
  table, **never** `daily_prices`, because `fill.py` fabricates rows there and
  measuring against it understates staleness by exactly the fabricated days.
- `expected_lag_days` — 1. The app treats `days_stale <= expected_lag_days` as
  normal, not as a warning.
- `fill_gap` — `null`, or `{"start": "…", "end": "…"}`. **Treat this as a canary,
  not a feature.** It reports the single bridging gap between the imported
  historical dataset and the snapshot era; a live `as_of` is years past it, so in
  normal operation it is always `null`. If it is ever non-null the prices are
  fabricated and something is wrong upstream — show a warning, as the CLI does.
  It is kept because it can genuinely change: snapshot retirement
  (`snapshot_retire.py`, a live plan in that repo) moves the gap boundary.

There is deliberately **no `historical_view` field**. It would be a function of
an `as_of` request parameter that v1 does not have, so no data change could ever
make it true — a client branch on it could never execute. When a future version
adds historical queries, adding the field back is purely additive.

## `GET /api/v1/stations`

```json stations-response
{
  "as_of": "2026-09-11",
  "generated_at": "2026-09-11T22:04:13+10:00",
  "routing_day": "2026-09-12",
  "served_at": "2026-09-12T07:01:55+10:00",
  "freshness": {
    "latest_observation_date": "2026-09-11",
    "days_stale": 1,
    "expected_lag_days": 1,
    "fill_gap": null
  },
  "network": {
    "average_price": 178.4,
    "drift_cents_per_day": -0.8,
    "drift_label": "falling"
  },
  "stations": [
    {
      "code": 414,
      "label": "BP Springwood",
      "group": "on_route",
      "price": 161.9,
      "delta_vs_network": -16.5,
      "probability_buy": 0.823,
      "route": {
        "restricted": false,
        "days": null,
        "reachable_on_routing_day": true,
        "next_reachable_date": "2026-09-12",
        "days_until_reachable": 0
      },
      "diversion": null,
      "cheapest_on_route": true
    },
    {
      "code": 261,
      "label": "7-Eleven Penrith South",
      "group": "off_route",
      "price": 157.7,
      "delta_vs_network": -20.7,
      "probability_buy": null,
      "route": {
        "restricted": true,
        "days": [2, 6],
        "reachable_on_routing_day": false,
        "next_reachable_date": "2026-09-13",
        "days_until_reachable": 1
      },
      "diversion": {
        "delta_vs_cheapest_on_route": -4.2,
        "worth_diverting": true,
        "threshold_cents": 3.0
      },
      "cheapest_on_route": false
    },
    {
      "code": 585,
      "label": "EG Ampol Emu Heights",
      "group": "unpriced",
      "price": null,
      "delta_vs_network": null,
      "probability_buy": null,
      "route": {
        "restricted": false,
        "days": null,
        "reachable_on_routing_day": true,
        "next_reachable_date": "2026-09-12",
        "days_until_reachable": 0
      },
      "diversion": null,
      "cheapest_on_route": false
    }
  ]
}
```

The third station demonstrates the point made below: `unpriced` is a *price*
classification, so this station is `group: "unpriced"` while still carrying a
real routing answer (`reachable_on_routing_day: true`). It is absent from the
`FavouriteServiceStationPriceGradientSignal` description because a station with
no recent price data has no gradient and is omitted by the caller.

### The `group` discriminator is not a routing partition

`group` is `on_route` | `off_route` | `unpriced`, but **the partition is on price
first and route second**: `unpriced` means `price is None` *regardless of
reachability*. So an unpriced station that is also off-route appears as
`group: "unpriced"` while its `route` block still describes the restriction. The
app should read `route.reachable_on_routing_day` when it wants a routing answer,
and `group` only for display grouping.

### Array order

`on_route` ascending by price, then `off_route` ascending by price — the CLI's
display order. Then `unpriced` **sorted by label**. That last one is the single
deliberate deviation from the CLI, which emits unpriced stations in
`PREFERRED_STATIONS` dict-insertion order; sorting by label keeps the response
deterministic and stops it silently reordering if someone edits `config.py`.

### Other notes

- `price`, `delta_vs_network` and `diversion` are `null` for `unpriced` stations.
- `route.days` uses Python weekday numbers (Mon=0 … Sun=6) to match
  `STATION_ROUTE_DAYS`. The app maps to display names.
- For a station with `restricted: false`, the server has no next-reachable
  date to pass through — `StationView.next_reachable()` short-circuits to
  `None` when there are no route days. The serializer therefore **emits
  `routing_day` with `days_until_reachable: 0`**, rather than `null`. This is a
  deliberate choice so the app needs no special case; it is stated because two
  implementers would otherwise split between the two answers.
- `days_until_reachable` is `0` for a station reachable on `routing_day`.
  Otherwise `next_reachable_date` is the first date **after** `routing_day`
  whose weekday is in `route.days`, and `days_until_reachable` is the gap
  between them. These three fields are mutually derivable and must agree: a
  fixture that violates the relationship is a bug in the fixture, not a case
  to implement.
- `diversion` is `null` for on-route stations, and for off-route stations when
  nothing priced is on route to compare against. Sign convention matches the CLI:
  `delta_vs_cheapest_on_route` is negative when the off-route station is cheaper,
  and `worth_diverting` is true at `<= -3.0`. `threshold_cents` echoes
  `DIVERSION_WORTH_CENTS`, a module constant rather than a tuned parameter.
- The station list is **read-only**. `PREFERRED_STATIONS` keys are hashed into an
  experiment grading identity, so the API must never accept mutations.

### Do not sort by `probability_buy`

The label's cheapness test is each station's own trailing percentile, so P(BUY)
is station-relative and **not comparable across stations**. Sorting by it points
at whichever pump has fallen furthest against its own history, which is routinely
the dearest one on the list. The app sorts by price. This caveat is why the
prose fields exist.

## `GET /api/v1/recommendation`

```json recommendation-response
{
  "as_of": "2026-09-11",
  "generated_at": "2026-09-11T22:04:13+10:00",
  "routing_day": "2026-09-12",
  "served_at": "2026-09-12T07:01:55+10:00",
  "freshness": {
    "latest_observation_date": "2026-09-11",
    "days_stale": 1,
    "expected_lag_days": 1,
    "fill_gap": null
  },
  "status": "ok",
  "verdict": "BUY",
  "fill_size": "bridge",
  "source": "model",
  "headline": "WORTH A STOP — BP Springwood @ 161.9c, but bridge, don't brim.",
  "reason": "Good local price, but the network is falling 0.8c/day — buy enough to get by and keep some tank for later.",
  "target_station": { "code": 414, "label": "BP Springwood", "price": 161.9 },
  "nearest_off_route": null,
  "cycle": {
    "day": 41,
    "length": 46.33,
    "beyond_expected_length": false,
    "pct_through": 0.8633714655730629,
    "last_cycle_min": 155.3,
    "last_cycle_max": 201.7
  },
  "network": {
    "average_price": 178.4,
    "drift_cents_per_day": -0.8,
    "drift_label": "falling"
  },
  "rules": {
    "verdict": "BUY",
    "mean_value": 0.5,
    "signals": [
      { "name": "AverageCycleTimeSignal",
        "recommendation": "BUY",
        "description": "cycle ending soon (86% through cycle; day 40 / 46.3)" },
      { "name": "AverageGradientAfterPeakSignal",
        "recommendation": "NEUTRAL",
        "description": "price has not flatlined (last 3 gradients: [-0.9, -0.8, -0.7])" },
      { "name": "AverageNearPreviousMinMaxSignal",
        "recommendation": "WAIT",
        "description": "price in middle of last cycle (current 178.4c; last cycle min 155.3c, max 201.7c)" },
      { "name": "FavouriteServiceStationPriceGradientSignal",
        "recommendation": "NEUTRAL",
        "description": "no preferred stations raising sharply (big raisers: none; non-raisers: BP Springwood @ -0.8, 7-Eleven Penrith South @ -0.6)" }
    ]
  }
}
```

### `status`

| Value | Condition | What the app shows |
|---|---|---|
| `ok` | At least one station is both priced and reachable today. | The decision screen. |
| `no_priced_station_on_route` | No station is both priced and reachable, but some priced station exists elsewhere. | A routing screen: nearest alternative and when it's next passed. |
| `no_price_data` | No preferred station has a price at all. | A data-problem screen. |

The middle value is deliberately **not** called `no_route_today`. The condition
tests priced-and-reachable, so a station that genuinely is on the route today but
has no price lands in `unpriced`, empties the on-route set, and triggers this
status. The CLI currently prints "Nothing on your route today" there, which is
false in that case; the wire contract should not inherit the wording bug.

### `verdict` and `fill_size` — two separate questions

Note the example above: the network is **falling**, so `fill_size` is `bridge`,
matching its own `reason` ("bridge, don't brim"). `brim` with a falling network
is a contradiction the mapping table forbids.

`_fill_advice` states its own design: **whether to buy at all is the model's
question; how much to put in is the network's.** Those are materially different
instructions — "FILL UP, brim it" and "WORTH A STOP, bridge don't brim" are both
buys — so they get two structured fields rather than one, and the app never has
to string-match `headline` to tell them apart.

- `verdict` — `BUY` | `WAIT` | `DONT_BUY`, or `null` when `status` is not `ok`.
- `fill_size` — `brim` | `bridge` | `minimum` | `null`.

There is **no `NEUTRAL` at this level.** `combine_signals` emits only the three
values, returning `WAIT` (with a NaN mean) even when every signal is neutral.
`NEUTRAL` exists solely per-signal, inside `rules.signals[].recommendation`,
where the four-value enum *is* correct.

The server's verbatim string is `"DON'T BUY"` — with a space and an apostrophe.
That is a poor wire token, so this one identifier **is** remapped, to `DONT_BUY`.
This is a deliberate exception to the verbatim rule for `rules.signals[].name`;
noted here so nobody reconciles the inconsistency in either direction.

The complete mapping over every branch of `_fill_advice`:

| Branch | `source` | `verdict` | `fill_size` |
|---|---|---|---|
| scored, buying, network not falling | `model` | `BUY` | `brim` |
| scored, buying, network falling | `model` | `BUY` | `bridge` |
| scored, not buying, another station clears the bar | `model` | `WAIT` | `null` |
| scored, not buying, nothing clears the bar | `model` | `WAIT` | `minimum` |
| unscored, rules say BUY | `rules` | `BUY` | `bridge` if falling else `brim` |
| unscored, rules say otherwise | `rules` | `WAIT` or `DONT_BUY` | `null` |

Two consequences the app can rely on: `DONT_BUY` is reachable **only** via
`source: "rules"`, and `{source: "model", verdict: "DONT_BUY"}` never occurs.

### Other fields

- `source` — **`model`** or **`rules`**. The invariant is one-directional:
  `source == "rules"` guarantees every `probability_buy` is `null`;
  `source == "model"` guarantees only that **at least one** station scored.
  Partial scoring is normal, not an edge case — `model_probabilities` inserts
  only the stations that returned a probability, and one is enough to flip
  `source`. So `probability_buy` is nullable **per station, independently of
  `source`**, and the app needs a display state for "model ran, this station
  unscored" (the CLI renders `-`). Provenance matters because scoring failures
  are swallowed by a bare `except` and a fallback otherwise looks like success.
- `headline` / `reason` — rendered prose, displayed verbatim. These encode
  judgement (brim vs bridge; the P(BUY) caveat above) that is error-prone to
  reconstruct client-side and would drift from the CLI over time. **Both are
  populated on all three statuses**, but `reason` is `null` on `no_price_data`,
  where the CLI has no second line. See the table below.
- `target_station` — the station being recommended. `null` unless `status` is
  `ok`.
- `nearest_off_route` — populated **only** when `status` is
  `no_priced_station_on_route`: `{code, label, price, next_reachable_date,
  days_until_reachable}`. Carries what the CLI says on that branch ("cheapest
  preferred station is X @ Yc, next passed in 3d") so `/recommendation` stands
  alone without joining against `/stations`.
- `rules` — the `--explain` breakdown, always present. Cheap, and it is the only
  verdict available when `source` is `rules`.
  - `rules.mean_value` is `number | null`. It is NaN when every signal is
    NEUTRAL, and **NaN is not valid JSON** — Python emits a bare `NaN` token that
    Swift's `JSONDecoder` rejects outright. Serialize it as `null`.
  - `rules.signals` **always has exactly four entries**, in this order.
    `evaluate_all_signals` is an unconditional four-element return with no
    branching, so there is no shorter array and no variable-length case for the
    app to handle. `NEUTRAL` entries are excluded from `mean_value` but still
    appear in the array. In the example above the directional values are
    `[1.0, 0.0]`, a mean of exactly 0.5, which meets the server's `>= 0.5` buy
    threshold — hence `verdict: "BUY"`.
  - `rules.signals[].name` is the **verbatim** identifier from the server
    (`AverageCycleTimeSignal`, `AverageGradientAfterPeakSignal`,
    `AverageNearPreviousMinMaxSignal`,
    `FavouriteServiceStationPriceGradientSignal`). No snake_case remapping: a mapping is a second source of truth that
    drifts the first time a signal is added.
- `drift_cents_per_day` is `null` when drift is unknown; `drift_label` is then
  `"unknown"`. Otherwise `rising` when `> 0.05`, `falling` when `< -0.05`, else
  `flat` — the CLI's thresholds.

### The cycle block

**`length` is a floating-point number, not an integer.** It is
`mean_cycle_length` — the mean inter-peak distance in days, computed as
`float(round(np.average(...), 2))`, so two-decimal values like `46.33` are the
norm and whole numbers are the exception. Typing it as an integer rejects the
**entire** response the first time a cycle averages 46.33 days: the same
one-field-kills-everything failure as an unguarded NaN. The CLI's
`cycle day 18/35` is a *rounded display*, not the stored value.

**`day` and `pct_through` are not the same measure.** `day` is **1-indexed**
(`days_since_last_peak + 1`), while `pct_through` uses the **0-indexed**
numerator (`days_since_last_peak / mean_cycle_length`). So
`day / length != pct_through`, and the app must not "correct" either one. The
relationship that *does* hold exactly is:

```
pct_through == (day - 1) / length
```

`pct_through` is **not rounded** — it is the raw quotient, which is why the
example carries its full precision. Display it rounded; don't store it rounded.

`pct_through` **can exceed 1.0** when a cycle runs long, so do not type it as a
0–1 fraction or feed it unclamped to a progress view.
**`beyond_expected_length` must be consumed, never recomputed.** The server
computes it as `day > round(length)`, the condition behind the CLI's `"46+"`
display — but `round` is not the same function in both languages. Python uses
banker's rounding, so `round(46.5)` is **46**; Swift's `.rounded()` is
half-away-from-zero and gives **47**. `length` is itself `round(x, 2)`, so a
value of exactly 46.5 is reachable, and on such a day a client that recomputed
this would disagree with the server about whether the cycle has overrun. Take
the boolean as sent. This is the same shape of trap as `day` versus
`pct_through`: two things that look mutually derivable and are not, quite.

Note also that `beyond_expected_length` is not equivalent to `pct_through > 1.0`
— at `day: 47, length: 46.5` the flag is true while `pct_through` is 0.989.

The app gets a better deal than the CLI here: the CLI clamps to `"46+"` and
loses the true day number, whereas `day` is always the real value.

### What can and cannot be NaN

**`rules.mean_value` is the only NaN-capable field on the wire.** This is
guaranteed, not incidental, and is worth knowing before someone "helpfully"
extends the null-serialization rule to the other numeric fields.

`combine_signals` returns `float("nan")` with no guard, which is why that field
is nullable. The cycle fields cannot reach it: `_mean_cycle_length` does return
NaN on fewer than two peaks, but `CycleDetector.detect` refuses to build a
`CycleState` from it — `len(peaks) < 2` returns `None` (`cycle.py:116`) and
`mean_cycle_length <= 0 or np.isnan(...)` returns `None` (`cycle.py:124`), and
`build_signals` turns a `None` state into an error rather than a payload.

Two invariants follow, which the app may rely on:

- **`cycle.length > 0` always**, so `pct_through` is always finite,
  non-negative, and division-safe.
- No NaN-bearing `CycleState` can exist, so `pct_through` and `length` need no
  null handling.

### Prose and verdict by status

| `status` | `headline` | `reason` | `verdict` / `fill_size` |
|---|---|---|---|
| `ok` | `_fill_advice` headline | `_fill_advice` reason | per the mapping above |
| `no_priced_station_on_route` | no priced station is reachable today | cheapest preferred station, price, and when next passed | `null` / `null` |
| `no_price_data` | no price data for any preferred station | `null` | `null` / `null` |

`rules.verdict` stays populated on **all** statuses — the rule signals are a
function of the cycle and network prices and do not depend on routing at all.
It is the only verdict available when the top-level one is `null`.

Note the trap this table closes: renaming the enum to
`no_priced_station_on_route` stopped the app inheriting the CLI's false "Nothing
on your route today", and then the same false claim would have arrived anyway as
a `headline` the app displays verbatim. The server session is correcting the CLI
wording in the same PR as the refactor, so there is one source of prose and the
API ships the corrected string. That is a visible, deliberate diff in
`signal-regression.yml`.

## Errors

The cache is missing until the first nightly run, and a from-scratch DB rebuild
on viking drops it. That must not surface as a stale or empty payload:

```
HTTP 503
{ "error": "not_generated",
  "detail": "No precomputed signal. Runs nightly after the price load.",
  "last_generated_at": null }
```

The app treats 503 as "try again tomorrow", not as an error worth alarming about.

## Storage of the precomputed blob

A table in the SQLite DB, written by the nightly job and read by the workbench
process. Not a file in the repo checkout — the daily script's first action is
`git pull --ff-only`. `db.py` owns all persistence by convention; WAL mode lets
the timer write while waitress reads.

**Implementation note (server-side, added post-#416 review):** each endpoint
reads the cache row independently and fresh per request — necessarily so,
since WAL is exactly what lets a new nightly row become visible without a
workbench restart. In the few-millisecond window where the nightly job's
`write_signal_cache` commit lands between two client calls, the two endpoints
can therefore read different rows and disagree, contradicting the opening
paragraph's guarantee. The window is narrow (one write, once nightly) and low
severity, but real. A client that wants to rule it out entirely should compare
**both** `generated_at` and `routing_day` across its `/stations` and
`/recommendation` calls, re-fetching on either mismatch; `generated_at` alone
is not enough, because `routing_day` is computed fresh per request (see
above) and can differ between the two calls even when both read the same
cached row — for example when the calls straddle midnight in Sydney. v1 has
no request parameter to pin a specific `generated_at`.

## Split of work between cached and per-request

Cached nightly (expensive, day-stable, a function of `as_of` and the DB): station
prices and `P(BUY)`, cycle state, network average and drift, freshness facts, the
rule verdict.

Computed per request (cheap, day-sensitive, pure Python over cached numbers): the
on-route/off-route partition, `next_reachable`, diversion arithmetic, and the
`headline`/`reason` fill advice. This is what keeps `routing_day` correct at any
hour instead of freezing the reachability answer at generation time.
