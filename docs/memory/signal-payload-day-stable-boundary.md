---
name: signal-payload-day-stable-boundary
description: SignalPayload fields must be a function of (as_of_date, database) only — nothing derived from the wall clock, even transitively. A field that reads `now` is unsafe to cache overnight.
metadata:
  type: project
---

`fuel_signal/signal.py`'s `build_signals` was split (#415) into
`compute_signal(...) -> SignalPayload` (expensive: cycle detection, per-station
price and P(BUY), drift, the rule verdict — the ~29s `load_history` cost) and
`render_text(payload, *, routing_day=None, now=None, explain=False)` (cheap:
station reachability, diversion arithmetic, freshness age, all formatting).

The reason for splitting on *cost*, not on *data vs presentation*: `routing_day`
and `as_of` are deliberately two dates (see PR #410 — collapsing them into one
was the bug that moved a Wed/Sun station on and off route depending on the hour
a historical `--as-of` was run). Station 261 is `frozenset({2, 6})`. If a
`SignalPayload` were
computed and cached at 22:00 Tuesday, a naive design would bake in that Tuesday's
reachability, and 07:00 Wednesday would still report it off-route — on the one
morning it's actually reachable. The fix isn't "recompute reachability" as a
special case; it's a hard rule about what may live on `SignalPayload` at all:

**Every field on `SignalPayload` must be a function of `(as_of_date, database)`
and nothing else.** Not `now`, not `routing_day`, not anything derived from
either — directly or transitively. `render_text` takes `routing_day`/`now` as
its own arguments, evaluated fresh at render time, specifically so a cached
payload read the next morning is still correct.

The freshness banner is the field most likely to violate this by accident: the
*observation date* (`last_real_price_date`) is day-stable and lives on the
payload; the *age* (`today - observation date`) is not, and is computed inside
`render_text` from a fresh `now`. Getting this backwards — caching the
already-computed age — silently under-reports staleness by exactly however long
the payload sat in the cache.

`test_rendering_the_same_payload_at_different_now_moves_only_the_freshness_banner`
in `tests/test_signal.py` is the mechanical check: it renders one payload at two
`now` values five days apart and asserts everything except the freshness line is
byte-identical. Run it (or extend it) before adding a field to `SignalPayload` —
a field that fails this test belongs in `render_text`'s arguments instead.

This boundary was the precondition for #416 (CLOSED via PR #425, 2026-09-12):
`fuel_signal/generate_signal_cache.py` computes and JSON-encodes the payload
once nightly (`fuel_signal/api_v1.py`'s `encode_cache_entry`), and the
`/api/v1/stations`/`/api/v1/recommendation` blueprint in `inspect.py` decodes
it and projects it against the real current Sydney date per request — the
same `payload` vs `routing_day`/`now` split as `render_text`, just over HTTP
instead of stdout. The iOS app consuming this lives in the sibling repo
`fps-app`, not in this one. See [[signal-cache-format-versioning]] for the
JSON encoding's own gotcha.
