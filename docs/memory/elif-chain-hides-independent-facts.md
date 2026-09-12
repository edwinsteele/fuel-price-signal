---
name: elif-chain-hides-independent-facts
description: Chaining mutually-exclusive branches for facts that aren't actually mutually exclusive lets the first branch checked permanently shadow information a later branch would have shown.
metadata:
  type: project
---

`render_text`'s route headline (`fuel_signal/signal.py`, `build_signals` /
`render_text`) computes two independent booleans off one `views` list:
*is anything reachable today with no price* (`on_route_unpriced`) and *is
anything priced but off route* (`off_route`). Both can be true at once — an
outage on the daily commute plus a priced weekly-run station is the ordinary
case, not an edge case.

#418's original fix (PR #424, first commit) chained them as `if on_route: ...
elif on_route_unpriced: ... elif off_route: ...`. That reads as "pick whichever
applies", but because `on_route_unpriced` was checked before `off_route`, and
4 of 5 `PREFERRED_STATIONS` are on the daily commute (see
`config.STATION_ROUTE_DAYS` — only station 261 is restricted), `on_route or
on_route_unpriced` is true almost every day live. The `off_route` branch —
the one that names the cheapest reachable alternative and says when it's next
passed — became dead code in production despite having correct logic and
passing tests, because the fixture-driven tests never populated both facts at
once (own-review finding on PR #424, caught by the repo owner running
`/code-review high` against the merged fix).

**The general trap:** when a function derives several booleans from the same
input and each becomes a report of a genuinely independent fact, an `elif`
chain silently picks one to report and drops the rest — and unit tests that
each set up only one of the facts at a time won't catch it, because in
isolation each branch's message is correct.

**The fix:** check each fact separately and compose the output from every one
that holds, rather than a single `if/elif` that returns after the first match.
Here: the headline picks one sentence (unpriced-but-reachable, vs.
"nothing reachable", vs. "no price data at all") but a second, independent
line — the off-route pointer — is appended whenever `off_route` is non-empty,
regardless of which headline sentence fired.

A same-review finding on the same PR: the fix's first cut also duplicated
output (stations named in the new headline were printed again by the
pre-existing "no price data" footer loop, since that loop iterates the same
unpriced set the headline draws from) and asserted a temporary-sounding "no
price data **yet**" on a code path also driven with historical `--as-of`
dates into permanent forward-fill gaps (`_station_price_on`'s BP Springwood
268-day gap). Both are instances of the same root cause: introducing a new
code path that reads from a set already consumed elsewhere, without checking
what else reads that same set.
