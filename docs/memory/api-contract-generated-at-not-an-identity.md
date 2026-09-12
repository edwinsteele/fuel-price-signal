---
name: api-contract-generated-at-not-an-identity
description: generated_at is whole-second ISO 8601 and the iOS client's JSONDecoder rejects fractional seconds outright — don't "fix" the cache-race gap in docs/api-contract.md by adding sub-second precision.
metadata:
  type: project
---

`docs/api-contract.md`'s implementation note (post-#416, under **Storage of the
precomputed blob**) documents a narrow race: two cache writes landing within the
same second produce an identical `generated_at`, so a client comparing
`generated_at` (and `routing_day`) to detect a stale/mixed pair can't catch that
case. Comparing both fields is a strong mitigation, not a guarantee — see the
note itself for the full reasoning.

**The trap:** the obvious fix is to widen `generated_at` to millisecond
precision and restore the guarantee. Don't. The iOS client
(`fuel-price-signal-app`'s `SignalClient`) decodes with
`JSONDecoder.dateDecodingStrategy = .iso8601`, which is Foundation's
`.withInternetDateTime` — it rejects fractional seconds. A payload carrying
`22:04:13.472+10:00` doesn't degrade gracefully; every decode on both
endpoints fails at once, for every client. This was flagged independently by
Codex reviewing the app-side vendoring of the same doc change (2026-09-12).

If the race is ever worth closing rather than documenting, the shape that
doesn't break the decoder is a **new** field — a monotonic cache row id or
write counter that clients compare instead of inferring identity from a
timestamp. That's a v2 conversation (tracked as an issue in
`fuel-price-signal-app`, not here) — the underlying category is that a
timestamp is being used as an identity, and identity isn't what a timestamp is
for. Second-resolution is just where that shows up first.
