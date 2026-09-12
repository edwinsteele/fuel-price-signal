---
name: signal-cache-format-versioning
description: fuel_signal.db.signal_cache stores a JSON blob whose SignalPayload fields are indexed by hand. A field rename/addition needs _CACHE_FORMAT_VERSION bumped, or an old row 500s instead of the documented 503.
metadata:
  type: reference
---

`fuel_signal/api_v1.py`'s `_encode_signal_payload`/`_decode_signal_payload` hand-index every
field of `SignalPayload` (and its nested `CycleState`/`CombinedVerdict`/`SignalEvaluation`)
when writing/reading the JSON blob stored in `db.signal_cache.payload_json`. There is no
generic (de)serializer — no `dataclasses.asdict`/`**kwargs` — so a field renamed, removed, or
added-as-required on `SignalPayload` (which its own docstring explicitly invites, per
[[signal-payload-day-stable-boundary]]) breaks `_decode_signal_payload` on any row written
before the code change.

That row is not hypothetical: a `signal_cache` row persists across a `git pull` + waitress
restart until the next nightly `fuel_signal.generate_signal_cache` run overwrites it. A
`SignalPayload` field change landing between two nightly runs leaves exactly one stale row for
one day.

The mitigation (added in review on PR #425, same day as the endpoints themselves):

- `encode_cache_entry` stamps a `"version": _CACHE_FORMAT_VERSION` key (currently `1`).
- `decode_cache_entry` raises `ValueError` on a version mismatch, and lets `KeyError`/`TypeError`
  propagate on a missing/malformed field either way.
- `inspect.py`'s `_load_signal_cache` catches `(KeyError, TypeError, ValueError)` around the
  decode call and treats it as "not generated" — the documented `HTTP 503`, not a bare 500.

**When you add/rename/remove a `SignalPayload` field:** bump `_CACHE_FORMAT_VERSION` in
`fuel_signal/api_v1.py`. The version bump doesn't make the old row *readable* — it just
guarantees the failure is a clean, immediate `ValueError` (version mismatch) instead of a
`KeyError` deep in field-by-field decoding, which is easier to diagnose but behaves identically
from the blueprint's point of view (both degrade to 503). Either way, the fix on the ground is
the same: run `generate_signal_cache` again.
