---
name: sydney-offset-timestamps-dont-sort-as-strings
description: ISO 8601 timestamps stored with a Sydney offset (signal_cache.generated_at, devices.last_seen) are not in chronological order as strings across the April DST fall-back hour — compare them in SQL with julianday(), which applies the offset.
metadata:
  type: project
---

This repo stores wall-clock times as ISO 8601 text with a Sydney offset
(`2026-10-10T07:01:55+11:00`), e.g. `signal_cache.generated_at` and, from #435,
`devices.first_seen`/`last_seen`. Text order matches time order only while
the offset stays the same. When Sydney leaves DST (first Sunday of April,
03:00+11:00 → 02:00+10:00), the hour 02:00–03:00 happens twice:
`02:15:00+10:00` (16:15Z) is **later** than `02:30:00+11:00` (15:30Z), but it
sorts first as a string. So a plain `ORDER BY last_seen` or
`WHERE last_seen < ?` is wrong for that hour, and a comparison against a `Z`
or epoch-derived time is wrong all year.

SQLite's `julianday()` parses a trailing `±HH:MM` or `Z` and converts to UTC,
so `ORDER BY julianday(col)` / `julianday(a) < julianday(b)` compare instants.
It returns NULL for text it can't parse, and NULL compares false, so a
malformed bound silently matches nothing. The `devices` helpers in `db.py`
(#435, PR #440) guard each bound with `SELECT julianday(?)` and raise if it
is NULL.

Python-side, compare `datetime.fromisoformat(...)` values (aware datetimes
compare by instant), never the strings.

Related: [[api-contract-generated-at-not-an-identity]].
