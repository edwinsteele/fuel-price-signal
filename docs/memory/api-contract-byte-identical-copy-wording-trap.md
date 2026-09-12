---
name: api-contract-byte-identical-copy-wording-trap
description: fuel-price-signal-app vendors docs/api-contract.md byte-for-byte, so any phrase anchored to "this repo"/"here"/"this copy" is true in at most one of the two homes and self-contradicts in the other.
metadata:
  type: project
---

`docs/api-contract.md` is canonical here; `fuel-price-signal-app` vendors it
byte-for-byte (its `make sync-contract` diffs against this repo's raw `main`
copy and fails on any difference). Because the two copies are identical bytes
in two different repos, any sentence that refers to its own location rather
than to itself as a document breaks when read from the other home.

Two instances hit in the same PR (#427, 2026-09-12):

- *"**This copy** is the server-side source of truth... App-side status is
  tracked in that repo, **not here**."* — read from the app repo, "this copy"
  is the vendored duplicate (not the source of truth) and "here" is exactly
  where app-side status *is* tracked — both clauses invert.
- Fixed by rephrasing to be about the document, not the reader's location:
  *"App-side implementation status is tracked in `fuel-price-signal-app`, not
  in this document."* — true regardless of which checkout you're reading.

**Rule for future edits near the status block or anywhere else in this file:**
name repos explicitly (`` `fuel-price-signal` ``, `` `fuel-price-signal-app` ``)
rather than using "this repo"/"here"/"this copy"/"that repo". If a sentence's
truth value depends on which of the two checkouts is open, it's wrong in one
of them.
