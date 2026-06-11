# Analysis — interaction-column probe for A in the shallow-elongated corner (#231)

**Verdict: REJECTED.** All three interaction columns fail gates 1 and 2, and
the target fold (7) gets *worse*, not better. Nothing graduates.

## Outcome against the four #231 gates

Sign convention: `Δ = run − R0`, negative is better. Median across 5 seeds is
the headline.

| run | fold-7 Δh25 | worst-fold Δh25 | helps h25 | net-pop Δ | gates |
|---|---|---|---|---|---|
| R1 `A_x_shallow_elong` | **+0.0402** ❌ | +0.0402 ❌ | 6/14 | +0.0012 ❌ | 1,2,3 fail |
| R2 `+A_x_other`        | **+0.0181** ❌ | +0.0579 ❌ | 6/14 | −0.0012 ✓ | 1,2 fail |
| R3 `A_x_smooth`        | **+0.0368** ❌ | +0.0393 ❌ | 6/14 | −0.0014 ✓ | 1,2 fail |

- **Gate 1** (fold-7 hard25 Δ ≤ −0.04): every run is the *opposite sign* —
  fold 7 regresses. Decisive fail.
- **Gate 2** (no fold's hard25 Δ worse than +0.01): worst-fold is +0.04 to
  +0.058 everywhere. Fail.
- **Gate 3** (net population Δ ≤ 0): R2/R3 are marginally negative but smaller
  than the cohort seed-std floor (≈0.008) — i.e. noise, not lift. R1 fails.
- **Gate 4** (row-level on `ext_descent_shallow`): moot once 1+2 fail.

## Not a seed artifact

The seed-variance gate flagged 3 of 56 cells, **all on fold 4** (a shock fold —
high variance there is expected). Fold 7's seed_std (≈0.037) is within the
cohort distribution and not flagged; its regression is stable across seeds
(median Δh25 +0.0402 ≥ mean +0.0139, so most seeds agree it worsened).

## What the probe established

#231 was built as a clean discriminator for the fold-7 failure:

- **representation problem** (the tree had the ingredients but couldn't combine
  them, per #214) → handing it the explicit product should fix fold 7;
- **signal-availability problem** (A doesn't carry the corner signal) → the
  product is noise and stays flat.

Fold 7 did neither — it got *worse*. That rules out the representation
hypothesis and confirms signal-availability, in its strong form: A's reading in
the shallow-elongated corner is not merely absent but **actively misleading**.

Mechanism (consistent with the step5 row-level result that A's per-row effect is
harmful on `ext_descent` rows): `A_x_shallow_elong` is "A's value, isolated to
exactly the corner where A is wrong, zero elsewhere". The greedy tree finds it
useful on train, splits on it, and amplifies the harm out of sample. A
misleading signal gets *used* and backfires (worse); only a noise signal would
be ignored (flat). This confirms the #214 oracle diagnostic's pessimistic read
outright.

## Conclusion for the track

The intra-series route to **rehabilitating A** in the shallow-elongated corner
is exhausted: #214 (raw axes) and #231 (explicit interaction) both failed, the
second one backwards. #212 still stands — A genuinely helps normal descent — we
simply cannot fix its corner behaviour from A.

## Forward path — corrected

The earlier default ("if #231 fails, reopen external data" per #215) is too
quick. What we have ruled out is narrow: **A** does not carry the corner signal.
We have *not* shown that no intra-series feature does — the #214 oracle
diagnostic tested A specifically and is silent about other candidates.

Next move (cheap, no model fitting): an **oracle-diagnostic sweep** over *other*
intra-series candidate signals (trough-proximity / days-since-last-meaningful-
drop, cross-LGA or cross-brand late-descent consensus, the triplet's B and C
re-examined in this corner). Apply the same train-only oracle existence check
used in #214. Two-stage discipline:

- a candidate that **doesn't** separate shallow-vs-steep within elongated under
  the oracle is dropped for free (you can't beat the oracle);
- a candidate that **does** is only a lead — it must then earn a PIT-safe proxy
  and pass paired-WFCV on the untouched folds. The oracle hit never ships.

External data becomes earned (not assumed) only if the sweep comes up empty
across several genuinely different signal families. Tracked as a follow-up
design issue.

## Outputs

- `runs.csv` — per-(fold, run, seed) log-losses (280 rows).
- `fold_run.csv` — per-(fold, run) aggregates (mean + median, delta vs R0, seed_std).
- `meta.json` — config, summary, seed-variance flags, definitions.
- `rowpreds.parquet` — per-row predictions (gitignored).
- `run.log` — captured stdout (gitignored). Total wall ≈ 853s.
