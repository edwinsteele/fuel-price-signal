# Analysis — shallow-elongated regime constraint for A (#214)

**Verdict:** the engineered two-axis features fail. Per-fold diagnostics + an
oracle diagnostic on the train segment show why: A's level *does* carry an
early elongated-vs-normal signal, but it does NOT cleanly distinguish
**shallow elongated** from **steep elongated** until very late phase
(>1.4) — and shallow-vs-steep within elongated is the actual discrimination
the failure regime requires (per step5e).

## Outcome against the four #214 gates

Headline: median across 5 seeds; seed-variance gate clean (no flagged
cells, n_flagged_gt_5x = 0 for both cohorts).

| Gate | Threshold | R_raw | R_composite |
|---|---|---|---|
| 1. Fold-7 hard25 Δll | ≤ −0.04 | **+0.0092** ❌ | 0.0000 ❌ |
| 2. Worst-fold hard25 Δll | ≤ +0.01 | **+0.034** ❌ | +0.018 ❌ |
| 3. Net population Δll | ≤ 0 | −0.0013 ✓ | −0.0001 ✓ |
| 4. Row-level on `ext_descent_shallow` | reduce | not pursued — see below | not pursued |

Both runs fail gates 1 and 2. R_composite is essentially a null result
(net population Δll ≈ 0, no fold-7 movement). R_raw helps 8/14 folds
slightly in aggregate but **opens a new fold-level regression** of +0.034
on the worst fold and makes the original target fold (7) marginally worse.

Gate 4 (row-level) was deprioritised after the oracle diagnostic showed
the mechanism the features were supposed to capture is not visible in A
in the early-to-mid phase range. Disambiguating "features helped on the
target rows but bled into other rows" vs "features didn't even help the
target rows" no longer changes the path forward.

## Why R_raw failed — the diagnostic evidence

Three diagnostic plots, each answering a different question.

### `phase_diagnostic.png` — does d/dt of A have a trough signature?

Plots median network_px_std (level) and median network_px_std_delta_3d
(rate) vs cycle_pct_through, no regime split, de-dup to one row per date.

Finding: **no**. Dispersion compresses sharply in the first 5% of the cycle
(just past peak: ~17c → ~7c by phase 0.25), then sits flat at its floor
through the trough zone and most of the cycle. d/dt is most negative at
the peak transition, near zero through the trough zone, and dips negative
again only in deep elongation (phase > 1.25). There is no characteristic
compression spike at the trough — the compression is already done by the
time we get there.

Implication: rate-of-change of A is not a trough detector. It's a "peak
just happened" detector and a "cycle still going" indicator.

### `phase_by_regime.png` — does A look different in the three descent regimes?

Same axes as above, but split rows into three buckets by row-wise (PIT-safe)
regime tags: `normal_descent`, `ext_descent_steep`, `ext_descent_shallow`.

Finding: A's level differs across regimes — visibly higher in the two
elongated buckets than in normal_descent throughout. But ext_descent_steep
and ext_descent_shallow track each other closely; the within-elongated
discrimination is weak at row level.

### `phase_oracle_cycles.png` — oracle view: pre-classified cycles, train-only

Pre-classifies whole cycles by their full-cycle properties (length >
training median, descent slope at trough vs −0.9 c/day), then plots
median A-level and median d/dt of A vs phase. Train segment only
(< 2021-11-01, before fold 1's val window).

```
                 phase 0.1-0.3   phase 0.4-0.55   phase 0.7-0.95   phase 1.0-1.5
normal             5.81c          5.49c             6.68c            (n/a)
elongated_steep    8.04c          7.22c             7.15c            5.36c
elongated_shallow  7.14c          7.24c             6.93c            7.65c
```

Findings:

1. **Elongated-vs-normal is visible in A's LEVEL from phase 0.2 onwards.**
   Normal cycles sit at 5-6c through mid-cycle, elongated cycles sit at
   7-8c. A ~2c gap from early descent. Information exists.

2. **A's DELTA carries a peak-approach signal, not a trough signal.**
   Normal cycles spike to +3c/3d at phase 0.7-0.8 (anticipating next peak).
   Elongated cycles stay flat at zero (not yet near their next peak).
   Useful, but for next-peak detection, not trough proximity.

3. **Shallow vs steep within elongated diverges only at phase > 1.4.**
   At phase 0.4-0.55 (trough zone): steep 7.22c, shallow 7.24c — basically
   indistinguishable. Only in deep elongation tail (1.4+) does the shallow
   line climb to 7.65c while steep drops to 5.36c. With only 8 shallow
   cycles in train, that tail is noisy.

Implication: the discrimination the failure regime requires
(shallow-within-elongated, per step5e) is NOT visible in A — neither in
level nor in rate — at the phase where it would be useful for prediction.

## Why this falsifies the R_raw hypothesis

R_raw's hypothesis was: provide the model with `elongation_ratio` and
`cycle_descent_slope_so_far` so it can carve out the (elongated, shallow)
corner and condition A's contribution there.

The oracle diagnostic shows two problems with that plan:

1. **The shallow-vs-steep distinction within elongated isn't strongly
   present in A in the early/mid phase range** where prediction matters.
   The model could only use the elongation_ratio + slope features to
   identify the regime, but A's value in that regime doesn't carry
   distinctive information for the model to condition on. Conditioning
   on regime is moot if A reads the same in both sub-regimes.

2. **R_raw's aggregate "helps" comes from the elongated-vs-normal
   distinction** (which IS visible in A from phase 0.2) — but that's the
   wrong target. Per step5e, steep elongated rows don't need different
   treatment; only shallow elongated do. R_raw was probably suppressing
   A's contribution on all elongated rows, helping shallow elongated but
   hurting steep elongated by approximately the same amount, with the
   net winning out marginally on some folds and losing on others.

The +0.034 worst-fold regression we now have under R_raw is consistent
with this: a fold heavy in steep-elongated rows would lose lift from A
that R_raw partially suppresses.

## Forward paths (not pursued in this PR)

### Reject and pivot to external data (#215)

Per #215's decision matrix: if #214 fails, the external-data case
strengthens (file a scoping issue for one cheap external signal, e.g.
Singapore Mogas weekly). The shallow-within-elongated discrimination is
genuinely hard to surface from intra-series data — the slow-descent
regime simply doesn't generate the cross-station coordination signals A
was built to read. External wholesale-vs-retail margin is a plausible
fallback because it's regime-orthogonal: a stuck-shallow descent often
reflects wholesale costs flat, which is observable independently.

### Or: replace A's construction with a phase-conditional residual

`A_residual = A - E[A | phase]` — encodes "is A unusual for its phase?"
rather than "is A high?". The phase-conditional baseline could be
constructed PIT-safely (rolling window of A's median at the same phase
over the previous N cycles). This would surface the elongated-vs-normal
discrimination cleanly. **But:** it does not solve shallow-vs-steep
within elongated, which is the actual failure regime. So it's a partial
fix that may help some folds and not the target one.

### Or: try a model-level intervention — RECOMMENDED next probe

The two ingredients needed to carve out the failure regime are both in
the feature set already:

- A's level cleanly separates elongated vs normal from phase 0.2 (oracle
  diagnostic).
- The descent slope (`cycle_descent_slope_so_far` or the constituent
  peak / price / days_since columns the baseline already has) separates
  shallow from steep directly.

R_raw had both. The tree didn't carve the corner because expressing
"A misreads when (elongated AND shallow)" requires THREE nested splits:
elongation indicator → slope → A-contribution within that leaf. Each
nested split needs enough data per leaf; shallow elongated is ~8% of
train data and the model may simply not have found the conditional.
R_composite gave the model the regime indicator as a single binary
column and was also flat — suggesting the gap is not "model doesn't
know the regime exists" but "model doesn't know how to use A differently
inside it."

The natural follow-up: **hand the tree the interaction directly as a
column.** Something like:

- `A_x_shallow_elong = network_px_std × is_extended_shallow_descent`
  (gives the tree the corner-conditioned A value in one split).
- Or smoother: `A × elongation_ratio × (1 + slope)` for continuous
  signal.
- Or a complementary `A_x_other = network_px_std × (1 − is_extended_shallow_descent)`
  so the tree can split on whether A's normal-regime interpretation
  applies.

LightGBM monotone constraints on A conditional on regime are an
alternative but more architecture-y; the interaction column is the
cheapest cleanest test of "is the failure mode about combination
representation, or about underlying signal availability?".

Cheap experiment, follows directly from what the diagnostics told us.

## Recommended next action

**Update #215** with the diagnostic finding. The intra-series late-descent
track has produced one piece of work (RAC_full graduated from #212) and
one rejected experiment (#214). Per #215's decision matrix, the next move
is to file a scoping issue for one cheap external signal as a probe.

Before that, the phase-conditional residual idea is worth a small Step 2
experiment — partial fix or not, it's cheap and tests whether A can be
made regime-aware enough to recover some of the lost lift. If yes, the
external-data case weakens slightly.

## Outputs

- `runs.csv` — per-(fold, run, seed) log-losses; 210 rows.
- `fold_run.csv` — per-(fold, run) aggregates (mean + median across seeds,
  delta vs R0, seed_std).
- `meta.json` — config, summary, seed-variance flags, definitions.
- `rowpreds.parquet` — per-row predictions for all (fold, run, seed)
  cells; would have supported gate 4 if pursued. Gitignored.
- `run.log` — captured stdout from the harness (gitignored).
- `phase_diagnostic.png/.py` — A level + rate vs phase, no regime split.
- `phase_by_regime.png/.py` — same axes, three row-wise regime buckets.
- `phase_oracle_cycles.png/.py` — full-cycle oracle classification,
  train-only segment.

## Memory updates

- `project_late_descent_investigation` — record the diagnostic finding +
  the path-forward branching.
- `feedback_features_via_experiment_first` — already created in this PR.
