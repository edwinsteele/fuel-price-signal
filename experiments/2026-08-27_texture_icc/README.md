# texture_icc — measuring the placebo texture ICC by SOURCE COLUMN

- **Date:** 2026-08-27
- **Branch:** `claude/3jj-23-ecb614`
- **SHA:** see the PR that lands `--same-source-column`
- **Status:** closed — 32 draws run 2026-08-27 (5.7h); `TEXTURE_ICC_BOUND` 0.391 → **0.274**
- **Bead:** `fps-3jj.23` (P1), which **blocks** `fps-3jj.22`

## Hypothesis

`placebo.TEXTURE_ICC_BOUND = 0.391` prices how much a placebo bank loses when its draws reuse
source columns. It is a **bound on the wrong quantity**: a 95% upper limit on the ICC of delta
grouped by texture *family* (`F(3,16) = 0.411`, `p = 0.75`, resolvable only down to 0.359),
used as a stand-in for the ICC by *column*. Family is coarser — two different
`days_since_trough_entry_<lga>` columns are same-family but different-column, while two
placebos from the *same* column are more alike than that — so the column ICC is plausibly
**larger**, and if it exceeds 0.391 then bars on wide candidates are too **easy**, the exact
bias this machinery exists to remove.

A bank whose draws are pinned to a few deliberate source columns, each at its own block seed,
separates the two channels a placebo's delta moves through:

- **alignment** — did *this* rearrangement happen to line up with the target? Varies with the
  block seed.
- **texture** — how much does a column *of this shape* perturb the fit at all, regardless of
  alignment? Distribution, cardinality, NaN rate, autocorrelation. Fixed by the source column.

Two draws from the same column share texture exactly, so grouping deltas by column and running
a one-way ANOVA gives the ICC directly: between-column variance is texture, within-column is
alignment.

## Finding 1 (before any fit): the design in the bead cannot discharge the bead

`power.py` prices each candidate design before any fit is spent. This is the repo's own rule
about null results ("a quiet result reads *could not see it*, never *it is not there*") applied
one step earlier — to the design, not to its output.

**Design A, the bead as written** — one pinned column, *n* seeds, variance ratio against the
committed 20-draw `noise_floor_k1.json`:

| draws | hours | smallest resolvable ICC | 95% upper bound if the point estimate lands at 0 |
|---|---|---|---|
| 10 | 1.9 | 0.661 | **0.587** |
| 15 | 2.9 | 0.583 | 0.557 |
| 20 | 3.8 | 0.539 | 0.539 |
| 30 | 5.7 | 0.489 | 0.519 |

Every figure in this section is a **one-sided 95% bound at the 5% tail** — the same tail
`texture_channel.py` used to produce the 0.391 being replaced, so the two are like-for-like.
An earlier revision priced Design A at the two-sided 0.975/0.025 pair while Design B used the
5% tail, printing 97.5% bounds and 95% bounds in the same column: that inflated Design A's
numbers (10 draws read 0.653 rather than 0.587) and inflated the gap between the designs.
Found in review of PR #337; the conclusion is unchanged, every number moved.

The bead budgets 10 draws. That design's **best possible outcome is an upper bound of 0.587** —
looser than the 0.391 already shipped — and it cannot call anything under 0.661 significant. Two
hours of fits would produce a *worse* number than the one being replaced, whatever the deltas
came out as. The cost is structural, not bad luck: it is an unpaired variance ratio between two
separately-computed banks, on `F(9, 19)`, and a variance ratio needs a lot of both.

**Design B — k pinned columns × m seeds, one-way ANOVA by column.** Same cost per draw, but it
estimates the by-column quantity *directly* (so the family-vs-column gap closes rather than
being restated), and it depends on no second artifact, hence on no second artifact's sampling
noise:

| k | m | draws | hours | smallest resolvable ICC | 95% upper bound at a point estimate of 0 |
|---|---|---|---|---|---|
| 4 | 5 | 20 | 3.8 | 0.309 | 0.606 |
| 5 | 4 | 20 | 3.8 | 0.339 | 0.548 |
| 6 | 5 | 30 | 5.7 | 0.245 | 0.414 |
| **8** | **4** | **32** | **6.1** | **0.262** | **0.376** |
| 8 | 5 | 40 | 7.6 | 0.208 | 0.321 |
| 12 | 5 | 60 | 11.4 | 0.166 | 0.232 |

**8 × 4 = 32 draws is the smallest design of any shape that can leave a bound tighter than the
0.391 it would replace.** That is what is queued.

The estimator itself was validated against synthetic data with a known ICC before being pointed
at anything real — essentially unbiased at 0.0/0.2/0.391/0.7, with 95%-upper-bound coverage
measured at 94.7–97.3%.

## Where the constant is load-bearing at all

Second half of `power.py`. Single-candidate bar in c/L, 20-draw bank, batch1's live ruler
(`noise_floor.json`: arity 3, 10 draws, mean +0.0251, std 0.0999), shipped `effective_n_draws`
over the real overlap structure — the ICC swept across its **entire** possible range:

| arity | ICC 0 | 0.2 | **0.274** (live) | 0.6 | 0.8 | 1.0 | spread |
|---|---|---|---|---|---|---|---|
| 1 | −0.152 | −0.152 | −0.152 | −0.152 | −0.152 | −0.152 | **0.000** |
| 2 | −0.152 | −0.152 | −0.152 | −0.152 | −0.152 | −0.152 | **0.000** |
| 3 | −0.152 | −0.153 | −0.153 | −0.155 | −0.156 | −0.157 | 0.005 |
| 4 | −0.152 | −0.154 | −0.155 | −0.159 | −0.161 | −0.163 | 0.011 |
| 6 | −0.152 | −0.156 | −0.158 | −0.166 | −0.171 | −0.177 | 0.025 |
| 10 | −0.152 | −0.161 | −0.165 | −0.183 | −0.196 | −0.211 | 0.059 |
| 20 | −0.152 | −0.175 | −0.185 | −0.249 | −0.316 | −0.431 | 0.280 |
| 35 | −0.152 | −0.200 | −0.226 | −0.535 | no band | no band | 0.383 |

(The highlighted column tracks `placebo.TEXTURE_ICC_BOUND`; it read 0.391 — bars −0.154 /
−0.156 / −0.171 / −0.203 / −0.287 — before this run. The *spread* column is a property of the
0→1 range and did not move.)

**"`TEXTURE_ICC_BOUND` sets EVERY bar" is true but misleading, and the correction matters.** At
arity 1–2 the constant's value is irrelevant *by construction* (no two draws share a column at
all). At arity 3–4 — the shape `docs/routines/generator.md` invites, and the shape all five of
batch1's candidates had — the constant's **whole range** moves the bar by ≤ 0.011 c/L, an order
of magnitude under the realised arbiter's own decision quantum. The measurement is not worth
running for candidates of that width.

It is worth running because the same document actively invites *wide* groups — "every feature
win in this project's history arrived as a group … the 35 LGA event features went in together"
— and at arity 10 the range is 0.059 c/L, at arity 20 it is 0.280, and at arity 35 the constant
decides between a −0.152 bar and no band at all. **The load-bearing case is real but it is the
wide one, not "every bar".** That is the sentence this run exists to replace with a number.

## The pinned set, and why these columns

`select_draws`' own even spread over the 49 **usable** columns (five
`days_since_trough_entry_<lga>` columns are all-NaN in the frozen frame and are excluded),
i.e. the same sampling rule the real bank uses:

```
cycle_pct_through                       days_since_trough_entry_georges_river
station_price_cents                     days_since_trough_entry_north_sydney
brand_mean_cents                        days_since_trough_entry_ryde
days_since_trough_entry_camden          days_since_trough_entry_wollondilly
```

**Not hand-picked, deliberately.** The ICC that `effective_n_draws` charges is the correlation
between two draws sharing a column *from the population the bank actually draws from*, so the
pinned set must be sampled the way the bank samples — not curated for texture diversity, which
would overstate the between-column variance and therefore the ICC. That the spread lands 5 of 8
on LGA trough counters is a property of batch1's lock (35 of its 49 usable columns are LGA
counters), and is the composition being priced.

It spans three texture shapes anyway — a phase fraction, two price levels, five counters — and
includes `station_price_cents`, the level-like column the bead names as the deliberate first
pick and the hardest case for any time-axis reordering. `measure_icc.py` prints the per-column
breakdown, so any family structure inside the estimate stays visible rather than pooled away.

**The table above rests on one environmental fact: exactly 5 of batch1's 54 declared columns
are all-NaN.** The overlap is invariant to *which* columns die (verified across eight different
5-subsets — identical `n_eff` every time) but not to *how many*: 50 / 49 / 48 usable columns
give `n_eff` 5.35 / 5.25 / 5.16 at 20 draws, arity 20, so one more dead column shifts every
wide-arity bar. That count was measured off the frozen frame, which is gitignored — so it could
not be re-checked from the repo. `compute_noise_floor` now stamps `all_nan_baseline_columns`
and `n_baseline_columns_available` into every floor it writes, and `bank_model` reads the stamp
in preference to its hardcoded default. **The queued 32-draw run is therefore its own
verification of the figure this table depends on.** Raised in the second round of PR #337
review.

**Pre-flight (real batch1 frame, 2,084,203 rows):** all 32 draws pass the `MAX_SELF_CORRELATION
= 0.6` screen, `|self_corr|` between 0.001 and 0.097, four draws per column, 32 distinct seeds
(30 from the historical pool, 2 from the unbounded extension). Screening cost 6s — so the 6.1h
is all fits, and the run cannot die on a screen failure hours in.

## How to invoke

Build the bank (the long one — ~6.1h; `--batches-dir` points at the primary clone because
`features.parquet` and `fuel_signal.db` are gitignored and live only there):

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor batch1 --arity 1 --n-draws 32 --batches-dir /Users/esteele/Code/fuel-price-signal/experiments/batches --out-name noise_floor_icc.json --same-source-column cycle_pct_through --same-source-column station_price_cents --same-source-column brand_mean_cents --same-source-column days_since_trough_entry_camden --same-source-column days_since_trough_entry_georges_river --same-source-column days_since_trough_entry_north_sydney --same-source-column days_since_trough_entry_ryde --same-source-column days_since_trough_entry_wollondilly 2>&1 | tee experiments/2026-08-27_texture_icc/run.log
```

Then read it:

```bash
PYTHONPATH=. uv run python experiments/2026-08-27_texture_icc/measure_icc.py --pinned /Users/esteele/Code/fuel-price-signal/experiments/batches/batch1/noise_floor_icc.json
```

The power table, any time — pure arithmetic over committed artifacts, seconds:

```bash
PYTHONPATH=. uv run python experiments/2026-08-27_texture_icc/power.py
```

## This bank is NOT a ruler, and the code enforces that

The property that lets a pinned bank measure the ICC — draws repeating a source column, so
their deltas share texture exactly — is precisely what makes its band **too narrow to grade
anything against**. Rather than rely on whoever runs it choosing a safe `--out-name`, the
payload is stamped `null_method: placebo_column_pinned_source`, and `dossier_tables._noise_band`
already refuses any floor whose `null_method` is not `placebo_column`. Written straight over
`noise_floor.json` it would disable grading, not corrupt it. There is a test for exactly that.

The pinned path also **refuses to substitute** a different column when one fails the screen,
where the ordinary path must substitute. Substituting here would quietly turn a same-column pair
into a different-column pair and bias the measured ICC toward **zero** — the direction that reads
"no problem". A pinned column that fails is retried at fresh seeds and then the call raises.

## Results

The 32-draw bank ran on 2026-08-27: **5.7h** wall (`wall_seconds` 20,392), 32/32 deltas, no
partial folds, `baseline_fingerprint 54:1a6ec2d84a69`, `tank_params 50/3.571/1d/10%`, seed 42,
14 windows — every field identical to the committed `noise_floor_k1.json` it is compared
against. Artifact: `experiments/batches/batch1/noise_floor_icc.json` (primary clone; the batch
dir is gitignored).

**The environmental fact the whole bar table rests on holds.** `all_nan_baseline_columns` came
back with **exactly 5** entries — `days_since_trough_entry_` for bayside, botany_bay,
hunters_hill, lane_cove, waverley — so `n_baseline_columns_available = 49`, the count the
overlap model assumed. This run is the first artifact to verify it rather than trust a
measurement taken off a gitignored frame; nothing above needed recomputing.

### Delta by source column

```
                                       n      mean       std
brand_mean_cents                       4 -0.013229  0.048784
cycle_pct_through                      4  0.013721  0.094981
days_since_trough_entry_camden         4  0.008968  0.114935
days_since_trough_entry_georges_river  4 -0.020326  0.174113
days_since_trough_entry_north_sydney   4  0.107071  0.126655
days_since_trough_entry_ryde           4  0.010841  0.167211
days_since_trough_entry_wollondilly    4  0.108898  0.071293
station_price_cents                    4  0.006315  0.094757
```

Whole bank: 32 draws, mean +0.0278, std 0.1152 c/L.

### The ICC by column — one-way ANOVA (primary)

| quantity | value |
|---|---|
| `F(7,24)` | 0.735 (`p` = 0.645) |
| ICC point estimate | **−0.071** — i.e. 0 |
| 95% CI, two-sided | [−0.229, +0.359] |
| **95% upper bound, one-sided 5% tail** | **+0.274** |
| smallest ICC this design could resolve | 0.262 |

`F < 1` means the between-column mean square came in *below* the within-column one: no texture
clustering visible at all. The point estimate is negative, which for ICC(1) means zero.

**It reads "could not see it", never "it is not there".** The design resolves to 0.262, so an
ICC of 0.05 and an ICC of 0.25 are both entirely consistent with this result. What it does rule
out, at the same 5% tail `0.391` itself was derived at, is anything above **0.274**.

### The bead's own construction, reported for continuity

| quantity | value |
|---|---|
| within-column variance (pinned bank) | 0.014124 |
| total variance (`noise_floor_k1.json`) | 0.006397 |
| ratio | 2.208 (`F(24,19)`, two-sided `p` = 0.082) |
| implied ICC | −1.208, CI [−4.178, +0.100] |
| smallest ICC resolvable | 0.574 |

As `power.py` predicted, this estimator sees nothing — it cannot call anything under 0.574. But
the *direction* is worth recording: the pinned bank's **alignment-only** variance came out 2.2×
the multi-column floor's **alignment + texture** variance, which the estimator's stated
assumption (a common `Var(alignment)` across banks) forbids. The difference is not resolvable
(the ratio's 95% CI, [0.90, 5.18], contains 1), so this is not evidence that the pinned set is
unrepresentative — but it is a second, independent reason the bead's original design could not
have discharged the bead: its load-bearing assumption is the one the data leans against. The
ANOVA needs no such assumption, which is why it is primary.

**The residual, stated rather than buried.** If that lean is real — the 8 pinned columns having
genuinely higher alignment variance than the average column — then the ANOVA's ICC is biased
**low** and 0.274 is optimistic. The pinned set was chosen by `select_draws`' own even spread
rather than curated, precisely so the estimate would be for the population the bank draws from,
so there is no mechanism proposed for such a bias; it is recorded as the open edge, and it is
the reason the constant ships as a bound rather than as the point estimate.

### A by-product: the 0.391 being replaced was not what it claimed (`fps-8o0`)

Checked in the same pass, because `fps-8o0` flagged that the defect fixed in `measure_icc.py`
still lives in `texture_channel.py`, which is what produced 0.391. It bites: batch1's `network`
texture family is a **singleton** (11 LGA counters, 3 cycle_magnitude, 3 price_level_other, 2
cycle_phase, 1 network). `texture_channel.py` filters groups with `if len(g) > 1` — dropping
that draw from the sums of squares — while computing `n = len(draws) = 20`, `k_bar = n / 5` and
`df2 = n - len(groups) = 16`. So the published `F(3,16) = 0.411 → 0.391` is a 19-draw/4-group F
carried on a 20-draw/5-group `df` and `k_bar`.

Recomputed consistently over all five families with the corrected estimator:

| | n | k | k_bar | F | p | ICC | 95% upper | resolvable |
|---|---|---|---|---|---|---|---|---|
| all 5 families, consistent (**correct**) | 20 | 5 | 3.20 | `F(4,15)` = 0.330 | 0.854 | −0.265 | **0.226** | 0.391 |
| singleton dropped, consistent | 19 | 4 | 3.82 | `F(3,15)` = 0.411 | 0.748 | −0.182 | 0.402 | 0.374 |
| *as `texture_channel.py` actually computes it* | 19/20 | 4/5 | 4.00 | `F(3,16)` = 0.411 | 0.748 | −0.173 | **0.391** | 0.359 |

The third row reproduces the shipped 0.391 and its published "could not resolve below 0.359"
exactly, which is what identifies the defect: the F comes from 19 draws over 4 groups, the `df2`
and `k_bar` from 20 over 5. Fixing only the filter (row 2) would have *raised* the constant to
0.402; fixing the whole computation (row 1) lowers it to 0.226. Both fixes had to land together
or the correction would have pointed the wrong way — which is why `fps-8o0` says check the
number before changing the script.

So the superseded constant was overstated on its own terms too. Two estimators, of two different
quantities, on two different artifacts, both landing under 0.391 and both with point estimates
at 0.

## Conclusion

**`placebo.TEXTURE_ICC_BOUND` = 0.274.** Replaced, not held: 0.274 is a one-sided 95% upper
bound derived exactly the way 0.391 was, on the quantity `effective_n_draws` actually charges
rather than on a coarser stand-in for it. The family-vs-column gap is closed rather than
re-stated — and it closed in the direction the bead could not call in advance, with the column
ICC *smaller* than the family bound, not larger. The pessimistic end still ships; the point
estimate (0) does not.

The three outcomes this experiment was set up to distinguish, against what happened:

- **ICC ≈ 0** (upper bound comfortably under ~0.2) → close `fps-3jj.22`. **Not met.** The point
  estimate is 0, but the bound is 0.274 and the design floors at 0.262. Reading a quiet F as
  "zero" is the exact move this repo refuses.
- **ICC ≈ 0.391** → the shipped value survives. **Not met**, and it did not survive.
- **ICC > 0.391** → bars on wide candidates are too easy; raise it. **Not met.**

The result landed between the first two, which is why the decision is "replace with the measured
bound" rather than any of the three pre-written verdicts.

### What moved

| candidate columns | effective draws | bar (1 candidate) | was, at 0.391 |
|---|---|---|---|
| 1–2 | 20.00 | −0.152 | −0.152 |
| 3 | 18.17 | −0.153 | −0.154 |
| 4 | 16.50 | −0.155 | −0.156 |
| 10 | 10.81 | −0.165 | −0.171 |
| 20 | 6.74 | −0.185 | −0.203 |
| 35 | 4.31 | −0.226 | −0.287 |

**Lowering the constant LOOSENS the bar** — less correlation, more effective draws, narrower
band, easier bar. That is the direction that admits false positives, which is why it moved only
on a direct measurement of the right quantity and why the bound rather than the estimate is what
ships. Every candidate batch1 has actually run (arity 1 and 3) moved by **0.001 c/L**, so no
already-written dossier's verdict is affected; a re-render of one would differ in the fourth
decimal place. `power.py`'s second table stands as written: at arity ≤ 4 this constant cannot
move a verdict, and any future write-up calling it "the input to every bar" without that
qualifier is overstating it.

### `fps-3jj.22`: demoted, not closed

The bead's rule was "close it if the ICC is near zero, because then overlap costs nothing". The
point estimate says exactly that; the bound does not, and the bound is what grades. At 0.274 an
arity-35 candidate still pays **0.074 c/L** purely for source-column overlap (down from 0.135),
an arity-20 one 0.033, an arity-10 one 0.013, and an arity-3 one 0.001. So `fps-3jj.22`'s
ceiling roughly halved and remains real only for candidates far wider than anything yet run.

It is also no longer the cheapest way to buy that 0.074 back. Drawing placebo sources from
outside the lock is a change to the null's construction; **taking more draws is not**, and the
bound is limited by the design's resolution rather than by anything about the data — 12 columns
× 5 seeds (60 draws, ~11.4h) lands near 0.23 even if the point estimate again comes out at 0,
and closer to 0.15 if `F` comes in below 1 again as it did here. If a wide candidate ever needs
the bar tightened, re-measure first.

## Followups

- `fps-3jj.22` — re-triaged: **demoted, not closed**. Ceiling roughly halved (0.135 → 0.074 c/L,
  and only at arity ~35); more draws is now the cheaper route to the same tightening.
- `fps-8o0` — the singleton defect is confirmed live and its effect quantified above. Fixing it
  no longer moves a shipped constant, since `TEXTURE_ICC_BOUND` is now the by-column number; the
  bead is a script correction plus a note that its published 0.391/0.359 were wrong.
- Done in the same change: `placebo.TEXTURE_ICC_BOUND` = 0.274 with its comment block rewritten,
  `docs/CONVENTIONS.md` § the ICC and both bar tables, `docs/routines/generator.md`'s bar quotes,
  `docs/routines/retrospective.md`'s `icc =` reference, and
  `experiments/2026-08-26_placebo_draw_independence/README.md` § what was still a bound.
- `tests/test_texture_icc_estimator.py` now pins the published tables at an explicit
  `PUBLISHED_ICC`, so the next move of the constant fails loudly with the list of documents to
  re-transcribe rather than leaving them silently stale.
