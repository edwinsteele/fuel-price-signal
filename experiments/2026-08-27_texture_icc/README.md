# texture_icc — measuring the placebo texture ICC by SOURCE COLUMN

- **Date:** 2026-08-27
- **Branch:** `claude/3jj-23-ecb614`
- **SHA:** see the PR that lands `--same-source-column`
- **Status:** open — instrument + power analysis landed, 32-draw bank queued
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

## Result so far: the design in the bead cannot discharge the bead

`power.py` prices each candidate design before any fit is spent. This is the repo's own rule
about null results ("a quiet result reads *could not see it*, never *it is not there*") applied
one step earlier — to the design, not to its output.

**Design A, the bead as written** — one pinned column, *n* seeds, variance ratio against the
committed 20-draw `noise_floor_k1.json`:

| draws | hours | smallest resolvable ICC | 95% upper bound if the point estimate lands at 0 |
|---|---|---|---|
| 10 | 1.9 | 0.729 | **0.653** |
| 15 | 2.9 | 0.650 | 0.622 |
| 20 | 3.8 | 0.604 | 0.604 |
| 30 | 5.7 | 0.552 | 0.584 |

The bead budgets 10 draws. That design's **best possible outcome is an upper bound of 0.653** —
looser than the 0.391 already shipped — and it cannot call anything under 0.729 significant. Two
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

| arity | ICC 0 | 0.2 | 0.391 | 0.6 | 0.8 | 1.0 | spread |
|---|---|---|---|---|---|---|---|
| 1 | −0.152 | −0.152 | −0.152 | −0.152 | −0.152 | −0.152 | **0.000** |
| 2 | −0.152 | −0.152 | −0.152 | −0.152 | −0.152 | −0.152 | **0.000** |
| 3 | −0.152 | −0.153 | −0.154 | −0.155 | −0.156 | −0.157 | 0.005 |
| 4 | −0.152 | −0.154 | −0.156 | −0.159 | −0.161 | −0.163 | 0.011 |
| 6 | −0.152 | −0.156 | −0.161 | −0.166 | −0.171 | −0.177 | 0.025 |
| 10 | −0.152 | −0.161 | −0.171 | −0.183 | −0.196 | −0.211 | 0.059 |
| 20 | −0.152 | −0.175 | −0.203 | −0.249 | −0.316 | −0.431 | 0.280 |
| 35 | −0.152 | −0.200 | −0.287 | −0.535 | no band | no band | 0.383 |

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

Pending the 32-draw run.

## Conclusion

Pending. The three outcomes and what each decides:

- **ICC ≈ 0** (upper bound comfortably under ~0.2) — every width collapses to the arity-1 bar of
  −0.152, overlap costs nothing, and `fps-3jj.22` buys literally nothing. Close it.
- **ICC ≈ 0.391** — the shipped value survives as a measurement of the right quantity rather
  than a bound on a coarser one. `fps-3jj.22` is worth doing as an optimisation: it would
  recover most of the 0.135 c/L an arity-35 candidate currently pays purely for overlap.
- **ICC > 0.391** — current bars on wide candidates are too **easy**. Raise the constant in the
  same change, and `fps-3jj.22` becomes the fix rather than an optimisation.

Whatever lands, `power.py`'s second table stands on its own: at arity ≤ 4 this constant cannot
move a verdict, and any future write-up claiming it "sets every bar" without that qualifier is
overstating it.

## Followups

- `fps-3jj.22` — blocked on this, re-triaged on the result.
- `docs/CONVENTIONS.md` § the ICC bound and `placebo.TEXTURE_ICC_BOUND`'s docstring both carry
  the measured value once the run lands.
- `experiments/2026-08-26_placebo_draw_independence/README.md` § "What is still a bound" points
  here.
