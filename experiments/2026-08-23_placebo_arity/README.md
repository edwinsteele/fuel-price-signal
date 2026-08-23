# Does the placebo band widen with arity? (bd fps-3jj.14)

**STATUS: measured 2026-08-24. The k=3 floor is now batch1's grading ruler.**

## The finding

| | k=1 | k=3 |
|---|---|---|
| draws | 20 | 10 |
| band mean | −0.0089 c/L | **+0.0251 c/L** |
| band std | 0.0800 c/L | **0.0999 c/L** |

**std ratio 1.25×, and the design cannot resolve it.** F(9,19) = 1.56, two-sided
p = 0.395, 95% confidence interval on the std ratio **[0.74, 2.40]**. The smallest ratio
n=10-vs-n=20 could have called significant is **1.70×**. So the honest statement is
"could not see it", NOT "it is not there" — the interval comfortably contains both no
widening and a doubling. Written this way deliberately; see
`feedback_correction_no_finer_than_estimator`.

**Two effects moved, in opposite directions, and neither is individually resolvable.**

- **Dilution** (mean −0.0089 → +0.0251): three junk columns degrade the fit more than one
  does, which moves the bar *easier*.
- **Opportunity** (std 0.0800 → 0.0999): three junk columns give the fit more places to
  find spurious structure, which moves the bar *harder*. This is the effect `fps-3jj.14`
  was opened to measure.

They partly cancel, which is why the net bar move is small. Decomposing the batch-of-5
bar (−0.2170 → −0.2706):

```
mean shift   +0.0340   (easier)   dilution
std widening −0.0519   (harder)   opportunity — the arity effect proper
n 20 → 10    −0.0357   (harder)   NOT arity: the t-penalty for a less certain band
```

**Two-thirds of the net hardening at batch level is draw count, not arity.** The k=3
ruler is stricter partly because it is a *less certain* ruler: `family_wise_z_threshold`
uses `df = n_draws − 1`, so z rises from 2.602 to 2.959 at n_candidates=5 purely because
there are half as many draws. Worth knowing before attributing the whole move to arity.

## The decision: k=3 is batch1's ruler

Promoted by rename (`fps-3jj.14`, 2026-08-24):

```
noise_floor.json     -> noise_floor_k1.json    (the arity baseline, retained)
noise_floor_k3.json  -> noise_floor.json       (the grading ruler)
```

**The decisive reason is not the statistics.** batch1's candidates are 3, 3, 3, 2, 2
columns, and `_noise_band` refuses a floor whose arity is *below* the run's. Under the
k=1 floor all five candidates would have dossiered `available: false` — batch1 would be
ungradeable. The promotion is a precondition, not a preference.

The statistics support it as the *conservative* choice rather than proving it necessary:
where the two rulers differ the k=3 one is net harder, and the one-sided guard means the
two 2-column candidates get a slightly-too-hard bar (disclosed as
`floor_arity_exceeds_run`) rather than a too-soft one. "Could not see a widening" is not
licence to keep grading wide arms on a narrow ruler — that is exactly the favourable bias
the bead exists to remove.

**batch1's bar is therefore −0.167 c/L judged singly, −0.271 c/L at the batch-of-5
family-wise correction.** Propagated to `docs/CONVENTIONS.md`, `docs/routines/generator.md`
and `docs/routines/dossier.md`.

## Rejected: re-running k=3 at 20 draws to match k=1's certainty

Considered and ruled out on 2026-08-24. Two reasons.

**It is not available.** The binding pool is the COLUMNS, not the seeds. Each draw needs
`arity` distinct source columns and the lock has 54, so 20 draws at k=3 would need 60.
The hard ceiling is `floor(54/3)` = **18 draws** — and 18 consumes every lock column,
leaving the `self_correlation` screen no substitution headroom. Substitution is not
hypothetical: it fired twice on the k=1 run (two all-NaN LGA columns). With an empty
fallback tail the run either hard-fails after hours of compute, or is forced to keep a
column that FAILED the screen — a placebo still correlated with its original, which
narrows the band and makes the bar easier. That is the bias direction this bead removes.

**It would not be worth it if it were.** Going 10 → 18 draws recovers 0.032 c/L of the
0.036 c/L draw-count penalty at n_candidates=5. That is below the realised arbiter's own
~0.05 c/L decision-flip quantum — precision the instrument downstream cannot express.

## The ruler self-destructs above ~4 columns

Found while writing the above, from the owner's observation that a candidate could have
one feature per LGA. Because the t-critical uses `df = n_draws - 1` and max draws is
`min(floor(54/k), floor(30/k))`, the bar does not degrade gracefully as arity rises — it
explodes, and almost none of the explosion is the arity effect. It is the collapsing draw
count. On batch1's k=3 band (mean +0.0251, std 0.0999), family-wise at 5 candidates:

| arity | max draws | z(5 cand) | bar | |
|---|---|---|---|---|
| 1 | 30 | 2.503 | −0.2249 | |
| 2 | 15 | 2.711 | −0.2457 | |
| 3 | 10 | 2.959 | −0.2705 | batch1 today |
| 4 | 7 | 3.360 | −0.3105 | generator.md already invites this |
| 5 | 6 | 3.635 | −0.3380 | |
| 6 | 5 | 4.105 | −0.3849 | |
| 10 | 3 | 8.042 | −0.7783 | 3x the largest win in project history |
| 13 | 2 | 38.972 | −3.8682 | absurd |
| 18+ | 1 | — | — | **no band possible** (needs n ≥ 2) |
| 35 | 0 | — | — | `select_draws` raises: 70 columns needed, 54 exist |

**This project's own motivating precedent for group candidates is ungradeable by this
ruler.** `docs/routines/generator.md` justifies multi-column candidates on the 35 LGA
features having gone in together. A 35-column candidate cannot have a noise floor
computed at all.

**And the gap is already live.** generator.md invites 2–4 columns. A 4-column candidate
gets a 7-draw floor, so its bar is **0.0400 c/L harder** than a 3-column candidate's —
purely from band thinness, nothing to do with its arity being higher. Two candidates in
the same batch, both legitimate under the generator's own rules, would be graded by
rulers of materially different strictness, and nothing in the dossier says so.

Filed as bd `fps-3jj.21` (P1), blocking batch2.

**It DOES matter for batch2**, because the Bonferroni correction grows with candidate
count: the same 10→18 draw-count penalty is 0.032 c/L at 5 candidates, 0.043 at 10, and
0.050 — a full flip — at 15. And at batch2's size the 54-column pool cannot supply both
higher arity and more draws, so something structural has to give (reusing source columns
across draws, at a cost in independence; or drawing placebos from outside the lock).
Filed as bd `fps-3jj.21`, blocking batch2.

## Provenance

k=3 run: `git_sha 55106bb`, `computed_at 2026-08-23T11:53Z`, `wall_seconds 8368` (2.3h),
`partial: false`. Both floors share `baseline_fingerprint 54:1a6ec2d84a69`,
`tank_params 50/3.571/1d/10%`, `seed 42`, `n_windows 14` and identical
`inner_fold_params` — `compare_arity.py` hard-stops if they ever diverge.

Run log `noise_floor_k3.log` is gitignored (`*.log`). It was 172 lines, every one
accounted for by 10 × (1 `loaded` + 14 `fold` + 1 `done`) + 10 draw lines + 2 trailer
lines: no warnings, no substitutions, no retries.

**No `experiments/ledger.yaml` entry.** This is an instrument finding — a measurement of
the ruler — not a candidate claim, and the ledger takes one entry per falsified claim.

## The question

`experiments/pipeline/noise_floor.py` built its null by adding exactly ONE placebo column
per draw. `docs/routines/generator.md` invites 2–4 column candidates, and three of
batch1's five are 3-column groups. A k-column arm gets more chances for the fit to find
something than a 1-column placebo arm does, so a k-column candidate graded against a
1-column band is graded against a ruler that does not know k — **in the candidate's
favour**, by an amount no single run can reveal.

The owner's decision on the bead was to NOT cap candidates at one column: batch1's primary
aim is (b), learning how this pipeline behaves, and constraining candidate shape to suit
the ruler would distort the thing being measured. So: measure the gap instead.

The question is deliberately **"is the band materially wider at k=3 than at k=1"**, not
"what is the k=3 band to three significant figures". A variance comparison does not need a
gate-quality band, which is why the k=3 run is 10 draws rather than 20.

## How to reproduce

Both floors are committed, so the comparison alone re-runs in a second:

```bash
PYTHONPATH=. uv run python experiments/2026-08-23_placebo_arity/compare_arity.py
```

To rebuild the k=3 floor from scratch (~2.3h as measured; 10 draws was the ceiling at the
time because the seed pool binds `n_draws * arity` at 30 — note the COLUMN pool binds
harder still, at `floor(54/3)` = 18):

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor batch1 \
    --arity 3 --n-draws 10 --out-name noise_floor_k3.json \
    2>&1 | tee experiments/batches/batch1/noise_floor_k3.log
```

That writes the side-file name. It was the grading ruler's name only AFTER the promotion
rename below — `compare_arity.py` reads the post-promotion names
(`noise_floor_k1.json` / `noise_floor.json`), so repoint it or redo the rename.

`compare_arity.py` hard-stops if the two floors disagree on `baseline_fingerprint`,
`tank_params` or `seed` — arity is supposed to be the only thing that differs — and
refuses a `partial: true` floor outright.

## What it reports, and how to read it

- Both bands' mean/std/n, and the **std ratio** k=3 / k=1.
- An F-test on the variance ratio with its 95% CI, **and the smallest ratio this design
  could call significant** (n=10 vs n=20 is not much power). A quiet result reads as
  "could not see it", never "it is not there" — see
  `feedback_correction_no_finer_than_estimator` for why this repo insists on writing down
  what a statistic can actually resolve.
- The decision-relevant number: **how far the bar moves, in c/L**, at both the
  single-candidate and the batch-of-5 correction. Read it against results.csv's 0.03–0.26
  c/L range for single-column features, and against the realised arbiter's decision-flip
  quantum (~0.05 c/L at 7d, smaller at batch1's 1d cadence).

## Known limitation of this comparison

Draws 1 and 10 of the k=3 bank are not independent: they reuse block seeds 97/101 on the
lock's most-correlated column pair, giving placebo columns correlated at 0.965 and 0.778
(measured 2026-08-23). So the bank has ~9 effective draws, not 10. batch1's committed
**k=1** floor carries the same defect (seeds 97 and 101 each appear twice), so it sits on
both sides of the ratio and largely cancels. Filed as bd `fps-3jj.20`; not a blocker for
this comparison, and the reason is that it is not a new defect — it predates arity.
