# Does the placebo band widen with arity? (bd fps-3jj.14)

**STATUS: measurement not yet run.** This directory holds the analysis machinery; the
finding replaces this section once `noise_floor_k3.json` exists. Committed ahead of the
result, against the usual "commit a lab book entry after results exist" convention,
because PR #332's merge plan depends on this script existing and a plan that references an
untracked file is not reviewable.

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

```bash
# 1. the k=3 floor (~1.9h). 10 draws is the ceiling: the draw pools bind on
#    n_draws * arity against a 30-seed pool, so max draws at arity k is floor(30/k).
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor batch1 \
    --arity 3 --n-draws 10 --out-name noise_floor_k3.json \
    2>&1 | tee experiments/batches/batch1/noise_floor_k3.log

# 2. the comparison
PYTHONPATH=. uv run python experiments/2026-08-23_placebo_arity/compare_arity.py
```

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
