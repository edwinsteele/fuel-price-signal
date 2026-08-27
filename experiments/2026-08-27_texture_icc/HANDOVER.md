# Handover prompt — fps-3jj.23, after the 32-draw run

Paste everything below the line into a fresh session.

---

Pick up bd `fps-3jj.23` (P1, `in_progress`). The instrument merged as PR #337 (squash `45736d4`);
the 32-draw measurement run has now finished. Your job is the analysis and the decision it feeds
— **no new fits should be needed.**

**Read these first, in order:**
1. `bd show fps-3jj.23` — the note dated 2026-08-27 supersedes the bead's own "How" section.
   Read the note before the description; the description specifies a design that was measured to
   be incapable of answering the question.
2. `experiments/2026-08-27_texture_icc/README.md` — the power analysis, the pinned-column
   rationale, and the three outcomes with what each decides.
3. `docs/CONVENTIONS.md`, the section on the ICC bound.

**Step 1 — read the result.** The run wrote `experiments/batches/batch1/noise_floor_icc.json`
(32 draws, 8 pinned columns x 4 seeds, arity 1). Run:

```bash
PYTHONPATH=. uv run python experiments/2026-08-27_texture_icc/measure_icc.py \
    --pinned experiments/batches/batch1/noise_floor_icc.json
```

It prints the by-column ANOVA ICC (**primary** — the quantity `effective_n_draws` actually
charges), the variance-ratio ICC against `noise_floor_k1.json` (secondary, reported for
continuity with how the bead framed it), both with CI and the design's own resolution limit, and
the bar at each arity under the shipped bound vs the measured value.

**Two things to check before trusting the output**, both cheap:
- `wall_seconds` should be ~22000 (~6.1h) and `n_draws` 32. A short run means it died partway.
- The payload now stamps `all_nan_baseline_columns`. It should list **exactly 5** columns. The
  published bar-vs-arity table depends on that count (50/49/48 usable → n_eff 5.35/5.25/5.16 at
  20 draws arity 20) and this run is the first artifact that verifies it. **If it isn't 5, the
  table in `CONVENTIONS.md` and `INDEX.md` needs recomputing** — say so loudly rather than
  quietly proceeding.

**Step 2 — the decision.** The bead's acceptance criteria allow three outcomes:

- **ICC ≈ 0** (upper bound comfortably under ~0.2) → every width collapses to the arity-1 bar of
  −0.152; overlap costs nothing; **close `fps-3jj.22`**, it buys nothing.
- **ICC ≈ 0.391** → the shipped value survives, now as a measurement of the *right* quantity
  rather than a bound on a coarser one. `fps-3jj.22` is worth doing as an optimisation.
- **ICC > 0.391** → current bars on wide candidates are too **easy**. Raise the constant in the
  same change; `fps-3jj.22` becomes the fix rather than an optimisation.

Note the design resolves down to 0.262 and, at a point estimate of 0, bounds at 0.376. A quiet
result reads "could not see it", **never** "it is not there" — this repo's standing rule.

**Step 3 — land it.** Remaining acceptance criteria:
- [ ] ICC stated as a number; `placebo.TEXTURE_ICC_BOUND` replaced with it, raised, or
      **explicitly held with the reason** (all three are permitted).
- [ ] `docs/CONVENTIONS.md` + the `TEXTURE_ICC_BOUND` docstring carry the measured value,
      including the bar table at each width.
- [ ] The family-vs-column gap resolved (the by-column ANOVA closes it) or re-stated as open.
- [ ] `fps-3jj.22` re-triaged on the result.
- [ ] This README's Results/Conclusion filled in, and `experiments/INDEX.md`'s row updated from
      `open`.
- [ ] `bd close fps-3jj.23` + `bd dolt push` — **only once the ICC is actually a number.** It was
      deliberately left open at merge because the measurement hadn't run.

Code changes to `experiments/pipeline/**` or `fuel_signal/**` need a PR (branch from
`origin/main` first). Changes confined to `experiments/2026-08-27_texture_icc/**` and
`experiments/INDEX.md` go direct to `main` — **and must be pushed, not just committed.**

**Context worth having:**
- The ICC is **inert at the arity actually in use.** Across its whole possible range it moves the
  bar 0.000 c/L at arity 1–2, 0.005 at 3, 0.011 at 4 — every width batch1 has run — against 0.059
  at arity 10 and 0.383 at 35. So a large measured ICC is not an emergency for current candidates;
  it matters for the wide groups `docs/routines/generator.md` invites. Don't overstate it.
- `fps-8o0` is **closed** (PR #340): the singleton-group defect fixed in `measure_icc.py` also
  existed in `texture_channel.py`, which is what *produced* 0.391. It bit — `network` is a
  singleton family — so the shipped constant it produced was computed on 19 draws and 4
  families, not 20 and 5. Recomputed correctly, 0.391's true upper bound was 0.226 (see this
  experiment's README, "the 0.391 being replaced was not what it claimed"). Doesn't change
  anything here — `TEXTURE_ICC_BOUND` had already moved to the by-column 0.274 before this was
  checked, so it was overstated on its own terms, not a live grading ruler.
- The pinned bank is **not a grading ruler** and the code enforces it — `null_method:
  placebo_column_pinned_source`, which `_noise_band` refuses. Don't promote it to
  `noise_floor.json`.
- Estimator tests live in `tests/test_texture_icc_estimator.py` (16 tests). If you change
  `_anova_icc`, they cover known-ICC recovery, one-sided-bound coverage, tail symmetry and the
  singleton case.
