# Leading indicators of Sydney E10 — AIP TGP exploration

- **Date:** 2026-06-20
- **Branch:** main
- **SHA:** 70be117 (baseline 54-feat)
- **Status:** open (candidate `tgp_delta_7d` clears the log-loss screen; realised/CpL arbiter pending — see Followups)

## Goal
Exploratory (no prior hypothesis): find an external upstream price series that leads
Sydney pump prices, is free + daily-pullable, and could become a feature. Look at it
side-by-side with Sydney E10 over history and decide if there's exploitable lead.
Surfaces the #215 "external wholesale/crude lead" thread the #262 headroom map pointed at.

## Data source (the find)
**AIP Sydney ULP Terminal Gate Price** — the literal wholesale input. Free xlsx,
**daily (weekday), 2004→present, c/L GST-inclusive** (same units as our pump data),
no scrape/FX needed. `data/AIP_TGP_2026-06-19.xlsx` (sheet "Petrol TGP", Sydney col).
Download page: aip.com.au → "Historical ULP and Diesel TGP data".

## The arc
1. **Overlay** (`overlay.py` → `tgp_vs_e10_overlay.png`, `tgp_vs_e10_calm_2023.png`).
   TGP is the *floor*; the E10 cycle rides on top. Cycle troughs "kiss" TGP. Retail
   margin (E10−TGP) mean ~13 c/L, range −8 (spike) to +60 (cycle peak). The lead is
   regime-dependent: contemporaneous in calm, multi-week in shocks (2020 crash sticky-
   down, 2022 spike reprice-up).
2. **Oracle diagnostic** (`diagnostic.py` → `gap_to_tgp_diagnostic.png`). Gap-to-TGP
   vs actual: **depth-remaining r²=0.63**, days-to-trough r²=0.47, **gap÷slope r²=0.19
   (WORSE)** → ship the raw gap, not the time projection. Kiss-gap at trough: mean
   −0.3, sd 3.1 c/L (troughs land on TGP, near-unbiased).
3. **Gap screen** (`paired_wfcv.py`, R0 vs R1=`station_minus_tgp_cents`). Δll_all by
   regime: normal **−0.012** (helps), shock **+0.011** (HURTS) — inverse of hypothesis.
   Mechanism: the kiss is to the floor *at trough time*; today's TGP is accurate when
   the floor is stable (calm), misleading when it's moving (shock).
4. **Velocity redesign** (`paired_wfcv_velocity.py`, near-factorial). `tgp_delta_7d`
   carries strong **independent** signal (R2 vel-only: normal −0.018 / shock −0.010 /
   pooled −0.015) and **rescues shock** (R3 gap+vel shock −0.005 vs R1 +0.011). Raw gap
   stays a shock drag even with velocity (R2 shock −0.010 beats R3 −0.005) until the
   explicit `gap×vel7` interaction tames it (R4 shock −0.009) — but the interaction is
   marginal (+0.003 pooled) and seed-unstable (fold 11 30.6×); dropped.
5. **Horizon sweep** (`paired_wfcv_velsweep.py`, velocity-only at 3/7/14d). **7d is a
   Goldilocks**: vel3 −0.018/−0.008, **vel7 −0.018/−0.010**, vel14 −0.005/−0.007
   (normal/shock). 3d too jumpy to read the *sustained* multi-week shock move; 14d too
   sluggish for the ~6-wk cycle. MA-smoothing parked (7d point-to-point already beats 3d).

## Conclusion
**`tgp_delta_7d` (7-day TGP momentum) is the candidate.** It clears the log-loss screen
including the shock regime where the #262 money is — unlike the raw gap. Velocity does
the shock work; the raw gap adds only calm-regime log-loss (low economic headroom) at
the cost of shock. Per project discipline the **realised/CpL backtest is the arbiter**,
not log-loss — so no graduation yet.

Notes: PIT = TGP ffilled across weekends + lagged 1 day (published weekday AM; lag is
conservative). hard25 cohort de-emphasised per session feedback; verdict read on
all-population Δ × regime.

## Followups
- **Realised/CpL backtest decision** (the open arbiter): does `tgp_delta_7d` convert to
  spend, and does the raw gap (R3 = gap+vel7) earn its place or is it dead weight in
  calm? Needs TGP plumbed into `PriceHistory` (`tgp_at`/velocity) + `decide()` and the
  feature into `features.py` — production work, follow-up issue. Harness wrinkle:
  `run_paired_realised_backtest` shares `feature_columns` across arms, so an added
  feature is two single-arm runs diffed (or extend the harness).
