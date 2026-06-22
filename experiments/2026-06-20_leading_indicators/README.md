# Leading indicators of Sydney E10 — AIP TGP exploration

- **Date:** 2026-06-20
- **Branch:** main
- **SHA:** 70be117 (baseline 54-feat)
- **Status:** done — `tgp_delta_7d` clears the realised/CpL arbiter (ships alone; production path = #271); the raw gap does NOT graduate.

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

6. **Realised/CpL arbiter** (`realised_tgp.py` → `*_realised.csv`, `run_realised.log`).
   The decision step: in-process injection of the candidates via the #269
   `ArmSpec.extra_feature_provider` seam (no production plumbing yet), 14 walk-forward
   folds Nov 2021→Apr 2025, three arms (baseline / vel7 / gap_vel7). **All three converge
   on τ=0.25** (= prod lock) on every fold → `own_tau == held_tau`, so there is **no
   operating-point contribution to net out** (`tau_diverges=False` everywhere); the whole
   Δ is pure feature effect. Pooled: baseline 189.79 c/L, **vel7 −0.039 c/L**, gap_vel7
   −0.136 c/L.

## Conclusion
**`tgp_delta_7d` (7-day TGP momentum) GRADUATES; the raw gap does NOT.** Production path
filed as **#271** (AIP TGP downloader/storage + `features.py` compute + `PriceHistory`
native source + `FEATURE_COLUMNS` bump) — ship `tgp_delta_7d` alone.

- **vel7 → graduates.** Pooled Δcpl = **−0.039 c/L**, narrow but right-shaped: the gains
  concentrate in the *expensive* folds (8 −0.51, 9 −0.30, 10 −0.67; all ~200+ c/L always-buy
  baselines), exactly where the #262 headroom lives. Log-loss screen and realised arbiter
  agree on direction.
- **raw gap → does NOT graduate.** gap_vel7 pools *better* (−0.136) but gets there by the
  **wrong trade**: it rescues low-headroom calm/falling folds (5 −0.13, 13 −0.28) while
  **zeroing out vel7's biggest gains in the expensive folds** (9 and 10 both → 0.000; 8
  cut from −0.51 to −0.21). It sells high-prize shock gains for low-prize calm gains.

**Why folds 9/10 read as exactly equal-and-opposite (zero gap_vel7 Δ).** Probed on the
calibrated decisions: it is *not* a bug. Across each fold's eval-date×station grid almost
every probability sits clearly on one side of τ for all three arms — they agree without
being close. Fold 10's entire vel7 −0.667 comes from a **single decision flip** (station
585, 21 Feb 2024 @ 189.9: baseline p=0.218 WAIT, vel7 p=0.292 BUY, gap_vel7 p=0.176 WAIT).
The raw gap there (6.3 c/L) pulls the probability back under τ and erases the one flip → gap_vel7
spend-identical to baseline. The exact-match is one cancelled marginal decision, not noise.

**Mechanism (complementary, not redundant).** `tgp_delta_7d` reads the *direction the floor
is moving*; `station_minus_tgp_cents` reads *how close you are to the floor*. When TGP is
rising fast (folds 9/10) vel7 correctly times a pre-spike buy, but the gap's "still far above
floor" brake cancels it — destroying value in the high-headroom regime. When TGP is
falling/flat (folds 5/13) vel7 turns over-pessimistic and misses genuine troughs; the gap's
floor-proximity signal rescues them — but in low-headroom periods. The pooled win is bought
in the cheap regime at the cost of the expensive one. **The TGP-above-retail inversion of
Sep–Oct 2022** (visible in `tgp_vs_e10_overlay.png`: crude fell fast post-2022-spike, lagging
refinery-gate TGP didn't, retailers sold *below* wholesale) is what trained the model to read
near-zero/negative gap as deepest-trough → explains the fold-5 gap rescue.

Notes: PIT = TGP ffilled across weekends + lagged 1 day (published weekday AM; lag is
conservative); live replay uses `.asof` (latest value ≤ decision date) to match
`PriceHistory.avg_price_at` on-or-before semantics. hard25 cohort de-emphasised per session
feedback; verdict read on all-population pooled CpL × regime.

## Followups
- **#271** — production path for `tgp_delta_7d` (the graduation work).
- **#270** — `experiments/lib` glue hoist for the external-series provider (gated on 2nd use).
