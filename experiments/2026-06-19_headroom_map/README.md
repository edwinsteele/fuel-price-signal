# Economic headroom map — model vs perfect-foresight oracle CPL by zone (#262)

- **Date:** 2026-06-19
- **Branch:** main (oracle code landed via PR #263, merged)
- **SHA:** de3934a (oracle); run on the same tree
- **Status:** open — ran 2026-06-19; cycle/regime thread rests, external-data (#215) is the surviving lead

## Hypothesis

#259 disproved the "late descent is a decision-skill soft spot" story: on chosen
(`~emergency`) fills the model pays a **flat ~175 c/L in every cycle regime**
(174.5 / 176.4 / 174.9). The headline regime saving% gradient was an artifact of
(a) emergency-fill dilution and (b) a regime-varying *always-buy* denominator —
**always-buy is the wrong yardstick.**

The one surviving question is generic and zone-agnostic: **is the flat ~175 near
the real floor, or does the model miss deeper troughs anywhere?** Only a
perfect-foresight oracle ceiling can answer it. We expect headroom
(`model_cpl − oracle_cpl`) to be **modest and roughly flat** — if so, the
late-descent / regime thread rests for good. A concentrated hot zone would
reopen it (subject to the caveat below).

## Leaky-ceiling, necessary-not-sufficient caveat

The oracle sees the future, so the gap is **leaky by construction**:

- A non-zero gap proves money **exists** in a zone (**necessary**) — an upper
  bound on recoverable cents.
- It does **not** prove a PIT-safe feature can **capture** it (**not sufficient**).
  The oracle's edge may live entirely in information that isn't available ahead
  of time (the ±1–2 day trough jitter #259 showed is mostly noise).
- Flat-bottom troughs **decouple log-loss from CPL**: sharpening the exact-day
  prediction (better log-loss) can gain ≈0 cents. WFCV log-loss is only a screen;
  any feature chasing a hot zone must still clear the realised arbiter.

So a **tie** in a zone is a hard "stop digging" signal; a **gap** is only
permission to keep looking.

## How to invoke this script

```bash
PYTHONPATH=. uv run python experiments/2026-06-19_headroom_map/headroom_map.py
```

Requires PR #263 merged (or run from its branch) — the script imports
`run_oracle_backtest`.

## Setup

- **Oracle** (`fuel_signal.backtest.run_oracle_backtest`, #263): exact DP over the
  feasible buy/wait sequences → lowest realised CPL under the **same tank
  dynamics** as `run_backtest`. Fit-free, deterministic. Tighter than the issue's
  v1 greedy (global optimum, run-dry paths pruned).
- **Model**: production 54-feat baseline through the #255 realised harness with
  `collect_fills=True`, 14-fold walk-forward, seed 42, isotonic — identical to the
  #259 gate-1 run. `inner_fold_params={"train_min_days":1095,...}` (the fold-1
  inner-OOF gotcha).
- **Same windows/stations/seed/tank** for both passes. Oracle runs over each fold's
  val window; both fill ledgers tagged by zone **at each fill's own date** and
  pooled within zone.
- **Axes (≥3):** cycle `regime` (3-band `cycle_pct_through`: normal <0.6 /
  late_descent 0.6–1.0 / overdue ≥1.0), `quarter` (season), `volatility` (network
  `network_px_std` terciles), plus `fold` (regime-over-time, cheap). Tags are
  network-wide per-date values, so fills join on **date alone** (a station+date
  join spuriously misses ~20% of fills — eval-grid dates a station didn't report).
- **Path-dependency caveat** (per #259): CPL is a tank-path metric, so this is
  "realised CPL conditional on filling in zone X" — a valid existence check, not
  clean causal attribution.

## Results

14-fold walk-forward, seed 42, isotonic, default tank. 574s. Outputs:
`headroom_map.csv`, `headroom_map.png`, `model_fills.parquet`,
`oracle_fills.parquet`. **Overall: model_cpl 189.79 − oracle_cpl 188.14 =
1.66 c/L.** Context: the model already beats always-buy by ≈3.6 c/L (#259), so it
has closed ~⅔ of the always-buy→oracle distance; 1.66 c/L is the residual ceiling.

**Regime axis — FLAT (late-descent thread rests).** Headroom late_descent 1.00 /
normal 1.50 / overdue 1.33. The hypothesised soft spot (late descent) has the
*lowest* headroom — even an oracle barely beats the model there. Independent
confirmation of #259's "flat ~175 chosen-only CPL" via a different construction
(oracle ceiling, not the always-buy yardstick). No money there ⇒ stop digging.

**Volatility axis — the one that lights up, but a HUMP not a ramp.** The
production-script terciles showed a tidy monotonic 0 → 1.96 → 3.59, but that was a
binning artifact (the top tercile, `network_px_std ≥ 11.2c`, merged a sharp peak
with a dead tail). Re-cut on fixed cent bands (post-hoc on the saved ledgers):

| network_px_std band | model fills | headroom c/L |
|---|---|---|
| <8c | 264 | −0.07 |
| 8–12c | 301 | 2.67 |
| **12–16c** | 107 | **7.09** |
| ≥16c | 77 | −0.53 |

Recoverable value concentrates at *elevated-but-not-chaotic* dispersion (12–16c,
~7 c/L), is ~0 in calm markets (<8c, model already on the floor), and is
**unresolvable in the extreme tail** (≥16c, only 77 fills). The two negative cells
are impossible as true per-zone bounds (oracle can't lose window-wide) → they
calibrate the **per-band noise floor at ≈±0.5 c/L**; 2.67 and 7.09 sit clear of
it, ≥16c does not.

**Time — episodes cluster on macro shocks, not on the dispersion metric.** Top
months: 2022-05 (13.2 c/L, post-invasion crude spike, but thin/spiky — 15 fills),
the contiguous **2023-08+09** (5.2, 5.8 — the robust episode, ~43 fills, = fold 8),
2024-03 (4.8), 2023-12 (3.3, vol 15.6c). Monthly buckets are thin/path-dependent →
**monthly noise floor ≈±3 c/L** (e.g. 2025-01 −3.6, 2023-10 −2.9 are pure noise).
Key nuance: the biggest month (2022-05) had only *moderate* `network_px_std` (10c),
so headroom ≠ f(contemporaneous dispersion) alone — there is a macro crude/wholesale
shock component the model's dispersion feature does not carry.

## Conclusion

**Open → the late-descent/regime thread rests for good; the surviving lead is
external wholesale/crude, not a cycle-phase feature.** The model is already near
the perfect-foresight floor (1.66 c/L overall); the only real headroom is ~7 c/L in
the 12–16c dispersion band, concentrated on macro-shock episodes (May-2022,
2023-H2). Signal-from-cycle-features is in diminishing returns.

**Leaky-ceiling caveat governs any next move.** 7 c/L in a band is an *upper bound*
on a prize that is largely unforecastable (the oracle's edge is foresight of how a
volatile period resolves; the model already sees `network_px_std`, so the gap is
"which way it breaks", i.e. the ±jitter #259 showed is noise). Necessary, not
sufficient. Any feature must clear the realised arbiter — WFCV log-loss is only a
screen, and flat-bottom troughs decouple the two.

## Followups

- Supersedes the Gate-2 portion of #259 (closed). #259 thread rests.
- The episodes point at the deferred **external-data hypothesis (#215)**: a
  wholesale/crude lead (Singapore Mogas / terminal gate price) is the natural
  feature family for the macro-shock headroom — the map and that old hunch agree on
  direction. Entry point if pursued: oracle existence check restricted to
  high-vol / shock rows → PIT-safe wholesale-lead proxy → realised arbiter.
- Lib dedup from this experiment's scaffolding tracked in #264 (chore).
