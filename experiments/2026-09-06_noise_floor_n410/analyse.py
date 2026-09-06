"""Grade the 410-station placebo bands against their five-station predecessors.

Phase 1 (fps-ajs, sections 1-5): reads the three committed banks in
experiments/batches/batch1/ and answers the one question fps-nas logged as
`not_tested` -- does the placebo band narrow with universe width the way the
fold-clustered sd did (exactly 2.00x)?

Phase 2 (sections 6-9): adds `noise_floor_n410_k3.json`, the arity-3 410-station
bank. Phase 1's bank is arity 1 and therefore grades NO batch1 candidate (all five
are arity 2 or 3, and `_noise_band` refuses a floor narrower than the run), so every
phase-1 grade of `tgp_cycle_displacement` was an analysis of the ruler, never a
verdict. The arity-3 bank is the first 410-station ruler that is legally admissible
for the candidates, and it is what sections 6-9 grade on.

Run: PYTHONPATH=. uv run python -m experiments.2026-09-06_noise_floor_n410.analyse
(or: PYTHONPATH=. uv run python experiments/2026-09-06_noise_floor_n410/analyse.py)
"""
import json
import pathlib
from collections import Counter

import numpy as np
from scipy import stats

from experiments.pipeline.dossier_tables import (
    _bank_admissibility,
    family_wise_z_threshold,
)

BATCH = pathlib.Path("experiments/batches/batch1")
# fps-nas: tgp_cycle_displacement's pooled delta at each width, from
# experiments/2026-09-05_arbiter_universe_width/README.md.
DELTA_FIVE, DELTA_BROAD = -0.2077, -0.1292


def load(name):
    d = json.loads((BATCH / name).read_text())
    return d, np.array(d["deltas_cpl_held"], dtype=float)


def band(x, n_eff):
    m, s = float(x.mean()), float(x.std(ddof=1))
    z = family_wise_z_threshold(1, n_eff)
    return m, s, z, m - z * s


def main():
    k3, x3 = load("noise_floor.json")
    k1, x1 = load("noise_floor_k1.json")
    w, xw = load("noise_floor_n410.json")
    wk3, xwk3 = load("noise_floor_n410_k3.json")

    axes = ("baseline_fingerprint", "tank_params", "null_method", "n_windows")
    ref = {a: k3.get(a) for a in axes}
    for name, d in [("noise_floor_k1.json", k1), ("noise_floor_n410.json", w),
                    ("noise_floor_n410_k3.json", wk3)]:
        bad = {a: (ref[a], d.get(a)) for a in axes if d.get(a) != ref[a]}
        if bad or d.get("partial"):
            raise SystemExit(f"{name} is not comparable: {bad or 'partial: true'}")

    print("=" * 78)
    print("1. THE FOUR BANKS")
    print("=" * 78)
    rows = [
        ("noise_floor.json", "5", 3, k3, x3),
        ("noise_floor_k1.json", "5", 1, k1, x1),
        ("noise_floor_n410.json", "410", 1, w, xw),
        ("noise_floor_n410_k3.json", "410", 3, wk3, xwk3),
    ]
    print(f"{'bank':<22}{'pop':>5}{'k':>3}{'n':>4}{'mean':>9}{'sd':>8}"
          f"{'relSE':>7}{'t(mean=0)':>11}{'p':>10}")
    for fn, pop, k, d, x in rows:
        n = len(x)
        m, s = x.mean(), x.std(ddof=1)
        t = m / (s / np.sqrt(n))
        p = 2 * stats.t.sf(abs(t), n - 1)
        print(f"{fn:<22}{pop:>5}{k:>3}{n:>4}{m:>+9.4f}{s:>8.4f}"
              f"{1/np.sqrt(2*(n-1)):>7.3f}{t:>+11.2f}{p:>10.2g}")

    print()
    print("=" * 78)
    print("2. DOES THE BAND NARROW? (same arity, k=1, the only clean comparison)")
    print("=" * 78)
    n1, n2 = len(x1), len(xw)
    s1, s2 = x1.std(ddof=1), xw.std(ddof=1)
    F = s1**2 / s2**2
    p = 2 * min(stats.f.sf(F, n1 - 1, n2 - 1), stats.f.cdf(F, n1 - 1, n2 - 1))
    lo = np.sqrt(F / stats.f.ppf(0.975, n1 - 1, n2 - 1))
    hi = np.sqrt(F / stats.f.ppf(0.025, n1 - 1, n2 - 1))
    print(f"  sd ratio (5-stn k=1 / 410-stn k=1) = {s1:.4f}/{s2:.4f} = {s1/s2:.3f}x")
    print(f"  F = {F:.3f}, df = ({n1-1}, {n2-1}), two-sided p = {p:.4f}")
    print(f"  95% CI on the sd ratio: [{lo:.3f}, {hi:.3f}]")
    print(f"    assumed 2.00x inside CI? {lo <= 2.0 <= hi}")
    print(f"    no narrowing (1.00x) inside CI? {lo <= 1.0 <= hi}")
    print(f"  fold-clustered sd narrowed 1.999x (fps-nas). Band point estimate: {s1/s2:.3f}x")
    print()
    print(f"  Confounded comparator the bead nominated (5-stn k=3 -> 410-stn k=1):"
          f" {x3.std(ddof=1)/s2:.3f}x")
    print("    -- mixes arity with width; not evidence about width.")

    print()
    print("=" * 78)
    print("2b. PAIRED ON SOURCE COLUMN (20 columns both k=1 banks drew)")
    print("=" * 78)
    A = {p["source_column"]: d for p, d in zip(k1["placebo_draws"], x1)}
    B = {p["source_column"]: d for p, d in zip(w["placebo_draws"], xw)}
    common = sorted(set(A) & set(B))
    a = np.array([A[c] for c in common])
    b = np.array([B[c] for c in common])
    d = b - a
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"  n = {len(common)} shared source columns (block seeds differ, so the")
    print("  shuffle is NOT held fixed -- column TEXTURE is what is paired).")
    print(f"    5-stn   mean {a.mean():+.4f}  sd {a.std(ddof=1):.4f}")
    print(f"    410-stn mean {b.mean():+.4f}  sd {b.std(ddof=1):.4f}")
    print(f"  paired diff (410 - 5): {d.mean():+.4f} c/L, t = {t:+.2f}, "
          f"p = {2*stats.t.sf(abs(t), len(d)-1):.4f}")
    print(f"  sd ratio on the shared subset alone: {a.std(ddof=1)/b.std(ddof=1):.3f}x "
          f"(full-bank estimate {s1/s2:.3f}x)")
    extra = sorted(set(B) - set(A))
    e = np.array([B[c] for c in extra])
    tt = stats.ttest_ind(b, e, equal_var=False)
    print("  is the mean shift a column-selection artifact? split the 410 bank:")
    print(f"    columns k=1 also drew (n={len(b)}): mean {b.mean():+.4f}")
    print(f"    columns only 410 drew (n={len(e)}): mean {e.mean():+.4f}")
    print(f"    Welch t = {tt.statistic:+.2f}, p = {tt.pvalue:.3f} -> "
          f"{'column-set dependent' if tt.pvalue < 0.05 else 'present in BOTH halves; not an artifact of which columns were drawn'}")

    print()
    print("=" * 78)
    print("3. THE BAR, AND WHAT MOVED IT")
    print("=" * 78)
    m1, sd1, z1, bar1 = band(x1, k1.get("effective_n_draws") or len(x1))
    m2, sd2, z2, bar2 = band(xw, w.get("effective_n_draws") or len(xw))
    print(f"  5-stn  k=1: mean {m1:+.4f}  sd {sd1:.4f}  z_gate {z1:.4f}  bar {bar1:+.4f} c/L")
    print(f"  410-stn k=1: mean {m2:+.4f}  sd {sd2:.4f}  z_gate {z2:.4f}  bar {bar2:+.4f} c/L")
    print(f"  bar moved {bar2-bar1:+.4f} c/L (delta is a COST, so a less-negative bar is EASIER)")
    print("  decomposition, holding one factor at the five-station value:")
    print(f"    mean shift alone : bar {m2 - z1*sd1:+.4f}  ({(m2-z1*sd1)-bar1:+.4f})")
    print(f"    sd  narrow alone : bar {m1 - z2*sd2:+.4f}  ({(m1-z2*sd2)-bar1:+.4f})")

    print()
    print("=" * 78)
    print("4. RE-GRADING tgp_cycle_displacement")
    print("=" * 78)
    print("  NOT a dossier-legal grade: this bank is arity 1, the candidate is arity 2.")
    print("  Reported as an analysis of the ruler, not as a verdict on the feature.")
    for label, delta, m, s, z in [
        ("five  vs 5-stn k=3 (fps-nas, as published)", DELTA_FIVE, x3.mean(), x3.std(ddof=1),
         family_wise_z_threshold(1, 10)),
        ("broad vs 5-stn k=3 (fps-nas, WRONG RULER) ", DELTA_BROAD, x3.mean(), x3.std(ddof=1),
         family_wise_z_threshold(1, 10)),
        ("broad vs 410-stn k=1 (the right width)    ", DELTA_BROAD, m2, sd2, z2),
    ]:
        zc = (delta - m) / s
        print(f"  {label}: z = {zc:+.3f} vs bar {z:.3f} -> "
              f"{'CLEARS' if abs(zc) >= z and zc < 0 else 'fails'}")
    print()
    print("  fps-nas predicted z ~ -3.1 from sd narrowing alone. Decomposing the gain from")
    print("  grading the broad delta on the 5-stn k=1 band to grading it on the 410 band:")

    def zc(m, s):
        return (DELTA_BROAD - m) / s

    z_start, z_end = zc(m1, sd1), zc(m2, sd2)
    z_sd_first, z_mean_first = zc(m1, sd2), zc(m2, sd1)
    print(f"    start  (mean {m1:+.4f}, sd {sd1:.4f}): z = {z_start:+.3f}")
    print(f"    end    (mean {m2:+.4f}, sd {sd2:.4f}): z = {z_end:+.3f}"
          f"   total {abs(z_end)-abs(z_start):+.3f} of |z|")
    nar_a = abs(z_sd_first) - abs(z_start)
    men_a = abs(z_end) - abs(z_sd_first)
    men_b = abs(z_mean_first) - abs(z_start)
    nar_b = abs(z_end) - abs(z_mean_first)
    print(f"    sd first   : narrowing {nar_a:+.3f}, then mean shift {men_a:+.3f}")
    print(f"    mean first : mean shift {men_b:+.3f}, then narrowing {nar_b:+.3f}")
    print("    -> z is a RATIO, so the split is ORDER-DEPENDENT; neither term is dominant.")
    print(f"       Shapley (mean of the two orders): narrowing {(nar_a+nar_b)/2:+.3f}, "
          f"mean shift {(men_a+men_b)/2:+.3f}")
    print("    (The BAR decomposition in section 3 IS exact -- bar = mean - z*sd is additive.)")

    print()
    print("=" * 78)
    print("5. CAN ANY batch1 CANDIDATE BE GRADED AGAINST THIS BANK?")
    print("=" * 78)
    for cand in sorted(pathlib.Path("experiments/candidates/batch1").glob("*/results.json")):
        r = json.loads(cand.read_text())
        meta = dict(r.get("meta", {}))
        arity = len(r.get("candidate", {}).get("columns") or [])
        as_is = _bank_admissibility(
            w, run_meta=meta, run_arity=arity, refuse_identity_mismatch=True,
            refuse_null_method_mismatch=True, refuse_narrow_arity=True,
            refuse_population_mismatch=True)
        meta_matched = dict(meta, station_population=w["station_population"])
        if_rerun = _bank_admissibility(
            w, run_meta=meta_matched, run_arity=arity, refuse_identity_mismatch=True,
            refuse_null_method_mismatch=True, refuse_narrow_arity=True,
            refuse_population_mismatch=True)
        print(f"  {cand.parent.name:<28} arity={arity}  as-committed: refused on "
              f"{as_is.get('axis')!s:<20} if re-run at 410: "
              f"{'OK' if if_rerun['ok'] else 'refused on ' + str(if_rerun.get('axis'))}")

    print()
    print("  draw hygiene of the 410 bank:")
    cols = [p["source_column"] for p in w["placebo_draws"]]
    sc = np.array([abs(p["self_correlation"]) for p in w["placebo_draws"]])
    print(f"    {len(cols)} draws, {len(set(cols))} distinct source columns, "
          f"{len(set(p['block_seed'] for p in w['placebo_draws']))} distinct block seeds")
    print(f"    |self_correlation| max {sc.max():.4f} (cap 0.60), mean {sc.mean():.4f}")
    print(f"    effective_n_draws {w['effective_n_draws']} of {w['n_draws']} nominal")

    print()
    print("=" * 78)
    print("6. PHASE 2 -- THE ARITY-3 410 BANK vs WHAT WAS PREDICTED FOR IT")
    print("=" * 78)
    print("  Predictions were extrapolated from phase 1 before the run, and are printed")
    print("  here so the run can falsify them. They are not findings.")
    m_w3, sd_w3 = xwk3.mean(), xwk3.std(ddof=1)
    ne_w3 = wk3["effective_n_draws"]
    z_w3 = family_wise_z_threshold(1, ne_w3)
    rows = [
        ("sd", 0.0556 * (x3.std(ddof=1) / x1.std(ddof=1)), sd_w3, "0.0556 x the 5-stn k1->k3 ratio"),
        ("effective_n_draws", 28.08, ne_w3, "40 draws x 3 cols from a 49-col pool"),
        ("z_gate", 1.7332, z_w3, "family_wise_z_threshold(1, n_eff)"),
        ("mean", 0.0561, float(m_w3), "assumed the k=1 410 bias carries"),
    ]
    print(f"  {'quantity':<20}{'predicted':>11}{'measured':>11}{'err':>10}  basis")
    for lab, pred, got, basis in rows:
        print(f"  {lab:<20}{pred:>11.4f}{got:>11.4f}{got - pred:>+10.4f}  {basis}")
    print("  -> the two ARITHMETIC predictions (n_eff, z_gate) are exact; both")
    print("     STATISTICAL ones (sd, mean) are wrong, and wrong in opposite directions.")

    print()
    print("=" * 78)
    print("7. THE 2x2: DOES ARITY BEHAVE THE SAME AT BOTH WIDTHS?")
    print("=" * 78)
    cell = {(5, 1): x1, (5, 3): x3, (410, 1): xw, (410, 3): xwk3}
    print("  mean (c/L):")
    print(f"    {'pop':>5}{'k=1':>10}{'k=3':>10}{'k3 - k1':>10}")
    for pop in (5, 410):
        a, b = cell[(pop, 1)], cell[(pop, 3)]
        print(f"    {pop:>5}{a.mean():>+10.4f}{b.mean():>+10.4f}{b.mean() - a.mean():>+10.4f}")
    print(f"    {'410-5':>5}{cell[(410,1)].mean()-cell[(5,1)].mean():>+10.4f}"
          f"{cell[(410,3)].mean()-cell[(5,3)].mean():>+10.4f}")
    inter = ((cell[(410, 3)].mean() - cell[(410, 1)].mean())
             - (cell[(5, 3)].mean() - cell[(5, 1)].mean()))
    se = np.sqrt(sum(v.var(ddof=1) / len(v) for v in cell.values()))
    print(f"    interaction {inter:+.4f}, SE {se:.4f}, z = {inter/se:+.2f}, "
          f"p = {2*stats.norm.sf(abs(inter/se)):.3f}")
    print("  sd (c/L):")
    print(f"    {'pop':>5}{'k=1':>10}{'k=3':>10}{'k3/k1':>10}")
    for pop in (5, 410):
        a, b = cell[(pop, 1)], cell[(pop, 3)]
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        F = sb**2 / sa**2
        lo = np.sqrt(F / stats.f.ppf(0.975, len(b) - 1, len(a) - 1))
        hi = np.sqrt(F / stats.f.ppf(0.025, len(b) - 1, len(a) - 1))
        print(f"    {pop:>5}{sa:>10.4f}{sb:>10.4f}{sb/sa:>9.3f}x  95% CI [{lo:.3f}, {hi:.3f}]")
    for k in (1, 3):
        a, b = cell[(5, k)], cell[(410, k)]
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        F = sa**2 / sb**2
        lo = np.sqrt(F / stats.f.ppf(0.975, len(a) - 1, len(b) - 1))
        hi = np.sqrt(F / stats.f.ppf(0.025, len(a) - 1, len(b) - 1))
        pF = 2 * min(stats.f.sf(F, len(a) - 1, len(b) - 1),
                     stats.f.cdf(F, len(a) - 1, len(b) - 1))
        print(f"    width narrowing at k={k}: {sa/sb:.3f}x, p = {pF:.3f}, "
              f"95% CI [{lo:.3f}, {hi:.3f}]")
    print("  is the 410 k=3 mean above zero? (the phase-1 headline, re-asked at arity 3)")
    for lab, n in [("nominal n = 40", float(len(xwk3))), (f"effective n = {ne_w3:.3f}", ne_w3)]:
        t = m_w3 / (sd_w3 / np.sqrt(n))
        print(f"    {lab:<24} t = {t:+.3f}, p = {2*stats.t.sf(abs(t), n-1):.4f}")
    tt = stats.ttest_ind(xw, xwk3, equal_var=False)
    print(f"    410 k=1 vs k=3 mean: {xw.mean():+.4f} vs {m_w3:+.4f}, "
          f"Welch t = {tt.statistic:+.2f}, p = {tt.pvalue:.3f}")
    tt = stats.ttest_ind(xwk3, x3, equal_var=False)
    print(f"    410 k=3 vs 5-stn k=3: {m_w3:+.4f} vs {x3.mean():+.4f}, "
          f"Welch t = {tt.statistic:+.2f}, p = {tt.pvalue:.3f}")
    a, b = xwk3[:20], xwk3[20:]
    tt = stats.ttest_ind(a, b, equal_var=False)
    print(f"    bank halves: first20 {a.mean():+.4f} vs last20 {b.mean():+.4f}, "
          f"Welch t = {tt.statistic:+.2f}, p = {tt.pvalue:.3f}")

    print("  the bar, at the arity that actually grades candidates (k=3):")
    print(f"    {'bank':<24}{'mean':>9}{'sd':>9}{'z_gate':>9}{'bar':>10}")
    for lab, d, v in [("5-stn k=3 (canonical)", k3, x3), ("410 k=3", wk3, xwk3),
                      ("410 k=1 (phase 1)", w, xw)]:
        n = d.get("effective_n_draws") or len(v)
        m, sv = v.mean(), v.std(ddof=1)
        zg = family_wise_z_threshold(1, n)
        print(f"    {lab:<24}{m:>+9.4f}{sv:>9.4f}{zg:>9.4f}{m - zg*sv:>+10.4f}")
    m_a, s_a, z_a = x3.mean(), x3.std(ddof=1), family_wise_z_threshold(1, 10)
    bar_a, bar_b = m_a - z_a * s_a, m_w3 - z_w3 * sd_w3
    print(f"    5-stn k=3 -> 410 k=3: bar moves {bar_b - bar_a:+.4f} c/L "
          f"(a less-negative bar is EASIER); exact additive split:")
    print(f"      mean shift      {m_w3 - m_a:+.4f}")
    print(f"      sd narrowing    {z_a*s_a - z_a*sd_w3:+.4f}  (at the 5-stn z_gate)")
    print(f"      more draws      {z_a*sd_w3 - z_w3*sd_w3:+.4f}  "
          f"(z_gate {z_a:.4f} at 10 draws -> {z_w3:.4f} at {ne_w3:.2f})")
    print("      -> the DRAW COUNT, not the population width, is the largest term.")

    print()
    print("=" * 78)
    print("8. THE FIRST LEGAL 410-STATION GRADE OF tgp_cycle_displacement")
    print("=" * 78)
    print("  The candidate is arity 2. An arity-3 bank is admissible (a floor grades any")
    print("  run of arity <= its own) but CONSERVATIVE: sd grows with arity, so the band")
    print("  is wider than a matched arity-2 ruler would be.")
    for lab, delta, m, sdv, zg, legal in [
        ("broad vs 410 k=3  (LEGAL, conservative)", DELTA_BROAD, m_w3, sd_w3, z_w3, True),
        ("broad vs 410 k=1  (phase 1; arity-refused)", DELTA_BROAD, xw.mean(),
         xw.std(ddof=1), family_wise_z_threshold(1, w.get("effective_n_draws") or len(xw)), False),
        ("broad vs 5-stn k=3 (fps-nas; wrong width)", DELTA_BROAD, x3.mean(),
         x3.std(ddof=1), family_wise_z_threshold(1, 10), False),
        ("five  vs 5-stn k=3 (as published)", DELTA_FIVE, x3.mean(),
         x3.std(ddof=1), family_wise_z_threshold(1, 10), True),
    ]:
        zc = (delta - m) / sdv
        print(f"  {lab:<42} z = {zc:+.3f} vs bar {zg:.3f} -> "
              f"{'CLEARS' if zc < 0 and abs(zc) >= zg else 'FAILS ':<6}"
              f"{'' if legal else '  [not a legal grade]'}")
    print(f"  predicted for this run: z ~ -3.2. Measured: {(DELTA_BROAD - m_w3)/sd_w3:+.3f}.")
    print("  The prediction was made against the arity-1 bank's sd and does not survive")
    print("  correcting the arity. The verdict flips with the ruler's arity, not the data.")
    print("  how close is the fail?")
    need = abs(DELTA_BROAD - m_w3) / z_w3
    print(f"    at the stamped bar it needs sd <= {need:.4f}; the bank has {sd_w3:.4f} "
          f"({sd_w3/need - 1:+.1%})")
    print(f"    equivalently the delta would have to reach {m_w3 - z_w3*sd_w3:+.4f} c/L "
          f"(it is {DELTA_BROAD:+.4f})")
    s_k2 = 0.5 * (xw.std(ddof=1) + sd_w3)
    print(f"    an arity-2 bank -- the MATCHED ruler, not run -- interpolates to sd ~ "
          f"{s_k2:.4f};")
    print(f"    at that sd and this mean the grade would be z = "
          f"{(DELTA_BROAD - m_w3)/s_k2:+.3f}, i.e. it could flip. Not measured.")

    print()
    print("=" * 78)
    print("9. WHAT THE ARITY-3 410 BANK UNLOCKS, AND ITS DRAW HYGIENE")
    print("=" * 78)
    for cand in sorted(pathlib.Path("experiments/candidates/batch1").glob("*/results.json")):
        r = json.loads(cand.read_text())
        meta = dict(r.get("meta", {}))
        arity = len(r.get("candidate", {}).get("columns") or [])
        as_is = _bank_admissibility(
            wk3, run_meta=meta, run_arity=arity, refuse_identity_mismatch=True,
            refuse_null_method_mismatch=True, refuse_narrow_arity=True,
            refuse_population_mismatch=True)
        if_rerun = _bank_admissibility(
            wk3, run_meta=dict(meta, station_population=wk3["station_population"]),
            run_arity=arity, refuse_identity_mismatch=True,
            refuse_null_method_mismatch=True, refuse_narrow_arity=True,
            refuse_population_mismatch=True)
        print(f"  {cand.parent.name:<28} arity={arity}  as-committed: refused on "
              f"{as_is.get('axis')!s:<20} if re-run at 410: "
              f"{'OK' if if_rerun['ok'] else 'refused on ' + str(if_rerun.get('axis'))}")
    draws = wk3["placebo_draws"]
    by_draw = {}
    for d in draws:
        by_draw.setdefault(d["draw"], []).append(d["source_column"])
    sc = np.array([abs(d["self_correlation"]) for d in draws])
    reuse = Counter(d["source_column"] for d in draws)
    print(f"    {len(by_draw)} draws x arity {sorted({len(v) for v in by_draw.values()})}"
          f" = {len(draws)} column-slots, {len(reuse)} distinct source columns of "
          f"{wk3['n_baseline_columns_available']} available")
    print(f"    per-column reuse min {min(reuse.values())} max {max(reuse.values())}; "
          f"duplicate column inside a draw: "
          f"{any(len(set(v)) < len(v) for v in by_draw.values())}")
    print(f"    {len({d['block_seed'] for d in draws})} distinct block seeds; "
          f"|self_correlation| max {sc.max():.4f} (cap 0.60), mean {sc.mean():.4f}")
    print(f"    effective_n_draws {ne_w3:.3f} of {wk3['n_draws']} nominal -- "
          f"reuse costs {wk3['n_draws'] - ne_w3:.2f} draws, and the bar pays for it "
          f"(z_gate {z_w3:.4f} vs {family_wise_z_threshold(1, 40):.4f} at 40)")


if __name__ == "__main__":
    main()
