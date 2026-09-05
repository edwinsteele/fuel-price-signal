"""fps-ajs: grade the 410-station placebo band against its five-station predecessors.

Reads the three committed banks in experiments/batches/batch1/ and answers the one
question fps-nas logged as `not_tested`: does the placebo band narrow with universe
width the way the fold-clustered sd did (exactly 2.00x)?

Run: PYTHONPATH=. uv run python -m experiments.2026-09-06_noise_floor_n410.analyse
(or: PYTHONPATH=. uv run python experiments/2026-09-06_noise_floor_n410/analyse.py)
"""
import json
import pathlib

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

    print("=" * 78)
    print("1. THE THREE BANKS")
    print("=" * 78)
    rows = [
        ("noise_floor.json", "5", 3, k3, x3),
        ("noise_floor_k1.json", "5", 1, k1, x1),
        ("noise_floor_n410.json", "410", 1, w, xw),
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


if __name__ == "__main__":
    main()
