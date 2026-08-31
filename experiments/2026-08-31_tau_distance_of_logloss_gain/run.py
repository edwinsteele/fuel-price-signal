"""Where does tgp_cycle_displacement's WFCV log-loss gain sit relative to the
decision threshold the realised arbiter actually applies?  (bd fps-e6i)

Pure lookup over committed artifacts — no model is fit here.  Two inputs:

  * ``experiments/candidates/batch1/tgp_cycle_displacement/rowpreds.parquet``
    — the WFCV screen's per-row RAW probabilities, R0 and candidate, 5 seeds.
  * ``experiments/batches/batch1/r0_cache.joblib`` — the realised arbiter's own
    per-fold baseline fit: the isotonic calibrator and the OOF-selected tau.

The two are the SAME MODEL on the SAME ROWS (asserted below, not assumed), so the
cache's isotonic map converts the screen's raw probabilities into the calibrated
scale the decision rule reads, exactly.

Sign convention (experiments/lib/gates.py): delta = candidate - R0; negative is better.
"""
from __future__ import annotations

import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

from experiments.lib.constants import BASELINE_COLUMNS
from experiments.lib.fit import per_row_log_loss
from fuel_signal import evaluate as _ev
from fuel_signal.features import load_features

REPO = pathlib.Path("/Users/esteele/Code/fuel-price-signal")
BATCH = REPO / "experiments/batches/batch1"
CAND = REPO / "experiments/candidates/batch1/tgp_cycle_displacement"
OUT = pathlib.Path(__file__).parent

PREFERRED = [261, 414, 429, 585, 18517]  # the arbiter's replay universe
TAU = 0.25
REALISED_SEED = 42


# --------------------------------------------------------------------------
# Isotonic geometry
# --------------------------------------------------------------------------
def iso_levels(iso) -> list[tuple[float, float, float]]:
    """Maximal flat runs of the fitted isotonic map as ``(x_start, x_end, y)``.

    sklearn's IsotonicRegression stores breakpoints and interpolates linearly
    between them, so a "step" is a maximal interval over which y_thresholds_ is
    constant; the intervals between two flat runs are the ramps.
    """
    x = np.asarray(iso.X_thresholds_, dtype=float)
    y = np.asarray(iso.y_thresholds_, dtype=float)
    runs: list[tuple[float, float, float]] = []
    i = 0
    while i < len(y):
        j = i
        while j + 1 < len(y) and y[j + 1] == y[i]:
            j += 1
        runs.append((x[i], x[j], y[i]))
        i = j + 1
    return runs


def tau_raw(iso, tau: float = TAU) -> float:
    """The raw-probability threshold equivalent to calibrated ``tau``.

    Isotonic is monotone, so ``p_cal >= tau``  <=>  ``p_raw >= tau_raw``: the whole
    diagnostic can be done in raw space once this point is located.
    """
    x = np.asarray(iso.X_thresholds_, dtype=float)
    y = np.clip(np.asarray(iso.y_thresholds_, dtype=float), 0.0, 1.0)
    if tau <= y[0]:
        return float(x[0])
    if tau > y[-1]:
        return float(x[-1])
    k = int(np.searchsorted(y, tau, side="left"))
    if y[k] == y[k - 1]:
        return float(x[k])
    frac = (tau - y[k - 1]) / (y[k] - y[k - 1])
    return float(x[k - 1] + frac * (x[k] - x[k - 1]))


def one_step_band(iso, tau: float = TAU) -> tuple[float, float]:
    """Raw-probability band spanning one isotonic level either side of ``tau``.

    A row inside it is at most one step of calibration resolution away from being
    on the other side of the threshold — the widest reading of "near tau" that the
    calibrator's own granularity supports.
    """
    xt = tau_raw(iso, tau)
    runs = iso_levels(iso)
    below = [r for r in runs if r[1] <= xt]
    above = [r for r in runs if r[0] >= xt]
    lo = below[-1][0] if below else runs[0][0]
    hi = above[0][1] if above else runs[-1][1]
    return float(lo), float(hi)


# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("tau-distance of the log-loss gain — tgp_cycle_displacement (fps-e6i)")
    print("=" * 78)

    cache = joblib.load(BATCH / "r0_cache.joblib")
    rp = pd.read_parquet(CAND / "rowpreds.parquet")
    rp["price_date"] = pd.to_datetime(rp["price_date"])
    folds = sorted(rp.fold.unique())
    seeds = sorted(rp.seed.unique())
    print(f"\nscreen: {len(rp):,} rows | folds {folds[0]}-{folds[-1]} | seeds {seeds}")
    print(f"arbiter replay universe: {PREFERRED} "
          f"({rp.station_code.nunique()} stations scored by the screen)")

    # ---- 0. the two artifacts are the same fit -----------------------------
    print("\n--- 0. Verify the screen's R0 fit IS the arbiter's baseline fit ---")
    feats = load_features(BATCH / "features.csv")
    worst = 0.0
    r0s = rp[(rp.run == "R0") & (rp.seed == REALISED_SEED)]
    for i, (_train, val) in enumerate(_ev.walk_forward_folds(feats), start=1):
        if val.empty or i not in (1, 7, 14):
            continue
        p_arb = cache.per_fold[i]["cal_pipe"].base_pipeline.predict_proba(
            val[BASELINE_COLUMNS])[:, 1]
        key = pd.DataFrame({
            "station_code": val["station_code"].to_numpy(),
            "price_date": pd.to_datetime(val["price_date"]).to_numpy(),
            "p_arb": p_arb,
        })
        m = r0s[r0s.fold == i].merge(key, on=["station_code", "price_date"],
                                     validate="1:1")
        worst = max(worst, float(np.abs(m.proba - m.p_arb).max()))
    print(f"max |screen R0 proba - arbiter base proba| over folds 1/7/14: {worst:.3e}")
    assert worst < 1e-6, "screen and arbiter are NOT the same fit — analysis invalid"
    print("-> identical to float32 precision. The cache's isotonic map applies exactly.")

    # ---- 1. the decision boundary, in raw space ---------------------------
    print("\n--- 1. Where the calibrated tau=0.25 rule actually sits in raw space ---")
    geom = {}
    rows = []
    for f in folds:
        iso = cache.per_fold[f]["cal_pipe"].calibrator
        own = float(cache.per_fold[f]["own_tau"])
        xt = tau_raw(iso, TAU)
        lo, hi = one_step_band(iso, TAU)
        geom[f] = {"own_tau": own, "tau_raw": xt, "band_lo": lo, "band_hi": hi,
                   "n_levels": len(iso_levels(iso))}
        rows.append({"fold": f, "own_tau": own, "tau_raw": xt,
                     "band_lo": lo, "band_hi": hi, "band_width": hi - lo,
                     "n_iso_levels": geom[f]["n_levels"]})
    gdf = pd.DataFrame(rows)
    print(gdf.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\ntau (calibrated) is {TAU} on {(gdf.own_tau == TAU).sum()}/{len(gdf)} folds.")
    print(f"tau_raw spans [{gdf.tau_raw.min():.4f}, {gdf.tau_raw.max():.4f}] "
          f"-- NOT 0.25. Measuring |p_raw - 0.25| would target the wrong point.")

    # ---- 2. per-row deltas -------------------------------------------------
    print("\n--- 2. Per-row probability and log-loss deltas ---")
    idx = ["fold", "station_code", "price_date"]
    wide = rp.pivot_table(index=idx + ["seed"], columns="run", values="proba")
    lab = rp.drop_duplicates(idx).set_index(idx)["label"].astype(int)
    w = wide.reset_index().set_index(idx)
    w["label"] = lab
    w["ll_R0"] = per_row_log_loss(w["label"].to_numpy(), w["R0"].to_numpy(float))
    w["ll_cand"] = per_row_log_loss(w["label"].to_numpy(), w["candidate"].to_numpy(float))
    w["d_ll"] = w["ll_cand"] - w["ll_R0"]
    w["d_p"] = w["candidate"] - w["R0"]

    def summarise(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
        """Collapse the 5 seeds to one row per (fold, station, date)."""
        g = frame.groupby(level=idx).agg(
            R0=("R0", "mean"), cand=("candidate", "mean"),
            d_ll=("d_ll", "mean"), d_p=("d_p", "mean"), label=("label", "first"))
        print(f"  [{tag}] {len(g):,} rows  sum d_ll = {g.d_ll.sum():+.4f} nats  "
              f"mean = {g.d_ll.mean():+.3e}")
        return g

    a = summarise(w, "seed-mean")
    one = w.reset_index()
    one = one[one.seed == REALISED_SEED].set_index(idx)
    b = one.groupby(level=idx).agg(
        R0=("R0", "mean"), cand=("candidate", "mean"),
        d_ll=("d_ll", "mean"), d_p=("d_p", "mean"), label=("label", "first"))
    print(f"  [seed {REALISED_SEED}] {len(b):,} rows  sum d_ll = {b.d_ll.sum():+.4f} nats")

    results: dict = {"geometry": rows, "arms": {}}

    # ---- 3. seed stability of the headline numbers ------------------------
    print("\n--- 3. Seed-by-seed stability of the three headline shares ---")
    print(f"{'seed':>6}{'sum d_ll':>12}{'pref-5 gain %':>15}{'one-step gain %':>17}"
          f"{'crossings %':>13}{'pref-5 crossings':>18}")
    per_seed = []
    ws = w.reset_index()
    for sd in seeds:
        gs = ws[ws.seed == sd].copy()
        gs["tau_raw"] = gs.fold.map(lambda f: geom[f]["tau_raw"])
        gs["lo"] = gs.fold.map(lambda f: geom[f]["band_lo"])
        gs["hi"] = gs.fold.map(lambda f: geom[f]["band_hi"])
        tot = float(gs.d_ll.sum())
        pref = gs.station_code.isin(PREFERRED)
        band = (gs.R0 >= gs.lo) & (gs.R0 <= gs.hi)
        cross = (gs.R0 >= gs.tau_raw) != (gs["candidate"] >= gs.tau_raw)
        rec = {"seed": int(sd), "sum_d_ll": tot,
               "pref_gain_share": float(gs.loc[pref, "d_ll"].sum() / tot),
               "one_step_gain_share": float(gs.loc[band, "d_ll"].sum() / tot),
               "crossing_share": float(cross.mean()),
               "n_pref_crossings": int((cross & pref).sum())}
        per_seed.append(rec)
        print(f"{sd:>6}{tot:>12.1f}{100*rec['pref_gain_share']:>14.2f}%"
              f"{100*rec['one_step_gain_share']:>16.2f}%"
              f"{100*rec['crossing_share']:>12.2f}%{rec['n_pref_crossings']:>18}")
    ps = pd.DataFrame(per_seed)
    print(f"{'sd':>6}{ps.sum_d_ll.std(ddof=1):>12.1f}"
          f"{100*ps.pref_gain_share.std(ddof=1):>14.2f}%"
          f"{100*ps.one_step_gain_share.std(ddof=1):>16.2f}%"
          f"{100*ps.crossing_share.std(ddof=1):>12.2f}%"
          f"{ps.n_pref_crossings.std(ddof=1):>18.1f}")
    results["per_seed"] = per_seed


    for tag, g in (("seed_mean", a), (f"seed{REALISED_SEED}", b)):
        results["arms"][tag] = analyse(g, geom, tag)

    crux(w, idx, seeds, results)

    plots(w, geom, results)

    (OUT / "facts.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT / 'facts.json'}")


def analyse(g: pd.DataFrame, geom: dict, tag: str) -> dict:
    print("\n" + "=" * 78)
    print(f"ANALYSIS [{tag}]")
    print("=" * 78)
    gg = g.reset_index()
    gg["tau_raw"] = gg.fold.map(lambda f: geom[f]["tau_raw"])
    gg["band_lo"] = gg.fold.map(lambda f: geom[f]["band_lo"])
    gg["band_hi"] = gg.fold.map(lambda f: geom[f]["band_hi"])
    gg["dist_R0"] = (gg.R0 - gg.tau_raw).abs()
    gg["dist_cand"] = (gg.cand - gg.tau_raw).abs()
    gg["abs_dp"] = gg.d_p.abs()
    gg["pref"] = gg.station_code.isin(PREFERRED)
    total = float(gg.d_ll.sum())
    out: dict = {"total_d_ll_nats": total, "n_rows": int(len(gg))}

    # --- population channel ------------------------------------------------
    print(f"\n[A] POPULATION CHANNEL — the arbiter only acts at {len(PREFERRED)} stations")
    print(f"{'group':<16}{'rows':>10}{'row %':>9}{'sum d_ll':>12}{'% of gain':>11}"
          f"{'d_ll/row':>12}")
    for name, mask in (("preferred 5", gg.pref), ("other 709", ~gg.pref)):
        sub = gg[mask]
        s = float(sub.d_ll.sum())
        print(f"{name:<16}{len(sub):>10,}{100*len(sub)/len(gg):>8.2f}%{s:>+12.4f}"
              f"{100*s/total:>10.1f}%{sub.d_ll.mean():>+12.3e}")
        out[f"pop_{name.split()[0]}"] = {
            "n": int(len(sub)), "row_share": len(sub) / len(gg),
            "sum_d_ll": s, "gain_share": s / total,
            "mean_d_ll": float(sub.d_ll.mean())}

    # --- tau-distance profile ---------------------------------------------
    print("\n[B] TAU-DISTANCE PROFILE — binned by |p_raw(R0) - tau_raw|, all rows")
    edges = [0, .01, .02, .05, .10, .20, .40, 1.0]
    gg["dbin"] = pd.cut(gg.dist_R0, bins=edges, include_lowest=True)
    print(f"{'|p-tau_raw|':<16}{'rows':>10}{'row %':>9}{'sum d_ll':>12}{'% of gain':>11}"
          f"{'mean|dp|':>11}")
    prof = []
    for iv, sub in gg.groupby("dbin", observed=True):
        s = float(sub.d_ll.sum())
        print(f"{str(iv):<16}{len(sub):>10,}{100*len(sub)/len(gg):>8.2f}%{s:>+12.4f}"
              f"{100*s/total:>10.1f}%{sub.abs_dp.mean():>11.4f}")
        prof.append({"bin": str(iv), "n": int(len(sub)), "row_share": len(sub)/len(gg),
                     "sum_d_ll": s, "gain_share": s/total,
                     "mean_abs_dp": float(sub.abs_dp.mean())})
    out["tau_distance_profile"] = prof

    # --- one isotonic step -------------------------------------------------
    print("\n[C] ONE ISOTONIC STEP OF TAU — the acceptance criterion")
    inband = (gg.R0 >= gg.band_lo) & (gg.R0 <= gg.band_hi)
    s = float(gg.loc[inband, "d_ll"].sum())
    print(f"  rows inside one isotonic step of tau : {inband.sum():,} "
          f"({100*inband.mean():.2f}% of rows)")
    print(f"  their share of the total log-loss gain: {100*s/total:.1f}%  "
          f"({s:+.4f} of {total:+.4f} nats)")
    print(f"  concentration ratio (gain% / row%)    : {(s/total)/inband.mean():.2f}x")
    out["one_isotonic_step"] = {"n": int(inband.sum()), "row_share": float(inband.mean()),
                                "sum_d_ll": s, "gain_share": s/total}
    pin = inband & gg.pref
    print(f"  ... and at the preferred 5           : {pin.sum():,} rows, "
          f"{float(gg.loc[pin,'d_ll'].sum()):+.4f} nats "
          f"({100*float(gg.loc[pin,'d_ll'].sum())/total:.2f}% of gain)")
    out["one_isotonic_step_preferred"] = {
        "n": int(pin.sum()), "sum_d_ll": float(gg.loc[pin, "d_ll"].sum()),
        "gain_share": float(gg.loc[pin, "d_ll"].sum()) / total}

    # --- threshold crossings ----------------------------------------------
    print("\n[D] THRESHOLD CROSSINGS — rows where the two arms would DECIDE differently")
    cross = (gg.R0 >= gg.tau_raw) != (gg.cand >= gg.tau_raw)
    s = float(gg.loc[cross, "d_ll"].sum())
    print(f"  all 714 stations : {cross.sum():,} rows ({100*cross.mean():.3f}%), "
          f"{s:+.4f} nats ({100*s/total:.1f}% of gain)")
    cp = cross & gg.pref
    sp = float(gg.loc[cp, "d_ll"].sum())
    print(f"  preferred 5      : {cp.sum():,} rows "
          f"({100*cp.sum()/max(gg.pref.sum(),1):.3f}% of their {gg.pref.sum():,}), "
          f"{sp:+.4f} nats ({100*sp/total:.2f}% of gain)")
    out["crossings"] = {"n_all": int(cross.sum()), "share_all": float(cross.mean()),
                        "sum_d_ll": s, "gain_share": s/total,
                        "n_preferred": int(cp.sum()), "sum_d_ll_preferred": sp}

    # robustness: the candidate arm's own tau_raw is unknown (its calibrator was
    # not cached), so sweep the ruler rather than trusting one point.
    print("\n  robustness — crossing count as tau_raw is swept +/-30% (all stations):")
    sweep = []
    for mult in (0.7, 0.85, 1.0, 1.15, 1.3):
        c = (gg.R0 >= gg.tau_raw * mult) != (gg.cand >= gg.tau_raw * mult)
        print(f"    x{mult:<5} {c.sum():>7,} rows ({100*c.mean():.3f}%)  "
              f"gain share {100*float(gg.loc[c,'d_ll'].sum())/total:>6.1f}%")
        sweep.append({"mult": mult, "n": int(c.sum()), "share": float(c.mean()),
                      "gain_share": float(gg.loc[c, "d_ll"].sum())/total})
    out["crossing_tau_sweep"] = sweep

    # --- AC1: where the biggest movers live -------------------------------
    print("\n[E] WHERE THE BIGGEST PROBABILITY MOVERS LIVE (acceptance criterion 1)")
    print(f"{'cohort':<22}{'n':>9}{'med |p-tau|':>13}{'med |p-tau|':>13}"
          f"{'P10 dist':>10}{'in band':>10}{'% of gain':>11}")
    print(f"{'':<22}{'':>9}{'R0':>13}{'cand':>13}{'R0':>10}{'R0':>10}{'':>11}")
    cohorts = [("all rows", gg)]
    for q, nm in ((0.90, "top 10% by |dp|"), (0.99, "top 1% by |dp|"),
                  (0.999, "top 0.1% by |dp|")):
        thr = gg.abs_dp.quantile(q)
        cohorts.append((nm, gg[gg.abs_dp >= thr]))
    ac1 = []
    for nm, sub in cohorts:
        ib = ((sub.R0 >= sub.band_lo) & (sub.R0 <= sub.band_hi)).mean()
        s = float(sub.d_ll.sum())
        print(f"{nm:<22}{len(sub):>9,}{sub.dist_R0.median():>13.4f}"
              f"{sub.dist_cand.median():>13.4f}{sub.dist_R0.quantile(.10):>10.4f}"
              f"{100*ib:>9.2f}%{100*s/total:>10.1f}%")
        ac1.append({"cohort": nm, "n": int(len(sub)),
                    "median_dist_R0": float(sub.dist_R0.median()),
                    "median_dist_cand": float(sub.dist_cand.median()),
                    "p10_dist_R0": float(sub.dist_R0.quantile(.10)),
                    "in_band_share": float(ib), "gain_share": s/total})
    out["biggest_movers"] = ac1

    # --- F: is the preferred-5 gain even resolvable on its own? ------------
    print("\n[F] POWER — per-row gain at the arbiter's stations vs the rest")
    print("    (fold-clustered: mean of 14 per-fold means, SE across folds)")
    print(f"{'group':<16}{'folds':>7}{'mean d_ll/row':>16}{'2.SE':>12}{'resolvable':>12}")
    out["power"] = {}
    for name, mask in (("preferred 5", gg.pref), ("other 709", ~gg.pref)):
        fm = gg[mask].groupby("fold").d_ll.mean()
        m = float(fm.mean()); se = float(fm.std(ddof=1) / np.sqrt(len(fm)))
        print(f"{name:<16}{len(fm):>7}{m:>+16.3e}{2*se:>12.3e}"
              f"{('yes' if abs(m) > 2*se else 'NO'):>12}")
        out["power"][name.split()[0]] = {"mean": m, "two_se": 2*se,
                                         "resolvable": bool(abs(m) > 2*se)}
    pm = out["power"]["preferred"]["mean"]; om = out["power"]["other"]["mean"]
    pse = out["power"]["preferred"]["two_se"] / 2
    print(f"    preferred-5 mean sits {abs(pm - om)/pse:.2f} SE from the network-wide "
          f"mean -> {'DIFFERENT' if abs(pm-om)/pse > 2 else 'no evidence the 5 differ'}")
    out["power"]["z_pref_vs_other"] = float(abs(pm - om) / pse)
    return out


def plots(w: pd.DataFrame, geom: dict, results: dict) -> None:
    """Three figures: the crux, the tau-distance profile, and where tau really sits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = pd.read_csv(OUT / "per_fold_screen_vs_arbiter.csv")

    # --- 1. the crux -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    for ax, col, title in (
            (axes[0], "screen_all", "screen scored on ALL 714 stations"),
            (axes[1], "screen_pref5", "screen scored on the ARBITER'S 5 stations")):
        x, y = tab[col].to_numpy(), tab.delta_cpl_own.to_numpy()
        cols = ["#c44e52" if r == "shock" else "#4c72b0" for r in tab.regime]
        ax.scatter(x, y, c=cols, s=70, zorder=3, edgecolor="white", linewidth=0.8)
        for xi, yi, f in zip(x, y, tab.fold):
            ax.annotate(str(f), (xi, yi), fontsize=7.5, color="#444",
                        xytext=(4, 4), textcoords="offset points")
        b, a0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b * xs + a0, color="#333", lw=1.6, ls="--", zorder=2)
        ax.axhline(0, color="#bbb", lw=0.8); ax.axvline(0, color="#bbb", lw=0.8)
        r = float(np.corrcoef(x, y)[0, 1])
        agree = int((np.sign(x) == np.sign(y)).sum())
        ax.set_title(f"{title}\nPearson r = {r:+.3f}   sign agreement {agree}/14",
                     fontsize=10.5)
        ax.set_xlabel("screen: per-fold $\\Delta$ log-loss (cand - R0)")
        ax.grid(alpha=.25, zorder=0)
    axes[0].set_ylabel("arbiter: per-fold $\\Delta$ CPL, own $\\tau$ (c/L)")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[
        Line2D([], [], marker="o", ls="", color="#c44e52", label="shock fold"),
        Line2D([], [], marker="o", ls="", color="#4c72b0", label="normal fold")],
        fontsize=8.5, loc="lower right")
    gap = results["crux"]["correlation_gap"]
    fig.suptitle("Same model, same folds, same seeds — only the STATION POPULATION "
                 "differs\n"
                 f"gap in r = {gap['observed']:+.3f}, 95% CI "
                 f"[{gap['ci95'][0]:+.3f}, {gap['ci95'][1]:+.3f}]  "
                 f"(tgp_cycle_displacement, fps-e6i)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "crux_screen_vs_arbiter.png", dpi=140)
    plt.close(fig)

    # --- 2. tau-distance profile ------------------------------------------
    prof = pd.DataFrame(results["arms"]["seed_mean"]["tau_distance_profile"])
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ix = np.arange(len(prof))
    ax.bar(ix - .2, 100 * prof.row_share, .4, label="% of rows", color="#b0b8c4")
    ax.bar(ix + .2, 100 * prof.gain_share, .4, label="% of log-loss gain",
           color="#4c72b0")
    ax.set_xticks(ix)
    ax.set_xticklabels([b.replace("(-0.001, ", "[0.0, ") for b in prof.bin],
                       rotation=20, fontsize=8.5)
    ax.set_xlabel(r"$|p_{raw}(R0) - \tau_{raw}|$   (distance from the decision boundary)")
    ax.set_ylabel("percent")
    step = results["arms"]["seed_mean"]["one_isotonic_step"]
    ax.set_title("The gain is NOT threshold-remote — it is threshold-INDIFFERENT\n"
                 f"one isotonic step of $\\tau$: {100*step['row_share']:.2f}% of rows "
                 f"carry {100*step['gain_share']:.1f}% of the gain "
                 f"({step['gain_share']/step['row_share']:.2f}x)", fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "tau_distance_profile.png", dpi=140)
    plt.close(fig)

    # --- 3. where tau actually sits ---------------------------------------
    g = pd.DataFrame(results["geometry"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    r0 = w.reset_index()
    r0 = r0[r0.seed == REALISED_SEED]
    ax.hist(r0["R0"], bins=180, color="#b0b8c4", log=True)
    ax.axvspan(g.tau_raw.min(), g.tau_raw.max(), color="#4c72b0", alpha=.35,
               label=r"$\tau_{raw}$ range across folds" + f"\n[{g.tau_raw.min():.3f}, "
                     f"{g.tau_raw.max():.3f}]")
    ax.axvline(0.25, color="#c44e52", lw=2, ls="--",
               label=r"$\tau = 0.25$ read as a RAW probability" + "\n(the wrong point)")
    ax.set_xlabel("raw LightGBM probability (R0, seed 42)")
    ax.set_ylabel("rows (log scale)"); ax.legend(fontsize=8.5)
    ax.set_title("The threshold is calibrated; the screen is not", fontsize=10.5)
    ax = axes[1]
    import joblib as _jl
    cache = _jl.load(BATCH / "r0_cache.joblib")
    xs = np.linspace(0, 0.6, 3000)
    for f in (1, 7, 14):
        iso = cache.per_fold[f]["cal_pipe"].calibrator
        ax.plot(xs, np.clip(iso.predict(xs), 0, 1), lw=1.6, label=f"fold {f}")
    ax.axhline(TAU, color="#c44e52", lw=1.6, ls="--", label=r"$\tau=0.25$ (calibrated)")
    ax.axvspan(g.tau_raw.min(), g.tau_raw.max(), color="#4c72b0", alpha=.3)
    ax.set_xlabel("raw probability"); ax.set_ylabel("calibrated probability")
    ax.set_title("Isotonic map: 0.25 calibrated $\\approx$ 0.11-0.16 raw", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(alpha=.3); ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.8)
    fig.tight_layout(); fig.savefig(OUT / "where_tau_sits.png", dpi=140)
    plt.close(fig)
    print("\nwrote crux_screen_vs_arbiter.png, tau_distance_profile.png, "
          "where_tau_sits.png")


def crux(w: pd.DataFrame, idx: list[str], seeds: list, results: dict) -> None:
    """The bead's actual puzzle: why the screen's per-fold delta doesn't track the
    arbiter's per-fold delta (6/14 sign agreement, corr -0.176).

    If the population channel is the cause, restricting the SCREEN to the 5 stations
    the arbiter replays should move that correlation; if threshold-distance is the
    cause, it should not.
    """
    print("\n" + "=" * 78)
    print("[G] CRUX — screen vs arbiter per-fold agreement, all stations vs the 5")
    print("=" * 78)
    facts = json.loads((CAND / "facts.json").read_text())
    real = pd.DataFrame(facts["breakdowns"]["per_fold"])[
        ["fold", "regime", "n_fills", "delta_cpl_own", "delta_ll_all_median"]]

    ws = w.reset_index()
    ws["pref"] = ws.station_code.isin(PREFERRED)

    def per_fold_delta_ll(frame: pd.DataFrame, paired: bool) -> pd.Series:
        """Per-fold screen delta, under either seed-aggregation convention.

        paired=True  -> median over seeds of the PAIRED per-seed delta.
        paired=False -> difference of the two arms' seed-medians, which is what
                        experiments/lib/aggregate.py publishes as
                        ``delta_ll_all_median``.  The two are not the same
                        statistic and disagree in sign on near-zero folds, so
                        every correlation below is reported under both.
        """
        if paired:
            by_seed = frame.groupby(["fold", "seed"]).d_ll.mean()
            return by_seed.groupby("fold").median()
        ll = frame.groupby(["fold", "seed"])[["ll_R0", "ll_cand"]].mean()
        med = ll.groupby("fold").median()
        return med.ll_cand - med.ll_R0

    tab = real.set_index("fold").join(pd.DataFrame({
        "screen_all": per_fold_delta_ll(ws, True),
        "screen_pref5": per_fold_delta_ll(ws[ws.pref], True),
        "screen_all_unpaired": per_fold_delta_ll(ws, False),
        "screen_pref5_unpaired": per_fold_delta_ll(ws[ws.pref], False),
    }))
    n_pref = ws[ws.pref].groupby("fold").size() / len(seeds)
    tab["n_rows_pref5"] = n_pref

    print(f"\n{'fold':>5}{'regime':>8}{'fills':>7}{'rows@5':>8}{'d_cpl_own':>12}"
          f"{'screen(714)':>13}{'screen(5)':>12}{'sign agree':>12}")
    for f, r in tab.iterrows():
        agree_all = "yes" if np.sign(r.delta_cpl_own) == np.sign(r.screen_all) else "no"
        agree_p = "yes" if np.sign(r.delta_cpl_own) == np.sign(r.screen_pref5) else "no"
        print(f"{f:>5}{r.regime:>8}{int(r.n_fills):>7}{int(r.n_rows_pref5):>8}"
              f"{r.delta_cpl_own:>+12.4f}{r.screen_all:>+13.5f}{r.screen_pref5:>+12.5f}"
              f"{agree_all + '/' + agree_p:>12}")

    print(f"\n{'comparison':<40}{'Pearson r':>12}{'Spearman':>11}{'sign agree':>13}")
    out = {}
    for nm, col in (
            ("paired  | screen ALL 714 stations", "screen_all"),
            ("paired  | screen PREFERRED 5 only", "screen_pref5"),
            ("published conv. | ALL 714 stations", "screen_all_unpaired"),
            ("published conv. | PREFERRED 5 only", "screen_pref5_unpaired")):
        r = float(tab.delta_cpl_own.corr(tab[col]))
        rho = float(tab.delta_cpl_own.corr(tab[col], method="spearman"))
        agree = int((np.sign(tab.delta_cpl_own) == np.sign(tab[col])).sum())
        print(f"{nm:<40}{r:>+12.3f}{rho:>+11.3f}{f'{agree}/14':>13}")
        out[col] = {"pearson": r, "spearman": rho, "sign_agree": agree}
    # n = 14 folds. Report the interval, not the point estimate, and bootstrap the
    # DIFFERENCE between the two correlations (that difference is the actual claim).
    print("\n  95% CIs (Fisher z, n=14 folds) and a fold-level bootstrap of the gap:")
    n = len(tab)
    for nm, col in (("ALL 714 stations", "screen_all"),
                    ("PREFERRED 5 only", "screen_pref5")):
        r = float(tab.delta_cpl_own.corr(tab[col]))
        z = np.arctanh(np.clip(r, -0.999999, 0.999999)); se = 1 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        print(f"    {nm:<20} r = {r:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
        out.setdefault(col, {}).update({"ci95": [float(lo), float(hi)]})
    rng = np.random.default_rng(0)
    a = tab.delta_cpl_own.to_numpy(); b = tab.screen_all.to_numpy(); c = tab.screen_pref5.to_numpy()
    diffs = []
    for _ in range(20000):
        k = rng.integers(0, n, n)
        if np.std(a[k]) == 0 or np.std(b[k]) == 0 or np.std(c[k]) == 0:
            continue
        diffs.append(np.corrcoef(a[k], c[k])[0, 1] - np.corrcoef(a[k], b[k])[0, 1])
    diffs = np.array(diffs)
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    obs = float(tab.delta_cpl_own.corr(tab.screen_pref5) - tab.delta_cpl_own.corr(tab.screen_all))
    print(f"    gap (pref5 - all)    r = {obs:+.3f}   95% CI [{d_lo:+.3f}, {d_hi:+.3f}]  "
          f"P(gap>0) = {float((diffs > 0).mean()):.3f}")
    out["correlation_gap"] = {"observed": obs, "ci95": [float(d_lo), float(d_hi)],
                              "p_gap_positive": float((diffs > 0).mean()),
                              "n_folds": int(n), "n_boot": int(len(diffs))}
    r_check = float(tab.delta_cpl_own.corr(tab.delta_ll_all_median))
    print(f"{'(published delta_ll_all_median, as a check)':<40}{r_check:>+12.3f}")
    out["published_check_pearson"] = r_check

    ov = ws.pref.sum() / len(ws)
    print(f"\nEvidence overlap: the arbiter reads {ws.pref.sum()//len(seeds):,} of the "
          f"screen's {len(ws)//len(seeds):,} rows = {100*ov:.2f}%.")
    print("Restricting the screen to the arbiter's own stations is the ONLY change "
          "between\nthe two correlation rows above — same model, same folds, same seeds.")
    results["crux"] = out
    tab.to_csv(OUT / "per_fold_screen_vs_arbiter.csv")
    print(f"wrote {OUT / 'per_fold_screen_vs_arbiter.csv'}")


if __name__ == "__main__":
    sys.exit(main())
