"""Per-zone headroom via consumption-interval attribution + bootstrap CIs (bd fps-1785999730023-4-264564ac).

The #262 headroom map attributes cost to PURCHASE EVENTS: it groups fills by the
zone of the fill date and pools spend/litres within each zone. That breaks the
`oracle <= model` bound at the zone level for two reasons named in the bd issue:

1. The arms fill on different dates and in different counts, so a zone's model
   numerator/denominator describe different fills over different litres than the
   oracle's. It is not a paired quantity.
2. Path coupling: the oracle buys cheaply just BEFORE an expensive stretch and
   coasts through it, so its in-zone fills are disproportionately expensive and
   the saving lands in the neighbouring zone.

Both show up as impossible negatives (a perfect-foresight oracle cannot lose):
>=16c band -0.53, 2025-01 -3.6, 2023-10 -2.9.

This script re-attributes cost to CONSUMPTION instead. The tank model is fully
deterministic -- 50 L tank, 50/14 L/day, evaluated every 7 days, so exactly 25 L
burn per interval and the level only ever takes values {0, 25, 50}. Both arms
therefore burn identical litres on identical dates regardless of strategy. Costing
each litre at the price it was BOUGHT for but booking it to the interval it is
BURNED in gives every zone an identical denominator across arms, which is what
makes the difference paired. It also puts the oracle's cheap pre-emptive buy into
the expensive stretch it was actually bought for, which is the path-coupling fix.

Reconstruction, not re-simulation: the fills already carry everything needed. All
1454 fills sit on a clean 7-day lattice, `emergency` pins the post-fill level (an
emergency half-fill tops up to 25 L, a chosen buy always fills to 50 L), and the
level is deterministic between fills.

What the checks do and do NOT cover. `check_reconciliation` asserts
attributed + leftover == purchased, which is fuel CONSERVATION -- it holds for any
queue replay that neither creates nor destroys litres, so on its own it would not
catch a wrong burn SCHEDULE (a mis-derived interval count would just inflate
leftover). The schedule is covered separately by the lattice assert in
`_replay_unit`: every inter-fill gap must be a whole number of evaluation
intervals, which is what makes `n_intervals` exact rather than a floor.

Usage:
  PYTHONPATH=. uv run python experiments/2026-08-20_headroom_attribution/attribution.py
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd

from experiments.lib.zones import assign_regime
from fuel_signal.backtest import TankParams
from fuel_signal.features import load_features

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "2026-06-19_headroom_map"
TANK = TankParams()
INTERVAL_LITRES = TANK.daily_consumption_litres * TANK.evaluation_interval_days  # 25.0
N_BOOT = 4000
RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------
# Consumption replay
# --------------------------------------------------------------------------
def _replay_unit(g: pd.DataFrame, mode: str = "fifo", label: str = "start"
                 ) -> tuple[list[dict], dict]:
    """Attribute one (fold, station, arm)'s fill costs to consumption intervals.

    Returns (interval rows, reconciliation dict). `g` is that unit's fills, any
    order.

    `mode` selects the inventory convention, which matters because a chosen buy on
    a half-full tank leaves two vintages in it at once:
      'fifo'    - the litre burned first is the litre bought first.
      'average' - the tank is a well-mixed pool; every litre burns at the running
                  weighted-average cost of what is in it.
    Physically a real tank blends, so 'average' is the more faithful model and
    'fifo' the conventional inventory one. Total cost over a unit is identical
    either way; only WHICH INTERVAL the cost lands in differs, which is exactly
    what this experiment measures -- so the conclusion must survive both.

    `label` is a SECOND, independent convention with the same property. A burn
    spans [d, d+7); stamping it with the start, the end, or the midpoint is a free
    choice with no physical content. It is not cosmetic: regime spells run 12-18
    days, so a 7-day stamp shift moves up to half a zone width of burn across a
    boundary. run_backtest itself applies depletion at the interval END (top of the
    next eval date), so 'end' is the engine-consistent choice and 'start' the
    naive one -- but neither is more true of the fuel, which burned throughout.
    """
    g = g.sort_values("date")
    first = g.iloc[0]
    # Post-fill level is pinned by the emergency flag: an emergency tops the tank
    # up to half, a chosen buy always goes to full (litres = size - level).
    level = 25.0 if bool(first.emergency) else TANK.tank_size_litres
    # Whatever sat in the tank before the first fill is un-priced prior inventory
    # (the run starts at 50%). FIFO burns it first; it carries no cost and is
    # excluded from both numerator and denominator.
    prior = max(0.0, level - float(first.litres))
    queue: list[list[float]] = []
    if prior > 0:
        queue.append([prior, float("nan")])
    queue.append([float(first.litres), float(first.price)])

    rows: list[dict] = []
    burned = 0.0
    prior_burned = 0.0
    dates = list(g["date"])
    for j in range(len(g)):
        if j > 0:
            r = g.iloc[j]
            level = 25.0 if bool(r.emergency) else TANK.tank_size_litres
            queue.append([float(r.litres), float(r.price)])
        # Burn forward until the next fill (or stop at the last fill: fuel still
        # in the tank at the window tail was bought but never consumed).
        if j + 1 >= len(g):
            break
        gap_days = (dates[j + 1] - dates[j]).days
        # Guards the burn SCHEDULE, which conservation cannot: an off-lattice gap
        # would silently floor n_intervals, under-burn, and hide the shortfall in
        # leftover while every conservation identity still balanced.
        if gap_days % TANK.evaluation_interval_days != 0:
            raise AssertionError(
                f"off-lattice gap {gap_days}d between fills at {dates[j]} and "
                f"{dates[j + 1]} -- interval count would be wrong")
        n_intervals = gap_days // TANK.evaluation_interval_days
        for t in range(n_intervals):
            want = min(INTERVAL_LITRES, level)
            level -= want
            # Whole days only: the zone lookup is keyed on calendar dates, so a
            # half-interval offset would land between them and drop every row.
            off_days = {"start": 0, "mid": 3, "end": 7}[label]
            when = dates[j] + pd.Timedelta(
                days=TANK.evaluation_interval_days * t + off_days)
            cost = 0.0
            litres_priced = 0.0
            remaining = want
            if mode == "average" and queue:
                # Blend the whole tank to one price, then draw from it. Un-priced
                # prior inventory stays segregated: it has no cost to average in.
                priced = [q for q in queue if not np.isnan(q[1])]
                pl = sum(q[0] for q in priced)
                if pl > 0:
                    avg = sum(q[0] * q[1] for q in priced) / pl
                    queue = ([q for q in queue if np.isnan(q[1])] + [[pl, avg]])
            while remaining > 1e-9 and queue:
                take = min(remaining, queue[0][0])
                if np.isnan(queue[0][1]):
                    prior_burned += take
                else:
                    cost += take * queue[0][1]
                    litres_priced += take
                queue[0][0] -= take
                remaining -= take
                if queue[0][0] <= 1e-9:
                    queue.pop(0)
            burned += want
            if litres_priced > 0:
                rows.append({"date": when, "litres": litres_priced, "spend_cents": cost})

    leftover = sum(q[0] for q in queue if not np.isnan(q[1]))
    leftover_cost = sum(q[0] * q[1] for q in queue if not np.isnan(q[1]))
    recon = {
        "purchased_litres": float(g["litres"].sum()),
        "purchased_spend": float(g["spend_cents"].sum()),
        "attributed_litres": sum(r["litres"] for r in rows),
        "attributed_spend": sum(r["spend_cents"] for r in rows),
        "leftover_litres": leftover,
        "leftover_spend": leftover_cost,
        "prior_burned": prior_burned,
        "burned_total": burned,
    }
    return rows, recon


def build_consumption_ledger(fills: pd.DataFrame, arm: str, mode: str = "fifo",
                             label: str = "start"
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, recs = [], []
    for (fold, station), g in fills.groupby(["fold", "station_code"], sort=True):
        rows, recon = _replay_unit(g, mode=mode, label=label)
        for r in rows:
            r.update({"fold": fold, "station_code": station, "arm": arm})
        out.extend(rows)
        recon.update({"fold": fold, "station_code": station, "arm": arm})
        recs.append(recon)
    return pd.DataFrame(out), pd.DataFrame(recs)


def check_reconciliation(recon: pd.DataFrame, arm: str) -> None:
    """Every litre bought is burned, still in the tank, or was prior inventory."""
    bad = 0
    lhs = recon["attributed_litres"] + recon["leftover_litres"]
    rhs = recon["purchased_litres"]
    if not np.allclose(lhs, rhs, atol=1e-6):
        bad += int((~np.isclose(lhs, rhs, atol=1e-6)).sum())
    lhs_s = recon["attributed_spend"] + recon["leftover_spend"]
    if not np.allclose(lhs_s, recon["purchased_spend"], atol=1e-4):
        bad += int((~np.isclose(lhs_s, recon["purchased_spend"], atol=1e-4)).sum())
    if bad:
        raise AssertionError(f"[{arm}] reconciliation failed on {bad} unit-checks")
    print(f"  [{arm}] reconciliation OK over {len(recon)} station-folds "
          f"(attributed {recon['attributed_litres'].sum():,.0f} L, "
          f"leftover {recon['leftover_litres'].sum():,.0f} L, "
          f"prior {recon['prior_burned'].sum():,.0f} L)", flush=True)


# --------------------------------------------------------------------------
# Zone tagging
# --------------------------------------------------------------------------
#: Fixed cent-bands, matching the #262 re-cut (its tercile cut was a binning
#: artifact), so the two runs' volatility rows are comparable. Edge semantics
#: differ in principle -- this uses [lo, hi), #262 used pd.cut's (lo, hi] -- which
#: is inert here: features.csv has no exact 8/12/16 values (min 2.48), and the
#: recomputed old estimator reproduces #262's published 7.09 and -0.53 to the
#: decimal, so the comparison is empirically sound even though the construction
#: is not byte-identical.
VOL_BANDS = [("<8c", -np.inf, 8.0), ("8-12c", 8.0, 12.0),
             ("12-16c", 12.0, 16.0), (">=16c", 16.0, np.inf)]


def _vol_band(x: float) -> str:
    if pd.isna(x):
        return "unmatched"
    for name, lo, hi in VOL_BANDS:
        if lo <= x < hi:
            return name
    return "unmatched"


def date_zone_lookup() -> pd.DataFrame:
    """Per-DATE cycle_pct_through + network_px_std, same construction as the #262
    map's `_date_zone_lookup` -- both are network-wide so a date alone keys them.
    Rebuilt here rather than reused so grid dates with NO fill still get tagged,
    which purchase-event attribution never needed."""
    feat = load_features()
    feat["price_date"] = pd.to_datetime(feat["price_date"])
    lk = (feat.groupby("price_date")[["cycle_pct_through", "network_px_std"]]
          .first().reset_index().rename(columns={"price_date": "date"}))
    # Drop before mapping, not after: assign_regime/_vol_band turn a NaN into the
    # STRING "unmatched", which survives a dropna on the mapped column and would
    # pool into the panels as a spurious extra zone. The #262 `_tag` this replaces
    # dropped on raw cycle_pct_through for exactly this reason.
    lk = lk.dropna(subset=["cycle_pct_through", "network_px_std"])
    lk["regime"] = lk["cycle_pct_through"].map(assign_regime)
    lk["volatility"] = lk["network_px_std"].map(_vol_band)
    lk["quarter"] = "Q" + lk["date"].dt.quarter.astype(str)
    return lk


# --------------------------------------------------------------------------
# Paired per-zone headroom + bootstrap
# --------------------------------------------------------------------------
def zone_panel(ledger: pd.DataFrame, axis: str, unit: str) -> pd.DataFrame:
    """Collapse the consumption ledger to (unit, zone, arm) -> litres, spend."""
    return (ledger.groupby([unit, axis, "arm"], observed=True)[["litres", "spend_cents"]]
            .sum().reset_index().rename(columns={unit: "unit", axis: "zone"}))


def _headroom(panel: pd.DataFrame) -> pd.Series:
    """model_cpl - oracle_cpl per zone from a (unit, zone, arm) panel."""
    agg = panel.groupby(["zone", "arm"], observed=True)[["litres", "spend_cents"]].sum()
    cpl = (agg["spend_cents"] / agg["litres"]).unstack("arm")
    if "model" not in cpl or "oracle" not in cpl:
        return pd.Series(dtype=float)
    return cpl["model"] - cpl["oracle"]


def bootstrap_ci(panel: pd.DataFrame, n_boot: int = N_BOOT, alpha: float = 0.05
                 ) -> pd.DataFrame:
    """Resample UNITS with replacement; percentile CI on per-zone headroom.

    Resampling whole units (not rows) keeps each unit's two arms together, so the
    pairing survives the resample -- that is the whole point of the paired
    denominator built above.
    """
    units = panel["unit"].unique()
    point = _headroom(panel)
    idx = {u: panel[panel["unit"] == u] for u in units}
    draws = []
    for _ in range(n_boot):
        pick = RNG.choice(units, size=len(units), replace=True)
        draws.append(_headroom(pd.concat([idx[u] for u in pick], ignore_index=True)))
    d = pd.DataFrame(draws)
    lo = d.quantile(alpha / 2)
    hi = d.quantile(1 - alpha / 2)
    return pd.DataFrame({
        "headroom_cpl": point,
        "ci_lo": lo, "ci_hi": hi,
        "ci_width": hi - lo,
        "boot_std": d.std(),
        "crosses_zero": (lo < 0) & (hi > 0),
    }).reindex(point.index)


def bootstrap_zone_contrasts(panel: pd.DataFrame, n_boot: int = N_BOOT,
                             alpha: float = 0.05) -> pd.DataFrame:
    """CI on the DIFFERENCE in headroom between every pair of zones.

    Overlapping per-zone CIs do NOT imply the zones are indistinguishable -- the
    two estimates share folds, so their errors are correlated and the contrast is
    tighter than either interval suggests. The claim "late descent is worse than
    normal" is a contrast, so it has to be tested as one.
    """
    units = panel["unit"].unique()
    idx = {u: panel[panel["unit"] == u] for u in units}
    point = _headroom(panel)
    zones = list(point.index)
    draws = []
    for _ in range(n_boot):
        pick = RNG.choice(units, size=len(units), replace=True)
        draws.append(_headroom(pd.concat([idx[u] for u in pick], ignore_index=True)))
    d = pd.DataFrame(draws)
    recs = []
    for i, a in enumerate(zones):
        for b in zones[i + 1:]:
            if a not in d or b not in d:
                continue
            diff = d[a] - d[b]
            lo, hi = diff.quantile(alpha / 2), diff.quantile(1 - alpha / 2)
            recs.append({"zone_a": a, "zone_b": b,
                         "diff": point[a] - point[b], "ci_lo": lo, "ci_hi": hi,
                         "separated": bool(lo > 0 or hi < 0),
                         "p_a_worse": float((diff > 0).mean())})
    return pd.DataFrame(recs)


def purchase_event_headroom(model: pd.DataFrame, oracle: pd.DataFrame,
                            axis: str) -> pd.Series:
    """The OLD estimator, recomputed here so old and new sit in one table."""
    def cpl(d):
        return d.groupby(axis, observed=True).apply(
            lambda g: g["spend_cents"].sum() / g["litres"].sum(), include_groups=False)
    return cpl(model) - cpl(oracle)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_ledger(model: pd.DataFrame, oracle: pd.DataFrame, lk: pd.DataFrame,
                 mode: str, label: str = "start", verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"\n[replay] intervals (cost={mode}, label={label})", flush=True)
    m_led, m_rec = build_consumption_ledger(model, "model", mode=mode, label=label)
    o_led, o_rec = build_consumption_ledger(oracle, "oracle", mode=mode, label=label)
    if verbose:
        check_reconciliation(m_rec, "model")
        check_reconciliation(o_rec, "oracle")
        pd.concat([m_rec, o_rec], ignore_index=True).to_csv(
            HERE / f"reconciliation_{mode}.csv", index=False)
    led = pd.concat([m_led, o_led], ignore_index=True).merge(
        lk, on="date", how="left", validate="many_to_one")
    lost = led["regime"].isna().sum()
    if lost and lost == len(led):
        raise AssertionError(
            f"every interval row lost the date join (label={label!r}) -- the burn "
            f"stamps are not on the lookup's calendar-date grid")
    if lost:
        print(f"  [warn] {lost}/{len(led)} interval rows lost the date join - dropping",
              flush=True)
        led = led.dropna(subset=["regime"])
    led["station_fold"] = led["fold"].astype(str) + ":" + led["station_code"].astype(str)

    # Common span per (fold, station): attribution stops at each arm's own last
    # fill, and the arms' last fills differ, so without this the arms cover
    # slightly different windows -- reintroducing the unpaired-denominator defect
    # this estimator exists to remove. Trim both to the earlier of the two ends.
    cut = (led.groupby(["fold", "station_code", "arm"], observed=True)["date"].max()
           .unstack("arm").min(axis=1).rename("cut"))
    before = len(led)
    led = led.merge(cut, on=["fold", "station_code"], how="left")
    led = led[led["date"] <= led["cut"]].drop(columns="cut")
    if verbose:
        print(f"  [span] trimmed {before - len(led)} interval rows to the common "
              f"per-station-fold window", flush=True)
    return led


def main() -> None:
    t0 = time.perf_counter()
    model = pd.read_parquet(SRC / "model_fills.parquet")
    oracle = pd.read_parquet(SRC / "oracle_fills.parquet")
    for d in (model, oracle):
        d["date"] = pd.to_datetime(d["date"])
    print(f"[load] model {len(model)} fills, oracle {len(oracle)} fills", flush=True)
    lk = date_zone_lookup()
    led = build_ledger(model, oracle, lk, "fifo")

    # Paired-denominator check: the whole claim of this estimator is that both
    # arms burn the same litres in the same zone. Report where they do not.
    print("\n[pairing] litres per zone per arm (identical => paired):", flush=True)
    chk = (led.groupby(["regime", "arm"], observed=True)["litres"].sum()
           .unstack("arm"))
    chk["diff_pct"] = 100 * (chk["model"] - chk["oracle"]).abs() / chk["oracle"]
    print(chk.round(1).to_string(), flush=True)

    out = {}
    for axis in ("regime", "volatility", "quarter"):
        old = purchase_event_headroom(
            model.assign(volatility=model["network_px_std"].map(_vol_band)),
            oracle.assign(volatility=oracle["network_px_std"].map(_vol_band)), axis)
        rows = {}
        for unit, label in (("fold", "fold"), ("station_code", "station"),
                            ("station_fold", "station_fold")):
            panel = zone_panel(led, axis, unit)
            ci = bootstrap_ci(panel)
            rows[label] = ci
        tab = rows["fold"].copy()
        tab = tab.rename(columns={c: f"fold_{c}" for c in
                                  ("ci_lo", "ci_hi", "ci_width", "boot_std", "crosses_zero")})
        for label in ("station", "station_fold"):
            tab[f"{label}_ci_lo"] = rows[label]["ci_lo"]
            tab[f"{label}_ci_hi"] = rows[label]["ci_hi"]
        tab["old_purchase_event"] = old
        tab["shift_vs_old"] = tab["headroom_cpl"] - tab["old_purchase_event"]
        tab.index.name = "zone"
        out[axis] = tab
        print(f"\n=== {axis.upper()} ===", flush=True)
        print(tab[["old_purchase_event", "headroom_cpl", "shift_vs_old",
                   "fold_ci_lo", "fold_ci_hi", "fold_crosses_zero"]].round(3).to_string(),
              flush=True)

    # --- Contrasts: an ordering claim is a CONTRAST, so test it as one. Run for
    # BOTH axes -- the volatility panel was originally asserted without this and
    # the assertion did not survive the test.
    for axis in ("regime", "volatility"):
        c = bootstrap_zone_contrasts(zone_panel(led, axis, "fold"))
        print(f"\n=== {axis.upper()} CONTRASTS (fifo/start, fold bootstrap) ===", flush=True)
        print(c.round(3).to_string(index=False), flush=True)
        c.to_csv(HERE / f"contrasts_{axis}.csv", index=False)

    # --- The headline: the full convention grid. Two independent free choices
    # (cost basis x interval label) with no physical content, 6 combinations.
    # If the zone ORDERING is not stable across this grid, the per-zone
    # decomposition is not identified and no amount of extra data fixes it.
    print("\n=== CONVENTION GRID: per-zone headroom under 6 estimators ===", flush=True)
    grid = []
    for cost in ("fifo", "average"):
        for label in ("start", "mid", "end"):
            g_led = build_ledger(model, oracle, lk, cost, label, verbose=False)
            for axis in ("regime", "volatility"):
                h = _headroom(zone_panel(g_led, axis, "fold"))
                for zone, v in h.items():
                    grid.append({"axis": axis, "cost": cost, "label": label,
                                 "zone": zone, "headroom_cpl": v})
    grid = pd.DataFrame(grid)
    grid.to_csv(HERE / "convention_grid.csv", index=False)
    for axis in ("regime", "volatility"):
        piv = (grid[grid.axis == axis]
               .pivot_table(index="zone", columns=["cost", "label"], values="headroom_cpl"))
        print(f"\n-- {axis} --", flush=True)
        print(piv.round(2).to_string(), flush=True)
        spread = (piv.max(axis=1) - piv.min(axis=1)).rename("convention_spread")
        between = piv.max(axis=0) - piv.min(axis=0)
        print(f"   per-zone spread ACROSS conventions: "
              f"{spread.round(2).to_dict()}", flush=True)
        print(f"   spread BETWEEN zones, per convention: "
              f"min {between.min():.2f} max {between.max():.2f}", flush=True)
        print(f"   -> identified only if between-zone > across-convention; "
              f"worst zone spread {spread.max():.2f} vs best between-zone "
              f"{between.max():.2f}", flush=True)
        piv.to_csv(HERE / f"convention_grid_{axis}.csv")
        out[axis]["convention_spread"] = spread
        neg = (grid[(grid.axis == axis)]["headroom_cpl"] < 0).sum()
        print(f"   impossible negatives across the grid: {neg}/{len(grid[grid.axis==axis])}",
              flush=True)

    for axis, tab in out.items():
        tab.to_csv(HERE / f"headroom_{axis}.csv")
    led.to_parquet(HERE / "consumption_ledger.parquet", index=False)
    print(f"\n[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
