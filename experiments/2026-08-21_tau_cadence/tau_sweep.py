"""fps-929: is the daily-cadence plateau an OPERATING-POINT limit or a SIGNAL limit?

fps-fii (experiments/2026-08-20_cadence_ceiling/) found the model plateaus where the
oracle does not:

    cadence   model    oracle   headroom   captured
    7d        189.67   188.14   1.54       71%
    2d        187.85   185.16   2.69       65%
    1d        187.82   184.85   2.97       64%

Model gains 1.82 c/L going 7d->2d, then 0.03 c/L going 2d->1d, while the oracle keeps
improving. Two explanations, implying completely different work:

  1. OPERATING-POINT limit. tau=0.25 was OOF-selected under weekly economics where
     nearly every decision is forced (67.6% of 7d fills are emergency fills). A model
     with ~12 wait-rungs instead of 1 should plausibly be choosier. Fix is free.
  2. SIGNAL-CONTENT limit. The features carry no more exploitable information and the
     plateau is real. Fix is features.

THIS SCRIPT IS ASYMMETRIC EVIDENCE, AND THAT IS DELIBERATE. It sweeps a FIXED tau over
the replay and reads off the best one, which is choosing tau with hindsight on the same
rows it is scored on. So:

  * A FLAT curve at 1d RULES OUT the operating-point explanation. If the best tau
    available in hindsight is no better than 0.25, no honest selection rule can do
    better either. That verdict is definitive.
  * A curve with a large dip does NOT establish a realisable gain. It UPPER-BOUNDS one.
    An honest re-pick has to select tau out-of-sample, and `_train_calibrate_select_tau`
    takes no TankParams at all -- it optimises a proxy expected-cents criterion over OOF
    predictions, not tank economics -- so making selection cadence-aware is a real code
    change, not a replay. That is follow-up work, gated on this dip being large enough
    to be worth it.

Reference point: at 7d every fold's OOF-selected tau came out at exactly 0.25, the
production lock (model_cadence_per_fold.csv), so the 7d sweep should pass through
~189.67 at tau=0.25. That is the harness's own self-check.

COST TRICK (inherited from fps-fii, exact rather than approximate): a fold's fit is
BOTH tank-independent and tau-independent. `_train_calibrate_select_tau` takes no
TankParams, and tau is only a threshold applied afterwards to already-calibrated
probabilities. So we pay ONE walk-forward fit at 7d through the real harness, keep its
`RealisedResult.baseline_cache` (each fold's `cal_pipe`), and replay those same fitted
models at every (cadence, tau) cell through the same `aggregate_backtest` economics.

RUN-DRY AUDIT, AND WHY IT IS NOT INHERITED: fps-fii audited dryness per CADENCE. A tau
sweep needs it per CELL, because tau changes it. A HIGHER tau buys less often, so it can
strand a tank that tau=0.25 kept wet -- and `run_backtest` clamps a negative tank to 0
and carries on rather than erroring. A cell with dry events has bought fewer litres than
the arms it is compared against and its CPL must not be quoted.

WINDOW LEVEL ONLY. Do not slice any of this by cycle zone -- per-zone economics is not
identified (fps-1785999730023-4-264564ac, fps-grp). Per-fold is fine: each (fold,
station) is an independent simulation with its own tank.

  PYTHONPATH=. uv run python experiments/2026-08-21_tau_cadence/tau_sweep.py 2>&1 \
    | tee experiments/2026-08-21_tau_cadence/run.log
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import time

import pandas as pd

from experiments.lib.constants import BASELINE_COLUMNS, BASELINE_FINGERPRINT
from experiments.lib.io import current_git_sha, write_meta
from experiments.lib.realised import ArmSpec, _plan_folds, run_paired_realised_backtest
from fuel_signal import db as _db
from fuel_signal.backtest import (
    AlwaysBuyStrategy,
    ModelStrategy,
    TankParams,
    _evaluation_dates,
    load_history,
    run_oracle_backtest,
)
from fuel_signal.backtest_phase2 import aggregate_backtest
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.features import load_features

HERE = pathlib.Path(__file__).resolve().parent
FII = HERE.parent / "2026-08-20_cadence_ceiling" / "model_cadence.py"

# _dry_audit is reused from fps-fii rather than copied: the audit's correctness rests on
# mirroring run_backtest's own level arithmetic exactly, and a second transcription is a
# second chance to get that wrong. This is its SECOND use, which is this repo's stated
# trigger for hoisting shared glue into experiments/lib (cf. fps-1785999729817-2) --
# filed as a follow-up rather than done here, so this experiment doesn't also become a
# refactor of the harness it depends on.
_spec = importlib.util.spec_from_file_location("_fii_model_cadence", FII)
_fii = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fii)
_dry_audit = _fii._dry_audit
DRY_TOL = _fii.DRY_TOL

# 7d is the production lock and the control; 1d is where the plateau lives. 2d is the
# nearest provably-safe intermediate cadence (fps-fii established 3d is STRUCTURALLY
# unusable -- a reachable decide-level sits above the emergency floor yet cannot survive
# one interval -- so it is not swept here at all).
CADENCES = (7, 2, 1)
FIT_CADENCE = 7

# Production tau is 0.25 and every fold's OOF selection agreed. The hypothesis under
# test says a finer grid should want a CHOOSIER model, so the grid extends well above
# 0.25 while staying dense around it.
TAUS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)

INNER_FOLDS = {"train_min_days": 1095, "val_days": 90, "step_days": 90}

_SUBSET = os.environ.get("TAU_FOLDS")
FOLD_SUBSET = [int(x) for x in _SUBSET.split(",")] if _SUBSET else None
_TAU_ENV = os.environ.get("TAU_GRID")
if _TAU_ENV:
    TAUS = tuple(float(x) for x in _TAU_ENV.split(","))
OUT = HERE / "smoke" if (FOLD_SUBSET or _TAU_ENV) else HERE


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    station_codes = list(PREFERRED_STATIONS)
    plans = _plan_folds(df, {}, FOLD_SUBSET)
    print(f"[folds] {len(plans)} windows, {len(station_codes)} stations, "
          f"{len(BASELINE_COLUMNS)} cols ({BASELINE_FINGERPRINT})", flush=True)
    print(f"[grid]  cadences {CADENCES}  x  taus {TAUS}", flush=True)

    # --- The one paid fit pass -------------------------------------------------------
    res = run_paired_realised_backtest(
        [ArmSpec("baseline", df)], BASELINE_COLUMNS, collect_fills=True, seed=42,
        inner_fold_params=INNER_FOLDS, tank=TankParams(evaluation_interval_days=FIT_CADENCE),
        fold_subset=FOLD_SUBSET,
    )
    cache = res.baseline_cache
    assert cache is not None, "held_tau=None should always yield a baseline_cache"
    own_taus = sorted({cache.per_fold[p.fold]["own_tau"] for p in plans})
    print(f"[fit] harness pass done in {res.meta['total_wall_seconds']:.1f}s; "
          f"{len(cache.per_fold)} folds cached; OOF-selected taus {own_taus}", flush=True)

    rows: list[dict] = []
    fold_rows: list[dict] = []
    fills: list[dict] = []

    for cadence in CADENCES:
        ct0 = time.perf_counter()
        tank = TankParams(evaluation_interval_days=cadence)
        eval_dates = sorted({
            d for p in plans for d in _evaluation_dates(p.val_start, p.val_end, cadence)
        })
        conn = _db.open_db()
        try:
            history = load_history(conn, station_codes, eval_dates=eval_dates)
        finally:
            conn.close()
        print(f"[cadence {cadence}d] history over {len(eval_dates)} eval dates "
              f"({time.perf_counter() - ct0:.1f}s)", flush=True)

        # always-buy and the oracle are tau-INVARIANT: computed once per cadence, not
        # once per cell. They are the fixed reference the whole sweep is read against.
        a_spend = a_litres = o_spend = o_litres = 0.0
        for p in plans:
            always = aggregate_backtest(
                history, AlwaysBuyStrategy(), station_codes, p.val_start, p.val_end, tank,
            )
            a_spend += always["total_spend_cents"]
            a_litres += always["total_litres"]
            for sc in station_codes:
                r = run_oracle_backtest(
                    history, sc, p.val_start, p.val_end, tank, collect_fills=True
                )
                if pd.isna(r.realised_cpl):
                    print(f"[warn] oracle NaN cadence={cadence} fold={p.fold} stn={sc}",
                          flush=True)
                    continue
                o_spend += r.total_spend_cents
                o_litres += r.total_litres
        always_cpl = a_spend / a_litres
        oracle_cpl = o_spend / o_litres
        print(f"[cadence {cadence}d] reference arms: always {always_cpl:.2f}  "
              f"oracle {oracle_cpl:.2f}", flush=True)

        for tau in TAUS:
            tt0 = time.perf_counter()
            m_spend = m_litres = 0.0
            cell_fills: list[dict] = []
            for p in plans:
                cf = cache.per_fold[p.fold]
                model = aggregate_backtest(
                    history,
                    ModelStrategy(pipeline=cf["cal_pipe"],
                                  feature_columns=BASELINE_COLUMNS, threshold=tau),
                    station_codes, p.val_start, p.val_end, tank, collect_fills=True,
                )
                m_spend += model["total_spend_cents"]
                m_litres += model["total_litres"]
                for fr in model["fills"]:
                    cell_fills.append({
                        "cadence": cadence, "tau": tau, "fold": p.fold, "arm": "model",
                        "date": fr.date, "station_code": fr.station_code,
                        "price": fr.price, "litres": fr.litres,
                        "spend_cents": fr.spend_cents, "emergency": fr.emergency,
                    })
                fold_rows.append({
                    "cadence": cadence, "tau": tau, "fold": p.fold,
                    "val_start": p.val_start, "val_end": p.val_end,
                    "model_cpl": model["cpl"], "n_fills": len(model["fills"]),
                })
            model_cpl = m_spend / m_litres
            # A cell can in principle produce no fills at all (a tau so high the model
            # never buys, with no emergency triggered either); an empty frame with no
            # columns would blow up _dry_audit's column access rather than reporting
            # the dryness that situation IS.
            cell_df = pd.DataFrame(
                cell_fills,
                columns=["cadence", "tau", "fold", "arm", "date", "station_code",
                         "price", "litres", "spend_cents", "emergency"],
            )
            audit = _dry_audit(cell_df, plans, cadence, tank, station_codes)
            fills.extend(cell_fills)
            emerg = float(cell_df["emergency"].mean() * 100) if len(cell_df) else float("nan")
            rows.append({
                "cadence_days": cadence, "tau": tau,
                "always_cpl": always_cpl, "model_cpl": model_cpl, "oracle_cpl": oracle_cpl,
                "captured_cpl": always_cpl - model_cpl,
                "headroom_cpl": model_cpl - oracle_cpl,
                "captured_pct": 100.0 * (always_cpl - model_cpl) / (always_cpl - oracle_cpl),
                "model_fills": len(cell_fills),
                "model_emergency_pct": emerg,
                "model_litres": m_litres,
                "dry_events": audit["dry_events"],
                "worst_dry_litres": audit["worst_dry_litres"],
                "cell_valid": audit["dry_events"] == 0,
                "wall_s": time.perf_counter() - tt0,
            })
            flag = "" if audit["dry_events"] == 0 else "  *** DRY - INVALID ***"
            print(f"[cadence {cadence}d tau {tau:.2f}] model {model_cpl:.3f}  "
                  f"headroom {model_cpl - oracle_cpl:.3f}  fills {len(cell_fills)} "
                  f"({emerg:.0f}% emerg)  dry {audit['dry_events']}"
                  f"  [{time.perf_counter() - tt0:.1f}s]{flag}", flush=True)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "tau_sweep.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "tau_sweep_per_fold.csv", index=False)
    pd.DataFrame(fills).to_parquet(OUT / "tau_fills.parquet", index=False)

    print("\n=== realised CPL vs tau, by cadence (window level, pooled) ===", flush=True)
    print(sweep.round(3).to_string(index=False), flush=True)

    print("\n=== verdict inputs ===", flush=True)
    for cadence in CADENCES:
        c = sweep[(sweep.cadence_days == cadence) & sweep.cell_valid]
        if c.empty:
            print(f"  cadence {cadence}d: NO VALID CELLS", flush=True)
            continue
        at25 = c[c.tau == 0.25]["model_cpl"]
        best = c.loc[c.model_cpl.idxmin()]
        base = float(at25.iloc[0]) if len(at25) else float("nan")
        print(f"  cadence {cadence}d: best tau {best.tau:.2f} -> {best.model_cpl:.3f} c/L; "
              f"tau=0.25 -> {base:.3f}; gain from re-picking {base - best.model_cpl:.3f} c/L "
              f"(spread across grid {c.model_cpl.max() - c.model_cpl.min():.3f})", flush=True)

    write_meta(OUT, {
        "issue": "fps-929",
        "git_sha": current_git_sha(),
        "baseline_fingerprint": BASELINE_FINGERPRINT,
        "n_folds": len(plans),
        "stations": station_codes,
        "cadences": list(CADENCES),
        "taus": list(TAUS),
        "fit_cadence": FIT_CADENCE,
        "oof_selected_taus": own_taus,
        "seed": 42,
        "inner_fold_params": INNER_FOLDS,
        "dry_tol_litres": DRY_TOL,
        "fit_wall_seconds": res.meta["total_wall_seconds"],
        "total_wall_seconds": time.perf_counter() - t0,
        "note": (
            "Fixed-tau sweep read in hindsight: a FLAT curve rules out the "
            "operating-point explanation, a dip only UPPER-BOUNDS a realisable gain. "
            "Window level only; per-zone economics is not identified (fps-grp)."
        ),
    })
    print(f"\n[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
