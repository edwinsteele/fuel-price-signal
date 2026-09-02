"""Stage 2: the MODEL arm at the same cadences, so reported headroom is like-for-like.

Stage 1 (oracle_cadence.py) showed the perfect-foresight ceiling falls as the
decision grid gets finer. That on its own does NOT mean headroom grows — a finer
grid hands the MODEL extra chances too. This stage replays the production
54-feature baseline at 7 / 3 / 1 day cadence over the identical folds, stations
and seed, so `model_cpl - oracle_cpl` is measured within one cadence.

COST TRICK (exact, not an approximation): a fold's fit is tank-independent.
`experiments.lib.realised._train_calibrate_select_tau(train_df, cols, seed,
inner_fold_params)` takes no TankParams — the calibrated pipeline and the OOF-
selected tau depend only on train rows, columns, seed and the inner fold grid.
Only the REPLAY (and the PIT lga_days_since eval-date grid the PriceHistory is
built on) is cadence-dependent. So we pay the ~9min walk-forward fit once at 7d
via the real harness, keep its `RealisedResult.baseline_cache` (which captures
each fold's `cal_pipe` + `own_tau`), and replay those same fitted models at the
finer cadences through the same `aggregate_backtest` economics the harness uses
internally. No scaffolding is reimplemented: the fit comes from the harness, the
economics come from the production engine.

RUN-DRY AUDIT: cadence is not a free knob. ``run_backtest`` clamps a tank that
would go negative to 0 and carries on (the emergency rule tests the CURRENT level,
not the post-depletion overshoot). Which decide-levels are reachable depends on
D = daily x interval, so changing the cadence can put a reachable level inside
the (floor*size, D) gap where the model can silently strand the tank and buy
fewer litres — that is a real cadence artefact and must invalidate the ceiling.

Before PR #358, ``run_oracle_backtest`` pruned run-dry paths outright, so
"either arm has a dry event" was a correct, if blunt, proxy for that failure: the
oracle could never legitimately have one, so any dry event anywhere meant the two
arms had stopped solving the same problem. PR #358 made the oracle clamp-and-
account through a no-price date exactly like ``run_backtest`` does (fps-32h), so
a real station closure now produces dry events in BOTH arms even though nothing
about the cadence choice is unsafe — the blunt proxy fires on every cadence a
closure falls in. The gate below re-aims at the invariant it was actually
protecting rather than loosening it: every dry event must be (1) attributable to
darkness — the station had no resolvable price at that date, mirroring
``run_backtest``'s own gap-forgiveness exactly, including its one-step grace on
the date price resumes (see ``_dry_audit``) — and (2) symmetric between arms,
since two arms entering the same closure at different tank levels can only
diverge in forgiven litres by a bounded, physical amount (see
``SYMMETRY_TOL_PER_GAP``). A dry event that fails either check is genuine cadence
stranding, and the ceiling stays invalid.

WINDOW LEVEL ONLY — do not slice any of this by cycle zone (fps-1785999730023-4-264564ac).

  PYTHONPATH=. uv run python experiments/2026-08-20_cadence_ceiling/model_cadence.py
"""
from __future__ import annotations

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
    PriceHistory,
    TankParams,
    _evaluation_dates,
    load_history,
    run_oracle_backtest,
)
from fuel_signal.backtest_phase2 import aggregate_backtest
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.features import load_features

HERE = pathlib.Path(__file__).resolve().parent
# fps-fii asks for 7/3/1. 2 is swept as well because 3d turns out to be
# STRUCTURALLY unusable, not merely noisy: at D = 10.71 L a reachable decide-level
# (7.14 L) sits above the 10% emergency floor yet cannot survive one interval, so
# the model can strand the tank and the oracle stops being a ceiling. 2d is the
# nearest cadence that is provably safe (verified in exact rational arithmetic over
# the reachable lattice), so it supplies the valid intermediate point the issue
# wanted 3d to be. 3d is still run and reported, flagged invalid, as the evidence.
CADENCES = (7, 3, 2, 1)
FIT_CADENCE = 7  # the cadence the harness pass runs at; the others reuse its fits

# Materiality floor for the run-dry audit, in litres — deliberately a PHYSICAL
# quantity (one millilitre), not a float epsilon. A genuine dry event is worth a
# whole interval's burn: >= 3.571 L even at the finest cadence swept here, i.e.
# ~3500x this threshold. What sits below it is replay drift, which shows up in the
# ORACLE arm specifically because run_oracle_backtest rounds every DP state level
# to 6dp per layer while this replay does not — at cadence 2 the wait chain lands
# on exactly 0.0 in exact arithmetic (confirmed in rational arithmetic: no
# reachable level lies in the open (floor*size, D) interval), and unrounded floats
# undershoot it by ~1.7e-6 L per step. An epsilon-scale threshold does NOT separate
# these two populations: 1e-6 still flags the drift. The audit also reports
# worst_dry_litres so the ~3500x separation is visible in the artifact rather than
# resting on the reader trusting this constant.
# The MODEL arm's replay is bit-faithful to run_backtest (which does no rounding
# either), so its counts are exact regardless.
DRY_TOL = 1e-3

# Symmetry tolerance for the darkness invariant, in litres — a PHYSICAL bound,
# picked the same way as DRY_TOL rather than as a fitted epsilon: the model and
# oracle arms buy on different schedules, so they can enter any one no-price gap
# at different tank levels, but each arm's level lives in [0, tank_size_litres]
# independently of what the other arm did — so the two arms' forgiven litres for
# that ONE gap can differ by at most one tank size (whichever arm entered fuller
# forgives less; entering at 0 vs entering full is the largest possible spread).
# Multiplied by a (fold, station) key's gap count below since separate closures
# each contribute their own independent tank-size's worth of possible divergence.
# Observed on real data (dry_audit.csv, pre-this-change): every cadence's
# model/oracle forgiven-litres gap is <15L against a single closure (one gap per
# key), well inside the one-tank-size bound this constant encodes.
SYMMETRY_TOL_PER_GAP = TankParams().tank_size_litres

# Same inner-OOF gotcha as #262/#259: fold 1's outer train is ~1825d, so the inner
# default (also 1825d) yields 0 folds. Kept identical so 7d reproduces 189.79.
INNER_FOLDS = {"train_min_days": 1095, "val_days": 90, "step_days": 90}

# Smoke knob: CADENCE_FOLDS=13,14 runs two folds instead of fourteen (~1min vs ~13min).
# Unset (the default) is the real run — every fold, which is what the writeup quotes.
_SUBSET = os.environ.get("CADENCE_FOLDS")
FOLD_SUBSET = [int(x) for x in _SUBSET.split(",")] if _SUBSET else None
# A smoke writes to its own subdir. Sharing filenames with the real run left a
# 1-fold model ledger sitting next to a 14-fold oracle ledger, which reads as a
# like-for-like comparison and is not one (caught in review of this experiment).
OUT = HERE / "smoke" if FOLD_SUBSET else HERE


def _dry_audit(
    fills: pd.DataFrame,
    plans: list,
    cadence: int,
    tank: TankParams,
    station_codes: list[int],
    history: PriceHistory,
) -> dict:
    """Replay run_backtest's own level arithmetic over a ledger, classifying every
    run-dry event as darkness (forgiven — matches run_backtest's own behaviour, the
    ceiling stays valid) or stranding (a real cadence failure, the ceiling is not).

    A "dry" event is a depletion step whose PRE-CLAMP level is negative: the engine
    silently does ``max(0.0, ...)`` there, so the driver covered more distance than
    the litres bought pay for. Post-#358 ``run_backtest`` and ``run_oracle_backtest``
    forgive a dry event caused by a no-price date identically (clamp-and-account, not
    prune), so a dry event no longer by itself implies the two arms disagree.

    A dry event is DARKNESS when the station has no resolvable price at that date,
    OR — mirroring ``run_backtest``'s ``gap_since_decide`` one-step grace exactly —
    no resolvable price at the immediately PRECEDING eval date: depletion is applied
    before the day's decide point, so the first day price resumes after a gap still
    absorbs one step of the gap's forgiveness even though a price is available that
    day (``run_backtest`` doesn't raise there either, for the same reason). Any other
    dry event has a price both today and yesterday — ``run_backtest`` would raise on
    it — so it's STRANDING.

    Also tracks each (fold, station) key's total forgiven litres and how many
    distinct no-price gaps it saw, so the caller can check the two arms' forgiven
    litres agree within the physical bound (``SYMMETRY_TOL_PER_GAP``) rather than
    assuming exact equality.

    Reconstruction is exact rather than approximate: on a BUY the engine sets the
    level to full and charges ``size - level`` litres, and on an emergency it sets
    the level to half and charges ``0.5*size - level``, so in both cases the post-
    fill level equals ``level + litres``.
    """
    size = tank.tank_size_litres
    depletion = tank.daily_consumption_litres * cadence
    dry_events = 0
    dry_litres = 0.0
    worst = 0.0
    steps = 0
    stranded_events = 0
    stranded_litres = 0.0
    litres_by_key: dict[tuple[int, int], float] = {}
    gaps_by_key: dict[tuple[int, int], int] = {}
    for p in plans:
        eval_dates = _evaluation_dates(p.val_start, p.val_end, cadence)
        for sc in station_codes:
            sub = fills[(fills["fold"] == p.fold) & (fills["station_code"] == sc)]
            by_date = dict(zip(sub["date"], sub["litres"], strict=True))
            level = 0.5 * size
            prev_price_none = False
            key = (p.fold, sc)
            for i, d in enumerate(eval_dates):
                price_now = history.station_price_at(sc, d)
                if price_now is None and not prev_price_none:
                    gaps_by_key[key] = gaps_by_key.get(key, 0) + 1
                if i > 0:
                    steps += 1
                    raw = level - depletion
                    if raw < -DRY_TOL:
                        dry_events += 1
                        dry_litres += -raw
                        worst = max(worst, -raw)
                        litres_by_key[key] = litres_by_key.get(key, 0.0) + (-raw)
                        if price_now is None or prev_price_none:
                            pass  # darkness: forgiven, exactly as run_backtest forgives it
                        else:
                            stranded_events += 1
                            stranded_litres += -raw
                    level = max(0.0, raw)
                if d in by_date:
                    level += by_date[d]
                prev_price_none = price_now is None
    return {
        "dry_events": dry_events,
        "dry_litres": dry_litres,
        "worst_dry_litres": worst,
        "depletion_steps": steps,
        "stranded_events": stranded_events,
        "stranded_litres": stranded_litres,
        "litres_by_key": litres_by_key,
        "gaps_by_key": gaps_by_key,
        # The a-priori condition: a reachable decide-level inside this open interval
        # is above the emergency floor yet cannot survive one interval.
        "gap_lo": tank.floor_fraction * size,
        "gap_hi": depletion,
    }


def _symmetry_excess_litres(model_audit: dict, oracle_audit: dict, tank_size: float) -> float:
    """Max, over (fold, station) keys, of how far the two arms' forgiven litres
    exceed the physical bound (see ``SYMMETRY_TOL_PER_GAP``). <= 0.0 means every
    key is within bound; a positive value names the worst violation in litres.
    """
    keys = set(model_audit["litres_by_key"]) | set(oracle_audit["litres_by_key"])
    worst = 0.0
    for k in keys:
        n_gaps = max(model_audit["gaps_by_key"].get(k, 0), oracle_audit["gaps_by_key"].get(k, 0), 1)
        bound = tank_size * n_gaps
        diff = abs(model_audit["litres_by_key"].get(k, 0.0) - oracle_audit["litres_by_key"].get(k, 0.0))
        worst = max(worst, diff - bound)
    return worst


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    station_codes = list(PREFERRED_STATIONS)
    plans = _plan_folds(df, {}, FOLD_SUBSET)
    print(f"[folds] {len(plans)} windows, {len(station_codes)} stations, "
          f"{len(BASELINE_COLUMNS)} cols ({BASELINE_FINGERPRINT})", flush=True)

    # --- The one paid fit pass: production baseline at 7d through the real harness --
    res = run_paired_realised_backtest(
        [ArmSpec("baseline", df)], BASELINE_COLUMNS, collect_fills=True, seed=42,
        inner_fold_params=INNER_FOLDS, tank=TankParams(evaluation_interval_days=FIT_CADENCE),
        fold_subset=FOLD_SUBSET,
    )
    cache = res.baseline_cache
    assert cache is not None, "held_tau=None should always yield a baseline_cache"
    print(f"[fit] harness pass done in {res.meta['total_wall_seconds']:.1f}s; "
          f"{len(cache.per_fold)} folds cached", flush=True)

    rows: list[dict] = []
    fold_rows: list[dict] = []
    fills: list[dict] = []
    audits: list[dict] = []

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

        m_spend = m_litres = a_spend = a_litres = o_spend = o_litres = 0.0
        for p in plans:
            cf = cache.per_fold[p.fold]
            model = aggregate_backtest(
                history,
                ModelStrategy(pipeline=cf["cal_pipe"], feature_columns=BASELINE_COLUMNS,
                              threshold=cf["own_tau"]),
                station_codes, p.val_start, p.val_end, tank, collect_fills=True,
            )
            always = aggregate_backtest(
                history, AlwaysBuyStrategy(), station_codes, p.val_start, p.val_end, tank,
            )
            fo_spend = fo_litres = 0.0
            for sc in station_codes:
                r = run_oracle_backtest(
                    history, sc, p.val_start, p.val_end, tank, collect_fills=True
                )
                if pd.isna(r.realised_cpl):
                    print(f"[warn] oracle NaN cadence={cadence} fold={p.fold} station={sc}",
                          flush=True)
                    continue
                fo_spend += r.total_spend_cents
                fo_litres += r.total_litres
                for fr in r.fills:
                    fills.append({
                        "cadence": cadence, "fold": p.fold, "arm": "oracle",
                        "own_tau": float("nan"), "date": fr.date,
                        "station_code": fr.station_code, "price": fr.price,
                        "litres": fr.litres, "spend_cents": fr.spend_cents,
                        "emergency": fr.emergency,
                    })

            for fr in model["fills"]:
                fills.append({
                    "cadence": cadence, "fold": p.fold, "arm": "model",
                    "own_tau": cf["own_tau"], "date": fr.date, "station_code": fr.station_code,
                    "price": fr.price, "litres": fr.litres, "spend_cents": fr.spend_cents,
                    "emergency": fr.emergency,
                })
            m_spend += model["total_spend_cents"]
            m_litres += model["total_litres"]
            a_spend += always["total_spend_cents"]
            a_litres += always["total_litres"]
            o_spend += fo_spend
            o_litres += fo_litres
            fold_rows.append({
                "cadence": cadence, "fold": p.fold, "own_tau": cf["own_tau"],
                "val_start": p.val_start, "val_end": p.val_end,
                "model_cpl": model["cpl"], "always_cpl": always["cpl"],
                "oracle_cpl": fo_spend / fo_litres if fo_litres > 0 else float("nan"),
            })

        model_cpl = m_spend / m_litres
        always_cpl = a_spend / a_litres
        oracle_cpl = o_spend / o_litres
        cad_fills = pd.DataFrame([f for f in fills if f["cadence"] == cadence])
        mf = cad_fills[cad_fills["arm"] == "model"]
        audit = {
            arm: _dry_audit(cad_fills[cad_fills["arm"] == arm], plans, cadence, tank,
                            station_codes, history)
            for arm in ("model", "oracle")
        }
        symmetry_excess = _symmetry_excess_litres(
            audit["model"], audit["oracle"], tank.tank_size_litres
        )
        audits.append({
            "cadence_days": cadence,
            "symmetry_excess_litres": symmetry_excess,
            **{f"{arm}_{k}": v for arm, a in audit.items() for k, v in a.items()
               if k not in ("litres_by_key", "gaps_by_key")},
        })
        ceiling_valid = (
            audit["model"]["stranded_events"] == 0
            and audit["oracle"]["stranded_events"] == 0
            and symmetry_excess <= 0.0
        )
        rows.append({
            "cadence_days": cadence,
            "always_cpl": always_cpl, "model_cpl": model_cpl, "oracle_cpl": oracle_cpl,
            "captured_cpl": always_cpl - model_cpl,        # value the model delivers
            "headroom_cpl": model_cpl - oracle_cpl,        # what an oracle still has
            "pot_cpl": always_cpl - oracle_cpl,            # total available
            "captured_pct": 100 * (always_cpl - model_cpl) / (always_cpl - oracle_cpl),
            "model_fills": len(mf),
            "model_emergency_pct": 100 * mf["emergency"].mean(),
            "model_mean_fill_litres": mf["litres"].mean(),
            "model_litres": m_litres,
            "model_dry_events": audit["model"]["dry_events"],
            "oracle_dry_events": audit["oracle"]["dry_events"],
            "ceiling_valid": ceiling_valid,
            "wall_s": time.perf_counter() - ct0,
        })
        r = rows[-1]
        print(
            f"[cadence {cadence:>2}d] always {always_cpl:.2f}  model {model_cpl:.2f}  "
            f"oracle {oracle_cpl:.2f}  captured {r['captured_cpl']:.2f} "
            f"({r['captured_pct']:.0f}%)  headroom {r['headroom_cpl']:.2f}  "
            f"fills {r['model_fills']} ({r['model_emergency_pct']:.0f}% emerg)  "
            f"dry {audit['model']['dry_events']}/{audit['model']['depletion_steps']} "
            f"{'VALID' if ceiling_valid else 'CEILING INVALID'}  "
            f"[{r['wall_s']:.1f}s]",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "model_cadence.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "model_cadence_per_fold.csv", index=False)
    pd.DataFrame(fills).to_parquet(OUT / "cadence_model_fills.parquet")
    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(OUT / "dry_audit.csv", index=False)

    print("\n=== all three arms vs decision cadence (window level, pooled) ===", flush=True)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
    print("\n=== run-dry audit (a cadence with stranded events or symmetry_excess_litres > 0 "
          "has an INVALID ceiling) ===", flush=True)
    print(audit_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
    print("\nReference (#262, 7d): always 193.41 / model 189.79 / oracle 188.14.", flush=True)

    meta_dir = OUT / "stage2_model"  # own dir so the two stages' meta.json do not collide
    meta_dir.mkdir(exist_ok=True)
    write_meta(meta_dir, {
        "stage": "2_model_arm",
        "git_sha": current_git_sha(),
        "cadences": list(CADENCES),
        "fit_cadence": FIT_CADENCE,
        "fit_reuse_rationale": (
            "_train_calibrate_select_tau takes no TankParams — cal_pipe and tau are "
            "cadence-invariant, so reusing the 7d fits at 3d/1d is exact."
        ),
        "n_folds": len(plans),
        "fold_subset": FOLD_SUBSET,
        "station_codes": station_codes,
        "seed": 42,
        "inner_fold_params": INNER_FOLDS,
        "baseline_fingerprint": BASELINE_FINGERPRINT,
        "harness_meta": res.meta,
        "summary": summary.to_dict("records"),
        "dry_audit": audit_df.to_dict("records"),
        "wall_seconds": time.perf_counter() - t0,
    }, baseline_columns=BASELINE_COLUMNS)
    print(f"[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
