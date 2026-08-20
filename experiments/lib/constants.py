from fuel_signal.features import LOCKED_FEATURE_COLUMNS, LOCKED_FEATURE_FINGERPRINT

SEEDS = (42, 43, 44, 45, 46)
SHOCK_FOLDS = frozenset({1, 4, 9, 13})
# LightGBM params shared across all experiment scripts — do not redefine per-script.
LGBM_DEFAULTS: dict = {"verbose": -1, "subsample": 0.8, "subsample_freq": 1}

# The locked production baseline (R0), re-exported from fuel_signal.features so an
# experiment script has one import for every shared constant (fps-zci).
#
# NEVER retype the group composition `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS +
# NETWORK_FEATURE_COLUMNS`, and never re-derive R0 by inspecting a features frame.
# Both failure modes have already cost a batch: fps-sa1 (discovery pulled the
# rejected Phase 4b brand troughs into a 64-column R0) and fps-zci (a sorted
# permutation of the right 54 columns fit a different model, worth 0.038 c/L on
# the arbiter). ORDER IS PART OF THE CONTRACT — do not sort this.
# A copy, not an alias: an in-place mutation here (`.append`, `.sort`) would
# otherwise reach straight through into fuel_signal's canonical list and corrupt the
# contract for every importer in the process — silently, since nothing re-reads the
# artifact mid-run.
BASELINE_COLUMNS: list[str] = list(LOCKED_FEATURE_COLUMNS)

# Identity of BASELINE_COLUMNS as '<n>:<sha12>', hashed over the ORDERED list.
# Stamp it into every meta.json (experiments.lib.io.write_meta does this for you)
# so two runs' comparability is a mechanical check, not an eyeball one.
BASELINE_FINGERPRINT: str = LOCKED_FEATURE_FINGERPRINT

# Attached to every economics figure cut on a per-row label (a candidate's add_axis)
# rather than on folds. Pooled CPL is a path-coupled total — a buy now changes what is
# possible later — so allocating it to a sub-period has no unique answer. This is the
# defect that withdrew every per-zone row of #262 (experiments/2026-08-20_headroom_
# attribution/, bd fps-1785999730023-4-264564ac) and the 2026-06-18 gate-1 per-regime
# saving% (experiments/2026-08-21_path_coupling_audit/, bd fps-grp). Fold cuts are safe
# for a mechanical reason: realised.py runs aggregate_backtest once per fold and
# aggregate_backtest runs run_backtest once per station with a fresh tank, so a fold-cut
# number is a sum of complete windows; a row label is a slice through one.
ROW_AXIS_ECONOMICS_CAVEAT: str = (
    "NOT IDENTIFIED — this cell allocates a path-coupled cost (pooled realised CPL) to a "
    "sub-period selected by a per-row label. That allocation has no unique answer: free "
    "bookkeeping conventions can move a cell further than the cells differ from each other, "
    "and it is a bias term, so more folds/stations/seeds do not shrink it. Report as colour, "
    "never as a finding. Fold-cut economics (per_fold, per_regime=SHOCK_FOLDS) are unaffected "
    "— each (fold, station) is an independent simulation with its own tank. "
    "See docs/CONVENTIONS.md § 'Bucketed results — check the convention spread before "
    "believing an ordering' and experiments/2026-08-21_path_coupling_audit/."
)
