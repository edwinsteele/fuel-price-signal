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
