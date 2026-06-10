SEEDS = (42, 43, 44, 45, 46)
SHOCK_FOLDS = frozenset({1, 4, 9, 13})
# LightGBM params shared across all experiment scripts — do not redefine per-script.
LGBM_DEFAULTS: dict = {"verbose": -1, "subsample": 0.8, "subsample_freq": 1}
