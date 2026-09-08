---
name: decide-feature-parity-gap
description: backtest.py's decide() recomputes features independently of features.py — a new feature family needs wiring in both.
metadata:
  type: project
---

fuel_signal/backtest.py's ModelStrategy.decide() (the realised backtest's live replay path) recomputes features from PriceHistory independently of fuel_signal/features.py's offline row-construction path (used by the WFCV screen). A new engineered-feature family (LGA/brand/network/TGP troughs etc.) landing in features.py does NOT automatically work in decide() -- it needs its own loop/cache wiring there too. This bit fps-3i7 (2026-08-18, PR #306): brand-trough columns worked fine offline for 3 months (commit 3f2dd29) but decide() only ever looped LGA_FEATURE_COUNCILS, so any feature_columns set including a brand-trough column (e.g. batch_freeze.py's resolve_baseline_columns()) aborted the realised backtest's baseline arm on the very first decide() call -- candidate-independent, since R0 carries no extra_feature_provider. When adding a new feature family, check decide() parity, not just features.py.
