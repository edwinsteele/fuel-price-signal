---
name: screen-and-arbiter-share-no-population
description: The WFCV screen and the realised arbiter share 0.71% of their rows — a network-wide screen delta doesn't forecast a five-station CPL delta.
metadata:
  type: project
---

The WFCV log-loss screen and the realised arbiter share 0.71% of their evidence. The screen scores every station (batch1: 798,649 rows, 714 stations); the arbiter replays only PREFERRED_STATIONS (261/414/429/585/18517) = 5,693 of those rows. realised.py trains on the full frame (train_df = a.df.loc[p.train_index]) and only the REPLAY is restricted, so the models are identical — it is the scoring population that differs, by 141x. Consequence measured on batch1/tgp_cycle_displacement (fps-e6i): per-fold corr(arbiter delta_cpl_own, screen delta_ll) is -0.096 paired / -0.176 published-convention over all 714 stations, and +0.478 / +0.543 with the screen restricted to the arbiter's five; sign agreement 7/14 and 5/14 vs 10/14 and 11/14. Population is the ONLY change (same model, folds, seeds, statistic); bootstrapped gap +0.574, 95% CI [+0.101, +1.033], P(gap>0)=0.991. The five stations are NOT special — their per-row gain sits 0.33 SE from the network-wide figure — they are just small enough that the effect cannot clear its own fold-clustered 2*SE even on log-loss, which is much smoother than realised CPL. So: a network-wide screen delta is not a forecast of a five-station CPL delta, and per-fold sign disagreement between them is the expected result, not a signal about the feature. fps-4z6 tracks reporting the restricted screen alongside.
