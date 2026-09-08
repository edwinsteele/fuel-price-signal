---
name: cadence-not-a-free-knob
description: Oracle-vs-model headroom is only valid at cadences where no reachable tank level lands in the run-dry interval: 1, 2 and 7 days.
metadata:
  type: project
---

Cadence is not a free knob in the backtest engine. run_backtest CLAMPS a dry tank to 0 and continues (its emergency rule tests the CURRENT level, not the post-depletion overshoot, despite its comment claiming otherwise), while run_oracle_backtest PRUNES run-dry paths. So oracle-vs-model headroom is only meaningful where no reachable level lands in the open (floor*size, D) interval, D = daily_consumption * evaluation_interval_days. Default TankParams: SAFE at 1, 2 and 7 days; INVALID at 3, 4, 5, 6 and 8-14 (at 3d that is 109 dry events / 389 L and headroom goes NEGATIVE, which is impossible for a true ceiling). Check the STARTING level (50% of tank) as well as the wait-chain rungs -- README.md:756's documented CLI example fails there and nowhere else. Before trusting any non-default cadence, audit it empirically (replay the engine's own level arithmetic over the fill ledger) rather than reasoning about the lattice. Fix + merge gate: bd fps-5mn.
