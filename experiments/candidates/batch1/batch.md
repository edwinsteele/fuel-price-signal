# Batch record — batch1

Written by `experiments.pipeline.redundancy` at screen time, before any bead was
filed. These are the numbers that justified filing; see that module for why the
gate sits where it does.

- Candidates: **5**
- Rows sampled: 2,084,203
- Cross-candidate |rho| gate: **0.85**
- Verdict: **PASS**

## Mechanism families (disclosure, never gated)

5 distinct family label(s) across 5 candidate(s).

| candidate | mechanism family |
|---|---|
| `lga_trough_propagation` | lead-lag-propagation |
| `network_move_breadth` | directional-network-breadth |
| `station_descent_dynamics` | station-own-price-dynamics |
| `stickiness_phase_saddle` | station-heterogeneity-interaction |
| `tgp_cycle_displacement` | wholesale-lead |

## Block R^2 against the existing column set

Flagged at 0.9; reported, never auto-rejected. Block figure is the mean of the candidate's member columns.

| candidate | n cols | block R^2 | max column R^2 |
|---|---|---|---|
| `lga_trough_propagation` | 3 | 0.685 | 0.855 |
| `network_move_breadth` | 3 | 0.522 | 0.671 |
| `station_descent_dynamics` | 3 | 0.408 | 0.642 |
| `stickiness_phase_saddle` | 2 | 0.450 | 0.722 |
| `tgp_cycle_displacement` | 2 | 0.574 | 0.583 |

## Pairwise |rho|

**Cross-candidate pairs are the hard gate. Within-candidate pairs are disclosure only** — a group whose members are related is usually what makes them one mechanism rather than three.

### Cross-candidate (gated)

| |rho| | a | b |
|---|---|---|
| 0.593 | `lga_trough_propagation.lga_trough_breadth_7d` | `network_move_breadth.network_fall_breadth_3d` |
| 0.545 | `lga_trough_propagation.lga_trough_breadth_7d` | `network_move_breadth.network_rise_breadth_3d` |
| 0.445 | `network_move_breadth.network_rise_breadth_3d` | `station_descent_dynamics.station_px_change_3d` |
| 0.440 | `network_move_breadth.network_fall_breadth_3d` | `station_descent_dynamics.station_px_change_3d` |
| 0.439 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `network_move_breadth.network_rise_breadth_delta_2d` |
| 0.426 | `network_move_breadth.network_rise_breadth_3d` | `station_descent_dynamics.station_px_change_14d` |
| 0.361 | `lga_trough_propagation.lga_trough_breadth_7d` | `station_descent_dynamics.station_px_change_3d` |
| 0.360 | `network_move_breadth.network_fall_breadth_3d` | `station_descent_dynamics.station_px_change_14d` |
| 0.336 | `lga_trough_propagation.lga_trough_breadth_7d` | `station_descent_dynamics.station_descent_decel` |
| 0.310 | `network_move_breadth.network_fall_breadth_3d` | `station_descent_dynamics.station_descent_decel` |
| 0.306 | `lga_trough_propagation.lga_trough_breadth_7d` | `network_move_breadth.network_rise_breadth_delta_2d` |
| 0.289 | `network_move_breadth.network_rise_breadth_delta_2d` | `station_descent_dynamics.station_descent_decel` |
| 0.285 | `network_move_breadth.network_rise_breadth_3d` | `station_descent_dynamics.station_descent_decel` |
| 0.278 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `station_descent_dynamics.station_px_change_14d` |
| 0.206 | `network_move_breadth.network_rise_breadth_delta_2d` | `station_descent_dynamics.station_px_change_3d` |
| 0.204 | `network_move_breadth.network_fall_breadth_3d` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.192 | `station_descent_dynamics.station_px_change_14d` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.166 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `station_descent_dynamics.station_descent_decel` |
| 0.161 | `station_descent_dynamics.station_px_change_14d` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.159 | `network_move_breadth.network_fall_breadth_3d` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.154 | `network_move_breadth.network_rise_breadth_3d` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.152 | `lga_trough_propagation.lga_leader_lead_days` | `station_descent_dynamics.station_px_change_14d` |
| 0.143 | `network_move_breadth.network_fall_breadth_3d` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.136 | `network_move_breadth.network_rise_breadth_delta_2d` | `station_descent_dynamics.station_px_change_14d` |
| 0.136 | `lga_trough_propagation.lga_leader_lead_days` | `network_move_breadth.network_rise_breadth_3d` |
| 0.132 | `network_move_breadth.network_rise_breadth_3d` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.130 | `network_move_breadth.network_rise_breadth_3d` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.124 | `lga_trough_propagation.lga_trough_breadth_7d` | `station_descent_dynamics.station_px_change_14d` |
| 0.106 | `lga_trough_propagation.lga_trough_breadth_7d` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.100 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `network_move_breadth.network_fall_breadth_3d` |
| 0.090 | `station_descent_dynamics.station_px_change_3d` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.090 | `lga_trough_propagation.lga_leader_lead_days` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.079 | `station_descent_dynamics.station_descent_decel` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.063 | `lga_trough_propagation.lga_leader_lead_days` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.059 | `station_descent_dynamics.station_px_change_3d` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.058 | `network_move_breadth.network_rise_breadth_delta_2d` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.044 | `station_descent_dynamics.station_px_change_3d` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.043 | `lga_trough_propagation.lga_leader_lead_days` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.042 | `stickiness_phase_saddle.abs_sticky_x_phase` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.040 | `station_descent_dynamics.station_px_change_14d` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.038 | `network_move_breadth.network_rise_breadth_delta_2d` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.037 | `lga_trough_propagation.lga_leader_lead_days` | `station_descent_dynamics.station_descent_decel` |
| 0.033 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `station_descent_dynamics.station_px_change_3d` |
| 0.032 | `lga_trough_propagation.lga_leader_lead_days` | `station_descent_dynamics.station_px_change_3d` |
| 0.031 | `network_move_breadth.network_rise_breadth_delta_2d` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.027 | `station_descent_dynamics.station_descent_decel` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.026 | `station_descent_dynamics.station_descent_decel` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.023 | `lga_trough_propagation.lga_leader_lead_days` | `network_move_breadth.network_rise_breadth_delta_2d` |
| 0.023 | `lga_trough_propagation.lga_leader_lead_days` | `network_move_breadth.network_fall_breadth_3d` |
| 0.021 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.021 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `network_move_breadth.network_rise_breadth_3d` |
| 0.020 | `stickiness_phase_saddle.abs_sticky_x_phase` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.019 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.016 | `lga_trough_propagation.lga_trough_breadth_7d` | `stickiness_phase_saddle.sticky_x_phase` |
| 0.015 | `lga_trough_propagation.lga_trough_breadth_7d` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.010 | `station_descent_dynamics.station_px_change_14d` | `stickiness_phase_saddle.sticky_x_phase` |
| 0.010 | `network_move_breadth.network_fall_breadth_3d` | `stickiness_phase_saddle.sticky_x_phase` |
| 0.009 | `lga_trough_propagation.lga_trough_breadth_7d` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.009 | `stickiness_phase_saddle.sticky_x_phase` | `tgp_cycle_displacement.tgp_cycle_displacement_frac_amp` |
| 0.008 | `network_move_breadth.network_rise_breadth_3d` | `stickiness_phase_saddle.sticky_x_phase` |
| 0.007 | `network_move_breadth.network_rise_breadth_delta_2d` | `stickiness_phase_saddle.sticky_x_phase` |
| 0.007 | `station_descent_dynamics.station_descent_decel` | `stickiness_phase_saddle.sticky_x_phase` |
| 0.005 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `stickiness_phase_saddle.abs_sticky_x_phase` |
| 0.002 | `stickiness_phase_saddle.sticky_x_phase` | `tgp_cycle_displacement.tgp_cycle_displacement_cents` |
| 0.002 | `station_descent_dynamics.station_px_change_3d` | `stickiness_phase_saddle.sticky_x_phase` |
| 0.001 | `lga_trough_propagation.lga_trough_breadth_delta_3d` | `stickiness_phase_saddle.sticky_x_phase` |
| 0.000 | `lga_trough_propagation.lga_leader_lead_days` | `stickiness_phase_saddle.sticky_x_phase` |

### Within-candidate (disclosure only)

| |rho| | candidate | a | b |
|---|---|---|---|
| 0.903 | `station_descent_dynamics` | `station_px_change_3d` | `station_descent_decel` |
| 0.853 | `tgp_cycle_displacement` | `tgp_cycle_displacement_cents` | `tgp_cycle_displacement_frac_amp` |
| 0.640 | `network_move_breadth` | `network_rise_breadth_3d` | `network_fall_breadth_3d` |
| 0.449 | `lga_trough_propagation` | `lga_trough_breadth_7d` | `lga_trough_breadth_delta_3d` |
| 0.400 | `station_descent_dynamics` | `station_px_change_3d` | `station_px_change_14d` |
| 0.362 | `network_move_breadth` | `network_rise_breadth_3d` | `network_rise_breadth_delta_2d` |
| 0.209 | `network_move_breadth` | `network_fall_breadth_3d` | `network_rise_breadth_delta_2d` |
| 0.090 | `lga_trough_propagation` | `lga_trough_breadth_delta_3d` | `lga_leader_lead_days` |
| 0.040 | `lga_trough_propagation` | `lga_trough_breadth_7d` | `lga_leader_lead_days` |
| 0.033 | `station_descent_dynamics` | `station_px_change_14d` | `station_descent_decel` |
| 0.024 | `stickiness_phase_saddle` | `sticky_x_phase` | `abs_sticky_x_phase` |
