# Round 13.883 — S6 merge semantics and S5–S9 gate

## Outcome

Round 13.883 passes.  The current code completes a fresh, strongly isolated
S5–S9 two-seed matrix: all ten 200-step episodes are persistable, all 2,000
planning attempts produce native joint trajectories, and no episode reports a
collision, out-of-road event, fallback, GT/mask conflict or committed-trajectory
failure.

The 0.1 rad committed heading envelope, PID gains, road boundary, dynamics and
5 m/7 m clearance contracts were not changed.

## S6 diagnosis

The pre-round failure was:

```text
scenario                         S6_background_merge_in
seed                             17
failure step                     170
failure                          committed_trajectory_tracking_deviation
agent                            agent0
execution elapsed                2.6 s
longitudinal error               0.021742 m
lateral error                    0.282880 m
heading error                    -0.103084 rad
heading limit                    0.1 rad
collision / out-of-road          false / false
```

The selected execution was an all-RIGHT lane change created at step 144 even
though RuleMaker's risk detector reported `risk_triggered=false`.  Its target
lane chain was valid, but the unnecessary maneuver entered a curved committed
path and crossed the heading envelope.  Five isolated repetitions with the
same seed, initial speed and scenario recipe selected KEEP and completed
200/200, showing that the nominal S6 KEEP path and the lateral PID were not the
root defect.  The unstable boundary was the generic MOBIL ranking being
allowed to turn sub-centimetre traffic-state variation into a no-risk lane
change.

S6 is a controlled longitudinal gap-creation scenario: its single merge actor
is inserted into the leader--middle gap, and the platoon yields or expands its
spacing.  RuleMaker now emits a joint KEEP proposal while the formation is
locked and the detector reports no risk.  If a real conflict is detected, the
existing unlocked joint-action search remains available.  KEEP is not a
trajectory fallback: Normal planner remains the final feasibility authority
and may reject it under the unchanged hard contracts.

Controller diagnostics now explicitly expose:

```text
actual_heading_rad
actual_yaw_rate_rad_s
preview_reference_heading_world_rad
first_reference_heading_world_rad
```

These fields separate world-path tangent, preview target and actual yaw
response without affecting the control output.

## S6 focused gate

```text
seed 17                         200 / 200, pass
seed 23                         200 / 200, pass
scenario triggered/realized     2 / 2
rule action                     KEEP for all three roles, all 400 steps
collision / out-of-road         0 / 0
failure                         none
```

Evidence:

```text
/tmp/bev-stage-census/round13_883_gate_s6_seed17/
/tmp/bev-stage-census/round13_883_gate_s6_seed23/
/tmp/bev-stage-census/round13_883_s6_seed17_trace/
```

## Fresh S5–S9 10 x 200 gate

Each episode ran in its own process, serially, with target speed 30 km/h and
the strict RuleMaker + Normal planner + adaptive control chain.

| Scenario | Seed | Steps | Native planning | Persistable | Failure | Planner P95 |
|---|---:|---:|---:|---:|---|---:|
| S5 | 17 | 200 | 200/200 | yes | none | 342.119 ms |
| S5 | 23 | 200 | 200/200 | yes | none | 335.254 ms |
| S6 | 17 | 200 | 200/200 | yes | none | 205.893 ms |
| S6 | 23 | 200 | 200/200 | yes | none | 204.240 ms |
| S7 | 17 | 200 | 200/200 | yes | none | 193.347 ms |
| S7 | 23 | 200 | 200/200 | yes | none | 184.548 ms |
| S8 | 17 | 200 | 200/200 | yes | none | 270.362 ms |
| S8 | 23 | 200 | 200/200 | yes | none | 266.379 ms |
| S9 | 17 | 200 | 200/200 | yes | none | 411.198 ms |
| S9 | 23 | 200 | 200/200 | yes | none | 420.255 ms |

The planner timings are offline expert-generation timings and are not assessed
against the learned model's 100 ms three-vehicle inference requirement.

Episode reports are stored under:

```text
/tmp/bev-stage-census/round13_883_full_<s5|s7|s8|s9>_seed<17|23>/
  <scenario>/metrices/episode_0000/expert_episode.json
/tmp/bev-stage-census/round13_883_gate_s6_seed<17|23>/
  S6_background_merge_in/metrices/episode_0000/expert_episode.json
```

## Regression

```text
focused RuleMaker/controller              100 passed
combined planner/controller/collection     224 passed
git diff --check                           passed
```

The combined suite covers RuleMaker, Normal planner, committed execution,
controller diagnostics, preview evaluation, joint collection, mode contract
and S7/S8 G-block geometry.

With this gate complete, the expert-chain prerequisite for diagnostic-64 is
satisfied.  Diagnostic-64 collection remains a separate long-running task and
was not started in this round.
