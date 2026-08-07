# Round 13.882a/b: S5 determinism and S8 feasibility

## Result

Round 13.882a is complete.  S5 now has a deterministic scenario-local random
stream and fixed physical vehicle type for every controlled actor.  The strict
in-process seed-23/seed-17/seed-23 sandwich test compares every state, action
and control step and passes.

Round 13.882b resolves the reported S8 `all_rule_proposals_infeasible`
failure, but the complete S8 episode gate remains blocked by a separate
closed-loop heading-tracking error.  Diagnostic-64 collection must not start.

## 13.882a: S5

The divergence first appeared only after the hard-brake trigger.  Two shared
global-state dependencies were removed:

- hard-brake and adjacent-vehicle recipe parameters now use a private seed
  derived from `(episode seed, scenario id, route)`;
- S5 controlled traffic uses `TrafficDefaultVehicle` instead of consuming the
  traffic manager's random vehicle-type stream.

Episode summaries now persist the derived scenario seed and all realized S5
brake parameters.  Consuming arbitrary values from the traffic manager RNG no
longer changes these values.

Acceptance:

```text
tests/test_bev_episode_determinism.py       1 passed (164.13 s)
S5 seed 17, 80 steps                       pass
S5 seed 23, 80 steps                       pass
```

## 13.882b: S8

The original production lattice rejected the exit at step 39 even though its
closest background gaps were 4.89--4.95 m.  An exact-state feasibility probe
kept the road footprint, kinematics, 5 m/7 m gaps and OBB checks unchanged and
found a safe joint plan.  The missing production dimensions were:

- intermediate braking accelerations including `-5 m/s2`;
- 0.5 s braking-duration resolution, including 1.5 s and 2.5 s;
- a `-2 m/s2` recovery segment;
- sufficient longitudinal-profile and local-pool coverage.

These values are enabled only for a non-KEEP S8 action.  Generic planner
search remains unchanged.  S8 also now freezes random traffic density at zero
and uses only its two explicitly configured lane-2 vehicles with a fixed
physical type.

After the change, seeds 17 and 23 are exactly reproducible and both start the
same native joint execution at step 38.  Neither reports
`all_rule_proposals_infeasible`.  Both then stop at step 41 with the identical
independent blocker:

```text
reason                         committed_trajectory_tracking_deviation
agent                          agent1
elapsed execution time         0.3 s
longitudinal error             0.145428 m
lateral error                  0.039059 m
heading error                  0.137151 rad
heading limit                  0.1 rad
```

The low spatial error combined with the heading-limit violation during the
initial exit curve identifies a planner/controller closed-loop executability
issue.  It is not evidence that the S8 traffic layout is dynamically
infeasible.  Therefore no vehicle was removed or moved and no safety boundary
was relaxed.

## Evidence

```text
/tmp/bev-stage-census/round13_882ab_s5_s9_seed17_23_80.json
/tmp/bev-stage-census/round13_882b_s8_seed17_dense_probe_after_prod.json
/tmp/bev-stage-census/round13_882b_s8_fixed_seed17.json
/tmp/bev-stage-census/round13_882b_s8_fixed_seed23.json
```

Focused planner/orchestrator/controller tests pass (`116 passed`), in addition
to the real sandwich determinism test.  The remaining S8
heading-tracking blocker must be handled in a separate controller/path
executability round before rerunning the full `10 x 200` gate or collecting
diagnostic-64.
