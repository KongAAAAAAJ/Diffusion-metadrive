# Round 13.881a — route-chain spatial geometry

## Outcome

The shared execution-path contract is implemented and the former S6/S8
`spatial path arc buffer is exhausted` failure is removed.  The round is not
declared fully accepted because the real S8 map exposes a finite-width road
geometry obstruction at the lane-2/exit-connector seam, and the existing
tracking contract stops the initial lane-0 to lane-1 commitment at 0.8 s.

No road, 5 m/7 m clearance, kinematic, or controller threshold was relaxed.

## Implemented contract

- S8 RIGHT remains one adjacent lane per maneuver:
  `lane0 -> lane1 -> lane2`.
- The incorrect upstream branch `3C0_1_ -> 4G1_0_` was removed from both
  RuleMaker and Normal planner target resolution.
- Once on lane 2, KEEP follows the navigation successor chain:
  `3C0_1_->4G0_0_ -> 4G0_0_->4G1_1_ -> 4G1_1_->4G1_2_ -> ...`.
- RuleMaker passes complete source and target lane chains to Normal planner.
- The candidate-pool cache key includes both immutable chains.
- `TrajectoryExecutionSpec` now separates:
  - nominal timed trajectory and `reference_arc_m`;
  - extended `spatial_path_world` and `path_arc_m`.
- The executor uses the nominal arc only for `s_ref/v_ref/a_ref`, while path
  projection, curvature limits, and spatial sampling use the extended path.
- Candidate and executor road audits receive the complete frozen chain.
- A non-mutating first-roll execution preflight runs before a proposal can be
  accepted; a failed preflight continues to the next ranked proposal.

## Real S8 geometry evidence

Navigation checkpoints for seeds 17/23 are:

```text
3C0_1_ -> 4G0_0_ -> 4G1_1_ -> 4G1_2_ -> 4G1_3_ -> ...
```

Main-road lane centers at their endpoint are respectively 10.5 m, 7.0 m, and
3.5 m from the exit connector start.  Therefore only lane 2 is the exit
approach; lane 0/1 commitments must finish on the current road first.

The constructed lane-2 route path has:

```text
arc length       295.678 m
maximum spacing    0.250 m
maximum curvature  0.131 1/m
```

It is continuous and uses the correct connector.  However, the lane-2 surface
ends exactly where the connector surface starts one lane width to the right.
Their finite-width surfaces have no longitudinal overlap.  The strict XL
footprint audit consequently rejects a corner at this seam even when all map
lanes are included.  Resolving that requires a map/junction drivable-surface
change, not a planner safety-boundary relaxation.

## Simulator smoke

### S8 seeds 17/23, 2x80

Output:

```text
/tmp/bev-round13-881a-s8-smoke2
```

Both episodes successfully create the correct adjacent-lane execution and run
eight committed rolls.  Neither uses the wrong branch or exhausts the spatial
buffer.  Both stop at step 8 because agent1 nominal longitudinal tracking lag
reaches 1.024 m against the existing 1.0 m threshold.  This round did not tune
the controller or widen the tracking envelope.

### S6 seeds 17/23, 2x200

Output:

```text
/tmp/bev-round13-881a-s6-smoke2
```

- seed 17 reaches 194 steps, then reports `all_rule_proposals_infeasible`;
- seed 23 reaches 198 steps, then a first committed roll is rejected for
  `yaw_rate_limit` (maximum curvature remains 0.232 1/m and lateral
  acceleration 4.97 m/s2).

The former seed-17 failure near step 155 was `spatial path arc buffer is
exhausted`; it no longer occurs in either run.  Thus the common S6 committed
execution foundation is fixed, while its late decision/dynamic feasibility is
separate follow-up work.

## Verification

Core regression:

```text
135 passed
```

The broader selected regression produced `179 passed, 2 failed`.  The two
failures are legacy semantic-anchor acceptance tests whose generator returns
`[9,8,2]` anchors while those tests expect heading in `[8,3]`; none of the
modified route/planner files participate in that generator.

`py_compile` and `git diff --check` pass.

## Stop condition

Further automatic tuning was stopped.  Full S8 acceptance now needs an explicit
choice to correct the G-block junction drivable geometry (or redefine the hard
road-footprint contract).  This round intentionally does neither.
