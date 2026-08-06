# Round 13.8 — longitudinal reference separation and feedback execution

## Outcome

The explicit longitudinal-reference and feedback-executable rolling contracts
are implemented.  The original Round 13.76 failure
`outside_reachable_distance` is removed, but the mandatory S5 `2 x 80` smoke
still fails the strict lateral-acceleration audit at the first committed roll.
Round 13.8 therefore remains blocked and the ten-seed run was not started.

No RuleMaker action, target lane, lane-change path, scenario parameter, 5 m/7 m
gap, hard kinematic boundary, reward, diffusion behavior, or dataset schema was
changed.

## Implemented behavior

- `LongitudinalTrackingReference` separates fixed-time arc, speed and
  acceleration from lateral waypoints.
- Fixed-world simulator branches no longer convert position lag into extra
  target speed.
- The committed executor projects each vehicle onto the accepted spatial path,
  rolls a four-second profile from the actual speed/arc state, and re-runs the
  complete world/local, road, background and pairwise safety audit.
- The accepted execution id, spatial path, target lane and lane-change direction
  remain unchanged; active execution still generates no RuleMaker proposal.
- Expert PID, formation-locked follower control, simulator branch control and
  online trajectory control share acceleration feedforward plus speed/position
  feedback.  Follower gap feedback is bounded to `+/-1 m/s2`; it is absent in
  S5 emergency-independent control.
- STOP references remain stationary after zero speed and controller integrals
  use saturation anti-windup.

## Actuator response evidence

Independent S1 resets at 24 km/h produced monotonic 0.5-second average response:

```text
throttle +1.00: about +0.42 m/s2
throttle -1.00: about -2.60 m/s2
```

The first 0.1-second sample is zero for every tested command because the
simulator actuator has delay.  The later branch trace contains substantially
larger transient acceleration, so a static acceleration-to-throttle fit is not
sufficient to satisfy the longitudinal tracking thresholds.  No further gain
tuning was performed.

## Regression verification

The Round 13.71--13.76 controller, planner, collector, evaluator and diagnostic
suite passed before the final evidence run:

```text
204 passed
```

## Longitudinal benchmark

S1 constant-speed, acceleration, braking and STOP references were all valid
under the hard trajectory audit (`12/12`).  The controller benchmark did not
meet the closed-loop limits:

```text
clean longitudinal P95: 7.84 m       (required <= 1.0 m)
clean longitudinal P99: 8.15 m       (required <= 1.5 m)
maximum continuous saturation: 2.4 s (required <= 1.0 s)
STOP terminal speed: 0.0028 m/s       (required <= 0.3 m/s)
```

This confirms that reference separation fixes the semantic speed error, but the
current delayed actuator plus simple static feedforward mapping is not yet an
accurate longitudinal plant controller.

## S5 smoke and stop decision

Configuration:

```text
scenario: S5_hard_brake_lead / R1_entry_straight
seeds: [17, 23]
horizon: 80
```

Aggregate result:

- hard-brake event and immediate unlock: `2/2`;
- collision and simulator out-of-road rate: `0`;
- collection-ready labels before failure: `42/42`;
- persistable episodes: `0/2`;
- both failures occur at committed elapsed time `0.1 s`;
- both failures are `committed_trajectory_kinematic_infeasible` for agent0;
- `outside_reachable_distance`: `0`;
- maximum lateral acceleration: about `6.083 m/s2`, above the unchanged
  `6.0 m/s2` hard limit;
- tracking error at failure is small (about 8.5 mm longitudinal, 3.6 mm lateral,
  0.050 rad heading).

Evidence, including videos, BEV mosaics, trajectories and episode JSON, is under
`outputs/bev-round13-8/final-smoke/`.

The plan requires stopping if the remaining failure is spatial-path/lateral
execution rather than longitudinal reference semantics.  Therefore no further
speed reduction, curvature adjustment, lane-change edit, fallback, tolerance
increase, or `10 x 200` run was performed.
