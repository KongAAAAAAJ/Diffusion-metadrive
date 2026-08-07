# Round 13.882a/b: S5 determinism and S8 feasibility

## Result

Round 13.882a is complete.  S5 now has a deterministic scenario-local random
stream and fixed physical vehicle type for every controlled actor.  The strict
in-process seed-23/seed-17/seed-23 sandwich test compares every state, action
and control step and passes.

Round 13.882b resolves the reported S8 `all_rule_proposals_infeasible`
failure and the subsequent closed-loop heading/lateral tracking errors.  The
complete S8 episode gate remains blocked by a separate background-prediction
contract mismatch.  Diagnostic-64 collection must not start.

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
initial exit curve identified a planner/controller closed-loop executability
issue.  The dense 0.1 s path showed the root cause: a `-6 m/s2` braking
profile, 3.5 s lane change and zero start delay compressed the 3.6 m lateral
transition into roughly 8 m of spatial arc.  Its dense lateral acceleration
was about 8.8 m/s2, despite the eight 0.5 s waypoints nearly passing the 6
m/s2 contract.

The production planner now audits every 0.1 s chord transition against the
same yaw-rate, curvature and lateral-acceleration limits before a candidate
enters joint search.  The controller also guards the exit's initial
left-to-right curvature reversal: if the first reachable tangent and the
nominal long-lookahead tangent have opposite signs, preview is capped at the
first 0.5 s waypoint.  Steering sign itself was verified correct and was not
changed.

With both fixes, seeds 17 and 23 no longer fail heading (`0.137 rad`) or
lateral (`0.544 m`) tracking.  They progress deterministically to step 48 and
then stop on the unchanged 5 m background hard boundary:

```text
reason                         committed_trajectory_background_unsafe
agent                          agent1
planned minimum gap            9.103910 m
real committed-roll gap        4.915710 / 4.915704 m (seed 17 / 23)
collision                      false
```

The size of the planned-to-real gap change rules out the small residual
tracking error as the sole cause.  Background lane/route prediction versus
committed execution must be audited next.  No vehicle was removed or moved,
and the 5 m boundary was not relaxed.

## Evidence

```text
/tmp/bev-stage-census/round13_882ab_s5_s9_seed17_23_80.json
/tmp/bev-stage-census/round13_882b_s8_seed17_dense_probe_after_prod.json
/tmp/bev-stage-census/round13_882b_s8_fixed_seed17.json
/tmp/bev-stage-census/round13_882b_s8_fixed_seed23.json
/tmp/bev-stage-census/round13_882b_s8_dense_preview2_seed17.json
/tmp/bev-stage-census/round13_882b_s8_dense_preview_seed23.json
```

Focused dense-path and controller tests pass.  The remaining S8 background
prediction blocker must be resolved before rerunning the full `10 x 200` gate
or collecting diagnostic-64.

## 13.882b2: background predictor versus committed roll

The scalar gap diagnostic was replaced by an evidence-preserving result that
records the defining background actor, dense time index and both predicted
poses.  The committed executor additionally records the actor's current pose,
speed, lane, navigation successor and policy class.  Seeds 17 and 23 now give
the same decomposition (apart from floating-point noise):

```text
                                  seed 17       seed 23
new candidate, 4 s audit          9.103910 m    9.103897 m
committed preflight, elapsed 0     7.263989 m    7.263975 m
committed roll, elapsed 1.0 s      4.915710 m    4.915704 m
minimum dense time                 4.0 s         4.0 s
background speed                   17.999973     17.999978 km/h
```

All three measurements refer to the same explicitly injected S8 background
vehicle.  Its speed is continuous and the minimum is always at the end of the
rolling prediction window.  The drop therefore has two independent parts:

1. the feedback-executable committed preflight is already about 1.84 m less
   clear than the candidate geometry used during joint selection;
2. advancing the four-second window by one second exposes another roughly
   2.35 m of closing motion while the committed longitudinal governor advances
   the terminal ego pose more slowly.

The predictor did contain a real junction bug: `_get_continuation_lane()`
mixed `navigation.next_ref_lanes` with every graph successor and selected the
geometrically closest connector.  At the S8 G-block this chose
`4G0_0_ -> 4G0_1_ lane 2` although the background policy route specifies
`4G0_0_ -> 4G1_1_ lane 0`.  Continuation selection is now route-first, with
graph geometry used only when navigation has no continuous successor.

That correction does not change the reported 9.10/7.26/4.92 m values because
the background vehicle remains on its current lane throughout the four-second
failure horizon.  It rules out lane/route prediction as the direct cause.
The remaining blocker is an acceptance-contract mismatch: the Normal planner
ranks the original candidate geometry, preflights only the first executable
four-second roll, and does not recursively audit the future rolling windows
that a committed maneuver will expose.  Fixing that requires a separate
planner/executor feasibility change; changing traffic, control gains or the
5 m boundary would hide rather than resolve it.

Evidence:

```text
/tmp/bev-stage-census/round13_882b2_contract_seed17.json
/tmp/bev-stage-census/round13_882b2_contract_seed23.json
```

The 13.882b2 predictor investigation is complete.  S8 episode acceptance is
still blocked, so the `10 x 200` gate and diagnostic-64 remain intentionally
disabled.

## 13.882b3: full committed rolling-horizon admission audit

The committed executor's feedback-executable window construction is now a
single shared implementation.  Both online `roll()` and plan admission use
the same:

- curvature-derived speed limit;
- longitudinal feedback governor;
- float32 world/local trajectory reconstruction;
- spatial-path resampling.

Before any commitment is registered, admission recursively advances an ideal
tracked state in 0.1 s decision steps from `elapsed=0` through
`completion_deadline`.  At every state it audits the next four-second window
against road footprint, absolute-time background predictions, 5 m background
clearance, three-agent OBB collision and 7 m platoon clearance.  The union of
the windows therefore ends at `completion_deadline + 4 s`; it no longer ends
at the first candidate's four-second horizon.

A deterministic regression constructs a slower ego trajectory with a closing
background actor for which the first four-second window remains above 5 m but
the fifth second falls below 5 m.  The ordinary first `roll()` passes and the
full admission audit rejects it before execution.  A matched-speed version
checks all 41 windows and passes with coverage through 8.0 s.

On the real S8 state, the old plan is now rejected at step 38 before an active
execution is created:

```text
                                  seed 17       seed 23
proposal step                     38            38
full-audit window elapsed         0.9 s         0.9 s
unsafe actor                      agent0        agent0
minimum background gap           4.476696 m    4.476682 m
relative dense index              28            28
full coverage endpoint            10.5 s        10.5 s
result                            proposal rejected before commitment
```

The RuleMaker currently supplies only the joint RIGHT exit proposal at this
state, so the externally visible result is the strict typed rejection
`all_rule_proposals_infeasible`.  This is the intended 13.882b3 behavior: the
system no longer accepts the plan and then fails ten control steps later.
It does not claim that S8 has become feasible.  A later S8 feasibility round
must search/select a different native trajectory or adjust the explicit S8
traffic design; it must not bypass this audit or relax the 5 m boundary.

Evidence:

```text
/tmp/bev-stage-census/round13_882b3_seed17_v3.json
/tmp/bev-stage-census/round13_882b3_seed23.json
```
