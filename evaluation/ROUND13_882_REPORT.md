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

## 13.882b4: candidate backtracking and controlled S8 feasibility

The planner now treats full committed-horizon admission as a candidate-level
decision inside a RuleMaker proposal.  If the lowest-cost short-horizon-safe
joint selection fails execution geometry or the recursive committed audit,
only that selection is excluded and the next safe selection is tried.  A
rejected selection cannot create a lane-change commitment or a fallback
trajectory.

Exhaustive evidence on the original S8 layout showed that this was necessary
but insufficient: all 208 short-horizon-safe selections at seed 17 failed the
full contract (202 background-clearance failures and 6 platoon-pair failures).
Moving only the front actor from `s=50 m` to `s=70 m` still exhausted all 76
selections (42 background and 34 pairwise failures).  Since the planner lattice
was exhausted under the unchanged 5 m/7 m boundaries, the controlled S8
background window was changed once to rear/front `s=5/80 m`, both at 18 km/h.

This layout admits a native joint exit plan and removes the former 9.10 m to
4.92 m committed-background failure.  During the resulting atomic execution,
two state/label inconsistencies were also corrected without changing the
trajectory:

- a commitment now remains active until the vehicle centre is within 0.25 m
  of the accepted target centreline, rather than merely having its footprint
  enter the target lane;
- once MetaDrive assigns a vehicle to the accepted target lane family, the
  retained segment is labelled KEEP, while the execution commitment remains
  active until geometric convergence.  Repeating LEFT/RIGHT at that point
  would mean a second lane change to the dynamic-anchor topology.

Seeds 17 and 23 both pass through step 70 with no collision, out-of-road,
fallback, committed safety failure or GT/mask conflict:

```text
/tmp/bev-stage-census/round13_882b4_step70_seed17.json
/tmp/bev-stage-census/round13_882b4_step70_seed23.json
```

The 80-step gate is not yet complete.  Near the commitment completion boundary
the current exclusion implementation repeatedly rebuilds the same pairwise
candidate conflict tables and did not return within the bounded diagnostic
run.  The run was stopped rather than weakening safety or making a second S8
traffic adjustment.  Round 13.882b4 therefore has a verified functional fix
through the original failure interval, but remains blocked on efficient,
semantics-preserving exhaustive full-horizon search before the `2 x 80`,
`10 x 200`, or diagnostic-64 gates can be claimed.

## 13.882b4.1: cached full-horizon joint search

The repeated-exclusion loop has been replaced by one cost-ordered enumeration
of the short-horizon-safe joint selections. Admission now maintains two strict
caches:

- `(agent, candidate, completion deadline)` stores the exact
  feedback-executable rolling windows plus kinematic, road, tracking and 5 m
  background audit result;
- `(candidate A, candidate B, completion deadline)` stores the OBB/7 m
  pairwise result over those same windows.

The ordinary committed executor's full-horizon audit now calls the same
candidate and pairwise audit primitives. A failed cache entry is reusable as
well as a successful one. Diagnostics retain aggregate rejection counts and
only a bounded example trace instead of serializing thousands of duplicate
failures.

The current-code seed-17 gate now returns deterministically at step 71 instead
of remaining inside repeated search. RuleMaker supplies one joint LEFT
proposal containing 4,821 short-horizon-safe combinations. Under the
executor-identical initial tracking audit, all 4,821 contain at least one
candidate outside the 0.1 rad heading envelope:

```text
candidate audit requests       9,639
candidate audits computed        127
candidate cache hits            9,512
full-horizon result             infeasible
failure                         committed_trajectory_tracking_deviation
```

Evidence:

```text
/tmp/bev-stage-census/round13_882b41_final_seed17_72.json
/tmp/bev-stage-census/round13_882b41_seed23_80.json
```

The seed-23 run independently reaches the same step, action tuple and 4,821
combination count. It was produced immediately before moving the initial
tracking predicate into the candidate cache, and therefore shows the same
infeasibility decomposed later as pairwise/preflight failures. The final
seed-17 run is the authoritative current-code result.

Consequently the search/caching objective of 13.882b4.1 is complete, but the
S8 `2 x 80` behavioral gate is not: it is `0/2`, with the first mandatory seed
already failing at step 71. Per the gate order, `10 x 200` and diagnostic-64
were not started. The next fix must investigate why the post-exit RuleMaker
requests a new all-LEFT maneuver while the three vehicles still differ from
that path's preview heading by more than 0.1 rad; it must not undo the cache,
relax the tracking envelope, or change the 5 m/7 m boundaries.

## 13.882b4.2: S8 decision and route-path semantic closure

The step-71 failure was a decision/path contract error rather than missing
planner search capacity.  All vehicles were already on the rightmost source
lane `(3C0_1_, 4G0_0_, 2)`.  KEEP follows that lane's navigation continuation
into the exit connector `(4G0_0_, 4G1_1_, 0)`, whereas the selected LEFT action
returned to the mainline continuation `(4G0_0_, 4G0_1_, 1)`.  Three corrections
close that semantic gap without changing any safety threshold:

- a lane-change commitment completes only after target-family membership,
  centreline convergence, footprint containment and target-lane heading
  convergence; merely entering the target lane no longer permits an immediate
  reverse decision;
- S8's rightmost-lane KEEP is represented as a required route action, and an
  apparent collision in RuleMaker's coarse ranking trajectory no longer
  deletes the only route-correct proposal.  The proposal is deferred to the
  Normal planner, which remains the sole final authority for dense footprint,
  dynamics, 5 m/7 m clearance and three-vehicle OBB safety;
- route-chain candidates and committed spatial paths anchor their initial
  tangent to the measured vehicle heading even when a MetaDrive lane omits the
  optional seam-transition marker.  This removes the artificial heading jump
  observed near the source-lane/exit-connector seam while retaining all hard
  kinematic and road audits.

The preview evaluator also now pins the expert contract's 30 km/h target speed.
Previously its implicit environment default was 90 km/h, which produced an
unrelated deterministic `committed_trajectory_deadline_missed` failure and did
not match the collection/probe configuration.

Diagnostic evidence for the original wrong LEFT decision and the corrected
route KEEP proposal is stored at:

```text
/tmp/bev-stage-census/round13_882b42_s8_rule_semantics_seed17.json
/tmp/bev-stage-census/round13_882b42_s8_route_keep_diag_seed17.json
```

The isolated-process gates now pass:

```text
2 x 80 seeds                       17, 23
10 x 200 seeds                     17,23,31,47,59,71,83,97,109,127
persistable episodes               10 / 10
native joint planning              2000 / 2000 (100%)
collision / out-of-road            0 / 0
failure reason                     none for all ten episodes
```

The combined RuleMaker, Normal planner, preview evaluator, joint collection,
mode-contract and S7/S8 G-block regression suite reports `206 passed`.
`git diff --check` is clean.

Each 200-step report is available under:

```text
/tmp/bev-stage-census/round13_882b42_gate_v3_seed<SEED>/
  S8_ego_exit_to_ramp/metrices/episode_0000/expert_episode.json
```

This completes the S8 decision/path semantic gate.  It does not relax the
tracking envelope, road boundary, dynamics, 5 m background clearance or 7 m
platoon clearance, and it does not add a synthetic trajectory fallback.
