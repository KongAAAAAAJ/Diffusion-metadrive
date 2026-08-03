# Round 13.73: S5--S9 Expert Preview and Collection Audit

## Scope

The evaluation uses the strict three-agent production chain:

```text
RuleMaker -> PlatoonNormalPlanner -> adaptive low-level controller
          -> JointBEVSampleBuilder contract checks
```

Each run uses sensorless simulator state. Camera and LiDAR observations are not
created. The fixed seeds are `17, 23, 31, 47, 59, 71, 83, 97, 109, 127`, with
a 200-step horizon and no retry substitution.

Artifacts:

- Baseline: `/media/kong/Elements_SE/Diffusion_Data/outputs/bev-round13-73/baseline`
- Post-fix: `outputs/bev-round13-73/postfix`
- Each evaluation contains 50 trajectory NPZ files, 50 top-down MP4 files,
  50 semantic-BEV MP4 files, per-episode JSON, and scenario/root JSON+CSV
  summaries.

## Metrics

`planning success` is the successful native joint plans divided by all planner
attempts. `collection-ready success` additionally requires dynamic anchors,
hard masks, expert trajectory kinematics, and GT labeling to pass. An episode
is persistable only if it runs without collision/out-of-road/fallback and has
no collection rejection.

### Baseline

| Scenario | Collision | Planning success | Collection-ready success | Persistable | Mean wall time | Mean sim time |
|---|---:|---:|---:|---:|---:|---:|
| S5 | 0% | 97.773% | 100.000% | 0% | 12.982 s | 4.39 s |
| S6 | 100% | 100.000% | 100.000% | 0% | 4.367 s | 1.40 s |
| S7 | 0% | 99.182% | 99.820% | 0% | 45.443 s | 12.12 s |
| S8 | 0% | 97.647% | 74.921% | 0% | 16.035 s | 4.15 s |
| S9 | 0% | 99.816% | 96.081% | 0% | 48.692 s | 16.31 s |

The original S6 summary used a generic `crash` label. Focused diagnostics
confirmed that the deterministic early failures were sidewalk/road-boundary
collisions rather than vehicle collisions.

### Post-fix

| Scenario | Collision | Planning success | Collection-ready success | Persistable | Mean wall time | Mean sim time |
|---|---:|---:|---:|---:|---:|---:|
| S5 | 0% | 97.722% | 96.353% | 0% | 16.912 s | 4.29 s |
| S6 | 0% | 99.660% | 97.584% | 0% | 42.614 s | 14.66 s |
| S7 | 0% | 98.705% | 92.145% | 0% | 30.530 s | 7.62 s |
| S8 | 0% | 98.911% | 68.688% | 0% | 38.243 s | 9.08 s |
| S9 | 10% | 100.000% | 99.837% | 60% | 47.415 s | 19.38 s |

The S6 wall time increase is expected: the baseline terminated after 1.4 s of
simulated time, while half of the post-fix episodes reached the 20 s horizon.

## Implemented corrections

1. Restored the complete RuleMaker joint score. Progress, lane-change costs,
   traffic clearance, platoon-pair safety, formation consistency, and close-pair
   terms are active again.
2. Replaced RuleMaker's stationary-current-position traffic scoring with
   timestamp-aligned, lane-following constant-speed prediction. The dense
   Normal planner predictor remains the hard safety authority.
3. Added arc-length preview to the simple lateral PID while retaining the
   original conservative PID gains.
4. Corrected the shared reachability and dynamic-anchor integration when a
   vehicle stops or reaches the speed cap inside a 0.5 s sample interval.
5. Exposed explicit failure classes, planning diagnostics, RuleMaker state,
   rejection details, collision agents, per-episode runtime, and artifact
   paths in the evaluator.
6. Restored top-down warning markers and selected/candidate trajectory overlays.

## Scenario findings

### S5: hard braking with occupied adjacent lanes

- All 10 episodes end with `normal_planner_no_safe_joint_combination` at
  simulator steps 40--46.
- Local trajectories exist for all three vehicles. At the terminal failure,
  every tested joint combination conflicts, so expanding the single-agent
  longitudinal search is not the missing capability.
- Video and state logs show that repeated RuleMaker lane decisions and unequal
  tracking progress distribute the three platoon vehicles across different
  lanes. The final KEEP decision cannot recover a collision-free four-second
  joint plan.
- The configured adjacent actors do not prevent an earlier lane-change plan for
  the entire horizon. The expert lacks maneuver commitment/hysteresis and a
  synchronized platoon lane-change completion constraint.

Conclusion: S5 has a RuleMaker/closed-loop synchronization problem, not a
Normal planner local-range problem. The next change should either add explicit
joint maneuver commitment or revise adjacent-vehicle placement so that the
intended braking solution is the only safe action. Safety gaps should not be
relaxed.

### S6: controlled background merge

- Collision rate improves from 10/10 to 0/10 after time-aware traffic scoring
  and preview control.
- Five episodes reach all 200 steps. Five terminate with a local candidate
  failure: four are lane-end urgent changes whose candidates violate yaw-rate
  or lateral-acceleration limits, and one is blocked by the controlled merge
  vehicle's hard corridor.
- The five full-horizon episodes are not persistable because curved-lane STOP
  anchors are rejected by `outside_reachable_distance`. The generator advances
  by lane arc length while the fixed eight-point validator measures Euclidean
  chord length; the small deficit exceeds the strict 1 mm tolerance.

Conclusion: the original S6 crash was primarily caused by incorrect RuleMaker
future-traffic scoring. Remaining failures are a lane-end action timing issue
and a curved-lane anchor distance representation issue. Do not loosen 5 m/7 m
safety gaps.

### S7: ego platoon merges from ramp

- All 10 episodes fail with agent2 having no local candidate.
- The leader and middle vehicle enter the mainline first. The rear vehicle does
  not receive the route-forced LEFT action until it has only a few metres of
  source lane remaining; every urgent completion violates the lane-end deadline.
- Collection also records repeated LEFT action versus hard-mask conflicts for
  agent2 before the final planner failure.

Conclusion: this is a route-trigger synchronization error. Increasing the
generic Normal planner search range cannot make a physically late merge valid.
The forced-merge trigger must account for the rear vehicle and reserve enough
distance for all three roles.

### S8: platoon exits to ramp

- All 10 episodes fail with one local pool empty: agent1 in 7 episodes and
  agent0 in 3.
- The exit action is forced while the three vehicles occupy different route
  segments. Candidate rejection is dominated by curvature/yaw/lateral
  acceleration, road footprint, and background-corridor checks.
- GT action/hard-mask conflicts are frequent: collection-ready success is only
  68.688%.

Conclusion: S8 has the same class of staggered route-action problem as S7,
plus a route-topology/mode-mask mismatch on the branch connector. It should be
fixed at the joint route-action and topology contract, not by accepting
kinematically invalid candidates.

### S9: narrow channel negotiation

- Six of ten episodes are fully persistable.
- Three episodes complete the horizon but reject one final expert trajectory
  for `acceleration_below_min`.
- Seed 47 has one `crash_sidewalk:agent1` at step 137 near the narrow-channel
  exit connector.
- KEEP remains selected for the whole representative successful episode; the
  earlier GT/mask mode conflict is removed.

Conclusion: S9 is the only current scenario close to collection readiness.
The remaining work is a connector-specific tracking/final-trajectory audit,
not RuleMaker mode selection.

## Acceptance conclusion

Round 13.73 evaluation infrastructure and the requested 50-episode visual/data
audit are complete. The expert data chain is **not** ready for the balanced
diagnostic-64 collection because S5, S6, S7, and S8 have zero persistable
episodes. The saved artifacts provide deterministic evidence for a separate
route-action synchronization round; no additional safety-boundary relaxation
or blind scenario tuning is justified by this run.
