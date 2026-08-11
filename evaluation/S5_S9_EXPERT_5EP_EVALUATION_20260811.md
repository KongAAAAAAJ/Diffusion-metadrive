# S5--S9 Expert Five-Episode Evaluation (2026-08-11)

## Outcome

The expert execution gate passes, but the scenario-realization review is only a
partial pass.

- All 25 episodes completed the frozen 200-step horizon.
- All 5,000 planning attempts returned native joint trajectories.
- Collision rate, out-of-road rate, and expert failure rate were all zero.
- All 25 episodes were persistable and all 4,750 collection-ready steps passed.
- S5, S6, and S8 completed their declared traffic recipes in every episode.
- S7 completed only 6 of 10 background-vehicle recipes in every episode.
- S9 completed only 1 of 2 background-vehicle recipes in every episode.
- S5 consistently reported `adjacent_lane_missing:left_side`; the right-side
  vehicle and hard-braking lead vehicle were realized.

The run therefore proves that the current RuleMaker/Normal-planner/adaptive
controller chain is stable on these five seeds. It does not prove that S7 and
S9 exercise the intended full traffic challenge, and S5's adjacent-lane
blocking is weaker than its two-sided recipe suggests.

## Frozen protocol

```text
expert chain       RuleMaker -> PlatoonNormalPlanner -> adaptive LQR/PID control
scenarios          frozen S5--S9 routes
seeds              17, 23, 31, 47, 59
episodes           5 per scenario, 25 total
horizon            200 simulator steps (20 s)
agents             3
traffic density    0 (only scenario-authored actors)
target speed       30 km/h
video fps          10
```

The execution used `evaluation.preview_and_evaluation` with `--evaluate`,
top-down video, semantic-BEV video, and exact trajectory output enabled. The
five scenarios were run in separate processes but with identical protocol and
seed lists.

## Quantitative results

| Scenario | Episodes | Native planning | Collision / road exit | Persistable | Recipes complete | Team PDMS | Min gap (m) | Max formation error (m) | Formation unlock | Mean expert planning (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S5 hard-brake lead | 5/5 | 1000/1000 | 0 / 0 | 5/5 | 2/2 in 5/5 | 0.498 | 14.76 | 14.02 | 5/5 | 432.65 |
| S6 background merge-in | 5/5 | 1000/1000 | 0 / 0 | 5/5 | 1/1 in 5/5 | 0.391 | 15.33 | 27.37 | 0/5 | 248.44 |
| S7 ego ramp merge | 5/5 | 1000/1000 | 0 / 0 | 5/5 | **6/10 in 0/5** | 0.390 | 13.26 | 25.87 | 0/5 | 272.84 |
| S8 ego ramp exit | 5/5 | 1000/1000 | 0 / 0 | 5/5 | 2/2 in 5/5 | 0.377 | 15.02 | 8.62 | 0/5 | 327.59 |
| S9 narrow-channel negotiation | 5/5 | 1000/1000 | 0 / 0 | 5/5 | **1/2 in 0/5** | 0.502 | 15.74 | 33.25 | 0/5 | 484.14 |

`Team PDMS` is the mean of the five episode-level `__team__.reward` values.
The minimum gap and maximum formation error are extrema across every saved
agent step. Expert planning latency is offline label-generation latency and is
not the learned model's sub-100-ms inference metric. The expert planner is not
real-time at this instrumentation level: scenario-average planning time ranges
from 248.44 ms (S6) to 484.14 ms (S9).

## Scenario-by-scenario assessment

### S5 -- hard-braking lead

- The lead vehicle and braking profile were realized in 5/5 episodes.
- Formation unlocked in all five episodes during the braking response, while
  the high-level rule action remained KEEP for all three vehicles.
- The leader reached essentially zero speed without collision or road exit.
- The left adjacent vehicle was missing in every episode; only the right
  adjacent actor spawned. This weakens the intended two-sided lane-blocking
  pressure and should be treated as a scenario-design defect even though the
  recipe currently marks itself complete.

Assessment: expert response passes; two-sided hazard realization needs repair
or an explicit one-sided contract.

### S6 -- background merge-in

- The controlled merge actor was injected and the recipe completed in 5/5.
- The expert used KEEP throughout and stayed formation-locked, resolving the
  interaction through longitudinal spacing/control rather than a lane change.
- All five episodes remained safe and collection-ready.

Assessment: passes the intended conservative-yield execution check. A future
evaluation should add an actor-relative merge-gap/TTC metric, since the current
minimum-gap summary primarily describes platoon spacing.

### S7 -- ego merges from ramp

- The three platoon vehicles performed the required LEFT merge; LEFT was
  selected for 653 of 3,000 agent decisions.
- Every episode was safe and planning-valid.
- Only 6 of the 10 authored mainline vehicles were successfully injected in
  every episode. The remaining attempts were blocked by agent-clearance and
  repeated `block_route_road` injection failures.

Assessment: expert merge execution passes, but the current evaluation is a
reduced-density version of the authored scenario. Scenario completeness fails.

### S8 -- ego exits to ramp

- All two background actors were eventually injected in every episode.
- The platoon performed the required RIGHT maneuver; RIGHT was selected for
  605 of 3,000 agent decisions.
- Transient spawn-blocked/injection-failed notes occurred while actors were
  retried, but the terminal recipe-complete flag was true for all episodes.

Assessment: passes both expert execution and final scenario realization.

### S9 -- narrow-channel negotiation

- All five episodes were safe, but all three agents selected KEEP throughout
  and formation never unlocked.
- Only one of two authored background actors was injected in every episode;
  the second repeatedly failed on `block_route_road`.
- The high team PDMS (0.502) should not be interpreted as evidence of strong
  negotiation because the missing actor and all-KEEP behavior indicate that
  the intended two-sided merge/split interaction was not fully exercised.

Assessment: expert stability passes; negotiation challenge and scenario
completeness fail.

## Artifacts

Root:

```text
outputs/s5_s9_expert_eval_5ep_parallel_20260811/
```

Each scenario contains:

```text
video/                  5 top-down trajectory-overlay MP4 files
semantic_bev_video/     5 three-role semantic-BEV mosaic MP4 files
trajectories/           5 exact trajectory NPZ files
metrices/episode_*/     per-episode JSON, plots, and diagnostics
metrices/episode_summary.csv
metrices/expert_collection_summary.json
metrices/formation_unlock_summary.json
```

Artifact audit:

```text
top-down videos         25
semantic-BEV videos     25
trajectory archives     25
expert episode JSON     25
PDMS episode JSON       25
total size              approximately 1.1 GiB
```

An earlier interrupted serial diagnostic remains under
`outputs/s5_s9_expert_eval_5ep_20260811/`; it is not part of the final 25-episode
evidence set.

## Recommended next action

Do not change safety or planner constraints. First repair or explicitly revise
the actor-realization contracts for S5-left, S7's four missing mainline actors,
and S9's missing merge/split actor. Then rerun the same five-seed protocol and
require `scenario_recipes_complete=true` in every episode before using this
visual audit as evidence that all five scenario designs are fully realized.
