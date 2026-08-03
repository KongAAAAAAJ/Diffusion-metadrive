# Round 13.74 — S5 emergency-independent expert-chain report

## Status

Implementation and regression tests are complete, but the S5 simulator acceptance
gate is **not passed**.  The corrected `2 x 80` smoke still ended with
`normal_planner_single_agent_no_safe_candidate` for both seeds.  In accordance
with the round's stop condition, no further behavioural change and no `10 x 200`
run were performed.

## Implemented contract

- Per-agent lane-change commitments retain the accepted LEFT/RIGHT action and
  target lane family until completion; reset clears all commitment state.
- S5's actual `hard_brake_lead` marker changes the coordinator from `LOCKED` to
  `EMERGENCY_INDEPENDENT` on the next planning decision.
- Emergency-independent decisions omit formation-action consistency terms.
- RuleMaker uses complete `leader -> middle -> rear` prefix search with pairwise
  OBB pruning; it retains all safe prefixes and can backtrack.
- Normal planner uses the same complete conditional expansion.  Formation penalty
  is omitted only in emergency-independent mode; kinematic, road, background and
  all three platoon-pair safety checks remain active.
- The expert collector uses per-agent `PIDTrajectoryController` while independent
  and `LQRFollowerController` while locked, resetting controller state on a
  transition.
- Preview artifacts record coordination mode, formation-constraint state,
  controller backend and commitment actions without changing the dataset schema.

## Tests

Command:

```bash
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
PYTHONPATH="$PWD:$PYTHONPATH" \
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest \
  tests/test_preview_and_evaluation.py \
  tests/test_rule_maker.py \
  tests/test_bev_trajectory_kinematic_contract.py \
  tests/test_joint_bev_collection.py \
  tests/test_platoon_normal_planner.py \
  tests/test_bev_expert_chain_audit.py -q
```

Result: `153 passed`.

`git diff --check` passed.

## Corrected S5 smoke

Configuration:

```text
scenario    S5_hard_brake_lead
route       R1_entry_straight
seeds       17, 23
horizon     80
policy      rule_maker + lattice + adaptive controller
```

Aggregate result:

| Metric | Result |
|---|---:|
| hard-brake event triggered | 2/2 |
| collision rate | 0% |
| out-of-road rate | 0% |
| native joint step success | 110/112 (98.21%) |
| collection-ready step success | 88/90 (97.78%) |
| persistable episodes | 0/2 |
| final failure | 2 x single-agent no-safe-candidate |

Both episodes remained locked and selected KEEP before the physical brake marker.
At decision index 30 both entered emergency-independent mode, and every later
recorded control used `PIDTrajectoryController`.  No committed action changed
directly from LEFT to RIGHT or RIGHT to LEFT.

## Failure attribution

The final failure is local to `agent0`; it is not a three-agent combination
failure and is not caused by formation scoring.

| Seed | Failure frame | Raw candidates | Main hard rejections | Formation penalty |
|---:|---:|---:|---|---|
| 17 | 60 | 252 | corridor 864, road 126, background collision 42, curvature 84 | disabled |
| 23 | 52 | 252 | corridor 648, road 84, background collision 162, lateral acceleration 6 | disabled |

At failure, the safe longitudinal corridors for the attempted lane change have
negative upper progress or are empty.  Thus RuleMaker finds a coarse pairwise-safe
LEFT action, but Normal planner cannot instantiate a trajectory that simultaneously
satisfies the 5 m background corridor, road footprint and kinematic limits.  This
is a remaining RuleMaker-to-Normal-planner feasibility mismatch.

Each episode also recorded one earlier rejected expert step for `agent2` with
`acceleration_below_min`.  The collector correctly marks the episode unusable;
this is a second blocker to the required complete persistence even if the final
agent0 failure were removed.

## Artifacts

- First smoke: `outputs/bev-round13-74/smoke/S5_hard_brake_lead/`
- Corrected smoke: `outputs/bev-round13-74/smoke_fix/S5_hard_brake_lead/`
- Aggregate JSON:
  `outputs/bev-round13-74/smoke_fix/S5_hard_brake_lead/metrices/expert_collection_summary.json`
- Per-episode JSON, top-down videos, semantic-BEV videos and exact trajectory NPZ
  files are stored below the corrected smoke directory.

## Required next decision

Round 13.74 must remain unaccepted.  A subsequent round should choose explicitly
between aligning RuleMaker's action-feasibility prediction with the Normal planner
or altering S5's physical gap timing.  This round did neither because the approved
boundary forbids S5 traffic changes and permits only one evidence-driven
correction after smoke failure.
