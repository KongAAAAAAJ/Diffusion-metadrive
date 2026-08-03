# Round 13.76 — committed joint-trajectory continuity

## Outcome

The committed-trajectory execution contract is implemented and its regression tests pass. The mandatory S5 `2 x 80` smoke fails the strict recursive kinematic audit, so the ten-seed run was not started and Round 13.76 is not marked accepted.

No controller, scenario parameter, safety gap, road boundary, kinematic limit, reward, diffusion behavior, or dataset schema was changed.

## Implemented behavior

- A selected native lane-change proposal now carries an immutable three-agent `JointTrajectoryExecutionPlan`.
- Each selected candidate stores its lane/path and longitudinal profile parameters and a fixed-time dense trajectory buffer extending beyond the original four-second label horizon.
- `JointTrajectoryExecutor` advances the original absolute-time path and returns a fresh future `0.5 ... 4.0 s` trajectory without restarting lateral progress.
- The whole three-agent plan is retained atomically; non-changing agents are not replanned independently during the commitment.
- RuleMaker advances risk and commitment state through `advance_committed_execution()` without generating proposals.
- Every roll rechecks world/local kinematics, tracking envelope, lane-chain validity, road footprint, background safety and pairwise joint safety.
- Any failure is typed and rejects the episode; no action switch, projection, braking fallback, or safety relaxation is available.
- Evaluation trajectory archives now include execution id/source/elapsed time, hard-audit status, tracking errors and minimum platoon gap. Native-planning and committed-roll latency are reported separately.

## Regression verification

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

Result: `163 passed`.

The new tests verify absolute-time rolling, three-agent atomic execution, tracking-deviation rejection and proposal-free RuleMaker state advancement.

## S5 smoke

Configuration:

```text
scenario: S5_hard_brake_lead / R1_entry_straight
seeds: [17, 23]
horizon: 80
```

Aggregate result:

- hard-brake trigger and immediate unlock: `2/2`, step 30;
- collision rate: `0`;
- out-of-road rate: `0`;
- collection-ready labels before failure: `42/42`;
- persistable episodes: `0/2`;
- failure reason: `committed_trajectory_kinematic_infeasible` for both episodes;
- execution id: `1` in both episodes;
- active execution generated zero new RuleMaker proposals;
- failed roll latency: approximately `3.82–3.84 ms`.

Both episodes accept a native joint plan at step 30 and enter `committed_roll` at elapsed `0.1 s`. The retained path then fails for `agent2`:

```text
world violation: outside_reachable_distance
local violation: outside_reachable_distance
tracking longitudinal error: about 0.0323 m
tracking lateral error: about 0.00041 m
tracking heading error: about 0.00048 rad
```

Thus path identity and proposal suppression work, but the original open-loop longitudinal profile is not recursively executable from the actual state even after only one control tick. The small pose error changes current speed/reachable-distance consistency enough for the unchanged `[-8, 5] m/s²` contract to reject the next four-second label.

Evidence is stored under `outputs/bev-round13-76/smoke/S5_hard_brake_lead/`, including top-down videos, semantic-BEV videos, trajectory NPZ files and per-episode execution diagnostics.

## Stop decision

The plan explicitly requires stopping when the retained trajectory fails because of tracking or recursive feasibility. Therefore:

- no second behavior correction was made;
- no kinematic tolerance was enlarged;
- no emergency trajectory or fallback was introduced;
- the `10 x 200` run was not launched.

The next technical decision must be handled separately: either make the expert trajectory a feedback-executable reference whose longitudinal state is part of the execution contract, or redesign the controller/reference-speed interface. That work belongs to the previously deferred longitudinal reference separation round, not Round 13.76.
