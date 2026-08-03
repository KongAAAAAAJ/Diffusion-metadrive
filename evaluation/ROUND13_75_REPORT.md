# Round 13.75 — RuleMaker/Normal Planner feasibility contract

## Outcome

The transactional feasibility contract is implemented and the related regression suite passes. The S5 simulation acceptance does **not** pass, so Round 13.75 is not marked complete.

No safety gap, road-footprint, kinematic limit, scenario parameter, controller, reward, diffusion, or dataset-schema boundary was relaxed.

## Implemented contract

- `RuleMaker.propose_joint_actions()` returns deterministically ranked joint proposals without creating lane-change commitments.
- `PlatoonNormalPlanner.plan_ranked()` tries proposals in RuleMaker rank order and accepts the first proposal with a native safe joint trajectory.
- `RuleMaker.accept_joint_action()` is the only point that registers a pending lane-change commitment.
- Active commitments fix the action and target lane family; infeasibility is reported as `committed_action_infeasible`.
- Exhausted proposals are reported as `all_rule_proposals_infeasible`; the strict expert chain never returns a synthetic fallback trajectory.
- RuleMaker coarse trajectories now contain the eight future timestamps `0.5 ... 4.0 s`; dense collision checking explicitly prepends the current pose.
- RuleMaker source/target lane metadata is passed unchanged to the planner.
- World-to-local conversion is shared and performed in float64 before the persisted float32 representation is audited.
- Candidate pools are cached across proposals and diagnostics record proposal attempts, selected rank, pool requests, and cache hits.
- S5 event detection is tied to the current orchestrator episode, avoiding stale hard-brake markers when an environment instance is reused.

## Regression verification

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

Result: `159 passed`.

`git diff --check` also passes.

## Simulation evidence

### Initial 10-seed run

This run exposed stale per-episode S5 trigger state and repeated full-horizon replanning of committed lane changes:

- episodes: 10
- collision rate: 0
- out-of-road rate: 0
- native planning step success: `495/505 = 98.02%`
- collection-ready success: `395/395 = 100%`
- persistable episodes: `0/10`
- terminal reason: `committed_action_infeasible` for all ten episodes
- mean/P95 wall time: `17.94/23.96 s`

Evidence: `outputs/bev-round13-75/full/S5_hard_brake_lead/` contains top-down videos, semantic-BEV videos, exact trajectory archives, episode diagnostics, and the aggregate report.

### One targeted correction and mandatory re-smoke

The permitted evidence-driven correction did two things:

1. bound S5 event detection to the current scenario episode;
2. make an accepted lane-change commitment use an absolute remaining completion deadline instead of restarting a fresh five-second maneuver at every 0.1-second planning tick.

The corrected `2 x 80` smoke produced:

- hard-brake event: `2/2` triggered at step 30;
- emergency-independent formation unlock: `2/2` started at step 30;
- collision rate: 0;
- out-of-road rate: 0;
- native planning step success: `122/124 = 98.39%`;
- collection-ready success: `102/102 = 100%`;
- persistable episodes: `0/2`;
- terminal reason: `committed_action_infeasible` for both episodes;
- mean/P95 wall time: `16.85/17.61 s`.

At failure, RuleMaker correctly kept the active lane-change action and every ranked proposal contained that same commitment. The planner exhausted all proposals because `agent0` had no native candidate:

- seed 17: the active LEFT commitment had elapsed 1.0 s; all candidates were rejected by the hard longitudinal corridor (`648` corridor rejections);
- seed 23: the newly accepted LEFT commitment failed on the next tick; candidates were rejected by the hard corridor, curvature audit, and six background-collision checks.

Pool caching was active (`6/12` and `2/6` cache hits at the terminal steps), so proposal iteration did not regenerate identical local pools.

Evidence: `outputs/bev-round13-75/smoke_fix/S5_hard_brake_lead/` contains both top-down/BEV videos, trajectory archives, episode-level proposal traces, formation-unlock summary, and aggregate metrics.

## Root cause and stop decision

The remaining blocker is not a RuleMaker-versus-Normal-Planner hard-feasibility disagreement. The proposal is feasible when accepted, but repeated receding-horizon regeneration does not guarantee that the same committed maneuver remains feasible from the next observed state. In other words, the expert chain lacks **recursive feasibility / accepted-trajectory continuity** for lane-change commitments.

The agreed stop condition therefore applies. The ten-seed acceptance was not rerun after the corrected smoke failed, and no further action scoring, scenario, safety, or controller tuning was attempted.

## Recommended next round

Handle commitment execution as a separate contract, for example by retaining the accepted native trajectory and tracking/re-anchoring its unexecuted suffix, with a formally audited contingency tube. This must preserve the existing 5 m/7 m gaps, OBB/road checks, and kinematic limits. It should not silently switch action, restart a full lane change each tick, or use a fallback trajectory.
