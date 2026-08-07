# Round 13.882: S5--S9 expert-chain gate

## Outcome

The round is **blocked**.  The required `10 x 200` gate was not started and
the diagnostic-64 collector was not started.

The development gate uses one MetaDrive episode per process.  At 80 steps,
seed 17 produced:

| Scenario | Result |
|---|---|
| S5 | pass, 80/80 |
| S6 | pass, 80/80 |
| S7 | pass, 80/80 |
| S8 | `all_rule_proposals_infeasible`, step 39 |
| S9 | pass, 80/80 |

S8 reproduced the same step-39 failure for seeds 17 and 23.  After extending
the chosen exit connector through its unambiguous downstream ramp successors,
the committed-horizon road rejections disappeared.  Agent2 then had 17
dynamically and geometrically valid candidates, but none satisfied the frozen
5 m background-vehicle gap.  No safety boundary or scenario parameter was
changed to force acceptance.

## Implemented contract fixes

- RuleMaker proposals are filtered against dynamic-anchor hard-valid action
  groups before Normal planner search.
- S7/S8 route geometry includes the merge/exit connector seams and continuous
  downstream lane chain where the successor is unambiguous.
- Candidate and committed execution use the same expanded drivable surfaces
  and dense XL footprint audit.
- The longitudinal governor keeps generated acceleration 0.001 m/s2 inside
  the hard boundary to survive float32 re-audit without changing the
  `[-8, 5] m/s2` validator.
- Curved-path travel uses arc distance and keeps the final moving segment long
  enough to satisfy the unchanged `0.25 1/m` curvature contract.
- Temporal longitudinal error and spatial cross-track error are separated in
  Frenet coordinates.
- PID preview receives bounded cross-track feedback.  A new execution id
  resets controller derivative/integral state, while committed rolls retain
  state continuously.
- Audit JSON records preview, cross-track, heading, longitudinal and actuator
  components per role.

## Determinism blocker

The in-process S5 sandwich test no longer passes.  Strong process isolation
also failed: two independent S5 seed-23 runs first diverged at step 31.  Agent0
speed was 26.85092 km/h in one process and 25.97634 km/h in the other, while
agent1 and agent2 still matched.  This points to the S5 lead-brake trigger or
lead longitudinal response, not engine teardown.

Evidence:

```text
/tmp/bev-stage-census/round13_882_isolated_s5_seed23_a.json
/tmp/bev-stage-census/round13_882_isolated_s5_seed23_b.json
```

## Saved smoke evidence

```text
/tmp/bev-stage-census/round13_882d_s5_seed17_preview80.json
/tmp/bev-stage-census/round13_882_gate_s6_seed17_80.json
/tmp/bev-stage-census/round13_882_gate_s7_seed17_80.json
/tmp/bev-stage-census/round13_882_gate_s9_seed17_80.json
/tmp/bev-stage-census/round13_882ac_s8_seed17_poolchain80.json
/tmp/bev-stage-census/round13_882_gate_s8_seed23_80.json
```

## Regression

Focused controller/planner tests passed (`82 passed`), and the broader run
reported `214 passed, 3 failed`:

- `test_bev_episode_determinism.py`: real determinism blocker described above.
- two legacy semantic-anchor assertions compare `[8,3]` source trajectories
  with the generator's current `[8,2]` anchors; unrelated to Round 13.882.

`git diff --check` passes.  The worktree remains intentionally uncommitted.

## Required next decision

Before resuming `10 x 200`, independently fix and verify:

1. deterministic S5 lead-brake trigger/actuation at step 31;
2. whether S8 scenario traffic timing should be changed, or whether its
   5 m-safe infeasibility is the intended negative case.

Do not start diagnostic-64 until both issues are resolved and the full gate is
10/10.
