# Round 13.87 — S5–S9 200-step expert-chain audit

## Outcome

Round 13.87 is blocked.  The complete S5–S9, two-seed matrix produced four
clean 200-step episodes and six typed failures:

| Scenario | Seed | Steps | Result |
|---|---:|---:|---|
| S5 | 17 | 200 | pass |
| S5 | 23 | 51 | `committed_trajectory_tracking_deviation` |
| S6 | 17 | 159 | `gt_action_group_has_no_valid_mode` |
| S6 | 23 | 200 | pass |
| S7 | 17 | 71 | `committed_trajectory_background_unsafe` |
| S7 | 23 | 67 | `committed_trajectory_pairwise_unsafe` |
| S8 | 17 | 2 | `committed_trajectory_kinematic_infeasible` |
| S8 | 23 | 2 | `committed_trajectory_kinematic_infeasible` |
| S9 | 17 | 200 | pass |
| S9 | 23 | 200 | pass |

All ten scenario events were reported triggered and realized.  Evidence:

```text
/tmp/bev-stage-census/round13_87_s5_s9_long_audit.json
```

## Failure evidence

- S5 seed 23 fails at step 50, committed elapsed time 0.7 s.  Agent0 has only
  0.015 m longitudinal and 0.084 m lateral error, but 0.117 rad heading error
  exceeds the committed tracking envelope.
- S6 seed 17 fails at step 158.  RuleMaker chooses LEFT for all three roles,
  while every LEFT dynamic anchor is hard-invalid.  The recorded violations
  include `first_waypoint_at_time_zero` and `outside_reachable_distance` (plus
  `acceleration_below_min` for followers).  This is an anchor/GT contract
  problem, not a Normal-planner local-candidate failure.
- S7 seed 17 fails on the first committed roll: the agent0 trajectory has a
  3.880 m predicted background gap, below the unchanged 5 m boundary.
- S7 seed 23 fails on the first committed roll: the agent1–agent2 gap is
  6.170 m, below the unchanged 7 m platoon boundary.
- Both S8 seeds fail on the first committed roll because agent0's retained
  spatial-path arc buffer is exhausted.  The accepted execution plan cannot
  supply the required rolling four-second longitudinal reference.

These are four distinct upstream contracts.  They must not be hidden by
loosening the road/gap boundaries or by retuning the longitudinal controller.

## Round 13.88 gate

Round 13.88 diagnostic-64 collection was not started.  Starting it with a
4/10 complete-episode rate would violate the frozen S5–S9 quotas and the rule
that only complete, native, contract-valid episodes may be persisted.

The next work should separately address:

1. S8 execution-path horizon construction;
2. S6 dynamic-anchor time semantics and GT action-group validity;
3. S7 accepted-plan versus first-roll 5 m/7 m safety consistency;
4. S5 heading tracking-envelope failure.

After those fixes, rerun this exact 10-episode matrix before enabling Round
13.88.

