# Round 13.861a: Process-local MetaDrive determinism

## Outcome

The in-process engine lifecycle contract passed.  A subprocess-per-episode
fallback was therefore not implemented.

The real S5 sandwich test runs the following sequence in one Python process:

```text
S5 seed 23 -> S5 seed 17 -> S5 seed 23
```

The two seed-23 signatures include every step's RuleMaker action, trajectory
source, failure reason, and the three vehicles' position, heading, speed, and
lane.  Three consecutive clean-process executions passed (108.82 s, 106.89 s,
and 106.10 s).

After every episode, the test also requires:

```text
BaseEngine.singleton is None
EngineCore.global_config is None
BaseEngine.COLORS_OCCUPIED is empty
BaseEngine.COLORS_FREE contains the complete color space
builtins.base is absent
```

This confirms that the existing `env.close()` path clears the relevant
MetaDrive/Panda process-level state.  No additional engine mutation or random
seed workaround was needed.

## Restored Round 13.86 audit

Command:

```bash
python -m evaluation.audit_bev_expert_chain \
  --config configs/dataset/data_collect_diagnostic64.yaml \
  --seeds 17 23 --max-steps 50 \
  --output /tmp/bev-stage-census/round13_86_chain_audit_after_13_861a.json
```

Result:

| Scenario | Seed 17 | Seed 23 |
|---|---|---|
| S5 | pass | pass |
| S6 | pass | pass |
| S7 | `all_rule_proposals_infeasible` | `all_rule_proposals_infeasible` |
| S9 | pass | pass |

The previous committed-executor out-of-road failures did not recur.  S7 is now
rejected before execution by the strict candidate contract.  Its failure is a
separate RuleMaker/planner feasibility issue and must not be hidden by changing
the controller or relaxing the road footprint boundary.

## Regression

```text
77 passed, 15 warnings
git diff --check: pass
```

