# Rounds 13.881–13.884 — per-scenario contract audit

## Outcome

The four failures from Round 13.87 were reviewed independently with three
read-only worker investigations and one primary-agent integration review.
Round 13.882 has a verified root-cause fix.  Rounds 13.881, 13.883 and 13.884
remain blocked by the strict stop condition; no safety boundary was relaxed.

## 13.881 — S8 spatial-path horizon: blocked

The original `spatial path arc buffer is exhausted` is caused by temporal and
spatial semantics being conflated.  The selected `-8 m/s2` stop-and-hold plan
contains about 8.6 s of timestamps but only 2.778 m of unique geometry.  The
first feedback roll from 6.67 m/s requires about 9.08 m.

A temporal/spatial separation prototype removed the buffer exception, but then
exposed the upstream geometry error: S8's downstream exit lane is treated as an
immediate adjacent lane about 11 m laterally away.  Non-stopping candidates all
fail the dense road-footprint audit; the stopping candidate only hid the error
by barely moving.  The prototype was removed rather than persisting a known
invalid exit path.

Required next change: build S8 geometry along the source-to-exit route lane
chain and perform execution preflight before accepting the proposal.

## 13.882 — S6 dynamic-anchor action semantics: fixed

The dynamic anchors already use the correct future times `0.5...4.0s`.  The
actual defect was a negative lane id:

```text
source lane id 0 + LEFT(-1) -> lane id -1
MetaDrive list[-1]          -> lane id 2
```

RuleMaker therefore labelled a physically rightward target as LEFT, while the
anchor topology correctly disabled LEFT.  RuleMaker and Normal planner now
reject negative target ids before `get_lane()` and strictly verify that the
returned lane index equals the requested index.  No invalid anchor is filled
or relabelled.

The original GT/mask conflict is gone.  Under the temporary spatial-path
prototype both seeds reached 170/170 steps, but after removing that unaccepted
prototype the strict 200-step run produced: seed 23 `200/200`; seed 17 failed
at step 155 with `spatial path arc buffer is exhausted`.  Thus the S6 semantic
bug is fixed, but Round 13.882 remains coupled to the unresolved 13.881
execution-path contract and is not marked complete.

## 13.883 — S7 first-roll safety consistency: blocked

The planner previously hard-rejected only OBB overlap.  Background 5 m
clearance was a soft score and the platoon 7 m clearance was absent from joint
selection, while the executor enforced both as hard constraints.

The planner now uses the same dense bumper-gap geometry before candidates enter
the pool and during joint prefix search.  Unit tests and planner regressions
pass.  Real S7 still fails:

- seed 17: background gap enters the shared corridor below 5 m on the first
  committed roll;
- seed 23: the earlier pair-gap failure is removed, but a later heading
  tracking deviation appears.

This points to a remaining cross-lane prediction/corridor timing discrepancy,
not a reason to reduce 5 m/7 m.

## 13.884 — S5 heading tracking: partially fixed, not accepted

The expert preview PID used a default `pid_dt=0.5s` although the simulator
decision period is 0.1 s.  Its default lateral gains also differed from the
online preview controller.  The controller now derives its default PID period
from `physics_world_step_size * decision_repeat` and uses the already validated
online defaults `Kp=1.6`, `Kd=0.12`.

The original seed-23 heading failure is removed in the 80-step smoke.  However,
seed 17 now exposes `all_rule_proposals_infeasible` after the newly aligned hard
gap checks.  Therefore the controller correction is retained, but Round 13.884
is not marked complete.

## Regression

```text
148 passed, 15 warnings
```

The checked set covers RuleMaker, Normal planner, committed execution,
longitudinal references, expert-chain audit and controller contracts.
