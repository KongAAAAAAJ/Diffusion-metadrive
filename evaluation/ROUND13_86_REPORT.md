# Round 13.86 — S5/S6/S7/S9 50-step expert-chain acceptance

## Outcome

Round 13.86 passes its frozen eight-episode matrix:

```text
scenarios: S5, S6, S7, S9
seeds:    17, 23
horizon:  50 steps
result:   8/8 pass
```

All eight episodes triggered and realized their configured event, reached the
50-step horizon, and reported no typed expert-chain failure.

Evidence:

```text
/tmp/bev-stage-census/round13_86_acceptance.json
```

## S7 road-audit correction

The remaining S7 rejection was traced to an XL-vehicle footprint corner at a
real topology-connected lane seam.  The point was about 9 mm before the
successor lane's numerical start and about 0.11 m past the predecessor lane's
numerical end.  It was therefore outside both individual lane surfaces even
though the lanes are connected.

The shared dense-footprint audit now:

- resolves the connected predecessor lane for both native candidates and the
  committed executor;
- admits points inside the quadrilateral joining a predecessor end to its
  connected successor start;
- refuses geometrically identical gaps when the lane graph has no matching
  topology connection.

This is a connected-lane seam model, not a global road-boundary tolerance.  The
planner and committed executor still use the same dense XL footprint audit and
the existing road, 5 m/7 m gap, OBB and kinematic boundaries.

## Regression

```text
141 passed, 15 warnings
git diff --check: pass
```

The regression set covers the Normal planner, expert-chain auditor, RuleMaker,
joint collector and collection runner.

