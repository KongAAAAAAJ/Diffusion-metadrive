# Round 13.94 Acceptance Report

## Scope

Round 13.94 addresses three independent gates:

1. reject S9 `RIGHT_HIGH` modes that cannot satisfy the calibrated execution
   contract before categorical sampling;
2. reduce semantic-BEV construction latency without changing raster pixels;
3. measure repeatability by running complete evaluator calls sequentially in one
   fixed Python process.

The source checkpoints in `evaluation/ROUND13_93_MANIFEST.json` remain
diagnostic-only.  Consequently all results below are diagnostic evidence and are
not eligible for formal paper conclusions.

## Implementation

- Added an execution-time mode mask which is a strict subset of the persisted
  physical hard mask.  Every retained coarse anchor must pass the same
  actuator-aware projection used by the final trajectory optimizer.  The mask is
  installed before Stage-1 or GRPO categorical sampling and log-probability
  calculation; it is not a post-sampling fallback and does not modify the dataset.
- Conservatively culled map geometry whose world AABB cannot intersect the
  rotated BEV view disk, and reused the scene adapter's rasterizer instance.
- Added fixed-process repeated evaluation and a behavior SHA256 which excludes
  timing but includes policy, safety, formation, efficiency and execution
  metrics.

## Full diagnostic gate

Command:

```bash
PYTHONHASHSEED=0 python -m evaluation.bev_four_model_evaluator \
  --manifest evaluation/ROUND13_93_MANIFEST.json \
  --output outputs/bev-round13-94/four_model_execution_mask.json \
  --run-mode diagnostic --device cuda --max-steps 100
```

This evaluates 10 S5--S9 episodes per model (40 episodes total).

| Model | Execution rejects (13.93 -> 13.94) | BEV P50 ms | BEV P95 ms | Tick P95 ms |
|---|---:|---:|---:|---:|
| A | 1 -> 0 | 73.63 | 89.33 | 117.63 |
| A+GRPO | 2 -> 0 | 74.37 | 89.97 | 118.41 |
| B | 2 -> 0 | 71.95 | 85.91 | 125.14 |
| B+GRPO | 2 -> 0 | 71.13 | 83.19 | 123.68 |

The S9 `RIGHT_HIGH` blocker is closed for this gate: all four policies completed
with zero final optimizer rejection.  The new mask removed a non-zero number of
modes online (mean 0.026--0.033 modes per joint state), proving that the
actuator-aware feasibility boundary was exercised.

BEV P95 decreased from 103.75--111.20 ms to 83.19--89.97 ms (approximately
18--20%).  Three-agent model inference remains below 21 ms P95.  The complete
planning tick is still above 100 ms P95 because it also contains anchors, hard
masking, execution-mask construction and trajectory optimization; Round 13.94
does not claim that the end-to-end latency gate is closed.

Artifact SHA256:

```text
4f31c0aaedcde7cac99f3aec4c52436c959993ac855a98e2f877335d969fbb0c
```

## Fixed-process repeatability

A two-repeat, fixed-process short gate was run with identical checkpoints,
scenario seeds, explicit model noise, `PYTHONHASHSEED=0`, deterministic Torch and
deterministic CUDA settings.  Timing was excluded from the behavior hash.

```text
repeat 1: 4a563046ca4af92edbda4e28870dc10c54059e4ca1094d66c10b0eba7dfc7a51
repeat 2: 9f4d4aea9d0c1055128a30ab8dcb26e1acceae6421444f60e459310792c9f66d
exact_behavior_match: false
```

Initial states and policy sampling matched, but MetaDrive/Bullet accumulated
centimetre-scale physics differences.  One example changed A's minimum
background clearance from approximately 5.009 m to 4.983 m, crossing the exact
5 m metric threshold.  Because the short deterministic gate already failed, a
second expensive full 40-episode repeat was not launched.  Re-running until a
matching hash appears would not satisfy the contract.

A formal evaluator invocation was also verified to reject the current source:

```text
formal GRPO requires an eligible Stage 1 checkpoint; diagnostic sources require explicit opt-in
```

This is intentional.  Metadata was not altered to bypass checkpoint eligibility.

## Tests

Focused and regression command:

```text
91 passed
```

Coverage includes execution-mask semantics, STOP retention, physical-mask
immutability, GRPO batch integration, raster pixel equivalence after conservative
culling, repeat hashing, Stage-1/GRPO policy regressions and simulator branch
regressions.  `git diff --check` passes.  The new chain contains no
camera/LiDAR/selector/legacy-GRPO dependency.

## Verdict

- S9 final-mode executability: **PASS** for the full diagnostic gate.
- BEV rasterization/build latency: **PASS** for the BEV P95 `<100 ms` target and
  materially improved, while full planning-tick P95 remains an open optimization
  item.
- Fixed-process exact repeatability: **FAIL** due to simulator physics drift.
- Formal repeated evaluation: **BLOCKED BY DESIGN** until eligible formal Stage-1
  and GRPO checkpoints exist; the strict refusal is verified.

Round 13.94 is therefore partially accepted, but the complete Round-13 formal
evidence gate is not yet closed.
