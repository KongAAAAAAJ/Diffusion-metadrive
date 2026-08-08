# Round 13.93: S5--S9 four-model diagnostic evaluation

## Outcome

The diagnostic four-model evaluation completed for the frozen S5--S9
scenario contract, holdout seeds `[31, 47]`, and `100` simulator steps per
episode (`10 episodes/model`, `40 episodes` total).

This round validates the evaluation infrastructure and latency contract.  It
does **not** validate formal policy quality: all four source checkpoints are
diagnostic-only, completion is impossible on several routes within the
100-step diagnostic horizon, and the observed safety/executability rates are
not acceptable as paper results.

Result artifact:

```text
outputs/bev-round13-93/four_model_diagnostic.json
SHA256 9a96cd00e240101d8dfad1d92b90beb0e06a9c554f1e005460cb3244b4058c3a
```

The report records and verifies the exact Stage 1/GRPO checkpoint hashes from
`evaluation/ROUND13_93_MANIFEST.json`.  It also records
`diagnostic_only=true`, `eligible_for_formal_conclusions=false`, the complete
S5--S9 scenario contract hash, fixed diffusion-noise seeds, deterministic CUDA
settings, and `PYTHONHASHSEED=0`.

## Results

| Model | Collision | Out of road | 5 m violation | 7 m violation | Execution rejection | Formation P95 | Inference P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 10% | 0% | 70% | 40% | 10% | 17.51 m | 9.64 ms |
| B | 0% | 0% | 60% | 50% | 20% | 18.01 m | 20.15 ms |
| A + GRPO | 0% | 0% | 80% | 40% | 20% | 16.31 m | 9.48 ms |
| B + GRPO | 10% | 0% | 70% | 30% | 20% | 18.08 m | 20.20 ms |

All models pass the frozen three-role model-inference P95 limit of `100 ms`.
The full planning tick is slower (`128.8--135.6 ms` P95), dominated by online
semantic-BEV construction (`~100 ms`) rather than model inference.  The
execution-time trajectory optimizer costs about `10 ms` P95.

Every execution rejection occurred in S9 near the end of the diagnostic
horizon.  In each case the leader sampled `RIGHT_HIGH`; even the selected
hard-valid coarse anchor could not be converted to the stricter calibrated
actuator profile, predominantly because of `acceleration_above_max`, with
occasional curvature, heading-alignment, or reachable-distance violations.
The evaluator terminates those episodes and records the exact reason; it never
executes a fallback trajectory.

## Infrastructure fixes made during the gate

- Added an immutable four-checkpoint manifest with strict file SHA256 checks.
- Required deterministic Torch algorithms, disabled TF32, fixed the cuBLAS
  workspace, and required `PYTHONHASHSEED=0` at process startup.
- Verified that every model receives the same reset pose, heading, and speed
  for each scenario/seed within the report (`atol=1e-8`).
- Counted an unprojectable selected action as an explicit execution rejection
  instead of aborting the whole evaluator or applying a fallback.
- Made terminal statistics robust to MetaDrive removing an agent object.
- Corrected the longitudinal STOP contract so an initial zero-speed sample may
  be followed by a valid accelerate--decelerate profile; once motion stops,
  the remaining profile must still be exactly stationary.

## Reproducibility boundary

Two independent 12-step runs with fixed Python hash seed and deterministic
CUDA still produced different behavior hashes after timing fields were
removed:

```text
5df064092ac9ed699f6d3c7220f959ae81f5b7d0a00309b8e3a9f4d906addb2d
21a194851e3e6ad24b358fec596ab5bd2fd476aa4de058b06df7f6044c1c21a4
```

The differences include small vehicle-state changes and occasional mode/gap
threshold changes.  Therefore exact cross-process bitwise reproducibility is
not claimed.  The final 40-episode run does enforce identical initial state
across the four models inside that run, so its comparative setup is fair, but
future formal evaluation should either establish a deterministic simulator
process contract or report confidence intervals over more seeds.

## Verification

```text
100 passed, 17 warnings
git diff --check: pass
```

The regression set covers the Stage 1 A/B integration, A/B GRPO core and
online training, joint reward, simulator branches, trajectory optimizer,
longitudinal reference/control calibration, tracking diagnostics, and the
four-model evaluator.

## Remaining blockers

1. Formal eligible Stage 1 and GRPO checkpoints do not exist yet.
2. S9 selected-mode executability must be improved through training reward,
   guidance, or mode feasibility—not by relaxing the execution optimizer.
3. The diagnostic policies have high hard-gap violation rates and nonzero
   collisions; no performance superiority can be concluded.
4. Exact cross-process simulator reproducibility remains unresolved.
5. End-to-end planning tick P95 exceeds 100 ms because BEV construction is
   slow, although the frozen model-inference requirement itself passes.

