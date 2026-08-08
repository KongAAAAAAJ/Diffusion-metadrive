# Round 13.92 — S5–S9 Online Joint GRPO Diagnostic Acceptance

## Result

Round 13.92 passes as a **diagnostic training-chain acceptance** for both
variants.  It does not establish a safety or performance improvement and none
of the generated checkpoints is eligible for formal training or paper results.

Both runs use the passed Round 13.91 holdout calibration reports, the frozen
S5–S9 scenario contract, development seeds 17/23, and holdout validation seeds
31/47.  Online sampling round-robins all ten `(scenario, seed)` buckets.  Each
bucket supplied exactly three model states; constant-reward groups were logged
and executed but correctly skipped as optimizer updates because their signed
GRPO advantages are identically zero.

| Metric | Variant A | Variant B |
|---|---:|---:|
| Optimizer updates | 20 | 20 |
| Sampled joint rollouts | 30 | 30 |
| Uninformative rollouts skipped | 10 | 10 |
| Environment steps | 462 | 462 |
| Samples per S5–S9/seed bucket | 3 | 3 |
| Optimized trajectory valid rate | 100% | 100% |
| Trajectory optimizer P50 / P95 | 31.32 / 34.31 ms | 31.49 / 36.01 ms |
| Validation unsafe count | 8 | 9 |
| Validation collision count | 2 | 2 |
| Validation out-of-road count | 0 | 0 |
| Validation simulator reward mean | -16.963 | -19.077 |

The uneven **update** count across buckets is expected and is not hidden: S5
frequently produces a constant unsafe reward for all four samples.  The exact
per-bucket sample/update counts are recorded in each `report.json`.  Sampling,
not synthetic gradient production, is balanced.

## Artifacts

- A run: `/tmp/bev-round13-92-final/A/run_2`
- B run: `/tmp/bev-round13-92-final/B/run_1`
- A `last.pt` SHA256:
  `b7dd7e0a77ea130983bf2ab387de890eb17e171555eec7705b5f552ea4b24fa4`
- A `best.pt` SHA256:
  `29ee20d2beb547ea1af247bfeeabbabb4e15a85e641b499eefc2f0973c30a0af`
- B `last.pt` SHA256:
  `cde31dfd47d3cd4e532d3771fa6475a5646510d2e30e4482b566c78183f0aad4`
- B `best.pt` SHA256:
  `f1faa131e16228299a1f00d76a72638bb981a864c1b35ff71371e250866e5173`

All four checkpoints round-trip through their strict A/B loader.  They retain
`diagnostic_only=true` and `eligible_for_formal_training=false`.

## Gradient and freeze audit

- A: all 53 decoder tensors and both mode-head tensors changed.
- B: all 60 decoder tensors, both mode-head tensors, all six predecessor action
  encoder tensors, and the residual gate changed.
- A/B: all 166 non-decoder/non-mode planner tensors are bitwise identical to
  their Stage 1 source checkpoint.
- A/B: frozen reference tensors are bitwise identical to the Stage 1 source.
- Every accepted update has finite non-zero decoder and mode-head gradients.
- B additionally has finite non-zero action-encoder and gate gradients on all
  recorded accepted updates.

## Contract fixes made during acceptance

1. Calibration JSON is reconstructed into the strict trajectory-optimizer
   dataclass before comparing both normalized content and SHA256.  JSON
   list/tuple representation differences no longer reject an identical config.
2. Constant group rewards are classified as uninformative and do not call an
   optimizer step; strict non-zero-gradient checks remain unchanged.
3. The execution optimizer repairs heading-only coarse-anchor mismatch by
   deriving smooth centered tangents from the unchanged XY path.  The selected
   mode and XY geometry are preserved, and yaw-rate, curvature, lateral
   acceleration, and heading-alignment audits remain hard gates.
4. Training samples round-robin all S5–S9 × seeds 17/23 buckets, rather than
   letting a long early episode dominate the 20 updates.
5. Per-module gradient norms and per-bucket sample/update counts are recorded.

## Remaining risks

- Raw selected diffusion trajectories were kinematically invalid in every
  accepted update batch; the execution optimizer made all executed actions
  valid, but its retained-raw fraction remained low.  This is evidence that
  model-level trajectory compliance is not solved by the diagnostic GRPO run.
- Fixed validation remains unsafe in most branches and did not improve over the
  Stage 1 policy in a way that can support an effectiveness claim.
- S5 has very weak within-group reward diversity.  Its states are represented
  in sampling, but most do not contribute gradient updates.
- Formal GRPO remains blocked until eligible Stage 1 checkpoints trained on the
  formal verified dataset exist.

Round 13.93 may now run the four-model diagnostic evaluator, but it must retain
these caveats and must not present the diagnostic checkpoints as formal results.
