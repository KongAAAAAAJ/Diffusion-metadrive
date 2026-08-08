# Round 13.95: Tolerance-Based Reproducibility Assessment

## Question

Round 13.95 tests whether fixed-seed, fixed-process closed-loop physics drift is
small enough that requiring bit-exact simulator trajectories can be replaced by
a layered reproducibility contract.

The evaluated checkpoints are still diagnostic-only.  This report assesses
experimental sensitivity; it does not make a formal model-performance claim.

## Frozen layered contract

The tolerances were fixed before the full repeat and were not adjusted after
observing the result.

Strict requirements:

- complete initial scene (platoon plus background actors) must hash exactly;
- identical model input and diffusion noise must produce bit-exact model output;
- per scenario/seed collision, out-of-road, completion and execution-rejection
  outcomes must agree;
- a 5 m/7 m threshold flip is only classified as a harmless boundary ambiguity
  when both measured gaps are within `0.10 m` of the threshold.

Continuous tolerances:

```text
distance/formation/gap       0.10 m
mean speed                   0.25 km/h
episode rate                 0.10
mean reward                  0.10
comfort metrics              0.10
mode/fraction metrics        0.02
```

Finally, the qualitative labels `better / equivalent / worse` are compared for
A+GRPO versus A, B+GRPO versus B, B versus A and B+GRPO versus A+GRPO across
safety, formation, completion and reward metrics.  Timing is excluded from the
behavior comparison.

## Experiment

One Python process sequentially executed two complete diagnostic evaluations:

```text
S5--S9
seeds [31, 47]
10 episodes/model/repeat
4 models
80 closed-loop episodes total
100 simulator steps/episode
```

Artifact:

```text
outputs/bev-round13-95/fixed_process_full.json
SHA256 5577b8c6d5ba1a75727c0f11df1c202d2685282c90cec0e25eac53d5dbe067fd
```

## Results

### Deterministic components

- Complete initial scene hash: exact across both repeats.
- Model input and explicit diffusion noise at the deterministic probe: exact.
- Model output for that identical input/noise: bit-exact for A, B, A+GRPO and
  B+GRPO, both within each run and across repeats.
- Final execution optimizer rejection: zero in both repeats for every model.
- Out-of-road: zero in both repeats for every model.

This localizes the observed divergence to accumulated closed-loop physics and
traffic interaction, not to checkpoint loading, BEV/model inference or diffusion
sampling.

### Critical discrete instability

The relaxed gate still failed.  Most importantly, B+GRPO in
`S7_ego_merge_from_ramp`, seed 47 changed from no collision to collision:

```text
repeat 1: collision=false, min background gap=-0.910 m, min platoon gap=1.538 m
repeat 2: collision=true,  min background gap=-0.043 m, min platoon gap=0.073 m
```

There were also six non-ambiguous 5 m/7 m safety-classification changes in S6
and S7.  They were not accepted as harmless threshold jitter because the paired
minimum gaps were not both within `0.10 m` of the boundary.

### Aggregate sensitivity

| Model | Collision rate repeat 1 -> 2 | 5 m violation | 7 m violation | Reward mean |
|---|---:|---:|---:|---:|
| A | 0.10 -> 0.10 | 0.70 -> 0.80 | 0.40 -> 0.40 | 1.974 -> 2.110 |
| A+GRPO | 0.00 -> 0.00 | 0.70 -> 0.70 | 0.40 -> 0.30 | 1.682 -> 1.724 |
| B | 0.00 -> 0.00 | 0.60 -> 0.60 | 0.50 -> 0.60 | 1.044 -> 1.057 |
| B+GRPO | 0.10 -> 0.20 | 0.70 -> 0.70 | 0.50 -> 0.40 | 1.116 -> 1.670 |

The comparison found 52 continuous-tolerance violations.  Notable changes
include formation P95 differences of `0.45--1.80 m`, maximum-spread differences
up to `10.17 m`, and a B+GRPO reward difference of `0.553`.

Eight qualitative comparison labels changed between repeats.  Examples:

- B+GRPO versus A+GRPO formation changed from `worse` to `better`;
- B+GRPO versus B formation changed from `equivalent` to `better`;
- B versus A 5 m safety changed from `equivalent` to `better`;
- B versus A 7 m safety changed from `equivalent` to `worse`.

Therefore the current diagnostic ranking is not robust to the observed
closed-loop divergence.

With only 10 episodes per model, a single episode changes a rate by 0.10.  For
reference, Wilson 95% intervals are approximately:

```text
0/10: [0.000, 0.278]
1/10: [0.018, 0.404]
2/10: [0.057, 0.510]
```

These intervals overlap substantially, so collision rates of 0.0, 0.1 and 0.2
cannot support a reliable superiority claim at this sample size.

## Decision

Bit-exact closed-loop physics should **not** remain a formal acceptance
requirement.  The layered contract is the more appropriate design:

1. exact initial scene and exact model inference for identical input/noise;
2. exact critical discrete outcomes within a repeat pair;
3. pre-declared continuous equivalence margins;
4. stable model-comparison labels;
5. formal multi-seed confidence intervals/bootstraps.

However, the current experiment does **not** pass that relaxed contract.  The
observed differences are not merely harmless millimetre/centimetre noise: one
collision outcome and multiple model-ranking conclusions changed.

Before a paper-level conclusion, the next formal evaluation should use eligible
checkpoints, at least the five frozen seeds per S5--S9 scenario (25 episodes per
model), multiple independent evaluation repeats, Wilson intervals for rates and
bootstrap confidence intervals for continuous metrics.  A model should only be
declared better when the confidence interval for its paired difference clears
the pre-declared equivalence margin.

## Verification

```text
94 passed
git diff --check: passed
```

The fixed-process evaluator now persists per-episode outcomes, complete initial
scene hashes, deterministic model probes, tolerance violations and conclusion
changes.  No checkpoint metadata or safety threshold was changed.

## Verdict

Round 13.95 infrastructure: **PASS**.

Current diagnostic reproducibility gate: **FAIL**.

Conclusion: exact simulator trajectories are unnecessary, but the present
10-episode diagnostic results are too sensitive to closed-loop divergence to be
used as stable comparative evidence.
