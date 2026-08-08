# Round 13.89 closeout report

## Outcome

Round 13.89a and 13.89b passed. Round 13.89c failed its development
calibration gate, so Round 13.89d, Round 13.89e, and Round 14 were not
started.

This is an intentional hard stop. The failed calibration reports are not
eligible inputs to online GRPO.

## 13.89a: balanced diagnostic-64

Dataset:

```text
/tmp/bev-round13-89a-diagnostic64
```

Collection result:

```text
attempted episodes: 37
stored episodes:    34
rejected episodes:   3
joint samples:      64
S5/S6/S7/S8/S9:     13/13/13/13/12
scenario contract:  57d6192878fbad5de1ad69997fb60e10c2ba941e5feabf8597a6c4846dcb3133
```

Rejected episodes were not persisted:

```text
all_rule_proposals_infeasible:             2
committed_trajectory_tracking_deviation:   1
```

Full verifier passed:

```text
schema v2 / complete scan:                 pass
expert trajectories:                      192/192 valid
hard-valid dynamic anchors:               1118/1118 valid
STOP anchors:                              192/192 valid
BEV compression ratio:                    6.4x
decode throughput:                        917.53 joint samples/s
train/val/test joint samples:              59/2/3
```

## 13.89b: diagnostic Stage 1 A/B

Both variants were initialized from ResNet18.a1 and trained with the new
dataset in `overfit_64` mode. No old Stage 1 checkpoint was loaded.

| Variant | Optimizer step | Loss reduction | Mode accuracy | GT-mode ADE | Result |
|---|---:|---:|---:|---:|---|
| A | 180 | 80.64% | 100% | 0.9493 m | pass |
| B | 160 | 78.05% | 100% | 0.9859 m | pass |

Artifacts:

```text
/tmp/bev-stage1-round13-89/A/run_1/checkpoints/diagnostic.pt
/tmp/bev-stage1-round13-89/B/run_1/checkpoints/diagnostic.pt
/tmp/bev-stage1-round13-89/validation.json
```

Strict checkpoint reload, fixed-noise deterministic open-loop inference, and
one real sensorless closed-loop step passed for both variants. All artifacts
remain diagnostic-only and are not eligible for formal training.

## 13.89c: proxy--simulator development calibration

Reports:

```text
/tmp/bev-round13-89-calibration-A-development.json
/tmp/bev-round13-89-calibration-B-development.json
```

| Metric | Required | A | B |
|---|---:|---:|---:|
| Informative groups | >=15 | 29 | 25 |
| Mean group Spearman | >=0.50 | 0.1833 | 0.1597 |
| Pairwise agreement | >=70% | 61.24% | 58.56% |
| False-safe count | 0 | 22 | 29 |
| Tracking lateral P95 | <=0.5 m | 1.785 m | 1.677 m |
| Tracking heading P95 | <=0.1 rad | 0.614 rad | 0.602 rad |
| Calibration passed | true | false | false |

The tracking envelope is also outside its permitted cap:

| Variant | Longitudinal P99 | Lateral P99 | Heading P99 |
|---|---:|---:|---:|
| A | 17.610 m | 3.055 m | 0.996 rad |
| B | 17.803 m | 2.662 m | 1.034 rad |

## Root cause

The current blocker is the final diffusion trajectory contract, not the
dynamic anchors and not merely a PID gain issue.

For both variants, all sampled final diffusion trajectories failed the shared
production kinematic audit:

```text
Variant A final diffusion trajectories: 0/360 valid
Variant B final diffusion trajectories: 0/360 valid
Variant A dynamic anchors:              526/526 valid
Variant B dynamic anchors:              526/526 valid
```

Violation counts:

| Violation | A | B |
|---|---:|---:|
| acceleration above maximum | 358 | 358 |
| acceleration below minimum | 307 | 296 |
| heading/translation misalignment | 359 | 360 |
| non-forward motion | 318 | 305 |
| outside reachable distance | 282 | 271 |
| curvature limit | 1 | 2 |

The controller consequently spent up to 3.7 s continuously saturated and
could not faithfully execute the sampled trajectories. Enlarging the tracking
envelope or starting GRPO with these rollouts would convert known invalid
actions into training data and violate the frozen safety contract.

## Gated remainder

The following were deliberately not run:

- independent holdout calibration on seeds 31 and 47;
- A/B 20-step online GRPO;
- A/B/A+GRPO/B+GRPO four-model evaluation;
- Round 14 legacy-code cleanup.

Before resuming, the planner must enforce or learn a dynamically executable
final trajectory representation. The fix must be validated on final diffusion
outputs, not only on coarse anchors. It must not be implemented as an unsafe
trajectory fallback, a relaxed kinematic boundary, or a larger proxy margin.
