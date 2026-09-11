# Risk-PACT pilot

This pilot adds the minimum components needed to validate a dynamic-risk-field + PACT-style post-training idea without modifying the existing GRPO path.

## Current scope
- Analytic anisotropic Gaussian dynamic risk field.
- Constant-velocity extrapolation of background actors from the current 8-D actor state.
- Dynamic level-set safety constraint with a softmax-weighted temporal maximum.
- Exact zero safety supervision for clearly safe trajectories.
- x0-space PACT-lite teacher: one normalized risk-gradient correction in clean trajectory space.
- Standalone sanity check and unit tests.

## Actor state convention
The current BEV planner normalizes background actor tokens with scales `[64, 32, 1, 1, 30, 30, 10, 4]`; the pilot therefore uses the convention:

`[x, y, cos(yaw), sin(yaw), vx, vy, length, width]`.

Verify this against the dataset producer before a formal run.

## Run the sanity check
```bash
PYTHONPATH=. python -m train.bev_risk_pact.sanity_check
```
Expected behavior: one teacher step must reduce trajectory risk for the synthetic unsafe example.

## Run tests
```bash
PYTHONPATH=. pytest -q tests/test_bev_risk_pact.py
```

## Next integration step
1. Obtain frozen old-policy x0 predictions from `BEVOnlyDiffusionPlanner.predict_denoised_candidates`.
2. Feed `[B,3,10,8,3]` candidates plus `[B,3,16,8]` background actor states into `build_x0_pact_teacher`.
3. Select the same mode/sample subset used for student training.
4. Update only the current mode-residual trajectory head in the first pilot.
5. Log before/after risk, violation rate, teacher displacement, and policy drift.

## Important limitation
`build_x0_pact_teacher` is intentionally a PACT-lite approximation. It makes a normalized Euclidean gradient step on the predicted clean trajectory. It is not mathematically identical to the reverse-KL score-space teacher in PACT. The strict score-space implementation should be added only after this pilot verifies that the risk field and level-set gradients are useful.

## Training-time visualization switch

Risk-field visualization is designed as a training-loop hook rather than a
standalone validation entry point.  For a smoke run, create the visualizer once
before the optimizer loop:

```python
from train.bev_risk_pact import (
    RiskPACTConfig,
    RiskPACTTrainingVisualizer,
    RiskPACTVisualizationConfig,
)

risk_cfg = RiskPACTConfig()
risk_visualizer = RiskPACTTrainingVisualizer(
    run_dir=run_dir,
    risk_config=risk_cfg,
    visualization_config=RiskPACTVisualizationConfig(
        enabled=True,
        interval_steps=5,
        start_step=0,
        max_events=4,
        batch_index=0,
        role_index=0,
        mode_index=0,
    ),
)
```

Immediately after constructing the PACT teacher in the training loop:

```python
teacher_result = build_x0_pact_teacher(
    old_trajectory,
    actor_state,
    actor_valid_mask,
    curriculum_scale=curriculum_scale,
    config=risk_cfg,
)

risk_visualizer.maybe_save(
    step=optimizer_step,
    actor_state=actor_state,
    actor_valid_mask=actor_valid_mask,
    old_trajectory=old_trajectory,
    teacher_result=teacher_result,
)
```

With the example settings, visualization events are emitted at optimizer steps
0, 5, 10, and 15 into:

```text
<run_dir>/risk_field_visualizations/
```

Each event contains four images: risk snapshots, dynamic safe-tube contours,
old-vs-teacher correction, and the per-horizon risk curve.

For a formal run, set only:

```python
RiskPACTVisualizationConfig(enabled=False)
```

When disabled, the hook does not import matplotlib and writes no images, so it
has effectively zero plotting overhead in formal training.

A YAML-style training configuration can mirror the same fields:

```yaml
risk_pact:
  visualization:
    enabled: true          # smoke: true; formal: false
    interval_steps: 5
    start_step: 0
    max_events: 4
    batch_index: 0
    role_index: 0
    mode_index: 0
    output_subdir: risk_field_visualizations
```
