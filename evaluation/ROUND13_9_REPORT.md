# Round 13.9 — execution-time trajectory optimization

## Frozen policy/execution boundary

```text
Stage 1 diffusion raw trajectory
  -> GRPO raw rollout and raw DDIM log-prob
  -> deterministic KinematicTrajectoryOptimizer
  -> proxy reward / simulator branch / trajectory controller
```

The transform never writes optimized waypoints into
`JointGRPORollout.selected_trajectories`, `chains_normalized`, or predecessor
history. Thus GRPO still optimizes the probability of the raw diffusion action;
the reward is measured on the deterministic action actually executed.

## Implementation

- `models/bev_planner/trajectory_optimizer.py`
  - selected-mode coarse anchor is the semantic trust reference;
  - raw XY/heading residual is bounded around the selected hard-valid anchor;
  - a deterministic descending line search retains the largest fraction whose
    float32 result passes the fixed 0.5 s production kinematic contract;
  - STOP uses the selected valid STOP anchor directly;
  - final float32 output must pass `validate_trajectory_kinematics()`.
- Only the selected three trajectories (or GRPO's `G x 3`) are transformed;
  the ten-mode candidate tensor and network structure are unchanged.
- Calibration, proxy reward, simulator branch, online GRPO action execution,
  fixed validation, Stage 1 closed-loop smoke, and four-model evaluation use the
  transformed action.
- Checkpoints/calibration reports freeze optimizer config and SHA256.

## Real checkpoint audit

Artifacts:

- `/tmp/bev-round13-9-optimizer-A.json`
- `/tmp/bev-round13-9-optimizer-B.json`

Eight real joint samples per variant (`G=4`, 96 trajectories/variant):

| Variant | raw valid | optimized valid | intervention ADE | FDE | optimizer P95 |
|---|---:|---:|---:|---:|---:|
| A | 0/96 | 96/96 | 3.163 m | 4.924 m | 11.47 ms |
| B | 0/96 | 96/96 | 3.098 m | 4.259 m | 12.09 ms |

Both reports assert `raw_grpo_rollout_unchanged=true`.
Both preserve sampled lane-change direction for 100% of audited lane-change
actions. Mean output distance to the selected coarse anchor is 0.325 m (A) and
0.263 m (B); retained raw fractions are 18.1% and 14.6% respectively.

## S5–S9 development calibration

Artifacts:

- `/tmp/bev-round13-9-calibration-A-development.json`
- `/tmp/bev-round13-9-calibration-B-development.json`

All 360 transformed diffusion trajectories per variant pass the production
kinematic audit. Raw validity remains 0/360, as expected for the current
diagnostic checkpoints.

| Variant | Spearman | Pairwise | False-safe | lateral P95 | heading P95 |
|---|---:|---:|---:|---:|---:|
| A | 0.741 | 86.8% | 11 | 0.262 m | 0.058 rad |
| B | 0.627 | 80.0% | 13 | 0.253 m | 0.060 rad |

Conclusion: final diffusion trajectory executability and tracking gates are
fixed at the action boundary, and both variants now meet development reward
ranking thresholds. Calibration remains blocked only by false-safe (A: 8
termination-semantics + 3 controller-tracking; B: 10 + 3). This is the existing
tracking-aware proxy/termination task, not a kinematic-output failure. Holdout
calibration and GRPO training remain blocked until that independent task is
closed. Future reward shaping or diffusion guidance should reduce raw-policy
violations and optimizer intervention, without changing this execution safety
contract.
