# ChassisFusion `skip_all_unsafe_group` semantic installer v3

This package is intentionally **not** a `git apply` unified diff. Your local BEV branch has diverged from the raw GitHub context after several incremental patches, so line-context patches can fail even when the intended logic is correct.

The installer locates the existing GRPO fields/functions semantically and performs an idempotent edit.

## Intended behavior

Keep your existing setting:

```yaml
grpo:
  advantage:
    unsafe_override_enabled: true
    unsafe_advantage_value: -1.0
```

and add:

```yaml
    skip_all_unsafe_group: true
```

For each valid `(vehicle, mode)` group with N=48 candidates:

- mixed safe/unsafe: unsafe candidates keep the configured fixed negative advantage, safe candidates keep standard GRPO z-score advantages;
- all safe: standard GRPO unchanged;
- **all unsafe** (`collision OR out_of_drivable` for all 48): all 48 advantages are zeroed and `signal_mode_mask=False`, so that group contributes no trajectory policy-gradient;
- BC/reference-KL and optional auxiliary safety losses remain governed by their existing valid-mode masks and are not globally disabled by this PG mask.

## Apply

Run from the repository root:

```bash
python3 /path/to/apply_skip_all_unsafe_group.py --check
python3 /path/to/apply_skip_all_unsafe_group.py --apply
```

The first command writes nothing. The second creates backups named `*.bak_skip_all_unsafe` before modifying files.

## Files touched

- `models/bev_planner/joint_grpo.py`
- `train/bev_joint_grpo_online/config.py`
- `train/bev_joint_grpo_online/train.py`
- `train/bev_joint_grpo_online/rollout.py`
- `train/bev_joint_grpo_online/runner.py`
- `configs/train/bev_joint_grpo.yaml`

The script does **not** change your current value of `unsafe_override_enabled`; it only adds/enables `skip_all_unsafe_group`.

## After applying

Recommended quick checks:

```bash
python3 -m py_compile \
  models/bev_planner/joint_grpo.py \
  train/bev_joint_grpo_online/config.py \
  train/bev_joint_grpo_online/train.py \
  train/bev_joint_grpo_online/rollout.py \
  train/bev_joint_grpo_online/runner.py

grep -R "skip_all_unsafe_group" -n \
  models/bev_planner/joint_grpo.py \
  train/bev_joint_grpo_online \
  configs/train/bev_joint_grpo.yaml
```
