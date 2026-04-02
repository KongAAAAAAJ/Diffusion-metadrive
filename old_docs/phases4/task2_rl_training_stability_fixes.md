# Phase 6B.2：RL 训练稳定性修复

> **前置依赖**：Phase 6B.1（surrogate motion prediction）已合入
> **目标**：修复 5 个导致 MA-GRPO 训练无法收敛的独立 bug
> **预期效果**：LR 不再单调衰减至地板值；BN running stats 不再漂移；beta_reg 调度与实际步数匹配；team_reward 不再恒定 -10；梯度裁剪阈值合理

## 修改内容

共涉及 3 个文件，5 处修改。每处修改独立，无交叉依赖。

---

### Fix 1：LR 双向自适应（替换单调衰减）

**文件**：`train/train_platoon_rl.py`，约第 650-654 行

**当前代码**：
```python
kl_threshold = float(config.get("kl_threshold", 5.0))
if float(metrics["kl"]) > kl_threshold:
    decay_factor = float(config.get("lr_decay_factor", 0.5))
    for group in trainer.optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * decay_factor, 1e-6)
```

**问题**：LR 只降不升，触发 6 次后就到地板值 1e-6，模型几乎不更新。

**替换为**：
```python
kl = float(metrics["kl"])
kl_target = float(config.get("kl_target", 8.0))
kl_high = kl_target * 1.5
kl_low = kl_target * 0.5
lr_max = float(config.get("lr", 5e-5))
lr_min = 1e-6
lr_decay = float(config.get("lr_decay_factor", 0.8))
lr_grow = float(config.get("lr_grow_factor", 1.05))
for group in trainer.optimizer.param_groups:
    if kl > kl_high:
        group["lr"] = max(float(group["lr"]) * lr_decay, lr_min)
    elif kl < kl_low:
        group["lr"] = min(float(group["lr"]) * lr_grow, lr_max)
```

**同时更新 config**（`configs/train/platoon_grpo_v2.yaml`）：
- 移除 `kl_threshold: 5.0`（不再使用）
- 添加 `kl_target: 8.0`
- 修改 `lr_decay_factor: 0.8`（从 0.5 改为更温和的 0.8）
- 添加 `lr_grow_factor: 1.05`

---

### Fix 2：冻结 Backbone 中 BatchNorm 的 running stats

**文件**：`train/train_platoon_rl.py`，`_apply_freeze_config()` 函数末尾（约第 290 行之后）

**问题**：backbone 的 ResNet34 包含 36 个 BatchNorm2d 层。`_apply_freeze_config` 只冻结了 `requires_grad`，但模型处于 `train()` 模式，BN 的 `running_mean` / `running_var` 仍在更新。RL 的有效 batch_size=1，导致 running stats 剧烈抖动，backbone 输出不稳定。

**在 `_apply_freeze_config` 函数末尾（`print(...)` 之前）添加**：
```python
if freeze_backbone:
    frozen_bn_count = 0
    for name, module in model.named_modules():
        if any(keyword in name for keyword in backbone_keywords):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
                frozen_bn_count += 1
    print(f"[freeze] Froze {frozen_bn_count} BatchNorm layers in backbone")
```

**注意**：需要在文件顶部确认已有 `from torch import nn`（或 `import torch.nn as nn`）。如果没有，需添加。

**额外**：为确保 BN 在整个训练过程中保持 eval 模式，需要在 `_apply_freeze_config` 中将冻结的 BN 注册为钩子，或者更简单的方案——在训练循环每次 `model.train()` 调用后重新 eval BN。但当前代码中训练循环没有显式调用 `model.train()`，所以只需确保 `_apply_freeze_config` 在 `model.to(device)` 之前或之后都能正确生效即可。

实际上，`module.eval()` 只影响该 module 的 forward 行为（不更新 running stats），但如果后续有人调用 `model.train()`，子模块会被递归设回 train 模式。为了防御这种情况，**同时设置 `module.track_running_stats = False`**，这样即使被设回 train 模式，BN 也不会更新 running stats。

---

### Fix 3：beta_reg 调度的 total_steps 与实际训练步数对齐

**文件**：`train/train_platoon_rl.py`，`build_runtime()` 函数中初始化 trainer 之前（约第 571 行之前）

**问题**：`_get_beta_reg()` 读取 `self.config["total_steps"]`，config yaml 中写死 `500`。实际训练如果传入 `--steps 10000`，beta_reg 在 step 350 后就永久停留在 `beta_min=0.1`，占训练 96.5% 的时间 KL 正则化不足。

**修改**：在 `build_runtime` 中创建 trainer 之前，将实际训练步数写入 config：

找到以下代码（约第 571 行）：
```python
trainer = MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=env, config=config)
```

在其**之前**添加一行：
```python
config["total_steps"] = int(steps or config.get("total_steps", 5))
```

---

### Fix 4：team_reward 碰撞惩罚从一票否决改为累加

**文件**：`evaluation/reward_terms.py`，`compute_team_reward()` 函数（第 75-104 行）

**问题**：`any_crash` 只要联合轨迹中任一 agent crash，整组就 `-collision_penalty`。在碰撞率 ~17% 时，8 组联合轨迹（每组 3 agent）中约 `1 - (1-0.17)^3 = 43%` 的组仍会触发 `-10`，team_reward 信号仍然很稀疏。

**当前代码**（第 91-96 行）：
```python
any_crash = False
for step_infos in per_agent_step_infos.values():
    for info in step_infos:
        total_formation_error += abs(_as_float(info, "formation_error", 0.0))
        total_progress += max(_as_float(info, "progress", 0.0), 0.0)
        any_crash = any_crash or _as_bool(info, "crash", False)
        count += 1
```

以及第 102 行：
```python
safety_term = -collision_penalty if any_crash else 0.0
```

**替换为**按 agent 计数碰撞的累加惩罚：
```python
crash_count = 0
num_agents = len(agent_ids)
for step_infos in per_agent_step_infos.values():
    agent_crashed = False
    for info in step_infos:
        total_formation_error += abs(_as_float(info, "formation_error", 0.0))
        total_progress += max(_as_float(info, "progress", 0.0), 0.0)
        agent_crashed = agent_crashed or _as_bool(info, "crash", False)
        count += 1
    if agent_crashed:
        crash_count += 1
```

以及将 safety_term 替换为：
```python
safety_term = -collision_penalty * (crash_count / max(num_agents, 1))
```

这样：0 个 agent crash → 0 惩罚，1/3 crash → -3.33，2/3 → -6.67，3/3 → -10.0。提供梯度友好的连续信号。

---

### Fix 5：梯度裁剪阈值放宽

**文件**：`configs/train/platoon_grpo_v2.yaml`

**当前**：`max_grad_norm: 5.0`

**修改为**：`max_grad_norm: 10.0`

**理由**：当可行 anchor 少时，梯度信号集中在少量轨迹上，范数偏大。5.0 过于激进地裁剪了有效梯度。放宽到 10.0 配合双向 LR 调节即可。

---

## configs/train/platoon_grpo_v2.yaml 最终状态

```yaml
group_size: 2
num_agents: 3
lr: 0.00005
kl_target: 8.0
lr_decay_factor: 0.8
lr_grow_factor: 1.05
max_grad_norm: 10.0
total_steps: 500

ddim_steps: 4
ddim_eta: 0.02
advantage_discount_gamma: 0.8

beta_reg_max: 1.0
beta_reg_min: 0.1
beta_reg_warmup_frac: 0.3
beta_reg_decay_frac: 0.4

lambda_local: 0.7
lambda_team: 0.3
joint_top_k: 2
num_joint_groups: 8
use_closedloop: true
closedloop_workers: 8

reward_config:
  delta_s_max: 10.0
  d_norm: 10.0
  d_safe: 8.0
  w_progress: 1.0
  w_formation: 0.5
  w_safety: 0.3
  w_collision: 10.0
  w_road: 5.0
  w_comfort: 0.1
  w_team_formation: 0.5
  w_team_safety: 1.0
  w_team_efficiency: 0.3
  w_team_collision: 10.0

traffic_density: 0.08
horizon: 100
max_env_steps_per_rollout: 20

freeze_backbone: true
freeze_tf_decoder: false
freeze_trajectory_head: false
```

变更：移除 `kl_threshold`，新增 `kl_target`、`lr_decay_factor`（值从0.5改0.8）、`lr_grow_factor`，修改 `max_grad_norm` 从 5.0 到 10.0。

---

## 不需要修改的文件

- `envs/platoon_env.py` — 本任务不涉及
- `train/ma_grpo_trainer.py` — `_get_beta_reg()` 逻辑不变，只是 config 中 `total_steps` 值被外部修正
- `train/closedloop_executor.py` — 不涉及
- `models/` 下所有文件 — 不涉及

---

## 验收测试

创建 `tests/acceptance/test_phase6b_task2.py`，包含以下测试：

### Test 1：`test_lr_bidirectional_adaptation`

```python
def test_lr_bidirectional_adaptation():
    """LR should decrease when KL > kl_target*1.5 and recover when KL < kl_target*0.5."""
    # 模拟训练循环中 LR 调节逻辑：
    #   1. 设置 kl_target=8.0, lr=5e-5
    #   2. 模拟 kl=15.0（> 12.0）连续 10 次 → LR 应下降但不低于 1e-6
    #   3. 模拟 kl=2.0（< 4.0）连续 50 次 → LR 应恢复到接近 5e-5
    #
    # 断言：
    #   - 高 KL 后 LR < 初始值
    #   - 低 KL 恢复后 LR > 衰减后的最低值
    #   - LR 始终在 [1e-6, 5e-5] 范围内
    #   - 恢复后 LR 与初始 LR 的差距 < 20%
```

### Test 2：`test_bn_frozen_in_backbone`

```python
def test_bn_frozen_in_backbone():
    """After _apply_freeze_config with freeze_backbone=True,
    all BatchNorm layers in backbone should have track_running_stats=False."""
    # 构造一个包含 BatchNorm 的简单模型，模拟 backbone 命名规则
    # 调用 _apply_freeze_config
    # 断言：
    #   - 所有名称含 backbone_keywords 的 BN 层：track_running_stats == False
    #   - 非 backbone 的 BN 层（如果有）：track_running_stats 不受影响
    #   - forward 通过不报错
```

### Test 3：`test_beta_reg_uses_actual_total_steps`

```python
def test_beta_reg_uses_actual_total_steps():
    """beta_reg schedule should span the actual training steps, not config default."""
    # 用 ToyPlanner 构造 MultiAgentGRPOTrainer
    # 设置 config["total_steps"] = 10000
    # 断言：
    #   - step 0: beta_reg == beta_reg_max (1.0)
    #   - step 3000 (warmup_end): beta_reg == 1.0
    #   - step 5000 (mid-decay): 0.1 < beta_reg < 1.0
    #   - step 7000 (decay_end): beta_reg == beta_reg_min (0.1)
    #   - step 9999: beta_reg == 0.1
```

### Test 4：`test_team_reward_proportional_crash_penalty`

```python
def test_team_reward_proportional_crash_penalty():
    """Team reward crash penalty should scale with fraction of crashed agents."""
    from evaluation.reward_terms import compute_team_reward
    config = {"w_team_collision": 10.0, "w_team_formation": 0.5,
              "w_team_safety": 1.0, "w_team_efficiency": 0.3}

    # Case 1: 0/3 agents crash → safety_term == 0
    # Case 2: 1/3 agents crash → safety_term == -10/3 ≈ -3.33
    # Case 3: 3/3 agents crash → safety_term == -10.0
    #
    # 断言：
    #   - reward_0crash > reward_1crash > reward_3crash
    #   - reward_1crash - reward_0crash ≈ -(10.0 * 1/3) * w_team_safety（在容差内）
    #   - reward_3crash ≈ 原来的 any_crash 结果（向后兼容极端情况）
```

### Test 5：`test_config_consistency`

```python
def test_config_consistency():
    """Verify platoon_grpo_v2.yaml has expected fields after modification."""
    import yaml
    config = yaml.safe_load(open("configs/train/platoon_grpo_v2.yaml"))

    # 断言：
    #   - "kl_threshold" not in config（已移除）
    #   - config["kl_target"] == 8.0
    #   - config["lr_decay_factor"] == 0.8
    #   - config["lr_grow_factor"] == 1.05
    #   - config["max_grad_norm"] == 10.0
```

### Test 6：`test_no_regression_on_existing_training_loop`

```python
def test_no_regression_on_existing_training_loop():
    """toy-single mode should still complete 5 training steps without error."""
    # 使用 build_runtime(mode="toy-single", ..., run_training=False)
    # 手动执行 5 步 trainer.update()
    # 断言：
    #   - 无异常
    #   - metrics 中包含 loss, kl, mean_reward 等字段
    #   - loss 是有限值
```

---

## 实施注意事项

1. **Fix 1 和 Fix 5 有关联**：都涉及 `platoon_grpo_v2.yaml`，确保最终 yaml 状态与本文档"最终状态"一致
2. **Fix 2 需要导入 `nn`**：检查 `train_platoon_rl.py` 顶部是否已有 `import torch.nn as nn` 或 `from torch import nn`
3. **Fix 3 的 `steps` 变量**：在 `build_runtime` 函数签名中已有 `steps: int` 参数，直接使用即可
4. **Fix 4 注意向后兼容**：当所有 agent 都 crash 时，新公式 `crash_count/num_agents = 1.0`，结果与原来 `any_crash=True` 完全一致
5. **运行现有测试确保无回归**：`/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase6a_task*.py tests/acceptance/test_phase6b_task1.py -x`
