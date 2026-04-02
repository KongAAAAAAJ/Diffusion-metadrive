# 5v2.1 参考策略正则（ref-reg loss）替代 IL loss

> **依赖**：无（可独立实施）
> **改动文件**：`train/ma_grpo_trainer.py`, `models/diffusion/diffusion_rl_scheduler.py`

## 背景

当前 IL loss 的 target 是模型自身生成的 best-of-group 轨迹（`chain[-1][best_idx]`），存在自我强化偏差。将其替换为参考策略正则：用冻结的预训练模型（teacher）约束 RL 微调策略不要偏离太远。

核心思想来自 BDPO 的 pathwise KL，但因为当前模型使用 `prediction_type="sample"`（预测 x_0 而非噪声 ε），简化为 x_0 空间的 L2 距离。

## 数学公式

```
L_ref-reg = (1/T_d) × Σ_{t=1}^{T_d} w_t × ||x0_θ(x_t, t, c) - x0_ref(x_t, t, c)||²

其中：
- x0_θ：当前策略在 diffusion step t 对干净轨迹的预测
- x0_ref：冻结参考策略在同一输入下的预测
- w_t = γ^(T_d - t)，γ=0.8（与 advantage discount 一致）
- T_d：diffusion step 数（当前为 4）
```

## 具体代码改动

### 改动 1：`DiffusionRLScheduler.replay_with_log_prob()` 支持返回 model_outputs

**文件**：`models/diffusion/diffusion_rl_scheduler.py`
**位置**：`replay_with_log_prob()` 方法（line 267）

当前状态：`return_model_outputs=True` 时已返回每步模型预测（`model_outputs: list[Tensor]`），此功能已存在，无需修改。

### 改动 2：新增 `DiffusionRLScheduler.compute_ref_predictions()`

**文件**：`models/diffusion/diffusion_rl_scheduler.py`
**位置**：在 `replay_with_log_prob()` 之后新增方法

```python
def compute_ref_predictions(
    self,
    ref_model,
    batch: dict,
    agent_id: str,
    diffusion_chain: List[Tensor],
) -> List[Tensor]:
    """用冻结参考模型对同一 diffusion chain 计算每步 x0 预测（无梯度）。

    Parameters
    ----------
    ref_model : PlatoonDiffusionPlanner (frozen)
    batch : 观测 batch
    agent_id : 目标 agent
    diffusion_chain : 采样阶段存储的去噪链 [step_num+1 个 Tensor]

    Returns
    -------
    ref_outputs : list[Tensor]
        每步参考模型的 x0 预测，length = step_num，
        每个 Tensor shape = [G, M, 8, 2]（归一化坐标）
    """
    # 实现要点：
    # 1. with torch.no_grad()
    # 2. 调用 ref_model.extract_rl_context(batch) 获取参考模型的 context
    # 3. 遍历 roll_timesteps，对 chain[idx] 调用 ref_model.predict_denoised_traj()
    # 4. 收集每步输出到 list 返回
```

### 改动 3：替换 `MultiAgentGRPOTrainer.compute_il_loss()` → `compute_ref_reg_loss()`

**文件**：`train/ma_grpo_trainer.py`
**位置**：删除 `compute_il_loss()` 方法（line 206-238），替换为：

```python
def compute_ref_reg_loss(self, rollouts: dict) -> Tensor:
    """参考策略正则损失：约束当前策略的 x0 预测不偏离参考策略太远。

    对每个 agent、每个 diffusion step，计算：
        L2(x0_θ(x_t, t, c), x0_ref(x_t, t, c))
    加权汇总后返回标量 loss。

    Returns
    -------
    ref_reg_loss : Tensor (scalar)
    """
    losses = []
    discount = torch.tensor(
        [self.advantage_gamma ** (self.scheduler.step_num - i - 1)
         for i in range(self.scheduler.step_num)],
        dtype=torch.float32, device=self.device,
    )

    for agent_id in rollouts["agent_ids"]:
        chain = rollouts[agent_id]["diffusion_chain"]

        # 1. 当前模型的每步 x0 预测（有梯度）
        _, current_preds = self.scheduler.replay_with_log_prob(
            self.model, rollouts["batch"], agent_id, chain,
            return_model_outputs=True,
        )
        # current_preds: list of [G, M, 8, 2], length = step_num

        # 2. 参考模型的每步 x0 预测（无梯度）
        ref_preds = self.scheduler.compute_ref_predictions(
            self.ref_model, rollouts["batch"], agent_id, chain,
        )
        # ref_preds: list of [G, M, 8, 2], length = step_num

        # 3. 逐步加权 L2
        step_losses = []
        for t, (curr, ref) in enumerate(zip(current_preds, ref_preds)):
            l2 = F.mse_loss(curr, ref.detach(), reduction='mean')
            step_losses.append(discount[t] * l2)

        losses.append(torch.stack(step_losses).sum())

    return torch.stack(losses).mean()
```

### 改动 4：更新 `MultiAgentGRPOTrainer.update()`

**文件**：`train/ma_grpo_trainer.py`
**位置**：`update()` 方法（line 240-302）

将：
```python
il_loss = self.compute_il_loss(rollouts)
has_positive = any(bool(torch.any(adv > 0).item()) for adv in advantages.values())
il_weight = self.il_weight_default if has_positive else self.il_weight_no_positive
loss = rl_loss + il_weight * il_loss
```

替换为：
```python
ref_reg_loss = self.compute_ref_reg_loss(rollouts)
# β_reg 调度：训练初期大（保护预训练能力），后期小（允许探索）
beta_reg = self._get_beta_reg()
loss = rl_loss + beta_reg * ref_reg_loss
```

### 改动 5：新增 `_get_beta_reg()` 方法

**文件**：`train/ma_grpo_trainer.py`

```python
def _get_beta_reg(self) -> float:
    """β_reg 退火调度：
    - 前 30% 训练步：β_reg = beta_reg_max（强保护）
    - 中期：线性衰减
    - 后 30%：β_reg = beta_reg_min（弱保护）
    """
    total = self.config.get("total_steps", 500)
    beta_max = float(self.config.get("beta_reg_max", 1.0))
    beta_min = float(self.config.get("beta_reg_min", 0.1))
    warmup_frac = float(self.config.get("beta_reg_warmup_frac", 0.3))
    decay_frac = float(self.config.get("beta_reg_decay_frac", 0.4))

    step = self.global_step
    warmup_end = int(total * warmup_frac)
    decay_end = int(total * (warmup_frac + decay_frac))

    if step < warmup_end:
        return beta_max
    elif step < decay_end:
        progress = (step - warmup_end) / max(decay_end - warmup_end, 1)
        return beta_max - progress * (beta_max - beta_min)
    else:
        return beta_min
```

### 改动 6：更新 metrics 返回

**文件**：`train/ma_grpo_trainer.py`
**位置**：`update()` 返回 dict

将 `il_loss` / `il_weight` 替换为 `ref_reg_loss` / `beta_reg`。

## 配置新增

**文件**：`configs/train/platoon_grpo.yaml`（或 v2.yaml）

```yaml
# 参考策略正则
beta_reg_max: 1.0        # 训练初期 β_reg
beta_reg_min: 0.1        # 训练后期 β_reg
beta_reg_warmup_frac: 0.3  # 前 30% 保持最大值
beta_reg_decay_frac: 0.4   # 中间 40% 线性衰减
```

## 验收标准

**交付物**：修改后的 `ma_grpo_trainer.py`、`diffusion_rl_scheduler.py`、`tests/acceptance/test_phase5v2_task1.py`

**验收指标**（8 项）：
1. `compute_ref_reg_loss()` 存在，返回有限标量 Tensor
2. `compute_ref_predictions()` 存在，返回 list of Tensor，长度 = step_num
3. ref_reg_loss > 0（当前模型与参考模型输出不同）
4. ref_reg_loss 可 backward 且梯度有限
5. `_get_beta_reg()` 在 step=0 返回 beta_reg_max，step=total_steps 返回 beta_reg_min
6. `update()` 返回 dict 包含 `ref_reg_loss` 和 `beta_reg` 字段
7. 旧的 `compute_il_loss()` 方法已删除
8. 连续 3 次 update，ref_reg_loss 有限且非 NaN

**验收命令**：
```bash
pytest tests/acceptance/test_phase5v2_task1.py -v
```

## 注意事项

1. `compute_ref_predictions()` 必须在 `torch.no_grad()` 下运行参考模型
2. `replay_with_log_prob()` 在同一个 `update()` 中会被调用两次（一次为 RL loss，一次为 ref-reg loss 的 current_preds）。可以优化为一次调用同时返回 log_prob 和 model_outputs，避免重复计算。具体做法：在 `compute_rl_loss()` 中传入 `return_model_outputs=True`，将 model_outputs 缓存后传给 `compute_ref_reg_loss()`
3. 保留 `compute_il_loss` 的旧代码作为注释或 `_compute_il_loss_legacy()`，以便回退对比
