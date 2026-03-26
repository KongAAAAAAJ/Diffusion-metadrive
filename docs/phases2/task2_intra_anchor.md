# 5v2.2 Intra-anchor Advantage

> **依赖**：无（可独立实施）
> **改动文件**：`train/ma_grpo_trainer.py`, `models/diffusion/diffusion_rl_scheduler.py`

## 背景

当前实现只使用 `mode_index=0`（第一个 anchor），浪费了 diffusion model 的多模态能力。DiffusionDriveV2 的核心设计是**每个 anchor 独立采样 G 条轨迹，在同一 anchor 内部比较 advantage**，避免不同意图之间粗暴比较导致 mode collapse。

## 当前代码问题

**`ma_grpo_trainer.py` line 102**：
```python
mode_index = torch.zeros((trajectory_all.shape[0],), dtype=torch.long, ...)
# 永远取 mode 0，其他 anchor 的轨迹被丢弃
```

**`diffusion_rl_scheduler.py` line 205**：
```python
plan_xy = th.plan_anchor[..., :2].unsqueeze(0).repeat(num_groups, 1, 1, 1)
# 已经对 M 个 anchor 都做了采样，但下游只用 mode 0
```

## 改动目标

- 每个 anchor k ∈ [0, M) 独立采样 G 条轨迹
- 每个 anchor 独立评估 reward → 独立计算 advantage
- RL loss 在所有 anchor 上求和

## 具体代码改动

### 改动 1：`collect_group_samples()` 支持多 anchor 评估

**文件**：`train/ma_grpo_trainer.py`
**位置**：`collect_group_samples()` 方法（line 89-134）

核心改动：不再把 mode_index 固定为 0，而是对每个 anchor 独立评估。

```python
def collect_group_samples(self, group_size: int = 4, obs: dict | None = None) -> dict:
    if obs is None:
        obs = self.current_obs if self.current_obs is not None else self.env.reset()
    self.current_obs = obs
    batch = _obs_to_tensor_batch(obs, self.device)
    with torch.no_grad():
        samples = self.scheduler.sample_with_log_prob(self.model, batch, num_groups=group_size)

    rollouts: dict = {"agent_ids": list(batch.keys()), "batch": batch}
    for agent_id in batch.keys():
        sampled = samples[agent_id]
        trajectory_all = sampled["trajectory"]      # [G, M, 8, 3]
        log_prob_all = sampled["log_prob"]          # [G, M, step_num]
        G, M = trajectory_all.shape[0], trajectory_all.shape[1]

        # --- 改动：对每个 anchor 独立评估 reward ---
        # rewards_per_anchor: [G, M]
        # crash_per_anchor: [G, M]
        # out_per_anchor: [G, M]
        rewards_per_anchor = torch.zeros(G, M, device=self.device)
        crash_per_anchor = torch.zeros(G, M, dtype=torch.bool, device=self.device)
        out_per_anchor = torch.zeros(G, M, dtype=torch.bool, device=self.device)
        formation_per_anchor = torch.zeros(G, M, device=self.device)

        for k in range(M):
            traj_k = trajectory_all[:, k, :, :]    # [G, 8, 3]
            evaluation = self._evaluate_trajectory_group(agent_id, traj_k)
            rewards_k = torch.tensor(
                [compute_trajectory_reward(si, self.reward_config) for si in evaluation["step_infos"]],
                dtype=torch.float32, device=self.device,
            )
            rewards_per_anchor[:, k] = rewards_k
            crash_per_anchor[:, k] = torch.as_tensor(evaluation["crash_flags"], dtype=torch.bool, device=self.device)
            out_per_anchor[:, k] = torch.as_tensor(evaluation["out_of_road_flags"], dtype=torch.bool, device=self.device)
            formation_per_anchor[:, k] = torch.tensor(
                [float(sum(float(s.get("formation_error", 0.0)) for s in si) / max(len(si), 1))
                 for si in evaluation["step_infos"]],
                dtype=torch.float32, device=self.device,
            )

        # 为 step_env_with_best 选出全局最优轨迹
        flat_idx = torch.argmax(rewards_per_anchor.view(-1))
        best_g = int(flat_idx // M)
        best_k = int(flat_idx % M)

        rollouts[agent_id] = {
            "trajectory_all": trajectory_all.detach(),       # [G, M, 8, 3]
            "log_prob_all": log_prob_all.detach(),           # [G, M, step_num]
            "reward_per_anchor": rewards_per_anchor,         # [G, M]
            "crash_per_anchor": crash_per_anchor,            # [G, M] bool
            "out_per_anchor": out_per_anchor,                # [G, M] bool
            "formation_per_anchor": formation_per_anchor,    # [G, M]
            "diffusion_chain": sampled["diffusion_chain"],   # list of [G, M, 8, 2]
            "best_g": best_g,
            "best_k": best_k,
            # 向后兼容：保留旧字段
            "trajectory": trajectory_all[:, best_k, :, :].detach(),  # [G, 8, 3]
            "reward": rewards_per_anchor[:, best_k],                  # [G]
            "crash_flag": crash_per_anchor[:, best_k],
            "out_of_road_flag": out_per_anchor[:, best_k],
            "formation_error": formation_per_anchor[:, best_k],
            "log_prob": log_prob_all[:, best_k, :].detach(),
            "mode_index": torch.full((G,), best_k, dtype=torch.long, device=self.device),
        }
    return rollouts
```

### 改动 2：`compute_advantages()` 按 anchor 独立归一化

**文件**：`train/ma_grpo_trainer.py`
**位置**：`compute_advantages()` 方法（line 155-183）

```python
def compute_advantages(self, rollouts: dict) -> dict:
    """Intra-anchor advantage：每个 anchor 内独立归一化。

    Returns
    -------
    advantages : dict[agent_id -> Tensor [G, M, step_num]]
    """
    advantages: dict = {}
    discount = torch.tensor(
        [self.advantage_gamma ** (self.scheduler.step_num - i - 1)
         for i in range(self.scheduler.step_num)],
        dtype=torch.float32, device=self.device,
    )

    for agent_id in rollouts["agent_ids"]:
        reward = rollouts[agent_id]["reward_per_anchor"]       # [G, M]
        crash = rollouts[agent_id]["crash_per_anchor"]         # [G, M]
        out = rollouts[agent_id]["out_per_anchor"]             # [G, M]

        # 1. 每个 anchor 独立归一化（dim=0 是 group 维度）
        mean_r = reward.mean(dim=0, keepdim=True)               # [1, M]
        std_r = reward.std(dim=0, unbiased=False, keepdim=True) # [1, M]
        normalized = (reward - mean_r) / (std_r + 1e-4)         # [G, M]

        # 2. 正样本过滤：只保留优于组内均值的
        positive_mask = reward > mean_r                          # [G, M]
        normalized = normalized.clamp(min=0.0) * positive_mask.float()

        # 3. 安全约束
        crash_mask = crash | out                                 # [G, M]
        # 用非碰撞样本归一化 scale
        non_crash_vals = normalized[~crash_mask]
        if non_crash_vals.numel() > 0:
            scale = non_crash_vals.std(unbiased=False)
            if torch.isfinite(scale) and float(scale.item()) > 1e-6:
                normalized = normalized / scale
        adv = torch.where(crash_mask, torch.full_like(normalized, -1.0), normalized)

        # 4. 时间折扣：扩展到 step_num 维度
        adv = adv.unsqueeze(-1) * discount.unsqueeze(0).unsqueeze(0)  # [G, M, step_num]

        advantages[agent_id] = adv

    return advantages
```

### 改动 3：`compute_rl_loss()` 支持多 anchor

**文件**：`train/ma_grpo_trainer.py`
**位置**：`compute_rl_loss()` 方法（line 186-204）

```python
def compute_rl_loss(self, rollouts: dict, advantages: dict) -> Tensor:
    """所有 anchor 的 RL loss 求和。"""
    losses = []
    for agent_id in rollouts["agent_ids"]:
        # replay 返回 [G, M, step_num]
        replay_all = self.scheduler.replay_with_log_prob(
            self.model, rollouts["batch"], agent_id,
            rollouts[agent_id]["diffusion_chain"],
        )  # [G, M, step_num]

        advantage = advantages[agent_id]  # [G, M, step_num]

        # 所有 anchor、所有 group 一起算 loss
        per_token_loss = -torch.exp(replay_all - replay_all.detach()) * advantage
        mask_nz = per_token_loss != 0
        rl_loss = (per_token_loss * mask_nz).sum() / mask_nz.sum().clamp(min=1)
        losses.append(rl_loss)

    return torch.stack(losses).mean()
```

### 改动 4：`step_env_with_best()` 适配

**文件**：`train/ma_grpo_trainer.py`
**位置**：`step_env_with_best()` 方法（line 136-153）

```python
def step_env_with_best(self, rollouts: dict) -> dict:
    best_actions = {}
    for agent_id in rollouts["agent_ids"]:
        best_g = rollouts[agent_id]["best_g"]
        best_k = rollouts[agent_id]["best_k"]
        best_actions[agent_id] = (
            rollouts[agent_id]["trajectory_all"][best_g, best_k]
            .detach().cpu().numpy()
        )

    obs, _, terminated, truncated, _ = self.env.step(best_actions)
    self.env_step_counter += 1
    if (terminated.get("__all__", False) or truncated.get("__all__", False)
            or (self.max_env_steps_per_rollout > 0
                and self.env_step_counter >= self.max_env_steps_per_rollout)):
        obs = self.env.reset()
        self.env_step_counter = 0
    self.current_obs = obs
    return obs
```

### 改动 5：`replay_with_log_prob()` 返回 shape 确认

**文件**：`models/diffusion/diffusion_rl_scheduler.py`

当前 `replay_with_log_prob()` 已经返回 `[G, M, step_num]`（line 331）。需确认 `compute_rl_loss` 不再做 mode_index 选择，而是直接在 [G, M, step_num] 上操作。

**删除 `compute_rl_loss` 中的 mode_index 索引代码**：
```python
# 删除这些行：
# mode_index = rollouts[agent_id]["mode_index"]
# replay_log_prob = replay_all[torch.arange(...), mode_index]
```

## 显存影响

当前：G 条轨迹 × 1 anchor → G 次评估
改后：G 条轨迹 × M anchors → G×M 次评估

- 评估是 surrogate（无梯度），显存增加可忽略
- RL loss 从 [G, step_num] → [G, M, step_num]，梯度计算量增加 M 倍
- 当前 M=8, G=4 → 额外显存约 3-4GB
- 若超 16GB，降 group_size 到 2（G=2, M=8 → 16 条 vs 当前 G=4, M=1 → 4 条，仍然更多）

## 验收标准

**交付物**：修改后的 `ma_grpo_trainer.py`、`tests/acceptance/test_phase5v2_task2.py`

**验收指标**（11 项）：
1. `collect_group_samples()` 返回 `reward_per_anchor` shape = `[G, M]`
2. `compute_advantages()` 返回 shape = `[G, M, step_num]`
3. **【核心】per-anchor 独立归一化验证**：构造两个 anchor 的 reward 差异明显（如 anchor 0 全为正，anchor 1 全为负），调用 `compute_advantages()` 后，两个 anchor 各自的非碰撞 advantage 均值应独立接近 0，而非受另一 anchor 的 reward 拉偏。具体断言：
   ```python
   # anchor 0 和 anchor 1 归一化后均值各自接近 0
   assert abs(float(advantages[agent_id][:, 0, :].mean())) < 0.5
   assert abs(float(advantages[agent_id][:, 1, :].mean())) < 0.5
   # 两个 anchor 的 advantage 方差不应完全相同（独立计算的证明）
   # 若使用全局归一化，两列 std 会相同；per-anchor 时各列 std 独立
   ```
4. **【核心】跨 anchor 归一化检测**：当 anchor 0 的 reward 均值远高于 anchor 1 时（如 anchor 0 reward=[10,10], anchor 1 reward=[-1,-1]），若使用全局归一化 anchor 1 会得到负 advantage，但 per-anchor 归一化下 anchor 1 内部 std=0 advantage 应全为 0（或按 positive_mask 归零）。测试需显式验证此场景
5. crash 轨迹 advantage = -1.0（在所有 anchor 上均成立）
6. `compute_rl_loss()` 可 backward，梯度有限
7. `step_env_with_best()` 执行的是全局最优轨迹（best_g, best_k 指向 reward_per_anchor 中最大值位置，不一定是 anchor 0）
8. 当 anchor k=0 的所有 reward 均低于 anchor k=1 时，`best_k` 应为 1
9. 旧的 `mode_index` 索引逻辑已从 `compute_rl_loss()` 中删除（不再做 `replay_all[arange, mode_index]` 切片）
10. 连续 5 次 update，loss 有限非 NaN
11. `reward_per_anchor` 中每列（每个 anchor）的求值结果来自独立的 `_evaluate_trajectory_group` 调用，验证方式：mock `_evaluate_trajectory_group`，确认它被调用了 M 次而非 1 次

**验收命令**：
```bash
pytest tests/acceptance/test_phase5v2_task2.py -v
```

## 注意事项

1. **DiffusionDriveV2 有 20 个 anchor，我们有 8 个**（`ego_fut_mode`）。但原理相同
2. **不要做跨 anchor 归一化**——这是 DiffusionDriveV2 的核心设计决策，否则 mode collapse
3. `evaluate_trajectory_group` 目前接受 `[G, 8, 3]`，需要循环 M 次调用（每次传一个 anchor 的 G 条轨迹）
4. 向后兼容：保留 `reward`、`trajectory`、`mode_index` 等旧字段，以便旧测试不崩溃
