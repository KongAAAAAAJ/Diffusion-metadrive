# 5v2.6 分层 Advantage 合并

> **依赖**：5v2.2（intra-anchor advantage）、5v2.5（joint group + team reward）
> **改动文件**：`train/ma_grpo_trainer.py`

## 背景

将单车局部 advantage 和多车联合 advantage 合并为最终 advantage，用于 RL loss 计算。

## 数学公式

```
A_final[i, k, g, t] = λ_local × γ^(T_d - t) × A_local[i, k, g]
                     + λ_team  × γ^(T_d - t) × A_team[i, m]

其中：
- i: 车辆 index
- k: anchor index
- g: group sample index
- t: diffusion step
- m: 该 (i, k, g) 所属的联合组 index
- λ_local = 0.7, λ_team = 0.3（可配置）
- γ = 0.8（时间折扣）
```

## 核心问题：联合组与单车样本的映射

每个联合组 m 由 N 辆车各选一条轨迹组成：`members_m = {agent_i: (g_i, k_i)}`。

team_advantage 的分配规则：
- 联合组 m 的 team_reward 经归一化后得到 A_team_m
- 对于联合组 m 中的车辆 i，其轨迹 (g_i, k_i) 获得 A_team_m
- 不在任何联合组中的 (g, k) 组合：A_team = 0（不受联合信号影响）

## 具体代码改动

### 改动 1：新增 `compute_team_advantages()` 方法

**文件**：`train/ma_grpo_trainer.py`

```python
def compute_team_advantages(
    self,
    rollouts: dict,
    joint_groups: List[Dict[str, Tuple[int, int]]],
    team_rewards: List[float],
) -> dict:
    """计算联合组的 team advantage。

    Parameters
    ----------
    rollouts : dict
        collect_group_samples 的输出。
    joint_groups : list of dict[agent_id -> (g_idx, k_idx)]
        联合组成员描述。
    team_rewards : list of float
        每个联合组的 team reward。

    Returns
    -------
    team_advantages : dict[agent_id -> Tensor [G, M, step_num]]
        与 local advantage shape 一致，未参与联合组的位置为 0。
    """
    agent_ids = rollouts["agent_ids"]
    G = rollouts[agent_ids[0]]["reward_per_anchor"].shape[0]
    M_anchors = rollouts[agent_ids[0]]["reward_per_anchor"].shape[1]
    step_num = self.scheduler.step_num

    # 1. 归一化 team_rewards
    tr = torch.tensor(team_rewards, dtype=torch.float32, device=self.device)
    if tr.numel() > 1:
        mean_tr = tr.mean()
        std_tr = tr.std(unbiased=False)
        normalized_tr = (tr - mean_tr) / (std_tr + 1e-4)
    else:
        normalized_tr = torch.zeros_like(tr)

    # 2. 正样本过滤 + hard failure
    normalized_tr = normalized_tr.clamp(min=0.0)
    # 如果联合组中任何车 crash，team advantage = -1
    for m_idx, group in enumerate(joint_groups):
        any_crash = False
        for agent_id, (g_idx, k_idx) in group.items():
            if rollouts[agent_id]["crash_per_anchor"][g_idx, k_idx]:
                any_crash = True
                break
        if any_crash:
            normalized_tr[m_idx] = -1.0

    # 3. 分配到 [G, M, step_num] tensor
    discount = torch.tensor(
        [self.advantage_gamma ** (step_num - i - 1) for i in range(step_num)],
        dtype=torch.float32, device=self.device,
    )

    team_advantages = {}
    for agent_id in agent_ids:
        adv = torch.zeros(G, M_anchors, step_num, device=self.device)
        for m_idx, group in enumerate(joint_groups):
            if agent_id in group:
                g_idx, k_idx = group[agent_id]
                adv[g_idx, k_idx, :] = normalized_tr[m_idx] * discount
        team_advantages[agent_id] = adv

    return team_advantages
```

### 改动 2：新增 `compute_combined_advantages()` 方法

**文件**：`train/ma_grpo_trainer.py`

```python
def compute_combined_advantages(
    self,
    local_advantages: dict,
    team_advantages: dict,
) -> dict:
    """合并局部和联合 advantage。

    A_final = λ_local × A_local + λ_team × A_team

    Parameters
    ----------
    local_advantages : dict[agent_id -> Tensor [G, M, step_num]]
    team_advantages : dict[agent_id -> Tensor [G, M, step_num]]

    Returns
    -------
    combined : dict[agent_id -> Tensor [G, M, step_num]]
    """
    lambda_local = float(self.config.get("lambda_local", 0.7))
    lambda_team = float(self.config.get("lambda_team", 0.3))

    combined = {}
    for agent_id in local_advantages:
        local = local_advantages[agent_id]
        team = team_advantages.get(agent_id, torch.zeros_like(local))
        combined[agent_id] = lambda_local * local + lambda_team * team

    return combined
```

### 改动 3：更新 `update()` 主流程

**文件**：`train/ma_grpo_trainer.py`
**位置**：`update()` 方法

```python
def update(self, rollouts: dict, joint_groups=None, team_rewards=None) -> dict:
    # 1. 局部 advantage（intra-anchor）
    local_advantages = self.compute_advantages(rollouts)

    # 2. 联合 advantage（如果有）
    if joint_groups is not None and team_rewards is not None:
        team_advantages = self.compute_team_advantages(
            rollouts, joint_groups, team_rewards
        )
        advantages = self.compute_combined_advantages(local_advantages, team_advantages)
    else:
        advantages = local_advantages

    # 3. 计算 loss
    self.optimizer.zero_grad(set_to_none=True)
    rl_loss = self.compute_rl_loss(rollouts, advantages)
    ref_reg_loss = self.compute_ref_reg_loss(rollouts)  # 5v2.1
    beta_reg = self._get_beta_reg()
    loss = rl_loss + beta_reg * ref_reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
    # ... (grad_norm 计算、optimizer.step、KL 监控同前)
```

## 训练循环改动

**文件**：`train/ma_grpo_trainer.py`
**方法**：`train()` 或外部训练循环

```python
def train(self, total_steps: int):
    metrics = []
    group_size = int(self.config.get("group_size", 4))
    top_k = int(self.config.get("joint_top_k", 2))
    num_joint_groups = int(self.config.get("num_joint_groups", 8))
    use_closedloop = bool(self.config.get("use_closedloop", False))

    obs = self.env.reset()
    self.env_step_counter = 0
    self.current_obs = obs

    for _ in range(int(total_steps)):
        rollouts = self.collect_group_samples(group_size=group_size, obs=obs)

        joint_groups = None
        team_rewards = None

        if use_closedloop and hasattr(self.env, 'get_state'):
            # 构建联合组
            from train.joint_group import (
                select_top_k_candidates, build_joint_groups,
                extract_joint_trajectories,
            )
            from train.closedloop_executor import ClosedLoopExecutor
            from evaluation.reward_terms import compute_team_reward

            per_agent_candidates = {}
            for agent_id in rollouts["agent_ids"]:
                candidates = select_top_k_candidates(
                    rollouts[agent_id]["reward_per_anchor"],
                    rollouts[agent_id]["crash_per_anchor"],
                    top_k=top_k,
                )
                per_agent_candidates[agent_id] = candidates

            joint_groups = build_joint_groups(
                per_agent_candidates, num_groups=num_joint_groups
            )

            # 闭环执行
            executor = ClosedLoopExecutor(self.env, self.reward_config)
            joint_trajectories = [
                extract_joint_trajectories(jg, rollouts)
                for jg in joint_groups
            ]
            exec_results = executor.execute_joint_groups(joint_trajectories)

            # 计算 team reward
            team_rewards = [
                compute_team_reward(r["step_infos"], self.reward_config)
                for r in exec_results
            ]

        m = self.update(rollouts, joint_groups, team_rewards)
        metrics.append(m)
        obs = self.step_env_with_best(rollouts)

    return metrics
```

## 配置新增

```yaml
# 联合组配置
lambda_local: 0.7           # 局部 advantage 权重
lambda_team: 0.3            # 联合 advantage 权重
joint_top_k: 2              # 每车选 top-K 候选
num_joint_groups: 8          # 联合组数量
use_closedloop: false        # 是否启用闭环执行（渐进式开启）

# team reward 权重
w_team_formation: 0.5
w_team_safety: 1.0
w_team_efficiency: 0.3
w_team_collision: 10.0
```

## 验收标准

**交付物**：修改后的 `ma_grpo_trainer.py`、`tests/acceptance/test_phase5v2_task6.py`

**验收指标**（7 项）：
1. `compute_team_advantages()` 返回 shape = `[G, M, step_num]`（与 local 一致）
2. 未参与联合组的 (g, k) 位置 team advantage = 0
3. 碰撞联合组的 team advantage = -1.0
4. `compute_combined_advantages()` 返回 = λ_local × local + λ_team × team
5. `update()` 支持 `joint_groups=None`（退化为纯局部 advantage）
6. **【修复要求】`use_closedloop=True` 时训练循环包含真实闭环执行**：现有测试（`test_train_with_closedloop_executes_joint_pipeline`）用 `monkeypatch` 将 `ClosedLoopExecutor` 和 `select_top_k_candidates` 全部替换为假对象，只验证"函数被调用"，不验证真实逻辑，属于无效测试。需替换为以下方案：使用真实 `ToyEnv`（已有 `get_state`/`set_state` stub）和 `ToyPlanner` 构造场景，在不 mock 核心函数的前提下运行 `train(total_steps=1)`，验证：(a) `exec_results` 中每组均含 `step_infos` 和 `crash_flags` 字段；(b) `team_rewards` 是长度 = `num_joint_groups` 的 float list；(c) `update()` 接收到非 None 的 `joint_groups` 和 `team_rewards`。允许对无关的底层 IO（如 checkpoint 写磁盘）使用 mock，但 executor、joint_group 构建、team_reward 计算必须真实执行
7. **【修复要求】连续 3 次 update（含联合组），loss 有限非 NaN**：现有测试只循环了 2 次，需改为循环 3 次，每次均断言 `torch.isfinite(torch.tensor(metrics["loss"]))` 且 `metrics["loss"]` 不为 NaN

**验收命令**：
```bash
pytest tests/acceptance/test_phase5v2_task6.py -v
```

## 注意事项

1. **λ_team 调度**：训练初期建议 λ_team=0，只用局部 advantage，待局部收敛后再开启 team advantage。可以用一个简单的阶梯调度
2. **team advantage 稀疏性**：[G, M] 中只有少数 (g, k) 参与了联合组，大部分为 0。RL loss 中 mask_nz 会自动跳过这些位置
3. **credit assignment 简化**：第一版所有车共享同一 team advantage。后续可以根据"谁导致了 crash"差异化
