# Phase 5：编队闭环强化微调

> **前置依赖**：Phase 4（planner + 迁移权重）、Phase 1（env + reward info 字段 + 场景集成）
> **后续 Phase**：Phase 6（评估需要训练好的 checkpoint）

## 目标
通过 MA-GRPO 学习危险工况下的编队合作技能。

---

## 全局约定（本 Phase 所需）

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### 关键 import
```python
from envs.platoon_env import PlatoonEnv
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.weight_migration import migrate_single_to_platoon
from models.diffusion.diffusion_rl_scheduler import DiffusionRLScheduler
from evaluation.reward_terms import compute_step_reward
from train.ma_grpo_trainer import MultiAgentGRPOTrainer
```

### 参考代码
- log_prob + RL loss：`/home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py`
- RL agent：同目录 `diffusiondrivev2_rl_agent.py`

### 设计决策
- **MA-GRPO + CTDE**：每车独立采样 group_size 条轨迹，advantage 采用标准化公式（见下方 Advantage 计算）
- **log_prob**：每个 DDIM 去噪 step 实时计算 Gaussian log_prob 并累积；训练时重放去噪链（chain replay）对 log_prob 求梯度。**不是**采样结束后一次性计算。参考 `DDIMScheduler_with_logprob`（参考代码 lines 540-676）
- **log_prob 公式**：`log_prob = sum_over(time,coord)[ -((x_t - μ)² / (2σ²)) - log(σ) - log(√2π) ]`，其中 μ = DDIM 均值预测，σ = `clip(η·√variance, min=0.1)`。`prev_sample.detach()` 保证梯度只流过 μ 不流过采样噪声
- **两阶段前向**：① 采样阶段（`forward_train_rl`）：`with torch.no_grad()` DDIM 去噪得到轨迹 + 中间状态链 `all_diffusion_output` + advantage；② 训练阶段（`get_rlloss`）：重放去噪链，用缓存的中间状态计算新 log_prob，与 advantage 相乘得到 RL loss
- **Loss 混合**：`loss = RL_loss + il_weight × IL_loss`。当存在正 advantage 样本时 `il_weight=0.1`，无正样本时 `il_weight=1.0`（IL 兜底防止策略崩溃）
- **Reference policy**：frozen copy，用于 KL 监控（`compute_kl`）。KL 作为日志指标和早停依据，不直接加入 loss（参考代码验证此方案更稳定）。若 KL > 阈值（默认 5.0）触发学习率衰减
- **Reward**：轨迹级 reward。每条轨迹在 env 中执行后，将 8 步 dense reward 求和为单个标量作为该轨迹的 reward（用于 advantage 计算）
- **显存**：3 车 × group_size=4 = 12 次前向 ≈ 12GB，需 gradient checkpointing

### 奖励公式（第一版）
```
r_t = 1.0·r_progress + 0.5·r_formation + 0.3·r_safety + r_collision + r_road + 0.1·r_comfort
```
| 项 | 公式 | 默认权重/常数 |
|----|------|-------------|
| `r_progress` | `Δs / Δs_max` | w=1.0 |
| `r_formation` | `-(|Δx-Δx_target|+|Δy-Δy_target|)/d_norm` | w=0.5 |
| `r_safety` | `-max(0, d_safe-d_min)/d_safe` | w=0.3 |
| `r_collision` | `-10.0`（碰撞时） | C=10.0 |
| `r_road` | `-5.0`（出路时） | C=5.0 |
| `r_comfort` | `-(0.5·|jerk|+0.5·|Δsteering|)` | w=0.1 |

全局 bonus：全员到达 +20.0，队形恢复 +5.0。

**轨迹级 reward 聚合**：diffusion 一次生成 8 步轨迹，每步在 env 中执行后得到 `r_t`，单条轨迹的 reward = `sum(r_t for t in 1..8)`。此标量用于 group 内 advantage 计算。

**info 字段依赖（Phase 1.6 提供）**：progress, jerk, delta_steering, formation_error, min_gap, crash, arrive_dest, out_of_road, speed_km_h

### Advantage 计算（对齐参考代码）
```python
# 1. 标准化：mean + std 归一化
mean_r = reward_group.mean(dim=group_dim)
std_r  = reward_group.std(dim=group_dim)
advantages = (reward_group - mean_r) / (std_r + 1e-4)

# 2. 正样本过滤：只保留优于 baseline 的轨迹
baseline = mean_r  # 或使用 best-of-group / GT reward
mask_positive = (reward_group > baseline)
advantages = advantages.clamp(min=0) * mask_positive.float()

# 3. 安全约束：碰撞/出路的轨迹强制 advantage = -1.0
crash_mask = (crash_flags != 0) | (out_of_road_flags != 0)
advantages = torch.where(crash_mask, torch.full_like(advantages, -1.0), advantages)

# 4. 时间折扣：对 diffusion step 维度加权（早期 step 权重小）
discount = torch.tensor([0.8 ** (step_num - i - 1) for i in range(step_num)])
advantages = advantages.unsqueeze(-1) * discount  # [B, G, step_num]
```

### 奖励管理规则
- 版本号（reward_v1, v2...），每次修改 config 切换
- 每次调整前后 ≥ 3 seed 对照
- 不得无日志同时改多项

### 硬件约束
- RTX 4080 16GB
- 推理 ≤ 100ms（3 车总计）
- MA-GRPO group_size=4 ≈ 12GB（需 gradient checkpointing）
- 超 16GB 时先降 group_size/batch_size，再考虑 gradient accumulation

---

## 核心接口签名

```python
# evaluation/reward_terms.py
def compute_step_reward(info: dict, config: dict) -> float:
    """按奖励公式计算 r_t。config 含 w_prog, w_form, w_safe, w_coll, w_road, w_comf 等。
    info 来自 PlatoonEnv.step() 的 per-agent info dict。"""

def compute_trajectory_reward(step_infos: list[dict], config: dict) -> float:
    """将 8 步 dense reward 聚合为单条轨迹的标量 reward。
    step_infos = [info_t1, info_t2, ..., info_t8]
    return sum(compute_step_reward(info, config) for info in step_infos)"""

# models/diffusion/diffusion_rl_scheduler.py
class DDIMSchedulerWithLogProb:
    """继承/改写 DDIMScheduler，在每个去噪 step 返回 Gaussian log_prob。
    参考实现：参考代码 lines 540-676（DDIMScheduler_with_logprob）。

    关键：step() 返回 (prev_sample, log_prob, prev_sample_mean)
    - prev_sample: 去噪后的样本 [B, G*N, 8, 2]
    - log_prob: Gaussian log_prob [B, G*N]，sum over (time, coord)
    - prev_sample_mean: DDIM 均值预测（用于 log_prob 计算）

    log_prob 公式：
        log_prob = sum[ -((x.detach() - μ)² / (2σ²)) - log(σ) - log(√2π) ]
        σ = clip(η·√variance, min=0.1)
        x.detach() 保证梯度只流过 μ
    """
    def step(self, model_output, timestep, sample, eta=0.0, prev_sample=None) -> tuple[Tensor, Tensor, Tensor]: ...

class DiffusionRLScheduler:
    """封装 DDIMSchedulerWithLogProb，提供两阶段前向接口。"""
    def __init__(self, ddim_config: dict):
        self.scheduler = DDIMSchedulerWithLogProb(...)

    def sample_with_log_prob(self, model, condition, num_groups: int = 4) -> dict:
        """采样阶段（no_grad）：DDIM 去噪 + 逐 step log_prob。
        返回 {
            "trajectory": Tensor [B, G*N, 8, 3],
            "log_prob": Tensor [B, G*N, step_num],
            "diffusion_chain": list[Tensor],  # 中间状态，用于 chain replay
        }"""

    def replay_with_log_prob(self, model, condition, diffusion_chain: list[Tensor]) -> Tensor:
        """训练阶段（有梯度）：重放去噪链，用缓存的中间状态重新计算 log_prob。
        返回 log_prob [B, G*N, step_num]（可 backward）"""

    def compute_kl(self, log_prob: Tensor, ref_log_prob: Tensor) -> Tensor:
        """KL 散度估计，用于监控和早停（不加入 loss）。"""

# train/ma_grpo_trainer.py
class MultiAgentGRPOTrainer:
    def __init__(self, model: PlatoonDiffusionPlanner, ref_model: PlatoonDiffusionPlanner, env: PlatoonEnv, config: dict):
        """ref_model 为 frozen copy，用于 KL 监控。"""
    def collect_group_samples(self, group_size: int = 4) -> dict:
        """采样阶段。返回 {
            agent_id: {
                trajectory: [G, 8, 3],
                log_prob: [G, step_num],
                reward: [G],  # 轨迹级标量 reward
                crash_flag: [G],  # 安全约束标记
                diffusion_chain: list[Tensor],
            }
        }"""
    def compute_advantages(self, rollouts: dict) -> dict:
        """标准化 advantage + 正样本过滤 + 安全约束 + 时间折扣。"""
    def compute_rl_loss(self, rollouts: dict, advantages: dict) -> Tensor:
        """chain replay 获取新 log_prob → importance ratio × advantage。
        per_token_loss = -exp(new_log_prob - new_log_prob.detach()) × advantage"""
    def compute_il_loss(self, rollouts: dict) -> Tensor:
        """IL 兜底损失：L1 regression loss against 当前最优轨迹。"""
    def update(self, rollouts: dict) -> dict:
        """loss = RL_loss + il_weight × IL_loss。
        il_weight = 0.1（有正样本时）| 1.0（无正样本时）。
        返回 {loss, kl, mean_reward, il_weight, grad_norm}"""
    def train(self, total_steps: int): ...

# train/train_platoon_rl.py — 训练入口
#   --mode toy-single: 单车 GRPO toy 验证
#   --mode platoon: 3 车编队训练
#   --config: yaml 配置文件
#   --steps: 训练步数
```

---

## 子任务与验收指标

### ☐ 5.1 创建奖励函数
- [ ] 待完成
- **交付物**：`evaluation/reward_terms.py`、`tests/acceptance/test_phase5_task1.py`
- **验收指标**（7 项）：
  1. `compute_step_reward(info, config)` 存在，返回 float
  2. `crash=True` 时 reward ≤ -10.0
  3. 无事件时 reward > 0
  4. `formation_error=1.0` 时 reward > `formation_error=5.0` 时 reward
  5. 所有权重可通过 config 覆盖
  6. 缺失 key 不抛异常（默认 0）
  7. 返回值为 float
- **验收命令**：`pytest tests/acceptance/test_phase5_task1.py -v`（7 PASSED）

### ☐ 5.2 创建 DiffusionRLScheduler（含 DDIMSchedulerWithLogProb）
- [ ] 待完成
- **交付物**：`models/diffusion/diffusion_rl_scheduler.py`、`models/diffusion/__init__.py`、`tests/acceptance/test_phase5_task2.py`
- **关键实现**：`DDIMSchedulerWithLogProb` 基于参考代码 `DDIMScheduler_with_logprob`（lines 540-676）实现。每个 DDIM step 返回 `(prev_sample, log_prob, prev_sample_mean)`。log_prob 使用 Gaussian 公式：`sum[ -((x.detach()-μ)² / (2σ²)) - log(σ) - log(√2π) ]`，σ = `clip(η·√var, min=0.1)`。
- **验收指标**（9 项）：
  1. `DDIMSchedulerWithLogProb` 类存在
  2. `DiffusionRLScheduler` 类存在
  3. `sample_with_log_prob()` 返回 dict 含 trajectory、log_prob、diffusion_chain
  4. trajectory shape 含 `[8, 3]`（或 `[8, 2]` + bezier yaw 补全）
  5. log_prob shape = `[B, G*N, step_num]`，每个值有限
  6. `replay_with_log_prob()` 用缓存 chain 重新计算 log_prob，结果可 backward
  7. `compute_kl()` 返回非负标量
  8. 多次 `sample_with_log_prob()` 输出不同（随机性）
  9. `replay_with_log_prob()` 的 log_prob 与采样阶段的 log_prob 数值一致（`atol=1e-4`）
- **验收命令**：`pytest tests/acceptance/test_phase5_task2.py -v`（9 PASSED）

### ☐ 5.3 创建 MultiAgentGRPOTrainer
- [ ] 待完成
- **交付物**：`train/ma_grpo_trainer.py`、`train/__init__.py`、`tests/acceptance/test_phase5_task3.py`
- **关键实现**：
  - `__init__` 接受 `model` + `ref_model`（frozen copy）+ `env` + `config`
  - `compute_advantages` 实现标准化公式：`(r - mean) / (std + 1e-4)` + 正样本过滤 + 安全约束（crash→-1.0）+ 时间折扣（`0.8^(step_num-i-1)`）
  - `update` 使用两阶段前向：① no_grad 采样得到 chain + advantage，② chain replay 得到新 log_prob，loss = RL_loss + il_weight × IL_loss
  - `il_weight` 自适应：有正 advantage → 0.1，无正 advantage → 1.0
- **验收指标**（10 项）：
  1. 类存在，接受 model + ref_model + env + config
  2. `collect_group_samples(4)` 返回正确结构
  3. 每 agent 含 trajectory `[G,8,3]`、log_prob `[G,step_num]`、reward `[G]`、diffusion_chain
  4. `compute_advantages` 输出经过 std 归一化（|advantage.std() - 1| < 0.5 或全零）
  5. 正样本过滤：crash 轨迹的 advantage = -1.0
  6. `update()` 返回 {loss, kl, mean_reward, il_weight, grad_norm}
  7. 连续 2 次 update loss 变化
  8. KL ∈ [0, 10)
  9. 无正样本时 il_weight == 1.0
  10. loss 包含 RL 和 IL 两部分，均为有限值
- **验收命令**：`pytest tests/acceptance/test_phase5_task3.py -v`（10 PASSED）

### ☐ 5.4 创建训练入口脚本
- [ ] 待完成
- **交付物**：`train/train_platoon_rl.py`、`configs/train/platoon_grpo.yaml`、`tests/acceptance/test_phase5_task4.py`
- **验收指标**（6 项）：
  1. 接受 `--config` 和 `--mode` 参数
  2. `--mode toy-single --steps 5` 可运行
  3. `--mode platoon --steps 5` 可运行
  4. 自动创建 tensorboard 日志
  5. yaml 含：group_size, lr, kl_threshold, il_weight_default, il_weight_no_positive, reward_config, total_steps, checkpoint_interval, ddim_steps, ddim_eta, advantage_discount_gamma
  6. 5 步 < 5 分钟
- **验收命令**：
  ```bash
  python train/train_platoon_rl.py --mode toy-single --steps 5 --render 0
  pytest tests/acceptance/test_phase5_task4.py -v
  ```

### ☐ 5.4b 全链路集成冒烟测试（env→planner→reward→trainer）
- [ ] 待完成
- **背景**：验证 Phase 1、4、5 代码**真正连通**，防止各模块各自通过但无法集成。
- **交付物**：`tests/acceptance/test_phase5_integration.py`
- **验收指标**（5 项）：
  1. 完整链路：`PlatoonEnv(multimodal).reset()` → planner.forward() → env.step(traj) → compute_step_reward(info) → reward 为有限 float
  2. 连续 3 步不报错
  3. `collect_group_samples(2)` 在真实 env 上可调用
  4. `compute_advantages()` + `update()` 可运行，loss 有限
  5. 无硬编码 shape 转换
- **验收命令**：`pytest tests/acceptance/test_phase5_integration.py -v`（5 PASSED）

### ☐ 5.5 单车 GRPO toy 实验
- [ ] 待完成
- **交付物**：日志、`tests/acceptance/test_phase5_task5.py`
- **验收指标**（5 项）：
  1. 100 步 loss 无发散
  2. 100 步后 mean_reward > 第 1 步
  3. KL < 1.0
  4. grad_norm < 100
  5. 显存 < 10GB
- **验收命令**：
  ```bash
  python train/train_platoon_rl.py --mode toy-single --steps 100 --render 0
  pytest tests/acceptance/test_phase5_task5.py -v
  ```

### ☐ 5.6 3 车编队 GRPO 训练启动
- [ ] 待完成
- **交付物**：日志、checkpoint、`tests/acceptance/test_phase5_task6.py`
- **验收指标**（7 项）：
  1. 稳定运行 ≥ 500 步
  2. 无 NaN loss、无 OOM
  3. 显存 < 15GB
  4. formation_error 有下降趋势（后 50 步 < 前 50 步）
  5. collision_rate 在减少
  6. 每 100 步 checkpoint 到 `checkpoints/platoon_rl/`
  7. tensorboard 含 8 曲线：loss, rl_loss, il_loss, kl, mean_reward, formation_error, collision_rate, grad_norm
- **验收命令**：
  ```bash
  python train/train_platoon_rl.py --config configs/train/platoon_grpo.yaml --steps 500 --render 0
  ls checkpoints/platoon_rl/step_*.ckpt | wc -l  # ≥ 5
  pytest tests/acceptance/test_phase5_task6.py -v
  ```

---

## 跨 Phase 数据流

```
Phase 1 PlatoonEnv(multimodal) ──→ trainer.env（环境实例）
Phase 1 info dict（含 progress/jerk/...）──→ compute_step_reward()
Phase 1 hazard_scenarios ──→ curriculum 场景切换

Phase 4 PlatoonDiffusionPlanner ──→ trainer.model
Phase 4 migrate_single_to_platoon() ──→ 训练初始权重

Phase 5 checkpoints/platoon_rl/ ──→ Phase 6 评估
Phase 5 训练日志 ──→ Phase 7 论文曲线
```

---

## 参考代码关键实现索引

> 实现本 Phase 时**必须参考**以下代码段，不可仅凭接口签名自行实现。

### DDIMSchedulerWithLogProb（最关键）
- **文件**：`reference_libs/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py`
- **类名**：`DDIMScheduler_with_logprob`（lines 540-676）
- **核心逻辑**（`step()` 方法，lines 560-676）：
  1. 标准 DDIM 去噪得到 `prev_sample_mean`（lines 596-630）
  2. eta > 0 时加乘性噪声和加性噪声得到 `prev_sample`（lines 638-666）
  3. 计算 Gaussian log_prob：`-((x.detach() - μ)² / (2σ²)) - log(σ) - log(√2π)`，σ = `clip(std_dev_t, min=0.1)`（lines 668-675）
  4. `prev_sample.detach()` 保证梯度只流过 `prev_sample_mean`（即模型预测）
  5. `log_prob.sum(dim=(-2, -1))` 对 time 和 coord 维度求和

### 两阶段前向（采样 + 训练）
- **采样阶段**：`forward_train_rl()`（lines 800-939）
  - `with torch.no_grad()` 进行 DDIM 去噪（**注意**：在参考代码中这是通过 V2TransfuserModel.forward 的 `old_pred` 机制实现的，line 258-259）
  - 收集 `all_diffusion_output`（中间状态链）和 `all_log_probs`
  - 计算 advantage（lines 886-932）
- **训练阶段**：`get_rlloss()`（lines 1028-1120）
  - 从 `old_pred['all_diffusion_output']` 取出缓存的去噪链（line 1030）
  - 拆分为 `chains`（当前状态）和 `chains_prev`（下一状态）（lines 1033-1034）
  - 重放每个 step：用当前模型重新预测，传入 `prev_sample=chains_prev[...,i]` 以复用采样时的 sample（line 1090）
  - 计算新的 `log_prob`（有梯度）
  - **RL loss**：`per_token_loss = -exp(log_prob - log_prob.detach()) × advantage`（line 1096）
  - **IL loss**：L1 regression loss against GT trajectory（lines 1107-1112）
  - **混合**：`loss = RL_loss + il_weight × IL_loss`，il_weight = 0.1（有正样本）/ 1.0（无正样本）（lines 1115-1119）

### Advantage 计算
- **标准化**：`(reward - mean) / (std + 1e-4)`（line 889）
- **正样本过滤**：`clamp(min=0) * (reward > GT_reward)`（lines 892-893）
- **安全约束**：collision / drivable_area 不满足时 advantage = -1.0（lines 900-902）
- **时间折扣**：`advantage × 0.8^(step_num-i-1)`（lines 926-932）

### 注意事项
1. 参考代码中 `ego_fut_mode=20`（20 个 anchor mode），我们的 MetaDrive planner 可能不同，按实际 `config.ego_fut_mode` 适配
2. 参考代码使用 PDM Scorer 离线评分，我们改用 env dense reward 聚合，但 advantage 计算流程保持一致
3. 参考代码的 `num_groups` 是跨 anchor mode 分组，我们是 per-vehicle group_size 条轨迹，概念不同但数学等价
4. `truncated noise`：参考代码从 timestep=8（非 1000）开始去噪（line 819），这是 DiffusionDrive 的关键加速技巧，应保留
