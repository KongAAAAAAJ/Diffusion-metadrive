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
- **MA-GRPO + CTDE**：每车独立采样 group_size 条轨迹（step-level），advantage = r_i - mean(r_group)
- **log_prob**：DDIM 采样完成后基于 mode logits + regression 分布计算，不对中间 step 求梯度
- **Reference policy**：frozen copy，KL_weight 初始 0.01
- **Reward**：每 step dense reward（见下方奖励公式）
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

**info 字段依赖（Phase 1.6 提供）**：progress, jerk, delta_steering, formation_error, min_gap, crash, arrive_dest, out_of_road, speed_km_h

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

# models/diffusion/diffusion_rl_scheduler.py
class DiffusionRLScheduler:
    def sample_with_log_prob(self, model, condition) -> tuple[Tensor, Tensor]:
        """DDIM 采样 + log_prob。返回 (trajectory, log_prob)"""
    def compute_kl(self, log_prob: Tensor, ref_log_prob: Tensor) -> Tensor: ...

# train/ma_grpo_trainer.py
class MultiAgentGRPOTrainer:
    def __init__(self, model: PlatoonDiffusionPlanner, env: PlatoonEnv, config: dict): ...
    def collect_group_samples(self, group_size: int = 4) -> list[dict]:
        """step-level 采样。返回 [{agent_id: {traj:[G,8,3], log_prob:[G], reward:[G]}}]"""
    def compute_advantages(self, rollouts) -> list[dict]: ...
    def update(self, rollouts) -> dict:  # {loss, kl, mean_reward}
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

### ☐ 5.2 创建 DiffusionRLScheduler
- [ ] 待完成
- **交付物**：`models/diffusion/diffusion_rl_scheduler.py`、`models/diffusion/__init__.py`、`tests/acceptance/test_phase5_task2.py`
- **验收指标**（6 项）：
  1. 类存在
  2. `sample_with_log_prob()` 返回 (trajectory, log_prob)，trajectory 含 `[8,3]`
  3. log_prob 有限值
  4. `compute_kl()` 返回非负标量
  5. log_prob 支持 backward
  6. 多次采样输出不同
- **验收命令**：`pytest tests/acceptance/test_phase5_task2.py -v`（6 PASSED）

### ☐ 5.3 创建 MultiAgentGRPOTrainer
- [ ] 待完成
- **交付物**：`train/ma_grpo_trainer.py`、`train/__init__.py`、`tests/acceptance/test_phase5_task3.py`
- **验收指标**（7 项）：
  1. 类存在
  2. `collect_group_samples(4)` 返回正确结构
  3. 每 agent 含 traj `[G,8,3]`、log_prob `[G]`、reward `[G]`
  4. advantage 均值 ≈ 0（|mean| < 0.1）
  5. `update()` 返回 {loss, kl, mean_reward}
  6. 连续 2 次 update loss 变化
  7. KL ∈ [0, 10)
- **验收命令**：`pytest tests/acceptance/test_phase5_task3.py -v`（7 PASSED）

### ☐ 5.4 创建训练入口脚本
- [ ] 待完成
- **交付物**：`train/train_platoon_rl.py`、`configs/train/platoon_grpo.yaml`、`tests/acceptance/test_phase5_task4.py`
- **验收指标**（6 项）：
  1. 接受 `--config` 和 `--mode` 参数
  2. `--mode toy-single --steps 5` 可运行
  3. `--mode platoon --steps 5` 可运行
  4. 自动创建 tensorboard 日志
  5. yaml 含：group_size, lr, kl_weight, reward_config, total_steps, checkpoint_interval
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
  7. tensorboard 含 6 曲线：loss, kl, mean_reward, formation_error, collision_rate, grad_norm
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
