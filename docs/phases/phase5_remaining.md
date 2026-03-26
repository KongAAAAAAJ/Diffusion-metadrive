# Phase 5 剩余任务：真实环境 RL 训练链路

> **背景**：Phase 5 的 RL 数学链路（DiffusionRLScheduler、MultiAgentGRPOTrainer 核心算法）已正确实现并通过测试。但训练脚本 `train/train_platoon_rl.py` 目前使用 ToyEnv + ToyPlanner 运行，从未连接真实 `PlatoonEnv` + `PlatoonDiffusionPlanner`。Tasks 5.5/5.6 产出的指标来自 ToyEnv，不代表真实训练能力。
>
> **本文件定义的任务**：将 RL 训练链路连接到真实仿真环境，使其能在 PlatoonEnv 上用 PlatoonDiffusionPlanner 进行闭环 RL 微调。

---

## 全局约定

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
# 测试用 Python（含 torch + diffusers + metadrive）:
#   /home/kong/anaconda3/envs/navsim/bin/python   （torch + diffusers，无 metadrive）
#   /home/kong/anaconda3/envs/meta_drive/bin/python （metadrive 环境，用于集成测试）
```

### 关键路径

| 用途 | 路径 |
|------|------|
| 单车 checkpoint | `/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt` |
| plan anchor | `metadrive/exp_dataset/metadrive_anchors_ppo.npy` |
| transfuser config | `from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config` |

### 禁止事项
1. **不得修改** `models/diffusion/diffusion_rl_scheduler.py`——该文件已经过严格审查并修复
2. **不得修改** `evaluation/reward_terms.py`——已验收通过
3. **不得修改** `models/platoon/platoon_diffusion_planner.py` 的 `extract_rl_context` / `predict_denoised_traj` 方法
4. **不得改变** `MultiAgentGRPOTrainer` 的核心算法（`compute_advantages`、`compute_rl_loss`、`compute_il_loss` 的数学逻辑）
5. **不得用 mock 或 stub 来通过验收测试**——所有集成测试必须使用真实 PlatoonEnv + 真实 PlatoonDiffusionPlanner

---

## 子任务 5.7：PlatoonEnv 添加 `evaluate_trajectory_group` 方法

### 目标
在 `PlatoonEnv` 上添加轨迹组评估能力，使 `MultiAgentGRPOTrainer.collect_group_samples` 能获得真实仿真奖励。

### 设计说明

GRPO 采样阶段需要对同一初始观测采样 G 条轨迹，并分别获得奖励。由于 MetaDrive 环境不支持状态快照/恢复（fork），我们采用**代理奖励 (surrogate reward)** 策略：

1. **前 8 步的 dense reward**：对每条轨迹，利用当前 env 状态 + 轨迹几何特征估算 8 步 reward（不实际执行 env.step）
2. **碰撞/出路检测**：利用轨迹坐标与当前已知道路边界、其他车辆位置进行几何碰撞检测
3. **执行最优轨迹**：评估完 G 条轨迹后，选择 reward 最高的一条实际执行 env.step（用于推进环境状态）

这样做的原因：
- 参考代码 DiffusionDriveV2 也使用离线评分（PDMScorer），不是闭环执行所有 G 条轨迹
- MetaDrive env 不支持 state snapshot/restore
- 代理奖励 + 选择执行最优轨迹，已足够提供 GRPO 所需的组内相对比较信号

### 交付物
- 修改 `envs/platoon_env.py`：添加 `evaluate_trajectory_group` 方法
- 新建 `tests/acceptance/test_phase5_task7.py`

### 接口签名

```python
# envs/platoon_env.py — PlatoonEnv 新方法
def evaluate_trajectory_group(
    self,
    agent_id: str,
    trajectories: np.ndarray,  # [G, 8, 3] (x, y, yaw) 物理坐标
) -> dict:
    """评估 G 条候选轨迹的奖励，不实际推进环境状态。

    利用当前环境状态（各车位置、道路几何）+ 轨迹坐标计算代理奖励：
    - progress: 轨迹末端 x 相对当前位置的前进距离
    - formation_error: 轨迹各步与目标队形的偏差
    - crash: 轨迹是否与已知障碍物/其他车辆重叠
    - out_of_road: 轨迹是否超出道路边界
    - min_gap: 轨迹各步与其他车辆的最小间距
    - jerk: 轨迹加速度变化
    - delta_steering: 轨迹航向变化

    Returns
    -------
    {
        "step_infos": list[list[dict]],  # [G][8]{info_keys}
        "crash_flags": list[bool],       # [G]
        "out_of_road_flags": list[bool],  # [G]
    }
    """
```

### 实现要求

1. **progress 计算**：用 `agent_id` 当前位置 + 轨迹 endpoint 计算沿道路方向的前进距离（可使用 `_compute_progress` 的逻辑或简单地使用 `trajectory[-1, 0] - current_pos[0]` 沿行驶方向投影）
2. **formation_error 计算**：利用 `get_formation_relation_state(agent_id)` 获取当前目标队形偏差，结合轨迹位移估算各步的编队误差
3. **碰撞检测**：轨迹各步坐标与当前其他车辆位置做简单距离检测（距离 < 安全阈值则 crash=True）。可使用 `_compute_min_gap` 的逻辑
4. **出路检测**：如果 MetaDrive 提供道路边界信息，检查轨迹是否越界；否则使用横向偏移阈值（如 |lateral| > 3.5m）
5. **不得调用 `env.step()`**——这是纯评估方法
6. 如果环境尚未 reset（`self.agents` 为空），抛出 RuntimeError

### 验收指标（6 项）

1. `PlatoonEnv` 有 `evaluate_trajectory_group` 方法
2. 输入 `trajectories` shape `[G, 8, 3]`，返回 dict 含 `step_infos`（list of G, each list of 8 dicts）、`crash_flags`（list of G bools）、`out_of_road_flags`（list of G bools）
3. 每个 step_info dict 含 keys: `progress`, `formation_error`, `min_gap`, `jerk`, `delta_steering`, `crash`, `out_of_road`
4. crash_flags 中至少有一种检测逻辑工作（对一条明显碰撞的轨迹返回 True）
5. 不同轨迹返回不同 reward（通过 `compute_trajectory_reward` 验证）
6. 调用后 env 内部状态不变（调用前后 `get_formation_relation_state` 返回相同值）

### 验收命令
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task7.py -v
```

---

## 子任务 5.8：训练脚本连接真实 PlatoonEnv + PlatoonDiffusionPlanner

### 目标
修改 `train/train_platoon_rl.py`，使 `--mode platoon` 使用真实 `PlatoonEnv` + `PlatoonDiffusionPlanner`（从单车 checkpoint 迁移权重），而非 ToyEnv + ToyPlanner。`--mode toy-single` 保留 ToyEnv 用于快速调试。

### 交付物
- 修改 `train/train_platoon_rl.py`
- 新建 `tests/acceptance/test_phase5_task8.py`

### 实现要求

1. `--mode platoon` 时：
   ```python
   from envs.platoon_env import PlatoonEnv
   from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
   from models.platoon.weight_migration import migrate_single_to_platoon
   from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config

   env = PlatoonEnv({
       "observation_mode": "multimodal",
       "use_render": bool(args.render),
       "num_agents": 3,
       "horizon": 100,
       "traffic_density": 0.04,
       "num_scenarios": 1,
   })

   config_tf = build_transfuser_config(
       "small",
       plan_anchor_path="metadrive/exp_dataset/metadrive_anchors_ppo.npy",
   )
   model = PlatoonDiffusionPlanner(config_tf, num_vehicles=3)
   model = migrate_single_to_platoon(CKPT_PATH, model)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   ref_model = copy.deepcopy(model).to(device)
   ```

2. `--mode toy-single` 保持现有 ToyEnv + ToyPlanner 不变

3. checkpoint 路径通过 `--ckpt` 参数指定，默认为 `/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt`

4. `collect_group_samples` 调用后，trainer 应将选中的最优轨迹在 env 中实际执行一步（调用 `env.step`），推进环境状态到下一步。在 `MultiAgentGRPOTrainer` 中添加 `step_env_with_best` 方法：

   ```python
   def step_env_with_best(self, rollouts: dict) -> dict:
       """用每个 agent 的最优轨迹执行 env.step，推进环境状态。
       返回 next_obs（用于下一步的 collect_group_samples）。"""
       best_actions = {}
       for agent_id in rollouts["agent_ids"]:
           rewards = rollouts[agent_id]["reward"]
           best_idx = int(torch.argmax(rewards).item())
           best_actions[agent_id] = rollouts[agent_id]["trajectory"][best_idx].detach().cpu().numpy()
       obs, reward, terminated, truncated, info = self.env.step(best_actions)
       if terminated.get("__all__", False) or truncated.get("__all__", False):
           obs = self.env.reset()
       return obs
   ```

5. 训练循环改为：每步 collect → update → step_env_with_best → 用新 obs 做下一步 collect。不再每步都调 `env.reset()`

6. `MultiAgentGRPOTrainer.collect_group_samples` 需要接受可选的 `obs` 参数（若提供则不调 env.reset）：
   ```python
   def collect_group_samples(self, group_size: int = 4, obs: dict = None) -> dict:
       if obs is None:
           obs = self.env.reset()
       batch = _obs_to_tensor_batch(obs, self.device)
       ...
   ```

7. 添加 gradient checkpointing（`torch.utils.checkpoint`）以控制显存在 16GB 以内

### 验收指标（7 项）

1. `--mode platoon --steps 2 --render 0` 可运行，无 crash
2. 使用真实 `PlatoonEnv`（验证 `isinstance(trainer.env, PlatoonEnv)`）
3. 使用真实 `PlatoonDiffusionPlanner`（验证 `isinstance(trainer.model, PlatoonDiffusionPlanner)`）
4. 2 步后 `update()` 返回 loss 为有限值
5. `collect_group_samples` 接受 `obs` 参数（不强制每步 reset）
6. `step_env_with_best` 方法存在，调用后 env 状态推进
7. `--mode toy-single --steps 2` 仍可运行（向后兼容）

### 验收命令
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task8.py -v
```

---

## 子任务 5.9：真实环境单车 RL 冒烟测试（替代 5.5）

### 目标
用真实 PlatoonDiffusionPlanner（单车）+ PlatoonEnv（1 车模式）跑 20 步 RL，证明链路端到端可工作。

### 交付物
- 运行产出 `outputs/phase5/real_single_summary.json`
- 新建 `tests/acceptance/test_phase5_task9.py`

### 实现要求

1. 运行命令：
   ```bash
   /home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py \
       --mode platoon --steps 20 --render 0 \
       --config configs/train/platoon_grpo.yaml
   ```
   （注意：此处使用 1 车环境配置 + PlatoonDiffusionPlanner，不用 ToyPlanner）

2. 可选：在 yaml 中新增 `num_agents: 1` 配置项来控制单车/多车

3. 产出 JSON summary 文件含：steps, loss[], mean_reward[], kl[], grad_norm[], gpu_peak_gb, has_nan_loss, has_oom

### 验收指标（5 项）

1. 20 步全部完成，无 exception
2. 所有 loss 值有限（无 NaN / Inf）
3. gpu_peak_gb < 14（单车 group_size=2 应 < 8GB）
4. grad_norm 全部 < 200
5. kl 全部 < 10

### 验收命令
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py \
    --mode platoon --steps 20 --render 0
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task9.py -v
```

---

## 子任务 5.10：3 车编队 RL 真实训练（替代 5.6）

### 目标
用真实 PlatoonDiffusionPlanner + PlatoonEnv（3 车）跑 100 步 RL，验证编队训练基本可行。

### 交付物
- 运行产出 `outputs/phase5/real_platoon_summary.json`
- checkpoints 到 `checkpoints/platoon_rl_real/`
- tensorboard 到 `logs/platoon_rl_real/`
- 新建 `tests/acceptance/test_phase5_task10.py`

### 验收指标（6 项）

1. 100 步全部完成，无 exception
2. 无 NaN loss、无 OOM
3. gpu_peak_gb < 16
4. 每 50 步 checkpoint 到 `checkpoints/platoon_rl_real/`（≥ 2 个 ckpt）
5. tensorboard 含 8 曲线：loss, rl_loss, il_loss, kl, mean_reward, formation_error, collision_rate, grad_norm
6. mean_reward 后 20 步均值 ≥ 前 20 步均值 - 1.0（允许训练初期波动，但不能持续恶化）

### 验收命令
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py \
    --mode platoon --steps 100 --render 0 \
    --checkpoint-dir checkpoints/platoon_rl_real \
    --log-dir logs/platoon_rl_real
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5_task10.py -v
```

---

## 执行顺序

```
5.7 (PlatoonEnv.evaluate_trajectory_group)
  → 5.8 (训练脚本连接真实 env + planner)
    → 5.9 (单车冒烟测试 20 步)
      → 5.10 (3 车编队训练 100 步)
```

每个子任务完成后必须运行验收命令并确认全部 PASSED。

---

## 注意事项

### 显存预算（RTX 4080 16GB）
- 单车 group_size=2: ~4GB
- 3 车 group_size=2: ~8GB
- 3 车 group_size=4: ~12GB（需 gradient checkpointing）
- 超出时先降 group_size 到 2，再降 ddim_steps 到 2

### 已知限制
- MetaDrive Panda3D 退出时可能 segfault (code 139)——不影响训练逻辑，env.close() 放 finally 块
- `use_render=False` 必须设置（无头环境）
- 如果 GPU 上无 CUDA 可用，允许 CPU 回退运行（速度慢但正确）

### 代码质量要求
- 每个 test 文件中为每个验收指标单独写一个 `test_*` 函数（不要 1 个函数覆盖所有指标）
- 不允许在测试中 mock 环境或模型的核心方法
- 所有新增代码必须有类型注解
