# GRPO Trajectory Refinement 实现计划

## 一、方案概述

在 selector 选出高层意图后，对已选轨迹加截断噪声，通过组内多条轨迹比较（GRPO），优中选优，得到更适合编队行驶的 refined 轨迹。

### 核心流水线

```
selector 选出 intent_idx → τ_selected [8, 3]
  → 截断噪声 t=8/1000（α≈0.992）→ G=4 条 noisy 变体
  → 10 步 DDIM 去噪 → G 条 refined 轨迹（典型调整 0.3~1.0m）
  → get_state/set_state 分支模拟 → G 个闭环 reward
  → 组内标准化 advantage（正向 clamp + 安全约束）
  → 执行 best refined → 实际 env step
  → replay diffusion chain with grad → policy gradient 更新 diff_decoder
```

### 设计共识

1. **三车并行 refine**：各车独立 refine，不需要链式传递。车间交叉影响为二阶小量（~0.03m/step vs 0.3~1.0m 调整量，偏差 3~10%）。
2. **只解冻 diff_decoder**：backbone、encoder、selector 全冻结，仅 trajectory head 的 diff_decoder 参数有梯度。
3. **分阶段训练**：selector MAPPO 收敛 → 冻结 selector 训练 refinement → 可选弱耦合联合微调。
4. **推理时 1 次 refinement**：不需要 G 组比较，增加 ~20ms 延迟，总延迟仍在 100ms 内。

---

## 二、关键参数

| 参数 | 值 | 来源/理由 |
|------|---|----------|
| G（组大小） | 4 | 平衡方差与显存；DiffusionDriveV2 用 G=4 |
| trunc_timestep | 8 | t=8/1000，α≈0.992，噪声极小 |
| DDIM steps | 10 | 参考 DiffusionDriveV2 训练时 step_num=10 |
| roll_step_range | 20 | 映射 10 步到 timesteps [18,16,...,0] |
| eta | 0.5 | DDIM 噪声系数，与 Phase 5 一致 |
| lr | 1e-5 ~ 3e-5 | 比 selector 的 3e-4 小一个量级 |
| warmup | 1 epoch | cosine decay |
| advantage clamp | min=0 | 只保留 positive advantage（优中选优） |
| 安全约束 | crash/out_of_road → advantage=-1 | 不从不安全轨迹学习 |
| temporal discount | 0.8^(step_num-i-1) | 早期 step 权重更高 |
| 训练 env steps | 5000~10000 | 预期 3000~5000 步可见 formation_error 下降 |

---

## 三、代码修改任务

### Task 1：恢复 DiffusionRLScheduler

**文件**：`models/diffusion/diffusion_rl_scheduler.py`

**操作**：从 git 历史恢复已删除的文件。该文件已包含：
- `DDIMSchedulerWithLogProb.step()` — 带 log_prob 的 DDIM step
- `DiffusionRLScheduler.sample_with_log_prob()` — Phase 1 采样（no_grad）
- `DiffusionRLScheduler.replay_with_log_prob()` — Phase 2 重放（with grad）
- `DiffusionRLScheduler.compute_ref_predictions()` — 参考模型预测
- `DiffusionRLScheduler.compute_kl()` — KL 监控

**需要小改**：
- 增加 `sample_single_anchor()` 方法：接受单条已选轨迹作为 anchor（而非 plan_anchor 全集），返回 G 条变体的 refined 轨迹、log_prob、diffusion_chain
- 签名：`sample_single_anchor(model, context, selected_traj_norm, num_groups=4) -> dict`

**验收**：
```python
# 单元测试：给定一条归一化轨迹，输出 G 条变体
scheduler = DiffusionRLScheduler(config)
result = scheduler.sample_single_anchor(planner, ctx, traj_norm, num_groups=4)
assert result["trajectory"].shape == (4, 1, 8, 3)
assert result["log_prob"].shape == (4, 1, 10)
assert len(result["diffusion_chain"]) == 11
```

---

### Task 2：新建 TrajectoryRefiner

**文件**：`models/platoon/trajectory_refiner.py`（新建，约 200 行）

**类设计**：

```python
class TrajectoryRefiner:
    """封装 GRPO refinement 的完整流程：采样、打分、advantage、replay。"""

    def __init__(self, planner, scheduler_config, refine_config):
        """
        Args:
            planner: PlatoonDiffusionPlanner（frozen backbone，解冻 diff_decoder）
            scheduler_config: DiffusionRLScheduler 配置
            refine_config: dict with keys:
                num_groups: int = 4
                advantage_clamp_min: float = 0.0
                crash_advantage: float = -1.0
                temporal_discount_base: float = 0.8
                il_weight: float = 0.1  # IL 正则化权重
        """

    def refine_and_score(
        self,
        env: PlatoonEnv,
        agent_id: str,
        selected_traj: np.ndarray,  # [8, 3]
        obs_batch: dict,            # planner 输入
        reward_config: dict,
    ) -> dict:
        """
        1. 提取 context（no_grad）
        2. 对 selected_traj 加截断噪声 → G 条变体
        3. DDIM 去噪 → G 条 refined 轨迹
        4. get_state/set_state 分支模拟 → G 个 reward
        5. 组内标准化 advantage
        6. 返回 best refined 轨迹 + advantage + chain 等

        Returns:
            {
                "best_trajectory": np.ndarray [8, 3],
                "best_idx": int,
                "rewards": Tensor [G],
                "advantages": Tensor [G],
                "diffusion_chain": list[Tensor],
                "log_prob": Tensor [G, 1, step_num],
                "all_trajectories": Tensor [G, 8, 3],
            }
        """

    def compute_loss(
        self,
        obs_batch: dict,
        agent_id: str,
        refine_result: dict,
    ) -> Tensor:
        """
        1. replay_with_log_prob → 带梯度的 log_prob
        2. GRPO loss: -exp(log_p - detach(log_p)) * advantage * discount
        3. 可选 IL loss: L1(refined, selected)
        4. 返回 total loss
        """

    def inference(
        self,
        selected_traj: np.ndarray,
        obs_batch: dict,
        agent_id: str,
    ) -> np.ndarray:
        """推理模式：只做 1 次 refinement（G=1），不需要打分和 advantage。"""
```

**依赖**：
- `DiffusionRLScheduler`（Task 1）
- `PlatoonDiffusionPlanner.extract_rl_context` / `predict_denoised_traj`
- `PlatoonEnv.get_state` / `set_state`
- `compute_step_reward` / `compute_team_reward`

**验收**：
```python
refiner = TrajectoryRefiner(planner, sched_cfg, refine_cfg)
result = refiner.refine_and_score(env, "agent_0", traj, obs, reward_cfg)
assert result["best_trajectory"].shape == (8, 3)
assert result["advantages"].shape == (4,)
loss = refiner.compute_loss(obs, "agent_0", result)
assert loss.requires_grad
```

---

### Task 3：PlatoonDiffusionPlanner 增加解冻控制

**文件**：`models/platoon/platoon_diffusion_planner.py`

**新增方法**：

```python
def unfreeze_diff_decoder(self) -> None:
    """解冻 trajectory head 的 diff_decoder，保持其余参数冻结。"""
    self.freeze_for_selector()  # 先全冻结
    th = self.model._trajectory_head
    for name, param in th.diff_decoder.named_parameters():
        param.requires_grad = True
    # 可选：解冻 time_mlp（时间嵌入）
    for param in th.time_mlp.parameters():
        param.requires_grad = True
```

**验收**：
```python
planner.unfreeze_diff_decoder()
trainable = sum(p.numel() for p in planner.parameters() if p.requires_grad)
total = sum(p.numel() for p in planner.parameters())
assert trainable < total * 0.1  # diff_decoder 参数量 < 总量 10%
assert trainable > 0
```

---

### Task 4：SelectorPlatoonEnv 增加 refinement 分支

**文件**：`envs/selector_platoon_env.py`

**修改 `step()` 方法**：

```python
def step(self, action_dict):
    # 原有逻辑：selector intent → 选轨迹
    traj_actions = {}
    refine_results = {}
    for agent_id, intent_idx in action_dict.items():
        selected_traj = self._cached_candidates[agent_id][intent_idx]

        if self.refiner is not None:
            # GRPO refinement（三车并行，各自独立）
            result = self.refiner.refine_and_score(
                env=self.base_env,
                agent_id=agent_id,
                selected_traj=selected_traj,
                obs_batch={agent_id: self._cached_obs[agent_id]},
                reward_config=self.reward_config,
            )
            traj_actions[agent_id] = result["best_trajectory"]
            refine_results[agent_id] = result
        else:
            traj_actions[agent_id] = selected_traj

    # 执行 best refined trajectory
    obs, reward, terminated, truncated, info = self.base_env.step(traj_actions)

    # 如果有 refine_results，存入 info 供训练使用
    if refine_results:
        info["_refine_results"] = refine_results

    # ... 后续 reward 计算、obs 构建等不变
```

**关键注意**：三车的 `refine_and_score` 是顺序调用但各自独立做 G 次分支 rollout。每次分支 rollout 后会 `set_state` 恢复到调用前状态。

**新增 `__init__` 参数**：
- `refiner: Optional[TrajectoryRefiner] = None`
- 由训练入口根据 mode 决定是否传入 refiner

**验收**：
```python
# 不传 refiner：行为与现有完全一致
env = SelectorPlatoonEnv(config, refiner=None)
# 传 refiner：step 后 info 包含 _refine_results
env = SelectorPlatoonEnv(config, refiner=refiner)
obs, r, d, t, info = env.step({a: 0 for a in env.agents})
assert "_refine_results" in info
```

---

### Task 5：新增训练配置

**文件**：`configs/train/platoon_selector_refine.yaml`（新建）

```yaml
# 继承 selector 配置
inherit: platoon_mappo.yaml

# Refinement 特有配置
refinement:
  enabled: true
  num_groups: 4                  # G 组变体
  trunc_timestep: 8              # 截断时间步
  ddim_steps: 10                 # DDIM 去噪步数
  roll_step_range: 20            # timestep 映射范围
  eta: 0.5                       # DDIM 噪声系数
  advantage_clamp_min: 0.0       # 只保留 positive advantage
  crash_advantage: -1.0          # 碰撞时 advantage
  temporal_discount_base: 0.8    # 时间折扣基数
  il_weight: 0.1                 # IL 正则化权重

# Refinement 优化器
refine_optimizer:
  lr: 2.0e-5
  weight_decay: 1.0e-4
  warmup_epochs: 1
  min_lr: 1.0e-6

# Selector 冻结
selector_frozen: true

# 训练步数建议
default_steps: 5000
```

**验收**：配置文件可被 `load_config()` 正确解析。

---

### Task 6：训练入口 mode 分支

**文件**：`train/train_selector.py`

**新增 `platoon-selector-refine-train` 分支**：

```python
elif args.mode == "platoon-selector-refine-train":
    # 1. 加载 selector checkpoint（已收敛）
    # 2. 构建 planner，调用 planner.unfreeze_diff_decoder()
    # 3. 构建 DiffusionRLScheduler + TrajectoryRefiner
    # 4. 构建带 refiner 的 SelectorPlatoonEnv
    # 5. 训练循环：
    #    - env.step() 内部做 GRPO refinement + 分支模拟打分
    #    - 从 info["_refine_results"] 取 advantage 和 chain
    #    - 调用 refiner.compute_loss() → backward → optimizer.step()
    #    - 同时 RLlib MAPPO 照常更新 selector（但 selector_frozen=True 时跳过）
    # 6. 定期保存 checkpoint、记录 TensorBoard 指标
```

**注意**：refinement 的梯度更新不走 RLlib，而是在 env wrapper 外层用独立的 PyTorch optimizer。RLlib 只负责 selector 的 rollout 收集和 MAPPO 更新。

**验收**：
```bash
python -m train.train_selector \
  --config configs/train/platoon_selector_refine.yaml \
  --total-env-steps 32
# 应能跑通 32 步不报错
```

---

### Task 7：Callbacks 增加 refinement 指标

**文件**：`train/selector_callbacks.py`

**新增指标**：

| 指标 | 含义 |
|------|------|
| `refine_reward_mean` | G 组 refined 轨迹的平均 reward |
| `refine_reward_best` | G 组中 best 的 reward |
| `refine_advantage_mean` | 平均 positive advantage |
| `refine_displacement_mean` | refined 轨迹与 selected 轨迹的平均位移差（监控 refinement 幅度） |
| `refine_loss` | GRPO loss 值 |
| `refine_kl` | 与冻结 reference 的 KL 散度（可选，用于监控偏离程度） |

**验收**：TensorBoard 中可看到上述指标曲线。

---

## 四、任务依赖关系

```
Task 1 (恢复 DiffusionRLScheduler)
  └──→ Task 2 (TrajectoryRefiner)
        ├──→ Task 4 (SelectorPlatoonEnv refinement 分支)
        └──→ Task 6 (训练入口)
Task 3 (unfreeze_diff_decoder)
  └──→ Task 6 (训练入口)
Task 5 (配置文件)
  └──→ Task 6 (训练入口)
Task 7 (Callbacks) ── 独立，可并行
```

**推荐执行顺序**：Task 1 → Task 3 → Task 2 → Task 5 → Task 4 → Task 6 → Task 7

---

## 五、推理流水线（训练完成后）

```python
# selector 选出 intent
intent = selector(obs)
selected_traj = candidates[intent]

# refinement（只做 1 次，G=1）
refined_traj = refiner.inference(selected_traj, obs_batch, agent_id)

# 执行
env.step({agent_id: refined_traj})
```

推理延迟分解：
- planner.forward_selector(): ~30ms（3 车）
- selector forward: ~2ms
- refinement (1 次 DDIM 10 步): ~20ms（3 车）
- env.step(): ~5ms
- **总计: ~57ms < 100ms**

---

## 六、显存预算

| 组件 | 显存 |
|------|------|
| Frozen backbone (3 车共享) | ~2.0 GB |
| Frozen selector + critic | ~0.1 GB |
| diff_decoder forward (G=4 × 3 车，分批) | ~3.0 GB |
| diff_decoder 梯度 | ~2.0 GB |
| 分支模拟 env state | ~0.5 GB |
| 其他 (optimizer states, activations) | ~2.0 GB |
| **总计** | **~9.6 GB < 16 GB** |

如显存不够，可将 G 从 4 降到 2，或对 3 车的 refine 串行执行（分批释放显存）。

---

## 七、风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 分支模拟 get_state/set_state 不完整 | 低（Phase 5v2 已验证） | 复用 Phase 5v2 的 ClosedLoopExecutor 测试 |
| Refinement 不收敛 | 中 | 监控 refine_displacement_mean，若趋近 0 说明没学到东西；加大 eta 或 trunc_timestep |
| Refinement 偏离太远 | 低（截断噪声很小） | 监控 refine_kl，超阈值时加大 IL 正则权重 |
| 显存溢出 | 低 | G=4 预算 ~10GB，RTX 4080 16GB 余量充足 |
| 训练速度太慢 | 中 | 每 env step 做 G 次分支 rollout 约 150ms；可用 gradient accumulation 减少 replay 频率 |
