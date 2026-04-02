# 工作规划：Selector + GRPO 双链训练方案

> 对应技术方案文档：`METHODS.md`
> 创建日期：2025-03-31

---

## 一、现有代码基础

以下组件**已完整实现**，可直接复用：

| 组件 | 文件 | 状态 |
|------|------|------|
| MAPPO selector 训练链 | `train/train_selector.py` | 完整，RLlib PPO 集成 |
| Selector 环境封装 | `envs/selector_platoon_env.py` | 完整，candidate 生成 + 奖励组合 |
| Selector 模型（Actor/Critic） | `models/selector/intent_selector.py` | 完整，MLP 策略网络 |
| RLlib 模型适配器 | `models/selector/rllib_selector_model.py` | 完整，obs/action 空间对接 |
| 多模态 candidate 生成 | `transfuser_model_v2.py:forward_test()` | 完整，返回 K 条 candidates + embeddings |
| RL 去噪上下文提取 | `platoon_diffusion_planner.py:extract_rl_context()` | 完整 |
| 单步去噪推理 | `platoon_diffusion_planner.py:predict_denoised_traj()` | 完整 |
| 编队关系编码 | `platoon_diffusion_planner.py:RelationEncoder` | 完整，12→64 维 |

以下组件**需要新建或重大修改**：

| 组件 | 说明 |
|------|------|
| MoE Task Decoder | `TrajectoryHead` 末端新增 Router + Experts + Output Heads |
| GRPO 微调训练器 | 全新模块，离线基于状态快照做 group 采样 + advantage 计算 |
| Rollout 状态保存机制 | MAPPO rollout 时保存 env state 到 replay buffer |
| 分支评估（Counterfactual Eval） | 从保存的状态重启仿真，评估 refined trajectory |
| 双链训练调度器 | 协调 MAPPO 在线链和 GRPO 离线链的更新频率 |

---

## 二、分阶段工作规划

### Phase A：验证 MAPPO Selector 基础性能

**目标**：确认 selector 能否在当前架构下学到有意义的 mode 选择策略。这是整个方案的前提假设。

**具体任务**：

- [ ] A1. 准备训练配置
  - 确认 `configs/train/selector.yaml` 参数合理（lr, batch_size, num_agents 等）
  - 确认预训练 checkpoint 路径和 anchor 文件路径正确
  - 设置合理的 K 值（建议从 K=9 开始，与现有 `SelectorPlatoonEnv` 默认值一致）

- [ ] A2. 运行 selector 训练
  - 执行 `scripts/run_train_selector.sh`
  - 监控 TensorBoard 指标：`episode_reward_mean`, `formation_error_mean`, `crash_rate`, `intent_entropy`

- [ ] A3. 评估训练结果
  - **收敛性**：reward 曲线是否上升并趋于稳定
  - **多样性**：intent_entropy 是否保持合理水平（不应坍缩到始终选同一个 mode）
  - **安全性**：crash_rate 是否下降
  - **编队性能**：formation_error 是否改善

- [ ] A4. 记录基线指标
  - 保存 selector-only 训练结果作为后续对比基线
  - 记录每个指标的最终值和收敛轮数

**验收标准**：
- reward 曲线收敛且高于随机选择 baseline
- crash_rate < 30%
- intent_entropy 保持在 [0.5, log(K)] 区间

**风险与应对**：
- 若 selector 完全不收敛：检查 candidate 质量（planner 生成的 K 条轨迹是否足够多样）
- 若 mode 坍缩：增大 entropy_coeff，或检查 candidate_summary 特征是否有区分度

---

### Phase B：Per-Mode Output Heads（简化版 MoE）

**目标**：将 `TrajectoryHead` 的 task_decoder 从单一共享改为 per-mode 独立 head，为后续 GRPO 微调提供参数隔离基础。

**前置条件**：Phase A 验证 selector 可收敛。

**具体任务**：

- [ ] B1. 分析当前 TrajectoryHead 的 task_decoder 结构
  - 读取 `transfuser_model_v2.py` 中 `CustomTransformerDecoder` 最后一层的输出处理逻辑
  - 确认当前 `plan_reg` 和 `plan_cls` output head 的结构（输入维度、层数）
  - 确认 `forward_train` 和 `forward_test` 中 output head 的调用方式

- [ ] B2. 实现 Per-Mode Output Heads
  - 新增 `PerModeDecoder` 类，包含 K 个独立的小 MLP（结构与原 output head 相同）
  - 每个 MLP 输入 `[B, D]`，输出 `plan_reg [B, T, 3]` + `plan_cls [B, 1]`
  - 初始化策略：从预训练 task_decoder 权重复制到每个 per-mode head

- [ ] B3. 修改 TrajectoryHead 的 forward 流程
  - `forward_train`：shared trunk 输出 `[B, K, D]` → 拆分为 K 个 `[B, D]` → 分别输入对应 mode 的 head
  - `forward_test`：同上，但只在最终步做 per-mode decode
  - `infer_multimodal`：保持返回所有 K 个 candidates 的接口不变

- [ ] B4. 新增 single-mode forward 接口
  - 新增 `forward_single_mode(mode_idx, traj_feature_i, ...)` 方法
  - 只对被选中的 mode 做 decode，供后续 GRPO 微调使用
  - 该方法只前向传播 `per_mode_heads[mode_idx]`

- [ ] B5. 验证预训练兼容性
  - 加载现有预训练 checkpoint
  - 运行 `forward_test`，确认输出轨迹与改造前一致（数值误差 < 1e-5）
  - 运行 selector 环境，确认 candidate 生成正常

- [ ] B6. 重新运行 selector 训练
  - 用新的 per-mode heads 架构重跑 Phase A 的训练
  - 对比性能指标，确认无明显退化

**验收标准**：
- 预训练权重迁移后推理输出不变
- selector 训练性能不低于 Phase A 基线

**代码修改范围**：
- `transfuser_model_v2.py`：~100-150 行新增/修改
- `platoon_diffusion_planner.py`：~20-30 行适配新接口

---

### Phase C：GRPO 离线微调链（核心难点）

**目标**：实现第二条训练链——基于 MAPPO rollout 保存的状态快照，对被选 mode 的 per-mode head 做 GRPO 微调。

**前置条件**：Phase B 的 per-mode heads 工作正常。

#### C1. Rollout 状态保存机制

- [ ] C1.1 扩展 SelectorPlatoonEnv 的 step() 方法
  - 在每个 step 结束后，保存以下数据到 replay buffer：
    ```python
    {
        "env_snapshot": {
            "scenario_seed": int,      # 场景随机种子
            "step_idx": int,           # 当前步骤
            "agent_positions": dict,   # 各车位置/朝向/速度
            "traffic_state": ...,      # 周围交通状态
        },
        "planner_context": {
            "ego_query": tensor,       # [1, 1, D]
            "agents_query": tensor,    # [1, N, D]
            "bev_feature": tensor,     # [1, C, H, W]
            "status_encoding": tensor, # [1, 1, D]
        },
        "selected_mode_idx": int,
        "selected_traj": tensor,       # [T, 3]
        "reward": float,
    }
    ```
  - Buffer 容量：最近 N 个 episode 的全部 step 数据

- [ ] C1.2 实现 replay buffer 存储类
  - 文件：`train/grpo_replay_buffer.py`
  - 支持按 episode 存取、随机采样、淘汰旧数据

#### C2. 分支评估机制

- [ ] C2.1 调研 MetaDrive 状态回放能力
  - 确认 MetaDrive 是否支持 `env.set_state()` 或等价的状态恢复 API
  - 如果不支持精确回放，评估以下替代方案：
    - **方案 a**：用相同 seed + 跳步到目标 step（精确但慢）
    - **方案 b**：只评估开环轨迹质量（不需要环境交互，用 reward_terms 直接计算）
    - **方案 c**：用短窗口闭环评估（从保存状态前向 rollout 5-10 步）

- [ ] C2.2 实现分支评估器
  - 文件：`train/counterfactual_evaluator.py`
  - 输入：env_snapshot + refined_trajectory
  - 输出：该 trajectory 的 reward
  - 根据 C2.1 的调研结果选择具体实现方案

#### C3. GRPO 训练器

- [ ] C3.1 实现 Group 采样
  - 从 replay buffer 取一个 (env_snapshot, planner_context, selected_mode_idx)
  - 对该 mode 的 anchor trajectory 复制 `num_group` 份（建议 num_group=4~8）
  - 对每份添加不同的截断噪声
  - 输入 planner 执行多步去噪，生成 `num_group` 条 refined trajectories

- [ ] C3.2 实现 GRPO Advantage 计算
  - 对 `num_group` 条 refined trajectories 分别做分支评估，得到 rewards
  - 组内归一化：`advantage_i = (reward_i - mean(rewards)) / std(rewards)`
  - 基准轨迹：原始 selected_traj 的 reward

- [ ] C3.3 实现 GRPO Loss
  - 文件：`train/grpo_trainer.py`
  - 核心 loss：
    ```python
    # policy_ratio = new_log_prob / old_log_prob
    # loss = -mean(advantage * min(ratio, clip(ratio, 1-eps, 1+eps)))
    # + beta * KL(new_policy || ref_policy)
    ```
  - 其中 ref_policy 是 GRPO 微调开始时的 planner 参数快照
  - KL 约束防止 per-mode head 偏离太远

- [ ] C3.4 实现参数隔离更新
  - 冻结 shared trunk 参数（`requires_grad = False`）
  - 只对 `per_mode_heads[selected_mode_idx]` 的参数计算梯度
  - 可选：对 Router（如果有）也更新

- [ ] C3.5 集成测试
  - 单元测试：确认梯度只流向目标 mode 的参数
  - 端到端测试：跑一个 mini GRPO 更新，确认 loss 下降

#### C4. 双链调度

- [ ] C4.1 实现训练调度器
  - 文件：`train/dual_chain_scheduler.py`
  - 逻辑：
    ```
    for epoch:
        # Chain 1: MAPPO selector 训练
        mappo_result = selector_algo.train()  # N steps
        save_rollout_data_to_buffer(mappo_result)

        # Chain 2: GRPO planner 微调（每 M 个 MAPPO epoch 执行一次）
        if epoch % grpo_interval == 0:
            grpo_batch = buffer.sample(batch_size)
            grpo_trainer.update(grpo_batch, planner)
    ```
  - GRPO 更新频率建议：每 5-10 个 MAPPO epoch 做一次

- [ ] C4.2 添加 planner 参数同步机制
  - GRPO 更新 planner 后，selector 环境中的 planner 需要同步新参数
  - 注意：参数更新后 mode embedding 语义可能变化，selector 需要适应期

**验收标准**：
- GRPO 微调后，被选 mode 的轨迹质量（reward）有可度量的提升
- 整体系统（selector + refined planner）性能优于 Phase A 的 selector-only 基线
- 未被选 mode 的输出不受影响（shared trunk 冻结验证）

**代码新增范围**：
- `train/grpo_replay_buffer.py`：~150 行
- `train/counterfactual_evaluator.py`：~200 行（取决于方案选择）
- `train/grpo_trainer.py`：~300 行
- `train/dual_chain_scheduler.py`：~200 行
- `envs/selector_platoon_env.py`：~50 行修改（状态保存）

---

### Phase D：完整 MoE Task Decoder（可选）

**目标**：将 Phase B 的 per-mode independent heads 升级为带 Router 的 MoE 结构。

**前置条件**：Phase C 验证 per-mode refinement 确实有效。

**具体任务**：

- [ ] D1. 实现 MoE Router
  - 输入 `traj_feature_i [B, D]`，输出 gating weights `[B, M]`
  - M 为 expert 数量（建议 M=4~8，不需要等于 K）
  - Top-k gating（k=2），使用 load balancing loss

- [ ] D2. 实现 Expert 池
  - M 个并行小 MLP（结构与 Phase B 的 per-mode head 相同）
  - 从 Phase B/C 训练好的 per-mode heads 中选取表现最好的作为初始化

- [ ] D3. 实现 Expert Fusion
  - 按 gating weights 对 expert 输出做加权求和
  - 输出接 output heads（plan_reg + plan_cls）

- [ ] D4. 替换 per-mode heads 并验证
  - 加载 Phase C 的 checkpoint
  - 重新跑 selector + GRPO 训练
  - 对比 per-mode heads vs MoE 的性能差异

**评估决策点**：
- 如果 Phase C 的 per-mode heads 已经足够好，Phase D 可以跳过
- MoE 的主要价值在于跨 mode 的策略复用（如"急刹" expert 被多个 mode 共享）
- 如果 K 较小（K ≤ 9），MoE 的优势可能不明显

---

## 三、关键技术风险与缓解措施

### 风险 1：MetaDrive 不支持状态快照回放

**影响**：Phase C 的分支评估机制无法实现精确的 counterfactual evaluation。

**缓解**：
- 优先调研 MetaDrive 的 `get_state()` / `set_state()` API
- 退路方案：使用开环评估（基于 reward_terms 直接计算轨迹质量，不做闭环仿真）
- 开环评估虽然不如闭环准确，但实现简单且训练信号仍然有效

### 风险 2：GRPO 更新导致 Mode Embedding 漂移

**影响**：planner 参数更新后，selector 之前学到的 mode embedding → action 映射失效。

**缓解**：
- 冻结 shared trunk，只更新 per-mode heads → embedding 来自 trunk，不会漂移
- GRPO 更新幅度用 KL 约束限制
- GRPO 更新频率不宜过高（每 5-10 个 MAPPO epoch 一次）

### 风险 3：Mode Collapse

**影响**：GRPO 反复强化某几个 mode，其他 mode 退化，selector 失去多样性。

**缓解**：
- 监控 intent_entropy 指标
- 对不同 mode 的 GRPO 更新次数做均衡（按被选频率加权）
- 可选：对 planner 添加 mode diversity 正则项

### 风险 4：训练不稳定

**影响**：两条链交替更新时相互干扰。

**缓解**：
- Phase A 先单独验证 selector
- Phase C 初期固定 selector，只调 GRPO
- 确认两条链各自稳定后再联合训练
- 用较大的 replay buffer 平滑数据分布变化

---

## 四、工作量总结

| Phase | 核心工作 | 预估工期 | 依赖 |
|-------|---------|---------|------|
| A | 验证 MAPPO selector | 1-2 周 | 无 |
| B | Per-mode output heads | 1 周 | Phase A 通过 |
| C | GRPO 离线微调链 | 2-3 周 | Phase B 完成 |
| D | 完整 MoE（可选） | 1 周 | Phase C 有效 |
| **总计** | | **5-7 周** | |

**关键里程碑**：
- Phase A 结束：确认方案前提假设成立（selector 可收敛）
- Phase B 结束：模型架构改造完成，可发布中间结果
- Phase C 结束：完整双链训练跑通，可做完整实验对比
