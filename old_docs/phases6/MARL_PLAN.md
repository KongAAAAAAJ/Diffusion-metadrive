# Phase 6：RLlib MAPPO Intent Selector 替换现有 GRPO 训练链

## Summary
在 [`docs/phases6/mappo_selector_plan.md`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/docs/phases6/mappo_selector_plan.md) 的基础上，正式落地为一条**直接替换**当前 GRPO 的新训练链：冻结单车 diffusion planner，只训练共享参数的意图 selector；训练框架采用 **RLlib PPOConfig + shared actor + centralized critic** 的 MAPPO 风格实现。

这版计划有 4 个关键收口：
- 不复用现有 `plan_cls_branch`；新增 selector 专用接口与新 head。
- 不把 raw 图像喂给 RLlib；env 包装层只输出 planner 提炼后的 compact context、mode embedding 和少量候选几何摘要。
- 不用 callback 注入 centralized critic 所需 joint obs；由 selector env 直接把 `global_state` 放进每个 agent 的 observation，降低 RLlib 接线复杂度。
- 直接替换当前 RL 训练入口，但**旧 GRPO 文件删除放到最后一轮 cleanup**，在 MAPPO 路径验收通过后执行，避免中途失去可运行训练链。

## Key Changes
### 1. 冻结 planner，并新增 selector 专用多模态导出接口
- 在 [`metadrive/policy/diffusion_policy/transfuser_model_v2.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/metadrive/policy/diffusion_policy/transfuser_model_v2.py) 新增 `infer_multimodal(features)`，不改现有 `forward()` 行为。
- 该接口固定返回：
  - `trajectory`: `[B, 8, 3]`
  - `trajectory_mode_idx`: `[B]`
  - `trajectory_mode_logits`: `[B, K]`
  - `trajectory_candidates`: `[B, K, 8, 3]`
  - `trajectory_mode_embedding`: `[B, K, D]`
- `trajectory_candidates` 直接来自 selector 之前的多模态输出，不能再只保留 `argmax` 后的一条。
- `trajectory_mode_embedding` 取 scene-conditioned 的最后一层 mode hidden，不取纯 anchor 初值。
- 在 [`models/platoon/platoon_diffusion_planner.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/models/platoon/platoon_diffusion_planner.py) 新增 `forward_selector(batch)`，按 agent 返回上述字段，并继续保留当前 `forward()` 供旧推理/验收使用。
- planner 在 selector 训练链中**全参数冻结**，并在训练日志里显式统计 `planner_trainable_params == 0`。

### 2. 新建 RLlib 专用 selector env，而不是改 `PlatoonEnv`
- 新增 [`envs/selector_platoon_env.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/envs/selector_platoon_env.py)，实现 RLlib `MultiAgentEnv` 包装层，内部持有一个真实 [`PlatoonEnv`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/envs/platoon_env.py) 和一个 worker-local frozen planner。
- `reset()` 统一返回 `(obs_dict, info_dict)`，包装掉当前 `PlatoonEnv.reset()` 只返回 `obs` 的差异。
- `step(action_dict)` 输入是 `{agent_id: intent_idx}`，包装层从本地 `candidate_cache` 取出对应的 `[8,3]` 轨迹，再调用 `base_env.step(traj_actions)`。不能走 `low_level_step()`。
- selector env 的每车 observation 固定是 `gym.spaces.Dict`，至少包含：
  - `agent_context`: planner 提炼后的本车紧凑上下文
  - `formation_relation_state`: `[12]`
  - `mode_embeddings`: `[K, D]`
  - `candidate_summary`: `[K, S]`
  - `global_state`: 所有车的 context/relation/summary 拼接后的 team-level 向量
  - `action_mask`: `[K]`，默认全 1，后续可扩展无效 mode 屏蔽
- 不把 raw camera/lidar 输入 RLlib。
- reward 由包装层重算，使用底层 `info` 中已有的 `progress / formation_error / min_gap / crash / out_of_road / speed_km_h`，组合成：
  - `r_i = lambda_local * r_local_i + lambda_team * r_team`
- selector env 保留底层 `info`，并新增：
  - `selected_intent`
  - `selector_reward`
  - `control_mode="trajectory"`
  - `intent_valid`
- training worker 强制 `use_render=False`，evaluation 再单独开 render。

### 3. 新增 shared actor + centralized critic，并用 RLlib PPO 落 MAPPO
- 新增 [`models/selector/intent_selector.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/models/selector/intent_selector.py)：
  - `IntentSelectorActor`
  - `IntentSelectorCritic`
- Actor 只消费 `agent_context + formation_relation_state + mode_embeddings + candidate_summary`，输出 `K` 个 logits。
- Critic 只消费 `global_state`，输出 scalar value。第一版不做 callback/postprocess 拼 joint obs，完全依赖 env 侧提供 `global_state`。
- 新增 [`models/selector/rllib_selector_model.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/models/selector/rllib_selector_model.py)，采用 **旧版兼容性更好的 `TorchModelV2`** 封装 actor/critic；不使用 RLModule。
- 训练入口新增 [`train/train_selector.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/train/train_selector.py)，基于 `PPOConfig` 配置：
  - shared policy mapping
  - centralized critic model
  - `framework="torch"`
  - rollout workers、checkpoint、callbacks、evaluation
- 新增 [`train/selector_callbacks.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/train/selector_callbacks.py)，记录：
  - `formation_error_mean`
  - `crash_rate`
  - `team_reward_mean`
  - `intent_entropy`
  - `intent_usage_{k}`
- 新增 [`configs/train/selector.yaml`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/configs/train/selector.yaml)。
- 训练依赖版本固定为 **`ray[rllib]==2.4.0`**；当前环境已安装该版本并完成 smoke train 验证。注意 Ray 2.4 仍支持 `TorchModelV2` 旧 API（`_enable_rl_module_api=False`），无需迁移到 RLModule。

### 4. 训练入口替换与旧链路退场
- [`train/train_selector.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/train/train_selector.py) 作为 MAPPO selector 唯一主入口。
- 默认 mode 改成 selector 训练语义，例如：
  - `platoon-selector-train`
  - `platoon-selector-eval`
- 旧的 GRPO 相关模块：
  - [`train/ma_grpo_trainer.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/train/ma_grpo_trainer.py)
  - [`train/joint_group.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/train/joint_group.py)
  - [`train/closedloop_executor.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/train/closedloop_executor.py)
  - [`models/diffusion/diffusion_rl_scheduler.py`](/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive/models/diffusion/diffusion_rl_scheduler.py)
  不在第一批实现中直接删除；先停止被训练入口引用，待 MAPPO 全量验收通过后统一 cleanup。
- 相关 README / AGENTS / 脚本默认命令全部改到 MAPPO selector 路径，GRPO 仅在历史对照文档中保留说明。

## Explicit Interface Decisions
- planner 导出接口：
  - `infer_multimodal(features) -> dict`
  - `forward_selector(batch) -> dict[agent_id -> selector_payload]`
- selector env 动作空间：
  - 每车 `Discrete(K)`
- selector env 观测空间：
  - 每车 `Dict(agent_context, formation_relation_state, mode_embeddings, candidate_summary, global_state, action_mask)`
- actor/critic 输入边界：
  - actor 不看 `global_state`
  - critic 不看 raw image/lidar
- centralized critic 实现方式：
  - **env 直接提供 `global_state`**
  - **不使用** callback 注入 joint obs
- RLlib API：
  - `PPOConfig + TorchModelV2`
  - 不使用 RLModule / MARLLIB
- rollout worker 资源策略：
  - 1 worker = 1 env = 1 frozen planner
  - 训练不启 render
- 替换策略：
  - 训练入口直接切 MAPPO
  - 旧 GRPO 文件最后清理，不作为新实现阻塞项

## Acceptance Criteria
### A. 单元与接口验收
1. planner selector 接口
- `forward_selector()` 对 3 车输入返回 3 个 agent key。
- 每个 agent 都有：
  - `trajectory_candidates.shape == (K, 8, 3)`
  - `trajectory_mode_logits.shape == (K,)`
  - `trajectory_mode_embedding.shape == (K, D)`
- `K` 必须等于当前模型真实 `ego_fut_mode`，不能写死。
- 同一输入、同一 seed 下输出稳定一致。
- planner 所有参数 `requires_grad=False`。

2. selector env 接口
- `reset()` 返回 `(obs_dict, info_dict)`。
- `step()` 返回 RLlib 多智能体五元组，且 `terminateds/truncateds` 含 `"__all__"`。
- 输入离散 action 后，底层实际收到的是 `{agent_id: np.ndarray(8,3)}` 轨迹动作。
- observation 中所有 Box/Dict shape 与 space 声明严格一致。
- `global_state` 在所有 agent observation 中存在且 shape 固定。
- 任意 crash/out_of_road 时，episode 以团队同步方式结束。

3. selector model 接口
- actor 前向输出 `(B, K)` logits。
- critic 前向输出 `(B,)` value。
- `value_function()` 只依赖 `global_state`。
- policy sample 出来的 action 全部落在 `[0, K-1]`。

### B. RLlib 集成验收
4. smoke train
- `train/train_selector.py` 能在 `meta_drive` 环境下启动至少 3 个 training iterations，无 import / shape / worker crash。
- RLlib 日志中出现：
  - `episode_reward_mean`
  - `custom_metrics/formation_error_mean`
  - `custom_metrics/crash_rate`
  - `custom_metrics/intent_entropy`
- checkpoint 能按配置间隔保存。

5. 冻结正确性
- 单次训练后：
  - selector actor/critic 参数发生变化
  - planner 权重哈希不变
- TensorBoard 或 summary 中显式记录：
  - `selector_grad_norm`
  - `planner_grad_norm == 0` 或 planner 无梯度

6. worker 闭环稳定性
- `num_rollout_workers=2`、每 worker 1 env、3 车 platoon 条件下，训练 20 iterations 不出现：
  - Panda3D render 初始化错误
  - `Not enough objects exist!`
  - agent key 集异常丢失
  - action/obs shape mismatch

### C. 功能性验收
7. selector 行为不是退化常数策略
- 训练 20 iterations 后，`intent_usage_k` 不能所有质量都集中到单一 mode 且熵长期接近 0。
- 至少存在 2 个以上 mode 被实际选择。

8. reward 与环境链路正确
- `selector_reward` 与 `info` 中 `formation_error/progress/crash/out_of_road` 的组合关系符合配置。
- crash 时 reward 明显下降，progress 增加时 reward 正向增加。

9. 最小效果验收
- 在固定 20 episode evaluation 下：
  - `formation_error_mean` 为有限值
  - `crash_rate` 为有限值
  - 不要求首版必须优于旧 GRPO，但必须能稳定完成评估并输出完整指标

### D. 替换与清理验收
10. 入口替换完成
- `scripts/run_marl_train.sh` 和 `scripts/run_train.sh` 默认走 MAPPO selector。
- README 中 RL 训练说明已改为 selector MAPPO。
- 旧 GRPO 训练入口不再是默认主路径。

11. cleanup 完成条件
- 仅当 A/B/C/D 通过后，才删除旧 GRPO 训练器与相关 acceptance。
- 删除后仍需保证：
  - 新训练入口可跑
  - Phase 6 selector 验收全绿
  - README/脚本无悬挂引用

## Recommended Parallel Work Split
### Agent 1：Planner 导出与 selector 特征面
- 负责 `transfuser_model_v2.py`、`platoon_diffusion_planner.py`
- 目标：导出 `trajectory_candidates / logits / mode_embeddings / agent_context`
- 写完先交接口测试和 shape 验收

### Agent 2：Selector env 与 reward 包装
- 负责 `envs/selector_platoon_env.py`
- 目标：离散 intent -> 轨迹动作映射、obs Dict 构造、reward 重算、cache 生命周期
- 写完先交 env reset/step/RLlib 兼容测试

### Agent 3：RLlib model + training entry
- 负责 `models/selector/*`、`train/train_selector.py`、`train/selector_callbacks.py`、`configs/train/selector.yaml`
- 目标：shared actor、centralized critic、PPOConfig、callbacks、checkpoint/logging
- 写完先交 3-iteration smoke train

### Agent 4：替换与验收
- 负责脚本、README、acceptance、最终 cleanup
- 前提：Agent 1/2/3 都已交付
- 目标：补 Phase 6 acceptance tests，切默认入口，最后删除旧 GRPO 路径

## Assumptions
- 当前环境已安装 `ray[rllib]==2.4.0`，训练入口已通过 smoke train 验证。
- 首版不做 selected-intent GRPO refinement，只做 selector-only MAPPO。
- 首版不做无效 candidate pruning，`action_mask` 默认全 1。
- 首版 centralized critic 采用 env 侧显式提供 `global_state`，不引入 callback 注入或更复杂的 RLlib trajectory postprocess。
- 旧 GRPO 链路被视为待替换目标，但在 MAPPO 主链稳定前不提前删除源码，以免失去回退与对照能力。
