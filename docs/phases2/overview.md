# Phase 5v2：编队闭环强化微调（升级版）

> **前置依赖**：Phase 5 所有子任务已完成（reward_terms、DiffusionRLScheduler、MultiAgentGRPOTrainer、train_platoon_rl.py 均已存在）
> **目标**：在现有 MA-GRPO 基础上，实施三项核心升级：
> 1. 参考策略正则替代 IL loss
> 2. Intra-anchor 分层 advantage + 多车联合 advantage
> 3. 半闭环 → 闭环训练

## 总体架构变更

```
现有架构：
  sample G 条（mode=0）→ surrogate reward → 单层 advantage → RL + IL loss

升级架构：
  sample G 条 × M anchors → intra-anchor local reward
  → 局部 top-K 组合为联合组 → 闭环执行 → team reward
  → 分层 advantage（local + team）
  → RL loss + ref-reg loss（替代 IL loss）
```

## 子任务列表

| 编号 | 任务 | 改动文件 | 依赖 | 文档 |
|------|------|----------|------|------|
| 5v2.1 | 参考策略正则（ref-reg loss） | `ma_grpo_trainer.py`, `diffusion_rl_scheduler.py` | 无 | [task1_ref_reg.md](task1_ref_reg.md) |
| 5v2.2 | Intra-anchor advantage | `ma_grpo_trainer.py`, `diffusion_rl_scheduler.py` | 无 | [task2_intra_anchor.md](task2_intra_anchor.md) |
| 5v2.3 | Env 状态保存/恢复 | `envs/platoon_env.py` | 无 | [task3_env_state.md](task3_env_state.md) |
| 5v2.4 | 闭环轨迹执行器 | `train/closedloop_executor.py`（新）, `envs/platoon_env.py` | 5v2.3 | [task4_closedloop_exec.md](task4_closedloop_exec.md) |
| 5v2.5 | 联合组构建与 team reward | `train/joint_group.py`（新）, `evaluation/reward_terms.py` | 5v2.4 | [task5_joint_group.md](task5_joint_group.md) |
| 5v2.6 | 分层 advantage 合并 | `ma_grpo_trainer.py` | 5v2.2, 5v2.5 | [task6_layered_advantage.md](task6_layered_advantage.md) |
| 5v2.7 | 训练入口升级 + 配置 | `train/train_platoon_rl.py`, `configs/train/platoon_grpo_v2.yaml` | 5v2.1-6 | [task7_train_entry.md](task7_train_entry.md) |
| 5v2.8 | 集成冒烟测试 | `tests/acceptance/test_phase5v2_integration.py`（新） | 5v2.7 | [task8_integration_test.md](task8_integration_test.md) |

## 实施顺序

```
阶段一（可并行）：
  5v2.1 ref-reg loss ──┐
  5v2.2 intra-anchor ──┤── 合并后可运行单车 RL 验证
  5v2.3 env state ─────┘

阶段二（串行）：
  5v2.4 闭环执行器 → 5v2.5 联合组 + team reward → 5v2.6 分层 advantage

阶段三：
  5v2.7 训练入口升级 → 5v2.8 集成测试
```

## 关键设计决策

1. **不引入 intent reward**：通过 intra-anchor 归一化保护 anchor 多样性（与 DiffusionDriveV2 一致），不额外惩罚偏离 anchor
2. **联合组采样策略**：局部 top-K（K=2）+ 随机组合 M 组（M=8），而非穷举
3. **ref-reg 在 x_0 空间计算**：当前模型用 sample prediction，直接在预测轨迹空间求 L2
4. **渐进式解冻**：阶段一只解冻 diff_decoder + RelationEncoder + status_encoding，阶段二考虑 LoRA
5. **team advantage 初始权重 λ_team=0.3**：避免多车信号在训练初期主导

## 硬件约束

- RTX 4080 16GB
- 推理 ≤ 100ms（3 车总计）
- 闭环执行 M=8 联合组 × 8 步 ≈ 额外 ~2s/GRPO step（可接受）
- 显存预算：backbone frozen + gradient checkpointing ≤ 15GB

## 参考代码索引

| 功能 | 文件 | 关键行 |
|------|------|--------|
| Intra-anchor advantage | `reference_libs/DiffusionDriveV2/.../diffusiondrivev2_model_rl.py` | lines 886-932 |
| Pathwise KL | `reference_libs/flow-rl/flowrl/agent/offline/bdpo/bdpo.py` | lines 55-60, 155-165 |
| DDIMScheduler log_prob | `reference_libs/DiffusionDriveV2/.../diffusiondrivev2_model_rl.py` | lines 540-676 |
