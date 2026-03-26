# AGENTS.md — 项目总纲

## 项目名称
面向紧急危险场景的自动驾驶编队端到端决策规划

## 文档结构
本文件是**精简索引**。每个 Phase 的完整任务说明、接口签名、验收指标在独立文件中：

| Phase | 文件 | 状态 | 摘要 |
|-------|------|------|------|
| 0 | [`docs/phases/phase0.md`](docs/phases/phase0.md) | ✅ 完成 | 接口冻结 |
| 1 | [`docs/phases/phase1.md`](docs/phases/phase1.md) | ✅ 完成 | 编队环境 (1.1~1.7✅) |
| 2 | [`docs/phases/phase2.md`](docs/phases/phase2.md) | ✅ 完成 | 数据采集 (2.1~2.4✅) |
| 3 | [`docs/phases/phase3.md`](docs/phases/phase3.md) | ☐ 待开始 | 单车训练（已完成，非 Codex） |
| 4 | [`docs/phases/phase4.md`](docs/phases/phase4.md) | ✅ 完成 | 编队 Planner (4.0~4.4✅) |
| 5 | [`docs/phases/phase5.md`](docs/phases/phase5.md) | ✅ 完成 | RL 微调基础版 (5.1~5.10✅) |
| 5v2 | [`docs/phases2/overview.md`](docs/phases2/overview.md) | ✅ 完成 | RL 微调升级版 (5v2.1~5v2.8✅) |
| 6 | [`docs/phases/phase6.md`](docs/phases/phase6.md) | ☐ 待开始 | 实验评估 |
| 7 | [`docs/phases/phase7.md`](docs/phases/phase7.md) | ☐ 待开始 | 论文材料 |

**Codex 执行规则**：每次只读取并执行**一个 Phase 文件**。完成后更新本文件中的状态列。

---

## 一、项目目标
1. 3~5 车编队决策规划系统（MetaDrive 仿真）
2. 危险工况下编队安全通行、协作避障、队形恢复
3. 优于从零训练多车 RL 和分层式编队方法
4. 可支撑 SCI 论文的完整实验链
5. 闭环推理时延 ≤ 100ms

## 二、方法总路线

**阶段一：单车扩散规划预训练** → **阶段二：编队闭环强化微调（升级版）**

- 模型：复用 `V2TransfuserModel`（ResNet+Transformer BEV + DDIMScheduler）
- 编队方案：3 车共享权重 + `RelationEncoder(MLP)` + `MA-GRPO(CTDE)`
- 控制层：LQR(纵向) + PD(横向)，封装在环境内部

### Phase 5v2 升级要点（已实施）

| 升级项 | 原方案 | 升级后 |
|--------|--------|--------|
| 正则化 | IL loss（行为克隆） | ref-reg loss（BDPO-style x₀ 空间 L2，β 退火） |
| Advantage | 单层全局归一化 | Intra-anchor per-anchor 归一化 + 分层 advantage |
| 训练闭环 | 半闭环（surrogate reward） | 全闭环（联合组 + ClosedLoopExecutor） |

---

## 三、核心接口定义（冻结）

### formation_relation_state（12 维）
2 邻居 × 6：Δx, Δy, Δheading, Δv, Δx_target（**目标值**）, Δy_target
拼接到 status[8] → 共 20 维 → `_status_encoding(20, tf_d_model)`

### PlatoonEnv action 空间
- 轨迹模式：`{agent_id: np.ndarray(8,3)}` → LQR+PD 内部转控制
- 低层模式：`{agent_id: np.ndarray(2,)}` → 直接 steer/throttle

### 奖励公式
```
# 单车局部 reward
r_t = 1.0·(Δs/Δs_max) + 0.5·(-formation_err/d_norm) + 0.3·(-safety_penalty)
    + collision(-10) + road(-5) + 0.1·(-comfort)

# 联合 team reward（Phase 5v2 新增）
r_team = w_formation·(-avg_formation_err/d_norm)
       + w_safety·(-collision_penalty if any_crash else 0)
       + w_efficiency·(avg_progress/delta_s_max)

# 分层 advantage
A_final[i,k,g,t] = λ_local × γ^(T_d-t) × A_local[i,k,g]
                 + λ_team  × γ^(T_d-t) × A_team[i,m]
# 默认: λ_local=0.7, λ_team=0.3, γ=0.8
```

---

## 四、Phase 依赖关系

```
Phase 0
    ├── Phase 1（编队环境）──┐
    ├── Phase 2（数据采集）──┼──→ Phase 4（编队 Planner）──→ Phase 5──→ Phase 5v2──→ Phase 6 ──→ Phase 7
    └── Phase 3（单车训练）──┘
```

**跨 Phase 数据流**：
```
P1 env obs (multimodal) ────→ P4 planner.forward()
P1 info dict ───────────────→ P5/5v2 compute_step_reward() / compute_team_reward()
  必含: progress, jerk, delta_steering, formation_error, min_gap, crash, arrive_dest, out_of_road
P1 get_state/set_state ─────→ P5v2 ClosedLoopExecutor（闭环执行恢复）
P3 best.ckpt ──────────────→ P4 weight_migration
P4 planner ─────────────────→ P5/5v2 trainer.model
P5v2 checkpoints ───────────→ P6 evaluation
```

**Phase 5v2 内部数据流**：
```
collect_group_samples()           # sample G×M 轨迹 [G, M, 8, 3]
  └──→ reward_per_anchor [G, M]   # 每 anchor 独立评估
        └──→ compute_advantages() # per-anchor 归一化 → [G, M, step_num]

select_top_k_candidates()         # 每车选 top-K 候选 (g, k)
  └──→ build_joint_groups()       # 随机组合 M_joint 个联合组
        └──→ ClosedLoopExecutor   # 闭环执行，get_state/set_state 恢复
              └──→ compute_team_reward()   # formation + safety + efficiency
                    └──→ compute_team_advantages() # [G, M, step_num], 稀疏分配
                          └──→ compute_combined_advantages() # λ_local·A_local + λ_team·A_team
                                └──→ compute_rl_loss() + compute_ref_reg_loss()
```

---

## 五、训练模式

| 模式 | 命令 | 启用功能 | 适用阶段 |
|------|------|----------|----------|
| `toy-single` | `--mode toy-single` | ref-reg + intra-anchor（无联合组） | 快速验证 |
| `platoon` | `--mode platoon` | ref-reg + intra-anchor（无联合组） | 半闭环稳定性 |
| `platoon-closedloop` | `--mode platoon-closedloop` | 全部升级（联合组 + 闭环执行） | 完整训练 |

### 渐进式训练流程
```bash
# 阶段 1：toy 验证（< 5 分钟）
python train/train_platoon_rl.py --mode toy-single --steps 5 --render 0

# 阶段 2：半闭环
python train/train_platoon_rl.py \
  --mode platoon --config configs/train/platoon_grpo_v2.yaml --steps 500 --render 0

# 阶段 3：全闭环（完整升级）
python train/train_platoon_rl.py \
  --mode platoon-closedloop --config configs/train/platoon_grpo_v2.yaml --steps 500 --render 0
```

---

## 六、环境与路径约定

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
# Python: /home/kong/anaconda3/envs/meta_drive/bin/python
```

### 关键 import
```python
from envs.platoon_env import PlatoonEnv
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.relation_encoder import RelationEncoder
from models.platoon.weight_migration import migrate_single_to_platoon
from models.diffusion.diffusion_rl_scheduler import DiffusionRLScheduler
from train.ma_grpo_trainer import MultiAgentGRPOTrainer
from train.closedloop_executor import ClosedLoopExecutor
from train.joint_group import select_top_k_candidates, build_joint_groups, extract_joint_trajectories
from evaluation.reward_terms import compute_trajectory_reward, compute_team_reward
```

### 默认路径
| 用途 | 路径 |
|------|------|
| 数据集 | `$DATA_DIR` 默认 `/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets` |
| 单车 checkpoint | `/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt` |
| 编队 checkpoint | `checkpoints/platoon_rl/` |
| plan anchor | `metadrive/exp_dataset/metadrive_anchors_ppo.npy` |
| 训练配置（v2） | `configs/train/platoon_grpo_v2.yaml` |
| 训练日志 | `logs/platoon_rl/`、`logs/platoon_closedloop_rl/`、`logs/toy_single_rl/` |

### 参考代码库
| 概念 | 路径 |
|------|------|
| log_prob + RL loss（intra-anchor GRPO） | `reference_libs/DiffusionDriveV2/.../diffusiondrivev2_model_rl.py` |
| Pathwise KL（BDPO ref-reg） | `reference_libs/flow-rl/flowrl/agent/offline/bdpo/bdpo.py` |
| Transfuser backbone | `reference_libs/DiffusionDriveV2/.../transfuser_backbone.py` |

### 新建脚本头部必加
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

---

## 七、硬件约束

| 项目 | 约束 |
|------|------|
| GPU | RTX 4080, 16GB |
| 推理时延 | ≤ 100ms（3 车总计） |
| Headless | 必须支持 `use_render=False` |
| 已知问题 | Panda3D 退出 segfault (139)，不影响逻辑 |

显存预算：backbone frozen + gradient checkpointing ≤ 15GB；G=2, M=3 anchor 约 8~10GB

---

## 八、目录结构

```
Diffusion-metadrive/
├── AGENTS.md                              ← 本文件（精简索引）
├── configs/
│   └── train/
│       └── platoon_grpo_v2.yaml           ← 当前 RL 训练配置（ref-reg/分层adv/闭环/并行闭环）
├── docs/
│   ├── phases/phase0~7.md                 ← 各 Phase 完整说明
│   └── phases2/                           ← Phase 5v2 子任务说明
│       ├── overview.md                    ← 8 子任务总览
│       ├── task1_ref_reg.md               ← 参考策略正则
│       ├── task2_intra_anchor.md          ← Intra-anchor advantage
│       ├── task3_env_state.md             ← Env 状态保存/恢复
│       ├── task4_closedloop_exec.md       ← 闭环执行器
│       ├── task5_joint_group.md           ← 联合组 + team reward
│       ├── task6_layered_advantage.md     ← 分层 advantage 合并
│       ├── task7_train_entry.md           ← 训练入口升级
│       └── task8_integration_test.md      ← 集成冒烟测试
├── envs/
│   └── platoon_env.py                     ← 编队环境（含 get/set_state）
├── evaluation/
│   ├── platoon_metrics.py
│   └── reward_terms.py                    ← compute_trajectory_reward + compute_team_reward
├── models/
│   ├── diffusion/
│   │   └── diffusion_rl_scheduler.py      ← sample/replay/ref log_prob
│   └── platoon/
│       ├── platoon_diffusion_planner.py
│       ├── relation_encoder.py
│       └── weight_migration.py
├── train/
│   ├── ma_grpo_trainer.py                 ← MultiAgentGRPOTrainer（含分层adv/ref-reg）
│   ├── train_platoon_rl.py                ← 训练入口（toy-single/platoon/platoon-closedloop）
│   ├── closedloop_executor.py             ← ClosedLoopExecutor（状态恢复式闭环执行）
│   └── joint_group.py                     ← 联合组构建与轨迹提取
├── tests/acceptance/
│   ├── test_phase1_task{1~7}.py
│   ├── test_phase2_task{1~4}.py
│   ├── test_phase4_task{0~4}.py
│   ├── test_phase5_task{1~10}.py + test_phase5_integration.py
│   └── test_phase5v2_task{1~7}.py + test_phase5v2_integration.py  ← Phase 5v2 验收
├── metadrive/exp_dataset/                 ← 专家数据与 anchor
├── scenarios/hazard_scenarios.py
├── scripts/run_acceptance.sh              ← 统一验收入口
├── logs/ outputs/ checkpoints/
```

---

## 九、Codex 执行原则

### 总原则
1. 最小可运行版本优先
2. 每阶段先跑通再扩功能
3. 不同时重构环境、模型、训练器
4. 所有改动配置化，不写死魔法数
5. 每阶段留可运行脚本和日志

### 允许
项目骨架、数据采集、模型实现、训练脚本、评估器、可视化、配置文件、论文材料

### 不允许擅自
改研究定义、扩大到全栈感知、加通信机制、换大模型、无对照堆 trick

### 工作方式
每个任务输出：修改目标、涉及文件、设计选择、最小测试、潜在风险

---

## 十、验收框架

### 规则
1. 每个子任务完成后**必须运行验收命令**，输出粘贴到 commit message
2. 验收脚本在 `tests/acceptance/test_phase{N}_task{M}.py`（5v2 系列为 `test_phase5v2_task{M}.py`）
3. 通过 `pytest tests/acceptance/test_phase{N}_task{M}.py -v` 运行
4. **不允许 mock 核心逻辑**来通过验收（参见 5v2.6 criterion 6 的教训）
5. 阈值是**硬性要求**
6. 统一入口：`bash scripts/run_acceptance.sh [phase_number]`

### 验收完成后
更新本文件状态表中对应 Phase 的状态列（☐ → 🔶 → ✅）

---

## 十一、基线与对照（Phase 6 需实现）

### 必做对照
1. 单车预训练 + 编队 RL vs 从零训练多车 (MAPPO)
2. 端到端合作 vs 分层合作 (PPO高层 + IDM低层)
3. 扩散 planner vs 非扩散 planner
4. 多模态 vs 去掉部分模态
5. 危险工况训练 vs 无危险工况训练

### Phase 5v2 消融实验（新增）
- ref-reg loss vs IL loss（β 退火 vs 固定权重）
- intra-anchor advantage vs 全局归一化
- λ_team=0 vs λ_team=0.3（有无联合组信号）
- 闭环执行 vs 半闭环（surrogate reward）

---

## 十二、成功标准

**最小成功**：编队环境跑通 + 单车 planner 稳定 + RL 微调流程跑通
**方法成功**：优于从零训练和分层框架，有量化证据
**论文成功**：主表 + 消融表 + 泛化实验 + 失败分析 + 清晰证据链

---

## 最终提醒
项目关键是形成清晰的**研究证据链**：
- 单车预训练为什么有用
- 端到端合作为什么优于分层
- 扩散规划为什么适合危险工况编队
- 开环到闭环的能力提升是否成立
- **ref-reg vs IL**：正则化方式对 pre-trained policy 保留的影响
- **intra-anchor GRPO**：多模态 anchor 的利用效率与 mode collapse 的关系

Codex 的所有实现都应服务于这条证据链。
