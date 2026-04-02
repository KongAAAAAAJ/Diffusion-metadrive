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

**阶段一：单车扩散规划预训练** → **阶段二：结构化候选池上的编队意图 selector 强化训练（RLlib MAPPO）** → **阶段三：条件化 trajectory refinement 局部细化**

- 模型：复用 `V2TransfuserModel`（ResNet+Transformer BEV + DDIMScheduler）
- 候选池：由单车行为基元与编队协作基元共同组成，避免仅依赖 k-means 压缩后的高频模式
- 编队方案：3 车共享 frozen planner + shared selector actor + centralized critic
- 细化方案：selector 后接条件化 trajectory refinement，仅做意图邻域内的局部连续修正
- 控制层：LQR(纵向) + PD(横向)，封装在环境内部

### 当前默认 RL 架构（Phase 6）

| 模块 | 当前默认实现 |
|------|--------------|
| Planner | 冻结 `PlatoonDiffusionPlanner`，按车导出 `K` 条多模态候选轨迹和 `mode_embeddings` |
| Candidate Pool | 组织为“单车行为基元 + 编队协作基元”双库，显式覆盖让行、压缩队形、协同汇入、局部脱队、重组恢复等协作意图 |
| Policy | RLlib MAPPO shared selector actor，动作空间为 `Discrete(K)`，负责高层意图选择 |
| Critic | centralized critic，仅消费 `global_state` |
| Refiner | GRPO 闭环 trajectory refinement：对已选轨迹加截断噪声（t=8/1000）生成 G=4 条变体，经 10 步 DDIM 去噪得到 refined 候选，用 `get_state/set_state` 分支模拟闭环打分，组内标准化 advantage 后通过 policy gradient 更新 trajectory head 的 `diff_decoder` |
| 环境包装 | `SelectorPlatoonEnv` 将 selector 的离散 intent 映射为 `{agent_id: np.ndarray(8,3)}` 轨迹动作；refinement 阶段在 step() 内对已选轨迹做 G 组分支 rollout 打分后执行最优 refined trajectory |
| 训练入口 | `train/train_selector.py` |

---

## 三、核心接口定义（冻结）

### formation_relation_state（12 维）
2 邻居 × 6：Δx, Δy, Δheading, Δv, Δx_target（**目标值**）, Δy_target
拼接到 status[8] → 共 20 维 → `_status_encoding(20, tf_d_model)`

### PlatoonEnv action 空间
- 轨迹模式：`{agent_id: np.ndarray(8,3)}` → LQR+PD 内部转控制
- 低层模式：`{agent_id: np.ndarray(2,)}` → 直接 steer/throttle

### 奖励公式（当前 selector MAPPO 默认配置）
```
# 单车局部 reward
r_local = 0.8·(Δs/Δs_max) + 1.0·(-formation_err/d_norm) + 0.8·(-safety_penalty)
        + collision(-10) + road(-5) + 0.05·(-comfort)

# 联合 team reward
r_team = 1.0·(-avg_formation_err/d_norm)
       + 1.5·(-team_collision_penalty)
       + 0.2·(avg_progress/delta_s_max)

# selector 最终 reward
r_selector = 0.45·r_local + 0.55·r_team
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

**当前 Phase 6 selector 训练数据流**：
```
env.reset()/env.step()
  └──→ frozen PlatoonDiffusionPlanner.forward_selector()
        └──→ trajectory_candidates [K, 8, 3] + mode_embeddings [K, D]
              └──→ selector actor 采样离散 intent
                    └──→ SelectorPlatoonEnv 将 intent 映射回轨迹动作
                          └──→ PlatoonEnv 闭环执行
                                └──→ compute_step_reward() + compute_team_reward()
                                      └──→ RLlib MAPPO 更新 shared actor + centralized critic
```

**扩展版 selector + GRPO refinement 数据流**：
```
env.reset()/env.step()
  └──→ frozen PlatoonDiffusionPlanner.forward_selector()
        └──→ K 条候选轨迹 + mode_embeddings
              └──→ selector actor 采样离散 intent → τ_selected
                    └──→ GRPO refinement（三车并行，互不感知对方 refinement 结果）
                          ├── 截断噪声 t=8/1000 → G=4 条变体
                          ├── 10 步 DDIM 去噪 → G 条 refined 轨迹
                          ├── get_state/set_state 分支模拟 → G 个 reward
                          ├── 组内标准化 advantage（正向 clamp + 安全约束）
                          └── 执行 best refined trajectory
                                └──→ step/team reward
                                      ├──→ RLlib MAPPO 更新 selector（阶段 2）
                                      └──→ GRPO policy gradient 更新 diff_decoder（阶段 3）
```

**Refinement 设计要点**：
1. **单一意图内微调**：refinement 不改变 selector 已选意图的语义，仅在截断噪声（t=8，α≈0.992）范围内修正速度、曲率、间距、时序等几何细节，典型位移调整量 0.3~1.0m。
2. **三车并行 refine**：各车独立 refine，不需要链式传递。车间 refinement 交叉影响为二阶小量（~0.03m/step），远小于 refinement 自身调整幅度（3~10% 偏差），不影响训练收敛。
3. **分支模拟打分**：利用 `PlatoonEnv.get_state/set_state` 对 G 条变体做真闭环 rollout 打分，而非启发式 proxy，保证 reward 信号准确。
4. **只解冻 diff_decoder**：backbone、encoder、selector 全冻结，仅 trajectory head 的 diff_decoder 参数有梯度，显存增量可控。
5. **分阶段训练**：先 selector MAPPO 收敛 → 再冻结 selector 训练 refinement → 可选弱耦合联合微调。
6. 候选池设计必须优先保证协作意图覆盖性，refinement 不能替代上游候选缺失。

---

## 五、训练模式

| 入口 | 命令 | 启用功能 | 适用阶段 |
|------|------|----------|----------|
| `train_selector.py` | `--config ... --pretrained-ckpt ... --total-env-steps ...` | 冻结 planner + RLlib MAPPO selector 训练 | 默认训练 |
| `scripts/test_platoon_rl_ckpt.py` | `--checkpoint ... --config ...` | selector checkpoint 评估 | 后续评估 |

### 当前推荐训练流程
```bash
# 阶段 1：单车扩散规划预训练
bash scripts/run_diffusion_train.sh

# 阶段 2：selector MAPPO smoke（320 env steps）
python -m train.train_selector \
  --config configs/train/platoon_mappo_smoke.yaml --pretrained-ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt --total-env-steps 320

# 阶段 3：selector MAPPO 正式训练（示例 8000 env steps）
python -m train.train_selector \
  --config configs/train/selector.yaml --pretrained-ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt --total-env-steps 8000

# 阶段 4：冻结 selector，GRPO 训练 trajectory refinement（5000~10000 env steps）
python -m train.train_selector \
  --config configs/train/selector.yaml --pretrained-ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt --total-env-steps 5000

# 阶段 5：selector + refinement 弱耦合联合微调（规划中）
# python -m train.train_selector \
#   --config configs/train/selector.yaml --pretrained-ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt --total-env-steps 3000
```

`train/train_selector.py` 和 `scripts/run_marl_train.sh` 里的 `--total-env-steps`，以及 `scripts/run_train.sh` 里的 `RL_STEPS`，统一表示**总环境步数（total env steps）**，不是 RLlib training iteration。训练入口会按 `rollout_fragment_length × num_rollout_workers × num_envs_per_worker` 自动换算 iteration 数。

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
from envs.selector_platoon_env import SelectorPlatoonEnv
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.relation_encoder import RelationEncoder
from models.platoon.weight_migration import migrate_single_to_platoon
from models.selector.intent_selector import IntentSelectorActor, IntentSelectorCritic
from models.platoon.trajectory_refiner import TrajectoryRefiner
from models.diffusion.diffusion_rl_scheduler import DiffusionRLScheduler
from train.train_selector import run_training
from evaluation.reward_terms import compute_step_reward, compute_team_reward
```

### 默认路径
| 用途 | 路径 |
|------|------|
| 数据集 | `$DATA_DIR` 默认 `/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets` |
| 单车 checkpoint | `/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt` |
| 编队 checkpoint | `/media/kong/Elements_SE/Diffusion_Data/outputs/selector/run_x/checkpoints` |
| plan anchor | `metadrive/exp_dataset/anchors.npy` |
| 训练配置 | `configs/train/selector.yaml` / `configs/train/platoon_mappo_smoke.yaml` |
| 训练日志 | `/media/kong/Elements_SE/Diffusion_Data/outputs/selector/run_x/tb` |

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

显存预算：backbone frozen + gradient checkpointing ≤ 15GB；selector 阶段约 4.5GB；refinement 阶段（G=4 分支 + diff_decoder 梯度）约 10~12GB

---

## 八、目录结构

```
Diffusion-metadrive/
├── AGENTS.md                              ← 本文件（精简索引）
├── configs/
│   └── train/
│       ├── selector.yaml                  ← selector MAPPO 正式训练配置
│       ├── platoon_mappo_smoke.yaml       ← selector MAPPO 冒烟训练配置
│       └── platoon_selector_refine.yaml   ← GRPO refinement 训练配置
├── docs/
│   ├── phases/phase0~7.md                 ← 各 Phase 完整说明
│   ├── phases2/                           ← Phase 5v2 子任务说明
│   └── phases6/
│       ├── mappo_selector_plan.md         ← selector MAPPO 训练链设计
│       └── grpo_refinement_plan.md        ← GRPO trajectory refinement 实现计划
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
│   │   └── diffusion_rl_scheduler.py      ← sample/replay/ref log_prob（从 git 恢复）
│   ├── platoon/
│   │   ├── platoon_diffusion_planner.py
│   │   ├── trajectory_refiner.py          ← GRPO refinement 封装（截断噪声 + DDIM + advantage）
│   │   ├── relation_encoder.py
│   │   └── weight_migration.py
│   └── selector/                          ← intent selector actor/critic 与 RLlib adapter
├── train/
│   ├── train_selector.py                  ← selector MAPPO 主训练入口
│   ├── selector_callbacks.py              ← RLlib 自定义指标回调
│   └── selector_callbacks.py              ← RLlib 自定义指标回调
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

### Phase 6 当前建议对照
- frozen planner + selector MAPPO vs 从零训练多车 MAPPO
- 真实 planner candidates vs 去掉 multimodal selector（单一 best-mode）
- centralized critic vs 去中心化 critic
- team-heavy reward (`lambda_team>lambda_local`) vs local-heavy reward
- selector + GRPO refinement vs selector only（验证 refinement 增益）
- GRPO refinement（G=4 组内比较）vs 直接执行已选轨迹（无 refinement baseline）

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
- 冻结 planner + 训练 selector 是否比直接微调整条 diffusion policy 更稳定
- 多模态 candidates + selector 是否真的带来更好的编队协同行为
- GRPO 组内比较在意图内微调是否比直接执行 planner 候选更适配编队场景

Codex 的所有实现都应服务于这条证据链。
