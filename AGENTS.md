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
| 3 | [`docs/phases/phase3.md`](docs/phases/phase3.md) | ☐ 待开始 | 单车训练 |
| 4 | [`docs/phases/phase4.md`](docs/phases/phase4.md) | ☐ 待开始 | 编队 Planner |
| 5 | [`docs/phases/phase5.md`](docs/phases/phase5.md) | ☐ 待开始 | RL 微调 |
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
**阶段一：单车扩散规划预训练** → **阶段二：编队闭环强化微调**
- 模型：复用 `V2TransfuserModel`（ResNet+Transformer BEV + DDIMScheduler）
- 编队方案：3 车共享权重 + RelationEncoder(MLP) + MA-GRPO(CTDE)
- 控制层：LQR(纵向) + PD(横向)，封装在环境内部

## 三、核心接口定义（冻结）

### formation_relation_state（12 维）
2 邻居 × 6：Δx, Δy, Δheading, Δv, Δx_target(**目标值**), Δy_target
拼接到 status[8] → 共 20 维 → `_status_encoding(20, tf_d_model)`

### PlatoonEnv action 空间
- 轨迹模式：`{agent_id: np.ndarray(8,3)}` → LQR+PD 内部转控制
- 低层模式：`{agent_id: np.ndarray(2,)}` → 直接 steer/throttle

### 奖励公式
```
r_t = 1.0·(Δs/Δs_max) + 0.5·(-formation_err/d_norm) + 0.3·(-safety_penalty)
    + collision(-10) + road(-5) + 0.1·(-comfort)
```

---

## 四、Phase 依赖关系

```
Phase 0
    ├── Phase 1（编队环境）──┐
    ├── Phase 2（数据采集）──┼──→ Phase 4（编队 Planner）──→ Phase 5（RL）──→ Phase 6 ──→ Phase 7
    └── Phase 3（单车训练）──┘
```

**跨 Phase 数据流**：
```
P1 env obs (multimodal) ────→ P4 planner.forward()
P1 info dict ───────────────→ P5 compute_step_reward()
  必含: progress, jerk, delta_steering, formation_error, min_gap, crash, arrive_dest, out_of_road
P1 hazard_scenarios ────────→ P5 curriculum / P6 evaluation
P3 best.ckpt ──────────────→ P4 weight_migration
P4 planner ─────────────────→ P5 trainer.model
P5 checkpoints ─────────────→ P6 evaluation
```

---

## 五、环境与路径约定

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
# Python: /home/kong/anaconda3/envs/meta_drive/bin/python
```

### 关键 import
```python
from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv      # 编队基类
from metadrive.policy.diffusion_policy.transfuser_model_v2 import V2TransfuserModel
from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from metadrive.policy.idm_policy import IDMPolicy  # IDMPolicy(vehicle, random_seed)
from metadrive.policy.diffusion_policy.transfuser_features import (
    stitch_three_cameras, lidar_to_histogram, build_status_feature,
)
```

### 默认路径
| 用途 | 路径 |
|------|------|
| 数据集 | `$DATA_DIR` 默认 `/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets` |
| 单车 checkpoint | `checkpoints/single_vehicle/best.ckpt` |
| 编队 checkpoint | `checkpoints/platoon_rl/` |
| plan anchor | `metadrive/exp_dataset/metadrive_anchors.npy` |

### 参考代码库
| 概念 | 路径 |
|------|------|
| log_prob + RL loss | `reference_libs/DiffusionDriveV2/.../diffusiondrivev2_model_rl.py` |
| 扩散模型主体 | `reference_libs/DiffusionDriveV2/.../diffusiondrivev2_model_sel.py` |
| Transfuser backbone | `reference_libs/DiffusionDriveV2/.../transfuser_backbone.py` |

### 新建脚本头部必加
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

---

## 六、硬件约束

| 项目 | 约束 |
|------|------|
| GPU | RTX 4080, 16GB |
| 推理时延 | ≤ 100ms（3 车总计） |
| Headless | 必须支持 `use_render=False` |
| 已知问题 | Panda3D 退出 segfault (139)，不影响逻辑 |

显存预算：单车推理 1.5GB / 3 车推理 4.5GB / MA-GRPO(G=4) ≈ 12GB

---

## 七、推荐目录结构

```
Diffusion-metadrive/
├── AGENTS.md                          ← 本文件（精简索引）
├── docs/phases/                       ← 每个 Phase 的完整说明
│   ├── phase0.md ~ phase7.md
├── metadrive/                         ← 现有代码，勿大改
├── envs/platoon_env.py               ← Phase 1
├── evaluation/                        ← Phase 1 + 5
│   ├── platoon_metrics.py
│   └── reward_terms.py
├── scenarios/hazard_scenarios.py      ← Phase 1
├── models/                            ← Phase 4
│   ├── platoon/{relation_encoder, planner, weight_migration}.py
│   └── diffusion/diffusion_rl_scheduler.py
├── train/                             ← Phase 5
│   ├── ma_grpo_trainer.py
│   └── train_platoon_rl.py
├── tests/acceptance/                  ← 每个子任务的 pytest 验收
├── scripts/
├── configs/
├── tools/
├── logs/ outputs/ checkpoints/ paper/
```

---

## 八、Codex 执行原则

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

## 九、验收框架

### 规则
1. 每个子任务完成后**必须运行验收命令**，输出粘贴到 commit message
2. 验收脚本在 `tests/acceptance/test_phase{N}_task{M}.py`
3. 通过 `pytest tests/acceptance/test_phase{N}_task{M}.py -v` 运行
4. **不允许 mock 核心逻辑**来通过验收
5. 阈值是**硬性要求**
6. 统一入口：`bash scripts/run_acceptance.sh [phase_number]`

### 验收完成后
更新本文件状态表中对应 Phase 的状态列（☐ → 🔶 → ✅）

---

## 十、基线与对照（Phase 6 需实现）

### 必做对照
1. 单车预训练 + 编队 RL vs 从零训练多车 (MAPPO)
2. 端到端合作 vs 分层合作 (PPO高层 + IDM低层)
3. 扩散 planner vs 非扩散 planner
4. 多模态 vs 去掉部分模态
5. 危险工况训练 vs 无危险工况训练

### 消融实验（10 项）
去掉预训练 / 关系建模 / 图像 / LiDAR / 编队相对关系 / 危险 curriculum / IL vs IL+RL / 不同规模 / 不同 diffusion step / 不同奖励权重

---

## 十一、成功标准

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

Codex 的所有实现都应服务于这条证据链。
