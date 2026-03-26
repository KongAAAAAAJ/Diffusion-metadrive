# 面向紧急危险场景的自动驾驶编队端到端决策规划

## 项目简介
本项目面向 MetaDrive 仿真环境中的紧急危险场景，研究并实现一个 3~5 车自动驾驶编队的端到端决策与规划系统。整体路线分为两个阶段：先进行单车扩散式轨迹规划预训练，再在多车编队场景中进行闭环强化微调，使系统能够在危险工况下完成安全通行、协作避障和队形恢复。

该仓库的目标不只是“跑通一个模型”，而是形成一条可复现实验链：从环境、数据、单车 planner、编队 planner、闭环 RL 微调，到最终实验评估与论文材料组织，均围绕研究证据链展开。

## 研究目标
- 构建 3~5 车编队决策规划系统，并在 MetaDrive 中闭环运行。
- 在危险工况下实现协作避障、编队穿行和队形恢复。
- 验证“单车扩散预训练 + 编队闭环 RL 微调”优于从零训练多车 RL 与分层编队方案。
- 为后续论文实验提供统一、可复现、可追踪的实现基础。
- 满足 3 车总推理时延不超过 100ms 的工程约束。

## 功能特性 / 核心能力
- 单车扩散轨迹规划预训练。
- 多模态观测输入，支持相机、LiDAR、状态特征与编队关系特征。
- 编队环境支持轨迹动作和低层控制动作两种模式。
- 编队关系建模，支持 `formation_relation_state` 注入 planner。
- 编队扩散 planner，支持从单车 checkpoint 迁移到多车共享权重架构。
- MA-GRPO 闭环强化微调，支持 ref-reg、intra-anchor advantage、team reward 与 layered advantage。
- 危险场景配置与 curriculum 基础设施。
- 按 Phase/Task 组织的 acceptance tests，便于逐阶段验收。

## 方法概述
### 总体路线
项目采用两阶段方法：

1. 单车扩散规划预训练。
2. 编队闭环强化微调。

对应关系为：

- 阶段一：使用 `V2TransfuserModel` 学习单车轨迹生成能力。
- 阶段二：在共享单车 backbone 的基础上，引入编队关系编码与多车协同强化学习。

### 核心模型组成
- Backbone：`V2TransfuserModel`，包含 ResNet + Transformer BEV + DDIMScheduler。
- 编队方案：多车共享权重 + `RelationEncoder` + `_status_encoding` 扩展。
- 训练算法：`MA-GRPO`，采用 CTDE 范式。
- 控制层：环境内置 LQR 纵向控制与 PD 横向控制。

### 关键接口
`formation_relation_state` 为 12 维特征，来自 2 个邻居的相对状态与目标相对位置，拼接到 `status[8]` 后形成 20 维输入。

`PlatoonEnv` 支持两类 action：
- 轨迹模式：`{agent_id: np.ndarray(8, 3)}`
- 低层模式：`{agent_id: np.ndarray(2,)}`

项目中的基础奖励形式为：

```text
r_t = 1.0·(Δs/Δs_max) + 0.5·(-formation_err/d_norm) + 0.3·(-safety_penalty)
    + collision(-10) + road(-5) + 0.1·(-comfort)
```

## 算法训练流程
### 阶段一：单车扩散规划预训练
1. 在 MetaDrive 中采集专家数据。
2. 从轨迹数据中抽取 plan anchors。
3. 训练单车 diffusion planner。
4. 获得单车最佳 checkpoint，作为编队 planner 的初始化来源。

### 阶段二：编队闭环强化微调
1. 将单车 planner 权重迁移到多车共享架构。
2. 采样多组、多 anchor 候选轨迹。
3. 计算 intra-anchor 局部 reward 与局部 advantage。
4. 构建 top-K 联合组并执行闭环轨迹。
5. 计算 team reward 与 layered advantage。
6. 用 RL loss + ref-reg loss 进行更新。

升级版 Phase 5v2 的训练结构可以概括为：

```text
sample G groups × M anchors
-> local reward / intra-anchor advantage
-> top-K joint group selection
-> closed-loop execution
-> team reward
-> combined advantage (local + team)
-> RL loss + ref-reg loss
```

## 环境要求
### 软件环境
- 操作系统：Linux
- Python 环境：建议使用仓库既有 `meta_drive` conda 环境
- 解释器路径：`/home/kong/anaconda3/envs/meta_drive/bin/python`
- 仓库根目录：`/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive`

### 硬件约束
- GPU：RTX 4080 16GB
- Headless：必须支持 `use_render=False`
- 推理时延目标：3 车总计不超过 100ms
- 已知问题：Panda3D 退出时可能出现 segfault 139，不影响核心逻辑

### 显存预算参考
- 单车推理：约 1.5GB
- 3 车推理：约 4.5GB
- MA-GRPO（`G=4`）：约 12GB

## 安装与环境配置
在仓库根目录执行：

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHON_BIN=/home/kong/anaconda3/envs/meta_drive/bin/python
```

若你直接使用项目约定环境，建议后续命令统一写成：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python <module_or_script>
```

## 数据准备与采集
### 1. 运行专家数据采集
项目默认数据目录为：

```bash
export OUTPUT_ROOT=/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets
```

运行默认单车专家采集：

```bash
bash scripts/run_dataset_collect.sh
```

该脚本支持通过环境变量覆盖关键参数，例如：

```bash
PYTHON_BIN=/home/kong/anaconda3/envs/meta_drive/bin/python \
TARGET_SAMPLES=50000 \
EXPERT_TYPE=idm \
COLLECTION_MODE=single \
START_SEED=10 \
LOW_TRAFFIC_DENSITY=0.08 \
HIGH_TRAFFIC_DENSITY=0.12 \
OUTPUT_ROOT=/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets \
bash scripts/run_dataset_collect.sh
```

若需要进行更广覆盖的路网采集，可切换为：

```bash
COLLECTION_MODE=phase2_plan bash scripts/run_dataset_collect.sh
```

### 2. 抽取 Plan Anchors
项目提供了专门的 anchor 抽取脚本：

```bash
bash scripts/run_abstract_anchors.sh
```

常见可复现写法如下：

```bash
PYTHON_BIN=/home/kong/anaconda3/envs/meta_drive/bin/python \
EXPERT_NAME=ppo \
NUM_ANCHORS=8 \
bash scripts/run_abstract_anchors.sh
```

默认输出位置通常为：
- `metadrive/exp_dataset/metadrive_anchors_ppo.npy`
- `metadrive/exp_dataset/metadrive_anchors_ppo.png`

## 单车扩散训练
### 1. 直接使用训练脚本
单车扩散训练的推荐入口为：

```bash
bash scripts/run_diffusion_train.sh
```

默认脚本会调用：
- 模块：`metadrive.policy.diffusion_policy.train_transfuser`
- 默认 model size：`small`
- 默认 dataset root：`/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_preprocessed`
- 默认 anchor 路径：`metadrive/exp_dataset/metadrive_anchors_ppo.npy`

例如：

```bash
PYTHON_BIN=/home/kong/anaconda3/envs/meta_drive/bin/python \
MODEL_SIZE=small \
EXPERT_NAME=ppo \
DATASET_ROOT=/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_preprocessed \
PLAN_ANCHOR_PATH=metadrive/exp_dataset/metadrive_anchors_ppo.npy \
MAX_EPOCHS=10 \
NUM_WORKERS=8 \
bash scripts/run_diffusion_train.sh
```

### 2. 其他相关脚本
以下脚本属于辅助或实验性质：
- `scripts/run_diffusion_preprocess.sh`：离线特征缓存实验脚本。
- `scripts/run_diffusion_convert_camera_layout.sh`：旧相机数据布局迁移脚本。
- `scripts/run_diffusion_open_loop_eval.sh`：开环评估脚本。
- `scripts/run_diffusion_test.sh`：测试脚本。

如果没有特殊需要，优先使用 `run_dataset_collect.sh`、`run_abstract_anchors.sh`、`run_diffusion_train.sh` 这三条主线入口。

## 编队 RL 微调
### 1. 推荐入口
编队 RL 的推荐入口是：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py \
  --mode platoon-closedloop \
  --config configs/train/platoon_grpo_v2.yaml \
  --steps 500 \
  --render 0
```

其中：
- `toy-single`：单车 toy 验证。
- `platoon`：3 车半闭环训练。
- `platoon-closedloop`：3 车闭环训练，启用联合组与 team reward。

### 2. 最小可运行命令
单车 toy 验证：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py \
  --mode toy-single \
  --config configs/train/platoon_grpo_v2.yaml \
  --steps 5 \
  --render 0
```

3 车半闭环：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py \
  --mode platoon \
  --config configs/train/platoon_grpo_v2.yaml \
  --steps 5 \
  --render 0
```

3 车闭环：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python train/train_platoon_rl.py \
  --mode platoon-closedloop \
  --config configs/train/platoon_grpo_v2.yaml \
  --steps 5 \
  --render 0
```

### 3. 配置文件
当前训练入口主要使用：
- `configs/train/platoon_grpo_v2.yaml`

当前训练入口默认使用 `platoon_grpo_v2.yaml`，用于 Phase 5v2 及之后的升级版闭环微调，包含：
- ref-reg 配置
- layered advantage 配置
- joint-group 配置
- freeze 策略
- reward config

### 4. 包装脚本说明
仓库中还存在：

```bash
bash scripts/run_marl_train.sh
```

但该脚本当前更像一个简化包装入口，参数较固定。对协作者和复现实验来说，优先建议直接调用 `train/train_platoon_rl.py`，因为它暴露的模式与配置更完整。

## 验收与测试
### 1. 运行单项 acceptance test
每个 Phase/Task 对应一个 pytest 验收脚本，例如：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5v2_task7.py -v
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5v2_integration.py -v
```

### 2. 统一入口
项目提供按阶段聚合的验收脚本：

```bash
bash scripts/run_acceptance.sh
```

只运行某一阶段：

```bash
bash scripts/run_acceptance.sh 5
```

按多个阶段运行：

```bash
bash scripts/run_acceptance.sh 4 5
```

### 3. 说明
- 所有子任务默认要求通过对应 `tests/acceptance/` 下的验收测试。
- 核心逻辑不允许依赖 mock 来“伪通过”验收。
- 完成阶段后应同步更新 `AGENTS.md` 中对应 Phase 状态。

## 项目结构
推荐从以下目录理解项目：

```text
Diffusion-metadrive/
├── AGENTS.md                         # 项目总纲、阶段状态、接口冻结与执行规则
├── README.md                         # 项目总览与复现入口
├── docs/
│   ├── phases/                       # Phase 0~7 的正式任务说明
│   └── phases2/                      # Phase 5v2 升级任务说明
├── envs/
│   └── platoon_env.py                # 编队环境、状态恢复、轨迹执行接口
├── scenarios/
│   └── hazard_scenarios.py           # 危险工况配置
├── evaluation/
│   ├── platoon_metrics.py            # 编队指标
│   └── reward_terms.py               # step reward / team reward
├── models/
│   ├── platoon/                      # 编队 planner、关系编码、权重迁移
│   └── diffusion/                    # diffusion RL scheduler 等
├── train/
│   ├── ma_grpo_trainer.py            # 多车 GRPO trainer
│   ├── closedloop_executor.py        # 闭环执行器
│   ├── joint_group.py                # 联合组构建
│   └── train_platoon_rl.py           # 编队 RL 训练入口
├── metadrive/
│   └── exp_dataset/                  # 数据采集、anchor 抽取等
├── scripts/                          # 运行脚本与辅助脚本
├── configs/                          # 训练配置
├── tests/acceptance/                 # 分阶段验收测试
├── checkpoints/                      # 训练权重输出
├── logs/                             # TensorBoard 与运行日志
└── outputs/                          # 训练摘要与实验输出
```

## 文档索引
### 核心文档
- [`AGENTS.md`](AGENTS.md)：项目总纲、接口定义、硬件约束、执行原则。
- [`docs/phases/phase0.md`](docs/phases/phase0.md) ~ [`docs/phases/phase7.md`](docs/phases/phase7.md)：各阶段正式任务文档。
- [`docs/phases2/overview.md`](docs/phases2/overview.md)：Phase 5v2 升级版 RL 微调概览。

### 当前阶段状态
根据 `AGENTS.md`：
- Phase 0：已完成
- Phase 1：已完成
- Phase 2：已完成
- Phase 3：单车训练已完成，但非由 Codex 实现
- Phase 4：已完成
- Phase 5：已完成
- Phase 6：待开始
- Phase 7：待开始

## 默认路径约定
- 数据集目录：`/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets`
- 单车 checkpoint：`checkpoints/single_vehicle/best.ckpt`
- 编队 checkpoint：`checkpoints/platoon_rl/`
- plan anchor：`metadrive/exp_dataset/metadrive_anchors.npy`
- Phase 5v2 anchor 常见路径：`metadrive/exp_dataset/metadrive_anchors_ppo.npy`

## 已知问题与注意事项
- Panda3D 退出时可能出现 segfault 139，通常不影响训练与测试逻辑。
- `use_render=False` 是默认推荐设置，尤其在服务器和批量实验中。
- 某些脚本仍保留历史兼容逻辑，研究复现时优先参考本 README 中列出的主入口命令。
- 如果文档与脚本行为发生冲突，应以当前仓库代码和 `AGENTS.md` 的约定为准。

## 研究证据链提醒
本项目的实现始终服务于以下研究问题：
- 单车预训练为什么有用。
- 端到端合作为什么优于分层合作。
- 扩散规划为什么适合危险工况编队。
- 从开环到闭环的能力提升是否成立。

如果你准备继续扩展 Phase 6 或 Phase 7，建议先阅读：

```bash
sed -n '1,260p' AGENTS.md
sed -n '1,260p' docs/phases/phase6.md
sed -n '1,260p' docs/phases/phase7.md
```
