# 面向紧急危险场景的自动驾驶编队端到端决策规划

## 项目简介
本项目面向 MetaDrive 仿真环境中的紧急危险场景，研究并实现一个 3~5 车自动驾驶编队的端到端决策与规划系统。整体路线分为两个阶段：先进行单车扩散式轨迹规划预训练，再在多车编队场景中构建“结构化行为基元候选池 + 共享参数意图 selector + 条件化 trajectory refinement”的分层决策链，使系统能够在危险工况下完成安全通行、协作避障和队形恢复。

该仓库的目标不只是“跑通一个模型”，而是形成一条可复现实验链：从环境、数据、单车 planner、编队 planner、闭环 RL 微调，到最终实验评估与论文材料组织，均围绕研究证据链展开。

## 研究目标
- 构建 3~5 车编队决策规划系统，并在 MetaDrive 中闭环运行。
- 在危险工况下实现协作避障、编队穿行和队形恢复。
- 验证“单车扩散预训练 + 编队闭环 RL 微调”优于从零训练多车 RL 与分层编队方案。
- 为后续论文实验提供统一、可复现、可追踪的实现基础。
- 满足 3 车总推理时延不超过 100ms 的工程约束。

## 功能特性 / 核心能力
- 单车扩散轨迹规划预训练。
- 结构化行为基元候选池，支持“单车行为基元 + 编队协作基元”双库组织。
- 多模态观测输入，支持相机、LiDAR、状态特征与编队关系特征。
- 编队环境支持轨迹动作和低层控制动作两种模式。
- 编队关系建模，支持 `formation_relation_state` 注入 planner。
- 编队扩散 planner，支持从单车 checkpoint 迁移到多车共享权重架构。
- RLlib MAPPO / value-guided selector 闭环强化训练，负责高层协作意图选择。
- 条件化 trajectory refinement 模块，负责在保持原始语义意图不变的前提下完成局部连续轨迹细化。
- 支持基于组级相对偏好的 refinement 优化接口（如 GRPO），用于细化同一高层意图下的局部几何与交互细节。
- 危险场景配置与 curriculum 基础设施。
- 按 Phase/Task 组织的 acceptance tests，便于逐阶段验收。

## 方法概述
### 总体路线
项目采用两阶段方法：

1. 单车扩散规划预训练。
2. 编队闭环强化训练：冻结基础 planner，训练 selector，并在后续阶段引入条件化 trajectory refinement。

对应关系为：

- 阶段一：使用 `V2TransfuserModel` 学习单车轨迹生成能力。
- 阶段二：冻结单车 diffusion planner，构建结构化行为基元候选池，导出多模态候选轨迹与 mode embedding，再用 RLlib MAPPO 训练共享的意图 selector。
- 阶段三：在 selector 输出高层意图后，引入条件化 trajectory refinement 模块，在意图语义不变的前提下进行局部轨迹细化，并通过组级偏好优化进一步提升细节质量。

### 核心模型组成
- Backbone：`V2TransfuserModel`，包含 ResNet + Transformer BEV + DDIMScheduler。
- 候选池：由单车行为基元与编队协作基元共同组成，避免仅依赖 k-means 压缩后的高频模式。
- 编队方案：多车共享权重 + `RelationEncoder` + `_status_encoding` 扩展。
- Selector：基于 `RLlib MAPPO` / value-guided 思路进行高层协作意图选择，采用 shared actor + centralized critic 的 CTDE 范式。
- Refinement：条件化轨迹细化模块，输入已选轨迹、编队上下文和邻车摘要，输出受限残差修正。
- 控制层：环境内置 LQR 纵向控制与 PD 横向控制。

### 关键接口
`formation_relation_state` 为 12 维特征，来自 2 个邻居的相对状态与目标相对位置，拼接到 `status[8]` 后形成 20 维输入。

`PlatoonEnv` 支持两类 action：
- 轨迹模式：`{agent_id: np.ndarray(8, 3)}`
- 低层模式：`{agent_id: np.ndarray(2,)}`

项目中的 selector 奖励由“单车局部 shaping + 团队协同 shaping”组成：

```text
r_local = 0.8·(Δs/Δs_max) + 1.0·(-formation_err/d_norm) + 0.8·(-safety_penalty)
        + collision(-10) + road(-5) + 0.05·(-comfort)

r_team  = 1.0·(-avg_formation_err/d_norm)
        + 1.5·(-team_collision_penalty)
        + 0.2·(avg_progress/delta_s_max)

r_selector = 0.45·r_local + 0.55·r_team
```

## 算法训练流程
### 阶段一：单车扩散规划预训练
1. 在 MetaDrive 中采集专家数据。
2. 从轨迹数据中抽取 plan anchors。
3. 训练单车 diffusion planner。
4. 获得单车最佳 checkpoint，作为编队 planner 的初始化来源。

### 阶段二：编队意图 selector 强化训练
1. 将单车 planner 权重迁移到多车共享架构。
2. 冻结 planner，按车导出 `K` 条多模态候选轨迹、mode logits 与 mode embeddings。
3. 在候选组织上引入“单车行为基元 + 编队协作基元”双库结构，尽量覆盖让行、压缩队形、协同汇入、局部脱队、重组恢复等关键协作意图。
4. 共享 selector actor 基于 compact context、编队关系特征和 mode embeddings 输出离散意图分布。
5. 选中的 intent 被映射回 `[8, 3]` 轨迹动作，并在 `PlatoonEnv` 中闭环执行。
6. RLlib MAPPO 用 `shared actor + centralized critic` 更新 selector。

### 阶段三：GRPO 闭环 trajectory refinement
1. Selector 选出高层意图后，对已选轨迹加**截断噪声**（t=8/1000，α≈0.992）生成 G=4 条变体。
2. 经 10 步 DDIM 去噪得到 G 条 refined 轨迹，典型位移调整量 0.3~1.0m，不改变意图语义。
3. 利用 `PlatoonEnv.get_state/set_state` 对 G 条变体做**真闭环分支模拟**打分（`compute_step_reward + compute_team_reward`）。
4. 组内标准化 advantage（正向 clamp + 碰撞/出路安全约束），通过 policy gradient 更新 trajectory head 的 `diff_decoder`。
5. **三车并行 refine**，互不感知对方 refinement 结果——车间交叉影响为二阶小量（~0.03m/step vs 0.3~1.0m 调整量，偏差 3~10%），不影响训练收敛。
6. 训练分阶段：先 selector MAPPO 收敛（~8000 env steps）→ 冻结 selector 训练 refinement（~5000-10000 env steps）→ 可选弱耦合联合微调。
7. 推理时只做 1 次 refinement（不需要 G 组比较），增加 ~20ms 延迟，总延迟仍在 100ms 内。

当前推荐训练结构可以概括为：

```text
current obs
-> frozen PlatoonDiffusionPlanner.forward_selector()
-> K 条候选轨迹 + mode_embeddings
-> shared selector actor samples one discrete intent per vehicle
-> selected intent trajectory τ_selected
-> GRPO refinement（三车并行）:
   -> 截断噪声 t=8 → G=4 条变体
   -> 10 步 DDIM 去噪 → G 条 refined 轨迹
   -> get_state/set_state 分支模拟 → G 个 reward
   -> 组内标准化 advantage → policy gradient 更新 diff_decoder
-> 执行 best refined trajectory in PlatoonEnv
-> RLlib MAPPO updates selector actor/critic (阶段 2)
-> GRPO policy gradient updates diff_decoder (阶段 3)
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
- 单车 planner 推理：约 1.5GB
- 3 车 frozen planner + selector rollout：约 4.5GB
- GRPO refinement 阶段（G=4 分支 + diff_decoder 梯度）：约 10~12GB
- RLlib MAPPO 训练峰值取决于 worker 数、fragment length 和 batch size

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

## 地图导出
如果你想把 `BaseMultiEnv` 当前生成的地图保存成一张俯视 PNG，可以直接运行：

```bash
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
/home/kong/anaconda3/envs/meta_drive/bin/python scripts/save_base_multi_env_map.py \
  --output /tmp/base_multi_env_map.png \
  --resolution 1024 \
  --start-seed 1 \
  --num-scenarios 1 \
  --use-hybrid-map 1 \
  --hybrid-map-sequence SSXCOCSS
```

说明：
- 输出的是 top-down map image，不是 3D 渲染截图。
- 脚本会先 `reset()` 生成地图，再读取 `env.current_map` 调用 `draw_top_down_map()` 保存。
- 可通过 `--map` 覆盖默认 map 配置，或通过 `--hybrid-map-sequence` 指定固定 hybrid 序列。

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
NUM_ANCHORS=9 \
bash scripts/run_abstract_anchors.sh
```

当前 anchor 抽取已切换为“7 类语义分桶 + 桶内 K-Medoids + 真实 medoid”流程。
注意：该流程要求输入 shard 已包含语义判定字段，需使用补采字段后的新数据重新采集。

默认输出位置通常为：
- `expert_dataset/metadrive_anchors_ppo.npy`
- `expert_dataset/metadrive_anchors_ppo.png`

## 单车扩散训练
### 1. 直接使用训练脚本
单车扩散训练的推荐入口为：

```bash
bash scripts/run_diffusion_train.sh
```

默认脚本会调用：
- 模块：`models.diffusion.train_transfuser`
- 默认 model size：`small`
- 默认 dataset root：`/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_preprocessed`
- 默认 anchor 路径：`expert_dataset/metadrive_anchors_ppo.npy`

例如：

```bash
PYTHON_BIN=/home/kong/anaconda3/envs/meta_drive/bin/python \
MODEL_SIZE=small \
EXPERT_NAME=ppo \
DATASET_ROOT=/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_preprocessed \
PLAN_ANCHOR_PATH=expert_dataset/metadrive_anchors_ppo.npy \
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

## 编队 RL 微调与轨迹细化
### 1. 推荐入口
编队 RL 的推荐入口是：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m train.train_selector \
  --config configs/train/selector.yaml \
  --pretrained-ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt \
  --total-env-steps 8000
```

其中：
- `train/train_selector.py` 是 selector MAPPO 的唯一主训练入口。
- `--pretrained-ckpt`：冻结单车预训练模型路径。
- `--total-env-steps`：总环境步数，不是 RLlib iteration。

GRPO refinement 训练示例：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m train.train_selector \
  --config configs/train/selector.yaml \
  --pretrained-ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt \
  --total-env-steps 5000
```

### 2. 最小可运行命令
3 车 selector 训练 dry-run：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m train.train_selector \
  --config configs/train/selector.yaml \
  --dry-run
```

3 车 selector 训练：

```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m train.train_selector \
  --config configs/train/selector.yaml \
  --pretrained-ckpt /media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt \
  --total-env-steps 320
```

注意：当前 `train/train_selector.py` 和 `scripts/run_marl_train.sh` 统一使用 `--total-env-steps`，`scripts/run_train.sh` 里的 `RL_STEPS` 也表示 **总环境步数（total env steps）**，不是 RLlib training iteration。训练入口会根据 `rollout_fragment_length × num_rollout_workers × num_envs_per_worker` 自动换算成所需 iteration 数。

### 3. 配置文件
当前训练入口主要使用：
- `configs/train/selector.yaml` — selector MAPPO 正式训练
- `configs/train/refine_grpo.yaml` — selected-mode GRPO refinement 训练

`selector.yaml` 用于 Phase 6 的 selector MAPPO 训练，包含：
- 冻结 planner 配置
- selector/MAPPO 超参数
- rollout worker 配置
- reward config

### 4. 包装脚本说明
仓库中还存在：

```bash
bash scripts/run_marl_train.sh
```

但该脚本当前更像一个简化包装入口，参数较固定。对协作者和复现实验来说，优先建议直接调用 `python -m train.train_selector ...`。

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
│   ├── platoon_env.py                # 编队环境、状态恢复、轨迹执行接口
│   └── reward_terms.py               # step reward / team reward
├── evaluation/
│   └── platoon_metrics.py            # 编队指标
├── scenarios/
│   └── definitions.py                # S1~S12 场景定义与触发配置
├── models/
│   ├── platoon/                      # 编队 planner、关系编码、权重迁移、trajectory refiner
│   ├── selector/                     # intent selector actor/critic 与 RLlib adapter
│   └── diffusion/                    # diffusion RL scheduler (log_prob + replay)
├── train/
│   ├── train_selector.py             # selector MAPPO 训练入口
│   ├── selector_callbacks.py         # RLlib platoon callbacks
│   └── selector_callbacks.py         # RLlib 自定义指标回调
├── expert_dataset/                   # 数据采集、anchor 抽取等
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
- [`docs/phases6/mappo_selector_plan.md`](docs/phases6/mappo_selector_plan.md)：selector MAPPO 训练链设计方案。
- [`docs/phases6/grpo_refinement_plan.md`](docs/phases6/grpo_refinement_plan.md)：GRPO trajectory refinement 实现计划。

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
- 单车 checkpoint：`/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt`
- 编队 checkpoint：`/media/kong/Elements_SE/Diffusion_Data/outputs/selector/run_x/checkpoints/`
- plan anchor：`expert_dataset/anchors.npy`

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
