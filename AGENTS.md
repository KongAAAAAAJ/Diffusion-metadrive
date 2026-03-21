# agents.md

## 项目名称
面向紧急危险场景的自动驾驶编队端到端决策规划

## 文档用途
本文件用于指导 Codex 在当前项目中执行科研规划、代码开发、实验评估、结果分析与论文支撑材料整理。执行时优先遵循本文件，不清楚时先保持最小可运行原则，并将不确定点整理为待确认项，而不是擅自扩大范围。

## 参考代码库
本文件提到的参考代码库位置位于/home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDrive和/home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDriveV2中。

**关键参考文件映射**（当文档中提到"参考 DiffusionDriveV2 的 XXX"时，查阅以下路径）：

| 概念 | 参考文件（相对项目根） |
|------|----------------------|
| log_prob 计算 + RL loss | `/home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py` |
| 扩散模型主体结构 | `/home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py` |
| Transfuser backbone | `/home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDriveV2/navsim/agents/diffusiondrivev2/transfuser_backbone.py` |
| RL agent 封装 | `/home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_agent.py` |
| RL config | `/home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDriveV2/navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py` |

## 文档使用规则
在每一项任务完成后检查文档中任务的完成情况，将已完成项进行标记。

## 零、环境准备与现有代码入口

### 环境安装
```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
pip install -e .          # 安装 metadrive 及核心依赖
pip install torch torchvision pytorch-lightning tensorboard  # ML 依赖
```

### 现有代码运行命令
```bash
# 数据采集
python -m metadrive.exp_dataset.collect_expert \
    --target-samples 500 --output-root <DATA_DIR> --dataset-name <NAME> \
    --expert-type ppo --trajectory-correction-enabled 1

# 单车训练
python -m metadrive.policy.diffusion_policy.train_transfuser \
    --model-size small --dataset-root <DATA_DIR> \
    --plan-anchor-path metadrive/exp_dataset/metadrive_anchors.npy

# 单车闭环测试
python -m metadrive.policy.diffusion_policy.test_transfuser_policy \
    --checkpoint <CKPT> --episodes 3 --render 0

# 运行已有测试
pytest metadrive/tests/test_policy/ -v
```

### Python 路径约定
新建的根目录模块（`envs/`、`models/`、`train/`、`evaluation/`、`scenarios/`）需要项目根在 `sys.path` 中。**所有新建脚本**头部必须加：
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # 或根据脚本深度调整
```
或者在运行时统一设置：
```bash
export PYTHONPATH="/home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive:$PYTHONPATH"
```

### 关键现有类的 import 语句
```python
# 编队环境基类
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
# 单车扩散模型
from metadrive.policy.diffusion_policy.transfuser_model_v2 import V2TransfuserModel
# 模型配置
from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
# IDM 跟驰策略（Phase 1 验收用）
from metadrive.policy.idm_policy import IDMPolicy
# 特征工具
from metadrive.policy.diffusion_policy.transfuser_features import (
    stitch_three_cameras,   # 3 视角拼接 → [3, 256, 1024]
    lidar_to_histogram,     # LiDAR → [1, 256, 256]
    build_status_feature,   # ego_state → [8]
)
```

### 默认路径约定
| 用途 | 约定路径 |
|------|---------|
| 数据集根目录 | `DATA_DIR` 环境变量，默认 `/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets` |
| 单车预训练 checkpoint | `checkpoints/single_vehicle/best.ckpt` |
| 编队训练 checkpoint | `checkpoints/platoon_rl/` |
| plan anchor 文件 | `metadrive/exp_dataset/metadrive_anchors.npy` |

### Headless 运行
训练和批量评估必须以 `use_render=False` 运行。确保无 display 时设置 `export DISPLAY=` 或使用 `xvfb-run`。

## 一、项目目标
构建一个基于 MetaDrive 的编队端到端决策规划系统，采用“单车扩散规划预训练 + 编队闭环强化微调”的两阶段路线，实现以下目标：
1. 形成可运行的 3~5 车编队决策规划项目代码。
2. 在危险工况下实现编队安全通行、协作避障、队形恢复与队形变换。
3. 相比从零训练多车智能体方法和分层式编队方法，获得更好的泛化性、闭环安全性与通行效率。
4. 形成可支撑高质量 SCI 论文的完整实验链、分析链和复现链。
5. 在线闭环推理时延尽量控制在 100ms 以内。

## 二、核心研究问题
1. 如何将单车预训练得到的基础驾驶能力迁移到编队合作决策，而不是从零开始训练多车策略。
2. 如何提升编队强化学习方法在复杂道路结构与危险工况中的泛化性。
3. 如何构建非分层、端到端的编队合作框架，使其在危险场景下优于模块化框架。
4. 如何在引入图像、LiDAR、结构化状态等多模态输入时，保证训练稳定性与推理实时性。

## 三、当前已知基础
### 已具备能力
- 已基本跑通第一阶段扩散模型模仿学习监督训练流程。
- 已在 MetaDrive 仿真器中实现单车扩散模型初步训练，核心模型为 `V2TransfuserModel`（`metadrive/policy/diffusion_policy/transfuser_model_v2.py`），主干采用 ResNet+Transformer 多尺度融合（`transfuser_backbone.py`），扩散轨迹头采用 `ConditionalUnet1D`（`modules/conditional_unet1d.py`），扩散调度器使用 `DDIMScheduler`。
- 已构建 PPO / IDM 双专家数据采集策略，核心入口为 `metadrive/exp_dataset/collect_expert.py`；包含轨迹几何修正（`trajectory_correction.py`）与 plan anchor 抽取（`abstract_anchors.py`）。
- 已完成多轮数据采集，数据集以 shards/splits 形式存储，字段规范已在 `docs/io_spec.md` 冻结。
- 当前单车模型已能实现沿车道行驶功能。

### 当前不足
- 单车能力尚未系统覆盖导航转向、换道、基础避障等完整技能。
- 编队环境与编队奖励体系尚未形成统一训练接口。
- 多车合作扩散规划器与 RL 微调流程尚未落地。
- 论文级对照实验、消融实验与危险场景 benchmark 尚未建立。

## 四、方法总路线
### 阶段一：单车扩散规划预训练
- 仿真器：MetaDrive
- 输入：结构化状态 + 图像 + LiDAR
- 输出：未来轨迹点
- 数据：在 MetaDrive 中用 PPO 专家 + 规则优化后处理微调进行自主采集（核心入口：`metadrive/exp_dataset/collect_expert.py`）
- 模型：复用现有 `V2TransfuserModel`（`metadrive/policy/diffusion_policy/transfuser_model_v2.py`），不另起架构
- 训练入口：`metadrive/policy/diffusion_policy/train_transfuser.py`（PyTorch Lightning）
- 目标：学习基础驾驶技能，包括
  - 按导航指令行驶（直行、左转、右转）
  - 自主换道
  - 基础避障
  - 稳定跟车与道路保持

### 阶段二：编队闭环强化微调
- 仿真器：MetaDrive
- 编队规模：第一版 3 辆车（验证后扩展到 5 辆）
- 模型初始化：加载单车阶段预训练扩散模型参数
- 方法参考：DiffusionDriveV2 的扩散策略强化微调思路
- 编队架构方案：
  - **第一版（完全共享）**：3 车共用同一个 `V2TransfuserModel`，每车独立前向传播，仅在 relation encoder 处交换编队信息。可直接加载单车预训练权重，推理时延 ≈ 3×单车。
  - **备选（共享 backbone + 分离 head）**：backbone 共享权重，但 trajectory head 内增加 cross-vehicle attention。在第一版方案表现不佳时启用。
- 关系编码：
  - 第一版采用 **MLP** 编码，输入 `formation_relation_state`（见第五节），拼接到 `status_feature` 后一起进入模型
  - 备选：Attention 编码（在 MLP 方案效果不足时作为提升策略）
- 编队队形策略：
  - 标准队形为**纵列**（单车道前后排列）
  - 编队内部间距按**车头时距** 0.5～1.0s 控制
  - 遇到危险时允许适当解散队形（拆分绕行、压缩通行等），离开危险区域后自动恢复纵列
  - 总原则：在保证安全和通行效率的前提下尽量保持纵列队形
- 控制执行层：
  - 纵向控制采用 **LQR** 实现速度跟踪和距离保持（编队跟驰精度要求高于单车）
  - 横向控制沿用现有 PD 控制
  - PID 纵向控制作为备选方案
- RL 微调方案：
  - 采用**多智能体 GRPO**（CTDE 架构）：每车独立采样与策略更新，通过共享 critic / reward baseline 实现协调
  - **不**用一套 GRPO 同时微调所有车的联合策略，以控制显存占用（RTX 4080 16GB 约束）
  - log_prob 计算方案与 DiffusionDriveV2 一致（在 denoising 采样结束后对最终轨迹计算）
- 目标：学习编队合作技能，包括
  - 按纵列队形协同行驶
  - 多车合作避障
  - 瓶颈路段协同通行
  - 队形自主解散与恢复
  - 危险工况下维持整体安全和较高通过率

## 五、输入输出定义
### 输入模态
#### 1. 结构化状态
至少包括：
- ego 状态（位置、速度、航向、加速度、历史运动信息）
- 邻车状态
- 编队相对关系状态（`formation_relation_state`，详见下文）
- 路由/导航指令
- 可选局部地图几何摘要

#### formation_relation_state 定义（第一版冻结）
每辆编队车辆相对于其他编队成员的关系向量，对每个邻居包含：
- `Δx`：纵向相对距离（ego-local 坐标系，前正后负）
- `Δy`：横向相对距离（ego-local 坐标系，左正右负）
- `Δheading`：航向差
- `Δv`：速度差（纵向）
- `Δx_target`：目标队形的纵向间距（**目标值**，不是偏差；如 0.5s × v_ego）
- `Δy_target`：目标队形的横向偏移（**目标值**；纵列队形下通常为 0）

对于 3 车纵列编队，每辆车最多 2 个邻居，因此 `formation_relation_state` 维度 = 2 × 6 = **12 维**（不足的邻居补零）。

**注入方式**：拼接到现有 `status_feature`（当前 8 维），形成 8 + 12 = 20 维输入，进入 `_status_encoding = Linear(20, tf_d_model)`。编队阶段需修改该层输入维度，单车阶段预训练权重中该层将部分重新初始化。

**编码方式**：第一版使用 MLP（2 层 ReLU + LayerNorm）。如果 MLP 方案效果不足，备选 cross-attention 编码。

#### 2. 图像输入
- 前向或多视角相机图像
- 第一版允许先从较少视角起步，再扩展到多视角
- 图像 backbone 优先复用现有 ResNet34 预训练结构

#### 3. LiDAR 输入
- 优先采用 MetaDrive 中可获得的 LiDAR-like observation 或点云/栅格化表示
- 第一版可采用 LiDAR raster / occupancy 风格表示，降低实现复杂度

### 输出形式
- 每辆车输出未来轨迹点序列
- 不输出直接控制量作为主输出
- 控制执行层可通过轨迹跟踪器转换成 steer/throttle/brake

## 六、场景范围
### 道路结构
必须覆盖以下结构：
- 多车道直道
- 多车道弯道
- 窄桥
- 交叉口
- 环岛
- 匝道汇入
- 匝道汇出

### 危险工况
必须显式构造会打破队形稳定性的危险场景，例如：
- 前方静态障碍迫使编队拆分绕行
- 动态车辆切入导致局部制动与重组
- 窄桥/瓶颈区域导致压缩通行
- 交叉口横向冲突导致协作博弈
- 汇入/汇出路段导致部分成员让行或变道
- 编队局部车辆受阻，其他车辆需协同调整队形

## 七、创新主线
### 主创新
提出一种面向紧急危险场景的编队端到端合作规划框架，实现单车能力到多车合作能力的迁移与增强。

### 次创新
1. 提出“单车能力泛化—多车能力提升”两阶段范式，提升编队方法泛化性。
2. 提出从开环模仿学习到闭环强化微调的能力进阶路线。
3. 将扩散规划模型引入编队合作决策场景，并探索危险工况下的多车协作轨迹生成。

### 论文写作建议
最终论文中不要把创新点平均展开，而应形成如下叙事：
- 主线：单车到编队的端到端合作范式
- 支撑点：扩散规划器作为可迁移基础策略
- 结果点：危险场景闭环合作显著优于从零训练与分层合作

## 八、评价指标
### 自动驾驶通用指标
- 成功率 / 到达率
- 碰撞率 / 无碰撞率
- Drivable area violation
- 平均通行时间
- 平均速度 / progress
- 舒适性（加速度、jerk、yaw rate）
- TTC / 最小安全裕度

### 编队专用指标
- 队形保持误差
- 跟驰误差（间距误差、速度误差、时距误差）
- 队形恢复时间
- 队形重构成功率
- 编队整体通过率
- 编队整体通行效率
- 编队最小安全间距
- 关键工况协作成功率

### 泛化指标
- 未见道路结构测试性能
- 未见危险工况测试性能
- 不同交通流密度下的鲁棒性
- 不同编队规模下的性能变化

### 工程指标
- 单步推理时延
- 平均闭环时延
- 显存占用
- 实时性达标率（100ms 以内）

## 九、基线与对照
### 单车基线
- PPO expert 直接行为回归
- 单车非扩散 planner
- 单车 RL planner
- 单车 Diffusion planner

### 编队基线
- 从零训练的多车 RL 方法
  - **实现路径**：优先使用 **MAPPO**（Multi-Agent PPO），或 MetaDrive 自带多智能体 RL 接口，尽量减少额外实现量
  - 可参考 EPyMARL / MARLlib 等开源库，复用 MultiAgentMetaDrive 环境
- 分层式编队框架（高层队形/决策 + 低层单车规划）
  - **实现路径**：高层采用 **PPO 集中式决策**（输入全局编队状态，输出队形目标 / 车道分配），低层采用 **IDM 规则规划**（按高层指令执行跟驰和换道）
- 无单车预训练的端到端多车扩散/强化方法
- 无关系建模的多车共享 planner

### 必做对照
1. 单车预训练 + 编队 RL vs 从零训练多车
2. 端到端合作 vs 分层合作
3. 扩散 planner vs 非扩散 planner
4. 有图像+LiDAR+结构化状态 vs 去掉部分模态
5. 危险工况训练 vs 无危险工况训练

## 十、消融实验清单
必须设计以下消融：
1. 去掉单车预训练
2. 去掉编队关系建模
3. 去掉图像模态
4. 去掉 LiDAR 模态
5. 去掉结构化状态中的编队相对关系
6. 去掉危险场景 curriculum
7. 仅 IL vs IL+RL
8. 不同编队规模
9. 不同 diffusion step / 候选轨迹数
10. 不同奖励权重组合

## 十一、Codex 执行原则
### 总原则
1. 优先做最小可运行版本。
2. 每一阶段先跑通，再扩功能。
3. 不要同时重构环境、模型、训练器三大模块。
4. 所有改动优先配置化，不写死魔法数。
5. 每个阶段必须留下可运行脚本、日志和结果文件。

### 对 Codex 的具体要求
#### 允许执行
- 项目骨架搭建
- 数据采集脚本与数据格式统一
- 模型模块实现与重构
- 训练脚本实现
- 评估器与指标实现
- 可视化脚本
- 实验配置文件生成
- 结果汇总表、图、日志解析
- 论文图表原始材料生成

#### 不允许擅自执行
- 擅自改变研究主任务定义
- 擅自扩大到视觉端到端全栈感知问题
- 擅自增加复杂通信机制或延迟建模，除非明确进入扩展阶段
- 擅自用未验证的大型模型替换主干架构
- 在没有对照的情况下堆叠过多 trick

### 工作方式
Codex 执行每个任务时，必须输出：
1. 修改目标
2. 涉及文件
3. 关键设计选择
4. 最小测试方式
5. 潜在风险

## 十二、推荐目录结构

> **说明**：`metadrive/` 是现有仿真器与单车训练主体，不得整体重构。编队阶段新增模块在项目根目录下按以下结构新建，避免污染现有代码树。

```text
Diffusion-metadrive/                        ← 项目根目录
├── AGENTS.md
├── docs/                                  
│   ├── problem_definition.md
│   ├── io_spec.md
│   └── metrics_spec.md
├── metadrive/                              ← ✅ 现有仿真器与单车代码，勿大改
│   ├── exp_dataset/                        ← ✅ 数据采集（collect_expert.py 等）
│   ├── policy/diffusion_policy/            ← ✅ 单车扩散模型（transfuser_model_v2.py 等）
│   └── envs/diffusion_envs/               ← ✅ 多车数据采集基础环境（base_multi_env.py）
├── configs/                                ← 新增：所有可配置超参
│   ├── env/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── eval/
├── envs/                                   ← 新增：编队环境封装
│   └── platoon_env.py
├── scenarios/                              ← 新增：危险工况注入
│   └── hazard_scenarios.py
├── models/                                 ← 新增：编队 planner
│   ├── platoon/
│   │   ├── platoon_diffusion_planner.py
│   │   ├── relation_encoder.py
│   │   └── weight_migration.py
│   └── diffusion/
│       └── diffusion_rl_scheduler.py
├── train/                                  ← 新增：训练脚本
│   ├── train_single_il.py
│   ├── train_platoon_rl.py
│   └── ma_grpo_trainer.py
├── evaluation/                             ← 新增：指标与奖励
│   ├── platoon_metrics.py
│   └── reward_terms.py
├── scripts/                                ← ✅ 已存在，继续扩充
├── tools/                                  ← 新增：数据统计与汇总工具
├── logs/
├── outputs/
├── checkpoints/
└── paper/
```

## 十三、阶段任务拆解（适合 Codex 直接执行）

### Phase 依赖关系

```text
Phase 0
    │
    ├── Phase 1（编队环境）──┐
    │                        ├──→ Phase 4（编队 Planner）──→ Phase 5（RL 微调）──→ Phase 6 ──→ Phase 7
    ├── Phase 2（数据采集）──┤
    │                        │
    └── Phase 3（单车训练）──┘
```

- Phase 1、2、3 **可并行**执行，互不依赖
- Phase 4 依赖 Phase 1（编队环境）+ Phase 3（单车预训练权重）
- Phase 5 依赖 Phase 4（编队 Planner）+ Phase 1（编队环境 + 奖励）
- Phase 6、7 串行依赖 Phase 5

### Phase 0：研究收敛与接口冻结
#### 目标
冻结第一版问题定义、输入输出接口、评估标准。
#### 任务
- 统一第一版输入模态接口：结构化状态 + 图像 + LiDAR
- 冻结输出接口：未来轨迹点
- 冻结编队规模：3~5 车，优先从 3 车开始
- 冻结第一版道路结构和危险工况集合
#### 验收标准
- 生成 `docs/problem_definition.md`
- 生成 `docs/io_spec.md`
- 生成 `docs/metrics_spec.md`

### Phase 1：MetaDrive 编队环境封装
#### 目标
完成多车编队环境和危险场景接口。
#### 任务
- 基于 `MultiAgentMetaDrive` 封装 3 车编队环境（继承而非从零建），参考 `metadrive/envs/diffusion_envs/base_multi_env.py`
- 支持纵列队形初始化：3 车同车道，初始间距 = 0.5s 车头时距 × 初始速度
- 支持相对位姿定义、路线分配
- 支持危险工况注入（见 `scenarios/hazard_scenarios.py`）
- 输出每车局部观测（复用 `DatasetCollectObservation`）+ 全局编队状态（`formation_relation_state`）
- 实现编队控制执行层：纵向 LQR（速度跟踪 + 距离保持），横向沿用 PD 控制
#### 核心文件与接口签名
```python
# envs/platoon_env.py
class PlatoonEnv(MultiAgentMetaDrive):
    """3 车编队环境，继承 MultiAgentMetaDrive。
    action 空间：每车输入未来轨迹 np.ndarray shape (8, 3)，即 [x, y, heading] × 8 步。
    环境内部通过 LQR（纵向）+ PD（横向）将轨迹转为 [steer, throttle/brake] 执行。
    控制层封装在环境内部，外部调用者只需提供轨迹。"""
    def __init__(self, config: dict): ...
    def reset(self) -> dict[str, dict]:           # {agent_id: obs_dict}
        ...
    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict, dict, dict, dict, dict]:
        """actions: {agent_id: trajectory} trajectory shape (8, 3)
        returns: obs, reward, terminated, truncated, info"""
        ...
    def get_formation_relation_state(self, agent_id: str) -> np.ndarray:
        ...  # shape (12,)
    def get_platoon_metrics(self) -> dict:
        ...  # collision_rate, formation_error, min_gap, ...

# scenarios/hazard_scenarios.py
def get_hazard_scenario_configs() -> list[dict]:
    """返回一组 MetaDrive map config，每组注入一种危险工况"""

# evaluation/platoon_metrics.py
class PlatoonMetrics:
    def update(self, info: dict): ...
    def compute(self) -> dict:  # {success_rate, collision_rate, formation_error, recovery_time, min_gap}
        ...
```
#### 子任务顺序
1. 先写 `envs/platoon_env.py`（环境骨架 + reset/step + `formation_relation_state` 计算）
2. 再写 `evaluation/platoon_metrics.py`（指标统计）
3. 最后写 `scenarios/hazard_scenarios.py`（危险工况配置）
4. 写验收脚本 `scripts/verify_phase1.py`

#### 验收标准
- 使用 MetaDrive 自带的 **IDMPolicy**（`from metadrive.policy.idm_policy import IDMPolicy`）在 `DatasetCollectEnv` 的 map 配置（`hybrid_map_sequence="SSXCOCSS"`）上完成 3 车编队 **全程 rollout**
- 初始条件：0.5s 车头时距，速度 20～30 km/h
- 全程无编队内碰撞，队形误差 < 2m
- 可记录碰撞率、成功率、队形误差、恢复时间、最小车间距共 5 项核心指标
- 支持 headless 运行
#### 验收命令
```bash
python scripts/verify_phase1.py --episodes 5 --render 0
# 预期输出：collision_rate=0.0, formation_error<2.0, 5项指标均有数值
```

### Phase 2：单车专家数据采集
#### 目标
得到覆盖关键技能的单车 IL 数据集。
#### 任务
- 复用现有 PPO / IDM 专家采集链路（核心：`metadrive/exp_dataset/collect_expert.py`），不重建平行数据体系
- 统一数据字段命名，字段规范以 `docs/io_spec.md` 为准
- 补充多道路结构、多危险工况场景的采集覆盖
- 统计数据分布并输出摘要报告
#### 核心文件
- `metadrive/exp_dataset/collect_expert.py`（已存在，主采集入口）
- `metadrive/exp_dataset/trajectory_correction.py`（已存在，轨迹修正）
- `scripts/run_dataset_collect.sh`（已存在，采集驱动脚本）
- `tools/check_dataset_stats.py`（待新建，数据分布统计）
#### 子任务顺序
1. 确认现有采集脚本可运行：`bash scripts/run_dataset_collect.sh`
2. 新建 `tools/check_dataset_stats.py`，统计数据分布
3. 扩充采集场景覆盖（修改 `scripts/run_dataset_collect.sh` 中的 map/seed 配置）
4. 运行统计并输出报告

#### 验收标准
- 形成可训练单车扩散模型的数据集
- 数据覆盖直行、转向、换道、避障基础技能
#### 验收命令
```bash
python tools/check_dataset_stats.py --dataset-root <DATA_DIR>
# 预期输出：样本总数、各场景覆盖数、轨迹分布范围统计
```

### Phase 3：单车扩散规划模型训练
#### 目标
形成可闭环运行的单车基础规划器，作为编队阶段模型初始化权重。
#### 任务
- 复用现有 `V2TransfuserModel` 实现（`metadrive/policy/diffusion_policy/transfuser_model_v2.py`），不另起架构
  - 主干：`TransfuserBackbone`（ResNet + Transformer 多尺度 BEV 融合）
  - 扩散轨迹头：`ConditionalUnet1D`（`modules/conditional_unet1d.py`）
  - 扩散调度器：`DDIMScheduler`
- 三模态输入均已接入：`camera_feature`、`lidar_feature`、`status_feature`（`transfuser_features.py`）
- 开环训练与验证，训练入口：`metadrive/policy/diffusion_policy/train_transfuser.py`
- 闭环 sanity check，评估入口：`scripts/run_diffusion_test.sh`
#### 核心文件
- `metadrive/policy/diffusion_policy/transfuser_model_v2.py`（已存在，主模型）
- `metadrive/policy/diffusion_policy/train_transfuser.py`（已存在，训练入口）
- `metadrive/policy/diffusion_policy/eval_transfuser_open_loop.py`（已存在，开环评估）
- `metadrive/policy/diffusion_policy/test_transfuser_policy.py`（已存在，闭环测试）
#### 额外任务：轨迹归一化配置化
- 当前 `transfuser_model_v2.py` 中轨迹归一化常量（x: [-1.2, 55.7], y: [-20, 26], heading: [-2, 1.9]）是单车 PPO 专家数据的统计值，写死在代码中
- 在本阶段结束前，将归一化参数提取到 `TransfuserConfig` 中配置化（`traj_norm_x_min/max`, `traj_norm_y_min/max`, `traj_norm_heading_min/max`）
- 同时在 Phase 2 数据采集时统计编队场景下各车轨迹分布范围，为编队阶段归一化参数提供依据

#### 子任务顺序
1. 将轨迹归一化硬编码提取到 `TransfuserConfig`（修改 `transfuser_config.py` + `transfuser_model_v2.py`）
2. 确认训练可运行：`bash scripts/run_diffusion_train.sh`
3. 训练至收敛，保存 checkpoint
4. 闭环测试验证

#### 验收标准
- 单车能稳定完成沿车道行驶
- 初步具备导航转向、基础换道与避障能力
- 轨迹归一化参数已从硬编码改为配置化
#### 验收命令
```bash
# 闭环测试
python -m metadrive.policy.diffusion_policy.test_transfuser_policy \
    --checkpoint <CKPT> --episodes 10 --render 0
# 预期：success_rate > 0.7, collision_rate < 0.2
# 检查归一化配置化
python -c "from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig; c=TransfuserConfig(); print(c.traj_norm_x_min, c.traj_norm_x_max)"
```

### Phase 4：编队端到端合作扩散规划器
#### 目标
将单车 planner 扩展为编队 planner（完全共享方案）。
#### 设计决策
- **不做编队级别的模仿学习（IL）**：编队阶段没有编队专家数据，直接从单车预训练权重初始化后进入 Phase 5 在线 RL
- **参数共享方案**：3 车共用同一个 `V2TransfuserModel` 实例，每车独立前向传播
- **关系编码注入**：将 12 维 `formation_relation_state` 拼接到 8 维 `status_feature`，修改 `_status_encoding` 输入维度为 20
- **权重迁移**：加载单车预训练 checkpoint 时，`_status_encoding` 层的前 8 维权重复用，后 12 维随机初始化（Kaiming init）
#### 任务
- 新建 `PlatoonDiffusionPlanner`，包装 `V2TransfuserModel`，管理 N 车的前向传播与关系编码注入
- 新建 `RelationEncoder`（MLP：12 → 64 → 12，2 层 ReLU + LayerNorm），输出拼接到 status_feature
- 实现单车权重加载与部分层重新初始化的工具函数
- 验证 3 车前向传播的输入输出 shape 正确性
#### 核心文件与接口签名
```python
# models/platoon/platoon_diffusion_planner.py
from metadrive.policy.diffusion_policy.transfuser_model_v2 import V2TransfuserModel
from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from models.platoon.relation_encoder import RelationEncoder

class PlatoonDiffusionPlanner(nn.Module):
    def __init__(self, config: TransfuserConfig, num_vehicles: int = 3):
        self.model = V2TransfuserModel(config)  # 单实例，共享权重
        self.relation_encoder = RelationEncoder(input_dim=12, hidden_dim=64, output_dim=12)

    def forward(self, batch: dict[str, dict]) -> dict[str, Tensor]:
        """batch[agent_id] = {
            "camera": Tensor [3, 256, 1024],         # 3 视角拼接
            "lidar": Tensor [1, 256, 256],            # BEV 栅格
            "status": Tensor [8],                     # build_status_feature() 输出
            "formation_relation_state": Tensor [12],  # 2邻居×6特征
        }
        return {agent_id: trajectory}  # trajectory shape [ego_fut_mode, 8, 3]
        """
        # 对每车：relation_encoder(formation_relation_state) → [12]
        #        cat(status, encoded_relation) → [20] → _status_encoding(20, tf_d_model)

# models/platoon/relation_encoder.py
class RelationEncoder(nn.Module):
    """MLP: input_dim(12) → hidden_dim(64, ReLU) → output_dim(12, LayerNorm)
    输出拼接到 status_feature [8] 后，共 [20] 维进入 _status_encoding"""

# models/platoon/weight_migration.py
def migrate_single_to_platoon(
    single_ckpt_path: str, platoon_model: PlatoonDiffusionPlanner
) -> PlatoonDiffusionPlanner:
    """加载单车权重，_status_encoding 前8维复用，后12维 Kaiming init"""
```
#### 子任务顺序
1. 先写 `relation_encoder.py`（最简单，无外部依赖）
2. 再写 `platoon_diffusion_planner.py`（依赖 relation_encoder + V2TransfuserModel）
3. 再写 `weight_migration.py`（依赖 platoon_diffusion_planner）
4. 写验收测试

#### 验收标准
- 3 车随机输入前向传播可运行，输出 shape 正确：每车 `trajectory [8, 3]`
- 成功加载单车预训练 checkpoint，`_status_encoding` 以外的层权重一致
- 单次 3 车前向传播时延 < 50ms（RTX 4080）
#### 验收命令
```bash
python -c "
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.weight_migration import migrate_single_to_platoon
import torch, time
model = PlatoonDiffusionPlanner(config)
model = migrate_single_to_platoon('<SINGLE_CKPT>', model)
# ... 构造 dummy batch, 测 forward shape 和时延
"
# 或写为 pytest: pytest tests/test_platoon_planner.py -v
```

### Phase 5：编队闭环强化微调
#### 目标
学习危险工况下的合作技能。
#### 设计决策
- **多智能体 GRPO + CTDE 架构**：
  - **Decentralized Execution**：每车独立运行自己的 `V2TransfuserModel`（共享权重），独立采样轨迹
  - **Centralized Training**：使用全局编队 reward 作为 GRPO 的 group baseline（同一场景下 group_size 组采样的 reward 均值），不需要额外的 critic 网络
  - **Per-vehicle 策略更新**：每车独立计算 policy gradient 并累加梯度（因为权重共享，等价于对共享参数求梯度的平均），不构建联合动作空间
  - 这样每步推理量 = 3 车 × group_size（建议 4），共 12 次前向传播，在 RTX 4080 16GB 下可行（需 gradient checkpointing）
- **log_prob 计算**：与 DiffusionDriveV2 一致——在 denoising 采样完成后，基于 trajectory head 输出的 mode logits 和 regression 分布计算 log_prob，不对中间 diffusion step 求梯度
- **Reference policy**：frozen copy 用于 KL 惩罚（`KL_weight` 可配置，建议初始 0.01），防止偏离单车预训练过远
- **Reward 粒度**：每环境 step 给 dense reward（包含队形误差、碰撞、进度等），不使用 episode-level sparse reward
#### 任务
- 实现 `MultiAgentGRPOTrainer`：管理 3 车共享权重的策略采样、reward 收集、梯度计算与更新
- 实现 `DiffusionRLScheduler`：封装 DDIM 采样 + log_prob 计算 + reference KL
- 实现奖励函数模块（详见第十四节），奖励版本配置化并留日志
- 支持 curriculum 训练（先简单场景，再逐步引入危险工况）
- 保存每步策略采样的 log_prob、reward、轨迹质量、队形误差等信息用于调试
- **前置验证（Phase 5 启动前）**：先做单车扩散 + GRPO 微调 toy 实验，验证 RL 梯度能否稳定流过扩散采样过程
#### 核心文件与接口签名
```python
# train/ma_grpo_trainer.py
class MultiAgentGRPOTrainer:
    def __init__(self, model: PlatoonDiffusionPlanner, env: PlatoonEnv, config: dict): ...
    def collect_group_samples(self, group_size: int = 4) -> list[dict]:
        """每个环境 step，每车采样 group_size 条候选轨迹（step-level，非 episode-level）。
        返回 [{agent_id: {traj: [G,8,3], log_prob: [G], reward: [G]}}]
        G = group_size。advantage = r_i - mean(r_group) 在 step 粒度计算。"""
    def compute_advantages(self, rollouts: list[dict]) -> list[dict]:
        """advantage = r_i - mean(r_group)"""
    def update(self, rollouts: list[dict]) -> dict:
        """返回 {loss, kl, mean_reward, ...}"""
    def train(self, total_steps: int): ...

# models/diffusion/diffusion_rl_scheduler.py（参考 /home/kong/diffusion_codes/Diffusion-meta/reference_libs/DiffusionDriveV2/.../diffusiondrivev2_model_rl.py）
class DiffusionRLScheduler:
    def sample_with_log_prob(self, model, condition) -> tuple[Tensor, Tensor]:
        """DDIM 采样 + 基于 mode logits/regression 分布计算 log_prob"""
    def compute_kl(self, log_prob: Tensor, ref_log_prob: Tensor) -> Tensor: ...

# evaluation/reward_terms.py
def compute_step_reward(info: dict, config: dict) -> float:
    """按第十四节公式计算 r_t，config 包含各项权重"""
```
#### 子任务顺序
1. 先写 `evaluation/reward_terms.py`（纯函数，无模型依赖）
2. 再写 `models/diffusion/diffusion_rl_scheduler.py`（参考 DiffusionDriveV2 的 RL model）
3. 再写 `train/ma_grpo_trainer.py`（整合前两者）
4. 写 `train/train_platoon_rl.py`（训练入口脚本）
5. **前置验证**：先跑单车扩散+GRPO toy 实验确认梯度稳定

#### 验收标准
- 训练可稳定运行（reward 曲线无发散，KL 在合理范围内）
- 相比 3 车独立使用单车策略，编队合作性能（成功率、碰撞率、队形误差）显著提升
- 单步推理时延 ≤ 100ms（3 车总计，不含 RL 训练开销）
#### 验收命令
```bash
# toy 验证（单车 GRPO 梯度测试）
python train/train_platoon_rl.py --mode toy-single --steps 100 --render 0
# 正式训练
python train/train_platoon_rl.py --config configs/train/platoon_grpo.yaml
# tensorboard 检查 reward 曲线
tensorboard --logdir logs/platoon_rl/
```

### Phase 6：实验评估与消融
#### 目标
形成论文主结果与消融结果。
#### 任务
- 跑主表实验
- 跑危险工况泛化实验
- 跑不同编队规模实验
- 跑模态消融和训练策略消融
- 整理失败案例
#### 核心文件
- `scripts/run_main_benchmarks.py`
- `scripts/run_ablations.py`
- `tools/aggregate_results.py`
#### 子任务顺序
1. 先跑主表实验（我方 vs 所有基线，覆盖所有道路结构）
2. 再跑危险工况泛化实验
3. 再跑消融实验（按第十节清单逐项）
4. 最后整理失败案例
#### 验收标准
- 形成主结果表、消融表、可视化图、失败案例图册
#### 验收命令
```bash
python scripts/run_main_benchmarks.py --config configs/eval/main.yaml --output outputs/main_results/
python scripts/run_ablations.py --config configs/eval/ablations.yaml --output outputs/ablation_results/
python tools/aggregate_results.py --input outputs/ --output paper/tables/
```

### Phase 7：论文材料生成
#### 目标
形成论文初稿所需全部材料。
#### 任务
- 自动汇总主结果
- 生成方法图和实验图表原始数据
- 整理核心创新论证链
- 输出论文结构大纲
#### 核心文件
- `paper/outline.md`
- `paper/figures/`
- `paper/tables/`
#### 验收标准
- 论文初稿材料齐备

## 十四、奖励设计（第一版数学定义）

> 所有权重为建议初始值，可在实验中调整。调整时必须记录版本号和日志，不得在无对照的情况下同时改多个项。

### 每步 dense reward（per vehicle per step）

```
r_t = w_prog · r_progress
    + w_form · r_formation
    + w_safe · r_safety
    + w_coll · r_collision
    + w_road · r_road
    + w_comf · r_comfort
```

### 各项定义

| 项 | 公式 | 建议权重 | 说明 |
|----|------|---------|------|
| `r_progress` | `Δs / Δs_max` | `w_prog = 1.0` | Δs = 沿路线纵向推进距离，Δs_max 为单步最大推进（归一化到 [0,1]） |
| `r_formation` | `-( \|Δx - Δx_target\| + \|Δy - Δy_target\| ) / d_norm` | `w_form = 0.5` | 连续相对位姿误差；d_norm = 目标间距（如 0.5s×v），使误差无量纲化 |
| `r_safety` | `-max(0, d_safe - d_min) / d_safe` | `w_safe = 0.3` | d_min = 与最近编队邻车的距离，d_safe = 安全阈值（如 3m）；仅在过近时触发 |
| `r_collision` | `-C_coll`（固定常数） | `C_coll = 10.0` | 发生碰撞时一次性惩罚，episode 同时标记 terminated |
| `r_road` | `-C_road`（固定常数） | `C_road = 5.0` | 驶出可行驶区域时一次性惩罚 |
| `r_comfort` | `-(α · \|jerk\| + β · \|Δsteering\|)` | `w_comf = 0.1` | α=0.5, β=0.5；抑制大幅急动作 |

### 编队全局 bonus（可选）

- 编队全员到达终点：额外 `+R_arrive`（建议 20.0）
- 队形恢复成功（从误差 > 阈值恢复到 < 阈值）：额外 `+R_recover`（建议 5.0）

### GRPO baseline

在 MA-GRPO 中，每个环境 step，每车采样 group_size 条候选轨迹（step-level sampling），group baseline = 该 step 下 group_size 条轨迹的 reward 均值，不需要额外 critic。advantage = `r_i - mean(r_group)`。最终执行的轨迹从 group 中选取（如 argmax reward 或按 advantage 加权）。

### 执行要求
- 奖励函数必须有版本号（`reward_v1`, `reward_v2`, ...），每次修改在 config 中切换
- 每次权重调整前后必须有 ≥ 3 个 seed 的对照实验
- 不得在无日志的情况下频繁同时改多个奖励项

## 十五、第一版技术策略建议
### 建议优先方案
- 优先从 3 车编队开始
- 优先做结构化状态 + 较简图像/LiDAR 融合
- 优先复用单车 diffusion 规划主干
- 优先实现共享参数 + 关系编码的轻量编队模型
- 优先保证闭环训练稳定，而不是一开始追求最复杂合作结构

### 暂不优先内容
- 超大规模编队
- 显式通信时延建模
- 大型 VLM/VLA 模型接入
- 原始点云高成本处理链
- 端到端感知替代结构化状态

## 十六、硬件与运行环境约束

| 项目 | 约束 |
|------|------|
| GPU | NVIDIA GeForce RTX 4080，显存 **16GB** |
| 推理时延目标 | 单步 ≤ 100ms（3 车总计） |
| Headless 运行 | **必须支持**（服务器无显示器训练与评估） |
| MetaDrive 渲染 | 训练和批量评估时关闭渲染（`use_render=False`），仅可视化调试时开启 |
| Python | ≥ 3.8（与现有 `setup.py` 一致） |
| PyTorch | ≥ 2.0（支持 `torch.compile` 加速推理，可选） |

**显存预算参考**（RTX 4080 16GB）：
- 单车 `V2TransfuserModel`（small）前向传播 ≈ 1.5GB，训练 batch=16 ≈ 8GB
- 3 车完全共享方案：推理 ≈ 4.5GB，训练 batch=8 ≈ 10GB
- MA-GRPO（per-vehicle, group_size=4）：每步 3×4=12 次前向 ≈ 12GB（需 gradient checkpointing）
- 如超出 16GB，优先降低 group_size 或 batch_size，其次考虑 gradient accumulation

## 十七、成功标准
### 最小成功
- 3 车编队环境与评估系统跑通
- 单车扩散 planner 能提供稳定初始化
- 编队 RL 微调流程跑通

### 方法成功
- 相比从零训练多车 RL，样本效率与泛化性明显改善
- 相比分层合作框架，危险工况下成功率或安全性显著提升
- 队形恢复与协作避障能力有可视化和量化证据

### 论文成功
- 有明确主表、消融表、泛化实验、失败分析
- 有清晰方法动机与实验支撑链

## 十八、Codex 下一步立即执行任务

### 立即执行：Phase 0 - 研究收敛与接口冻结
达到Phase 0验收标准

### 立即执行：Phase 1 — 编队环境（优先级最高）
- [ ] 1.1 创建 `envs/__init__.py` 和 `envs/platoon_env.py`，实现 `PlatoonEnv` 骨架（继承 `MultiAgentMetaDrive`，3 车纵列初始化，`formation_relation_state` 计算）
- [ ] 1.2 创建 `evaluation/__init__.py` 和 `evaluation/platoon_metrics.py`，实现 5 项指标统计
- [ ] 1.3 创建 `scenarios/__init__.py` 和 `scenarios/hazard_scenarios.py`，至少 3 种危险工况配置
- [ ] 1.4 创建 `scripts/verify_phase1.py`，用 IDM 跟驰跑 3 车 rollout 并输出指标
- [ ] 1.5 运行 `python scripts/verify_phase1.py --episodes 5 --render 0` 通过验收

### 立即执行：Phase 2 — 数据采集（可与 Phase 1 并行）
- [ ] 2.1 确认 `bash scripts/run_dataset_collect.sh` 可正常运行
- [ ] 2.2 创建 `tools/check_dataset_stats.py`，统计样本数、场景覆盖、轨迹分布范围
- [ ] 2.3 扩充采集配置，覆盖多道路结构和多场景
- [ ] 2.4 运行统计脚本验收

### 立即执行：Phase 3 — 单车训练（可与 Phase 1 并行）
- [ ] 3.1 将 `transfuser_model_v2.py` 中轨迹归一化硬编码提取到 `TransfuserConfig`
- [ ] 3.2 确认训练脚本可运行，启动训练
- [ ] 3.3 闭环测试，保存可用的单车 checkpoint

### Phase 1+3 完成后：Phase 4 — 编队 Planner
- [ ] 4.1 创建 `models/platoon/relation_encoder.py`
- [ ] 4.2 创建 `models/platoon/platoon_diffusion_planner.py`
- [ ] 4.3 创建 `models/platoon/weight_migration.py`
- [ ] 4.4 验证 3 车前向传播 shape 正确，权重迁移正确

### Phase 4 完成后：Phase 5 — RL 微调
- [ ] 5.1 创建 `evaluation/reward_terms.py`
- [ ] 5.2 创建 `models/diffusion/diffusion_rl_scheduler.py`
- [ ] 5.3 创建 `train/ma_grpo_trainer.py`
- [ ] 5.4 创建 `train/train_platoon_rl.py`
- [ ] 5.5 单车 GRPO toy 实验验证梯度稳定
- [ ] 5.6 启动 3 车编队 GRPO 训练

## 十九、最终提醒
本项目的关键不是堆叠模块，而是形成一条清晰的研究证据链：
- 单车预训练为什么有用
- 端到端合作为什么优于分层合作
- 扩散规划为什么适合危险工况下的编队协作
- 从开环到闭环的能力提升是否真实成立

Codex 的所有实现都应服务于这条证据链。