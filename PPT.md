# PPT 内容框架：面向紧急危险场景的自动驾驶编队端到端决策规划

> 总页数：28 页 | 科研汇报风格 | 逻辑线：问题 → 方法 → 系统 → 实验 → 展望

---

## 第一部分：引言与问题定义（5 页）

### P1 — 封面
- 标题：面向紧急危险场景的自动驾驶编队端到端决策规划
- 副标题：基于扩散模型的多车协作轨迹规划与闭环强化微调
- 作者 / 单位 / 日期

### P2 — 研究背景
- 自动驾驶编队的应用价值（高速公路物流、紧急救援车队、军事护航）
- 编队驾驶的核心能力：安全避障、队形保持、协作变道、危险恢复
- 配图：编队行驶示意图（正常 → 遇到障碍 → 编队变形 → 恢复）

### P3 — 现状与挑战
- **挑战 1：规划粒度**——传统 RL（PPO/MAPPO）输出单步 action，难以产生时空连贯轨迹
- **挑战 2：多模态行为**——确定性策略无法表达"向左/向右绕行"等多模态决策
- **挑战 3：多车协调**——独立学习 vs 集中式学习的可扩展性与信用分配难题
- **挑战 4：安全约束**——危险场景中碰撞代价极高，从零试错不可接受

### P4 — 相关工作与局限
- 左列：**单车扩散规划**（Diffuser, DiffusionDrive, DiffusionDriveV2）
  - 优点：多模态轨迹生成；局限：仅限单车，无编队协调
- 中列：**多车 RL 编队**（MAPPO, QMIX, MADDPG）
  - 优点：多车联合优化；局限：从零训练效率低，单步 action 无轨迹级规划
- 右列：**分层编队方案**（PPO 高层 + IDM 低层）
  - 优点：分层解耦；局限：高低层目标不一致，端到端梯度断裂
- 底部一行：本工作的定位——**结合两者优势**

### P5 — 研究问题与贡献
- **核心问题**：如何将单车扩散规划能力高效迁移到多车编队，并通过闭环 RL 微调使其适应危险场景？
- **贡献 1**：提出单车预训练 + 编队 RL 微调的两阶段范式（预训练提供 prior，RL 提供 adaptation）
- **贡献 2**：设计 MA-GRPO 算法——多 anchor 采样 + intra-anchor 优势 + 分层 advantage + ref-reg 正则
- **贡献 3**：实现端到端闭环训练框架——联合组评估 + team reward + 状态恢复机制
- **贡献 4**：在 MetaDrive 危险场景中验证编队安全性、协调性和泛化能力

---

## 第二部分：技术路线总览（2 页）

### P6 — 总体技术路线
- 两阶段流程图（上下或左右排布）：
  - **阶段一：单车扩散规划预训练**
    - 专家数据采集 → Anchor 提取 → Transfuser 骨干 + DDIM 训练
  - **阶段二：编队闭环强化微调**
    - 权重迁移 → 关系编码注入 → MA-GRPO 闭环训练
- 标注关键数据流：单车 checkpoint → 编队 planner → 闭环评估 → 策略更新

### P7 — 系统架构总图
- 编队系统模块关系图：
  - **环境层**：PlatoonEnv（MetaDrive 3 车编队 + 背景交通流 + 状态保存/恢复）
  - **模型层**：PlatoonDiffusionPlanner（共享骨干 + RelationEncoder + StatusEncoding）
  - **训练层**：MA-GRPO Trainer + ClosedLoopExecutor + JointGroup
  - **评估层**：RewardTerms（step reward + team reward）+ PlatoonMetrics
- 用箭头标注 obs / action / reward / advantage 的流向

---

## 第三部分：研究内容——阶段一（4 页）

### P8 — 单车扩散规划模型
- 模型结构图：V2TransfuserModel
  - 输入：前视 RGB + LiDAR BEV + 车辆状态 (8-dim)
  - 骨干：ResNet-34 + Transformer BEV 融合
  - 输出头：DDIMScheduler → 8 步轨迹 (8×3: x, y, heading)
- 推理过程：T=4 步 DDIM 去噪，条件引导 anchor 选择

### P9 — 专家数据采集与 Anchor 提取
- 数据采集流程：IDM 专家策略在 MetaDrive 中采集 50k 轨迹
- Anchor 提取：K-Means 聚类获取 8 个代表性轨迹模式（anchor）
- 配图：anchor 可视化（8 条代表性轨迹形态）
- 作用：anchor 作为扩散模型去噪时的条件信号，保障多模态输出

### P10 — 数据预处理与离线特征缓存
- 三阶段 pipeline：collect → anchors → preprocess
- Preprocess 阶段：将 RGB/LiDAR/状态对齐并提取离线特征，加速训练
- 单车 IL 训练：监督学习拟合专家轨迹分布，获得 best checkpoint

### P11 — 权重迁移：单车 → 编队
- 迁移策略图解：
  - 共享部分（冻结）：ResNet + Transformer BEV 骨干
  - 新增模块：RelationEncoder (MLP) + `_status_encoding` 扩展 (8→20 dim)
  - 微调部分：diff_decoder + 新增模块
- formation_relation_state 定义：2 邻居 × 6 维（Δx, Δy, Δheading, Δv, Δx_target, Δy_target）= 12 维
- 拼接后：status(8) + relation(12) = 20 维 → `_status_encoding(20, tf_d_model)`

---

## 第四部分：研究内容——阶段二 MA-GRPO（8 页）

### P12 — MA-GRPO 算法总览
- 算法流程图（一个训练 step 的完整流程）：
  1. `collect_group_samples()`：每车采样 G=2 组 × M=3 anchors → 生成候选轨迹
  2. 局部 reward 评估 → intra-anchor advantage
  3. `select_top_k_candidates()`：每车选 K=2 个最优 (group, anchor) 对
  4. `build_joint_groups()`：随机组合 M_joint=8 个联合组
  5. `ClosedLoopExecutor`：闭环执行 → team reward
  6. 分层 advantage 合并 → RL loss + ref-reg loss → 梯度更新

### P13 — 多 Anchor 采样与 Intra-Anchor Advantage
- 左图：M=3 anchor 条件下的多模态轨迹采样示意
- 右图：Intra-anchor 归一化 vs 全局归一化的对比
  - 全局归一化：高 reward anchor 主导 → mode collapse
  - Intra-anchor：每 anchor 内部归一化 → 保持多样性
- 公式：
  ```
  A_local[g, m] = (R[g,m] - mean_g(R[:,m])) / (std_g(R[:,m]) + eps)
  ```

### P14 — 联合组构建与闭环评估
- 联合组构建逻辑图：
  - 每车 top-K=2 → 3 车联合排列组合 → 随机抽取 M_joint=8 组
- 闭环执行流程：
  - 保存初始状态 → 依次执行 8 个联合组 × 8 步 env.step()
  - 每组评估后恢复到初始状态 → 保证组间公平比较
- 状态恢复：get_state() / set_state() 覆盖编队车 + 背景交通流

### P15 — Team Reward 设计
- Team reward 公式：
  ```
  r_team = w_formation · (-avg_formation_err / d_norm)
         + w_safety · (-collision_penalty)
         + w_efficiency · (avg_progress / delta_s_max)
  ```
- 设计动机：
  - 局部 reward 只看单车表现 → 可能出现个体贪婪、全局不优
  - Team reward 引入编队级全局信号 → 促进协作
- 权重设定：w_team_formation=0.5, w_team_safety=1.0, w_team_efficiency=0.3

### P16 — 分层 Advantage：局部 + 团队
- 分层 advantage 公式：
  ```
  A_final[i,k,g,t] = lambda_local × gamma^(T-t) × A_local[i,k,g]
                    + lambda_team  × gamma^(T-t) × A_team[i,m]
  ```
- 关键设计：
  - lambda_local=0.7, lambda_team=0.3：局部信号主导，团队信号辅助
  - gamma=0.8 作用于 DDIM 去噪步 T=4，而非 MDP 时间步
  - 含义：早期去噪步（决定轨迹大方向）获得更大权重

### P17 — Ref-Reg 正则替代 IL Loss
- 动机：传统 IL loss（行为克隆）强制拟合专家 → 限制 RL 探索空间
- Ref-Reg（BDPO-style）：在 x_0 预测空间计算 L2 距离
  ```
  L_ref = beta * ||x_0_pred - x_0_ref||^2
  ```
- beta 自适应退火：beta_max=1.0 → beta_min=0.1（warmup 30% → decay 40%）
- 作用：训练前期保持接近预训练策略（稳定），后期放松约束（允许探索）

### P18 — 训练循环与工程优化
- 训练主循环伪代码：
  ```
  for step in range(total_steps):
      samples = collect_group_samples()         # 采样
      trainer.update(samples)                    # 梯度更新
      step_env_with_best(best_trajectory)        # 环境推进
  ```
- 工程优化：
  - Backbone 冻结 → 显存 < 10GB（RTX 4080 友好）
  - KL 自适应学习率：KL > threshold 时缩小 lr
  - OOM 自动恢复：捕获 CUDA OOM → 跳过当前 step
  - Gradient checkpointing

### P19 — 闭环执行器性能优化（Phase 6A）
- 问题诊断饼图：99.1% 时间在闭环评估 → 瓶颈是 env.reset()
- 优化方案：
  - Step 1：扩展 state 覆盖背景交通流（正确性修复）
  - Step 2：去除 reset，纯 set_state() 恢复 → 54s → 3s（18× 加速）
  - Step 3：多进程并行执行 → 3s → 0.5s（108× 加速）
- 性能对比表

---

## 第五部分：实验设计（6 页）

### P20 — 实验平台与环境设置
- MetaDrive 仿真平台介绍
- 编队环境配置：3 车编队、背景交通流（density=0.04）、多车道公路
- 危险场景类型：前车急刹、侧方切入、路障、多车混合工况
- 传感器配置：前视 RGB (320×180) + LiDAR BEV + 车辆状态
- 硬件：RTX 4080 16GB, 24 核 CPU

### P21 — 评价指标体系
- **安全类**：碰撞率 (collision_rate)、最小车间距 (min_gap)、TTC
- **编队类**：队形误差 (formation_error)、恢复时间 (recovery_time)、编队通过率
- **效率类**：成功率 (success_rate)、平均车速 (avg_speed)
- **舒适类**：横向加速度、纵向 jerk、方向盘变化率
- **工程类**：单步推理时延、显存占用

### P22 — 主表实验设计（方法对比）
- 对比方法表格：

| 方法 | 类型 | 说明 |
|------|------|------|
| Ours (MA-GRPO) | 预训练 + 闭环 RL | 单车扩散预训练 → 编队 MA-GRPO 闭环微调 |
| MAPPO | 从零多车 RL | MultiAgentMetaDrive + MAPPO，单步 action |
| 分层方案 | PPO 高层 + IDM 低层 | 集中式决策 + 底层跟驰 |
| 无预训练 | 随机初始化 + RL | 编队 planner 随机初始化 + GRPO |
| 纯 IL | 仅预训练 | 单车扩散训练 → 直接迁移，无 RL 微调 |

- 评估场景：正常跟驰、前车急刹、侧方切入、混合工况
- 每个方法 × 5 随机种子 × 100 episodes

### P23 — 消融实验设计
- **消融组 A：训练范式**
  - A1: 去掉单车预训练（随机初始化）
  - A2: 去掉 RL 微调（纯 IL 迁移）
  - A3: 开环 RL vs 闭环 RL
- **消融组 B：MA-GRPO 组件**
  - B1: 全局归一化 vs intra-anchor 归一化
  - B2: lambda_team=0 vs 0.3（有/无 team reward）
  - B3: IL loss vs ref-reg loss
  - B4: beta 固定 vs 自适应退火
- **消融组 C：模型结构**
  - C1: 去掉 RelationEncoder（relation 置零）
  - C2: 去掉 RGB 模态
  - C3: 去掉 LiDAR 模态

### P24 — 泛化性实验设计
- **道路泛化**：训练场景 vs 未见道路布局
- **交通密度泛化**：density = 0.02 / 0.04 / 0.08 / 0.12
- **编队规模泛化**：3 车 vs 4 车 vs 5 车
- **扩散步数敏感性**：T = 2 / 4 / 8
- **Anchor 数量敏感性**：M = 2 / 4 / 8

### P25 — 可视化与定性分析（预留）
- 典型场景轨迹可视化：前车急刹场景下各方法的轨迹对比
- 编队队形变化过程：正常 → 避障变形 → 恢复
- Anchor 选择分布：不同场景下 anchor 被选中的频率热力图
- Attention 可视化（如适用）：Transformer BEV 关注区域

---

## 第六部分：总结与展望（3 页）

### P26 — 研究总结
- 回顾核心问题与贡献（与 P5 呼应）
- 技术路线的有效性：两阶段范式 + MA-GRPO + 闭环训练
- 关键发现（预留实验结论占位）：
  - 发现 1：单车预训练显著加速编队 RL 收敛
  - 发现 2：intra-anchor 归一化有效防止 mode collapse
  - 发现 3：闭环 team reward 提升编队协调性
  - 发现 4：ref-reg 优于 IL loss 在探索-利用平衡上

### P27 — 局限性与未来工作
- **当前局限**：
  - 仿真到实车的 sim-to-real gap
  - 编队规模可扩展性（当前 3-5 车）
  - 通信延迟未建模（假设完美通信）
- **未来方向**：
  - 异构编队（不同车型混编）
  - 通信约束下的去中心化方案
  - 更复杂道路（交叉路口、匝道合流）
  - Sim-to-real：域随机化 + 实车部署

### P28 — 致谢 / Q&A
- 感谢页 + 联系方式
- 备用 Q&A 页

---

## 附录：设计备注

### 图表清单（需制作）
| 页码 | 图表 | 类型 |
|------|------|------|
| P2 | 编队行驶示意图 | 示意图 |
| P6 | 两阶段技术路线图 | 流程图 |
| P7 | 系统架构总图 | 模块关系图 |
| P8 | V2TransfuserModel 结构图 | 网络结构图 |
| P9 | Anchor 可视化 | 轨迹图 |
| P11 | 权重迁移示意图 | 对比图 |
| P12 | MA-GRPO 算法流程图 | 流程图 |
| P13 | Intra-anchor vs 全局归一化 | 对比图 |
| P14 | 联合组构建逻辑图 | 示意图 |
| P19 | 性能瓶颈诊断饼图 | 饼图/条形图 |
| P25 | 轨迹可视化 | 仿真截图 + 轨迹叠加 |
