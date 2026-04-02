一、全文说明与阅读导航

本文从三个层次介绍整体方法。第一部分介绍系统级方案，即高层 `selector` 与低层 `diffusion_planner` 如何分工、如何协同训练。第二部分分析该方案相对于原始 DiffusionDriveV2 训练方式的优势，重点讨论信用分配、训练稳定性与效率。第三部分聚焦于 `TrajectoryHead` 的结构改造，具体说明共享去噪主干与 `MoE task decoder` 的模块组成、输入输出关系以及多模态预训练与单模态微调时的调用方式。

其中，第一部分回答“整体方案是什么”；第二部分回答“为什么这样设计”；第三部分回答“模型内部具体怎么实现”。三部分共同构成从系统目标、训练逻辑到模块结构的完整说明。

二、研究背景与研究目标

本项目面向 MetaDrive 仿真环境中的紧急危险场景，研究并实现一个 3~5 车自动驾驶编队的端到端决策与规划系统。与常规单车自动驾驶任务相比，编队场景不仅要求每辆车具备独立安全通行能力，还要求多车之间能够在动态交通参与者、突发障碍物和高风险道路条件下保持协同决策与一致行动。因此，该问题同时具有多车协同、闭环控制、高风险工况和长时序规划等多重挑战。

本项目的研究目标是：构建一条“单车扩散预训练 + 编队闭环 RL 微调”的整体技术路线，使系统能够在危险工况下具备安全通行、协作避障和队形恢复等能力。具体而言，系统需要先利用单车扩散规划模型学习多模态轨迹生成与通用场景先验，再在编队闭环环境中通过强化学习进一步学习高层 mode 选择和低层局部 refinement，从而在保证规划表达能力的同时提高闭环执行效果、协同能力与策略稳定性。

在这一研究背景下，本文后续提出的整体方法重点回答三个问题：第一，如何将单车多模态扩散规划能力迁移到多车编队场景；第二，如何在高层 mode 选择与低层轨迹 refinement 之间实现清晰的信用分配；第三，如何在危险场景下同时兼顾安全性、协同性与训练效率。

三、方案介绍

2.1 总体思路

本方案将原始 DiffusionDriveV2 的“多模态生成与微调一体化训练”改为“高层离散选择与低层局部细化解耦训练”。整体思路是：先由冻结的多模态 `diffusion_planner` 生成一组 `candidate modes`，再由 `selector` 通过 MAPPO 在闭环仿真中学习选择单个 mode；随后复用 MAPPO rollout 过程中保存的环境状态与交互数据，对被选 mode 对应的局部 `diffusion_planner` 进行 GRPO 微调。这样，`selector` 负责学习“选哪个 mode”，`diffusion_planner` 负责学习“在该 mode 内如何局部优化”，从而实现更清晰的信用分配和更稳定的训练过程。

2.2 模型设计概览

从系统结构上看，模型由两个核心模块组成。

第一部分是多模态 `diffusion_planner`。它的作用是在给定当前观测条件下输出 `K` 个 `candidate modes` 轨迹。为了支持后续只在已选 mode 内做局部 refinement，本文将原始 `TrajectoryHead` 改造为“共享 `shared trunk` + `MoE task decoder`”的结构。`shared trunk` 负责统一的条件去噪建模，尽量复用原有多模态预训练权重；`MoE task decoder` 负责在单个 mode 特征上学习局部 refinement 策略，并在后续微调中作为主要更新对象。

第二部分是 `selector`。它接收 `diffusion_planner` 输出的 `candidate modes`、轨迹特征以及当前编队状态，通过 MAPPO 学习高层离散决策，从多个候选中选择一个最优 mode。被选中的轨迹随后既作为环境执行动作，也作为后续 `mode 内 refinement` 的基准轨迹。

2.3 训练设计概览

本方案采用两条相互衔接、但职责分离的训练链。

第一条链是 `selector` 的在线训练链。在每个环境 step 中，冻结 `diffusion_planner`，由其生成多个 `candidate modes`；`selector` 选出其中一个 mode 并送入闭环仿真环境执行；在 episode 结束后，根据真实闭环 reward 计算 PPO advantage，并用 MAPPO 更新 `selector` 的 actor 和 critic。与此同时，保存该 rollout 过程中的环境状态、周围车辆运动信息、被选 mode 轨迹及相关上下文。

第二条链是 `diffusion_planner` 的离线回放式微调链。从 MAPPO rollout 缓冲中取出某个时间步的环境状态快照和被选 mode，将该 mode 轨迹复制为 `num_group` 组并添加截断噪声，再输入 `diffusion_planner` 执行多步去噪，生成一组局部 refined 轨迹；随后从保存的环境状态出发做分支闭环评估，计算各组 refined 轨迹的 reward，并构造相对基准轨迹的组内 advantage，用 GRPO 仅更新该 mode 对应的局部 planner 参数。这样既保证 `mode 内 refinement` 只发生在同一语义邻域内，又能将轨迹变化控制在较小范围内。

2.4 阶段小结

总体而言，本方案通过“MAPPO 训练 `selector` + 回放式 GRPO 微调 mode 内 `diffusion_planner`”的双层设计，将高层 mode 选择与低层轨迹细化清晰分离。一方面，闭环仿真保证了 `selector` 的训练目标与真实任务一致；另一方面，基于回放状态的局部分支评估使 `diffusion_planner` 能在不重新进行大规模在线交互的情况下完成 `mode 内 refinement`。最终，这一方案既保留了多模态规划器的表达能力，又提高了强化学习阶段的稳定性、可解释性与训练效率。

下文首先分析这一方案的主要优势，然后进一步展开 `TrajectoryHead` 的结构改造，说明共享 `shared trunk` 与 `MoE task decoder` 的具体模块组成和数据流。

四、方案优势分析

3.1 充分复用了多模态扩散规划器的先验能力

冻结的 `diffusion_planner` 可以稳定提供覆盖性较好的 `candidate modes` 集合，`selector` 不需要从零学习连续轨迹生成，只需要学习“在当前场景下选哪一个候选更合适”；而后续的 GRPO 微调也不是重新学习整条轨迹，而是在同一 mode 内做小范围修正。这样既保留了预训练 planner 的多模态表达能力，又显著降低了强化学习搜索空间，提高了训练稳定性和样本效率。

3.2 信用分配更明确

本方案将“高层意图选择”和“低层轨迹细化”明确拆开。`selector` 只负责在多个 `candidate modes` 中做离散决策，`diffusion_planner` 只负责在已选 mode 的局部邻域内做连续 refinement。这样两部分的职责边界清晰，信用分配也更明确，避免了最终闭环 reward 同时混杂 mode 选择误差和轨迹细化误差的问题。相比直接端到端用一个连续策略同时完成 mode 选择和轨迹生成，这种分层解耦结构更容易训练，也更容易分析。

3.3 兼顾了闭环真实性和训练效率

`selector` 的训练直接依赖闭环仿真 reward，因此优化目标和真实任务表现是一致的；同时，`diffusion_planner` 的微调不必重新进行完整在线交互，而是复用 MAPPO rollout 中保存的环境状态做分支回放评估，在保证闭环打分真实性的同时减少额外环境采样成本。换句话说，它把昂贵的在线交互主要留给高层策略，把低层 planner 微调转化为基于状态快照的局部闭环优化，这在工程上更加高效。

3.4 更符合 mode 内局部细化的研究设定

由于 refinement 被显式约束在已选 mode 的邻域内，微调前后轨迹差异可以被控制在较小范围内，planner 不会在训练过程中破坏原始 mode 的语义结构。这一点非常重要，因为它保证了上游多模态候选池负责“覆盖不同意图”，下游 refinement 负责“提升该意图的闭环执行效果”，两者分工明确，也更有利于形成完整、清晰的研究证据链。总体来看，这一方案兼具结构清晰、训练稳定、样本利用率高和研究解释性强等多方面优势。

五、模型设计方案具体说明

在整体训练框架中，`TrajectoryHead` 是承接多模态轨迹生成与后续 `mode 内 refinement` 的关键模块，因此需要单独说明其结构改造。

`TrajectoryHead` 是整个扩散规划模型中的轨迹生成核心模块，位于感知编码与最终轨迹输出之间。上游 backbone 和 transformer decoder 先从相机、激光雷达以及车辆状态中提取场景上下文特征，如 `bev_feature`、`ego_query` 和 `agents_query`；`TrajectoryHead` 再以这些上下文特征为条件，对带噪的多模态轨迹锚点进行条件去噪，输出未来轨迹及其对应的模态得分。也就是说，`TrajectoryHead` 在整个项目中承担的是“将环境理解结果转化为可执行未来轨迹”的作用，是连接感知表征、多模态规划生成和后续强化学习微调的关键桥梁。

本文中的“`shared trunk`”特指 `TrajectoryHead` 中从轨迹初始化、位置编码、时间编码到条件交互式 `diff_decoder` 的共享去噪主干；“`MoE task decoder`”特指负责输出轨迹增量与模态分数的末端解码模块。

本方案将 `TrajectoryHead` 设计为两级结构：前半部分是共享的多模态条件去噪主干，后半部分是 `MoE task decoder`。`shared trunk` 负责在统一场景条件下对多模态带噪轨迹进行条件建模，学习通用的场景理解和去噪表示；`MoE task decoder` 负责针对单个轨迹 mode 做局部 refinement 策略建模。这样既能最大程度复用原有多模态预训练能力，又能在后续微调时只对单个 mode 做局部优化。

5.1 整体结构

整个 `TrajectoryHead` 由以下几个部分组成：

1. 多模态轨迹初始化模块
2. 轨迹位置编码模块
3. diffusion 时间步编码模块
4. 共享多模态条件去噪主干
5. 单模态特征拆分模块
6. `MoE task decoder`
7. 轨迹恢复与去噪迭代模块

其中，前四部分共同构成 `shared trunk`，后面三部分构成 mode-specific 解码与优化部分。

5.2 共享主干结构

`shared trunk` 接收多条带噪轨迹和环境上下文，对每个轨迹 mode 提取统一的条件去噪特征。共享主干的输入包括：

- `noisy_traj_points [B, K, T, C]`
  表示 `B` 个样本中，每个样本包含 `K` 条带噪轨迹，每条轨迹长度为 `T`，每个轨迹点维度为 `C`
- `bev_feature`
  场景 BEV 特征图
- `agents_query`
  周围车辆特征表示
- `ego_query`
  自车特征表示
- `time_embed`
  diffusion 时间步编码
- `status_encoding`
  车辆状态编码

`shared trunk` 内部由以下子模块组成。

5.2.1 轨迹锚点初始化与加噪模块

系统首先从预定义的多模态轨迹锚点 `plan_anchor` 出发，得到初始轨迹模板。在 diffusion 训练或采样时，对这些轨迹锚点添加截断噪声，得到多条 noisy trajectories：

`plan_anchor -> norm_odo -> add_noise -> noisy_traj_points`

输出：

- `noisy_traj_points [B, K, T, C]`

5.2.2 轨迹位置编码模块

对每条带噪轨迹进行位置编码。每条轨迹按时间顺序编码为一个高维位置嵌入，再通过一个 MLP 投影到统一 hidden dimension。

子模块：

- `gen_sineembed_for_position`
- `plan_anchor_encoder`

输入：

- `noisy_traj_points [B, K, T, C]`

输出：

- `traj_feature_init [B, K, D]`

其中 `D` 为共享隐藏维度。

5.2.3 时间步编码模块

对 diffusion 时间步 `t` 进行正弦位置编码，再通过 MLP 映射为时间条件向量。

子模块：

- `SinusoidalPosEmb`
- `time_mlp`

输入：

- `timesteps [B]`

输出：

- `time_embed [B, 1, D]`

5.2.4 条件交互式共享去噪主干

这是 `shared trunk` 的核心，对每个 mode 的轨迹特征进行条件建模。其结构由一个或多个共享 `Transformer Decoder Layer` 组成，每层内部包含以下子模块。

`a. Cross-BEV Attention`

轨迹特征与 `BEV` 特征交互，让每个轨迹 mode 感知局部道路结构、障碍物和空间布局。

输入：

- `traj_feature [B, K, D]`
- `noisy_traj_points [B, K, T, C]`
- `bev_feature`

输出：

- 更新后的轨迹特征 `[B, K, D]`

`b. Cross-Agent Attention`

轨迹特征与周围车辆特征交互，让每个 mode 感知邻车状态、相对位置和潜在交互风险。

输入：

- 当前轨迹特征 `[B, K, D]`
- `agents_query`

输出：

- 更新后的轨迹特征 `[B, K, D]`

`c. Cross-Ego Attention`

轨迹特征与 `ego_query` 交互，让每个 mode 与当前自车状态、自车规划意图保持一致。

输入：

- 当前轨迹特征 `[B, K, D]`
- `ego_query`

输出：

- 更新后的轨迹特征 `[B, K, D]`

`d. FFN + LayerNorm`

对融合后的轨迹特征做非线性变换和归一化，增强表示能力。

输入：

- 当前轨迹特征 `[B, K, D]`

输出：

- 更新后的轨迹特征 `[B, K, D]`

`e. Time Modulation`

利用 diffusion 时间步编码对轨迹特征做条件调制，使网络在不同噪声强度下采用不同的去噪行为。

输入：

- 轨迹特征 `[B, K, D]`
- `time_embed [B, 1, D]`

输出：

- 时间条件调制后的轨迹特征 `[B, K, D]`

经过一层或多层共享 decoder 后，得到最终共享轨迹特征：

输出：

- `traj_feature_shared [B, K, D]`

到这一步为止，`shared trunk` 完成了对多模态轨迹的统一条件建模。

5.3 单模态特征拆分

`shared trunk` 输出的是多模态特征：

- `traj_feature_shared [B, K, D]`

下一步将其按 mode 拆分，对每一个 mode 单独处理：

- 第 `i` 个 mode 特征记为 `traj_feature_i [B, D]`

在多模态预训练阶段：

- `K` 个 mode 特征都会依次送入后续的 `MoE task decoder`

在 GRPO 微调阶段：

- 只取被 `selector` 选中的单个 mode 特征 `traj_feature_i`
- 只对该 mode 做局部 refinement

5.4 MoE task decoder

原有共享 `task_decoder` 被替换为 `MoE task decoder`。它只处理单个 mode 的特征。

其结构由以下部分组成。

`1. Router`

输入：

- 单个 mode 的轨迹特征 `traj_feature_i [B, D]`

输出：

- 对多个 expert 的 gating 权重 `[B, M]`

作用：

- 自动决定当前特征应由哪些 expert 负责处理

`2. Experts`

- 包含 `M` 个并行 expert
- 每个 expert 是一个小型 MLP 或轻量残差网络

输入：

- `traj_feature_i [B, D]`

输出：

- 局部 refinement hidden feature `[B, D]`

`3. Expert Fusion`

根据 `Router` 的 gating 权重，对多个 expert 输出加权融合。

输出：

- 融合后的单模态 refined feature `[B, D]`

`4. Output Heads`

输入：

- 融合后的 refined feature

输出：

- `plan_reg_i [B, T, 3]`
- `plan_cls_i [B, 1]`

这里不手工指定某个 mode 对应某个 expert，而是让 `Router` 自动学习“跨 mode 的策略簇”。

5.5 轨迹恢复与去噪迭代

`MoE task decoder` 输出的是该 mode 的轨迹增量和分数：

- `plan_reg_i [B, T, 3]`
- `plan_cls_i [B, 1]`

然后将轨迹增量加回到原始 noisy trajectory 上：

`denoised_traj_i = noisy_traj_i + delta_i`

再结合 DDIM / RL scheduler 完成多步去噪迭代。

在多模态预训练中：

- 对 `K` 个 mode 全部进行 `shared trunk` 编码和 `MoE task decoder` 解码

在 GRPO 微调中：

- 仅对被选 mode 的特征做 group 复制、加噪、去噪和局部优化

5.6 输入输出关系总结

下面从“多模态预训练”和“单模态 GRPO 微调”两个运行阶段，重新串联前文模块的调用顺序。

5.6.1 多模态预训练阶段

输入：

- `noisy_traj_points [B, K, T, C]`
- `bev_feature`
- `agents_query`
- `ego_query`
- `time_embed`
- `status_encoding`

流程：

1. 多模态 noisy trajectories 编码
2. `shared trunk` 条件去噪建模
3. 得到 `traj_feature_shared [B, K, D]`
4. 拆分为 `K` 个单模态特征
5. 每个 mode 特征依次输入 `MoE task decoder`
6. 输出 `K` 个轨迹结果和 mode score

5.6.2 单模态 GRPO 微调阶段

输入：

- `noisy_traj_points [B, 1, T, C]`
- 与预训练阶段相同的环境条件特征

流程：

1. 单个 mode 的 noisy trajectory 输入 `shared trunk`
2. 得到单模态特征 `traj_feature_i [B, D]`
3. 输入 `MoE task decoder`
4. 输出该 mode 的 refined trajectory
5. 用 GRPO 只优化 `Router + Experts + Output Heads`

5.7 模块流程图说明

下图中的模块名称与第 4.2 至第 4.5 节的模块定义一一对应，可直接作为实现结构与训练数据流的视觉摘要。

```text
                     +----------------------------------+
                     |         TrajectoryHead           |
                     +----------------------------------+

 Input:
 noisy_traj_points [B,K,T,C]
 bev_feature
 agents_query
 ego_query
 time_step

        |
        v
+------------------------------+
| Trajectory Anchor + AddNoise |
+------------------------------+
        |
        v
+------------------------------+
| Position Encoding            |
| gen_sineembed_for_position   |
| + plan_anchor_encoder        |
+------------------------------+
        |
        v
 traj_feature_init [B,K,D]

 time_step
    |
    v
+------------------------------+
| Time Encoding                |
| SinusoidalPosEmb + time_mlp  |
+------------------------------+
        |
        v
 time_embed [B,1,D]

        traj_feature_init + scene context
        |
        v
+--------------------------------------------------+
| Shared Diffusion Trunk                           |
| - Cross-BEV Attention                            |
| - Cross-Agent Attention                          |
| - Cross-Ego Attention                            |
| - FFN + LayerNorm                                |
| - Time Modulation                                |
| - (stacked decoder layers)                       |
+--------------------------------------------------+
        |
        v
 traj_feature_shared [B,K,D]

        |
        v
+------------------------------+
| Per-Mode Feature Split       |
| traj_feature_i [B,D]         |
+------------------------------+
        |
        v
+--------------------------------------------------+
| MoE Task Decoder                                 |
| Router -> Experts -> Expert Fusion -> Heads      |
+--------------------------------------------------+
        |
        +----------------------+
        |                      |
        v                      v
 plan_reg_i [B,T,3]      plan_cls_i [B,1]

        |
        v
+------------------------------+
| Trajectory Restore           |
| noisy_traj_i + delta_i       |
+------------------------------+
        |
        v
 denoised trajectory_i
```

六、结语

综上，本文方案在系统层面采用“MAPPO 训练 `selector` + 回放式 GRPO 微调 mode 内 `diffusion_planner`”的双层训练设计，在模块层面采用“共享 `shared trunk` + `MoE task decoder`”的 `TrajectoryHead` 改造方案。前者解决了高层 mode 选择与低层局部 refinement 的信用分配问题，后者则在最大程度复用原有多模态预训练能力的同时，为单 mode 局部优化提供了足够灵活的表达能力。两者共同支撑了“高层选 mode、低层调 mode”的整体方法路线。

七、工作规划

Phase A（1-2周）：验证 MAPPO selector 基础性能

当前代码已可运行，直接跑 train_selector.py
评估 selector 是否能学到有意义的 mode 选择策略
如果 selector 已经能收敛到不错的性能，Phase B 的复杂度可以大幅降低
Phase B（1周）：per-mode output heads（不做 MoE）

将 task_decoder 改为 K 个独立小 MLP
多模态预训练中照常训练所有 head
这是完整 MoE 的简化版，可以先验证 per-mode 精修的价值
Phase C（2-3周）：离线 GRPO 微调（核心难点）

先用简化的 counterfactual 评估：用预存的场景参数重启仿真做局部 rollout
实现 GRPO loss 和参数更新
验证信号质量
Phase D（1周）：完整 MoE（可选）

如果 Phase C 证明 per-mode refinement 有效，再考虑是否需要 Router
很可能 Phase B 的简单版就够用