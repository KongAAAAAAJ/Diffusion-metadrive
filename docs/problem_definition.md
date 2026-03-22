# Phase 0 Problem Definition

## Research Goal

本项目面向紧急危险场景，目标是在 MetaDrive 上构建一个“单车扩散规划预训练 + 编队闭环强化微调”的端到端自动驾驶编队决策规划系统。

第一版主线固定为：

1. 先训练单车基础驾驶能力。
2. 再将单车能力迁移到 3 车纵列编队合作决策。
3. 最终形成支持论文实验链、复现链和结果分析链的项目结构。

## Main Research Questions

后续所有实现都应围绕以下问题组织：

1. 单车预训练为什么能帮助编队合作学习，而不是从零训练多车策略。
2. 端到端合作框架为什么在危险工况下优于分层式编队框架。
3. 扩散规划为什么适合危险场景中的多车协作轨迹生成。
4. 在引入结构化状态、图像、LiDAR 多模态输入后，如何兼顾训练稳定性与在线实时性。

## Current Project Baseline

### Already Available

当前仓库已经具备以下基础：

- 单车 expert 数据采集链路，核心入口为 `metadrive/exp_dataset/collect_expert.py`
- PPO / IDM 双专家支持，以及 PPO 轨迹模式识别与几何修正
- plan anchor 抽取链路，核心入口为 `metadrive/exp_dataset/abstract_anchors.py`
- 单车 diffusion 训练链路，核心入口为 `metadrive/policy/diffusion_policy/train_transfuser.py`
- 单车闭环测试入口为 `metadrive/policy/diffusion_policy/test_transfuser_policy.py`

### Not Yet Available

当前还未落地的关键部分：

- 编队环境统一接口
- 危险工况注入接口
- 编队奖励与指标模块
- 多车扩散 planner 与 RL 微调链路
- 论文级 benchmark、对照和消融实验脚本

## Frozen Phase 0 Decisions

为避免后续阶段反复返工，当前冻结以下决定：

- 第一版编队规模固定为 `3 车`
- 编队队形固定为 `纵列`
- planner 主输出固定为 `未来轨迹点序列`
- 不输出直接控制量作为主输出
- 输入模态目标固定为 `结构化状态 + 图像 + LiDAR`
- 第一批危险工况固定覆盖：
  - 前方静态障碍迫使绕行
  - 动态车辆切入
  - 瓶颈 / 窄桥压缩通行
  - 匝道汇入 / 汇出

## Phase Mapping

### Phase 1

实现基于 `MultiAgentMetaDrive` 的 3 车编队环境、危险场景接口和编队指标接口。

### Phase 2

继续复用现有单车 expert 采集主链路，补齐数据统计和多场景覆盖，不新建平行数据体系。

### Phase 3

继续复用现有单车 diffusion 训练与闭环测试链路，固化可迁移到编队阶段的单车 planner 初始化能力。

## Success Criteria

### Minimum Success

- 3 车编队环境和编队指标系统可以运行
- 单车 diffusion planner 可作为稳定初始化策略
- 编队 RL 微调流程可以跑通

### Method Success

- 相比从零训练多车 RL，有明显的样本效率或泛化性优势
- 相比分层合作框架，在危险工况下具备更好的成功率、安全性或通行效率

### Paper Success

- 形成主结果表、消融表、泛化实验和失败案例
- 结论能够回到“单车预训练迁移”和“端到端合作优于分层”的主线
