# Phase 0 Metrics Specification

## Purpose

本文件冻结单车阶段、编队阶段和工程侧的核心指标命名与口径，避免后续训练、评估和论文实验各写一套。

## Single-Vehicle Metrics

### Closed-Loop

单车闭环阶段保留以下核心指标：

- `success_rate`
- `crash_rate`
- `out_of_road_rate`
- `avg_reward`
- `avg_length`

### Open-Loop

单车开环阶段至少保留：

- 轨迹误差
- 按 `trajectory_mode` 分桶的误差统计
- 轨迹模式分布

### Dataset Quality

单车数据集质量统计至少包括：

- 各 `trajectory_mode` 数量与占比
- `mean_abs_lateral_before/after`
- `final_abs_lateral_before/after`
- 几何修正强度分布

## Platoon Metrics

第一版编队阶段冻结以下核心指标：

- `success_rate`
- `collision_rate`
- `drivable_violation_rate`
- `progress`
- `formation_error`
- `recovery_time`
- `min_inter_vehicle_gap`
- `latency_ms`

### Definitions

- `success_rate`：编队整体完成任务并到达目标的 episode 占比
- `collision_rate`：任一编队成员发生关键碰撞的 episode 占比
- `drivable_violation_rate`：任一编队成员越出可行驶区域的 episode 占比
- `progress`：编队整体路线推进或平均通行效率
- `formation_error`：编队成员与目标队形之间的误差
- `recovery_time`：队形被破坏后恢复到目标容差范围内所需时间
- `min_inter_vehicle_gap`：episode 中任意两车的最小车间距
- `latency_ms`：在线推理单步或平均时延

## Benchmark Reporting Contract

### Main Table

主结果表至少报告：

- `success_rate`
- `collision_rate`
- `progress`
- `formation_error`
- `latency_ms`

### Ablation

必须覆盖以下消融方向：

- 去掉单车预训练
- 去掉编队关系建模
- 去掉图像模态
- 去掉 LiDAR 模态
- 去掉危险场景 curriculum
- 仅 IL vs IL + RL

### Generalization

泛化实验至少覆盖：

- 未见道路结构
- 未见危险工况
- 不同交通流密度
- 不同编队规模

## Engineering Metrics

考虑 `AGENTS.md` 对实时性的要求，工程指标冻结为：

- `single_step_latency_ms`
- `average_latency_ms`
- `gpu_memory_mb`
- `realtime_ratio_under_100ms`

第一版至少保证能稳定记录：

- `single_step_latency_ms`
- `average_latency_ms`
