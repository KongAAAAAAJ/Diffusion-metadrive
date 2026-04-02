# Phase 0 Input / Output Specification

## Purpose

本文件冻结单车阶段与编队阶段的输入输出接口，作为后续环境、数据、模型、评估实现的统一约束。

## Single-Vehicle Dataset Interface

### Source Of Truth

当前单车样本由 `metadrive/exp_dataset/collect_expert.py` 构造，经 `shards/ + splits/` 目录组织后进入单车训练与评估流程。

### Frozen Fields

第一版单车样本以以下字段为准：

- `ego_state`
- `other_states`
- `agent_states`
- `agent_labels`
- `lidar`
- `bev_raster`
- `bev_semantic_map`
- `left_camera`
- `front_camera`
- `right_camera`
- `state_275`
- `trajectory`
- `trajectory_raw`
- `trajectory_mode`
- `trajectory_correction_strength`
- `trajectory_mean_abs_lateral_before`
- `trajectory_mean_abs_lateral_after`
- `trajectory_final_abs_lateral_before`
- `trajectory_final_abs_lateral_after`
- `ego_pose_world`
- `action`
- `traffic_density`

其中：

- `trajectory` 是训练主监督标签
- `trajectory_raw` 是 expert 原始未来轨迹
- `trajectory_mode` 是轨迹修正模块判别出的行为模式

## Single-Vehicle Planner Interface

### Inputs

单车 planner 的目标输入模态固定为：

- 结构化状态
- 图像
- LiDAR

现有代码中已接入的特征形式包括：

- `camera_feature`
- `lidar_feature`
- `status_feature`
- `ego_state`

### Outputs

单车 planner 输出固定为：

- `future trajectory points`

约束：

- 输出是未来轨迹点，不是直接控制量
- 控制执行层通过轨迹跟踪器将轨迹转成 `steer/throttle/brake`

## Platoon Observation Interface

### Per-Vehicle Observation

每辆编队车辆至少输出：

- `ego_state`
- `camera`
- `lidar`
- `route_command`
- `neighbor_states`

### Formation Relation State

第一版冻结 `formation_relation_state` 为 12 维。

对每个邻居包含：

- `Δx`
- `Δy`
- `Δheading`
- `Δv`
- `Δx_target`
- `Δy_target`

3 车纵列时，每车最多 2 个邻居，不足补零，因此总维度为 `2 × 6 = 12`。

## Platoon Planner Interface

编队 planner 输出固定为：

- 每辆车各自的未来轨迹点序列，shape `(8, 3)`

语义保持和单车阶段一致：

- 每车输出未来轨迹
- 环境内部用控制层将轨迹转换为底层控制执行

## Dataset / Runtime Contract

当前单车训练与评估目录契约固定为：

- `dataset_root/shards`
- `dataset_root/splits/train.txt`
- `dataset_root/splits/val.txt`

后续若编队阶段复用相同的数据装载接口，也应遵守相同目录契约。
