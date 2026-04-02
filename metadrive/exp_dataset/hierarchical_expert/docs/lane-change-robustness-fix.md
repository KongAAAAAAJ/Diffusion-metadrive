# Lane Change Robustness Fix — 提升换道成功率

> Date: 2026-03-26
> Status: Approved
> Goal: 10 episode 中至少 7 次到达终点 (当前: 前几次换道出界导致 episode 失败)

## 1. Root Cause Analysis

换道时车辆超出道路边界有 **5 个叠加原因**:

| # | 原因 | 影响 |
|---|------|------|
| 1 | 轨迹规划假设匀速, 实际 IDM 加减速导致偏离 | 车辆跟不上/超过轨迹, Pure Pursuit 大幅修正 |
| 2 | Frenet→Cartesian 在弯道段偏差大 | 轨迹点本身就可能在路外 |
| 3 | 换道决策太频繁 (threshold=0.2, cooldown=1.5s) | 高频尝试增加出界概率 |
| 4 | 换道期间 IDM 不减速 | 高速换道更容易失控 |
| 5 | min_duration=3.0s 太短 | 轨迹曲率大, 跟踪困难 |

## 2. Fix Strategy: 3 Tiers

### Tier 1: Parameter Tuning (最高优先级, 效果最直接)

修改 `driving_style.py` 中的默认参数:

```python
# DrivingStyleProfile 默认值修改
lane_change_threshold: float = 0.2  →  0.45    # 减少不必要换道
lane_change_cooldown: int = 30      →  80      # 冷却 4s (80*0.05s)
desired_lateral_speed: float = 1.0  →  0.7     # 更温和的横向运动
```

修改 `trajectory_planner.py` 中的 `LaneChangeTrajectoryConfig`:

```python
min_duration: float = 3.0  →  4.5    # 最短换道时间 4.5s
max_duration: float = 6.0  →  8.0    # 允许更长的换道
```

修改 `trajectory_tracker.py` 中 Pure Pursuit 参数:

```python
PP_LOOKAHEAD_SPEED_GAIN: float = 0.85  →  0.65   # 更近的前视 = 更紧跟踪
PP_MIN_LOOKAHEAD: float = 4.5          →  3.5     # 低速时不要看太远
PP_MAX_LOOKAHEAD: float = 12.0         →  10.0    # 高速时也不要看太远
MAX_STEER_DELTA_PER_STEP: float = 0.035 → 0.025  # 更平滑的转向变化率
```

同步修改 `STYLE_PRESETS` 中对应值 (conservative/aggressive 端也要调整):

```python
# conservative preset
lane_change_threshold: 0.5  →  0.7
lane_change_cooldown: 30    →  100
desired_lateral_speed: 0.7  →  0.5

# aggressive preset
lane_change_threshold: 0.05  →  0.25
lane_change_cooldown: 30     →  50
desired_lateral_speed: 1.5   →  1.0
```

### Tier 2: Speed Management During Lane Change (关键结构改动)

**文件**: `hierarchical_policy.py`

在 EXECUTING / ABORTING 状态下, 限制 IDM 目标速度:

```python
# 在 act() 中, acceleration() 调用前:
LANE_CHANGE_SPEED_FACTOR = 0.75  # 换道期间目标速度降为 75%

if self.manager.state in (ManeuverState.EXECUTING, ManeuverState.ABORTING):
    # 临时降低目标速度
    original_target_speed = self.target_speed
    self.target_speed = original_target_speed * LANE_CHANGE_SPEED_FACTOR
    acc = self.acceleration(acc_front_obj, acc_front_dist)
    self.target_speed = original_target_speed  # 恢复
else:
    acc = self.acceleration(acc_front_obj, acc_front_dist)
```

**原理**: 降速使轨迹更容易跟踪, 且车辆动力学余量更大。

### Tier 3: Trajectory Boundary Validation (防御性检查)

**文件**: `trajectory_planner.py` 的 `plan()` 方法

在 `_build_point_lane` 之后, 验证轨迹点是否在道路内:

```python
def _validate_within_road(self, points: np.ndarray, source_lane, target_lane) -> bool:
    """Check that trajectory points don't exceed road boundaries.

    Uses a simple heuristic: each point's lateral offset from source_lane
    should not exceed the distance to target_lane center + margin.
    """
    margin = 1.0  # [m] 允许的额外偏移
    for point in points:
        _, lat_from_source = source_lane.local_coordinates(point)
        _, lat_from_target = target_lane.local_coordinates(point)
        # 点应该在 source 和 target 之间, 不超出 margin
        if abs(lat_from_target) > source_lane.width * 0.5 + margin:
            return False
    return True
```

在 `plan()` 中, `_build_point_lane` 后加一道检查:

```python
def plan(self, ego_position, ego_heading, ego_speed, source_lane, target_lane, direction, urgency=0.0):
    # ... existing code ...
    for _ in range(3):
        coeffs = self._solve_quintic_coefficients(...)
        if self._validate_curvature(coeffs, duration, ego_speed):
            trajectory = self._build_point_lane(source_lane, longitudinal, ego_speed, coeffs, duration)
            # NEW: boundary check
            if self._validate_within_road(trajectory.center_line_points, source_lane, target_lane):
                return trajectory
            # trajectory goes off-road, try longer duration
        duration = min(duration * 1.3, self.config.max_duration * 2.0)
    return None  # refuse to execute unsafe trajectory
```

### Tier 3b: Road Curvature Guard (可选, 与 Tier 3 互补)

**文件**: `lane_change_manager.py` 的 `_handle_idle()`

在决定换道前检查当前路段曲率:

```python
def _is_road_too_curved(self, ego, source_lane) -> bool:
    """Don't start lane change on sharp curves."""
    ego_s, _ = source_lane.local_coordinates(ego.position)
    lane_length = float(source_lane.length)
    # 采样前方 30m 的航向变化
    sample_distance = min(30.0, lane_length - ego_s)
    if sample_distance < 5.0:
        return True  # 接近车道末端, 不换道
    heading_start = source_lane.heading_theta_at(max(ego_s, 0.0))
    heading_end = source_lane.heading_theta_at(min(ego_s + sample_distance, lane_length))
    heading_change = abs(heading_end - heading_start)
    # 30m 内航向变化 > 15° 认为弯道太急
    return heading_change > 0.26  # ~15 degrees
```

在 `_handle_idle()` 的候选评估之前调用:

```python
def _handle_idle(self, ego, perception, routing_target_lane, ...):
    if self.cooldown_timer > 0:
        ...
    # NEW: curvature guard (skip for mandatory lane changes)
    urgency, is_mandatory, forced_direction = self._compute_route_urgency(...)
    if not is_mandatory and self._is_road_too_curved(ego, routing_target_lane or ego.lane):
        return routing_target_lane or ego.lane, acc_front_obj, acc_front_dist
    # ... rest of existing code ...
```

## 3. Execution Order

```
Tier 1 (parameters) → run 10 episodes → evaluate
    ↓ if success rate < 70%
Tier 2 (speed management) → run 10 episodes → evaluate
    ↓ if still < 70%
Tier 3 (boundary validation + curvature guard) → run 10 episodes → evaluate
```

**建议先只做 Tier 1 + Tier 2**, 这两步已经能解决大部分问题。Tier 3 作为保底。

## 4. Acceptance Criteria

### Tier 1: Parameter Tuning

- [ ] 1.1 `DrivingStyleProfile` 默认值: `lane_change_threshold=0.45`, `lane_change_cooldown=80`, `desired_lateral_speed=0.7`
- [ ] 1.2 `LaneChangeTrajectoryConfig` 默认值: `min_duration=4.5`, `max_duration=8.0`
- [ ] 1.3 `PurePursuitTracker`: `PP_LOOKAHEAD_SPEED_GAIN=0.65`, `PP_MIN_LOOKAHEAD=3.5`, `PP_MAX_LOOKAHEAD=10.0`, `MAX_STEER_DELTA_PER_STEP=0.025`
- [ ] 1.4 `STYLE_PRESETS` conservative/aggressive 端同步更新
- [ ] 1.5 `StyleSampler.sample()` 仍能正确插值新的默认值范围

### Tier 2: Speed Management

- [ ] 2.1 `hierarchical_policy.py` 中定义 `LANE_CHANGE_SPEED_FACTOR = 0.75`
- [ ] 2.2 `act()` 中 EXECUTING/ABORTING 状态下临时降低 `self.target_speed` 后调用 `self.acceleration()`
- [ ] 2.3 `self.target_speed` 在 `acceleration()` 调用后恢复原值
- [ ] 2.4 IDLE 状态下 `acceleration()` 行为不变

### Tier 3: Boundary Validation (若需要)

- [ ] 3.1 `trajectory_planner.py` 新增 `_validate_within_road()` 方法
- [ ] 3.2 `plan()` 在 `_validate_curvature` 通过后额外调用 boundary check
- [ ] 3.3 boundary check 失败时增大 duration 重试, 最终返回 None
- [ ] 3.4 `lane_change_manager.py` 新增 `_is_road_too_curved()` 方法
- [ ] 3.5 `_handle_idle()` 在非强制换道前调用 curvature guard

### Integration Test

- [ ] 4.1 `run_expert.py` 运行 10 episodes, `arrive_dest >= 7`
- [ ] 4.2 `out_of_road` 事件 ≤ 2 次
- [ ] 4.3 换道仍能正常发生 (不是因为全部被阻止而通过)

## 5. Modified Files Summary

| File | Tier | Change Type |
|------|------|-------------|
| `driving_style.py` | 1 | 参数修改 |
| `trajectory_planner.py` | 1+3 | 参数修改 + 新增 boundary check |
| `trajectory_tracker.py` | 1 | 参数修改 |
| `hierarchical_policy.py` | 2 | 新增换道减速逻辑 |
| `lane_change_manager.py` | 3 | 新增弯道检查 |
