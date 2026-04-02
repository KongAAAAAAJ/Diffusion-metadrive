# 换道轨迹生成重写：世界坐标混合 → Frenet 坐标系五次横向偏移

> Date: 2026-03-27
> 目标文件: `metadrive/exp_dataset/hierarchical_expert/trajectory_planner.py`
> 关联文件: `metadrive/exp_dataset/hierarchical_expert/lane_change_manager.py`

---

## 问题诊断

运行 `run_expert.py` 时，车辆换道过程中频繁触发 `out_of_road`。根本原因是 `QuinticLaneChangePlanner` 生成的参考轨迹本身存在几何缺陷。

### 缺陷 1：世界坐标系线性插值在弯道上偏离道路（最核心）

`_sample_blended_points` 第 123-125 行：

```python
source_point = source_lane.position(source_longitudinal, start_lateral)
target_point = target_lane.position(target_longitudinal, 0.0)
points.append((1.0 - alpha) * source_point + alpha * target_point)
```

在弯道上，source_lane 和 target_lane 是不同曲率的同心弧。世界坐标系中对两条弧上的点做线性插值，中间点会：
- 内侧换道时切入弯道内侧，偏离车道
- 外侧换道时甩向弯道外侧，超出道路边界
- S 弯路段曲率方向变化时，产生扭曲的 S 形轨迹

### 缺陷 2：PointLane 的 1m 最小段长合并导致轨迹锯齿化

`InterpolatingLine._get_properties` 会合并间距 < 1m 的连续点。当前设置 `num_sample_points=60`，在 ego_speed=10 m/s、duration=5s 时，总轨迹长约 50m，点间距约 0.83m。大量采样点被合并，平滑的五次曲线退化为粗糙折线（可能只剩 10-15 个有效段），Pure Pursuit 跟踪器看到的是锯齿形轨迹。

### 缺陷 3：车道末端 clip 导致轨迹退化

第 109-110 行：

```python
source_longitudinal = np.clip(..., 0.0, source_length)
target_longitudinal = np.clip(..., 0.0, target_length)
```

车辆接近路段末尾时 `source_remaining` 很小，大量采样点被 clip 到车道末端位置聚集。这些聚集点被 PointLane 的 1m 规则合并后，换道轨迹变成近乎垂直于行驶方向的急转弯。

---

## 修改方案：Frenet 坐标系五次横向偏移

核心思路：不再分别在两条车道上取点做世界坐标系插值，改为在 source_lane 的 Frenet 坐标系中直接用五次多项式规划横向偏移。

```
轨迹点 = source_lane.position(s(t), d(t))
其中：
  s(t) = start_longitudinal + ego_speed * t    （纵向：匀速前进）
  d(t) = 五次多项式(start_lateral → target_offset)  （横向：平滑过渡）
```

轨迹天然沿 source_lane 几何形状生成，自动跟随弯道曲率。

---

## 执行步骤

### 步骤 1：新增 `_sample_frenet_points` 方法

在 `QuinticLaneChangePlanner` 中新增方法，替代 `_sample_blended_points`：

```python
def _sample_frenet_points(
    self,
    source_lane,
    start_longitudinal: float,
    ego_speed: float,
    start_lateral: float,
    target_offset: float,
    duration: float,
) -> np.ndarray:
    """在 source_lane 的 Frenet 坐标系中生成换道轨迹点。

    使用五次多项式控制横向偏移 d(t): start_lateral → target_offset，
    边界条件 d'(0)=d''(0)=d'(T)=d''(T)=0，保证起止横向速度和加速度为零。
    纵向方向以 ego_speed 匀速前进。
    """
    # 自适应采样：确保相邻点间距 >= 1.5m，避免 PointLane 的 1m 合并
    travel_distance = ego_speed * duration
    num_points = max(int(travel_distance / 1.5), 20)
    num_points = min(num_points, self.config.num_sample_points)
    times = np.linspace(0.0, duration, num_points)

    # 五次多项式：横向偏移从 start_lateral 到 target_offset
    coeffs = self._solve_quintic_coefficients(
        start_lateral, 0.0, 0.0,   # d(0), d'(0), d''(0)
        target_offset, 0.0, 0.0,   # d(T), d'(T), d''(T)
        duration,
    )

    source_length = float(source_lane.length)
    points = []
    for t in times:
        s = float(np.clip(start_longitudinal + ego_speed * float(t), 0.0, source_length))
        d, _, _ = self._evaluate_quintic(coeffs, float(t))
        point = np.asarray(source_lane.position(s, d), dtype=np.float64)
        points.append(point)

    return np.asarray(points, dtype=np.float64)
```

### 步骤 2：在 `plan()` 中添加车道末端防护并切换到 Frenet 采样

修改 `plan()` 方法：

```python
def plan(self, ego_position, ego_heading, ego_speed, source_lane, target_lane, direction, urgency=0.0):
    del direction
    longitudinal, lateral = source_lane.local_coordinates(ego_position)
    source_remaining = max(float(source_lane.length) - float(longitudinal), 0.0)

    # 车道末端防护：剩余距离不足则拒绝换道
    min_required = float(ego_speed) * self.config.min_duration * 0.3
    if source_remaining < max(min_required, 5.0):
        return None

    enforce_curvature = source_remaining > max(float(source_lane.width) * 1.5, 5.0)

    # 计算 target_offset：target_lane 中心线在 source_lane Frenet 坐标系中的横向偏移
    ref_s = float(np.clip(longitudinal, 0.0, float(source_lane.length)))
    ref_pos = source_lane.position(ref_s, 0.0)
    target_s = float(np.clip(target_lane.local_coordinates(ref_pos)[0], 0.0, float(target_lane.length)))
    target_center = target_lane.position(target_s, 0.0)
    _, target_offset = source_lane.local_coordinates(target_center)

    duration = self._compute_duration(ego_speed, target_offset - lateral, urgency)

    for _ in range(3):
        points = self._sample_frenet_points(
            source_lane=source_lane,
            start_longitudinal=float(longitudinal),
            ego_speed=float(ego_speed),
            start_lateral=float(lateral),
            target_offset=float(target_offset),
            duration=duration,
        )
        # 计算终点纵向位置用于 heading 验证
        terminal_target_longitudinal = float(np.clip(
            target_s + float(ego_speed) * duration,
            0.0,
            float(target_lane.length),
        ))
        if self._validate_world_path(
            points,
            source_lane,
            target_lane,
            terminal_target_longitudinal,
            enforce_curvature=enforce_curvature,
        ):
            return self._build_point_lane(points, width=float(source_lane.width))
        duration = min(duration * 1.3, self.config.max_duration * 2.0)
    return None
```

### 步骤 3：同步更新 `plan_abort`

`plan_abort` 也改用 Frenet 坐标系，从当前横向偏移平滑回到 0：

```python
def plan_abort(self, ego_position, ego_heading, ego_speed, source_lane) -> PointLane:
    longitudinal, lateral = source_lane.local_coordinates(ego_position)
    duration = float(np.clip(
        abs(lateral) / max(self.style.desired_lateral_speed * 0.8, 0.3),
        2.0, 5.0,
    ))
    for _ in range(3):
        points = self._sample_frenet_points(
            source_lane=source_lane,
            start_longitudinal=float(longitudinal),
            ego_speed=float(ego_speed),
            start_lateral=float(lateral),
            target_offset=0.0,  # 回到 source_lane 中心线
            duration=duration,
        )
        if self._validate_world_path(points, source_lane, source_lane,
                                     float(np.clip(longitudinal + ego_speed * duration, 0.0, float(source_lane.length)))):
            return self._build_point_lane(points, width=float(source_lane.width))
        duration = min(duration * 1.3, 8.0)
    # 最后兜底：即使验证不通过也返回轨迹（abort 必须返回）
    return self._build_point_lane(points, width=float(source_lane.width))
```

### 步骤 4：删除废弃代码

`_sample_blended_points` 方法在步骤 1-3 完成后不再被调用，**删除**它。

### 步骤 5：放宽 corridor_margin 验证（可选）

Frenet 坐标系轨迹在大曲率弯道上的世界坐标点可能比世界坐标混合轨迹略微偏离走廊。如果验证通过率过低，将 `LaneChangeTrajectoryConfig.corridor_margin` 从 `0.2` 增大到 `0.5`。仅在步骤 1-3 完成后运行测试发现过多 `plan() → None` 时执行此步。

---

## 不需要修改的部分

| 文件 | 原因 |
|------|------|
| `trajectory_tracker.py` | Pure Pursuit 跟踪器不关心轨迹如何生成，只消费 PointLane |
| `lane_change_manager.py` | Manager 调用 `planner.plan()` 和 `planner.plan_abort()` 的接口不变 |
| `safety_assessor.py` | 安全评估与轨迹形状无关 |
| `hierarchical_policy.py` | 上层策略调用接口不变 |
| `driving_style.py` | 风格参数不变 |

---

## 验收条件

### AC-1：基本功能（必须通过）

运行以下命令：

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/new_edit_expert/Diffusion-metadrive
python -m metadrive.exp_dataset.run_expert --expert-type idm --episodes 10 --render 0 --print-episode-summary 1 --print-run-summary 1
```

**通过标准：**
- 10 个 episode 中至少 **7 次** `arrive_dest=True`
- `out_of_road=True` 的 episode 不超过 **2 次**
- 运行过程中存在换道行为（`lane_changes >= 1` 在 run summary 中）

### AC-2：轨迹几何质量（必须通过）

添加以下临时调试代码到 `_sample_frenet_points` 末尾，运行 3 个 episode 后删除：

```python
# DEBUG: 验证 Frenet 轨迹几何质量
deltas = np.diff(points, axis=0)
segment_lengths = np.linalg.norm(deltas, axis=1)
assert np.all(segment_lengths >= 0.5), f"存在过短段: min={segment_lengths.min():.3f}m"
headings = np.arctan2(deltas[:, 1], deltas[:, 0])
heading_jumps = np.abs(np.diff(np.unwrap(headings)))
assert np.all(heading_jumps < 0.3), f"存在航向突变: max={heading_jumps.max():.3f}rad"
```

**通过标准：**
- 所有换道轨迹的相邻点间距 >= 0.5m（无点聚集）
- 所有换道轨迹的相邻段航向变化 < 0.3 rad（无锯齿/急转弯）

### AC-3：回归保护（必须通过）

对比修改前后的 10-episode run summary：
- `crash=True` 的次数不增加（允许相同或减少）
- `lane_changes` 总数不低于修改前的 50%（换道功能仍然活跃）

### AC-4：接口兼容（必须通过）

- `plan()` 的函数签名不变：`(ego_position, ego_heading, ego_speed, source_lane, target_lane, direction, urgency) → Optional[PointLane]`
- `plan_abort()` 的函数签名不变：`(ego_position, ego_heading, ego_speed, source_lane) → PointLane`
- `LaneChangeTrajectoryConfig` 的 dataclass 字段只增不删（向后兼容）

---

## 修改前的基线数据

运行修改前的代码记录基线（用于 AC-3 对比）：

```bash
python -m metadrive.exp_dataset.run_expert --expert-type idm --episodes 10 --render 0 --print-run-summary 1
```

将 `[Run Summary]` 输出记录为基线。
