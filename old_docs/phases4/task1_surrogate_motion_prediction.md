# Phase 6B.1：Surrogate 评估引入他车运动预测

> **前置依赖**：无
> **目标**：修复 `evaluate_trajectory_group()` 中他车位置冻结在 t=0 导致 75% 假阳性碰撞的 bug
> **预期效果**：后车/中车可行 anchor 从 0/8 恢复到 4/8，碰撞率从 ~79% 降至 ~50%

## 问题根因

`_surrogate_min_gap()` 和 `_surrogate_formation_error()` 使用**当前时刻（t=0）**的他车世界坐标，而 ego 轨迹是 4 秒未来预测（8 步 × 0.5s）。编队车间距仅 9.21m，任何 dx>8m 的 anchor 中间步都会"穿越"冻结的前车位置，触发假碰撞。

实际上编队车辆以 ~7 m/s 同步前进，相对位置变化很小。

## 修改文件

仅修改 `envs/platoon_env.py`，不动其他文件。

## 修改内容

### 1. 新增 `_agent_velocity_ms()` 方法

在 `_agent_speed_km_h()` 方法之后（约第 431 行），新增：

```python
def _agent_velocity_ms(self, agent_id: str) -> np.ndarray:
    """Returns [vx, vy] in world frame (m/s). Falls back to heading-aligned speed."""
    vehicle = self.agents.get(agent_id)
    if vehicle is None:
        return np.zeros(2, dtype=np.float32)
    velocity = getattr(vehicle, "velocity", None)
    if velocity is not None:
        return np.asarray(velocity[:2], dtype=np.float32)
    speed = self._agent_speed_km_h(agent_id) / 3.6
    heading = float(vehicle.heading_theta)
    return np.asarray([speed * np.cos(heading), speed * np.sin(heading)], dtype=np.float32)
```

### 2. 修改 `evaluate_trajectory_group()` 中的他车位置构建

当前代码（约第 582-586 行）：
```python
other_poses = {
    other_id: self._agent_pose(other_id)
    for other_id in self._agent_ids
    if other_id != agent_id and self._agent_pose(other_id) is not None
}
```

替换为同时收集速度信息：
```python
other_poses_t0 = {}
other_velocities = {}
for other_id in self._agent_ids:
    if other_id == agent_id:
        continue
    pose = self._agent_pose(other_id)
    if pose is None:
        continue
    other_poses_t0[other_id] = pose
    other_velocities[other_id] = self._agent_velocity_ms(other_id)
```

### 3. 在轨迹逐步评估循环中，计算时变他车位置

当前内层循环（约第 605 行）中，`other_poses` 是固定的。改为在每个 step 计算外推位置。

在 `for step_idx, world_pose in enumerate(trajectory_world):` 循环**开头**（第 606 行之后），添加：
```python
t = (step_idx + 1) * 0.5  # 轨迹时间步长 0.5s
other_poses_at_t = {}
for other_id, pose_t0 in other_poses_t0.items():
    vel = other_velocities[other_id]
    other_poses_at_t[other_id] = np.asarray(
        [pose_t0[0] + vel[0] * t, pose_t0[1] + vel[1] * t, pose_t0[2]],
        dtype=np.float32,
    )
```

然后将后续调用中的 `other_poses` 全部替换为 `other_poses_at_t`：
- 第 609 行：`self._surrogate_formation_error(agent_id, world_pose, other_poses)` → `other_poses_at_t`
- 第 610 行：`self._surrogate_min_gap(world_pose, other_poses)` → `other_poses_at_t`

### 4. 恢复碰撞阈值

当前第 618 行的 Problem K 临时修复：
```python
crash = crash or bool(min_gap < max(1.5, 0.3 * self._vehicle_length_m(agent_id)))
```

恢复为物理合理的阈值（半车长，约 2.87m）：
```python
crash = crash or bool(min_gap < self._vehicle_length_m(agent_id) * 0.5)
```

> 说明：运动预测修复了假阳性碰撞的根源，不再需要人为压低碰撞阈值。半车长 (2.87m) 是两车中心距小于一个车长时的合理碰撞判定。

### 5. 提取 `0.5` 为类属性（可选但推荐）

在 `PlatoonConfig` dataclass 中添加：
```python
trajectory_dt: float = 0.5  # 秒，轨迹规划器每个 waypoint 的时间间隔
```

然后 step 3 中的 `0.5` 替换为 `self.platoon_config.trajectory_dt`。

## 不需要修改的文件

- `_surrogate_min_gap()` 和 `_surrogate_formation_error()` 的函数签名和内部逻辑**不变**，它们接收的 `other_poses` 参数本身就是时变的了
- `closedloop_executor.py` — 闭环执行使用真实 env.step，不受影响
- `ma_grpo_trainer.py` — 不直接调用 surrogate
- `train_platoon_rl.py` — 不直接调用 surrogate

## 验收测试

创建 `tests/acceptance/test_phase6b_task1.py`，包含以下测试：

### Test 1：`test_motion_prediction_eliminates_false_crashes`

验证运动预测消除了后车的假阳性碰撞：

```python
def test_motion_prediction_eliminates_false_crashes():
    """Rear agent should have 0 crash anchors when other agents
    are predicted to move forward at platoon speed."""
    # 构造 3 车编队场景：
    #   agent0 at x=0, agent1 at x=9.21, agent2 at x=18.42
    #   所有车速 25 km/h (6.94 m/s)，朝向 heading=0（沿 x 轴正方向）
    #
    # 加载 metadrive_anchors.npy 作为 8 个 anchor 轨迹
    # 对 agent0（后车）调用 evaluate_trajectory_group
    #
    # 断言：
    #   - crash_flags 中 crash=True 的数量 == 0
    #   - 此前（静态他车）crash 数量为 6/8
```

实现方式：可以直接 mock `_agent_pose` 和 `_agent_velocity_ms` 返回编队位置/速度，mock `_road_half_width` 返回大值（避免 out_of_road 干扰），然后调用 `evaluate_trajectory_group`。或者如果不方便 mock MetaDrive 内部对象，可以构造一个 `PlatoonEnv` 子类/桩类只实现必要的方法。

### Test 2：`test_slow_anchor_correctly_crashes_middle_agent`

验证慢速 anchor 对中间车的碰撞判定是物理正确的：

```python
def test_slow_anchor_correctly_crashes_middle_agent():
    """Anchor 0 (avg 2.7 m/s) should cause rear-end crash for middle agent
    because the rear agent (moving at 6.94 m/s) catches up."""
    # agent1（中车）使用 anchor 0（dx=9.5m, avg 2.7m/s）
    # agent0（后车）以 6.94m/s 匀速追上来
    # 断言 crash=True（物理正确的碰撞）
```

### Test 3：`test_lead_agent_no_crash_on_straight_anchors`

```python
def test_lead_agent_no_crash_on_straight_anchors():
    """Lead agent has no vehicle ahead; straight anchors (4, 7) should not crash."""
    # agent2（前车）使用 anchor 4 和 7
    # 断言 crash=False
```

### Test 4：`test_surrogate_does_not_mutate_env_state`

```python
def test_surrogate_does_not_mutate_env_state():
    """evaluate_trajectory_group must not change env internal state."""
    # 调用前后 get_formation_relation_state 不变
    # 已有 assert 在代码中，此测试显式验证
```

### Test 5：`test_velocity_fallback`

```python
def test_velocity_fallback():
    """_agent_velocity_ms falls back to heading-aligned speed when vehicle.velocity is unavailable."""
    # Mock vehicle 无 velocity 属性
    # 验证返回 speed * [cos(heading), sin(heading)]
```

### Test 6：`test_collision_rate_improvement`（数值回归测试）

```python
def test_collision_rate_improvement():
    """With motion prediction, average crash rate across 3 agents × 8 anchors
    should be below 30% (was ~75% per rear/mid agent before fix)."""
    # 对 3 个 agent 分别 evaluate 8 个 anchor
    # 统计 crash_flags 中 True 的比例
    # 断言 < 0.30
    #   理论值：(0 + 2 + 2) / 24 = 16.7%（仅慢速 anchor 对中/前车）
```

## 数值验证基准

修复前后的 anchor 碰撞矩阵（`✗` = crash，`!` = out_of_road，`✓` = feasible）：

|  | Anchor 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--|---------|---|---|---|---|---|---|---|
| **修复前** |
| Agent0(rear) | ✗ | ✗ | ! | ! | ✗ | ✗! | ✗! | ✗ |
| Agent1(mid)  | ✗ | ✗ | ! | ! | ✗ | ✗! | ✗! | ✗ |
| Agent2(lead) | ✓ | ✓ | ! | ! | ✓ | ! | ! | ✓ |
| **修复后** |
| Agent0(rear) | ✓ | ✓ | ! | ! | ✓ | ! | ! | ✓ |
| Agent1(mid)  | ✗ | ✗ | ! | ! | ✓ | ! | ! | ✓ |
| Agent2(lead) | ✗ | ✗ | ! | ! | ✓ | ! | ! | ✓ |

- 修复前 crash rate: (6+6+0)/(3×8) = **50%**（不含 out_of_road）
- 修复后 crash rate: (0+2+2)/(3×8) = **16.7%**（仅物理正确的慢速追尾）
- feasible anchor: 修复前 (0+0+4)/24 = **16.7%** → 修复后 (4+2+2)/24 = **33.3%**

## 实施注意事项

1. **只改 `envs/platoon_env.py`**，不要改 `_surrogate_min_gap` 和 `_surrogate_formation_error` 的函数签名
2. **`trajectory_dt=0.5`** 来自 transfuser_config（time_horizon=4.0s / 8 poses = 0.5s），在 surrogate 中是轨迹评估的时间步长，与 MetaDrive 的 physics_world_step_size（0.02s × decision_repeat=5 = 0.1s）无关
3. **不要修改闭环执行器**（`closedloop_executor.py`），它使用真实 env.step 不受此 bug 影响
4. 原有的 `assert np.allclose(current_relation, ...)` 校验保留不动
5. 运行现有 phase6a 测试确保无回归：`pytest tests/acceptance/test_phase6a_task*.py -x`
