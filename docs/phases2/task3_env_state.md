# 5v2.3 Env 状态保存/恢复

> **依赖**：无（可独立实施）
> **改动文件**：`envs/platoon_env.py`

## 背景

闭环训练需要对多个联合组分别在环境中执行，执行后恢复到同一起点。MetaDrive 不原生支持 env fork，需要实现轻量的状态保存/恢复接口。

## 需要保存的状态

分析 `PlatoonEnv` 的内部状态（line 117-133, 249-301, 311-321）：

| 状态 | 类型 | 来源 |
|------|------|------|
| 每辆车位置 (x, y) | float×2 | `vehicle.position` |
| 每辆车航向 heading | float | `vehicle.heading_theta` |
| 每辆车速度 | float×2 | `vehicle.velocity` |
| 每辆车前轮转角 | float | `vehicle.steering` (如有) |
| `_last_actions` | dict | 内部追踪 |
| `_last_progress_refs` | dict | 进度追踪基准 |
| `_last_info` | dict | 上一步 info |
| `env_step_counter`（在 trainer 中） | int | 步数计数 |

**不需要保存**：
- 场景布局（reset 后固定不变）
- 传感器配置（固定）
- 交通参与者状态（当前编队训练只有编队车辆，无交通流）

## 具体代码改动

### 改动 1：新增 `get_state()` 方法

**文件**：`envs/platoon_env.py`
**位置**：在 `evaluate_trajectory_group()` 之后（约 line 641）

```python
def get_state(self) -> dict:
    """保存当前环境状态的快照，用于后续 set_state() 恢复。

    Returns
    -------
    state : dict
        包含所有需要恢复的状态信息。
    """
    vehicle_states = {}
    for agent_id in self._agent_ids:
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            continue
        vehicle_states[agent_id] = {
            "position": np.array(vehicle.position, dtype=np.float64).copy(),
            "heading": float(vehicle.heading_theta),
            "velocity": np.array(vehicle.velocity, dtype=np.float64).copy(),
        }

    return {
        "vehicle_states": vehicle_states,
        "_last_actions": {k: v.copy() if isinstance(v, np.ndarray) else v
                          for k, v in self._last_actions.items()} if hasattr(self, '_last_actions') else {},
        "_last_progress_refs": dict(self._last_progress_refs) if hasattr(self, '_last_progress_refs') else {},
        "_last_info": dict(self._last_info) if hasattr(self, '_last_info') else {},
    }
```

### 改动 2：新增 `set_state()` 方法

**文件**：`envs/platoon_env.py`
**位置**：在 `get_state()` 之后

```python
def set_state(self, state: dict) -> None:
    """恢复之前通过 get_state() 保存的环境状态。

    Parameters
    ----------
    state : dict
        get_state() 的返回值。

    Notes
    -----
    仅恢复车辆动力学状态和内部追踪变量。
    不恢复场景布局和传感器配置（这些在 reset 后不变）。
    恢复后不会自动生成新的观测，调用者需要手动获取。
    """
    vehicle_states = state["vehicle_states"]
    for agent_id, vs in vehicle_states.items():
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            continue
        # MetaDrive vehicle state restoration
        # vehicle.set_position 和 set_heading 是 BaseVehicle 的方法
        vehicle.set_position(vs["position"].tolist())
        vehicle.set_heading_theta(vs["heading"])
        vehicle.set_velocity(vs["velocity"].tolist())

    if state.get("_last_actions"):
        self._last_actions = state["_last_actions"]
    if state.get("_last_progress_refs"):
        self._last_progress_refs = state["_last_progress_refs"]
    if state.get("_last_info"):
        self._last_info = state["_last_info"]
```

### 改动 3：新增 `get_current_obs()` 方法

**文件**：`envs/platoon_env.py`
**位置**：在 `set_state()` 之后

```python
def get_current_obs(self) -> dict:
    """获取当前状态下的观测（不执行 step）。

    用于 set_state() 后获取恢复状态的观测。

    Returns
    -------
    obs : dict[str, dict[str, np.ndarray]]
    """
    raw_obs = {}
    for agent_id in self._agent_ids:
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            continue
        # 直接从 observation 模块获取当前观测
        obs_instance = self.observations.get(agent_id)
        if obs_instance is not None and hasattr(obs_instance, 'observe'):
            raw_obs[agent_id] = obs_instance.observe(vehicle)
        else:
            # fallback：返回 reset 时缓存的观测格式
            raw_obs[agent_id] = self._format_agent_observation(agent_id, None)
    return self._augment_observations(raw_obs)
```

## MetaDrive API 兼容性

需要验证 MetaDrive 的 `BaseVehicle` 是否支持以下方法：

| 方法 | 说明 | 备选方案 |
|------|------|----------|
| `vehicle.set_position([x, y])` | 设置位置 | `vehicle.origin.setPos(x, y, z)` (Panda3D) |
| `vehicle.set_heading_theta(heading)` | 设置航向 | `vehicle.origin.setH(heading_deg)` |
| `vehicle.set_velocity([vx, vy])` | 设置速度 | 通过 Bullet physics: `vehicle.chassis.node().setLinearVelocity(...)` |

**如果某个方法不存在，需要用底层 Panda3D / Bullet API 实现**。在验收测试中会检测。

## 验收标准

**交付物**：修改后的 `envs/platoon_env.py`、`tests/acceptance/test_phase5v2_task3.py`

**验收指标**（9 项）：
1. `get_state()` 返回 dict，包含 `vehicle_states` 字段
2. `vehicle_states` 包含所有 active agent 的 position, heading, velocity
3. `set_state(state)` 不抛异常
4. **round-trip 一致性（ToyEnv）**：ToyEnv 上 `get_state()` → `set_state()` → `get_state()` 不报错，返回的 vehicle 信息与原始一致（stub 实现）
5. **闭环验证（ToyEnv）**：ToyEnv 上 `get_state()` → `env.step(action)` → `set_state(saved)` → 正常执行，无异常
6. `get_current_obs()` 返回格式与 `reset()` 一致的 obs dict（key 集合相同，各 value 为 np.ndarray）
7. **【新增】Panda3D fallback 路径覆盖**：测试中构造一个没有 `set_position` 方法、但有 `origin.setPos` 的 mock vehicle 对象，调用 `set_state()` 时应走 Panda3D fallback 路径而非抛出 AttributeError。验证方式：构造 `mock_vehicle`，赋予其 `origin.setPos`、`origin.setH`、`chassis.node().setLinearVelocity` 方法，确认 `set_state()` 成功调用这些方法
8. **【新增】API 探测日志**：`set_state()` 在内部应通过 `hasattr` 判断走哪条路径，测试需验证当两条路径均不存在时，`set_state()` 抛出明确的 `RuntimeError`（而非 `AttributeError` 或静默失败），错误信息中应包含 agent_id 和缺失的方法名
9. ToyEnv 的 `get_state()`、`set_state()`、`get_current_obs()` stub 方法存在且可调用，`get_state()` 返回 dict，`set_state()` 为 no-op，`get_current_obs()` 返回 obs dict

**验收命令**：
```bash
pytest tests/acceptance/test_phase5v2_task3.py -v
```

## 注意事项

1. **MetaDrive 版本兼容**：不同版本的 MetaDrive vehicle API 可能不同。需要在真实 MetaDrive 环境中验证，ToyEnv 中也需要实现 stub 版本
2. **ToyEnv 也需要 get_state/set_state**：在 `train/train_platoon_rl.py` 的 `ToyEnv` 类中添加对应的 stub 方法（返回空 dict，set_state 为 no-op），以便 toy 模式下也能跑通
3. **多车同步恢复**：`set_state()` 必须在一次调用中恢复所有车辆，不能分开调用（否则中间状态不一致）
4. **观测一致性**：`set_state()` 后获取的观测应与 `get_state()` 时一致。如果传感器（camera/lidar）依赖渲染管线，可能需要额外的 `env.render()` 调用来刷新
5. **set_state() 的 fallback 实现要求**：当前代码（`platoon_env.py:688-757`）已有 Panda3D fallback，但测试未覆盖该路径。需在 `set_state()` 的实现中确认逻辑为：先 `hasattr(vehicle, 'set_position')` → 走标准 API；否则 `hasattr(vehicle, 'origin')` → 走 Panda3D API；两者均无则 `raise RuntimeError(f"Cannot restore state for {agent_id}: no supported position API")`
