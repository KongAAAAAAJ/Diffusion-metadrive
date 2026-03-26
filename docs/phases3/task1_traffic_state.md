# 6A.1 扩展 get_state/set_state 覆盖背景交通流

> **依赖**：无
> **改动文件**：`envs/platoon_env.py`

## 背景

当前 `PlatoonEnv.get_state()` 仅保存编队被控车辆（agent0/1/2）的状态（位置、速度、航向），不保存背景交通流（NPC 车辆）状态。闭环执行器在 `set_state()` 恢复时，被控车回到保存点，但背景车位置是错误的。

## MetaDrive 交通管理 API

MetaDrive 已有完整的背景车辆访问接口：

```python
# 获取所有背景交通车辆对象
traffic_vehicles = env.engine.traffic_manager.traffic_vehicles  # List[BaseVehicle]

# 每辆车的状态
for vehicle in traffic_vehicles:
    vehicle.position        # np.array [x, y]
    vehicle.velocity        # np.array [vx, vy]
    vehicle.heading_theta   # float (radians)
    vehicle.speed           # float (m/s)

# 设置车辆状态（BaseVehicle 继承自 BaseObject）
vehicle.set_position([x, y])
vehicle.set_heading_theta(heading)
# 速度需通过底层 Bullet 物理引擎设置（见现有 set_state 中 _restore_velocity 的实现）

# 车辆唯一标识
vehicle.name  # str，如 "default_0"、"traffic_vehicle_3"
```

## 具体改动

### `envs/platoon_env.py` — `get_state()` 增加背景交通流保存

在现有 `get_state()` 方法的 return dict 中新增 `"traffic_states"` 字段：

```python
def get_state(self) -> dict:
    # ... 现有的 vehicle_states（编队车）保存逻辑不变 ...

    # ── 新增：保存背景交通流状态 ──
    traffic_states = {}
    try:
        if hasattr(self, 'engine') and self.engine is not None:
            tm = getattr(self.engine, 'traffic_manager', None)
            if tm is not None:
                for vehicle in tm.traffic_vehicles:
                    traffic_states[vehicle.name] = {
                        "position": np.asarray(vehicle.position, dtype=np.float64).copy(),
                        "heading": float(vehicle.heading_theta),
                        "velocity": np.asarray(
                            getattr(vehicle, "velocity", (0.0, 0.0)),
                            dtype=np.float64
                        ).copy(),
                    }
    except Exception:
        pass  # 在 ToyEnv 或无 engine 环境中安全跳过

    return {
        "vehicle_states": vehicle_states,
        "traffic_states": traffic_states,   # ← 新增
        "_last_actions": last_actions,
        "_last_progress_refs": last_progress,
        "_last_info": last_info,
    }
```

### `envs/platoon_env.py` — `set_state()` 增加背景交通流恢复

在现有 `set_state()` 方法末尾新增背景交通流恢复逻辑：

```python
def set_state(self, state: dict) -> None:
    # ... 现有的编队车恢复逻辑不变 ...

    # ── 新增：恢复背景交通流状态 ──
    traffic_states = state.get("traffic_states", {})
    if traffic_states:
        try:
            tm = getattr(self.engine, 'traffic_manager', None) if hasattr(self, 'engine') and self.engine is not None else None
            if tm is not None:
                # 建立 name → vehicle 的快速查找表
                name_to_vehicle = {v.name: v for v in tm.traffic_vehicles}
                for vname, vstate in traffic_states.items():
                    vehicle = name_to_vehicle.get(vname)
                    if vehicle is None:
                        continue
                    # 恢复位置
                    position = np.asarray(vstate["position"], dtype=np.float64)
                    if hasattr(vehicle, "set_position"):
                        vehicle.set_position(position.tolist())
                    # 恢复航向
                    heading = float(vstate["heading"])
                    if hasattr(vehicle, "set_heading_theta"):
                        vehicle.set_heading_theta(heading)
                    # 恢复速度（复用现有 _restore_velocity 的逻辑模式）
                    velocity = np.asarray(vstate["velocity"], dtype=np.float64)
                    if hasattr(vehicle, "set_velocity"):
                        vehicle.set_velocity(velocity.tolist())
                    else:
                        chassis = getattr(vehicle, "chassis", None)
                        node = chassis.node() if chassis is not None and hasattr(chassis, "node") else None
                        if node is not None and hasattr(node, "setLinearVelocity"):
                            try:
                                from panda3d.core import Vec3
                                node.setLinearVelocity(Vec3(float(velocity[0]), float(velocity[1]), 0.0))
                            except Exception:
                                pass
        except Exception:
            pass  # 安全降级
```

### `ToyEnv` 适配

`ToyEnv.get_state()` 已返回一个简单 dict，`set_state()` 也是空操作。无需修改——`traffic_states` 字段缺失时 `set_state()` 自动跳过。

## 验收标准

**交付物**：修改后的 `envs/platoon_env.py`、`tests/acceptance/test_phase6a_task1.py`（新建）

**验收指标**（6 项）：

1. **get_state() 返回 traffic_states 字段**：在 `PlatoonEnv` 真实环境中 reset 后，`get_state()` 返回的 dict 包含 `"traffic_states"` 键，值为非空 dict（因 `traffic_density=0.04`，至少有 0~2 辆背景车，故接受空 dict 但类型必须为 dict）
2. **traffic_states 格式正确**：每个背景车条目包含 `"position"`（长度 2 的数组）、`"heading"`（float）、`"velocity"`（长度 2 的数组）
3. **set_state() 可恢复背景车位置**：reset → get_state → step 若干步 → set_state → 再次 get_state，两次 get_state 的 `traffic_states` 中同名车辆的 position 差值 < 1.0（允许物理引擎微小误差）
4. **编队车恢复不受影响**：现有 `vehicle_states` 的保存/恢复行为不变，原有 Phase 5v2 的 `test_phase5v2_integration.py::test_env_state_roundtrip_toy` 仍通过
5. **ToyEnv 兼容**：`ToyEnv.get_state()` 和 `set_state()` 不报错（不要求包含 `traffic_states`）
6. **无 import 污染**：不在 `platoon_env.py` 顶层新增 import，所有 MetaDrive 内部 API 访问仅通过已有的 `self.engine` 属性

**验收命令**：
```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
PYTHONPATH="$PWD:$PYTHONPATH" /home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase6a_task1.py -v
```

## 测试代码框架

```python
"""tests/acceptance/test_phase6a_task1.py"""
import pytest
import numpy as np

try:
    from envs.platoon_env import PlatoonEnv
    METADRIVE_AVAILABLE = True
except Exception:
    METADRIVE_AVAILABLE = False

from train.train_platoon_rl import ToyEnv


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_get_state_contains_traffic_states():
    """验收指标 1：get_state 返回 traffic_states 字段"""
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        state = env.get_state()
        assert "traffic_states" in state
        assert isinstance(state["traffic_states"], dict)
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_traffic_state_format():
    """验收指标 2：traffic_states 格式正确"""
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        state = env.get_state()
        for vname, vstate in state["traffic_states"].items():
            assert isinstance(vname, str)
            assert "position" in vstate
            assert "heading" in vstate
            assert "velocity" in vstate
            pos = np.asarray(vstate["position"])
            vel = np.asarray(vstate["velocity"])
            assert pos.shape == (2,) or pos.shape == (3,)  # MetaDrive 可能返回 [x,y] 或 [x,y,z]
            assert isinstance(vstate["heading"], float)
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_set_state_restores_traffic_positions():
    """验收指标 3：set_state 可恢复背景车位置"""
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        saved = env.get_state()
        # 执行若干步让背景车移动
        dummy_action = {f"agent{i}": np.zeros((8, 3), dtype=np.float32) for i in range(3)}
        for _ in range(3):
            try:
                env.step(dummy_action)
            except Exception:
                break
        # 恢复状态
        env.set_state(saved)
        restored = env.get_state()
        # 检查同名背景车的位置差
        for vname in saved["traffic_states"]:
            if vname not in restored["traffic_states"]:
                continue
            pos_saved = np.asarray(saved["traffic_states"][vname]["position"][:2])
            pos_restored = np.asarray(restored["traffic_states"][vname]["position"][:2])
            assert np.linalg.norm(pos_saved - pos_restored) < 1.0, (
                f"Traffic vehicle {vname} position drift: {pos_saved} vs {pos_restored}"
            )
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_agent_state_still_works():
    """验收指标 4：编队车恢复不受影响"""
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        saved = env.get_state()
        assert "vehicle_states" in saved
        assert len(saved["vehicle_states"]) == 3
        dummy_action = {f"agent{i}": np.zeros((8, 3), dtype=np.float32) for i in range(3)}
        try:
            env.step(dummy_action)
        except Exception:
            pass
        env.set_state(saved)
        restored = env.get_state()
        for agent_id in saved["vehicle_states"]:
            pos_s = np.asarray(saved["vehicle_states"][agent_id]["position"][:2])
            pos_r = np.asarray(restored["vehicle_states"][agent_id]["position"][:2])
            assert np.linalg.norm(pos_s - pos_r) < 0.5
    finally:
        env.close()


def test_toy_env_compatible():
    """验收指标 5：ToyEnv 兼容"""
    env = ToyEnv(num_agents=2, mode="platoon")
    state = env.get_state()
    env.set_state(state)  # 不应报错
```

## 注意事项

1. **MetaDrive position 维度**：`vehicle.position` 可能返回 `[x, y]` 或 `[x, y, z]`（3D），保存时用 `np.asarray(vehicle.position)` 取全量，恢复时 `set_position` 只需要 `[x, y]` 或完整坐标
2. **traffic_vehicles 可能为空**：低密度场景下可能没有背景车，`traffic_states` 为空 dict 是合法的
3. **set_state 中的车辆匹配**：使用 `vehicle.name` 作为匹配键。如果 `env.reset()` 重新创建了场景，背景车的 name 可能改变——但这不影响我们的场景（我们不再调用 reset，而是直接 set_state）
4. **异常安全**：所有交通流相关操作用 try/except 包裹，确保在 ToyEnv 或无 engine 环境中不报错
5. **不修改 MetaDrive 内部代码**：所有操作通过 `self.engine.traffic_manager.traffic_vehicles` 公开属性访问
