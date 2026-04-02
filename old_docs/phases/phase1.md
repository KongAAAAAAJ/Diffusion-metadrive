# Phase 1：MetaDrive 编队环境封装

> **前置依赖**：Phase 0
> **后续 Phase**：Phase 4（需要 1.1~1.7 全部完成）、Phase 5（需要 1.6 info 字段）、Phase 6（需要 1.7 场景集成）
> **可并行**：Phase 2、Phase 3

## 目标
完成多车编队环境、危险场景接口、指标系统，并通过 IDM 跟驰验证。

---

## 全局约定（本 Phase 所需）

### 环境与路径
```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
# Python 解释器：/home/kong/anaconda3/envs/meta_drive/bin/python
```

### 关键 import
```python
# 编队环境基类（Phase 1 必须继承此类，不是 MultiAgentMetaDrive）
from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv
# IDM 跟驰策略（验收用）—— 构造函数：IDMPolicy(control_object, random_seed)
from metadrive.policy.idm_policy import IDMPolicy
# 指标
from evaluation.platoon_metrics import PlatoonMetrics
```

### formation_relation_state 定义（冻结）
每辆车相对邻居的 6 维向量 × 最多 2 邻居 = **12 维**：
- `Δx`（纵向距离，ego-local，前正后负）、`Δy`（横向）、`Δheading`、`Δv`
- `Δx_target`：目标纵向间距（**目标值**，非偏差；如 0.5s × v_ego）
- `Δy_target`：目标横向偏移（纵列队形下通常为 0）

不足邻居补零。

### 硬件约束
- GPU：RTX 4080 16GB
- Headless 运行：`use_render=False`
- 已知问题：Panda3D 退出时可能 segfault（exit code 139），不影响 Python 逻辑

---

## 核心接口签名

```python
# envs/platoon_env.py
class PlatoonEnv(BaseMultiEnv):
    """3 车编队环境。
    action 空间：
      - 轨迹模式：{agent_id: np.ndarray (8, 3)} — 环境内部 LQR+PD 转为 steer/throttle
      - 低层模式：{agent_id: np.ndarray (2,)} — 直接 [steer, throttle]
    """
    def __init__(self, config: dict): ...
    def reset(self) -> dict[str, dict]: ...          # {agent_id: obs_dict}
    def step(self, actions) -> tuple[dict, dict, dict, dict, dict]: ...
    def get_formation_relation_state(self, agent_id: str) -> np.ndarray: ...  # (12,)
    def get_platoon_metrics(self) -> dict: ...

# evaluation/platoon_metrics.py
class PlatoonMetrics:
    def start_episode(self): ...
    def update(self, info: dict): ...
    def end_episode(self): ...
    def compute(self) -> dict: ...  # {success_rate, collision_rate, formation_error, recovery_time, min_inter_vehicle_gap}

# scenarios/hazard_scenarios.py
def get_hazard_scenario_configs() -> list[dict]: ...
```

---

## 子任务与验收指标

### ✅ 1.1 创建 PlatoonEnv 骨架
- [x] 完成
- **交付物**：`envs/__init__.py`、`envs/platoon_env.py`
- **验收指标**（`tests/acceptance/test_phase1_task1.py`，10 项）：
  1. `PlatoonEnv` 继承 `BaseMultiEnv`（不是 `MultiAgentMetaDrive`）
  2. `reset()` 返回 dict，含 3 个 agent key：`agent0, agent1, agent2`
  3. 每个 agent obs 为 dict 或 ndarray，非 None
  4. `get_formation_relation_state("agent0")` → shape `(12,)`
  5. `step({id: np.zeros((2,))})` 返回 5-tuple
  6. `step({id: np.zeros((8,3))})` 返回 5-tuple
  7. `info[agent_id]` 含 `control_mode` 字段
  8. `info[agent_id]` 含 `formation_error` 和 `min_gap`（float）
  9. `terminated` 和 `truncated` 含 `"__all__"`
  10. 初始间距 = `vehicle_length + headway_time × speed`，误差 < 1.0m
- **验收命令**：`pytest tests/acceptance/test_phase1_task1.py -v`（10 PASSED）

### ✅ 1.2 创建 PlatoonMetrics
- [x] 完成
- **交付物**：`evaluation/__init__.py`、`evaluation/platoon_metrics.py`
- **验收指标**（`tests/acceptance/test_phase1_task2.py`，6 项）：
  1. 有 `start_episode()`、`update()`、`end_episode()`、`compute()` 四个方法
  2. `compute()` 返回含 5 key 的 dict，值均为 float
  3. 精度测试：2 episode 喂入已知数据，验证 `success_rate=0.5, collision_rate=0.5, formation_error=2.0, min_inter_vehicle_gap=2.0`（容差 1e-3）
  4. 空 episode `compute()` 返回全 0.0
  5. recovery_time 测试：5 步超阈 + 1 步回阈 → recovery_time=5.0
  6. 所有返回值为 float
- **验收命令**：`pytest tests/acceptance/test_phase1_task2.py -v`（6 PASSED）

### ✅ 1.3 创建 HazardScenarios
- [x] 完成
- **交付物**：`scenarios/__init__.py`、`scenarios/hazard_scenarios.py`
- **验收指标**（`tests/acceptance/test_phase1_task3.py`，5 项）：
  1. `get_hazard_scenario_configs()` 返回 list，长度 ≥ 3
  2. 每个 config 含 `name`(str)、`description`(str)、`env_overrides`(dict)
  3. 包含：`static_obstacle_detour`、`dynamic_cut_in`、`bottleneck_narrow_bridge`
  4. 每个 `env_overrides` 至少 1 key
  5. 每个 `description` ≥ 10 字符
- **验收命令**：`pytest tests/acceptance/test_phase1_task3.py -v`（5 PASSED）

### ✅ 1.4 创建 verify_phase1.py
- [x] 完成
- **交付物**：`scripts/verify_phase1.py`
- **验收指标**（`tests/acceptance/test_phase1_task4.py`，5 项）：
  1. 文件无语法错误
  2. `parse_args(["--episodes","5","--render","0"])` → `episodes=5, render=0`
  3. `parse_args(["--max-steps","100"])` → `max_steps=100`
  4. import `PlatoonEnv` 和 `IDMPolicy`
  5. `main()` 接受 `argv` 参数
- **验收命令**：`pytest tests/acceptance/test_phase1_task4.py -v`（5 PASSED）

### ✅ 1.5 Phase 1 集成 rollout 验收
- [x] 完成
- **交付物**：`tests/acceptance/test_phase1_task5.py`、运行日志
- **验收指标**（硬性）：
  1. `verify_phase1.py --episodes 5 --render 0` 运行完成（退出码 0 或 139）
  2. summary 行 5 项指标不全为 0.000
  3. `collision_rate` = 0.0
  4. `formation_error` < 2.0
  5. `min_inter_vehicle_gap` > 0.0
  6. `success_rate` > 0.0
  7. 5 个 episode 均跑完
- **验收命令**：
  ```bash
  /home/kong/anaconda3/envs/meta_drive/bin/python scripts/verify_phase1.py --episodes 5 --render 0 2>&1 | tee logs/phase1_acceptance.log
  ```
  **实际结果**：
  - `episode=1 success=1 collision=0 formation_error=0.401 min_gap=9.212`
  - `episode=2 success=1 collision=0 formation_error=0.401 min_gap=9.212`
  - `episode=3 success=1 collision=0 formation_error=0.401 min_gap=9.212`
  - `episode=4 success=1 collision=0 formation_error=0.401 min_gap=9.212`
  - `episode=5 success=1 collision=0 formation_error=0.401 min_gap=9.212`
  - `summary: success_rate=1.000 collision_rate=0.000 formation_error=0.401 recovery_time=0.000 min_inter_vehicle_gap=9.212`

### ☐ 1.6 补充 info 字段（对接 Phase 5 奖励函数）
- [ ] 待完成
- **背景**：Phase 5 的 `compute_step_reward(info)` 需要 `progress`、`jerk`、`delta_steering` 等字段，当前 `_build_info_dict()` 未提供。
- **交付物**：更新 `envs/platoon_env.py`、`tests/acceptance/test_phase1_task6.py`
- **验收指标**（4 项）：
  1. `info[agent_id]` 包含：`formation_error`、`min_gap`、`crash`(bool)、`arrive_dest`(bool)、`out_of_road`(bool)、`progress`(float,Δs)、`jerk`(float)、`delta_steering`(float)、`speed_km_h`(float)
  2. IDM rollout 3 步后 `progress` > 0
  3. `jerk` 和 `delta_steering` 为有限值
- **验收命令**：`pytest tests/acceptance/test_phase1_task6.py -v`（4 PASSED）

### ☐ 1.7 HazardScenarios 集成到 PlatoonEnv
- [ ] 待完成
- **背景**：`hazard_scenarios.py` 当前是孤立模块。此任务将其连接到 env，使 Phase 5 curriculum 和 Phase 6 评估可切换场景。
- **交付物**：更新 `envs/platoon_env.py`、`tests/acceptance/test_phase1_task7.py`
- **验收指标**（4 项）：
  1. `PlatoonEnv({"hazard_scenario": "dynamic_cut_in"})` 可 reset
  2. `PlatoonEnv({"hazard_scenario": "static_obstacle_detour"})` 的 traffic_density 自动为 0.15
  3. `PlatoonEnv({"hazard_scenario": None})` 行为不变
  4. 所有 `get_hazard_scenario_configs()` 返回的场景名均可作为参数传入
- **验收命令**：`pytest tests/acceptance/test_phase1_task7.py -v`（4 PASSED）

---

## 跨 Phase 数据流（本 Phase 输出 → 下游消费）

```
PlatoonEnv.reset()/step() → obs dict ──→ Phase 4 PlatoonDiffusionPlanner.forward()
  lidar_state 模式: {"obs": ndarray, "formation_relation_state": ndarray[12]}
  multimodal 模式（Phase 4.0 升级后）: {"camera":[3,256,1024], "lidar":[1,256,256], "status":[8], "formation_relation_state":[12]}

PlatoonEnv.step() → info dict ──→ Phase 5 compute_step_reward(info)
  必须包含: progress, jerk, delta_steering, formation_error, min_gap, crash, arrive_dest, out_of_road, speed_km_h

PlatoonEnv({"hazard_scenario": "xxx"}) ──→ Phase 5 curriculum / Phase 6 evaluation
```
