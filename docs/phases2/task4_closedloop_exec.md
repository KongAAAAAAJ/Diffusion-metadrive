# 5v2.4 闭环轨迹执行器

> **依赖**：5v2.3（env state save/load）
> **改动文件**：`train/closedloop_executor.py`（新建）, `envs/platoon_env.py`（小改）

## 背景

当前 surrogate 评估（`evaluate_trajectory_group`）使用静态 other_poses 估算 reward，无法捕捉多车交互。闭环执行器将轨迹实际执行到环境中，获得真实的交互 reward。

## 设计

```
输入：env（保存状态后）+ joint_actions {agent_id: trajectory [8, 3]}
输出：step_infos, total_reward, crash_flag, formation_metrics

流程：
  1. env.get_state() 保存起点
  2. 逐步执行 joint_actions（8 步）
  3. 每步收集 info（progress, formation_error, min_gap, crash, ...）
  4. env.set_state(saved) 恢复起点
  5. 返回执行结果
```

## 具体代码

### 新建 `train/closedloop_executor.py`

```python
"""闭环轨迹执行器：在环境中实际执行联合轨迹，收集真实交互 reward。"""

from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np


class ClosedLoopExecutor:
    """在 PlatoonEnv 中闭环执行联合轨迹组。

    Parameters
    ----------
    env : PlatoonEnv
        编队环境实例。
    reward_config : dict
        reward 计算参数，传给 compute_trajectory_reward。
    horizon : int
        每条轨迹的步数（默认 8，与 diffusion output 一致）。
    """

    def __init__(self, env, reward_config: dict, horizon: int = 8):
        self.env = env
        self.reward_config = dict(reward_config or {})
        self.horizon = horizon

    def execute_joint_trajectory(
        self,
        joint_actions: Dict[str, np.ndarray],
    ) -> dict:
        """在环境中实际执行一组联合轨迹（所有车同时行动）。

        Parameters
        ----------
        joint_actions : dict[agent_id -> np.ndarray [8, 3]]
            每辆车的规划轨迹。

        Returns
        -------
        result : dict
            - "step_infos": dict[agent_id -> list[dict]]  # 每步 info
            - "crash_flags": dict[agent_id -> bool]
            - "out_of_road_flags": dict[agent_id -> bool]
            - "terminated": bool  # 是否提前终止
        """
        # 实现要点：
        # 1. 保存当前 env 状态
        saved_state = self.env.get_state()

        step_infos: Dict[str, List[dict]] = {aid: [] for aid in joint_actions}
        crash_flags: Dict[str, bool] = {aid: False for aid in joint_actions}
        out_flags: Dict[str, bool] = {aid: False for aid in joint_actions}
        terminated = False

        try:
            for step_idx in range(self.horizon):
                # 构建当前步的动作：取轨迹的第 step_idx 个点
                step_actions = {}
                for agent_id, traj in joint_actions.items():
                    # traj shape [8, 3] → 取第 step_idx 步
                    step_actions[agent_id] = traj[step_idx:step_idx+1]  # [1, 3]

                # 执行一步
                # 注意：env.step() 接受完整轨迹 [8, 3]，内部会转为控制量
                # 这里需要逐步执行，传入单步轨迹
                # 但 trajectory_to_control 需要 [8, 3]，所以传入从当前步到末尾的子轨迹
                sub_traj_actions = {}
                for agent_id, traj in joint_actions.items():
                    sub_traj_actions[agent_id] = traj[step_idx:]  # 从当前步到末尾

                obs, reward, term, trunc, info = self.env.step(sub_traj_actions)

                # 收集每辆车的 info
                for agent_id in joint_actions:
                    agent_info = info.get(agent_id, {})
                    step_infos[agent_id].append(agent_info)
                    if agent_info.get("crash", False):
                        crash_flags[agent_id] = True
                    if agent_info.get("out_of_road", False):
                        out_flags[agent_id] = True

                # 检查终止
                if term.get("__all__", False) or trunc.get("__all__", False):
                    terminated = True
                    break

        finally:
            # 无论如何都恢复状态
            self.env.set_state(saved_state)

        return {
            "step_infos": step_infos,
            "crash_flags": crash_flags,
            "out_of_road_flags": out_flags,
            "terminated": terminated,
        }

    def execute_joint_groups(
        self,
        joint_groups: List[Dict[str, np.ndarray]],
    ) -> List[dict]:
        """批量执行多个联合组，每组执行后恢复状态。

        Parameters
        ----------
        joint_groups : list of dict[agent_id -> np.ndarray [8, 3]]
            M 个联合组，每个组包含所有车的轨迹。

        Returns
        -------
        results : list[dict]
            每个联合组的执行结果。
        """
        # 保存一次起点
        saved_state = self.env.get_state()
        results = []

        for group in joint_groups:
            # 恢复到起点
            self.env.set_state(saved_state)
            result = self.execute_joint_trajectory(group)
            results.append(result)

        # 最终恢复到起点
        self.env.set_state(saved_state)
        return results
```

### 闭环执行的 step 方式说明

**当前 `env.step()` 的行为**（platoon_env.py line 289-301）：
- 接受 `{agent_id: trajectory [8, 3]}`
- 内部调用 `trajectory_to_control()` 转为控制量 `[2]`（steering, accel）
- 只执行**一步**低层控制

**闭环执行的两种策略**：

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **A: 逐步执行** | 每步取轨迹第 t 个点，转控制量执行 | 真实闭环 | 需要 8 次 env.step |
| **B: 一次执行** | 直接传完整轨迹，让 env 执行 1 步 | 简单 | 不是真正的多步闭环 |

**推荐策略 A**，但需要注意：
- `trajectory_to_control()` 取轨迹第一个点做 PD/LQR，所以逐步传入 `traj[step_idx:]` 是正确的
- 8 步闭环执行意味着每个联合组需要 8 次 `env.step()`
- M=8 组 → 64 次 `env.step()`，在 MetaDrive 中约 1-2 秒

## 验收标准

**交付物**：`train/closedloop_executor.py`、`tests/acceptance/test_phase5v2_task4.py`

**验收指标**（7 项）：
1. `ClosedLoopExecutor` 类存在，构造函数接受 env + reward_config
2. `execute_joint_trajectory()` 返回 dict 含 step_infos, crash_flags, out_of_road_flags
3. step_infos 每个 agent 有 ≤ 8 步（可因 early termination 少于 8）
4. 执行后 env 状态恢复（get_state() 与执行前一致）
5. crash 场景正确标记
6. `execute_joint_groups(M_groups)` 返回 M 个 result
7. 每个 result 的 crash_flags 独立（不同组可以有不同 crash 状态）

**验收命令**：
```bash
pytest tests/acceptance/test_phase5v2_task4.py -v
```

## 注意事项

1. **env.step() 只执行一步**：当前 `PlatoonEnv.step()` 接受 `[8, 3]` 轨迹但只执行一步控制。闭环执行需要调用 8 次 `env.step()` 才能走完整条轨迹
2. **逐步执行时的轨迹截断**：第 t 步应该传入 `traj[t:]`（从第 t 步到末尾），而非只传 `traj[t:t+1]`，因为 `trajectory_to_control()` 需要后续点来计算曲率
3. **ToyEnv 适配**：在 `ToyEnv` 中也实现 `get_state()` / `set_state()`（stub），以便测试
4. **性能优化**：如果 8 步逐步执行太慢，可以考虑只执行前 3-4 步（短视闭环），用 surrogate 补全剩余步。但首先实现完整 8 步版本
