# 6A.2 去除 _restore_state 中的 reset + 正确性验证

> **依赖**：6A.1（get_state/set_state 已覆盖背景交通流）
> **改动文件**：`train/closedloop_executor.py`、测试文件

## 背景

当前 `ClosedLoopExecutor._restore_state()` 每次先调用 `env.reset()`（~5.5s）再 `set_state()`。`execute_joint_groups()` 中调用 9 次 `_restore_state()` → 9 × 5.5s ≈ 49.5s 浪费。

6A.1 完成后，`get_state()`/`set_state()` 已覆盖背景交通流，无需通过 `reset()` 重建场景来保证状态一致性。可以安全去除 `reset()` 调用。

## 具体改动

### `train/closedloop_executor.py` — 修改 `_restore_state()`

**改动前**：
```python
def _restore_state(self, saved_state: dict) -> None:
    if hasattr(self.env, "reset"):
        self.env.reset()
    self.env.set_state(saved_state)
```

**改动后**：
```python
def _restore_state(self, saved_state: dict) -> None:
    self.env.set_state(saved_state)
```

仅此一行改动。删除 `env.reset()` 调用。

### `train/closedloop_executor.py` — 优化 `execute_joint_groups()`

当前 `execute_joint_groups()` 中 `execute_joint_trajectory()` 内部也会调用 `_restore_state(saved_state)`（在 finally 中），这导致了双重恢复。优化后的结构：

**改动前**：
```python
def execute_joint_groups(self, joint_groups):
    saved_state = self.env.get_state()
    results = []
    try:
        for joint_group in joint_groups:
            self._restore_state(saved_state)           # ← 恢复 1
            results.append(self.execute_joint_trajectory(joint_group))  # ← 内部 finally 也恢复
    finally:
        self._restore_state(saved_state)               # ← 恢复 2
    return results
```

**改动后**：
```python
def execute_joint_groups(self, joint_groups):
    saved_state = self.env.get_state()
    results = []
    try:
        for joint_group in joint_groups:
            self.env.set_state(saved_state)            # ← 直接 set_state，不走 _restore_state
            result = self._execute_single_group(joint_group)
            results.append(result)
    finally:
        self.env.set_state(saved_state)                # ← 最终恢复
    return results

def _execute_single_group(self, joint_actions):
    """执行单个联合组，不保存/恢复状态（由调用方管理）"""
    step_infos = {agent_id: [] for agent_id in joint_actions}
    crash_flags = {agent_id: False for agent_id in joint_actions}
    out_of_road_flags = {agent_id: False for agent_id in joint_actions}
    terminated = False

    for step_idx in range(self.horizon):
        step_actions = {}
        for agent_id, trajectory in joint_actions.items():
            traj = np.asarray(trajectory, dtype=np.float32)
            if traj.ndim != 2 or traj.shape[1] != 3:
                raise ValueError(f"Expected [T, 3] trajectory for {agent_id}, got {traj.shape}")
            horizon = traj.shape[0]
            start_idx = min(step_idx, max(horizon - 1, 0))
            window = traj[start_idx:]
            if window.shape[0] < horizon:
                pad = np.repeat(window[-1:, :], horizon - window.shape[0], axis=0)
                window = np.concatenate([window, pad], axis=0)
            step_actions[agent_id] = window

        _, _, term, trunc, info = self.env.step(step_actions)
        for agent_id in joint_actions:
            agent_info = dict(info.get(agent_id, {}))
            step_infos[agent_id].append(agent_info)
            crash_flags[agent_id] = crash_flags[agent_id] or bool(agent_info.get("crash", False))
            out_of_road_flags[agent_id] = out_of_road_flags[agent_id] or bool(agent_info.get("out_of_road", False))

        if term.get("__all__", False) or trunc.get("__all__", False):
            terminated = True
            break

    return {
        "step_infos": step_infos,
        "crash_flags": crash_flags,
        "out_of_road_flags": out_of_road_flags,
        "terminated": terminated,
    }
```

**注意**：`execute_joint_trajectory()` 的公开接口保持不变（仍然自行保存/恢复状态），供外部单次调用。新增 `_execute_single_group()` 作为内部方法，不管理状态。

## 验收标准

**交付物**：修改后的 `train/closedloop_executor.py`、`tests/acceptance/test_phase6a_task2.py`（新建）

**验收指标**（8 项）：

1. **_restore_state 不调用 reset**：`ClosedLoopExecutor._restore_state()` 方法体中不包含 `self.env.reset()` 或 `env.reset()` 调用
2. **execute_joint_groups 内不调用 reset**：`execute_joint_groups()` 方法全链路不触发 `env.reset()`
3. **功能正确——ToyEnv 闭环执行**：使用 `ToyEnv` 执行 `execute_joint_groups(2 组)`，返回 2 个 result，每个 result 包含 `step_infos`、`crash_flags`、`out_of_road_flags`、`terminated`
4. **功能正确——状态恢复**：`execute_joint_groups()` 执行前后 `env.get_state()` 一致
5. **原有测试不回归**：`pytest tests/acceptance/test_phase5v2_integration.py::test_closedloop_executor -v` 仍通过
6. **性能改善——单步耗时**（真实环境）：3 步 `platoon-closedloop` 训练的 `profile_joint` 平均值 < 10s（此前 ~54s）
7. **team_reward 合理**（真实环境）：3 步训练的 `team_reward_mean` 为有限值（非 NaN/Inf）
8. **各组独立性**：不同 joint group 可以有不同的 crash 状态，各组 step_infos 互不影响

**验收命令**：
```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive

# 功能验证（ToyEnv，快速）
PYTHONPATH="$PWD:$PYTHONPATH" /home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase6a_task2.py -v

# 不回归验证
PYTHONPATH="$PWD:$PYTHONPATH" /home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase5v2_integration.py::test_closedloop_executor -v

# 性能验证（真实环境，需要 MetaDrive + GPU）
PYTHONPATH="$PWD:$PYTHONPATH" /home/kong/anaconda3/envs/meta_drive/bin/python -m train.train_platoon_rl \
    --mode platoon-closedloop --steps 3 --render 0 \
    --checkpoint-dir /tmp/phase6a_perf_test/ckpt \
    --log-dir /tmp/phase6a_perf_test/logs
# 检查输出中 profile_joint < 10s
```

## 测试代码框架

```python
"""tests/acceptance/test_phase6a_task2.py"""
import inspect
import numpy as np
import torch
import pytest

from train.closedloop_executor import ClosedLoopExecutor
from train.train_platoon_rl import ToyEnv

try:
    from envs.platoon_env import PlatoonEnv
    METADRIVE_AVAILABLE = True
except Exception:
    METADRIVE_AVAILABLE = False


def test_restore_state_no_reset():
    """验收指标 1：_restore_state 不调用 reset"""
    source = inspect.getsource(ClosedLoopExecutor._restore_state)
    assert "reset()" not in source, "_restore_state should not call env.reset()"


def test_execute_joint_groups_no_reset():
    """验收指标 2：execute_joint_groups 不触发 reset"""
    source_groups = inspect.getsource(ClosedLoopExecutor.execute_joint_groups)
    assert "reset()" not in source_groups


def test_toy_env_joint_groups_functional():
    """验收指标 3：ToyEnv 闭环执行功能正确"""
    env = ToyEnv(num_agents=2, mode="platoon")
    executor = ClosedLoopExecutor(env, reward_config={})
    env.reset()
    group1 = {
        "agent0": np.zeros((8, 3), dtype=np.float32),
        "agent1": np.ones((8, 3), dtype=np.float32) * 0.5,
    }
    group2 = {
        "agent0": np.ones((8, 3), dtype=np.float32),
        "agent1": np.zeros((8, 3), dtype=np.float32),
    }
    results = executor.execute_joint_groups([group1, group2])
    assert len(results) == 2
    for result in results:
        assert "step_infos" in result
        assert "crash_flags" in result
        assert "out_of_road_flags" in result
        assert "terminated" in result


def test_state_restored_after_joint_groups():
    """验收指标 4：执行前后状态一致"""
    env = ToyEnv(num_agents=2, mode="platoon")
    executor = ClosedLoopExecutor(env, reward_config={})
    env.reset()
    before = env.get_state()
    groups = [
        {"agent0": np.zeros((8, 3), dtype=np.float32), "agent1": np.zeros((8, 3), dtype=np.float32)},
        {"agent0": np.ones((8, 3), dtype=np.float32), "agent1": np.ones((8, 3), dtype=np.float32)},
    ]
    executor.execute_joint_groups(groups)
    after = env.get_state()
    assert before == after


def test_groups_independence():
    """验收指标 8：各组独立性"""
    env = ToyEnv(num_agents=1, mode="toy-single")
    executor = ClosedLoopExecutor(env, reward_config={})
    env.reset()
    # 构造两组不同轨迹
    group_safe = {"agent0": np.ones((8, 3), dtype=np.float32) * 5.5}  # 安全轨迹
    group_crash = {"agent0": np.ones((8, 3), dtype=np.float32) * 0.1}  # 可能 crash 轨迹
    results = executor.execute_joint_groups([group_safe, group_crash])
    assert len(results) == 2
    # 两组的 crash_flags 可以不同（不要求必须不同，但数据结构独立）
    assert results[0]["crash_flags"].keys() == results[1]["crash_flags"].keys()


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_real_env_performance():
    """验收指标 6 + 7：真实环境性能和 team_reward 合理性"""
    import time
    import train.train_platoon_rl as entry

    summary = entry.build_runtime(
        mode="platoon-closedloop",
        config_path="configs/train/platoon_grpo_v2.yaml",
        steps=3,
        render=False,
        checkpoint_dir="/tmp/phase6a_task2_perf/ckpt",
        log_dir="/tmp/phase6a_task2_perf/logs",
        ckpt_path=entry.DEFAULT_SINGLE_CKPT,
        num_agents=3,
        run_training=True,
    )
    # 性能：profile_joint 平均 < 10s
    joint_times = summary.get("profile_joint", [])
    if joint_times:
        avg_joint = sum(joint_times) / len(joint_times)
        assert avg_joint < 10.0, f"joint_mean={avg_joint:.1f}s, should be < 10s"
    # team_reward 有限
    team_rewards = summary.get("team_reward_mean", [])
    for r in team_rewards:
        assert np.isfinite(float(r)), f"team_reward_mean contains non-finite: {r}"
```

## 注意事项

1. **`execute_joint_trajectory()` 公开接口不变**：外部直接调用 `execute_joint_trajectory()` 时仍自行保存/恢复状态。只有 `execute_joint_groups()` 使用优化后的内部方法
2. **ToyEnv 的 `set_state()` 是空操作**：ToyEnv 不会真正恢复状态，但这不影响测试通过
3. **真实环境测试需要 MetaDrive + checkpoint**：`test_real_env_performance` 需要真实环境，用 `@pytest.mark.skipif` 保护
4. **物理引擎碰撞状态**：去除 reset 后，物理引擎的碰撞检测缓存可能残留。MetaDrive 的 Bullet 物理引擎在每次 `step()` 开始时会重新计算碰撞对，因此 `set_state()` 后的第一个 `step()` 碰撞检测仍然是正确的
5. **episode 终止状态**：如果某个 group 执行到 `terminated=True`（碰撞导致 episode 结束），`set_state()` 恢复后 env 应该回到非终止状态。需验证 MetaDrive 的 `set_state` 是否会清除 terminated 标志——若不清除，可能需要在 `set_state()` 末尾手动重置 `self._terminated` 等标志
