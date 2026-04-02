# 6A.3 多进程并行闭环执行（可选）

> **依赖**：6A.2（_restore_state 已去除 reset，状态保存/恢复正确）
> **改动文件**：`train/closedloop_executor.py`、`train/train_platoon_rl.py`、`configs/train/platoon_grpo_v2.yaml`
> **状态**：⏳ 可选任务，6A.2 验证通过后再决定是否执行

## 背景

6A.2 完成后，闭环评估从 ~54s 降至 ~3s（单进程串行 8 组 × 8 步 env.step）。本任务进一步将 8 个 joint group 分发到多进程并行执行，利用 24 核 CPU 实现 ~0.5s 的目标。

## 设计

### 核心思路

每个 worker 进程持有独立的 `PlatoonEnv` 实例。主进程将 `saved_state` + `joint_actions` 发送给 worker，worker 在自己的 env 中 `set_state()` 后执行 8 步，返回结果。

```
主进程:
  saved_state = env.get_state()
  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
  │ W1  │ │ W2  │ │ W3  │ │ W4  │  ... (num_workers 个)
  │env1 │ │env2 │ │env3 │ │env4 │
  │grp1 │ │grp2 │ │grp3 │ │grp4 │
  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
     │       │       │       │
     └───────┴───────┴───────┘
              ↓
     results = [r1, r2, r3, r4, ...]
```

### 关键约束

1. **env 实例不可跨进程共享**：MetaDrive env 内部包含 Panda3D 渲染引擎和 Bullet 物理引擎，不可 pickle。每个 worker 必须自行创建 env
2. **saved_state 必须可 pickle**：6A.1 中保存的 state 是纯 Python dict + numpy array，可以 pickle
3. **worker 生命周期管理**：env 创建开销约 3-5s，不能每次都新建。使用持久化 worker pool

## 具体改动

### `train/closedloop_executor.py` — 新增 `ParallelClosedLoopExecutor`

```python
"""在 closedloop_executor.py 中新增 ParallelClosedLoopExecutor 类"""

import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import Dict, List, Optional

import numpy as np


def _worker_loop(env_config: dict, task_queue: Queue, result_queue: Queue, horizon: int):
    """Worker 进程主循环：创建独立 env，等待任务，执行后返回结果。"""
    from envs.platoon_env import PlatoonEnv
    env = PlatoonEnv(env_config)
    env.reset()

    while True:
        task = task_queue.get()
        if task is None:  # 毒丸信号 → 退出
            break
        task_id, saved_state, joint_actions = task
        try:
            env.set_state(saved_state)
            result = _execute_single_group_standalone(env, joint_actions, horizon)
            result_queue.put((task_id, result, None))
        except Exception as exc:
            result_queue.put((task_id, None, str(exc)))

    env.close()


def _execute_single_group_standalone(env, joint_actions: dict, horizon: int) -> dict:
    """在给定 env 中执行单个联合组（与 ClosedLoopExecutor._execute_single_group 逻辑一致）"""
    step_infos = {agent_id: [] for agent_id in joint_actions}
    crash_flags = {agent_id: False for agent_id in joint_actions}
    out_of_road_flags = {agent_id: False for agent_id in joint_actions}
    terminated = False

    for step_idx in range(horizon):
        step_actions = {}
        for agent_id, trajectory in joint_actions.items():
            traj = np.asarray(trajectory, dtype=np.float32)
            traj_len = traj.shape[0]
            start_idx = min(step_idx, max(traj_len - 1, 0))
            window = traj[start_idx:]
            if window.shape[0] < traj_len:
                pad = np.repeat(window[-1:, :], traj_len - window.shape[0], axis=0)
                window = np.concatenate([window, pad], axis=0)
            step_actions[agent_id] = window

        _, _, term, trunc, info = env.step(step_actions)
        for agent_id in joint_actions:
            agent_info = dict(info.get(agent_id, {}))
            step_infos[agent_id].append(agent_info)
            crash_flags[agent_id] = crash_flags[agent_id] or bool(agent_info.get("crash", False))
            out_of_road_flags[agent_id] = out_of_road_flags[agent_id] or bool(
                agent_info.get("out_of_road", False)
            )
        if term.get("__all__", False) or trunc.get("__all__", False):
            terminated = True
            break

    return {
        "step_infos": step_infos,
        "crash_flags": crash_flags,
        "out_of_road_flags": out_of_road_flags,
        "terminated": terminated,
    }


class ParallelClosedLoopExecutor:
    """多进程并行闭环执行器。

    Parameters
    ----------
    env : PlatoonEnv
        主进程环境实例（用于 get_state/set_state）。
    env_config : dict
        用于在 worker 中创建独立 env 的配置。
    reward_config : dict
        reward 计算参数。
    num_workers : int
        worker 进程数（默认 4）。
    horizon : int
        每条轨迹执行步数（默认 8）。
    """

    def __init__(
        self,
        env,
        env_config: dict,
        reward_config: dict,
        num_workers: int = 4,
        horizon: int = 8,
    ):
        self.env = env
        self.env_config = dict(env_config)
        self.reward_config = dict(reward_config or {})
        self.num_workers = int(num_workers)
        self.horizon = int(horizon)
        self._workers: List[Process] = []
        self._task_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._started = False

    def start(self):
        """启动 worker 进程池。"""
        if self._started:
            return
        ctx = mp.get_context("spawn")  # 避免 fork 与 CUDA/Panda3D 冲突
        self._task_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        for _ in range(self.num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(self.env_config, self._task_queue, self._result_queue, self.horizon),
                daemon=True,
            )
            p.start()
            self._workers.append(p)
        self._started = True

    def stop(self):
        """停止所有 worker 进程。"""
        if not self._started:
            return
        for _ in self._workers:
            self._task_queue.put(None)  # 毒丸
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._workers.clear()
        self._started = False

    def execute_joint_groups(self, joint_groups: List[Dict[str, np.ndarray]]) -> List[dict]:
        """并行执行多个联合组。"""
        if not self._started:
            self.start()

        saved_state = self.env.get_state()

        # 分发任务
        for task_id, group in enumerate(joint_groups):
            self._task_queue.put((task_id, saved_state, group))

        # 收集结果
        results_map = {}
        for _ in range(len(joint_groups)):
            task_id, result, error = self._result_queue.get(timeout=60)
            if error is not None:
                raise RuntimeError(f"Worker error on group {task_id}: {error}")
            results_map[task_id] = result

        # 恢复主进程 env 状态
        self.env.set_state(saved_state)

        # 按原始顺序返回
        return [results_map[i] for i in range(len(joint_groups))]

    def __del__(self):
        self.stop()
```

### `train/train_platoon_rl.py` — 在 `_build_joint_training_inputs()` 中支持并行执行器

```python
def _build_joint_training_inputs(trainer, rollouts):
    if not bool(trainer.config.get("use_closedloop", False)) or not hasattr(trainer.env, "get_state"):
        return None, None

    # ... 现有的 select_top_k + build_joint_groups 逻辑 ...

    joint_trajectories = [extract_joint_trajectories(group, rollouts) for group in joint_groups]

    # 根据配置选择串行或并行执行器
    num_closedloop_workers = int(trainer.config.get("closedloop_workers", 0))
    if num_closedloop_workers > 1:
        # 使用并行执行器
        if not hasattr(trainer, '_parallel_executor') or trainer._parallel_executor is None:
            from train.closedloop_executor import ParallelClosedLoopExecutor
            trainer._parallel_executor = ParallelClosedLoopExecutor(
                env=trainer.env,
                env_config=trainer.config.get("_env_config", {}),
                reward_config=trainer.reward_config,
                num_workers=num_closedloop_workers,
            )
        exec_results = trainer._parallel_executor.execute_joint_groups(joint_trajectories)
    else:
        # 使用串行执行器（默认）
        executor = ClosedLoopExecutor(trainer.env, trainer.reward_config)
        exec_results = executor.execute_joint_groups(joint_trajectories)

    team_rewards = [compute_team_reward(result["step_infos"], trainer.reward_config) for result in exec_results]
    return joint_groups, team_rewards
```

### `configs/train/platoon_grpo_v2.yaml` — 新增并行配置

在现有配置末尾追加：

```yaml
# 并行闭环执行配置（6A.3）
closedloop_workers: 0  # 0 = 串行（默认），>1 = 并行 worker 数
```

### `train/train_platoon_rl.py` — 保存 env_config 供 worker 使用

在 `build_runtime()` 的 platoon 模式分支中，构建 `env_config` 后将其存入 config：

```python
elif mode in {"platoon", "platoon-closedloop"}:
    env_config_dict = {
        "observation_mode": "multimodal",
        "use_render": False,  # worker 中强制关闭渲染
        "num_agents": resolved_num_agents,
        "horizon": int(config.get("horizon", 100)),
        "traffic_density": float(config.get("traffic_density", 0.04)),
        "num_scenarios": int(config.get("num_scenarios", 1)),
        "use_hybrid_map": bool(config.get("use_hybrid_map", True)),
        "hybrid_map_sequence": str(config.get("hybrid_map_sequence", "SSXCOCSS")),
    }
    config["_env_config"] = env_config_dict
    env = _build_platoon_env(config, render=render, num_agents=resolved_num_agents)
    # ...
```

## 验收标准

**交付物**：修改后的 `train/closedloop_executor.py`、`train/train_platoon_rl.py`、`configs/train/platoon_grpo_v2.yaml`、`tests/acceptance/test_phase6a_task3.py`（新建）

**验收指标**（7 项）：

1. **ParallelClosedLoopExecutor 类存在**：`from train.closedloop_executor import ParallelClosedLoopExecutor` 不报错
2. **start/stop 生命周期**：可以正常 `start()` → `execute_joint_groups()` → `stop()`，不泄漏进程
3. **closedloop_workers=0 时行为不变**：默认使用串行执行器，profile_joint 与 6A.2 结果一致
4. **closedloop_workers=4 时功能正确**：4 worker 并行执行 8 个 joint group，返回结果数量正确，team_reward 为有限值
5. **并行与串行结果数值接近**：对同一组 joint_groups，并行和串行执行的 team_reward 差异 < 20%（物理引擎浮点误差导致的差异是可接受的）
6. **性能改善**：`closedloop_workers=4` 时，3 步训练的 `profile_joint` 平均值 < 2s（此前串行 ~3s）
7. **配置文件更新**：`platoon_grpo_v2.yaml` 包含 `closedloop_workers` 字段

**验收命令**：
```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive

# 功能验证
PYTHONPATH="$PWD:$PYTHONPATH" /home/kong/anaconda3/envs/meta_drive/bin/python -m pytest tests/acceptance/test_phase6a_task3.py -v

# 性能验证（串行基线）
PYTHONPATH="$PWD:$PYTHONPATH" /home/kong/anaconda3/envs/meta_drive/bin/python -m train.train_platoon_rl \
    --mode platoon-closedloop --steps 3 --render 0 \
    --checkpoint-dir /tmp/phase6a_serial/ckpt --log-dir /tmp/phase6a_serial/logs

# 性能验证（并行 4 workers）—— 需要在 yaml 中临时设 closedloop_workers: 4
PYTHONPATH="$PWD:$PYTHONPATH" /home/kong/anaconda3/envs/meta_drive/bin/python -m train.train_platoon_rl \
    --mode platoon-closedloop --steps 3 --render 0 \
    --checkpoint-dir /tmp/phase6a_parallel/ckpt --log-dir /tmp/phase6a_parallel/logs
```

## 测试代码框架

```python
"""tests/acceptance/test_phase6a_task3.py"""
import pytest
import numpy as np

from train.closedloop_executor import ClosedLoopExecutor

try:
    from train.closedloop_executor import ParallelClosedLoopExecutor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

try:
    from envs.platoon_env import PlatoonEnv
    METADRIVE_AVAILABLE = True
except Exception:
    METADRIVE_AVAILABLE = False


def test_parallel_executor_importable():
    """验收指标 1：ParallelClosedLoopExecutor 可导入"""
    from train.closedloop_executor import ParallelClosedLoopExecutor
    assert ParallelClosedLoopExecutor is not None


@pytest.mark.skipif(not METADRIVE_AVAILABLE or not PARALLEL_AVAILABLE, reason="MetaDrive or parallel executor unavailable")
def test_parallel_lifecycle():
    """验收指标 2：start/stop 生命周期正常"""
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        executor = ParallelClosedLoopExecutor(
            env=env,
            env_config={"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1},
            reward_config={},
            num_workers=2,
        )
        executor.start()
        executor.stop()
        # 不应有残留进程
        assert len(executor._workers) == 0
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE or not PARALLEL_AVAILABLE, reason="MetaDrive or parallel executor unavailable")
def test_parallel_execution_functional():
    """验收指标 4：并行执行功能正确"""
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    env_config = {"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1}
    try:
        env.reset()
        executor = ParallelClosedLoopExecutor(
            env=env, env_config=env_config, reward_config={}, num_workers=2,
        )
        groups = []
        for _ in range(4):
            groups.append({
                f"agent{i}": np.random.randn(8, 3).astype(np.float32) for i in range(3)
            })
        results = executor.execute_joint_groups(groups)
        assert len(results) == 4
        for r in results:
            assert "step_infos" in r
            assert "crash_flags" in r
        executor.stop()
    finally:
        env.close()


def test_config_has_closedloop_workers():
    """验收指标 7：配置文件包含 closedloop_workers"""
    import yaml
    from pathlib import Path
    data = yaml.safe_load(Path("configs/train/platoon_grpo_v2.yaml").read_text())
    assert "closedloop_workers" in data
```

## 注意事项

1. **使用 `spawn` context**：`mp.get_context("spawn")` 而非默认的 `fork`，避免 CUDA context 和 Panda3D 引擎在 fork 时出问题
2. **worker 中关闭渲染**：`use_render=False`，避免多进程同时争抢 GPU 渲染资源
3. **env_config 的序列化**：传递给 worker 的是纯 dict 配置，由 worker 自行创建 PlatoonEnv 实例
4. **超时保护**：`result_queue.get(timeout=60)` 防止 worker 卡死导致主进程永久等待
5. **daemon=True**：worker 设为 daemon，主进程退出时自动清理
6. **内存约束**：4 个 worker × ~1GB/env ≈ 4GB，在可用 40GB 内存中完全安全
7. **saved_state 通过 Queue 传递**：每次 `execute_joint_groups` 调用时序列化一次 state，通过 Queue 发送给所有 worker。state 中的 numpy array 会被 pickle 序列化
8. **本任务为可选**：如果 6A.2 完成后 ~3s/step 已足够，可以跳过本任务
