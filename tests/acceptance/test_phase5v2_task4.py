from __future__ import annotations

import copy

import numpy as np

from train.closedloop_executor import ClosedLoopExecutor


class StubEnv:
    def __init__(self):
        self._state = {"tick": 0, "position": {"agent0": 0.0, "agent1": 0.0}}

    def get_state(self):
        return copy.deepcopy(self._state)

    def set_state(self, state):
        self._state = copy.deepcopy(state)

    def step(self, actions):
        info = {}
        for agent_id, traj in actions.items():
            advance = float(np.asarray(traj)[0, 0]) if len(traj) > 0 else 0.0
            self._state["position"][agent_id] += advance
            crash = bool(agent_id == "agent1" and advance > 1.5)
            info[agent_id] = {
                "progress": advance,
                "formation_error": abs(self._state["position"]["agent0"] - self._state["position"]["agent1"]),
                "min_gap": 0.5 if crash else 5.0,
                "jerk": 0.0,
                "delta_steering": 0.0,
                "crash": crash,
                "out_of_road": False,
            }
        self._state["tick"] += 1
        terminated = {agent_id: bool(agent_info["crash"]) for agent_id, agent_info in info.items()}
        truncated = {agent_id: False for agent_id in info}
        terminated["__all__"] = any(terminated.values())
        truncated["__all__"] = False
        reward = {agent_id: 0.0 for agent_id in info}
        obs = {agent_id: {} for agent_id in info}
        return obs, reward, terminated, truncated, info


def _joint_actions(step_value: float) -> dict[str, np.ndarray]:
    return {
        "agent0": np.full((8, 3), [step_value, 0.0, 0.0], dtype=np.float32),
        "agent1": np.full((8, 3), [step_value, 0.0, 0.0], dtype=np.float32),
    }


def test_closed_loop_executor_executes_and_restores_env_state():
    env = StubEnv()
    executor = ClosedLoopExecutor(env, reward_config={}, horizon=8)
    before = env.get_state()

    result = executor.execute_joint_trajectory(_joint_actions(1.0))

    assert set(result.keys()) == {"step_infos", "crash_flags", "out_of_road_flags", "terminated", "profile"}
    assert set(result["profile"].keys()) == {"restore_reset", "restore_set_state", "restore_total", "step_execution"}
    assert len(result["step_infos"]["agent0"]) == 8
    assert env.get_state() == before


def test_closed_loop_executor_marks_crash_and_early_termination():
    env = StubEnv()
    executor = ClosedLoopExecutor(env, reward_config={}, horizon=8)

    result = executor.execute_joint_trajectory(_joint_actions(2.0))

    assert result["crash_flags"]["agent1"] is True
    assert result["terminated"] is True
    assert len(result["step_infos"]["agent0"]) <= 8


def test_execute_joint_groups_returns_independent_results():
    env = StubEnv()
    executor = ClosedLoopExecutor(env, reward_config={}, horizon=8)

    results = executor.execute_joint_groups([_joint_actions(1.0), _joint_actions(2.0)])

    assert len(results) == 2
    assert results[0]["crash_flags"]["agent1"] is False
    assert results[1]["crash_flags"]["agent1"] is True
    assert env.get_state() == {"tick": 0, "position": {"agent0": 0.0, "agent1": 0.0}}
