import numpy as np

from envs.co_preference_platoon_env import CoPreferencePlatoonEnv


class _FakeBaseEnv:
    def __init__(self):
        self.last_actions = None
        self._obs = {
            "agent0": {
                "camera": np.zeros((3, 4, 4), dtype=np.float32),
                "lidar": np.zeros((1, 4, 4), dtype=np.float32),
                "status": np.zeros((8,), dtype=np.float32),
                "formation_relation_state": np.zeros((12,), dtype=np.float32),
                "topology_polylines": {
                    "current": np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
                    "left": np.asarray([[0.0, 3.5], [10.0, 3.5]], dtype=np.float32),
                    "right": None,
                    "branch": None,
                },
            }
        }

    def reset(self):
        return self._obs

    def step(self, actions):
        self.last_actions = actions
        return self._obs, {"agent0": 0.0}, {"agent0": False, "__all__": False}, {"agent0": False, "__all__": False}, {
            "agent0": {"progress": 1.0}
        }


class _FakePlanner:
    def __init__(self):
        self.preference_points = None

    def forward_with_preference(self, batch, preference_points):
        self.preference_points = preference_points
        return {"agent0": np.ones((8, 3), dtype=np.float32)}


def test_co_preference_env_maps_action_to_preference_point_and_steps_planner() -> None:
    base_env = _FakeBaseEnv()
    planner = _FakePlanner()
    env = CoPreferencePlatoonEnv({"base_env": base_env, "planner": planner})

    obs = env.reset()
    next_obs, reward, terminated, truncated, info = env.step({"agent0": {"topology_choice": 1, "s": 0.5}})

    np.testing.assert_allclose(planner.preference_points["agent0"], np.asarray([5.0, 3.5], dtype=np.float32))
    np.testing.assert_allclose(base_env.last_actions["agent0"], np.ones((8, 3), dtype=np.float32))
    assert "co_preference_target_point" in info["agent0"]
    assert "co_preference_reward" in info["agent0"]
    assert "agent0" in next_obs
    assert "agent0" in reward
    assert terminated["__all__"] is False
    assert truncated["__all__"] is False
