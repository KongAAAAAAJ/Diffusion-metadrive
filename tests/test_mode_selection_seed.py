from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.wrap_platoon_env import ModeSelectionSB3Env


class _FakeVehicle:
    speed_km_h = 0.0
    position = np.zeros((2,), dtype=np.float32)
    heading_theta = 0.0
    LENGTH = 4.5
    WIDTH = 2.0


class _SeedRecordingBaseEnv:
    def __init__(self):
        self.config = {}
        self.agents = {f"agent{i}": _FakeVehicle() for i in range(3)}
        self.start_index = 0
        self.num_scenarios = 0
        self.reset_seeds = []
        self.step_actions = []
        self._obs = {
            f"agent{i}": {
                "camera": np.zeros((3, 4, 4), dtype=np.float32),
                "lidar": np.zeros((1, 4, 4), dtype=np.float32),
                "status": np.full((8,), i, dtype=np.float32),
                "formation_relation_state": np.full((12,), i + 0.5, dtype=np.float32),
            }
            for i in range(3)
        }

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return self._obs

    def step(self, actions):
        self.step_actions.append(actions)
        reward = {f"agent{i}": 1.0 for i in range(3)}
        terminated = {f"agent{i}": False for i in range(3)}
        terminated["__all__"] = False
        truncated = {f"agent{i}": False for i in range(3)}
        truncated["__all__"] = False
        info = {f"agent{i}": {} for i in range(3)}
        return self._obs, reward, terminated, truncated, info


class _FakePlanner:
    def __init__(self):
        self.export_calls = 0
        self.batches = []
        self.config = SimpleNamespace(
            mode_keep_lane_count=1,
            mode_lane_change_left_count=1,
            mode_lane_change_right_count=1,
            mode_emergency_stop_count=1,
            target_line_num_points=8,
            target_point_prediction_horizon_s=4.0,
        )

    def export_mode_selection(self, batch):
        self.export_calls += 1
        self.batches.append(
            {
                agent_id: {
                    key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                    for key, value in sample.items()
                }
                for agent_id, sample in batch.items()
            }
        )
        agent_ids = list(batch)
        candidates = np.zeros((len(agent_ids), 4, 8, 3), dtype=np.float32)
        for agent_idx in range(len(agent_ids)):
            for mode_idx in range(4):
                candidates[agent_idx, mode_idx, :, 0] = float(mode_idx)
        logits = np.zeros((len(agent_ids), 4), dtype=np.float32)
        masks = np.ones((len(agent_ids), 4), dtype=bool)
        return {
            "agent_ids": agent_ids,
            "trajectory_candidates": candidates,
            "raw_cls_logits": logits,
            "masked_cls_logits": logits,
            "mode_valid_mask": masks,
            "pretrained_argmax_mode": np.zeros((len(agent_ids),), dtype=np.int64),
        }


def _make_env_and_planner():
    base_env = _SeedRecordingBaseEnv()
    planner = _FakePlanner()
    env = ModeSelectionSB3Env(
        {
            "base_env": base_env,
            "planner": planner,
            "num_agents": 3,
            "num_modes": 4,
            "seed": 100,
            "seed_offset": 7,
        }
    )

    def fake_dynamic_features():
        coarse = np.zeros((4, 8, 2), dtype=np.float32)
        for mode_idx in range(4):
            coarse[mode_idx, :, 0] = float(mode_idx)
        return {
            "coarse_trajectories": coarse,
            "mode_valid_mask": np.ones((4,), dtype=bool),
        }

    env._build_dynamic_mode_features_fallback = fake_dynamic_features  # type: ignore[method-assign]
    env._build_dynamic_mode_features = lambda vehicle: fake_dynamic_features()  # type: ignore[method-assign]
    return env, planner


def test_mode_selection_env_passes_reproducible_episode_seed_to_base_env():
    base_env = _SeedRecordingBaseEnv()
    env = ModeSelectionSB3Env(
        {
            "base_env": base_env,
            "planner": _FakePlanner(),
            "num_agents": 3,
            "num_modes": 4,
            "seed": 100,
            "seed_offset": 7,
        }
    )

    env.reset()
    env.reset()
    env.reset(seed=200)

    assert base_env.reset_seeds == [107, 108, 209]


def test_mode_selection_env_wraps_episode_seed_into_base_env_scenario_range():
    base_env = _SeedRecordingBaseEnv()
    base_env.start_index = 1
    base_env.num_scenarios = 1
    env = ModeSelectionSB3Env(
        {
            "base_env": base_env,
            "planner": _FakePlanner(),
            "num_agents": 3,
            "num_modes": 4,
            "seed": 1,
        }
    )

    env.reset()
    env.reset()
    env.reset(seed=7)

    assert base_env.reset_seeds == [1, 1, 1]


def test_step_executes_same_export_without_selected_mode_reexport():
    env, planner = _make_env_and_planner()

    env.reset()
    env.step(np.asarray([2, 1, 3], dtype=np.int64))

    assert planner.export_calls == 2


def test_selected_mode_endpoint_does_not_overwrite_preference_point():
    env, planner = _make_env_and_planner()

    env.reset()
    env.step(np.asarray([2, 1, 3], dtype=np.int64))

    first_export_batch = planner.batches[0]
    second_export_batch = planner.batches[1]
    assert first_export_batch["agent0"]["preference_point"].tolist() == [0.0, 0.0]
    assert second_export_batch["agent0"]["preference_point"].tolist() == [0.0, 0.0]
