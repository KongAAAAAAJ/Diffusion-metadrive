import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym  # type: ignore

from envs.wrap_platoon_env import ModeSelectionSB3Env
from envs.platoon_env import PlatoonEnv
from envs.reward_terms import compute_step_reward, compute_team_reward
from models.diffusion.test_transfuser_policy import _missing_controlled_agents


class _FakeBaseEnv:
    def __init__(self, config=None, drop_agent_after_step=None):
        self.config = dict(config or {})
        self.last_actions = None
        self.agents = {}
        self.drop_agent_after_step = drop_agent_after_step
        self._obs = {
            f"agent{i}": {
                "camera": np.zeros((3, 4, 4), dtype=np.float32),
                "lidar": np.zeros((1, 4, 4), dtype=np.float32),
                "status": np.full((8,), i, dtype=np.float32),
                "formation_relation_state": np.full((12,), i + 0.5, dtype=np.float32),
            }
            for i in range(3)
        }

    def reset(self):
        return self._obs

    def step(self, actions):
        self.last_actions = actions
        obs = dict(self._obs)
        if self.drop_agent_after_step is not None:
            obs.pop(self.drop_agent_after_step, None)
        return obs, {agent_id: float(idx) for idx, agent_id in enumerate(actions)}, {
            **{agent_id: False for agent_id in actions},
            "__all__": False,
        }, {**{agent_id: False for agent_id in actions}, "__all__": False}, {
            agent_id: {"progress": 1.0 + idx, "formation_error": 0.0, "min_gap": 10.0}
            for idx, agent_id in enumerate(actions)
        }


class _FakePlanner:
    def __init__(self, num_agents=3, num_modes=4, feature_dim=6):
        self.num_agents = num_agents
        self.num_modes = num_modes
        self.feature_dim = feature_dim

    def export_mode_selection(self, batch):
        agent_ids = list(batch)
        candidates = []
        features = []
        logits = []
        masks = []
        for agent_idx, _agent_id in enumerate(agent_ids):
            agent_candidates = np.zeros((self.num_modes, 8, 3), dtype=np.float32)
            for mode_idx in range(self.num_modes):
                agent_candidates[mode_idx, :, 0] = mode_idx
                agent_candidates[mode_idx, :, 1] = agent_idx
            candidates.append(agent_candidates)
            features.append(np.full((self.num_modes, self.feature_dim), agent_idx + 1.0, dtype=np.float32))
            logits.append(np.arange(self.num_modes, dtype=np.float32))
            mask = np.ones((self.num_modes,), dtype=bool)
            if agent_idx == 1:
                mask[-1] = False
            masks.append(mask)
        return {
            "agent_ids": agent_ids,
            "trajectory_candidates": np.stack(candidates, axis=0),
            "cls_feature": np.stack(features, axis=0),
            "raw_cls_logits": np.stack(logits, axis=0),
            "masked_cls_logits": np.where(np.stack(masks, axis=0), np.stack(logits, axis=0), -1e9),
            "mode_valid_mask": np.stack(masks, axis=0),
            "pretrained_argmax_mode": np.argmax(np.where(np.stack(masks, axis=0), np.stack(logits, axis=0), -1e9), axis=-1),
        }


def test_mode_selection_env_masks_and_executes_selected_candidates(tmp_path):
    base_env = _FakeBaseEnv()
    planner = _FakePlanner()
    debug_log_path = tmp_path / "mode_selection_debug.jsonl"
    env = ModeSelectionSB3Env(
        {
            "base_env": base_env,
            "planner": planner,
            "num_agents": 3,
            "debug_log_path": str(debug_log_path),
            "reward_config": {
                "w_progress": 1000.0,
                "w_team_efficiency": 1000.0,
                "reward_clip": 0.0,
            },
        }
    )

    obs, _ = env.reset()

    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert env.action_masks().shape == (12,)
    assert env.action_masks().reshape(3, 4)[1, 3] == 0
    assert obs["agent_relation_states"].shape == (3, 12)
    assert obs["trajectory_candidates"].shape == (3, 4, 8, 3)

    next_obs, reward, terminated, truncated, info = env.step(np.asarray([2, 1, 3], dtype=np.int64))

    assert reward == np.mean([0.0, 1.0, 2.0])
    assert info["reward"] == reward
    assert terminated is False
    assert truncated is False
    assert next_obs["agent_mode_masks"].shape == (3, 4)
    assert base_env.last_actions["agent0"].shape == (2,)
    assert info["executed_mode"] == [2, 1, 3]
    assert info["selected_trajectory_endpoint"][0] == [2.0, 0.0]
    assert info["invalid_mode_rate"] == 0.0
    assert info["pretrained_argmax_mode"] == [3, 2, 3]
    assert info["terminated"] is False
    assert info["truncated"] is False
    assert info["safety_flags"]["agent0"] == {"crash": False, "terminal_crash": False, "out_of_road": False}
    assert info["base_crash_flags"]["agent0"]["crash"] is False
    assert "crash_sidewalk" in info["base_crash_flags"]["agent0"]
    assert info["vehicle_position_before"] == {}
    assert info["vehicle_position_after"] == {}
    assert info["candidate_endpoints"][1][3] == [3.0, 1.0]
    assert '"executed_mode": [2, 1, 3]' in debug_log_path.read_text(encoding="utf-8")
    assert '"safety_flags"' in debug_log_path.read_text(encoding="utf-8")
    assert '"base_crash_flags"' in debug_log_path.read_text(encoding="utf-8")


def test_mode_selection_env_rejects_invalid_mode_without_fallback():
    env = ModeSelectionSB3Env({"base_env": _FakeBaseEnv(), "planner": _FakePlanner(), "num_agents": 3})
    env.reset()

    try:
        env.step(np.asarray([0, 3, 0], dtype=np.int64))
    except ValueError as exc:
        assert "invalid mode" in str(exc)
    else:
        raise AssertionError("invalid mode should raise instead of falling back")


def test_platoon_env_does_not_terminate_on_sidewalk_crash_only():
    env = PlatoonEnv.__new__(PlatoonEnv)
    env._agent_ids = ["agent0", "agent1", "agent2"]

    terminated, truncated = PlatoonEnv._enforce_platoon_episode_end(
        env,
        {"agent0": False, "agent1": False, "agent2": False, "__all__": False},
        {"agent0": False, "agent1": False, "agent2": False, "__all__": False},
        {
            "agent0": {"crash": False},
            "agent1": {"crash": False},
            "agent2": {"crash": True, "crash_sidewalk": True, "out_of_road": False},
        },
    )

    assert terminated["__all__"] is False
    assert truncated["__all__"] is False


def test_reward_terms_do_not_treat_sidewalk_as_vehicle_collision():
    config = {"w_collision": 10.0, "w_team_collision": 10.0, "reward_clip": 0.0}
    sidewalk_info = {"crash": True, "crash_sidewalk": True, "out_of_road": False, "progress": 0.0}
    vehicle_crash_info = {"crash": True, "crash_vehicle": True, "progress": 0.0}

    assert compute_step_reward(sidewalk_info, config) > -10.0
    assert compute_step_reward(vehicle_crash_info, config) <= -10.0
    assert compute_team_reward({"agent0": [sidewalk_info]}, config) == 0.0
    assert compute_team_reward({"agent0": [vehicle_crash_info]}, config) < 0.0


def test_mode_selection_env_keeps_fixed_agent_observation_shape_when_base_env_omits_agent():
    env = ModeSelectionSB3Env(
        {
            "base_env": _FakeBaseEnv(drop_agent_after_step="agent2"),
            "planner": _FakePlanner(),
            "num_agents": 3,
            "num_modes": 4,
            "relation_state_dim": 12,
        }
    )
    env.reset()

    obs, *_ = env.step(np.asarray([2, 1, 3], dtype=np.int64))

    assert obs["agent_relation_states"].shape == (3, 12)
    assert obs["trajectory_candidates"].shape == (3, 4, 8, 3)
    assert env.action_masks().shape == (12,)
    assert env._last_export["agent_ids"] == ["agent0", "agent1", "agent2"]


def test_mode_selection_env_terminates_when_controlled_agent_disappears():
    env = ModeSelectionSB3Env(
        {
            "base_env": _FakeBaseEnv(drop_agent_after_step="agent2"),
            "planner": _FakePlanner(),
            "num_agents": 3,
            "num_modes": 4,
        }
    )
    env.reset()

    _obs, _reward, terminated, truncated, info = env.step(np.asarray([2, 1, 3], dtype=np.int64))

    assert terminated is True
    assert truncated is False
    assert info["missing_agent_ids"] == ["agent2"]


def test_mode_selection_env_applies_configured_s1_s4_scenario_before_base_env_build():
    captured = {}

    def _factory(config):
        captured.update(config)
        return _FakeBaseEnv(config)

    env = ModeSelectionSB3Env(
        {
            "base_env_factory": _factory,
            "planner": _FakePlanner(),
            "num_agents": 3,
            "scenario_ids": ["S1_free_cruise_straight", "S4_curve_following"],
            "scenario_index": 1,
        }
    )

    assert captured["scenario_id"] == "S4_curve_following"
    assert captured["local_route"] == "R2_entry_curve"
    env.reset()


def test_missing_controlled_agents_detects_disappeared_platoon_member():
    assert _missing_controlled_agents({"agent0": {}, "agent2": {}}, ["agent0", "agent1", "agent2"]) == ["agent1"]
    assert _missing_controlled_agents({"agent0": {}, "agent1": {}, "agent2": {}}, ["agent0", "agent1", "agent2"]) == []
