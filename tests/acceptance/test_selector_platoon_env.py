from __future__ import annotations

import numpy as np
import pytest

from evaluation.reward_terms import compute_step_reward, compute_team_reward


class DummyBaseEnv:
    def __init__(self, num_agents: int = 2):
        self.num_agents = num_agents
        self.agent_ids = [f"agent{i}" for i in range(num_agents)]
        self.last_actions = None
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        obs = {
            agent_id: {
                "obs": np.full((4,), float(i + 1), dtype=np.float32),
                "formation_relation_state": np.full((12,), float(i), dtype=np.float32),
            }
            for i, agent_id in enumerate(self.agent_ids)
        }
        return obs

    def get_current_obs(self):
        return self.reset()

    def step(self, actions):
        self.last_actions = {k: np.asarray(v, dtype=np.float32).copy() for k, v in actions.items()}
        obs = self.reset()
        reward = {agent_id: 0.0 for agent_id in self.agent_ids}
        terminated = {agent_id: False for agent_id in self.agent_ids}
        truncated = {agent_id: False for agent_id in self.agent_ids}
        terminated["__all__"] = False
        truncated["__all__"] = False
        info = {
            agent_id: {
                "progress": 1.0 + i,
                "formation_error": 0.5 + i,
                "min_gap": 6.0,
                "jerk": 0.1,
                "delta_steering": 0.2,
                "crash": False,
                "out_of_road": False,
                "speed_km_h": 20.0 + i,
            }
            for i, agent_id in enumerate(self.agent_ids)
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        return None


class TerminatingBaseEnv(DummyBaseEnv):
    def __init__(self, mode: str):
        super().__init__(num_agents=2)
        self.mode = mode

    def step(self, actions):
        self.last_actions = {k: np.asarray(v, dtype=np.float32).copy() for k, v in actions.items()}
        obs = self.reset()
        reward = {agent_id: 0.0 for agent_id in self.agent_ids}
        terminated = {agent_id: False for agent_id in self.agent_ids}
        truncated = {agent_id: False for agent_id in self.agent_ids}
        info = {
            agent_id: {
                "progress": 1.0 + i,
                "formation_error": 0.5 + i,
                "min_gap": 6.0,
                "jerk": 0.1,
                "delta_steering": 0.2,
                "crash": False,
                "out_of_road": False,
                "speed_km_h": 20.0 + i,
            }
            for i, agent_id in enumerate(self.agent_ids)
        }

        if self.mode == "crash":
            info["agent1"]["crash"] = True
        elif self.mode == "out_of_road":
            info["agent1"]["out_of_road"] = True
        elif self.mode == "removed":
            obs.pop("agent1", None)
            terminated["agent1"] = True
        elif self.mode == "removed_all":
            obs.clear()
            terminated["agent0"] = True
            terminated["agent1"] = True
        else:
            raise ValueError(self.mode)

        terminated["__all__"] = False
        truncated["__all__"] = False
        return obs, reward, terminated, truncated, info


class DummyCandidateGenerator:
    def __init__(self, k: int = 3):
        self.k = k
        self.calls = 0

    def generate(self, agent_id, agent_obs, base_env=None, k=None):
        del agent_obs, base_env
        self.calls += 1
        k = self.k if k is None else int(k)
        base = np.linspace(0.0, 7.0, 8, dtype=np.float32)
        trajs = []
        for mode_idx in range(k):
            traj = np.zeros((8, 3), dtype=np.float32)
            traj[:, 0] = base + float(mode_idx)
            traj[:, 1] = float(mode_idx) * 0.25 + (0.1 if agent_id == "agent1" else 0.0)
            traj[:, 2] = float(mode_idx) * 0.05
            trajs.append(traj)
        return np.stack(trajs, axis=0)


class DummyPlannerSelector:
    def __init__(self, k: int = 3, embedding_dim: int = 4):
        self.k = k
        self.embedding_dim = embedding_dim
        self.calls = 0

    def forward_selector(self, batch):
        self.calls += 1
        payload = {}
        for agent_id in batch.keys():
            agent_index = int(agent_id.removeprefix("agent")) + 1
            candidates = np.zeros((self.k, 8, 3), dtype=np.float32)
            embeddings = np.zeros((self.k, self.embedding_dim), dtype=np.float32)
            for mode_idx in range(self.k):
                candidates[mode_idx, :, 0] = np.linspace(0.0, 7.0, 8, dtype=np.float32) + float(mode_idx)
                candidates[mode_idx, :, 1] = float(mode_idx) * 0.5
                candidates[mode_idx, :, 2] = float(mode_idx) * 0.1
                embeddings[mode_idx] = float(agent_index) * 10.0 + np.arange(self.embedding_dim, dtype=np.float32) + mode_idx
            payload[agent_id] = {
                "trajectory": candidates[0],
                "trajectory_candidates": candidates,
                "trajectory_mode_logits": np.linspace(0.0, 1.0, self.k, dtype=np.float32),
                "trajectory_mode_embedding": embeddings,
            }
        return payload


def _make_env():
    from envs.selector_platoon_env import SelectorPlatoonEnv

    base_env = DummyBaseEnv(num_agents=2)
    generator = DummyCandidateGenerator(k=3)
    env = SelectorPlatoonEnv(
        {
            "base_env": base_env,
            "candidate_generator": generator,
            "num_agents": 2,
            "K": 3,
            "agent_context_dim": 4,
            "candidate_summary_dim": 5,
            "reward_config": {
                "w_progress": 1.0,
                "w_formation": 0.5,
                "w_safety": 0.3,
                "w_collision": 10.0,
                "w_road": 5.0,
                "w_comfort": 0.1,
                "w_team_formation": 0.5,
                "w_team_safety": 1.0,
                "w_team_efficiency": 0.3,
                "w_team_collision": 10.0,
                "delta_s_max": 5.0,
                "d_norm": 10.0,
                "d_safe": 8.0,
            },
        }
    )
    return env, base_env, generator


def test_selector_env_reset_structure():
    env, _, _ = _make_env()
    obs, info = env.reset()
    assert set(obs.keys()) == {"agent0", "agent1"}
    assert set(info.keys()) == {"agent0", "agent1"}

    agent0 = obs["agent0"]
    assert set(agent0.keys()) == {
        "agent_context",
        "formation_relation_state",
        "mode_embeddings",
        "candidate_summary",
        "global_state",
        "action_mask",
    }
    assert tuple(agent0["agent_context"].shape) == (4,)
    assert tuple(agent0["formation_relation_state"].shape) == (12,)
    assert tuple(agent0["mode_embeddings"].shape) == (3, 5)
    assert tuple(agent0["candidate_summary"].shape) == (3, 5)
    assert tuple(agent0["action_mask"].shape) == (3,)
    assert agent0["action_mask"].dtype == np.float32
    assert np.allclose(agent0["action_mask"], 1.0)
    assert agent0["global_state"].ndim == 1


def test_selector_env_step_maps_intent_to_cached_trajectory():
    env, base_env, generator = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step({"agent0": 2, "agent1": 1})
    assert generator.calls == 4
    assert base_env.last_actions is not None
    assert np.allclose(base_env.last_actions["agent0"], generator.generate("agent0", None, None, 3)[2])
    assert np.allclose(base_env.last_actions["agent1"], generator.generate("agent1", None, None, 3)[1])
    assert set(obs.keys()) == {"agent0", "agent1"}
    assert set(reward.keys()) == {"agent0", "agent1"}
    assert "__all__" in terminated
    assert "__all__" in truncated
    assert set(info.keys()) == {"agent0", "agent1"}
    assert info["agent0"]["selected_intent"] == 2
    assert info["agent1"]["selected_intent"] == 1
    assert info["agent0"]["control_mode"] == "trajectory"
    assert info["agent0"]["intent_valid"] is True


def test_selector_env_reward_combination_matches_helpers():
    env, _, _ = _make_env()
    env.reset()
    _, reward, _, _, info = env.step({"agent0": 0, "agent1": 0})
    reward_config = env.reward_config
    expected_local = {
        agent_id: compute_step_reward(info[agent_id], reward_config)
        for agent_id in ("agent0", "agent1")
    }
    expected_team = compute_team_reward({agent_id: [info[agent_id]] for agent_id in ("agent0", "agent1")}, reward_config)
    for agent_id in ("agent0", "agent1"):
        expected = env.lambda_local * expected_local[agent_id] + env.lambda_team * expected_team
        assert np.isclose(reward[agent_id], expected)
        assert np.isclose(info[agent_id]["team_reward"], expected_team)


def test_selector_env_action_space_and_global_state_are_consistent():
    env, _, _ = _make_env()
    obs, _ = env.reset()
    for agent_id, agent_obs in obs.items():
        assert tuple(agent_obs["global_state"].shape) == env.single_observation_space.spaces["global_state"].shape
        assert env.single_action_space.n == 3


def test_selector_env_uses_planner_mode_embeddings_directly():
    from envs.selector_platoon_env import SelectorPlatoonEnv

    planner = DummyPlannerSelector(k=3, embedding_dim=4)
    env = SelectorPlatoonEnv(
        {
            "base_env": DummyBaseEnv(num_agents=2),
            "candidate_generator": planner,
            "num_agents": 2,
            "K": 3,
            "agent_context_dim": 4,
            "candidate_summary_dim": 5,
            "mode_embedding_dim": 4,
        }
    )

    obs, _ = env.reset()
    expected_agent0 = np.stack(
        [10.0 + np.arange(4, dtype=np.float32) + mode_idx for mode_idx in range(3)],
        axis=0,
    )
    expected_agent1 = np.stack(
        [20.0 + np.arange(4, dtype=np.float32) + mode_idx for mode_idx in range(3)],
        axis=0,
    )

    assert np.allclose(obs["agent0"]["mode_embeddings"], expected_agent0)
    assert np.allclose(obs["agent1"]["mode_embeddings"], expected_agent1)
    assert planner.calls == 1


def test_selector_env_batches_forward_selector_once_per_refresh():
    from envs.selector_platoon_env import SelectorPlatoonEnv

    planner = DummyPlannerSelector(k=3, embedding_dim=4)
    base_env = DummyBaseEnv(num_agents=3)
    env = SelectorPlatoonEnv(
        {
            "base_env": base_env,
            "candidate_generator": planner,
            "num_agents": 3,
            "K": 3,
            "agent_context_dim": 4,
            "candidate_summary_dim": 5,
            "mode_embedding_dim": 4,
        }
    )

    env.reset()
    assert planner.calls == 1

    env.step({"agent0": 0, "agent1": 1, "agent2": 2})
    assert planner.calls == 2


def test_selector_env_accepts_torch_candidate_outputs():
    torch = pytest.importorskip("torch")
    from envs.selector_platoon_env import SelectorPlatoonEnv

    class TorchPlannerSelector:
        def forward_selector(self, batch):
            payload = {}
            for agent_id in batch.keys():
                candidates = torch.zeros((3, 8, 3), dtype=torch.float32)
                embeddings = torch.zeros((3, 4), dtype=torch.float32)
                for mode_idx in range(3):
                    candidates[mode_idx, :, 0] = torch.linspace(0.0, 7.0, 8) + float(mode_idx)
                    candidates[mode_idx, :, 1] = float(mode_idx) * 0.5
                    candidates[mode_idx, :, 2] = float(mode_idx) * 0.1
                    embeddings[mode_idx] = float(mode_idx) + torch.arange(4, dtype=torch.float32)
                payload[agent_id] = {
                    "trajectory_candidates": candidates,
                    "trajectory_mode_embedding": embeddings,
                }
            return payload

    env = SelectorPlatoonEnv(
        {
            "base_env": DummyBaseEnv(num_agents=2),
            "candidate_generator": TorchPlannerSelector(),
            "num_agents": 2,
            "K": 3,
            "agent_context_dim": 4,
            "candidate_summary_dim": 5,
            "mode_embedding_dim": 4,
        }
    )

    obs, _ = env.reset()
    assert obs["agent0"]["mode_embeddings"].dtype == np.float32
    assert obs["agent0"]["candidate_summary"].dtype == np.float32


def test_selector_env_invalid_intent_raises():
    env, _, _ = _make_env()
    env.reset()

    with pytest.raises(ValueError, match="Invalid intent"):
        env.step({"agent0": 99, "agent1": 0})


def test_selector_env_missing_agent_action_raises():
    env, _, _ = _make_env()
    env.reset()

    with pytest.raises(KeyError, match="Missing selector action"):
        env.step({"agent0": 1})


@pytest.mark.parametrize("mode", ["crash", "out_of_road", "removed"])
def test_selector_env_terminates_episode_when_any_agent_fails(mode):
    from envs.selector_platoon_env import SelectorPlatoonEnv

    env = SelectorPlatoonEnv(
        {
            "base_env": TerminatingBaseEnv(mode),
            "candidate_generator": DummyCandidateGenerator(k=3),
            "num_agents": 2,
            "K": 3,
            "agent_context_dim": 4,
            "candidate_summary_dim": 5,
        }
    )

    env.reset()
    obs, reward, terminated, truncated, info = env.step({"agent0": 0, "agent1": 1})

    assert terminated["__all__"] is True
    assert terminated["agent0"] is True
    assert terminated["agent1"] is True
    assert truncated["__all__"] is False
    assert set(reward.keys()) == {"agent0", "agent1"}
    assert set(info.keys()).issubset(set(obs.keys()))
    assert "agent0" in env.get_last_step_infos()
    assert "agent1" in env.get_last_step_infos()


def test_selector_env_terminal_infos_are_subset_of_obs_when_all_agents_disappear():
    from envs.selector_platoon_env import SelectorPlatoonEnv

    env = SelectorPlatoonEnv(
        {
            "base_env": TerminatingBaseEnv("removed_all"),
            "candidate_generator": DummyCandidateGenerator(k=3),
            "num_agents": 2,
            "K": 3,
            "agent_context_dim": 4,
            "candidate_summary_dim": 5,
        }
    )

    obs, _ = env.reset()
    assert set(obs.keys()) == {"agent0", "agent1"}

    obs, reward, terminated, truncated, info = env.step({"agent0": 0, "agent1": 1})

    assert obs == {}
    assert info == {}
    assert terminated["__all__"] is True
    assert terminated["agent0"] is True
    assert terminated["agent1"] is True
    assert truncated["__all__"] is False
    assert set(reward.keys()) == {"agent0", "agent1"}


def test_selector_env_returns_empty_infos_when_team_terminates_for_rllib():
    from envs.selector_platoon_env import SelectorPlatoonEnv

    env = SelectorPlatoonEnv(
        {
            "base_env": TerminatingBaseEnv("crash"),
            "candidate_generator": DummyCandidateGenerator(k=3),
            "num_agents": 2,
            "K": 3,
            "agent_context_dim": 4,
            "candidate_summary_dim": 5,
        }
    )

    env.reset()
    obs, reward, terminated, truncated, info = env.step({"agent0": 0, "agent1": 1})

    assert set(info.keys()).issubset(set(obs.keys()))
    assert set(obs.keys()) == {"agent0", "agent1"}
    assert terminated["__all__"] is True
    assert truncated["__all__"] is False
    assert "agent0" in env.get_last_step_infos()
    assert "agent1" in env.get_last_step_infos()

