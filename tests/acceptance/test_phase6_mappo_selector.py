from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
import torch
import yaml

from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner


def _build_batch(num_agents: int = 3) -> dict[str, dict[str, torch.Tensor]]:
    batch = {}
    for idx in range(num_agents):
        batch[f"agent{idx}"] = {
            "camera": torch.zeros(3, 256, 1024, dtype=torch.float32),
            "lidar": torch.zeros(1, 256, 256, dtype=torch.float32),
            "status": torch.zeros(8, dtype=torch.float32),
            "formation_relation_state": torch.zeros(12, dtype=torch.float32),
        }
    return batch


def test_platoon_planner_forward_selector_exports_multimodal_candidates():
    config = build_transfuser_config(
        "small",
        plan_anchor_path="metadrive/exp_dataset/anchors.npy",
        ego_fut_mode=3,
        tf_d_model=256,
        tf_d_ffn=512,
        trajectory_decoder_layers=1,
    )
    planner = PlatoonDiffusionPlanner(config=config, num_vehicles=3)
    outputs = planner.forward_selector(_build_batch())

    assert set(outputs.keys()) == {"agent0", "agent1", "agent2"}
    for payload in outputs.values():
        assert payload["trajectory"].shape == (8, 3)
        assert payload["trajectory_candidates"].shape == (config.ego_fut_mode, 8, 3)
        assert payload["trajectory_mode_logits"].shape == (config.ego_fut_mode,)
        assert payload["trajectory_mode_embedding"].shape[0] == config.ego_fut_mode


def test_planner_can_be_frozen_for_selector_training():
    config = build_transfuser_config(
        "small",
        plan_anchor_path="metadrive/exp_dataset/anchors.npy",
        ego_fut_mode=3,
        tf_d_model=256,
        tf_d_ffn=512,
        trajectory_decoder_layers=1,
    )
    planner = PlatoonDiffusionPlanner(config=config, num_vehicles=3)
    planner.freeze_for_selector()
    assert all(not parameter.requires_grad for parameter in planner.parameters())


def test_selector_env_maps_discrete_intent_to_trajectory(monkeypatch):
    from envs.selector_platoon_env import SelectorPlatoonEnv

    class FakeBaseEnv:
        def __init__(self):
            self.last_actions = None

        def reset(self):
            return {
                "agent0": {"formation_relation_state": np.zeros(12, dtype=np.float32)},
                "agent1": {"formation_relation_state": np.zeros(12, dtype=np.float32)},
            }

        def step(self, actions):
            self.last_actions = actions
            obs = self.reset()
            reward = {"agent0": 0.0, "agent1": 0.0}
            terminated = {"agent0": False, "agent1": False, "__all__": False}
            truncated = {"agent0": False, "agent1": False, "__all__": False}
            info = {
                "agent0": {"formation_error": 0.0, "progress": 1.0, "crash": False, "out_of_road": False, "min_gap": 10.0, "speed_km_h": 10.0},
                "agent1": {"formation_error": 0.0, "progress": 1.0, "crash": False, "out_of_road": False, "min_gap": 10.0, "speed_km_h": 10.0},
            }
            return obs, reward, terminated, truncated, info

        def close(self):
            return None

    class FakePlanner:
        def __init__(self):
            self.frozen = True

        def forward_selector(self, batch):
            payload = {}
            for agent_id in batch.keys():
                candidates = torch.zeros(3, 8, 3, dtype=torch.float32)
                candidates[1, :, 0] = 1.0
                payload[agent_id] = {
                    "trajectory": candidates[0],
                    "trajectory_candidates": candidates,
                    "trajectory_mode_logits": torch.zeros(3),
                    "trajectory_mode_embedding": torch.zeros(3, 16),
                    "agent_context": torch.zeros(32),
                }
            return payload

    env = SelectorPlatoonEnv(
        {
            "num_agents": 2,
            "K": 3,
            "base_env": FakeBaseEnv(),
            "candidate_generator": FakePlanner(),
            "agent_context_dim": 32,
            "mode_embedding_dim": 16,
            "candidate_summary_dim": 6,
        }
    )
    obs, info = env.reset()
    assert set(obs.keys()) == {"agent0", "agent1"}
    assert set(info.keys()) == {"agent0", "agent1"}

    obs, reward, terminated, truncated, info = env.step({"agent0": 1, "agent1": 1})
    assert env.base_env.last_actions["agent0"].shape == (8, 3)
    assert float(env.base_env.last_actions["agent0"][0, 0]) == 1.0
    assert "__all__" in terminated
    assert "__all__" in truncated
    assert "global_state" in obs["agent0"]
    assert "selector_reward" in info["agent0"]
    assert isinstance(reward["agent0"], float)


def test_selector_env_default_path_builds_real_planner_generator(monkeypatch):
    from envs import selector_platoon_env as selector_env_mod

    fake_generator = object()
    calls = []

    def fake_builder(config):
        calls.append(dict(config))
        return fake_generator

    monkeypatch.setattr(selector_env_mod, "_build_default_candidate_generator", fake_builder)

    class FakeBaseEnv:
        def reset(self):
            return {
                "agent0": {"obs": np.zeros(4, dtype=np.float32), "formation_relation_state": np.zeros(12, dtype=np.float32)},
                "agent1": {"obs": np.zeros(4, dtype=np.float32), "formation_relation_state": np.zeros(12, dtype=np.float32)},
            }

        def close(self):
            return None

    env = selector_env_mod.SelectorPlatoonEnv(
        {
            "num_agents": 2,
            "K": 3,
            "base_env": FakeBaseEnv(),
            "pretrained_ckpt": "/tmp/fake.ckpt",
            "anchor_path": "metadrive/exp_dataset/anchors.npy",
            "model_size": "small",
        }
    )

    assert env.generator is fake_generator
    assert len(calls) == 1
    assert calls[0]["pretrained_ckpt"] == "/tmp/fake.ckpt"


def test_selector_actor_and_critic_shapes():
    from models.selector.intent_selector import IntentSelectorActor, IntentSelectorCritic

    actor = IntentSelectorActor(agent_context_dim=32, relation_dim=12, mode_embedding_dim=16, summary_dim=6, num_modes=3)
    critic = IntentSelectorCritic(global_state_dim=128)

    logits = actor(
        agent_context=torch.zeros(4, 32),
        formation_relation_state=torch.zeros(4, 12),
        mode_embeddings=torch.zeros(4, 3, 16),
        candidate_summary=torch.zeros(4, 3, 6),
    )
    values = critic(torch.zeros(4, 128))
    assert logits.shape == (4, 3)
    assert values.shape == (4,)


def test_train_selector_require_rllib_is_consistent():
    from train import train_selector as entry

    if entry._RAY_AVAILABLE:
        entry.require_rllib()
    else:
        with pytest.raises(RuntimeError, match="ray\\[rllib\\]"):
            entry.require_rllib()


def test_total_env_steps_resolves_to_iterations():
    from train import train_selector as entry

    cfg = {
        "train_batch_size": 4000,
        "rollout_fragment_length": 32,
        "num_rollout_workers": 2,
        "num_envs_per_worker": 1,
        "total_env_steps": 300,
    }

    assert entry.estimate_env_steps_per_iteration(cfg) == 4000
    assert entry.resolve_max_iterations(cfg) == 1

def test_phase6_config_exists():
    config_path = Path("configs/train/selector.yaml")
    assert config_path.exists()


def test_selector_reward_config_is_team_and_safety_leaning():
    main_config = yaml.safe_load(Path("configs/train/selector.yaml").read_text(encoding="utf-8"))
    smoke_config = yaml.safe_load(Path("configs/train/platoon_mappo_smoke.yaml").read_text(encoding="utf-8"))

    for cfg in (main_config, smoke_config):
        reward = cfg["reward_config"]
        assert cfg["lambda_team"] > cfg["lambda_local"]
        assert reward["reward_clip"] == 8.0
        assert reward["w_formation"] >= reward["w_progress"]
        assert reward["w_team_formation"] >= reward["w_formation"]
        assert reward["w_team_safety"] >= reward["w_safety"]
        assert reward["w_team_efficiency"] <= reward["w_progress"]
        assert reward["w_comfort"] <= 0.05


def test_selector_reward_prefers_safe_coordination_and_clips_outliers():
    from evaluation.reward_terms import compute_step_reward, compute_team_reward

    config = yaml.safe_load(Path("configs/train/selector.yaml").read_text(encoding="utf-8"))["reward_config"]

    safe_step = {
        "progress": 1.0,
        "formation_error": 0.2,
        "min_gap": 8.5,
        "jerk": 0.05,
        "delta_steering": 0.05,
        "crash": False,
        "out_of_road": False,
    }
    greedy_step = {
        "progress": 1.4,
        "formation_error": 2.5,
        "min_gap": 3.0,
        "jerk": 0.35,
        "delta_steering": 0.25,
        "crash": False,
        "out_of_road": False,
    }
    assert compute_step_reward(safe_step, config) > compute_step_reward(greedy_step, config)

    safe_team = {
        f"agent{i}": [
            {
                "progress": 1.0,
                "formation_error": 0.2,
                "crash": False,
            }
        ]
        for i in range(3)
    }
    greedy_team = {
        f"agent{i}": [
            {
                "progress": 1.5,
                "formation_error": 2.5,
                "crash": False,
            }
        ]
        for i in range(3)
    }
    assert compute_team_reward(safe_team, config) > compute_team_reward(greedy_team, config)

    extreme_step = {
        "progress": 100.0,
        "formation_error": 100.0,
        "min_gap": 0.0,
        "jerk": 100.0,
        "delta_steering": 100.0,
        "crash": True,
        "out_of_road": True,
    }
    assert compute_step_reward(extreme_step, {**config, "reward_clip": 3.0}) == -3.0
    assert (
        compute_team_reward(
            {f"agent{i}": [extreme_step] for i in range(3)},
            {**config, "reward_clip": 3.0},
        )
        == -3.0
    )


def test_run_training_writes_summary_with_numpy_metrics(tmp_path, monkeypatch):
    from train import train_selector as entry

    class FakeAlgo:
        def __init__(self):
            self.save_calls = []

        def train(self):
            return {
                "episode_reward_mean": np.float32(1.25),
                "custom_metrics": {
                    "formation_error_mean": np.float32(0.5),
                    "intent_usage": np.asarray([1.0, 2.0], dtype=np.float32),
                },
                "non_json_type": dict,
            }

        def save(self, checkpoint_dir):
            self.save_calls.append(str(checkpoint_dir))
            path = Path(checkpoint_dir) / f"checkpoint_{len(self.save_calls):06d}"
            path.mkdir(parents=True, exist_ok=True)
            (path / "marker.txt").write_text("ok", encoding="utf-8")
            return str(path)

        def stop(self):
            return None

    class FakeConfig:
        def build(self):
            return FakeAlgo()

    monkeypatch.setattr(entry, "build_rllib_config", lambda cfg: (FakeConfig(), {"num_agents": 3}))
    monkeypatch.setattr(entry.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(entry.ray, "shutdown", lambda: None)

    summary = entry.run_training(
        {
            "output_root": str(tmp_path),
            "max_iterations": 1,
            "checkpoint_freq": 1,
        }
    )

    run_dir = Path(summary["run_dir"])
    summary_path = run_dir / "summary.json"
    assert summary_path.exists()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["checkpoints"]
    assert summary_payload["best_checkpoint_path"] is not None
    assert summary_payload["best_checkpoint_iteration"] == 1
    assert summary_payload["best_checkpoint_metric"] == "episode_reward_mean"
    assert summary_payload["best_checkpoint_mode"] == "max"
    assert "env_profile_summary" not in summary_payload
    assert summary_payload["last_result"]["episode_reward_mean"] == 1.25
    assert summary_payload["last_result"]["non_json_type"] == "dict"


def test_run_training_tracks_best_checkpoint_on_improvement(tmp_path, monkeypatch):
    from train import train_selector as entry

    class FakeAlgo:
        def __init__(self):
            self.train_calls = 0
            self.save_calls = []

        def train(self):
            self.train_calls += 1
            reward = 1.0 if self.train_calls == 1 else 2.5
            return {"episode_reward_mean": np.float32(reward)}

        def save(self, checkpoint_dir):
            self.save_calls.append(str(checkpoint_dir))
            path = Path(checkpoint_dir) / f"checkpoint_{len(self.save_calls):06d}"
            path.mkdir(parents=True, exist_ok=True)
            (path / "marker.txt").write_text("ok", encoding="utf-8")
            return str(path)

        def stop(self):
            return None

    class FakeConfig:
        def build(self):
            return FakeAlgo()

    monkeypatch.setattr(entry, "build_rllib_config", lambda cfg: (FakeConfig(), {"num_agents": 3}))
    monkeypatch.setattr(entry.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(entry.ray, "shutdown", lambda: None)

    summary = entry.run_training(
        {
            "output_root": str(tmp_path),
            "max_iterations": 2,
            "checkpoint_freq": 2,
        }
    )

    assert summary["best_checkpoint_iteration"] == 2
    assert summary["best_checkpoint_value"] == 2.5
    assert summary["best_checkpoint_path"] is not None
    assert Path(summary["best_checkpoint_path"]).exists()


def test_iteration_log_line_contains_training_and_reward_metrics():
    from train import train_selector as entry

    line = entry.build_iteration_log_line(
        2,
        10,
        {
            "num_env_steps_sampled": 320,
            "num_agent_steps_sampled": 960,
            "episodes_total": 6,
            "episode_reward_mean": np.float32(1.5),
            "episode_len_mean": np.float32(48.0),
            "custom_metrics": {
                "formation_error_mean_mean": np.float32(0.42),
                "crash_rate_mean": np.float32(0.01),
                "team_reward_mean_mean": np.float32(-0.12),
                "intent_entropy_mean": np.float32(0.97),
            },
            "info": {
                "learner": {
                    "shared_selector": {
                        "learner_stats": {
                            "total_loss": np.float32(0.31),
                            "policy_loss": np.float32(-0.08),
                            "vf_loss": np.float32(0.22),
                            "entropy": np.float32(1.73),
                        }
                    }
                }
            },
        },
    )

    assert "[train][iter 2/10]" in line
    assert "env_steps=320" in line
    assert "ep_reward=1.500" in line
    assert "loss=0.310" in line
    assert "policy_loss=-0.080" in line
    assert "formation=0.420" in line
    assert "team_reward=-0.120" in line
    assert "intent_entropy=0.970" in line


def test_training_complete_log_line_contains_elapsed_and_run_dir():
    from train import train_selector as entry

    line = entry.build_training_complete_log_line(
        {
            "iterations": 5,
            "target_env_steps": 320,
            "checkpoints": ["/tmp/a", "/tmp/b"],
            "run_dir": "/tmp/run_1",
        },
        72.27,
    )

    assert "[train][done]" in line
    assert "elapsed=72.3s" in line
    assert "iterations=5" in line
    assert "target_env_steps=320" in line
    assert "checkpoints=2" in line
    assert "run_dir=/tmp/run_1" in line


def test_selector_callbacks_emit_finite_metrics_for_empty_episode():
    from train.selector_callbacks import PlatoonFormationCallbacks

    class FakeEpisode:
        def __init__(self):
            self.user_data = {}
            self.custom_metrics = {}

    callbacks = PlatoonFormationCallbacks()
    episode = FakeEpisode()
    callbacks.on_episode_start(worker=None, base_env=None, policies=None, episode=episode, env_index=0)
    callbacks.on_episode_end(worker=None, base_env=None, policies=None, episode=episode)

    assert episode.custom_metrics["formation_error_mean"] == 0.0
    assert episode.custom_metrics["crash_rate"] == 0.0
    assert episode.custom_metrics["team_reward_mean"] == 0.0
    assert episode.custom_metrics["intent_entropy"] == 0.0


def test_collect_episode_infos_normalizes_agent_reward_tuple_keys():
    from train.selector_callbacks import _collect_episode_infos

    class FakeEpisode:
        def __init__(self):
            self._agent_to_last_info = {}
            self.agent_rewards = {
                ("agent0", "shared_selector"): 1.0,
                ("agent1", "shared_selector"): 2.0,
            }

        def last_info_for(self, agent_id="agent0"):
            mapping = {
                "agent0": {"selected_intent": 1, "formation_error": 0.5, "team_reward": 0.25},
                "agent1": {"selected_intent": 2, "formation_error": 0.2, "team_reward": 0.25},
            }
            return mapping.get(agent_id)

    infos = _collect_episode_infos(FakeEpisode())

    assert len(infos) == 2
    assert sorted(info["selected_intent"] for info in infos) == [1, 2]


def test_collect_episode_infos_prefers_selector_env_last_step_infos():
    from train.selector_callbacks import _collect_episode_infos

    class FakeSelectorEnv:
        def get_last_step_infos(self):
            return {
                "agent0": {"selected_intent": 3, "formation_error": 1.2, "team_reward": -0.5},
                "agent1": {"selected_intent": 1, "formation_error": 0.4, "team_reward": -0.5},
            }

    class FakeBaseEnv:
        def get_sub_environments(self):
            return [FakeSelectorEnv()]

    class FakeEpisode:
        def __init__(self):
            self._agent_to_last_info = {}

    infos = _collect_episode_infos(FakeEpisode(), base_env=FakeBaseEnv(), env_index=0)

    assert len(infos) == 2
    assert sorted(info["selected_intent"] for info in infos) == [1, 3]
