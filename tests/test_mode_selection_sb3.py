import numpy as np
import pytest
import torch

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym  # type: ignore

from envs.mode_selection_sb3_env import ModeSelectionSB3Env
from envs.platoon_env import PlatoonEnv
from evaluation.reward_terms import compute_step_reward, compute_team_reward
from models.mode_selection.sb3_mode_cls_policy import (
    ModeClsActorCriticCore,
)
from train.test_mode_cls_sb3 import (
    PretrainedArgmaxModeModel,
    _termination_reason_from_info,
    draw_multivehicle_multimodal_overlay,
    draw_vehicle_footprint_overlay,
    evaluate_mode_cls_policy,
    resolve_ppo_checkpoint,
)
from metadrive.policy.diffusion_policy.test_transfuser_policy import (
    _build_ppo_actor_obs,
    _missing_controlled_agents,
    _predict_ppo_modes,
)


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


def test_mode_cls_core_trainable_scope_and_forward():
    core = ModeClsActorCriticCore(num_agents=3, num_modes=4, relation_dim=16, trajectory_embed_dim=32)

    trainable = {name for name, param in core.named_parameters() if param.requires_grad}
    assert any(name.startswith("relation_encoder") for name in trainable)
    assert any(name.startswith("trajectory_encoder") for name in trainable)
    assert any(name.startswith("actor_head") for name in trainable)
    assert any(name.startswith("value_head") for name in trainable)

    obs = {
        "agent_relation_states": torch.ones((2, 3, 12), dtype=torch.float32),
        "trajectory_candidates": torch.ones((2, 3, 4, 8, 3), dtype=torch.float32),
        "agent_mode_masks": torch.ones((2, 3, 4), dtype=torch.bool),
        "global_state": torch.zeros((2, 60), dtype=torch.float32),
        "pretrained_logits": torch.zeros((2, 3, 4), dtype=torch.float32),
    }
    logits, values, metrics = core(obs)
    assert logits.shape == (2, 3, 4)
    assert values.shape == (2,)
    assert "KL_to_pretrained" in metrics


def test_maskable_ppo_fake_env_smoke_runs_one_rollout():
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("sb3_contrib")
    from models.mode_selection.sb3_mode_cls_policy import require_sb3

    env = ModeSelectionSB3Env(
        {
            "base_env": _FakeBaseEnv(),
            "planner": _FakePlanner(),
            "num_agents": 3,
            "num_modes": 4,
            "relation_state_dim": 12,
        }
    )
    MaskablePPO, Policy = require_sb3()
    model = MaskablePPO(
        Policy,
        env,
        policy_kwargs={
            "num_agents": 3,
            "num_modes": 4,
            "relation_dim": 16,
            "trajectory_embed_dim": 32,
        },
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        verbose=0,
    )

    model.learn(total_timesteps=4, progress_bar=False)

    assert hasattr(model.policy, "mode_cls_core")


def test_pretrained_policy_source_uses_argmax_baseline():
    env = ModeSelectionSB3Env(
        {
            "base_env": _FakeBaseEnv(),
            "planner": _FakePlanner(),
            "num_agents": 3,
            "num_modes": 4,
            "relation_state_dim": 12,
        }
    )
    obs, _ = env.reset()
    model = PretrainedArgmaxModeModel()

    action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())

    assert np.asarray(action).shape == (3,)
    assert np.asarray(action).tolist() == [3, 2, 3]


def test_resolve_ppo_checkpoint_prefers_explicit_file_and_supports_run_dir(tmp_path):
    run_dir = tmp_path / "run_1"
    final_ckpt = run_dir / "checkpoints" / "final" / "sb3_model.zip"
    final_ckpt.parent.mkdir(parents=True)
    final_ckpt.write_text("fake", encoding="utf-8")
    explicit_ckpt = tmp_path / "explicit.zip"
    explicit_ckpt.write_text("fake", encoding="utf-8")

    assert resolve_ppo_checkpoint(explicit_ckpt, run_dir) == explicit_ckpt
    assert resolve_ppo_checkpoint("", run_dir) == final_ckpt


def test_termination_reason_prefers_safety_flags():
    assert (
        _termination_reason_from_info(
            {
                "terminated": True,
                "safety_flags": {
                    "agent1": {"crash": True, "out_of_road": False},
                    "agent2": {"crash": False, "out_of_road": True},
                },
            }
        )
        == "agent1_crash__agent2_out_of_road"
    )
    assert _termination_reason_from_info({"truncated": True}) == "truncated"


def test_evaluate_mode_cls_policy_collects_closed_loop_metrics(tmp_path):
    class FakePolicy:
        def __init__(self):
            self.calls = 0

        def predict(self, obs, deterministic=True, action_masks=None):
            self.calls += 1
            assert deterministic is True
            assert action_masks.shape == (12,)
            return np.asarray([2, 1, 3], dtype=np.int64), None

    env = ModeSelectionSB3Env(
        {
            "base_env": _FakeBaseEnv(),
            "planner": _FakePlanner(),
            "num_agents": 3,
            "num_modes": 4,
            "relation_state_dim": 12,
            "debug_log_path": str(tmp_path / "ppo_test_debug.jsonl"),
        }
    )

    summary = evaluate_mode_cls_policy(FakePolicy(), env, episodes=2, max_steps=3, deterministic=True)

    assert summary["episodes"] == 2
    assert summary["total_steps"] == 6
    assert summary["invalid_mode_rate"] == 0.0
    assert summary["argmax_mismatch_rate"] > 0.0
    assert summary["mode_selection_counts"]["agent0"]["2"] == 6
    assert (tmp_path / "ppo_test_debug.jsonl").read_text(encoding="utf-8").count("\n") == 6


def test_draw_multivehicle_multimodal_overlay_highlights_selected_mode():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    candidates_world = {
        "agent0": np.asarray(
            [
                [[10, 10], [20, 10], [30, 10]],
                [[10, 20], [20, 20], [30, 20]],
            ],
            dtype=np.float32,
        )
    }

    canvas = draw_multivehicle_multimodal_overlay(
        frame,
        candidates_world,
        selected_modes={"agent0": 1},
        projector=lambda point: np.asarray(point, dtype=np.float32),
    )

    assert canvas.shape == frame.shape
    assert int(canvas.sum()) > 0
    assert not np.array_equal(canvas[20, 20], canvas[10, 20])


def test_draw_vehicle_footprint_overlay_draws_actual_vehicle_box():
    class Vehicle:
        position = [20.0, 20.0]
        heading_theta = 0.0
        LENGTH = 8.0
        WIDTH = 4.0

    class BaseEnv:
        agents = {"agent0": Vehicle()}

    class Env:
        base_env = BaseEnv()

    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    canvas = draw_vehicle_footprint_overlay(
        frame,
        Env(),
        projector=lambda point: np.asarray(point, dtype=np.float32),
    )

    assert canvas.shape == frame.shape
    assert int(canvas.sum()) > 0


def test_ppo_actor_obs_pads_missing_agent_slots_for_fixed_sb3_space():
    obs = {
        "agent0": {
            "status": np.ones((8,), dtype=np.float32),
            "formation_relation_state": np.ones((12,), dtype=np.float32),
        },
        "agent2": {
            "status": np.full((8,), 2.0, dtype=np.float32),
            "formation_relation_state": np.full((12,), 2.0, dtype=np.float32),
        },
    }
    candidates = np.ones((2, 4, 8, 3), dtype=np.float32)
    masks = np.ones((2, 4), dtype=bool)
    raw_logits = np.ones((2, 4), dtype=np.float32)

    actor_obs = _build_ppo_actor_obs(
        obs=obs,
        agent_ids=["agent0", "agent2"],
        policy_agent_ids=["agent0", "agent1", "agent2"],
        candidates=candidates,
        mode_valid_mask=masks,
        raw_logits=raw_logits,
    )

    assert actor_obs["agent_relation_states"].shape == (3, 12)
    assert actor_obs["trajectory_candidates"].shape == (3, 4, 8, 3)
    assert actor_obs["agent_mode_masks"].shape == (3, 4)
    assert actor_obs["agent_mode_masks"][1].tolist() == [True, False, False, False]
    assert np.allclose(actor_obs["agent_relation_states"][1], 0.0)


def test_missing_controlled_agents_detects_disappeared_platoon_member():
    assert _missing_controlled_agents({"agent0": {}, "agent2": {}}, ["agent0", "agent1", "agent2"]) == ["agent1"]
    assert _missing_controlled_agents({"agent0": {}, "agent1": {}, "agent2": {}}, ["agent0", "agent1", "agent2"]) == []


def test_predict_ppo_modes_returns_fixed_slot_actions_and_active_subset_can_be_mapped():
    class DummyModel:
        def predict(self, obs, deterministic=True, action_masks=None):
            assert obs["agent_relation_states"].shape == (3, 12)
            assert np.asarray(action_masks).shape == (12,)
            return np.asarray([2, 0, 3], dtype=np.int64), None

    full_mask = np.zeros((3, 4), dtype=bool)
    full_mask[0, 2] = True
    full_mask[1, 0] = True
    full_mask[2, 3] = True
    actions = _predict_ppo_modes(
        DummyModel(),
        {"agent_relation_states": np.zeros((3, 12), dtype=np.float32)},
        full_mask,
        deterministic=True,
    )

    policy_agent_ids = ["agent0", "agent1", "agent2"]
    exported_ids = ["agent0", "agent2"]
    active_actions = [actions[policy_agent_ids.index(agent_id)] for agent_id in exported_ids]
    assert active_actions == [2, 3]
