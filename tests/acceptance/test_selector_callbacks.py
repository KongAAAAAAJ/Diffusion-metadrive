from __future__ import annotations

import math

import pytest

from train.selector_callbacks import PlatoonFormationCallbacks


class DummyEpisode:
    def __init__(self):
        self.user_data = {}
        self.custom_metrics = {}
        self._agent_to_last_info = {}


def _make_callback() -> PlatoonFormationCallbacks:
    return PlatoonFormationCallbacks()


def test_selector_callbacks_empty_episode_metrics_stay_finite():
    callback = _make_callback()
    episode = DummyEpisode()

    callback.on_episode_start(worker=None, base_env=None, policies=None, episode=episode, env_index=0)
    callback.on_episode_end(worker=None, base_env=None, policies=None, episode=episode)

    assert math.isfinite(float(episode.custom_metrics["formation_error_mean"]))
    assert math.isfinite(float(episode.custom_metrics["crash_rate"]))
    assert math.isfinite(float(episode.custom_metrics["team_reward_mean"]))
    assert episode.custom_metrics["formation_error_mean"] == pytest.approx(0.0)
    assert episode.custom_metrics["crash_rate"] == pytest.approx(0.0)
    assert episode.custom_metrics["team_reward_mean"] == pytest.approx(0.0)
    assert abs(float(episode.custom_metrics["intent_entropy"])) < 1e-6


def test_selector_callbacks_use_selector_reward_when_team_reward_missing():
    callback = _make_callback()
    episode = DummyEpisode()
    episode._agent_to_last_info = {
        "agent0": {
            "formation_error": 1.5,
            "crash": False,
            "selector_reward": 2.25,
            "selected_intent": 1,
        }
    }

    callback.on_episode_start(worker=None, base_env=None, policies=None, episode=episode, env_index=0)
    callback.on_episode_step(worker=None, base_env=None, episode=episode, env_index=0)
    callback.on_episode_end(worker=None, base_env=None, policies=None, episode=episode)

    assert episode.custom_metrics["formation_error_mean"] == pytest.approx(1.5)
    assert episode.custom_metrics["crash_rate"] == pytest.approx(0.0)
    assert episode.custom_metrics["team_reward_mean"] == pytest.approx(2.25)
    assert math.isfinite(float(episode.custom_metrics["team_reward_mean"]))


def test_selector_callbacks_do_not_emit_profile_metrics_to_custom_metrics():
    callback = _make_callback()
    episode = DummyEpisode()

    class DummySelectorEnv:
        def get_last_step_infos(self):
            return {
                "agent0": {
                    "formation_error": 1.0,
                    "crash": False,
                    "team_reward": 0.5,
                    "selected_intent": 2,
                }
            }

    class DummyBaseEnv:
        def get_sub_environments(self):
            return [DummySelectorEnv()]

    callback.on_episode_start(worker=None, base_env=DummyBaseEnv(), policies=None, episode=episode, env_index=0)
    callback.on_episode_step(worker=None, base_env=DummyBaseEnv(), episode=episode, env_index=0)
    callback.on_episode_end(worker=None, base_env=DummyBaseEnv(), policies=None, episode=episode)

    assert episode.custom_metrics["formation_error_mean"] == pytest.approx(1.0)
    assert episode.custom_metrics["crash_rate"] == pytest.approx(0.0)
    assert episode.custom_metrics["team_reward_mean"] == pytest.approx(0.5)
    assert abs(float(episode.custom_metrics["intent_entropy"])) < 1e-6
    assert episode.custom_metrics["intent_usage_2"] == pytest.approx(1.0)
    assert all(not key.startswith("profile_") for key in episode.custom_metrics)
