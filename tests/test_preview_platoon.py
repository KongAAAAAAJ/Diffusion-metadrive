from __future__ import annotations

from pathlib import Path

import numpy as np

from metadrive.exp_dataset import preview_platoon as module


class _FakeVehicle:
    def __init__(self) -> None:
        self.position = np.asarray([0.0, 0.0], dtype=np.float32)
        self.heading_theta = 0.0


class _FakeEnv:
    def __init__(self, episode_specs):
        self._episode_specs = list(episode_specs)
        self._episode_idx = -1
        self._step_idx = 0
        self.reset_seeds = []
        self.spawn_seeds = []
        self.agents = {}
        self.engine = type(
            "Engine",
            (),
            {
                "spawn_manager": type(
                    "SpawnManager",
                    (),
                    {"set_episode_spawn_seed": lambda inner_self, seed: self.spawn_seeds.append(seed)},
                )()
            },
        )()

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self._episode_idx += 1
        self._step_idx = 0
        self.agents = {f"agent{i}": _FakeVehicle() for i in range(3)}
        return {"agent0": {}, "agent1": {}, "agent2": {}}

    def low_level_step(self, actions):
        spec = self._episode_specs[self._episode_idx]
        done_step = spec["done_step"]
        terminate = self._step_idx >= done_step
        info = {
            "agent0": {"crash": False, "crash_vehicle": False},
            "agent1": {"crash": bool(spec.get("crash", False)), "crash_vehicle": bool(spec.get("crash", False))},
            "agent2": {"crash": False, "crash_vehicle": False},
        }
        self._step_idx += 1
        return {}, {}, {"__all__": terminate}, {"__all__": False}, info

    def close(self):
        return None


class _FakePolicy:
    def __init__(self):
        self.control_object = None

    def act(self):
        return np.zeros((2,), dtype=np.float32)


def test_run_preview_uses_start_seed_for_each_episode(monkeypatch, tmp_path: Path) -> None:
    fake_env = _FakeEnv([{"done_step": 2}, {"done_step": 2}])
    written = []

    monkeypatch.setattr(module, "_pick_local_route", lambda scenario_id: "R0")
    monkeypatch.setattr(module, "_capture_frame", lambda env, lead_id, heading_up, agent_ids: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_write_video", lambda path, frames, fps: written.append((path, len(frames), fps)))
    monkeypatch.setattr(module, "_make_idm_policies", lambda env, agent_ids, seed: {aid: _FakePolicy() for aid in agent_ids})
    video_dir = module.run_preview(
        scenario_id="S1_free_cruise_straight",
        local_route=None,
        num_agents=3,
        num_episodes=2,
        output_root=tmp_path,
        heading_up=False,
        traffic_density=0.10,
        start_seed=59,
        fps=10,
        env_factory=lambda config: fake_env,
    )

    assert fake_env.spawn_seeds == [59, 60]
    assert fake_env.reset_seeds == [None, None]
    assert video_dir.exists()
    assert len(written) == 2


def test_run_preview_retries_episode_when_it_ends_immediately(monkeypatch, tmp_path: Path) -> None:
    fake_env = _FakeEnv(
        [
            {"done_step": 0, "crash": True},
            {"done_step": 3, "crash": False},
        ]
    )
    written = []

    monkeypatch.setattr(module, "_pick_local_route", lambda scenario_id: "R0")
    monkeypatch.setattr(module, "_capture_frame", lambda env, lead_id, heading_up, agent_ids: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_write_video", lambda path, frames, fps: written.append((path, len(frames), fps)))
    monkeypatch.setattr(module, "_make_idm_policies", lambda env, agent_ids, seed: {aid: _FakePolicy() for aid in agent_ids})
    module.run_preview(
        scenario_id="S1_free_cruise_straight",
        local_route=None,
        num_agents=3,
        num_episodes=1,
        output_root=tmp_path,
        heading_up=False,
        traffic_density=0.10,
        start_seed=59,
        fps=10,
        env_factory=lambda config: fake_env,
    )

    assert fake_env.spawn_seeds == [59, 60]
    assert fake_env.reset_seeds == [None, None]
    assert len(written) == 1
    assert written[0][1] > 1
