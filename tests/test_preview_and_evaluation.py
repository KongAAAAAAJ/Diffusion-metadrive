from __future__ import annotations

from pathlib import Path
import sys
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import preview_and_evaluation as module


class _FakeVehicle:
    def __init__(self) -> None:
        self.position = np.asarray([0.0, 0.0], dtype=np.float32)
        self.velocity = np.asarray([0.0, 0.0], dtype=np.float32)
        self.heading_theta = 0.0
        self.speed_km_h = 0.0


class _FakeEnv:
    def __init__(self, episode_specs):
        self._episode_specs = list(episode_specs)
        self._episode_idx = -1
        self._step_idx = 0
        self.reset_seeds = []
        self.spawn_seeds = []
        self.applied_roles = []
        self.agents = {}
        self.config = {
            "pid_dt": 0.5,
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
        }
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
        for idx, (agent_id, vehicle) in enumerate(self.agents.items()):
            action = np.asarray(actions.get(agent_id, np.zeros(2, dtype=np.float32)), dtype=np.float32)
            vehicle.position = vehicle.position + np.asarray([1.0 + idx, 0.1 * idx], dtype=np.float32)
            vehicle.speed_km_h = float(vehicle.speed_km_h + 3.6 * float(action[1]) + 1.0)
            vehicle.velocity = vehicle.velocity + np.asarray([float(action[1]) + 0.1, 0.0], dtype=np.float32)
            vehicle.heading_theta = float(vehicle.heading_theta + 0.01 * float(action[0]))
        info = {
            "agent0": {"crash": False, "crash_vehicle": False},
            "agent1": {"crash": bool(spec.get("crash", False)), "crash_vehicle": bool(spec.get("crash", False))},
            "agent2": {"crash": False, "crash_vehicle": False},
        }
        self._step_idx += 1
        return {}, {}, {"__all__": terminate}, {"__all__": False}, info

    def close(self):
        return None

    def apply_dynamic_roles(self, roles):
        self.applied_roles.append(dict(roles))


class _FakePolicy:
    def __init__(self):
        self.control_object = None

    def act(self):
        return np.zeros((2,), dtype=np.float32)


class _FakeFrameCanvas:
    def pos2pix(self, x, y):
        return (400, 400)


class _FakeRenderer:
    position = np.asarray([0.0, 0.0], dtype=np.float32)
    _frame_canvas = _FakeFrameCanvas()

    def _world_to_screen_position(self, position, off):
        return np.asarray([40.0, 40.0], dtype=np.float32)


class _FakeRuleMaker:
    def reset(self, env, agent_ids):
        return None

    def compute(self, env, agent_ids, planner_batch):
        self._last_debug = {
            "agent_ids": list(agent_ids),
            "best_actions": {"agent0": 0},
            "best_score": 1.0,
            "dynamic_roles": {"agent0": "leader"},
            "candidates_by_agent": {
                "agent0": [
                    {
                        "action": -1,
                        "score": -1.0,
                        "trajectory_world": [[0.0, 0.0], [5.0, 3.5]],
                        "target_point": [5.0, 3.5],
                        "selected": False,
                    },
                    {
                        "action": 0,
                        "score": 1.0,
                        "trajectory_world": [[0.0, 0.0], [5.0, 0.0]],
                        "target_point": [5.0, 0.0],
                        "selected": True,
                    },
                    {
                        "action": 1,
                        "score": -2.0,
                        "trajectory_world": [[0.0, 0.0], [5.0, -3.5]],
                        "target_point": [5.0, -3.5],
                        "selected": False,
                    },
                ]
            },
        }
        env._preview_rule_maker_debug = self._last_debug
        return {"agent0": {"action": 0, "target_point": np.asarray([5.0, 0.0], dtype=np.float32)}}

    def get_last_debug(self):
        return getattr(self, "_last_debug", None)


class _FakeLatticePlanner:
    def __init__(self):
        self.calls = []
        self._last_debug = None

    def plan(self, env, decisions):
        self.calls.append(decisions)
        trajectory = np.asarray(
            [[float(i), 0.0, 0.0] for i in range(8)],
            dtype=np.float32,
        )
        self._last_debug = {
            "agent0": {
                "fallback_used": False,
                "fallback_reason": None,
                "best_index": 1,
                "candidate_count": 2,
                "candidates": [
                    {
                        "score": 3.0,
                        "selected": False,
                        "trajectory_world": (trajectory + np.asarray([0.0, 1.0, 0.0], dtype=np.float32)).tolist(),
                    },
                    {
                        "score": 1.0,
                        "selected": True,
                        "trajectory_world": trajectory.tolist(),
                    },
                ],
            }
        }
        return {"agent0": trajectory}

    def get_last_debug(self):
        return self._last_debug


def test_overlay_platoon_labels_draws_scenario_marked_traffic_vehicle() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    traffic_vehicle = _FakeVehicle()
    traffic_vehicle.scenario_warning_marker = "!"
    env = type(
        "Env",
        (),
        {
            "top_down_renderer": _FakeRenderer(),
            "agents": {},
            "engine": type(
                "Engine",
                (),
                {
                    "traffic_manager": type(
                        "TrafficManager",
                        (),
                        {"_traffic_vehicles": [traffic_vehicle]},
                    )()
                },
            )(),
        },
    )()

    labelled = module._overlay_platoon_labels(frame, env, [])

    assert labelled.sum() > frame.sum()


def test_overlay_platoon_labels_draws_green_triangle_for_leader_only() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    vehicle = _FakeVehicle()
    env_leader = type(
        "Env",
        (),
        {
            "top_down_renderer": _FakeRenderer(),
            "agents": {"agent0": vehicle},
            "engine": type("Engine", (), {"traffic_manager": None})(),
            "get_agent_role": lambda self, agent_id: "leader",
        },
    )()
    env_follower = type(
        "Env",
        (),
        {
            "top_down_renderer": _FakeRenderer(),
            "agents": {"agent0": vehicle},
            "engine": type("Engine", (), {"traffic_manager": None})(),
            "get_agent_role": lambda self, agent_id: "follower",
        },
    )()

    leader_frame = module._overlay_platoon_labels(frame.copy(), env_leader, ["agent0"])
    follower_frame = module._overlay_platoon_labels(frame.copy(), env_follower, ["agent0"])

    assert leader_frame.sum() > follower_frame.sum()


def test_overlay_rule_maker_debug_draws_candidate_and_selected_trajectories() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    env = type(
        "Env",
        (),
        {
            "top_down_renderer": _FakeRenderer(),
            "agents": {},
            "engine": type("Engine", (), {"traffic_manager": None})(),
        },
    )()
    debug = {
        "agent_ids": ["agent0"],
        "candidates_by_agent": {
            "agent0": [
                {"action": -1, "trajectory_world": [[0.0, 0.0], [5.0, 3.5]], "selected": False},
                {"action": 0, "trajectory_world": [[0.0, 0.0], [5.0, 0.0]], "selected": True},
                {"action": 1, "trajectory_world": [[0.0, 0.0], [5.0, -3.5]], "selected": False},
            ]
        },
    }

    drawn = module._overlay_rule_maker_debug(frame, env, debug)

    assert drawn.sum() > frame.sum()


def test_pipeline_factory_runs_strict_joint_expert_chain(monkeypatch) -> None:
    fake_rule_maker = _FakeRuleMaker()
    fake_planner = _FakeLatticePlanner()

    class FakeExpert:
        def __init__(self, env, agent_ids):
            assert tuple(agent_ids) == ("agent0", "agent1", "agent2")
            self.env = env
            self.agent_ids = agent_ids
            self.rule_maker = fake_rule_maker
            self.planner = fake_planner
            self.pid_controller = SimpleNamespace(get_last_debug=lambda: {})
            self.lqr_controller = self.pid_controller

        def plan(self, env):
            decisions = self.rule_maker.compute(
                env,
                self.agent_ids,
                planner_batch={},
            )
            # The fake RuleMaker only populates agent0; make the strict joint
            # test inputs explicit for all roles.
            decisions = {
                agent_id: decisions.get(
                    agent_id,
                    {
                        "action": 0,
                        "target_point": np.asarray(
                            [5.0, 0.0],
                            dtype=np.float32,
                        ),
                    },
                )
                for agent_id in self.agent_ids
            }
            trajectories = {
                agent_id: np.asarray(
                    [[float(index), 0.0, 0.0] for index in range(8)],
                    dtype=np.float32,
                )
                for agent_id in self.agent_ids
            }
            fake_planner.calls.append(decisions)
            fake_planner._last_debug = {
                agent_id: {
                    "candidates": [
                        {
                            "score": 0.0,
                            "selected": True,
                            "trajectory_world": trajectory.tolist(),
                        }
                    ]
                }
                for agent_id, trajectory in trajectories.items()
            }
            env.apply_dynamic_roles(
                {
                    "agent0": "leader",
                    "agent1": "follower",
                    "agent2": "follower",
                }
            )
            return SimpleNamespace(
                controls={
                    agent_id: np.zeros(2, dtype=np.float32)
                    for agent_id in self.agent_ids
                },
                trajectories_world=trajectories,
                rule_actions={
                    agent_id: 0 for agent_id in self.agent_ids
                },
            )

    monkeypatch.setattr(module, "RulePlannerExpert", FakeExpert)

    env = type(
        "Env",
        (),
        {
            "agents": {
                f"agent{index}": _FakeVehicle()
                for index in range(3)
            },
            "_last_planner_batch": {"agent0": {"coarse_trajectories": np.zeros((3, 8, 2), dtype=np.float32)}},
            "config": {
                "target_speed_km_h": 30.0,
                "lookahead_index": 2,
                "controller_type": "stabilized",
            },
            "applied_roles": [],
            "apply_dynamic_roles": lambda self, roles: self.applied_roles.append(dict(roles)),
        },
    )()

    action_fn = module._build_pipeline_factory(
        "rule_maker",
        "lattice",
        "pid",
    )(env, ["agent0", "agent1", "agent2"], 59)
    actions = action_fn(env)

    assert "agent0" in actions
    assert actions["agent0"].shape == (2,)
    assert actions["agent0"].dtype == np.float32
    assert len(fake_planner.calls) == 1
    assert fake_planner.calls[0]["agent0"]["action"] == 0
    assert np.allclose(fake_planner.calls[0]["agent0"]["target_point"], [5.0, 0.0])
    assert getattr(env, "_preview_rule_maker_debug", None) is not None
    assert getattr(env, "_preview_planning_debug", None)["planning_policy"] == "lattice"
    assert getattr(env, "_preview_planning_debug", None)["coordinate_frame"] == "world"
    assert np.asarray(env._preview_planning_debug["trajectories_by_agent"]["agent0"]).shape == (8, 3)
    assert env.applied_roles == [
        {
            "agent0": "leader",
            "agent1": "follower",
            "agent2": "follower",
        }
    ]


def test_collect_episode_step_record_includes_control_debug() -> None:
    env = type(
        "Env",
        (),
        {
            "agents": {"agent0": _FakeVehicle()},
            "_preview_control_debug": {
                "agent0": {
                    "mode": "follower_lqr",
                    "lat_error": 0.25,
                    "heading_error": -0.1,
                    "K_lat": [2.0, 3.0],
                }
            },
        },
    )()

    record = module._collect_episode_step_record(
        env=env,
        agent_ids=["agent0"],
        step_idx=0,
        actions={"agent0": np.asarray([0.1, 0.2], dtype=np.float32)},
        info={"agent0": {"crash": False}},
        pdms={},
        planning_debug={},
        previous_speed_mps={},
        dt=0.5,
    )

    assert record["control_debug"]["agent0"]["mode"] == "follower_lqr"
    assert record["control_debug"]["agent0"]["lat_error"] == pytest.approx(0.25)


def test_collect_episode_step_record_prefers_explicit_control_and_preserves_planning_debug() -> None:
    env = type(
        "Env",
        (),
        {
            "agents": {"agent0": _FakeVehicle()},
            "_preview_control_debug": {"agent0": {"source": "preview"}},
        },
    )()
    planning_debug = {
        "planning_policy": "diffusion_refine_grpo",
        "coordinate_frame": "world",
        "agent_ids": ["agent0"],
        "trajectories_by_agent": {"agent0": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
        "candidates_by_agent": {"agent0": [{"mode": 0, "selected": True}]},
        "planner_debug": {"agent0": {"selected_mode": 0}},
    }

    record = module._collect_episode_step_record(
        env=env,
        agent_ids=["agent0"],
        step_idx=0,
        actions={"agent0": np.asarray([0.1, 0.2], dtype=np.float32)},
        info={"agent0": {"crash": False}},
        pdms={},
        planning_debug=planning_debug,
        control_debug={"agent0": {"source": "explicit"}},
        previous_speed_mps={},
        dt=0.5,
    )

    assert record["control_debug"] == {"agent0": {"source": "explicit"}}
    assert record["planning"]["coordinate_frame"] == "world"
    assert record["planning"]["planner_debug"] == {"agent0": {"selected_mode": 0}}


def test_overlay_planning_debug_draws_lattice_candidates_and_selected_trajectory() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    env = type(
        "Env",
        (),
        {
            "top_down_renderer": _FakeRenderer(),
            "agents": {},
            "engine": type("Engine", (), {"traffic_manager": None})(),
        },
    )()
    debug = {
        "planning_policy": "lattice",
        "agent_ids": ["agent0"],
        "candidates_by_agent": {
            "agent0": [
                {"score": 3.0, "trajectory_world": [[0.0, 0.0], [5.0, 1.0]], "selected": False},
                {"score": 1.0, "trajectory_world": [[0.0, 0.0], [5.0, 0.0]], "selected": True},
            ]
        },
        "trajectories_by_agent": {"agent0": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]},
    }

    drawn = module._overlay_planning_debug(frame, env, debug)

    assert drawn.sum() > frame.sum()


def test_compute_step_pdms_uses_planning_debug_not_rule_maker_debug(monkeypatch) -> None:
    fake_rule_maker = _FakeRuleMaker()
    del fake_rule_maker
    calls = {}

    monkeypatch.setattr(
        module,
        "compute_pdms_reward_batch",
        lambda trajectory, *args, **kwargs: (
            np.ones((1,), dtype=np.float32),
            {
                "reward": np.asarray([1.0], dtype=np.float32),
                "progress": np.asarray([float(trajectory[0, -1, 0])], dtype=np.float32),
                "formation_lon": np.asarray([0.0], dtype=np.float32),
                "formation_lat": np.asarray([0.0], dtype=np.float32),
                "speed": np.asarray([0.0], dtype=np.float32),
                "comfort": np.asarray([0.0], dtype=np.float32),
                "consistency": np.asarray([0.0], dtype=np.float32),
                "gate": np.asarray([1.0], dtype=np.float32),
            },
        ),
    )
    monkeypatch.setattr(module, "compute_pairwise_formation_reward", lambda *args, **kwargs: (np.ones(1), np.ones(1)))

    env = _FakeEnv([{"done_step": 0}])
    env.agents = {"agent0": _FakeVehicle()}
    env._preview_planning_debug = {
        "trajectories_by_agent": {
            "agent0": [[float(i), 0.0, 0.0] for i in range(8)],
        }
    }
    rule_debug = {
        "candidates_by_agent": {
            "agent0": [
                {"selected": True, "trajectory_world": [[100.0, 0.0], [101.0, 0.0]]},
            ]
        }
    }

    results = module._compute_step_pdms(
        env,
        ["agent0"],
        rule_debug,
        {"agent0": {"crash": False, "crash_vehicle": False}},
        {
            "desired_gap_m": 10.0,
            "lon_decay_m": 10.0,
            "lat_decay_m": 3.0,
            "progress_s_max": 30.0,
            "waypoint_decay_gamma": 0.9,
        },
    )

    calls.update(results)
    assert calls["agent0"]["progress"] == 7.0


def test_publication_agent_color_is_stable_and_colorblind_friendly() -> None:
    assert module._publication_agent_color("agent0") == "#0072B2"
    assert module._publication_agent_color("agent1") == "#D55E00"
    assert module._publication_agent_color("agent2") == "#009E73"
    assert module._publication_agent_color("agent8") == "#0072B2"
    assert module._publication_agent_color("__team__") == "#333333"


def test_save_publication_figure_uses_high_dpi_and_tight_bbox(tmp_path: Path) -> None:
    calls = []

    class _FakeFigure:
        def savefig(self, *args, **kwargs):
            calls.append((args, kwargs))

    module._save_publication_figure(_FakeFigure(), tmp_path / "nested" / "plot.png")

    assert calls
    assert calls[0][0][0] == tmp_path / "nested" / "plot.png"
    assert calls[0][1]["dpi"] == 300
    assert calls[0][1]["bbox_inches"] == "tight"
    assert calls[0][1]["facecolor"] == "white"


def test_plot_episode_pdms_empty_records_writes_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"

    module._plot_episode_pdms(path, [])

    assert path.exists()
    assert path.stat().st_size > 0


def test_extract_team_pdms_series_falls_back_to_agent_mean() -> None:
    episode = [
        {"pdms": {"agent0": {"reward": 1.0}, "agent1": {"reward": 3.0}}},
        {"pdms": {"__team__": {"reward": 5.0}, "agent0": {"reward": 1.0}}},
        {"pdms": {"agent0": {"progress": 2.0}}},
    ]

    series = module._extract_team_pdms_series(episode, "reward")

    np.testing.assert_allclose(series, np.asarray([2.0, 5.0, np.nan]), equal_nan=True)


def test_aggregate_pdms_across_episodes_aligns_by_longest_step_and_ignores_nan() -> None:
    episodes = [
        [
            {"pdms": {"agent0": {"reward": 1.0}, "agent1": {"reward": 3.0}}},
            {"pdms": {"agent0": {"reward": 5.0}, "agent1": {"reward": 7.0}}},
        ],
        [
            {"pdms": {"__team__": {"reward": 4.0}}},
        ],
    ]

    aggregated = module._aggregate_pdms_across_episodes(episodes)

    assert aggregated["steps"] == [0, 1]
    assert list(aggregated["metrics"].keys()) == ["reward"]
    np.testing.assert_allclose(aggregated["metrics"]["reward"]["mean"], np.asarray([3.0, 6.0]))
    np.testing.assert_allclose(aggregated["metrics"]["reward"]["std"], np.asarray([1.0, 0.0]))


def test_main_defaults_match_preview_scenario_script(monkeypatch) -> None:
    captured = {}

    def fake_run_scenario(**kwargs):
        captured.update(kwargs)
        return Path("/tmp/videos")

    monkeypatch.setattr(sys, "argv", ["preview_platoon.py"])
    monkeypatch.setattr(module, "run_scenario", fake_run_scenario)

    module.main()

    assert captured["scenario_id"] == "S7_ego_merge_from_ramp"
    assert captured["local_route"] is None
    assert captured["num_agents"] == 3
    assert captured["num_episodes"] == 3
    assert captured["output_root"] == Path("/media/kong/Elements_SE/Diffusion_Data/outputs/run_results")
    assert captured["heading_up"] is False
    assert captured["traffic_density"] == 0.0
    assert captured["start_seed"] == 11
    assert captured["fps"] == 10
    assert captured["decision_policy"] == "rule_maker"
    assert captured["planning_policy"] == "lattice"
    assert captured["control_policy"] == "adaptive"


def test_main_rejects_lqr_lateral_override_args(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preview_platoon.py",
            "--lqr-lat-q1",
            "2.0",
            "--lqr-lat-q2",
            "3.0",
            "--lqr-lat-r",
            "0.2",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()


def test_main_accepts_horizon_override_arg(monkeypatch) -> None:
    captured = {}

    def fake_run_scenario(**kwargs):
        captured.update(kwargs)
        return Path("/tmp/videos")

    monkeypatch.setattr(sys, "argv", ["preview_platoon.py", "--horizon", "456"])
    monkeypatch.setattr(module, "run_scenario", fake_run_scenario)

    module.main()

    assert captured["horizon"] == 456


def test_main_forwards_horizon_override_to_all_scenarios(monkeypatch) -> None:
    captured = {}

    def fake_run_all_scenarios(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(sys, "argv", ["preview_platoon.py", "--all-scenarios", "--horizon", "456"])
    monkeypatch.setattr(module, "run_all_scenarios", fake_run_all_scenarios)

    module.main()

    assert captured["horizon"] == 456


def test_run_preview_uses_start_seed_for_each_episode(monkeypatch, tmp_path: Path) -> None:
    fake_env = _FakeEnv([{"done_step": 2}, {"done_step": 2}])
    written = []

    monkeypatch.setattr(module, "_pick_local_route", lambda scenario_id: "R0")
    monkeypatch.setattr(module, "_capture_topdown_frame", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_write_video", lambda path, frames, fps: written.append((path, len(frames), fps)))
    monkeypatch.setattr(
        module,
        "_build_pipeline_factory",
        lambda *args, **kwargs: (lambda env, agent_ids, seed: (lambda env: {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids})),
    )
    video_dir = module.run_scenario(
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
    assert len(written) == 4


def test_run_scenario_forwards_horizon_to_env_config(monkeypatch, tmp_path: Path) -> None:
    fake_env = _FakeEnv([{"done_step": 1}])
    captured_config = {}

    monkeypatch.setattr(module, "_pick_local_route", lambda scenario_id: "R0")
    monkeypatch.setattr(module, "_capture_topdown_frame", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_write_video", lambda path, frames, fps: None)
    monkeypatch.setattr(
        module,
        "_build_pipeline_factory",
        lambda *args, **kwargs: (lambda env, agent_ids, seed: (lambda env: {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids})),
    )

    module.run_scenario(
        scenario_id="S1_free_cruise_straight",
        local_route=None,
        num_agents=3,
        num_episodes=1,
        output_root=tmp_path,
        heading_up=False,
        traffic_density=0.10,
        start_seed=59,
        fps=10,
        env_factory=lambda config: (captured_config.update(config) or fake_env),
        horizon=123,
    )

    assert captured_config["horizon"] == 123


def test_run_preview_retries_episode_when_it_ends_immediately(monkeypatch, tmp_path: Path) -> None:
    fake_env = _FakeEnv(
        [
            {"done_step": 0, "crash": True},
            {"done_step": 3, "crash": False},
        ]
    )
    written = []

    monkeypatch.setattr(module, "_pick_local_route", lambda scenario_id: "R0")
    monkeypatch.setattr(module, "_capture_topdown_frame", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_write_video", lambda path, frames, fps: written.append((path, len(frames), fps)))
    monkeypatch.setattr(
        module,
        "_build_pipeline_factory",
        lambda *args, **kwargs: (lambda env, agent_ids, seed: (lambda env: {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids})),
    )
    module.run_scenario(
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
        max_episode_retries=2,
    )

    assert fake_env.spawn_seeds == [59, 60]
    assert fake_env.reset_seeds == [None, None]
    assert len(written) == 2
    assert all(item[1] > 1 for item in written)


def test_format_episode_stop_reason_reports_crash_and_out_of_road_agents() -> None:
    info = {
        "agent0": {"crash": False, "out_of_road": True},
        "agent1": {"crash_vehicle": True, "out_of_road": False},
        "agent2": {"crash_object": True, "out_of_road": True},
    }

    message = module._format_episode_stop_reason(
        step_idx=7,
        terminated={"__all__": True},
        truncated={"__all__": False},
        info=info,
    )

    assert "step=7" in message
    assert "reason=crash,out_of_road" in message
    assert "crash_agents=agent1,agent2" in message
    assert "out_of_road_agents=agent0,agent2" in message


def test_format_episode_stop_reason_reports_truncation_without_agent_failures() -> None:
    message = module._format_episode_stop_reason(
        step_idx=3,
        terminated={"__all__": False},
        truncated={"__all__": True},
        info={"agent0": {"crash": False, "out_of_road": False}},
    )

    assert message == "Episode stopped: step=3 reason=truncated"


def test_run_single_episode_prints_stop_reason(monkeypatch, capsys) -> None:
    fake_env = _FakeEnv([{"done_step": 0, "crash": True}])
    monkeypatch.setattr(module, "_capture_topdown_frame", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    action_fn_factory = lambda env, agent_ids, seed: (  # noqa: E731
        lambda env: {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids}
    )

    module._run_single_episode(
        fake_env,
        ["agent0", "agent1", "agent2"],
        "agent0",
        False,
        59,
        action_fn_factory,
    )

    captured = capsys.readouterr()
    assert "Episode stopped: step=0 reason=crash" in captured.out
    assert "crash_agents=agent1" in captured.out


def test_run_single_episode_records_with_metadrive_low_level_step_dt(monkeypatch) -> None:
    fake_env = _FakeEnv([{"done_step": 0}])
    fake_env.config = {
        "pid_dt": 0.5,
        "physics_world_step_size": 0.03,
        "decision_repeat": 4,
    }
    recorded_dts = []

    def fake_collect_episode_step_record(**kwargs):
        recorded_dts.append(kwargs["dt"])
        return {"step_idx": kwargs["step_idx"], "vehicles": {}}

    monkeypatch.setattr(module, "_capture_topdown_frame", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_collect_episode_step_record", fake_collect_episode_step_record)
    action_fn_factory = lambda env, agent_ids, seed: (  # noqa: E731
        lambda env: {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids}
    )

    module._run_single_episode(
        fake_env,
        ["agent0", "agent1", "agent2"],
        "agent0",
        False,
        59,
        action_fn_factory,
        episode_step_records=[],
    )

    assert recorded_dts == [pytest.approx(0.12)]


def test_run_preview_writes_placeholder_when_no_topdown_frames_are_captured(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_env = _FakeEnv([{"done_step": 3}, {"done_step": 3}, {"done_step": 3}])

    monkeypatch.setattr(module, "_pick_local_route", lambda scenario_id: "R8_narrow_channel")
    monkeypatch.setattr(module, "_capture_topdown_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_build_pipeline_factory",
        lambda *args, **kwargs: (lambda env, agent_ids, seed: (lambda env: {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids})),
    )

    written = []
    monkeypatch.setattr(
        module,
        "_write_video",
        lambda path, frames, fps: written.append((path, len(frames), fps)),
    )
    module.run_scenario(
        scenario_id="S9_narrow_channel_negotiation",
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

    assert len(written) == 2
    assert written[0][1] == 1


def test_evaluate_writes_episode_metrics_json_and_plots(monkeypatch, tmp_path: Path) -> None:
    fake_env = _FakeEnv([{"done_step": 1}, {"done_step": 1}])

    def fake_pipeline_factory(*args, **kwargs):
        def factory(env, agent_ids, seed):
            del seed

            def action_fn(env):
                trajectory = [[float(i), 0.0, 0.0] for i in range(8)]
                env._preview_planning_debug = {
                    "planning_policy": "lattice",
                    "agent_ids": list(agent_ids),
                    "trajectories_by_agent": {aid: trajectory for aid in agent_ids},
                    "candidates_by_agent": {
                        aid: [{"selected": True, "trajectory_world": trajectory, "score": 0.0}]
                        for aid in agent_ids
                    },
                }
                return {aid: np.asarray([0.1, 0.2], dtype=np.float32) for aid in agent_ids}

            return action_fn

        return factory

    monkeypatch.setattr(module, "_pick_local_route", lambda scenario_id: "R0")
    monkeypatch.setattr(module, "_capture_topdown_frame", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_write_video", lambda path, frames, fps: path.write_bytes(b"video"))
    monkeypatch.setattr(module, "_build_pipeline_factory", fake_pipeline_factory)
    monkeypatch.setattr(
        module,
        "build_platoon_metric_params",
        lambda config: {
            "desired_gap_m": 10.0,
            "lon_decay_m": 10.0,
            "lat_decay_m": 3.0,
            "progress_s_max": 30.0,
            "waypoint_decay_gamma": 0.9,
        },
    )
    monkeypatch.setattr(
        module,
        "compute_pdms_reward_batch",
        lambda *args, **kwargs: (
            np.ones((1,), dtype=np.float32),
            {
                "reward": np.asarray([1.0], dtype=np.float32),
                "progress": np.asarray([2.0], dtype=np.float32),
                "formation_lon": np.asarray([0.3], dtype=np.float32),
                "formation_lat": np.asarray([0.4], dtype=np.float32),
                "speed": np.asarray([0.5], dtype=np.float32),
                "comfort": np.asarray([0.6], dtype=np.float32),
                "consistency": np.asarray([0.7], dtype=np.float32),
                "gate": np.asarray([1.0], dtype=np.float32),
            },
        ),
    )
    monkeypatch.setattr(module, "compute_pairwise_formation_reward", lambda *args, **kwargs: (np.ones(1), np.ones(1)))

    module.run_scenario(
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
        evaluate=True,
    )

    metrics_dir = tmp_path / "S1_free_cruise_straight" / "metrices"
    episode_json = metrics_dir / "episode_0000" / "metrice.json"
    assert episode_json.exists()
    assert (metrics_dir / "episode_0000" / "metrice.png").exists()
    assert (metrics_dir / "episode_0001" / "metrice.json").exists()
    assert (metrics_dir / "episode_0001" / "metrice.png").exists()
    assert (metrics_dir / "ave_metrice.png").exists()
    results_dir = metrics_dir / "episode_0000" / "results"
    for name in [
        "planned_trajectories.png",
        "xy.png",
        "speed_time.png",
        "accel_time.png",
        "heading_time.png",
        "distance.png",
        "steer_time.png",
        "throttle_time.png",
    ]:
        assert (results_dir / name).exists()

    data = json.loads(episode_json.read_text())
    assert data["episode"] == 0
    assert data["seed"] == 59
    assert data["video_path"].endswith("episode_0000.mp4")
    assert data["formation_unlock"] == {
        "episode": 0,
        "unlocked_triggered": False,
        "unlocked_ranges": [],
    }
    assert data["pdms"]["agent0"]["reward"] == 1.0
    first_step = data["steps"][0]
    assert first_step["pdms"]["agent0"]["reward"] == 1.0
    assert first_step["actions"]["agent0"]["steer"] == pytest.approx(0.1)
    assert first_step["actions"]["agent0"]["throttle"] == pytest.approx(0.2)
    assert "agent0" in first_step["planning"]["trajectories_by_agent"]
    assert "accel_mps2" in first_step["vehicles"]["agent0"]
    unlock_summary = json.loads((metrics_dir / "formation_unlock_summary.json").read_text())
    assert unlock_summary["formation_unlock_trigger_count"] == 0
    assert unlock_summary["formation_unlock_trigger_rate"] == 0.0
    assert len(unlock_summary["formation_unlock_events"]) == 2
    assert not (tmp_path / "S1_free_cruise_straight" / "reports" / "metrics").exists()


def test_evaluate_false_still_writes_expert_collection_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_env = _FakeEnv([{"done_step": 1}])

    monkeypatch.setattr(module, "_pick_local_route", lambda scenario_id: "R0")
    monkeypatch.setattr(module, "_capture_topdown_frame", lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_write_video", lambda path, frames, fps: path.write_bytes(b"video"))
    monkeypatch.setattr(
        module,
        "_build_pipeline_factory",
        lambda *args, **kwargs: (lambda env, agent_ids, seed: (lambda env: {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids})),
    )

    module.run_scenario(
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
        evaluate=False,
    )

    metrics_dir = tmp_path / "S1_free_cruise_straight" / "metrices"
    assert (metrics_dir / "expert_collection_summary.json").exists()
    assert not (metrics_dir / "formation_unlock_summary.json").exists()
