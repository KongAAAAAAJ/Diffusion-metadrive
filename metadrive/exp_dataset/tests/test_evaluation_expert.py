from __future__ import annotations

import math
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import Mock

import numpy as np


def _load_module():
    metadrive_pkg = ModuleType("metadrive")
    metadrive_pkg.__path__ = []
    exp_dataset_pkg = ModuleType("metadrive.exp_dataset")
    exp_dataset_pkg.__path__ = []
    hierarchical_pkg = ModuleType("metadrive.exp_dataset.hierarchical_expert")
    hierarchical_pkg.__path__ = []
    sys.modules["metadrive"] = metadrive_pkg
    sys.modules["metadrive.exp_dataset"] = exp_dataset_pkg
    sys.modules["metadrive.exp_dataset.hierarchical_expert"] = hierarchical_pkg

    expert_policy_module = ModuleType("metadrive.exp_dataset.expert_idm_policy")
    expert_policy_module.ExpertIDMPolicy = type("ExpertIDMPolicy", (), {})
    driving_style_module = ModuleType("metadrive.exp_dataset.hierarchical_expert.driving_style")
    driving_style_module.DrivingStyleProfile = type("DrivingStyleProfile", (), {})
    hierarchical_policy_module = ModuleType("metadrive.exp_dataset.hierarchical_expert.hierarchical_policy")
    hierarchical_policy_module.HierarchicalExpertIDMPolicy = type("HierarchicalExpertIDMPolicy", (), {})
    sys.modules["metadrive.exp_dataset.expert_idm_policy"] = expert_policy_module
    sys.modules["metadrive.exp_dataset.hierarchical_expert.driving_style"] = driving_style_module
    sys.modules["metadrive.exp_dataset.hierarchical_expert.hierarchical_policy"] = hierarchical_policy_module

    module_path = Path(__file__).resolve().parents[1] / "evaluation_expert.py"
    spec = importlib.util.spec_from_file_location("evaluation_expert", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluation_expert = _load_module()
EpisodeMetrics = evaluation_expert.EpisodeMetrics
DEFAULT_OUTPUT_DIR = evaluation_expert.DEFAULT_OUTPUT_DIR
aggregate_episode_metrics = evaluation_expert.aggregate_episode_metrics
build_base_env_config = evaluation_expert.build_base_env_config
build_episode_video_path = evaluation_expert.build_episode_video_path
build_topdown_render_kwargs = evaluation_expert.build_topdown_render_kwargs
build_run_output_dir = evaluation_expert.build_run_output_dir
build_summary_json_path = evaluation_expert.build_summary_json_path
compute_heading_up_rotation_deg = evaluation_expert.compute_heading_up_rotation_deg
compute_jerk_stats = evaluation_expert.compute_jerk_stats
compute_ttc = evaluation_expert.compute_ttc
sync_topdown_camera_with_agent = evaluation_expert.sync_topdown_camera_with_agent
summarize_metrics = evaluation_expert.summarize_metrics


def test_build_expert_policy_creates_expert_idm_policy():
    vehicle = object()
    created = {}
    original_policy_cls = getattr(evaluation_expert, "ExpertIDMPolicy", None)
    try:
        evaluation_expert.ExpertIDMPolicy = Mock(
            side_effect=lambda control_object, random_seed: created.update(
                {
                    "vehicle": control_object,
                    "random_seed": random_seed,
                }
            ) or "policy"
        )

        policy = evaluation_expert.build_expert_policy(vehicle, random_seed=11)

        assert policy == "policy"
        assert created["vehicle"] is vehicle
        assert created["random_seed"] == 11
    finally:
        if original_policy_cls is not None:
            evaluation_expert.ExpertIDMPolicy = original_policy_cls


def test_compute_ttc_returns_gap_over_closing_speed():
    assert math.isclose(compute_ttc(ego_speed_ms=12.0, front_speed_ms=9.0, gap=30.0), 10.0)


def test_compute_ttc_returns_inf_without_positive_closing_speed_or_gap():
    assert math.isinf(compute_ttc(ego_speed_ms=8.0, front_speed_ms=8.0, gap=10.0))
    assert math.isinf(compute_ttc(ego_speed_ms=8.0, front_speed_ms=10.0, gap=10.0))
    assert math.isinf(compute_ttc(ego_speed_ms=8.0, front_speed_ms=3.0, gap=0.0))


def test_compute_jerk_stats_uses_second_order_speed_difference():
    avg_abs_jerk, max_abs_jerk = compute_jerk_stats([0.0, 1.0, 3.0, 6.0], dt=1.0)

    assert math.isclose(avg_abs_jerk, 1.0)
    assert math.isclose(max_abs_jerk, 1.0)


def test_compute_jerk_stats_falls_back_to_zero_for_short_sequences():
    assert compute_jerk_stats([1.0, 2.0], dt=0.1) == (0.0, 0.0)


def test_aggregate_episode_metrics_averages_continuous_values_and_merges_flags():
    metrics = aggregate_episode_metrics(
        agent_metrics=[
            EpisodeMetrics(
                route_completion=0.8,
                arrive_dest=True,
                crash=False,
                out_of_road=False,
                lane_change_count=1.0,
                min_ttc=4.0,
                avg_speed_km_h=30.0,
                avg_abs_jerk=0.4,
                max_abs_jerk=0.8,
                episode_length=20,
                episode_reward=10.0,
            ),
            EpisodeMetrics(
                route_completion=0.4,
                arrive_dest=True,
                crash=True,
                out_of_road=False,
                lane_change_count=3.0,
                min_ttc=float("inf"),
                avg_speed_km_h=50.0,
                avg_abs_jerk=0.6,
                max_abs_jerk=1.2,
                episode_length=20,
                episode_reward=6.0,
            ),
        ]
    )

    assert math.isclose(metrics.route_completion, 0.6)
    assert metrics.arrive_dest is True
    assert metrics.crash is True
    assert metrics.out_of_road is False
    assert math.isclose(metrics.lane_change_count, 2.0)
    assert math.isinf(metrics.min_ttc)
    assert math.isclose(metrics.avg_speed_km_h, 40.0)
    assert math.isclose(metrics.avg_abs_jerk, 0.5)
    assert math.isclose(metrics.max_abs_jerk, 1.0)
    assert metrics.episode_length == 20
    assert math.isclose(metrics.episode_reward, 8.0)


def test_summarize_metrics_reports_mean_rates_and_preserves_per_episode():
    episodes = [
        EpisodeMetrics(
            route_completion=1.0,
            arrive_dest=True,
            crash=False,
            out_of_road=False,
            lane_change_count=2.0,
            min_ttc=5.0,
            avg_speed_km_h=36.0,
            avg_abs_jerk=0.3,
            max_abs_jerk=0.9,
            episode_length=100,
            episode_reward=12.0,
        ),
        EpisodeMetrics(
            route_completion=0.5,
            arrive_dest=False,
            crash=True,
            out_of_road=True,
            lane_change_count=0.0,
            min_ttc=float("inf"),
            avg_speed_km_h=18.0,
            avg_abs_jerk=0.7,
            max_abs_jerk=1.1,
            episode_length=80,
            episode_reward=4.0,
        ),
    ]

    summary = summarize_metrics(episodes)

    assert summary.n_episodes == 2
    assert math.isclose(summary.success_rate, 0.5)
    assert math.isclose(summary.collision_rate, 0.5)
    assert math.isclose(summary.out_of_road_rate, 0.5)
    assert math.isclose(summary.mean_lane_change_count, 1.0)
    assert math.isclose(summary.mean_route_completion, 0.75)
    assert math.isinf(summary.mean_min_ttc)
    assert math.isclose(summary.mean_avg_speed_km_h, 27.0)
    assert math.isclose(summary.mean_avg_abs_jerk, 0.5)
    assert math.isclose(summary.mean_max_abs_jerk, 1.0)
    assert summary.per_episode == episodes


def test_summarize_metrics_handles_empty_input():
    summary = summarize_metrics([])

    assert summary.n_episodes == 0
    assert math.isclose(summary.success_rate, 0.0)
    assert math.isclose(summary.collision_rate, 0.0)
    assert math.isclose(summary.out_of_road_rate, 0.0)
    assert math.isclose(summary.mean_lane_change_count, 0.0)
    assert math.isclose(summary.mean_route_completion, 0.0)
    assert math.isinf(summary.mean_min_ttc)
    assert math.isclose(summary.mean_avg_speed_km_h, 0.0)
    assert math.isclose(summary.mean_avg_abs_jerk, 0.0)
    assert math.isclose(summary.mean_max_abs_jerk, 0.0)
    assert summary.per_episode == []


def test_output_paths_use_default_root_and_expected_names():
    run_dir = build_run_output_dir(DEFAULT_OUTPUT_DIR, expert_name="idm", seed=7, timestamp="20260331_120000")

    assert run_dir == "/media/kong/Elements_SE/Diffusion_Data/outputs/expert/idm_seed7_20260331_120000"
    assert build_episode_video_path(run_dir, 1).endswith("/episode_001.mp4")
    assert build_episode_video_path(run_dir, 12).endswith("/episode_012.mp4")
    assert build_summary_json_path(run_dir).endswith("/summary.json")


def test_build_base_env_config_prefers_agent0_and_configures_topdown_height():
    config = build_base_env_config(
        seed=3,
        n_episodes=5,
        topdown_camera_height=180.0,
        env_config={"traffic_density": 0.2},
    )

    assert config["use_render"] is False
    assert config["start_seed"] == 3
    assert config["num_scenarios"] == 5
    assert config["prefer_track_agent"] == "agent0"
    assert math.isclose(config["top_down_camera_initial_z"], 180.0)
    assert math.isclose(config["traffic_density"], 0.2)


def test_build_base_env_config_uses_standard_physics_control():
    config = build_base_env_config(seed=3, n_episodes=5)

    assert "control_mode" not in config
    assert "teleport_trajectory_steps" not in config
    assert "teleport_trajectory_dim" not in config


def test_build_topdown_render_kwargs_keeps_camera_following_agent():
    kwargs = build_topdown_render_kwargs(expert_name="idm", episode_index=2, step_count=15, screen_size=720, film_size=2400)

    assert kwargs["mode"] == "top_down"
    assert kwargs["window"] is False
    assert kwargs["screen_size"] == (720, 720)
    assert kwargs["film_size"] == (2400, 2400)
    assert kwargs["target_agent_heading_up"] is False
    assert "camera_position" not in kwargs
    assert "text" not in kwargs


def test_compute_heading_up_rotation_deg_matches_heading_up_convention():
    assert math.isclose(compute_heading_up_rotation_deg(0.0), 90.0)
    assert math.isclose(compute_heading_up_rotation_deg(math.pi / 2), 0.0)
    assert math.isclose(compute_heading_up_rotation_deg(-math.pi / 2), 180.0)


def test_sync_topdown_camera_with_agent_updates_existing_renderer_position():
    renderer = SimpleNamespace(position=None)
    env = SimpleNamespace(
        agents={"agent0": SimpleNamespace(position=np.asarray([12.5, -3.0], dtype=np.float32))},
        top_down_renderer=renderer,
    )

    sync_topdown_camera_with_agent(env, "agent0")

    assert renderer.position == (12.5, -3.0)


def test_parse_args_uses_standard_low_level_control_cli():
    args = evaluation_expert.parse_args(["--episodes", "2", "--seed", "9"])

    assert not hasattr(args, "expert")
    assert args.episodes == 2
    assert args.seed == 9
    assert not hasattr(args, "action_mode")
