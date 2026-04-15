from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from metadrive.policy.diffusion_policy.test_transfuser_policy import (
    MULTIMODAL_OTHER_COLOR,
    MULTIMODAL_SELECTED_COLOR,
    SCENARIO_BY_ID,
    ScenarioRouteSelection,
    _capture_2d_topdown_frame,
    _capture_3d_topdown_frame,
    _apply_episode_route_config,
    _compute_plot_view_bounds,
    _extract_road_topology,
    _record_step_visualization,
    _resolve_episode_scenario_route,
    _save_step_image,
    _normalize_visualization_args,
    _save_trajectory_plot,
    _write_video,
    parse_args,
)
from metadrive.obs.diff_obs.top_down_state_obs_multi_channel import DatasetCollectObservation
from metadrive.policy.diffusion_policy.transfuser_callback import render_closed_loop_prediction


def test_parse_args_defaults_enable_headless_2d_outputs():
    args = parse_args(["--checkpoint", "dummy.ckpt"])

    assert args.render == 0
    assert args.scenario_id == "S1_free_cruise_straight"
    assert args.save_3d_video == 0
    assert args.save_2d_video == 1
    assert args.save_trajectory_plot == 1
    assert args.save_step_images == 0
    assert args.step_image_interval == 1
    assert args.video_fps == 10
    assert args.topdown_camera_height == 80.0


def test_normalize_visualization_args_enables_render_for_3d_video():
    args = parse_args(["--checkpoint", "dummy.ckpt", "--render", "0", "--save-3d-video", "1"])

    normalized = _normalize_visualization_args(args)

    assert normalized.render == 1


def test_resolve_episode_scenario_route_uses_allowed_routes_only():
    rng = np.random.RandomState(7)

    selection = _resolve_episode_scenario_route("S5_hard_brake_lead", rng)

    assert selection.scenario_id == "S5_hard_brake_lead"
    assert selection.local_route in SCENARIO_BY_ID["S5_hard_brake_lead"].allowed_local_routes
    assert len(selection.ego_main_route_block_ids) >= 1


def test_resolve_episode_scenario_route_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown scenario_id"):
        _resolve_episode_scenario_route("S404_missing", np.random.RandomState(0))


def test_apply_episode_route_config_updates_env_and_engine_global_config():
    env = types.SimpleNamespace(
        config={},
        engine=types.SimpleNamespace(global_config={}),
    )
    selection = ScenarioRouteSelection(
        scenario_id="S8_ego_exit_to_ramp",
        local_route="R6_exit_to_ramp",
        route_preset="ramp_merge",
        ego_main_route_block_ids=("g0", "s_ramp0", "c0_ramp0"),
    )

    _apply_episode_route_config(env, selection)

    assert env.config["scenario_id"] == "S8_ego_exit_to_ramp"
    assert env.config["local_route"] == "R6_exit_to_ramp"
    assert env.config["route_preset"] == "ramp_merge"
    assert env.config["ego_main_route_block_ids"] == ["g0", "s_ramp0", "c0_ramp0"]
    assert env.engine.global_config["scenario_id"] == "S8_ego_exit_to_ramp"


def test_capture_3d_topdown_frame_returns_rgb_and_tracks_ego():
    perceive_calls = {}

    class FakeCamera:
        def perceive(self, **kwargs):
            perceive_calls.update(kwargs)
            return np.zeros((6, 8, 4), dtype=np.uint8)

    ego = types.SimpleNamespace(position=np.asarray([3.0, -2.0], dtype=np.float32))
    env = types.SimpleNamespace(
        engine=types.SimpleNamespace(main_camera=FakeCamera(), origin=object()),
        agents={"agent0": ego},
    )

    frame = _capture_3d_topdown_frame(env, camera_height=90.0)

    assert frame.shape == (6, 8, 3)
    assert perceive_calls["position"] == (3.0, -2.0, 90.0)
    assert perceive_calls["hpr"] == (0, -90, 0)
    assert perceive_calls["to_float"] is False


def test_capture_3d_topdown_frame_returns_none_without_active_agent():
    env = types.SimpleNamespace(
        engine=types.SimpleNamespace(main_camera=object(), origin=object()),
        agents={},
    )

    frame = _capture_3d_topdown_frame(env, camera_height=90.0)

    assert frame is None


def test_capture_2d_topdown_frame_swaps_surfarray_axes():
    render_calls = {}
    ego = types.SimpleNamespace(position=np.asarray([1.0, 2.0], dtype=np.float32))

    class FakeEnv:
        agents = {"agent0": ego}

        def render(self, **kwargs):
            render_calls.update(kwargs)
            return np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    frame = _capture_2d_topdown_frame(FakeEnv(), screen_size=512, film_size=2048)

    assert frame.shape == (6, 4, 3)
    assert render_calls["mode"] == "top_down"
    assert render_calls["window"] is False
    assert render_calls["screen_size"] == (512, 512)
    assert render_calls["film_size"] == (2048, 2048)
    assert render_calls["camera_position"] == (1.0, 2.0)


def test_capture_2d_topdown_frame_returns_none_without_active_agent():
    class FakeEnv:
        agents = {}

        def render(self, **kwargs):
            raise AssertionError("render should not be called when no active agent exists")

    frame = _capture_2d_topdown_frame(FakeEnv())

    assert frame is None


def test_record_step_visualization_uses_cached_ego_when_active_agent_is_gone():
    class FakeEgo:
        position = np.asarray([10.0, 20.0], dtype=np.float64)

        def convert_to_world_coordinates(self, waypoint, position):
            return np.asarray([position[0] + waypoint[0], position[1] + waypoint[1]], dtype=np.float64)

    actual_positions = []
    planned_trajectories = []
    final_info = {
        "predicted_trajectory": np.asarray(
            [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        )
    }

    _record_step_visualization(
        ego_before_step=FakeEgo(),
        ego_xy_before_step=np.asarray([10.0, 20.0], dtype=np.float64),
        final_info=final_info,
        episode_length=1,
        save_trajectory_plot=True,
        actual_positions=actual_positions,
        planned_trajectories=planned_trajectories,
        multimodal_trajectories=[],
    )

    assert np.allclose(actual_positions[0], np.asarray([10.0, 20.0], dtype=np.float64))
    assert planned_trajectories[0][0] == 0
    planned_world = np.asarray(planned_trajectories[0][1])
    assert planned_world.shape == (2, 2)
    assert np.allclose(planned_world[0], np.asarray([11.0, 22.0], dtype=np.float64))


def test_record_step_visualization_collects_multimodal_candidates():
    class FakeEgo:
        position = np.asarray([5.0, 6.0], dtype=np.float64)

        def convert_to_world_coordinates(self, waypoint, position):
            return np.asarray([position[0] + waypoint[0], position[1] + waypoint[1]], dtype=np.float64)

    multimodal_trajectories = []
    final_info = {
        "trajectory_candidates": np.asarray(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.5, 0.0]],
                [[1.0, 1.0, 0.0], [2.0, 1.5, 0.0]],
            ],
            dtype=np.float32,
        ),
        "trajectory_mode_idx": 1,
    }

    _record_step_visualization(
        ego_before_step=FakeEgo(),
        ego_xy_before_step=np.asarray([5.0, 6.0], dtype=np.float64),
        final_info=final_info,
        episode_length=3,
        save_trajectory_plot=True,
        actual_positions=[],
        planned_trajectories=[],
        multimodal_trajectories=multimodal_trajectories,
    )

    assert multimodal_trajectories[0][0] == 2
    assert multimodal_trajectories[0][2] == 1
    candidates_world = multimodal_trajectories[0][1]
    assert candidates_world.shape == (2, 2, 2)
    assert np.allclose(candidates_world[0, 0], np.asarray([6.0, 6.0], dtype=np.float64))


def test_extract_road_topology_returns_lane_boundaries():
    class FakeLane:
        width = 4.0

        def get_polyline(self, interval=2, lateral=0.0):
            return np.asarray([[0.0, lateral], [10.0, lateral]], dtype=np.float64)

    env = types.SimpleNamespace(
        engine=types.SimpleNamespace(
            current_map=types.SimpleNamespace(
                road_network=types.SimpleNamespace(get_all_lanes=lambda: [FakeLane()])
            )
        )
    )

    boundaries = _extract_road_topology(env)

    assert len(boundaries) == 2
    assert np.allclose(boundaries[0], np.asarray([[0.0, 2.0], [10.0, 2.0]], dtype=np.float64))
    assert np.allclose(boundaries[1], np.asarray([[0.0, -2.0], [10.0, -2.0]], dtype=np.float64))


def test_plot_view_bounds_include_local_road_topology_without_showing_whole_map():
    actual_positions = [
        np.asarray([10.0, 10.0], dtype=np.float64),
        np.asarray([15.0, 12.0], dtype=np.float64),
        np.asarray([20.0, 15.0], dtype=np.float64),
    ]
    planned_trajectories = [
        (0, [np.asarray([22.0, 17.0], dtype=np.float64), np.asarray([24.0, 19.0], dtype=np.float64)]),
    ]
    multimodal_trajectories = [
        (
            1,
            np.asarray(
                [
                    [[18.0, 13.0], [23.0, 18.0]],
                    [[16.0, 9.0], [21.0, 8.0]],
                ],
                dtype=np.float64,
            ),
            0,
        )
    ]
    road_boundaries = [
        np.asarray([[8.0, 7.0], [12.0, 8.0], [18.0, 10.0], [25.0, 13.0]], dtype=np.float64),
        np.asarray([[8.0, 13.0], [12.0, 14.0], [18.0, 16.0], [25.0, 19.0]], dtype=np.float64),
        np.asarray([[120.0, 120.0], [150.0, 150.0]], dtype=np.float64),
    ]

    xlim, ylim = _compute_plot_view_bounds(
        actual_positions,
        planned_trajectories,
        multimodal_trajectories,
        road_boundaries=road_boundaries,
        padding=1.5,
        road_margin=3.0,
    )

    assert xlim[0] <= 10.0 and xlim[1] >= 24.0
    assert ylim[0] <= 7.0 and ylim[1] >= 19.0
    assert xlim[1] < 40.0
    assert ylim[1] < 40.0


def test_multimodal_plot_colors_match_requested_palette():
    assert MULTIMODAL_SELECTED_COLOR == "orange"
    assert MULTIMODAL_OTHER_COLOR == "#8FD3FF"


def test_dataset_collect_observation_observe_caches_latest_observation():
    observation = object.__new__(DatasetCollectObservation)
    observation.state_observe = lambda vehicle: np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    observation._split_lidar_observation = lambda vehicle: (
        np.asarray([4.0, 5.0], dtype=np.float32),
        np.asarray([6.0, 7.0], dtype=np.float32),
    )
    observation.topdown_obs = types.SimpleNamespace(observe=lambda vehicle: np.ones((4, 4, 3), dtype=np.float32))
    observation._observe_rgb_views = lambda vehicle: {
        "rgb_left": np.full((2, 2, 3), 0.1, dtype=np.float32),
        "rgb_front": np.full((2, 2, 3), 0.2, dtype=np.float32),
        "rgb_right": np.full((2, 2, 3), 0.3, dtype=np.float32),
    }
    observation.current_observation = None

    ret = DatasetCollectObservation.observe(observation, vehicle=object())

    assert observation.current_observation is ret
    assert set(ret.keys()) == {"ego_state", "others_state", "lidar", "topdown", "rgb_left", "rgb_front", "rgb_right"}
    assert np.allclose(ret["ego_state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_render_closed_loop_prediction_supports_missing_gt():
    features = {
        "camera_feature": np.zeros((3, 256, 768), dtype=np.float32),
        "lidar_feature": np.zeros((1, 256, 256), dtype=np.float32),
        "status_feature": np.zeros((19,), dtype=np.float32),
    }
    predictions = {
        "trajectory": np.zeros((8, 3), dtype=np.float32),
        "trajectory_mode_idx": 2,
    }

    image = render_closed_loop_prediction(
        features=features,
        predictions=predictions,
        config=types.SimpleNamespace(
            bev_background_color="#000000",
            bev_semantic_classes={1: ("road", "#111111"), 2: ("traffic", "#222222"), 3: ("ego", "#333333")},
            bev_pixel_size=0.5,
        ),
        anchors=np.zeros((8, 8, 2), dtype=np.float32),
        overlay_all_anchors=True,
        metadata_text=["episode=0 step=1"],
    )

    assert image.shape == (512, 1024, 3)


def test_save_step_image_writes_episode_step_png(tmp_path: Path):
    output_dir = tmp_path / "closed_loop"
    final_info = {
        "camera_feature": np.zeros((3, 256, 768), dtype=np.float32),
        "lidar_feature": np.zeros((1, 256, 256), dtype=np.float32),
        "status_feature": np.zeros((19,), dtype=np.float32),
        "predicted_trajectory": np.zeros((8, 3), dtype=np.float32),
        "trajectory_mode_idx": 1,
    }

    output_path = _save_step_image(
        final_info=final_info,
        config=types.SimpleNamespace(
            bev_background_color="#000000",
            bev_semantic_classes={1: ("road", "#111111"), 2: ("traffic", "#222222"), 3: ("ego", "#333333")},
            bev_pixel_size=0.5,
        ),
        output_dir=output_dir,
        episode_idx=0,
        step_idx=3,
        anchors=np.zeros((8, 8, 2), dtype=np.float32),
        overlay_all_anchors=True,
        metadata_text=["episode=0 step=3"],
    )

    assert output_path == output_dir / "step_images" / "episode_000" / "step_00003.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_trajectory_plot_writes_png(tmp_path: Path):
    actual_positions = [
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.5], dtype=np.float64),
        np.asarray([2.0, 1.0], dtype=np.float64),
    ]
    planned_trajectories = [
        (0, [np.asarray([0.8, 0.2], dtype=np.float64), np.asarray([1.6, 0.4], dtype=np.float64)]),
        (2, [np.asarray([2.8, 1.2], dtype=np.float64), np.asarray([3.5, 1.4], dtype=np.float64)]),
    ]
    multimodal_trajectories = [
        (
            0,
            np.asarray(
                [
                    [[0.8, 0.2], [1.6, 0.4]],
                    [[0.7, -0.1], [1.2, -0.2]],
                ],
                dtype=np.float64,
            ),
            0,
        )
    ]
    road_boundaries = [
        np.asarray([[0.0, -1.0], [3.0, -1.0]], dtype=np.float64),
        np.asarray([[0.0, 2.0], [3.0, 2.0]], dtype=np.float64),
    ]
    output_path = tmp_path / "trajectory_plots" / "episode_000.png"

    _save_trajectory_plot(
        actual_positions,
        planned_trajectories,
        multimodal_trajectories,
        road_boundaries,
        str(output_path),
        episode_idx=0,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_write_video_delegates_to_mediapy(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_write_video(path, frames, fps):
        captured["path"] = path
        captured["fps"] = fps
        captured["frames"] = frames

    monkeypatch.setitem(__import__("sys").modules, "mediapy", types.SimpleNamespace(write_video=fake_write_video))
    video_path = tmp_path / "videos_2d" / "episode_000.mp4"
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

    _write_video(str(video_path), frames, fps=12)

    assert captured["path"] == str(video_path)
    assert captured["fps"] == 12
    assert len(captured["frames"]) == 1
