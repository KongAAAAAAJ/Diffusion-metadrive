from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock
from enum import Enum

import numpy as np


def _make_episode_spec(
    route_preset="mainline",
    local_route="R1_entry_straight",
    traffic_density=0.1,
    spawn_seed=1,
    idm_variant=None,
    scenario_id="S1_free_cruise_straight",
):
    return collect_expert.EpisodeSpec(
        route_preset=route_preset,
        local_route=local_route,
        traffic_density=traffic_density,
        spawn_seed=spawn_seed,
        idm_variant=idm_variant,
        scenario_id=scenario_id,
    )


class _StubEpisodeSpecSampler:
    def __init__(self, specs):
        self._specs = list(specs)

    def __call__(self, config, rng):
        outer = self

        class _Sampler:
            def __init__(self):
                self._index = 0

            def sample(self):
                if self._index >= len(outer._specs):
                    return outer._specs[-1]
                spec = outer._specs[self._index]
                self._index += 1
                return spec

            def fast_forward(self, n_episodes):
                self._index = min(len(outer._specs), self._index + int(n_episodes))

        return _Sampler()


def _load_module():
    metadrive_package = ModuleType("metadrive")
    metadrive_package.__path__ = []
    exp_dataset_package = ModuleType("metadrive.exp_dataset")
    exp_dataset_package.__path__ = []
    policy_package = ModuleType("metadrive.policy")
    policy_package.__path__ = []
    diffusion_policy_package = ModuleType("metadrive.policy.diffusion_policy")
    diffusion_policy_package.__path__ = []
    sys.modules.setdefault("metadrive", metadrive_package)
    sys.modules.setdefault("metadrive.exp_dataset", exp_dataset_package)
    sys.modules.setdefault("metadrive.policy", policy_package)
    sys.modules.setdefault("metadrive.policy.diffusion_policy", diffusion_policy_package)

    class _StubConfig(dict):
        pass

    class _StubTrajectoryCorrectionContext:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _StubOutOfRoadByReferenceLaneRule:
        def __init__(self, *args, **kwargs):
            return None

        def check_sample(self, sample):
            return type(
                "Result",
                (),
                {
                    "passed": True,
                    "rejection_reasons": [],
                    "missing_reference_lane_points": 0,
                },
            )()

    class _StubTrajectoryFilterPipeline:
        def __init__(self, rules):
            self.rules = list(rules)

        def check_sample(self, sample):
            verdicts = [rule.check_sample(sample) for rule in self.rules]
            passed = all(verdict.passed for verdict in verdicts)
            rejection_reasons = []
            missing_reference_lane_points = 0
            for verdict in verdicts:
                rejection_reasons.extend(getattr(verdict, "rejection_reasons", []))
                missing_reference_lane_points += int(getattr(verdict, "missing_reference_lane_points", 0))
            return type(
                "Result",
                (),
                {
                    "passed": passed,
                    "rejection_reasons": rejection_reasons,
                    "missing_reference_lane_points": missing_reference_lane_points,
                },
            )()

    class _StubExpertIDMConfig:
        distance_wanted = 10.0
        time_wanted = 1.5
        delta = 10.0
        acc_factor = 1.0
        deacc_factor = -5.0
        normal_speed_kmh = 30.0
        max_speed_kmh = 100.0
        enable_lane_change = True
        lane_change_freq = 50
        lane_change_speed_increase = 10.0
        safe_lane_change_distance = 15.0
        max_long_dist = 30.0
        heading_pid_kp = 1.7
        heading_pid_ki = 0.01
        heading_pid_kd = 3.5
        lateral_pid_kp = 0.3
        lateral_pid_ki = 0.002
        lateral_pid_kd = 0.05

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    expert_idm_module = ModuleType("metadrive.exp_dataset.expert_idm_policy")
    expert_idm_module.ExpertIDMConfig = _StubExpertIDMConfig
    expert_idm_module.ExpertIDMPolicy = type("ExpertIDMPolicy", (), {})

    base_vehicle_module = ModuleType("metadrive.component.vehicle.base_vehicle")
    base_vehicle_module.BaseVehicle = type("BaseVehicle", (), {})

    env_module = ModuleType("metadrive.envs.diffusion_envs.base_multi_env")
    env_module.DatasetCollectEnv = type("DatasetCollectEnv", (), {})

    ppo_module = ModuleType("metadrive.examples.ppo_expert")
    ppo_module.expert = lambda *args, **kwargs: np.zeros(2, dtype=np.float32)

    trajectory_module = ModuleType("metadrive.exp_dataset.trajectory_correction")
    trajectory_module.TrajectoryCorrectionContext = _StubTrajectoryCorrectionContext
    trajectory_module.TrajectoryMode = Enum(
        "TrajectoryMode",
        {
            "KEEP_LANE": 0,
            "LANE_CHANGE_LEFT": 1,
            "LANE_CHANGE_RIGHT": 2,
        },
    )
    trajectory_module.classify_trajectory_mode = lambda *args, **kwargs: trajectory_module.TrajectoryMode.KEEP_LANE
    trajectory_module.correct_trajectory_geometry = lambda trajectory, *args, **kwargs: (trajectory, {})

    idm_module = ModuleType("metadrive.policy.idm_policy")
    idm_module.FrontBackObjects = type("FrontBackObjects", (), {"get_find_front_back_objs": staticmethod(lambda *args, **kwargs: None)})
    idm_module.IDMPolicy = type("IDMPolicy", (), {"MAX_LONG_DIST": 100.0})

    transfuser_features_module = ModuleType("metadrive.policy.diffusion_policy.transfuser_features")
    transfuser_features_module.BoundingBox2DIndex = type(
        "BoundingBox2DIndex",
        (),
        {"size": staticmethod(lambda: 5), "POINT": slice(0, 2)},
    )
    mode_context_module = ModuleType("metadrive.policy.diffusion_policy.mode_context")
    mode_context_module.build_mode_context_from_sample = lambda sample: None
    mode_context_module.build_mode_context_from_vehicle = lambda *args, **kwargs: None

    mode_labeler_module = ModuleType("metadrive.policy.diffusion_policy.mode_labeler")
    mode_labeler_module.label_hierarchical_mode = lambda *args, **kwargs: 0
    mode_labeler_module.label_mode_from_expert_decision = lambda *args, **kwargs: 0

    class _StubModeTrajectoryOutput:
        coarse_trajectories = np.zeros((1, 3, 2), dtype=np.float32)
        mode_valid_mask = np.ones((1,), dtype=bool)

    class _StubModeTrajectoryGenerator:
        num_mode_slots = 1
        mode_slots = ()

        def __init__(self, *args, **kwargs):
            return None

        def generate(self, *args, **kwargs):
            return _StubModeTrajectoryOutput()

    mode_trajectory_generator_module = ModuleType("metadrive.policy.diffusion_policy.mode_trajectory_generator")
    mode_trajectory_generator_module.ModeTrajectoryGenerator = _StubModeTrajectoryGenerator

    mode_visualization_module = ModuleType("metadrive.policy.diffusion_policy.mode_visualization")
    mode_visualization_module.ModeOverlayRenderContext = type("ModeOverlayRenderContext", (), {})
    mode_visualization_module.overlay_mode_trajectories_on_frame = lambda *args, **kwargs: args[0].frame if args else None
    mode_visualization_module.pick_recommended_mode = lambda *args, **kwargs: 0

    class _StubTransfuserConfig:
        mode_keep_high_speed_mps = 8.3
        mode_keep_medium_speed_mps = 5.0
        mode_keep_low_speed_mps = 2.0
        mode_emergency_decel_mps2 = 4.0
        mode_keep_lane_count = 3
        mode_lane_change_left_count = 3
        mode_lane_change_right_count = 3
        mode_emergency_stop_count = 1
        ego_fut_mode = 10

    transfuser_config_module = ModuleType("metadrive.policy.diffusion_policy.transfuser_config")
    transfuser_config_module.build_transfuser_config = lambda *args, **kwargs: _StubTransfuserConfig()
    transfuser_config_module.diffusion_model_config_to_overrides = lambda *args, **kwargs: {}
    transfuser_config_module.load_diffusion_model_config = lambda *args, **kwargs: {}
    transfuser_config_module.resolve_model_config_value = lambda *args, **kwargs: None

    utils_module = ModuleType("metadrive.utils")
    utils_module.Config = _StubConfig

    dataset_module = ModuleType("metadrive.exp_dataset.metadrive_dataset")
    dataset_module.split_shards = lambda **kwargs: {"train": [], "val": [], "test": []}

    trajectory_filter_module = ModuleType("metadrive.exp_dataset.trajectory_filter")
    trajectory_filter_module.OutOfRoadByReferenceLaneRule = _StubOutOfRoadByReferenceLaneRule
    trajectory_filter_module.TrajectoryFilterPipeline = _StubTrajectoryFilterPipeline

    scenario_orchestrator_module = ModuleType("metadrive.exp_dataset.scenario_orchestrator")
    scenario_orchestrator_module.ScenarioOrchestrator = type(
        "ScenarioOrchestrator",
        (),
        {
            "__init__": lambda self, definition, local_route: None,
            "reset": lambda self, env, agent_id: None,
            "before_step": lambda self, env, agent_id, step_count: None,
            "get_episode_summary": lambda self: {
                "scenario_id": "unknown",
                "scenario_triggered": False,
                "scenario_realized": False,
                "scenario_trigger_step": None,
                "scenario_realized_step": None,
                "scenario_notes": [],
            },
        },
    )

    injected_module_names = [
        expert_idm_module,
        base_vehicle_module,
        env_module,
        ppo_module,
        trajectory_module,
        idm_module,
        transfuser_features_module,
        mode_context_module,
        mode_labeler_module,
        mode_trajectory_generator_module,
        mode_visualization_module,
        transfuser_config_module,
        utils_module,
        dataset_module,
        trajectory_filter_module,
        scenario_orchestrator_module,
    ]
    for module in injected_module_names:
        sys.modules[module.__name__] = module

    route_definitions_path = Path(__file__).resolve().parents[1] / "route_definitions.py"
    route_definitions_spec = importlib.util.spec_from_file_location(
        "metadrive.exp_dataset.route_definitions",
        route_definitions_path,
    )
    route_definitions_module = importlib.util.module_from_spec(route_definitions_spec)
    assert route_definitions_spec is not None
    assert route_definitions_spec.loader is not None
    sys.modules[route_definitions_spec.name] = route_definitions_module
    route_definitions_spec.loader.exec_module(route_definitions_module)

    scenario_definitions_path = Path(__file__).resolve().parents[1] / "scenario_definitions.py"
    scenario_definitions_spec = importlib.util.spec_from_file_location(
        "metadrive.exp_dataset.scenario_definitions",
        scenario_definitions_path,
    )
    scenario_definitions_module = importlib.util.module_from_spec(scenario_definitions_spec)
    assert scenario_definitions_spec is not None
    assert scenario_definitions_spec.loader is not None
    sys.modules[scenario_definitions_spec.name] = scenario_definitions_module
    scenario_definitions_spec.loader.exec_module(scenario_definitions_module)

    local_traffic_spawner_path = Path(__file__).resolve().parents[1] / "local_traffic_spawner.py"
    local_traffic_spawner_spec = importlib.util.spec_from_file_location(
        "metadrive.exp_dataset.local_traffic_spawner",
        local_traffic_spawner_path,
    )
    local_traffic_spawner_module = importlib.util.module_from_spec(local_traffic_spawner_spec)
    assert local_traffic_spawner_spec is not None
    assert local_traffic_spawner_spec.loader is not None
    sys.modules[local_traffic_spawner_spec.name] = local_traffic_spawner_module
    local_traffic_spawner_spec.loader.exec_module(local_traffic_spawner_module)

    module_path = Path(__file__).resolve().parents[1] / "collect_expert.py"
    spec = importlib.util.spec_from_file_location("collect_expert", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for injected in injected_module_names:
        sys.modules.pop(injected.__name__, None)
    if sys.modules.get("metadrive.policy.diffusion_policy") is diffusion_policy_package:
        sys.modules.pop("metadrive.policy.diffusion_policy", None)
    if sys.modules.get("metadrive.policy") is policy_package:
        sys.modules.pop("metadrive.policy", None)
    if sys.modules.get("metadrive.exp_dataset") is exp_dataset_package:
        sys.modules.pop("metadrive.exp_dataset", None)
    if sys.modules.get("metadrive") is metadrive_package:
        sys.modules.pop("metadrive", None)
    sys.modules.pop(route_definitions_spec.name, None)
    sys.modules.pop(scenario_definitions_spec.name, None)
    sys.modules.pop(local_traffic_spawner_spec.name, None)
    return module


collect_expert = _load_module()
ExpertCollectorConfig = collect_expert.ExpertCollectorConfig
ShardWriter = collect_expert.ShardWriter
build_trajectory_mode_summary = collect_expert.build_trajectory_mode_summary
detect_existing_state = collect_expert.detect_existing_state
parse_args = collect_expert.parse_args
run_collection = collect_expert.run_collection
write_manifest = collect_expert.write_manifest
derive_sample_lateral_decision = collect_expert.derive_sample_lateral_decision


def test_derive_sample_lateral_decision_uses_future_reference_lane_window():
    assert derive_sample_lateral_decision(1, (1, 1, 0, 0)) == -1
    assert derive_sample_lateral_decision(1, (1, 1, 2, 2)) == 1
    assert derive_sample_lateral_decision(1, (1, 1, 1, 1)) == 0
    assert derive_sample_lateral_decision(-1, (1, 0, 0)) == 0


def test_build_episode_samples_includes_future_reference_metadata(monkeypatch):
    config = ExpertCollectorConfig(
        expert_type="idm",
        horizon_steps=8,
        target_stride_steps=1,
        sample_stride_steps=1,
        trajectory_num_poses=3,
        save_raw_trajectory=True,
    )

    monkeypatch.setattr(
        collect_expert,
        "classify_trajectory_mode",
        lambda trajectory, context: 0,
    )

    frames = []
    for idx in range(9):
        frames.append(
            {
                "ego_state": np.asarray([idx], dtype=np.float32),
                "other_states": np.asarray([idx], dtype=np.float32),
                "lidar": np.asarray([idx], dtype=np.float32),
                "bev_raster": np.zeros((1, 1, 1), dtype=np.uint8),
                "bev_semantic_map": np.zeros((1, 1, 1), dtype=np.uint8),
                "agent_states": np.zeros((1, 1), dtype=np.float32),
                "agent_labels": np.zeros((1,), dtype=np.int64),
                "left_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "front_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "right_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "state_275": np.zeros((275,), dtype=np.float32),
                "ego_pose_world": np.asarray([float(idx), 0.1 * idx, 0.0], dtype=np.float32),
                "reference_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "action": np.asarray([0.0, 0.0], dtype=np.float32),
                "ego_speed_km_h": np.asarray(10.0, dtype=np.float32),
                "front_object_distance": np.asarray(100.0, dtype=np.float32),
                "front_object_speed_km_h": np.asarray(5.0, dtype=np.float32),
                "lane_index": np.asarray(0, dtype=np.int16),
                "reference_lane_index": np.asarray(idx % 2, dtype=np.int16),
                "reference_longitudinal": np.asarray(float(idx), dtype=np.float32),
                "reference_lateral": np.asarray(0.1 * idx, dtype=np.float32),
                "lane_width": np.asarray(4.0 + 0.1 * idx, dtype=np.float32),
                "current_ref_lane_count": np.asarray(1, dtype=np.int16),
                "next_ref_lane_count": np.asarray(1, dtype=np.int16),
            }
        )

    samples = collect_expert.build_episode_samples(frames, config, traffic_density=0.2)

    assert len(samples) == 1
    sample = samples[0]
    assert sample["future_ego_pose_world"].shape == (3, 3)
    assert sample["future_reference_pose_world"].shape == (3, 3)
    assert sample["future_reference_lane_index"].shape == (3,)
    assert sample["future_lane_width"].shape == (3,)
    np.testing.assert_allclose(sample["future_ego_pose_world"][:, 0], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(sample["future_reference_pose_world"][:, 0], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(sample["future_reference_lane_index"], np.asarray([1, 0, 1], dtype=np.int16))
    np.testing.assert_allclose(sample["future_lane_width"], np.asarray([4.1, 4.2, 4.3], dtype=np.float32))


def test_build_episode_samples_tags_route_and_local_route():
    config = ExpertCollectorConfig(
        expert_type="idm",
        horizon_steps=8,
        target_stride_steps=1,
        sample_stride_steps=1,
        trajectory_num_poses=3,
    )
    frames = []
    for idx in range(9):
        frames.append(
            {
                "ego_state": np.asarray([idx], dtype=np.float32),
                "other_states": np.asarray([idx], dtype=np.float32),
                "lidar": np.asarray([idx], dtype=np.float32),
                "bev_raster": np.zeros((1, 1, 1), dtype=np.uint8),
                "bev_semantic_map": np.zeros((1, 1, 1), dtype=np.uint8),
                "agent_states": np.zeros((1, 1), dtype=np.float32),
                "agent_labels": np.zeros((1,), dtype=np.int64),
                "left_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "front_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "right_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "state_275": np.zeros((275,), dtype=np.float32),
                "ego_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "reference_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "action": np.asarray([0.0, 0.0], dtype=np.float32),
                "ego_speed_km_h": np.asarray(10.0, dtype=np.float32),
                "front_object_distance": np.asarray(100.0, dtype=np.float32),
                "front_object_speed_km_h": np.asarray(5.0, dtype=np.float32),
                "lane_index": np.asarray(0, dtype=np.int16),
                "reference_lane_index": np.asarray(0, dtype=np.int16),
                "reference_longitudinal": np.asarray(float(idx), dtype=np.float32),
                "reference_lateral": np.asarray(0.0, dtype=np.float32),
                "lane_width": np.asarray(4.0, dtype=np.float32),
                "current_ref_lane_count": np.asarray(1, dtype=np.int16),
                "next_ref_lane_count": np.asarray(1, dtype=np.int16),
            }
        )

    samples = collect_expert.build_episode_samples(
        frames,
        config,
        traffic_density=0.2,
        route_id="ramp_merge",
        local_route="R7_merge_core",
    )

    assert samples
    assert samples[0]["route_id"] == "ramp_merge"
    assert samples[0]["local_route"] == "R7_merge_core"


def test_build_episode_samples_tags_scenario_id():
    config = ExpertCollectorConfig(
        expert_type="idm",
        horizon_steps=8,
        target_stride_steps=1,
        sample_stride_steps=1,
        trajectory_num_poses=3,
    )
    frames = []
    for idx in range(9):
        frames.append(
            {
                "ego_state": np.asarray([idx], dtype=np.float32),
                "other_states": np.asarray([idx], dtype=np.float32),
                "lidar": np.asarray([idx], dtype=np.float32),
                "bev_raster": np.zeros((1, 1, 1), dtype=np.uint8),
                "bev_semantic_map": np.zeros((1, 1, 1), dtype=np.uint8),
                "agent_states": np.zeros((1, 1), dtype=np.float32),
                "agent_labels": np.zeros((1,), dtype=np.int64),
                "left_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "front_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "right_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "state_275": np.zeros((275,), dtype=np.float32),
                "ego_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "reference_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "action": np.asarray([0.0, 0.0], dtype=np.float32),
                "ego_speed_km_h": np.asarray(10.0, dtype=np.float32),
                "front_object_distance": np.asarray(100.0, dtype=np.float32),
                "front_object_speed_km_h": np.asarray(5.0, dtype=np.float32),
                "lane_index": np.asarray(0, dtype=np.int16),
                "reference_lane_index": np.asarray(0, dtype=np.int16),
                "reference_longitudinal": np.asarray(float(idx), dtype=np.float32),
                "reference_lateral": np.asarray(0.0, dtype=np.float32),
                "lane_width": np.asarray(4.0, dtype=np.float32),
                "current_ref_lane_count": np.asarray(1, dtype=np.int16),
                "next_ref_lane_count": np.asarray(1, dtype=np.int16),
            }
        )

    samples = collect_expert.build_episode_samples(
        frames,
        config,
        traffic_density=0.2,
        route_id="mainline",
        local_route="R1_entry_straight",
        scenario_id="S3_straight_following",
    )

    assert samples
    assert samples[0]["scenario_id"] == "S3_straight_following"


def test_build_episode_samples_tags_episode_id():
    config = ExpertCollectorConfig(
        expert_type="idm",
        horizon_steps=8,
        target_stride_steps=1,
        sample_stride_steps=1,
        trajectory_num_poses=3,
    )
    frames = []
    for idx in range(9):
        frames.append(
            {
                "ego_state": np.asarray([idx], dtype=np.float32),
                "other_states": np.asarray([idx], dtype=np.float32),
                "lidar": np.asarray([idx], dtype=np.float32),
                "bev_raster": np.zeros((1, 1, 1), dtype=np.uint8),
                "bev_semantic_map": np.zeros((1, 1, 1), dtype=np.uint8),
                "agent_states": np.zeros((1, 1), dtype=np.float32),
                "agent_labels": np.zeros((1,), dtype=np.int64),
                "left_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "front_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "right_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "state_275": np.zeros((275,), dtype=np.float32),
                "ego_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "reference_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "action": np.asarray([0.0, 0.0], dtype=np.float32),
                "ego_speed_km_h": np.asarray(10.0, dtype=np.float32),
                "front_object_distance": np.asarray(100.0, dtype=np.float32),
                "front_object_speed_km_h": np.asarray(5.0, dtype=np.float32),
                "lane_index": np.asarray(0, dtype=np.int16),
                "reference_lane_index": np.asarray(0, dtype=np.int16),
                "reference_longitudinal": np.asarray(float(idx), dtype=np.float32),
                "reference_lateral": np.asarray(0.0, dtype=np.float32),
                "lane_width": np.asarray(4.0, dtype=np.float32),
                "current_ref_lane_count": np.asarray(1, dtype=np.int16),
                "next_ref_lane_count": np.asarray(1, dtype=np.int16),
            }
        )

    samples = collect_expert.build_episode_samples(
        frames,
        config,
        traffic_density=0.2,
        route_id="mainline",
        local_route="R1_entry_straight",
        episode_index=17,
    )

    assert samples
    assert samples[0]["episode_id"] == 17


def test_build_episode_samples_tags_idm_variant():
    config = ExpertCollectorConfig(
        expert_type="idm",
        horizon_steps=8,
        target_stride_steps=1,
        sample_stride_steps=1,
        trajectory_num_poses=3,
    )
    frames = []
    for idx in range(9):
        frames.append(
            {
                "ego_state": np.asarray([idx], dtype=np.float32),
                "other_states": np.asarray([idx], dtype=np.float32),
                "lidar": np.asarray([idx], dtype=np.float32),
                "bev_raster": np.zeros((1, 1, 1), dtype=np.uint8),
                "bev_semantic_map": np.zeros((1, 1, 1), dtype=np.uint8),
                "agent_states": np.zeros((1, 1), dtype=np.float32),
                "agent_labels": np.zeros((1,), dtype=np.int64),
                "left_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "front_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "right_camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "camera": np.zeros((1, 1, 3), dtype=np.uint8),
                "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "state_275": np.zeros((275,), dtype=np.float32),
                "ego_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "reference_pose_world": np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
                "action": np.asarray([0.0, 0.0], dtype=np.float32),
                "ego_speed_km_h": np.asarray(10.0, dtype=np.float32),
                "front_object_distance": np.asarray(100.0, dtype=np.float32),
                "front_object_speed_km_h": np.asarray(5.0, dtype=np.float32),
                "lane_index": np.asarray(0, dtype=np.int16),
                "reference_lane_index": np.asarray(0, dtype=np.int16),
                "reference_longitudinal": np.asarray(float(idx), dtype=np.float32),
                "reference_lateral": np.asarray(0.0, dtype=np.float32),
                "lane_width": np.asarray(4.0, dtype=np.float32),
                "current_ref_lane_count": np.asarray(1, dtype=np.int16),
                "next_ref_lane_count": np.asarray(1, dtype=np.int16),
            }
        )

    samples = collect_expert.build_episode_samples(
        frames,
        config,
        traffic_density=0.2,
        route_id="mainline",
        local_route="R1_entry_straight",
        idm_variant="aggressive",
    )

    assert samples
    assert samples[0]["idm_variant"] == "aggressive"


def test_build_episode_rngs_uses_single_episode_rng():
    config = ExpertCollectorConfig(start_seed=11, spawn_seed_offset=100000)

    episode_rng = collect_expert.build_episode_rngs(config)

    sampled = [int(episode_rng.randint(0, 2**31 - 1)) for _ in range(3)]

    expected_rng = np.random.RandomState(config.start_seed)
    expected = [int(expected_rng.randint(0, 2**31 - 1)) for _ in range(3)]

    assert sampled == expected


def test_episode_spec_sampler_uses_single_rng_for_scenario_route_density_spawn_and_idm_variant():
    config = ExpertCollectorConfig(
        start_seed=5,
        scenario_weights={
            "S1_free_cruise_straight": 0.7,
            "S2_free_cruise_curve": 0.3,
        },
        local_route_weights={
            "R1_entry_straight": 1.0,
            "R2_entry_curve": 1.0,
            "R3_mainline_straight": 1.0,
            "R5_ramp_curve": 1.0,
            "R9_post_split_curve": 1.0,
        },
        idm_variant_weights={"default": 0.2, "aggressive": 0.8},
    )
    episode_rng = collect_expert.build_episode_rngs(config)
    sampler = collect_expert.EpisodeSpecSampler(config, episode_rng)

    sampled = [sampler.sample() for _ in range(3)]

    expected_rng = np.random.RandomState(config.start_seed)
    expected = []
    for _ in range(3):
        scenario_id = expected_rng.choice(
            ["S1_free_cruise_straight", "S2_free_cruise_curve"],
            p=[0.7, 0.3],
        )
        if scenario_id == "S1_free_cruise_straight":
            local_route = expected_rng.choice(["R1_entry_straight", "R3_mainline_straight"], p=[0.5, 0.5])
        else:
            local_route = expected_rng.choice(["R2_entry_curve", "R5_ramp_curve", "R9_post_split_curve"], p=[1 / 3, 1 / 3, 1 / 3])
        density = float(expected_rng.uniform(config.traffic_density_min, config.traffic_density_max))
        spawn_seed = int(expected_rng.randint(0, 2**31 - 1))
        idm_variant = expected_rng.choice(["default", "aggressive"], p=[0.2, 0.8])
        route = collect_expert.get_required_preset(local_route)
        expected.append(
            (
                scenario_id,
                route,
                local_route,
                density,
                spawn_seed,
                None if idm_variant == "default" else idm_variant,
            )
        )

    actual = [
        (
            spec.scenario_id,
            spec.route_preset,
            spec.local_route,
            spec.traffic_density,
            spec.spawn_seed,
            spec.idm_variant,
        )
        for spec in sampled
    ]
    assert actual == expected


def test_fast_forward_episode_rngs_advances_episode_spec_sampler():
    config = ExpertCollectorConfig(
        start_seed=5,
        local_route_weights={"R1_entry_straight": 0.5, "R5_ramp_curve": 0.5},
        idm_variant_weights={"default": 0.5, "conservative": 0.5},
    )
    episode_rng = collect_expert.build_episode_rngs(config)

    collect_expert.fast_forward_episode_rngs(episode_rng, config, completed_episodes=2)
    sampler = collect_expert.EpisodeSpecSampler(config, episode_rng)
    next_spec = sampler.sample()

    expected_rng = np.random.RandomState(config.start_seed)
    expected_sampler = collect_expert.EpisodeSpecSampler(config, expected_rng)
    expected_sampler.fast_forward(2)
    expected_next = expected_sampler.sample()

    assert next_spec == expected_next


def test_rollout_episode_sets_spawn_seed_before_reset(monkeypatch):
    recorded = {"spawn_seed": None}

    class DummySpawnManager:
        def set_episode_spawn_seed(self, seed):
            recorded["spawn_seed"] = int(seed)

    class DummyVehicle:
        position = np.asarray([0.0, 0.0], dtype=np.float32)
        heading_theta = 0.0

    class DummyEnv:
        def __init__(self):
            self.engine = type("Engine", (), {"spawn_manager": DummySpawnManager()})()
            self.agents = {"agent0": DummyVehicle()}

        def reset(self):
            return (
                {
                    "agent0": {
                        "ego_state": np.zeros((1,), dtype=np.float32),
                        "others_state": np.zeros((1,), dtype=np.float32),
                        "lidar": np.zeros((1,), dtype=np.float32),
                        "topdown": np.zeros((1, 1, 1), dtype=np.float32),
                        "rgb_left": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_front": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_right": np.zeros((1, 1, 3), dtype=np.float32),
                    }
                },
                {},
            )

        def step(self, actions):
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(collect_expert, "build_frame", lambda *args, **kwargs: {"frame": 1})
    monkeypatch.setattr(collect_expert, "ppo_expert", lambda *args, **kwargs: np.zeros((2,), dtype=np.float32))

    frames, map_geometry, video_frames, base_traffic_count = collect_expert.rollout_episode(
        DummyEnv(),
        ExpertCollectorConfig(expert_type="ppo", max_episode_steps=1),
        episode_spawn_seed=42,
        episode_index=1,
    )

    assert recorded["spawn_seed"] == 42
    assert frames[0]["frame"] == 1
    assert int(frames[0]["expert_lateral_decision"]) == 0
    assert float(frames[0]["expert_target_speed_km_h"]) == 30.0
    assert map_geometry is None
    assert video_frames == []
    assert base_traffic_count == 0


def test_rollout_episode_passes_custom_idm_config_to_build_expert_policy(monkeypatch):
    recorded = {"idm_config": None}

    class DummySpawnManager:
        def set_episode_spawn_seed(self, seed):
            return None

    class DummyVehicle:
        position = np.asarray([0.0, 0.0], dtype=np.float32)
        heading_theta = 0.0

    class DummyPolicy:
        def act(self):
            return np.zeros((2,), dtype=np.float32)

    class DummyEnv:
        def __init__(self):
            self.engine = type("Engine", (), {"spawn_manager": DummySpawnManager()})()
            self.agents = {"agent0": DummyVehicle()}

        def reset(self):
            return (
                {
                    "agent0": {
                        "ego_state": np.zeros((1,), dtype=np.float32),
                        "others_state": np.zeros((1,), dtype=np.float32),
                        "lidar": np.zeros((1,), dtype=np.float32),
                        "topdown": np.zeros((1, 1, 1), dtype=np.float32),
                        "rgb_left": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_front": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_right": np.zeros((1, 1, 3), dtype=np.float32),
                    }
                },
                {},
            )

        def step(self, actions):
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

    def fake_build_expert_policy(vehicle, random_seed, idm_config=None):
        recorded["idm_config"] = idm_config
        return DummyPolicy()

    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(collect_expert, "build_frame", lambda *args, **kwargs: {"frame": 1})
    monkeypatch.setattr(collect_expert, "build_expert_policy", fake_build_expert_policy)

    custom_idm_config = collect_expert.ExpertIDMConfig(normal_speed_kmh=41.0)
    frames, map_geometry, video_frames, base_traffic_count = collect_expert.rollout_episode(
        DummyEnv(),
        ExpertCollectorConfig(expert_type="idm", max_episode_steps=1),
        episode_spawn_seed=1,
        episode_index=1,
        idm_config=custom_idm_config,
    )

    assert frames[0]["frame"] == 1
    assert int(frames[0]["expert_lateral_decision"]) == 0
    assert float(frames[0]["expert_target_speed_km_h"]) == 30.0
    assert map_geometry is None
    assert video_frames == []
    assert base_traffic_count == 0
    assert recorded["idm_config"] is custom_idm_config


def test_rollout_episode_collects_video_frames_when_enabled(monkeypatch):
    captured = {"count": 0}

    class DummySpawnManager:
        def set_episode_spawn_seed(self, seed):
            return None

    class DummyVehicle:
        position = np.asarray([0.0, 0.0], dtype=np.float32)
        heading_theta = 0.0

    class DummyEnv:
        def __init__(self):
            self.engine = type("Engine", (), {"spawn_manager": DummySpawnManager()})()
            self.agents = {"agent0": DummyVehicle()}

        def reset(self):
            return (
                {
                    "agent0": {
                        "ego_state": np.zeros((1,), dtype=np.float32),
                        "others_state": np.zeros((1,), dtype=np.float32),
                        "lidar": np.zeros((1,), dtype=np.float32),
                        "topdown": np.zeros((1, 1, 1), dtype=np.float32),
                        "rgb_left": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_front": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_right": np.zeros((1, 1, 3), dtype=np.float32),
                    }
                },
                {},
            )

        def step(self, actions):
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(collect_expert, "build_frame", lambda *args, **kwargs: {"frame": 1})
    monkeypatch.setattr(collect_expert, "ppo_expert", lambda *args, **kwargs: np.zeros((2,), dtype=np.float32))
    monkeypatch.setattr(
        collect_expert,
        "capture_episode_topdown_frame",
        lambda env, episode_index, step_count: (
            captured.__setitem__("count", captured["count"] + 1)
            or np.zeros((2, 2, 3), dtype=np.uint8)
        ),
    )

    frames, map_geometry, video_frames, base_traffic_count = collect_expert.rollout_episode(
        DummyEnv(),
        ExpertCollectorConfig(expert_type="ppo", max_episode_steps=1, save_videos=True),
        episode_spawn_seed=42,
        episode_index=1,
    )

    assert frames == [{"frame": 1}]
    assert map_geometry is None
    assert len(video_frames) == 1
    assert captured["count"] == 1
    assert base_traffic_count == 0


def test_rollout_episode_skips_video_capture_when_disabled(monkeypatch):
    captured = {"count": 0}

    class DummySpawnManager:
        def set_episode_spawn_seed(self, seed):
            return None

    class DummyVehicle:
        position = np.asarray([0.0, 0.0], dtype=np.float32)
        heading_theta = 0.0

    class DummyEnv:
        def __init__(self):
            self.engine = type("Engine", (), {"spawn_manager": DummySpawnManager()})()
            self.agents = {"agent0": DummyVehicle()}

        def reset(self):
            return (
                {
                    "agent0": {
                        "ego_state": np.zeros((1,), dtype=np.float32),
                        "others_state": np.zeros((1,), dtype=np.float32),
                        "lidar": np.zeros((1,), dtype=np.float32),
                        "topdown": np.zeros((1, 1, 1), dtype=np.float32),
                        "rgb_left": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_front": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_right": np.zeros((1, 1, 3), dtype=np.float32),
                    }
                },
                {},
            )

        def step(self, actions):
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(collect_expert, "build_frame", lambda *args, **kwargs: {"frame": 1})
    monkeypatch.setattr(collect_expert, "ppo_expert", lambda *args, **kwargs: np.zeros((2,), dtype=np.float32))
    monkeypatch.setattr(
        collect_expert,
        "capture_episode_topdown_frame",
        lambda env, episode_index, step_count: (
            captured.__setitem__("count", captured["count"] + 1)
            or np.zeros((2, 2, 3), dtype=np.uint8)
        ),
    )

    frames, map_geometry, video_frames, base_traffic_count = collect_expert.rollout_episode(
        DummyEnv(),
        ExpertCollectorConfig(expert_type="ppo", max_episode_steps=1, save_videos=False),
        episode_spawn_seed=42,
        episode_index=1,
    )

    assert frames == [{"frame": 1}]
    assert map_geometry is None
    assert video_frames == []
    assert captured["count"] == 0
    assert base_traffic_count == 0


def test_rollout_episode_spawns_base_traffic_when_local_route_present(monkeypatch):
    recorded = {"calls": []}

    class DummySpawnManager:
        def set_episode_spawn_seed(self, seed):
            return None

    class DummyVehicle:
        position = np.asarray([0.0, 0.0], dtype=np.float32)
        heading_theta = 0.0

    class DummySpawner:
        def spawn_base_traffic(self, env, ego_vehicle, local_route, traffic_density, rng):
            recorded["calls"].append(
                {
                    "env": env,
                    "ego_vehicle": ego_vehicle,
                    "local_route": local_route,
                    "traffic_density": traffic_density,
                    "sample": float(rng.rand()),
                }
            )
            return 3

    class DummyEnv:
        def __init__(self):
            self.config = {"local_route": "R5_ramp_curve", "traffic_density": 0.12}
            self.engine = type("Engine", (), {"spawn_manager": DummySpawnManager()})()
            self.agents = {"agent0": DummyVehicle()}

        def reset(self):
            return (
                {
                    "agent0": {
                        "ego_state": np.zeros((1,), dtype=np.float32),
                        "others_state": np.zeros((1,), dtype=np.float32),
                        "lidar": np.zeros((1,), dtype=np.float32),
                        "topdown": np.zeros((1, 1, 1), dtype=np.float32),
                        "rgb_left": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_front": np.zeros((1, 1, 3), dtype=np.float32),
                        "rgb_right": np.zeros((1, 1, 3), dtype=np.float32),
                    }
                },
                {},
            )

        def step(self, actions):
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(collect_expert, "build_frame", lambda *args, **kwargs: {"frame": 1})
    monkeypatch.setattr(collect_expert, "ppo_expert", lambda *args, **kwargs: np.zeros((2,), dtype=np.float32))
    monkeypatch.setattr(collect_expert, "LocalTrafficSpawner", DummySpawner)

    frames, map_geometry, video_frames, base_traffic_count = collect_expert.rollout_episode(
        DummyEnv(),
        ExpertCollectorConfig(expert_type="ppo", max_episode_steps=1),
        episode_spawn_seed=7,
        episode_index=1,
        local_route="R5_ramp_curve",
    )

    assert frames == [{"frame": 1}]
    assert map_geometry is None
    assert video_frames == []
    assert base_traffic_count == 3


def test_capture_episode_topdown_frame_preserves_topdown_image_axes(monkeypatch):
    rendered = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    class DummyEnv:
        top_down_renderer = None

        def render(self, **kwargs):
            return rendered.copy()

    monkeypatch.setattr(collect_expert, "get_primary_agent_id", lambda env: "agent0")
    monkeypatch.setattr(collect_expert, "sync_topdown_camera_with_agent", lambda env, agent_id: (0.0, 0.0))
    monkeypatch.setattr(collect_expert, "build_topdown_render_kwargs", lambda **kwargs: {"mode": "top_down"})
    monkeypatch.setattr(collect_expert, "overlay_text_on_frame", lambda frame, lines: frame)
    monkeypatch.setattr(collect_expert, "build_overlay_lines", lambda episode_index, step_count: [])

    frame = collect_expert.capture_episode_topdown_frame(DummyEnv(), episode_index=1, step_count=2, heading_up=False)

    assert frame.shape == rendered.shape
    np.testing.assert_array_equal(frame, rendered)
    assert len(recorded["calls"]) == 1
    assert recorded["calls"][0]["local_route"] == "R5_ramp_curve"
    assert recorded["calls"][0]["traffic_density"] == 0.12


def test_build_expert_policy_passes_vehicle_seed_and_idm_config():
    vehicle = object()
    created = {}
    idm_config = object()

    original_expert = getattr(collect_expert, "Expert", None)
    try:
        collect_expert.Expert = Mock(
            side_effect=lambda control_object, random_seed, idm_config=None: (
                created.update(
                    {
                        "vehicle": control_object,
                        "random_seed": random_seed,
                        "idm_config": idm_config,
                    }
                )
                or "policy"
            )
        )

        policy = collect_expert.build_expert_policy(vehicle, random_seed=7, idm_config=idm_config)

        assert policy == "policy"
        assert created["vehicle"] is vehicle
        assert created["random_seed"] == 7
        assert created["idm_config"] is idm_config
    finally:
        if original_expert is not None:
            collect_expert.Expert = original_expert


def test_build_expert_idm_config_uses_collector_config_overrides():
    config = collect_expert.ExpertCollectorConfig(
        expert_idm_distance_wanted=6.0,
        expert_idm_time_wanted=1.1,
        expert_idm_enable_lane_change=False,
        expert_idm_lane_change_freq=19,
        expert_idm_heading_pid_kp=2.3,
        expert_idm_heading_pid_ki=0.11,
        expert_idm_heading_pid_kd=4.6,
        expert_idm_lateral_pid_kp=0.44,
        expert_idm_lateral_pid_ki=0.006,
        expert_idm_lateral_pid_kd=0.07,
    )

    built = collect_expert.build_expert_idm_config(config)

    assert built.distance_wanted == 6.0
    assert built.time_wanted == 1.1
    assert built.enable_lane_change is False
    assert built.lane_change_freq == 19
    assert built.heading_pid_kp == 2.3
    assert built.heading_pid_ki == 0.11
    assert built.heading_pid_kd == 4.6
    assert built.lateral_pid_kp == 0.44
    assert built.lateral_pid_ki == 0.006
    assert built.lateral_pid_kd == 0.07


def test_scenario_expert_overrides_apply_before_idm_variant():
    config = collect_expert.ExpertCollectorConfig()

    idm_config = collect_expert.build_expert_idm_config(config)
    scenario_overrides = collect_expert.SCENARIO_EXPERT_OVERRIDES["S4_curve_following"]
    idm_config = collect_expert.apply_idm_overrides(idm_config, scenario_overrides)
    variant_overrides = collect_expert.IDM_VARIANT_CONFIGS["aggressive"]
    idm_config = collect_expert.apply_idm_overrides(idm_config, variant_overrides)

    assert idm_config.time_wanted == 0.8
    assert idm_config.enable_lane_change is False
    assert idm_config.normal_speed_kmh == 38.0


def test_parse_args_defaults_save_videos_to_false(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["collect_expert.py"])

    config = parse_args()

    assert config.save_videos is False


def _write_shard(path: Path, count: int) -> None:
    values = np.arange(count, dtype=np.int64)
    np.savez_compressed(path, sample_id=values)


def test_detect_existing_state_prefers_manifest_counts_and_existing_mode_counts(tmp_path):
    shard_dir = tmp_path / "shards"
    report_dir = tmp_path / "reports"
    shard_dir.mkdir()
    report_dir.mkdir()
    _write_shard(shard_dir / "shard_000000.npz", 3)
    _write_shard(shard_dir / "shard_000001.npz", 4)
    manifest = {
        "collected_samples": 99,
        "episodes": 7,
        "trajectory_correction": {
            "mode_counts": {
                "keep_lane": 12,
                "lane_change_left": 3,
            }
        },
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    state = detect_existing_state(shard_dir, report_dir)

    assert state["shard_count"] == 2
    assert state["total_samples"] == 99
    assert state["total_episodes"] == 7
    assert state["mode_counts"] == manifest["trajectory_correction"]["mode_counts"]


def test_detect_existing_state_falls_back_to_scanning_shards(tmp_path):
    shard_dir = tmp_path / "shards"
    report_dir = tmp_path / "reports"
    shard_dir.mkdir()
    report_dir.mkdir()
    _write_shard(shard_dir / "shard_000000.npz", 3)
    _write_shard(shard_dir / "shard_000001.npz", 5)

    state = detect_existing_state(shard_dir, report_dir)

    assert state["shard_count"] == 2
    assert state["total_samples"] == 8
    assert state["total_episodes"] == 0
    assert state["mode_counts"] == {}


def test_shard_writer_resumes_from_existing_index(tmp_path):
    writer = ShardWriter(tmp_path, samples_per_shard=2, start_shard_index=3)

    writer.add_samples(
        [
            {"value": np.asarray([1], dtype=np.int64)},
            {"value": np.asarray([2], dtype=np.int64)},
        ]
    )

    assert (tmp_path / "shard_000003.npz").exists()


def test_write_manifest_appends_resume_history(tmp_path):
    dataset_root = tmp_path / "dataset"
    report_dir = dataset_root / "reports"
    report_dir.mkdir(parents=True)
    existing_manifest = {
        "collected_samples": 10,
        "episodes": 2,
        "resume_history": [{"timestamp": "20260331_100000", "added_samples": 10}],
    }
    (report_dir / "manifest.json").write_text(json.dumps(existing_manifest), encoding="utf-8")

    config = ExpertCollectorConfig(output_root=tmp_path, dataset_name="dataset", resume=True)
    write_manifest(
        config=config,
        dataset_root=dataset_root,
        report_dir=report_dir,
        total_samples=16,
        total_episodes=4,
        split_summary={"train": ["shard_000000.npz"], "val": [], "test": []},
        correction_summary={"mode_counts": {"keep_lane": 16}},
        filter_summary={"enabled": True, "accepted": 16, "rejected": 0},
        collection_wall_time_sec=1.5,
        resume=True,
    )

    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["collected_samples"] == 16
    assert manifest["episodes"] == 4
    assert len(manifest["resume_history"]) == 2
    assert manifest["resume_history"][-1]["added_samples"] == 6
    assert manifest["trajectory_filter"]["enabled"] is True


def test_write_manifest_includes_route_distribution(tmp_path):
    dataset_root = tmp_path / "dataset"
    report_dir = dataset_root / "reports"
    report_dir.mkdir(parents=True)

    config = ExpertCollectorConfig(
        output_root=tmp_path,
        dataset_name="dataset",
        local_route_weights={"R1_entry_straight": 0.5, "R5_ramp_curve": 0.5},
    )
    write_manifest(
        config=config,
        dataset_root=dataset_root,
        report_dir=report_dir,
        total_samples=16,
        total_episodes=4,
        split_summary={"train": ["shard_000000.npz"], "val": [], "test": []},
        correction_summary={"mode_counts": {"keep_lane": 16}},
        filter_summary={"enabled": True, "accepted": 16, "rejected": 0},
        collection_wall_time_sec=1.5,
        route_counts={"mainline": 10, "ramp_merge": 6},
        local_route_counts={"R1_entry_straight": 9, "R5_ramp_curve": 7},
        resume=False,
    )

    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["route_distribution"] == {"mainline": 10, "ramp_merge": 6}
    assert manifest["local_route_distribution"] == {"R1_entry_straight": 9, "R5_ramp_curve": 7}


def test_write_manifest_includes_scenario_distribution(tmp_path):
    dataset_root = tmp_path / "dataset"
    report_dir = dataset_root / "reports"
    report_dir.mkdir(parents=True)

    config = ExpertCollectorConfig(output_root=tmp_path, dataset_name="dataset")
    write_manifest(
        config=config,
        dataset_root=dataset_root,
        report_dir=report_dir,
        total_samples=16,
        total_episodes=4,
        split_summary={"train": ["shard_000000.npz"], "val": [], "test": []},
        correction_summary={"mode_counts": {"keep_lane": 16}},
        filter_summary={"enabled": True, "accepted": 16, "rejected": 0},
        collection_wall_time_sec=1.5,
        route_counts={"mainline": 10, "ramp_merge": 6},
        local_route_counts={"R1_entry_straight": 9, "R5_ramp_curve": 7},
        scenario_counts={"S1_free_cruise_straight": 9, "S7_ego_merge_from_ramp": 7},
        resume=False,
    )

    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["scenario_distribution"] == {
        "S1_free_cruise_straight": 9,
        "S7_ego_merge_from_ramp": 7,
    }


def test_run_collection_resume_fast_forwards_rng_and_reuses_existing_counts(monkeypatch, tmp_path):
    sampled_specs: list[collect_expert.EpisodeSpec] = []
    applied_route_block_ids: list[tuple[str, ...]] = []

    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)
            self.engine = type(
                "Engine",
                (),
                {
                    "spawn_manager": type("SpawnManager", (), {"set_episode_spawn_seed": lambda self, seed: None})(),
                    "global_config": {},
                },
            )()

        def close(self):
            return None

    class DummyWriter:
        last_instance = None

        def __init__(self, output_dir, samples_per_shard, start_shard_index=0):
            self.output_dir = output_dir
            self.samples_per_shard = samples_per_shard
            self.start_shard_index = start_shard_index
            self.added = []
            DummyWriter.last_instance = self

        def add_samples(self, samples):
            self.added.extend(samples)
            return len(samples)

        def close(self):
            return 0

    def fake_rollout_episode(env, config, episode_spawn_seed, episode_index, idm_config=None, local_route=""):
        sampled_specs.append(
            collect_expert.EpisodeSpec(
                scenario_id=str(env.config.get("scenario_id")),
                route_preset=str(env.config.get("route_preset")),
                local_route=str(env.config.get("local_route")),
                traffic_density=float(env.config.get("traffic_density")),
                spawn_seed=int(episode_spawn_seed),
                idm_variant=None,
            )
        )
        applied_route_block_ids.append(tuple(env.config.get("ego_main_route_block_ids", ())))
        return ([{"frame": 1}], None, [], 2)

    def fake_build_episode_samples(*args, **kwargs):
        return [
            {
                "trajectory_mode": np.asarray(0, dtype=np.int64),
                "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32),
            }
        ]

    written_manifest = {}

    def fake_write_manifest(
        config,
        dataset_root,
        report_dir,
        total_samples,
        total_episodes,
        split_summary,
        correction_summary,
        collection_wall_time_sec,
        filter_summary=None,
        route_counts=None,
        local_route_counts=None,
        scenario_counts=None,
        resume=False,
    ):
        written_manifest.update(
            {
                "config": config,
                "dataset_root": dataset_root,
                "report_dir": report_dir,
                "total_samples": total_samples,
                "total_episodes": total_episodes,
                "split_summary": split_summary,
                "correction_summary": correction_summary,
                "filter_summary": filter_summary,
                "route_counts": route_counts,
                "local_route_counts": local_route_counts,
                "scenario_counts": scenario_counts,
                "collection_wall_time_sec": collection_wall_time_sec,
                "resume": resume,
            }
        )

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", DummyWriter)
    monkeypatch.setattr(collect_expert, "rollout_episode", fake_rollout_episode)
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(collect_expert, "build_episode_samples", fake_build_episode_samples)
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(collect_expert, "write_manifest", fake_write_manifest)
    monkeypatch.setattr(
        collect_expert,
        "get_route_blocks",
        lambda route_name: {
            "R1_entry_straight": ("s0",),
            "R3_mainline_straight": ("s_main0",),
            "R5_ramp_curve": ("s_ramp0", "c0_ramp0"),
            "R9_post_split_curve": ("c4",),
        }[route_name],
    )
    monkeypatch.setattr(
        collect_expert,
        "detect_existing_state",
        lambda shard_dir, report_dir: {
            "shard_count": 4,
            "total_samples": 5,
            "total_episodes": 2,
            "mode_counts": {"keep_lane": 5},
            "route_counts": {"mainline": 5},
            "local_route_counts": {"R1_entry_straight": 5},
            "scenario_counts": {"S1_free_cruise_straight": 5},
        },
    )

    config = ExpertCollectorConfig(
        output_root=tmp_path,
        dataset_name="resume_dataset",
        target_samples=6,
        samples_per_shard=2,
        traffic_density_min=0.1,
        traffic_density_max=0.9,
        resume=True,
    )

    run_collection(config)

    expected_rng = np.random.RandomState(config.start_seed)
    expected_sampler = collect_expert.EpisodeSpecSampler(config, expected_rng)
    expected_sampler.fast_forward(2)
    expected_next_spec = expected_sampler.sample()

    assert DummyWriter.last_instance is not None
    assert DummyWriter.last_instance.start_shard_index == 4
    assert [(spec.route_preset, spec.local_route, spec.traffic_density, spec.spawn_seed) for spec in sampled_specs] == [
        (expected_next_spec.route_preset, expected_next_spec.local_route, expected_next_spec.traffic_density, expected_next_spec.spawn_seed)
    ]
    assert applied_route_block_ids == [tuple(collect_expert.get_route_blocks(expected_next_spec.local_route))]
    assert written_manifest["total_samples"] == 6
    assert written_manifest["total_episodes"] == 3
    assert written_manifest["resume"] is True
    assert written_manifest["correction_summary"]["mode_counts"]["keep_lane"] == 6
    assert written_manifest["filter_summary"]["accepted"] == 1
    assert written_manifest["filter_summary"]["rejected"] == 0
    assert written_manifest["route_counts"]["mainline"] >= 5
    assert written_manifest["local_route_counts"]["R1_entry_straight"] >= 5
    assert written_manifest["scenario_counts"]["S1_free_cruise_straight"] >= 5
    assert build_trajectory_mode_summary(written_manifest["correction_summary"]["mode_counts"], 6)["keep_lane"]["count"] == 6


def test_run_collection_filters_rejected_samples_before_writer(monkeypatch, tmp_path):
    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)
            self.engine = type("Engine", (), {"global_config": {}})()

        def close(self):
            return None

    class DummyWriter:
        last_instance = None

        def __init__(self, *args, **kwargs):
            self.added = []
            DummyWriter.last_instance = self

        def add_samples(self, samples):
            self.added.append(list(samples))
            return len(samples)

        def close(self):
            return 0

    class DummyFilter:
        def check_sample(self, sample):
            passed = bool(sample["keep"])
            return type(
                "Result",
                (),
                {
                    "passed": passed,
                    "rejection_reasons": ([] if passed else ["out_of_road"]),
                    "missing_reference_lane_points": 0,
                },
            )()

    captured_manifest = {}

    def fake_write_manifest(*args, **kwargs):
        captured_manifest["filter_summary"] = kwargs.get("filter_summary")

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", DummyWriter)
    monkeypatch.setattr(
        collect_expert,
        "EpisodeSpecSampler",
        _StubEpisodeSpecSampler([_make_episode_spec(traffic_density=0.1, spawn_seed=1)]),
    )
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index, idm_config=None, local_route="": ([{"frame": 1}], None, [], 0),
    )
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(
        collect_expert,
        "build_episode_samples",
        lambda *args, **kwargs: [
            {"keep": True, "trajectory_mode": np.asarray(0, dtype=np.int64), "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32), "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32), "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32), "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32), "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32)},
            {"keep": False, "trajectory_mode": np.asarray(0, dtype=np.int64), "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32), "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32), "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32), "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32), "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32)},
        ],
    )
    monkeypatch.setattr(collect_expert, "build_trajectory_filter_pipeline", lambda config: DummyFilter())
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(collect_expert, "write_manifest", fake_write_manifest)

    run_collection(
        ExpertCollectorConfig(
            output_root=tmp_path,
            dataset_name="filtered_dataset",
            target_samples=1,
            trajectory_filter_enabled=True,
        )
    )

    assert DummyWriter.last_instance is not None
    assert len(DummyWriter.last_instance.added) == 1
    assert len(DummyWriter.last_instance.added[0]) == 1
    assert captured_manifest["filter_summary"]["accepted"] == 1
    assert captured_manifest["filter_summary"]["rejected"] == 1
    assert captured_manifest["filter_summary"]["rejection_reasons"]["out_of_road"] == 1


def test_run_collection_saves_qual_and_unqual_visualizations(monkeypatch, tmp_path):
    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)
            self.engine = type("Engine", (), {"global_config": {}})()

        def close(self):
            return None

    class DummyWriter:
        def add_samples(self, samples):
            return len(samples)

        def close(self):
            return 0

    class DummyFilter:
        def check_sample(self, sample):
            passed = bool(sample["keep"])
            return type(
                "Result",
                (),
                {
                    "passed": passed,
                    "rejection_reasons": ([] if passed else ["out_of_road"]),
                    "missing_reference_lane_points": 0,
                },
            )()

    captured_paths = []

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: DummyWriter())
    monkeypatch.setattr(
        collect_expert,
        "EpisodeSpecSampler",
        _StubEpisodeSpecSampler([_make_episode_spec(traffic_density=0.1, spawn_seed=1)]),
    )
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index, idm_config=None, local_route="": ([{"frame": 1}], None, [], 0),
    )
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: [])
    monkeypatch.setattr(
        collect_expert,
        "build_episode_samples",
        lambda *args, **kwargs: [
            {
                "keep": True,
                "trajectory": np.zeros((2, 3), dtype=np.float32),
                "trajectory_mode": np.asarray(0, dtype=np.int64),
                "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32),
                "_trajectory_raw": np.zeros((2, 3), dtype=np.float32),
                "_sample_index": 3,
                "_current_pose": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            },
            {
                "keep": False,
                "trajectory": np.ones((2, 3), dtype=np.float32),
                "trajectory_mode": np.asarray(0, dtype=np.int64),
                "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32),
                "_trajectory_raw": np.ones((2, 3), dtype=np.float32),
                "_sample_index": 5,
                "_current_pose": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            },
        ],
    )
    monkeypatch.setattr(collect_expert, "build_trajectory_filter_pipeline", lambda config: DummyFilter())
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(collect_expert, "write_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        collect_expert,
        "save_trajectory_visualization",
        lambda output_path, **kwargs: captured_paths.append(Path(output_path)),
    )

    run_collection(
        ExpertCollectorConfig(
            output_root=tmp_path,
            dataset_name="viz_dataset",
            target_samples=1,
            trajectory_filter_enabled=True,
            trajectory_visualization_enabled=True,
            expert_type="idm",
        )
    )

    assert any(str(path).endswith("reports/trajectory_visualizations/qual/ep_00001_sample_00003_keep_lane.png") for path in captured_paths)
    assert any(str(path).endswith("reports/trajectory_visualizations/unqual/ep_00001_sample_00005_keep_lane.png") for path in captured_paths)


def test_run_collection_does_not_save_filter_visualizations_when_disabled(monkeypatch, tmp_path):
    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)
            self.engine = type("Engine", (), {"global_config": {}})()

        def close(self):
            return None

    class DummyWriter:
        def add_samples(self, samples):
            return len(samples)

        def close(self):
            return 0

    class DummyFilter:
        def check_sample(self, sample):
            passed = bool(sample["keep"])
            return type(
                "Result",
                (),
                {
                    "passed": passed,
                    "rejection_reasons": ([] if passed else ["out_of_road"]),
                    "missing_reference_lane_points": 0,
                },
            )()

    captured = {"calls": 0}

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: DummyWriter())
    monkeypatch.setattr(
        collect_expert,
        "EpisodeSpecSampler",
        _StubEpisodeSpecSampler([_make_episode_spec(traffic_density=0.1, spawn_seed=1)]),
    )
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index, idm_config=None, local_route="": ([{"frame": 1}], None, [], 0),
    )
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: [])
    monkeypatch.setattr(
        collect_expert,
        "build_episode_samples",
        lambda *args, **kwargs: [
            {
                "keep": False,
                "trajectory": np.zeros((2, 3), dtype=np.float32),
                "trajectory_mode": np.asarray(0, dtype=np.int64),
                "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32),
                "_trajectory_raw": np.zeros((2, 3), dtype=np.float32),
                "_sample_index": 1,
                "_current_pose": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            },
            {
                "keep": True,
                "trajectory": np.zeros((2, 3), dtype=np.float32),
                "trajectory_mode": np.asarray(0, dtype=np.int64),
                "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32),
                "_trajectory_raw": np.zeros((2, 3), dtype=np.float32),
                "_sample_index": 3,
                "_current_pose": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            },
        ],
    )
    monkeypatch.setattr(collect_expert, "build_trajectory_filter_pipeline", lambda config: DummyFilter())
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(collect_expert, "write_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        collect_expert,
        "save_trajectory_visualization",
        lambda *args, **kwargs: captured.__setitem__("calls", captured["calls"] + 1),
    )

    run_collection(
        ExpertCollectorConfig(
            output_root=tmp_path,
            dataset_name="viz_disabled_dataset",
            target_samples=1,
            trajectory_filter_enabled=True,
            trajectory_visualization_enabled=False,
            expert_type="idm",
        )
    )

    assert captured["calls"] == 0


def test_run_collection_uses_standard_physics_env_config(monkeypatch, tmp_path):
    captured = {}

    class DummyEnv:
        def __init__(self, config):
            captured["env_config"] = dict(config)
            self.config = dict(config)
            self.engine = type("Engine", (), {"global_config": {}})()

        def close(self):
            return None

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: type("W", (), {"add_samples": lambda self, samples: 0, "close": lambda self: 0})())
    monkeypatch.setattr(
        collect_expert,
        "EpisodeSpecSampler",
        _StubEpisodeSpecSampler([_make_episode_spec(traffic_density=0.1, spawn_seed=1)]),
    )
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index, idm_config=None, local_route="": ([], None, [], 0),
    )
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(collect_expert, "build_episode_samples", lambda *args, **kwargs: [])
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(collect_expert, "write_manifest", lambda *args, **kwargs: None)

    run_collection(
        ExpertCollectorConfig(
            output_root=tmp_path,
            dataset_name="control_mode_dataset",
            target_samples=0,
            trajectory_num_poses=8,
        )
    )

    assert "control_mode" not in captured["env_config"]
    assert "teleport_trajectory_steps" not in captured["env_config"]
    assert "teleport_trajectory_dim" not in captured["env_config"]


def test_run_collection_writes_episode_videos_when_enabled(monkeypatch, tmp_path):
    captured = {"paths": []}

    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)
            self.engine = type("Engine", (), {"global_config": {}})()

        def close(self):
            return None

    class DummyWriter:
        def add_samples(self, samples):
            return len(samples)

        def close(self):
            return 0

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: DummyWriter())
    monkeypatch.setattr(
        collect_expert,
        "EpisodeSpecSampler",
        _StubEpisodeSpecSampler([_make_episode_spec(traffic_density=0.1, spawn_seed=1)]),
    )
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index, idm_config=None, local_route="": ([{"frame": 1}], None, [np.zeros((2, 2, 3), dtype=np.uint8)], 0),
    )
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(
        collect_expert,
        "build_episode_samples",
        lambda *args, **kwargs: [
            {
                "trajectory_mode": np.asarray(0, dtype=np.int64),
                "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32),
            }
        ],
    )
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(collect_expert, "write_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        collect_expert,
        "write_episode_video",
        lambda video_path, frames, fps: captured["paths"].append((video_path, len(frames), fps)),
    )

    run_collection(
        ExpertCollectorConfig(
            output_root=tmp_path,
            dataset_name="video_dataset",
            target_samples=1,
            save_videos=True,
            video_fps=12,
        )
    )

    assert len(captured["paths"]) == 1
    video_path, frame_count, fps = captured["paths"][0]
    assert str(video_path).endswith("reports/videos/S1_free_cruise_straight/episode_000001.mp4")
    assert frame_count == 1
    assert fps == 12


def test_build_episode_video_path_uses_scenario_subdirectory():
    path = collect_expert.build_episode_video_path(
        Path("/tmp/videos"),
        3,
        scenario_id="S7_ego_merge_from_ramp",
    )

    assert path == Path("/tmp/videos/S7_ego_merge_from_ramp/episode_000003.mp4")


def test_parse_args_reads_max_episodes(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_expert.py",
            "--target-samples",
            "50",
            "--max-episodes",
            "3",
        ],
    )

    config = parse_args()

    assert config.target_samples == 50
    assert config.max_episodes == 3


def test_parse_args_reads_mode_generate_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_expert.py",
            "--mode-generate-enabled",
            "true",
            "--mode-generate-frame-limit",
            "12",
            "--mode-generate-include-invalid",
            "false",
        ],
    )

    config = parse_args()

    assert config.mode_generate_enabled is True
    assert config.mode_generate_frame_limit == 12
    assert config.mode_generate_include_invalid is False


def test_run_collection_stops_when_max_episodes_is_reached(monkeypatch, tmp_path):
    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)
            self.engine = type("Engine", (), {"global_config": {}})()

        def close(self):
            return None

    class DummyWriter:
        def add_samples(self, samples):
            return len(samples)

        def close(self):
            return 0

    sampled_specs = [
        _make_episode_spec(traffic_density=0.1, spawn_seed=1),
        _make_episode_spec(traffic_density=0.2, spawn_seed=2, scenario_id="S3_straight_following"),
    ]
    rollout_calls = []
    manifest_capture = {}

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: DummyWriter())
    monkeypatch.setattr(collect_expert, "EpisodeSpecSampler", _StubEpisodeSpecSampler(sampled_specs))
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index, idm_config=None, local_route="": (
            rollout_calls.append((episode_spawn_seed, episode_index)) or [{"frame": episode_index}],
            None,
            [],
            0,
        ),
    )
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(
        collect_expert,
        "build_episode_samples",
        lambda *args, **kwargs: [
            {
                "trajectory_mode": np.asarray(0, dtype=np.int64),
                "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32),
            }
        ],
    )
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(
        collect_expert,
        "write_manifest",
        lambda *args, **kwargs: manifest_capture.update(
            total_samples=kwargs.get("total_samples", args[3] if len(args) > 3 else None),
            total_episodes=kwargs.get("total_episodes", args[4] if len(args) > 4 else None),
        ),
    )

    run_collection(
        ExpertCollectorConfig(
            output_root=tmp_path,
            dataset_name="max_episode_dataset",
            target_samples=50,
            max_episodes=1,
        )
    )

    assert rollout_calls == [(1, 1)]
    assert manifest_capture["total_episodes"] == 1
    assert manifest_capture["total_samples"] == 1


def test_run_collection_does_not_write_episode_videos_when_disabled(monkeypatch, tmp_path):
    captured = {"calls": 0}

    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)
            self.engine = type("Engine", (), {"global_config": {}})()

        def close(self):
            return None

    class DummyWriter:
        def add_samples(self, samples):
            return len(samples)

        def close(self):
            return 0

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: DummyWriter())
    monkeypatch.setattr(
        collect_expert,
        "EpisodeSpecSampler",
        _StubEpisodeSpecSampler([_make_episode_spec(traffic_density=0.1, spawn_seed=1)]),
    )
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index, idm_config=None, local_route="": ([{"frame": 1}], None, [np.zeros((2, 2, 3), dtype=np.uint8)], 0),
    )
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(
        collect_expert,
        "build_episode_samples",
        lambda *args, **kwargs: [
            {
                "trajectory_mode": np.asarray(0, dtype=np.int64),
                "trajectory_mean_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(0.0, dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(0.0, dtype=np.float32),
                "trajectory_correction_strength": np.asarray(0.0, dtype=np.float32),
            }
        ],
    )
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(collect_expert, "write_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        collect_expert,
        "write_episode_video",
        lambda video_path, frames, fps: captured.__setitem__("calls", captured["calls"] + 1),
    )

    run_collection(
        ExpertCollectorConfig(
            output_root=tmp_path,
            dataset_name="video_dataset",
            target_samples=1,
            save_videos=False,
        )
    )

    assert captured["calls"] == 0
