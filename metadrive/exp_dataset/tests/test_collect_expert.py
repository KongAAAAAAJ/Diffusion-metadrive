from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock
from enum import Enum

import numpy as np


def _load_module():
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

    utils_module = ModuleType("metadrive.utils")
    utils_module.Config = _StubConfig

    dataset_module = ModuleType("metadrive.exp_dataset.metadrive_dataset")
    dataset_module.split_shards = lambda **kwargs: {"train": [], "val": [], "test": []}

    trajectory_filter_module = ModuleType("metadrive.exp_dataset.trajectory_filter")
    trajectory_filter_module.OutOfRoadByReferenceLaneRule = _StubOutOfRoadByReferenceLaneRule
    trajectory_filter_module.TrajectoryFilterPipeline = _StubTrajectoryFilterPipeline

    injected_module_names = [
        expert_idm_module,
        base_vehicle_module,
        env_module,
        ppo_module,
        trajectory_module,
        idm_module,
        transfuser_features_module,
        utils_module,
        dataset_module,
        trajectory_filter_module,
    ]
    for module in injected_module_names:
        sys.modules[module.__name__] = module

    module_path = Path(__file__).resolve().parents[1] / "collect_expert.py"
    spec = importlib.util.spec_from_file_location("collect_expert", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for injected in injected_module_names:
        sys.modules.pop(injected.__name__, None)
    return module


collect_expert = _load_module()
ExpertCollectorConfig = collect_expert.ExpertCollectorConfig
ShardWriter = collect_expert.ShardWriter
build_trajectory_mode_summary = collect_expert.build_trajectory_mode_summary
detect_existing_state = collect_expert.detect_existing_state
parse_args = collect_expert.parse_args
run_collection = collect_expert.run_collection
write_manifest = collect_expert.write_manifest


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


def test_build_episode_rngs_uses_independent_streams():
    config = ExpertCollectorConfig(start_seed=11, spawn_seed_offset=100000)

    traffic_rng, spawn_rng = collect_expert.build_episode_rngs(config)

    traffic_values = [
        float(traffic_rng.uniform(config.traffic_density_min, config.traffic_density_max))
        for _ in range(3)
    ]
    spawn_values = [int(spawn_rng.randint(0, 2**31 - 1)) for _ in range(3)]

    expected_traffic_rng = np.random.RandomState(config.start_seed)
    expected_spawn_rng = np.random.RandomState(config.start_seed + config.spawn_seed_offset)
    expected_traffic = [
        float(expected_traffic_rng.uniform(config.traffic_density_min, config.traffic_density_max))
        for _ in range(3)
    ]
    expected_spawn = [int(expected_spawn_rng.randint(0, 2**31 - 1)) for _ in range(3)]

    assert traffic_values == expected_traffic
    assert spawn_values == expected_spawn


def test_fast_forward_episode_rngs_advances_traffic_and_spawn_streams_independently():
    config = ExpertCollectorConfig(start_seed=5, spawn_seed_offset=100000)
    traffic_rng, spawn_rng = collect_expert.build_episode_rngs(config)

    collect_expert.fast_forward_episode_rngs(traffic_rng, spawn_rng, config, completed_episodes=2)

    next_density = float(traffic_rng.uniform(config.traffic_density_min, config.traffic_density_max))
    next_spawn_seed = int(collect_expert.sample_episode_spawn_seed(spawn_rng))

    expected_traffic_rng = np.random.RandomState(config.start_seed)
    expected_spawn_rng = np.random.RandomState(config.start_seed + config.spawn_seed_offset)
    for _ in range(2):
        expected_traffic_rng.uniform(config.traffic_density_min, config.traffic_density_max)
        expected_spawn_rng.randint(0, 2**31 - 1)

    expected_density = float(expected_traffic_rng.uniform(config.traffic_density_min, config.traffic_density_max))
    expected_spawn_seed = int(expected_spawn_rng.randint(0, 2**31 - 1))

    assert next_density == expected_density
    assert next_spawn_seed == expected_spawn_seed


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

    frames, map_geometry, video_frames = collect_expert.rollout_episode(
        DummyEnv(),
        ExpertCollectorConfig(expert_type="ppo", max_episode_steps=1),
        episode_spawn_seed=42,
        episode_index=1,
    )

    assert recorded["spawn_seed"] == 42
    assert frames == [{"frame": 1}]
    assert map_geometry is None
    assert video_frames == []


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

    frames, map_geometry, video_frames = collect_expert.rollout_episode(
        DummyEnv(),
        ExpertCollectorConfig(expert_type="ppo", max_episode_steps=1, save_videos=True),
        episode_spawn_seed=42,
        episode_index=1,
    )

    assert frames == [{"frame": 1}]
    assert map_geometry is None
    assert len(video_frames) == 1
    assert captured["count"] == 1


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

    frames, map_geometry, video_frames = collect_expert.rollout_episode(
        DummyEnv(),
        ExpertCollectorConfig(expert_type="ppo", max_episode_steps=1, save_videos=False),
        episode_spawn_seed=42,
        episode_index=1,
    )

    assert frames == [{"frame": 1}]
    assert map_geometry is None
    assert video_frames == []
    assert captured["count"] == 0


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


def test_run_collection_resume_fast_forwards_rng_and_reuses_existing_counts(monkeypatch, tmp_path):
    sampled_densities: list[float] = []
    sampled_spawn_seeds: list[int] = []

    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)
            self.engine = type(
                "Engine",
                (),
                {"spawn_manager": type("SpawnManager", (), {"set_episode_spawn_seed": lambda self, seed: sampled_spawn_seeds.append(int(seed))})()},
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

    def fake_sample_traffic_density(rng, config):
        value = float(rng.uniform(config.traffic_density_min, config.traffic_density_max))
        sampled_densities.append(value)
        return value

    def fake_rollout_episode(env, config, episode_spawn_seed, episode_index):
        sampled_spawn_seeds.append(int(episode_spawn_seed))
        return ([{"frame": 1}], None, [])

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
                "collection_wall_time_sec": collection_wall_time_sec,
                "resume": resume,
            }
        )

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", DummyWriter)
    monkeypatch.setattr(collect_expert, "sample_traffic_density", fake_sample_traffic_density)
    monkeypatch.setattr(collect_expert, "rollout_episode", fake_rollout_episode)
    monkeypatch.setattr(collect_expert, "resolve_map_visualization_geometry", lambda **kwargs: None)
    monkeypatch.setattr(collect_expert, "build_episode_samples", fake_build_episode_samples)
    monkeypatch.setattr(collect_expert, "split_shards", lambda **kwargs: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(collect_expert, "write_manifest", fake_write_manifest)
    monkeypatch.setattr(
        collect_expert,
        "detect_existing_state",
        lambda shard_dir, report_dir: {
            "shard_count": 4,
            "total_samples": 5,
            "total_episodes": 2,
            "mode_counts": {"keep_lane": 5},
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

    expected_traffic_rng = np.random.RandomState(config.start_seed)
    expected_spawn_rng = np.random.RandomState(config.start_seed + config.spawn_seed_offset)
    for _ in range(2):
        expected_traffic_rng.uniform(config.traffic_density_min, config.traffic_density_max)
        expected_spawn_rng.randint(0, 2**31 - 1)
    expected_next_density = float(expected_traffic_rng.uniform(config.traffic_density_min, config.traffic_density_max))
    expected_next_spawn_seed = int(expected_spawn_rng.randint(0, 2**31 - 1))

    assert DummyWriter.last_instance is not None
    assert DummyWriter.last_instance.start_shard_index == 4
    assert sampled_densities == [expected_next_density]
    assert sampled_spawn_seeds == [expected_next_spawn_seed]
    assert written_manifest["total_samples"] == 6
    assert written_manifest["total_episodes"] == 3
    assert written_manifest["resume"] is True
    assert written_manifest["correction_summary"]["mode_counts"]["keep_lane"] == 6
    assert written_manifest["filter_summary"]["accepted"] == 1
    assert written_manifest["filter_summary"]["rejected"] == 0
    assert build_trajectory_mode_summary(written_manifest["correction_summary"]["mode_counts"], 6)["keep_lane"]["count"] == 6


def test_run_collection_filters_rejected_samples_before_writer(monkeypatch, tmp_path):
    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)

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
    monkeypatch.setattr(collect_expert, "sample_traffic_density", lambda rng, config: 0.1)
    monkeypatch.setattr(collect_expert, "sample_episode_spawn_seed", lambda rng: 1)
    monkeypatch.setattr(collect_expert, "rollout_episode", lambda env, config, episode_spawn_seed, episode_index: ([{"frame": 1}], None, []))
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
    monkeypatch.setattr(collect_expert, "sample_traffic_density", lambda rng, config: 0.1)
    monkeypatch.setattr(collect_expert, "sample_episode_spawn_seed", lambda rng: 1)
    monkeypatch.setattr(collect_expert, "rollout_episode", lambda env, config, episode_spawn_seed, episode_index: ([{"frame": 1}], None, []))
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
    monkeypatch.setattr(collect_expert, "sample_traffic_density", lambda rng, config: 0.1)
    monkeypatch.setattr(collect_expert, "sample_episode_spawn_seed", lambda rng: 1)
    monkeypatch.setattr(collect_expert, "rollout_episode", lambda env, config, episode_spawn_seed, episode_index: ([{"frame": 1}], None, []))
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

        def close(self):
            return None

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: type("W", (), {"add_samples": lambda self, samples: 0, "close": lambda self: 0})())
    monkeypatch.setattr(collect_expert, "sample_traffic_density", lambda rng, config: 0.1)
    monkeypatch.setattr(collect_expert, "sample_episode_spawn_seed", lambda rng: 1)
    monkeypatch.setattr(collect_expert, "rollout_episode", lambda env, config, episode_spawn_seed, episode_index: ([], None, []))
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

        def close(self):
            return None

    class DummyWriter:
        def add_samples(self, samples):
            return len(samples)

        def close(self):
            return 0

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: DummyWriter())
    monkeypatch.setattr(collect_expert, "sample_traffic_density", lambda rng, config: 0.1)
    monkeypatch.setattr(collect_expert, "sample_episode_spawn_seed", lambda rng: 1)
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index: ([{"frame": 1}], None, [np.zeros((2, 2, 3), dtype=np.uint8)]),
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
    assert str(video_path).endswith("reports/videos/episode_000001.mp4")
    assert frame_count == 1
    assert fps == 12


def test_run_collection_does_not_write_episode_videos_when_disabled(monkeypatch, tmp_path):
    captured = {"calls": 0}

    class DummyEnv:
        def __init__(self, config):
            self.config = dict(config)

        def close(self):
            return None

    class DummyWriter:
        def add_samples(self, samples):
            return len(samples)

        def close(self):
            return 0

    monkeypatch.setattr(collect_expert, "DatasetCollectEnv", DummyEnv)
    monkeypatch.setattr(collect_expert, "ShardWriter", lambda *args, **kwargs: DummyWriter())
    monkeypatch.setattr(collect_expert, "sample_traffic_density", lambda rng, config: 0.1)
    monkeypatch.setattr(collect_expert, "sample_episode_spawn_seed", lambda rng: 1)
    monkeypatch.setattr(
        collect_expert,
        "rollout_episode",
        lambda env, config, episode_spawn_seed, episode_index: ([{"frame": 1}], None, [np.zeros((2, 2, 3), dtype=np.uint8)]),
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
