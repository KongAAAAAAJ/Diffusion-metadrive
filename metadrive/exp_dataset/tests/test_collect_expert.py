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

    stub_module = ModuleType("metadrive.exp_dataset.hierarchical_expert")
    stub_module.HierarchicalExpertPolicy = type("HierarchicalExpertPolicy", (), {})
    stub_module.HierarchicalExpertIDMPolicy = type("HierarchicalExpertIDMPolicy", (), {})
    driving_style_module = ModuleType("metadrive.exp_dataset.hierarchical_expert.driving_style")
    driving_style_module.DrivingStyleProfile = type("DrivingStyleProfile", (), {})

    base_vehicle_module = ModuleType("metadrive.component.vehicle.base_vehicle")
    base_vehicle_module.BaseVehicle = type("BaseVehicle", (), {})

    env_module = ModuleType("metadrive.envs.diffusion_envs.base_multi_env")
    env_module.DatasetCollectEnv = type("DatasetCollectEnv", (), {})

    ppo_module = ModuleType("metadrive.examples.ppo_expert")
    ppo_module.expert = lambda *args, **kwargs: np.zeros(2, dtype=np.float32)

    trajectory_module = ModuleType("metadrive.exp_dataset.trajectory_correction")
    trajectory_module.TrajectoryCorrectionContext = type("TrajectoryCorrectionContext", (), {})
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

    injected_module_names = [
        stub_module,
        driving_style_module,
        base_vehicle_module,
        env_module,
        ppo_module,
        trajectory_module,
        idm_module,
        transfuser_features_module,
        utils_module,
        dataset_module,
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


def test_build_hierarchical_expert_policy_uses_explicit_default_style():
    vehicle = object()
    created = {}

    original_expert = getattr(collect_expert, "Expert", None)
    original_driving_style = getattr(collect_expert, "DrivingStyleProfile", None)
    try:
        collect_expert.Expert = Mock(
            side_effect=lambda vehicle, random_seed, style_profile: (
                created.update(
                    {
                        "vehicle": vehicle,
                        "random_seed": random_seed,
                        "style_profile": style_profile,
                    }
                )
                or "policy"
            )
        )
        collect_expert.DrivingStyleProfile = Mock(side_effect=original_driving_style)

        policy = collect_expert.build_hierarchical_expert_policy(vehicle, random_seed=7)

        assert policy == "policy"
        assert created["vehicle"] is vehicle
        assert created["random_seed"] == 7
        assert isinstance(created["style_profile"], original_driving_style)
        collect_expert.DrivingStyleProfile.assert_called_once_with()
    finally:
        if original_expert is not None:
            collect_expert.Expert = original_expert
        if original_driving_style is not None:
            collect_expert.DrivingStyleProfile = original_driving_style


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
        collection_wall_time_sec=1.5,
        resume=True,
    )

    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["collected_samples"] == 16
    assert manifest["episodes"] == 4
    assert len(manifest["resume_history"]) == 2
    assert manifest["resume_history"][-1]["added_samples"] == 6


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
    assert build_trajectory_mode_summary(written_manifest["correction_summary"]["mode_counts"], 6)["keep_lane"]["count"] == 6


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
