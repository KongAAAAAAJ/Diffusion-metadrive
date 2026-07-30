from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from expert_dataset.collect_joint_bev import (
    AgentRole,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    JointBEVSample,
    JointEpisodeRollout,
)
from expert_dataset import run_joint_bev_collection as runner
from expert_dataset.joint_bev_storage import PACKED_BEV_FIELD
from expert_dataset.semantic_bev_codec import PACKED_BEV_SHAPE, unpack_semantic_bev
from models.bev_planner.mode_contract import ModeIndex


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "joint.yaml"
    path.write_text(
        f"""
dataset:
  name: joint_test
  output_root: {tmp_path / "datasets"}
split:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 17
collection:
  target_joint_steps: 10
  start_seed: 23
  max_episodes: 4
  max_episode_steps: 20
  resume: false
  scenario_weights:
    S5_hard_brake_lead: 1.0
    S6_background_merge_in: 2.0
  traffic_density_min: 0.0
  traffic_density_max: 0.03
env_config:
  num_agents: 3
  horizon: 20
  observation_mode: bev_gt
  use_render: false
  image_observation: false
  sensors: {{}}
  target_speed_km_h: 30.0
{extra}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _sample() -> JointBEVSample:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    values["mode_valid_mask"][:, ModeIndex.STOP] = True
    values["gt_mode"][:] = int(ModeIndex.STOP)
    return JointBEVSample(**values)


def test_load_config_is_strict_and_fingerprint_excludes_run_target(
    tmp_path: Path,
) -> None:
    config = runner.load_run_config(_write_config(tmp_path))
    assert config.dataset_root == tmp_path / "datasets/joint_test"
    assert config.target_joint_steps == 10
    assert config.env_config["sensors"] == {}
    assert config.env_config["observation_mode"] == "bev_gt"
    changed_target = runner.replace(config, target_joint_steps=100, max_episodes=20, resume=True)
    assert changed_target.immutable_fingerprint() == config.immutable_fingerprint()

    with pytest.raises(ValueError, match="unknown dataset config sections"):
        runner.load_run_config(_write_config(tmp_path, "unexpected: {}\n"))


def test_config_rejects_sensor_or_non_boolean_resume(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "sensors: {}", "sensors: {range_sensor: enabled}"
    )
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="empty sensors"):
        runner.load_run_config(config_path)

    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "resume: false", "resume: 'false'"
    )
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="resume must be a boolean"):
        runner.load_run_config(config_path)


def test_episode_sampling_is_deterministic_by_global_episode_index(
    tmp_path: Path,
) -> None:
    config = runner.load_run_config(_write_config(tmp_path))
    first = [runner.sample_episode_spec(config, index) for index in range(10)]
    second = [runner.sample_episode_spec(config, index) for index in range(10)]
    assert first == second
    assert {spec.scenario_id for spec in first} <= set(config.scenario_weights)
    assert all(spec.local_route for spec in first)
    assert all(0.0 <= spec.traffic_density <= 0.03 for spec in first)


def test_diagnostic64_config_and_event_sampling_are_strict() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_diagnostic64.yaml")
    )
    assert sum(config.diagnostic_scenario_quotas.values()) == 64
    assert tuple(config.diagnostic_scenario_quotas.values()) == (
        13,
        13,
        13,
        13,
        12,
    )
    samples = tuple(_sample() for _ in range(4))
    rollout = JointEpisodeRollout(
        samples=samples,
        simulator_steps=50,
        rejected_joint_steps=0,
        failure_reason=None,
        terminated=False,
        truncated=True,
        sample_step_indices=(10, 20, 30, 40),
        scenario_summary={
            "scenario_triggered": True,
            "scenario_trigger_step": 20,
            "scenario_realized": True,
            "scenario_realized_step": 20,
        },
    )
    selected, steps, trigger = runner._select_diagnostic_samples(
        rollout,
        decision_dt_s=0.1,
        offsets_s=(0.0, 1.0),
        remaining=2,
    )
    assert len(selected) == 2
    assert steps == (20, 30)
    assert trigger == 20
    with pytest.raises(runner.JointCollectionError, match="never triggered"):
        runner._select_diagnostic_samples(
            runner.replace(
                rollout,
                scenario_summary={"scenario_triggered": False},
            ),
            decision_dt_s=0.1,
            offsets_s=(0.0, 1.0),
            remaining=2,
        )


def test_cli_overrides_only_operational_collection_limits(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = runner.parse_args(
        [
            "--config",
            str(config_path),
            "--dataset-root",
            str(tmp_path / "override"),
            "--target-joint-steps",
            "2",
            "--max-episodes",
            "1",
            "--max-episode-steps",
            "11",
            "--resume",
            "1",
        ]
    )
    assert config.dataset_root == (tmp_path / "override").resolve()
    assert config.target_joint_steps == 2
    assert config.max_episodes == 1
    assert config.max_episode_steps == 11
    assert config.resume is True


def test_run_collection_commits_joint_episode_with_real_store(
    tmp_path: Path, monkeypatch
) -> None:
    config = runner.load_run_config(_write_config(tmp_path))
    config = runner.replace(
        config,
        dataset_root=tmp_path / "output",
        target_joint_steps=1,
        max_episodes=1,
        max_episode_steps=11,
    )

    class _FakeEnv:
        def __init__(self, env_config):
            self.config = dict(env_config)
            self.platoon_config = type(
                "Config",
                (),
                {"traffic_density": 0.0, "initial_speed_km_h": 0.0},
            )()
            self.engine = type(
                "Engine",
                (),
                {
                    "global_config": {},
                    "spawn_manager": type(
                        "Spawn", (), {"set_episode_spawn_seed": lambda self, seed: None}
                    )(),
                },
            )()
            self.closed = False

        def set_runtime_scenario_route(self, scenario_id, local_route):
            self.config.update(
                {"scenario_id": scenario_id, "local_route": local_route}
            )

        def close(self):
            self.closed = True

    fake_env = _FakeEnv(config.env_config)
    monkeypatch.setattr(
        runner, "SensorlessJointBEVPlatoonEnv", lambda env_config: fake_env
    )
    monkeypatch.setattr(
        runner,
        "collect_joint_episode",
        lambda env, max_steps: JointEpisodeRollout(
            samples=(_sample(),),
            simulator_steps=max_steps,
            rejected_joint_steps=0,
            failure_reason=None,
            terminated=False,
            truncated=True,
        ),
    )

    summary = runner.run_collection(config)

    assert summary["stored_episodes"] == 1
    assert summary["total_joint_samples"] == 1
    assert fake_env.closed
    split = next(
        name for name, values in summary["splits"].items() if values["episodes"] == 1
    )
    episode_dir = next((config.dataset_root / split / "episodes").iterdir())
    packed = np.load(
        episode_dir / f"{PACKED_BEV_FIELD}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    assert isinstance(packed, np.memmap)
    assert packed.shape == (1, 3, *PACKED_BEV_SHAPE)
    np.testing.assert_array_equal(unpack_semantic_bev(packed[0]), _sample().bev)
    assert not (episode_dir / "bev.npy").exists()
