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
    JointEpisodeSidecar,
)
from expert_dataset import run_joint_bev_collection as runner
from expert_dataset.joint_bev_storage import PACKED_BEV_FIELD
from expert_dataset.semantic_bev_codec import PACKED_BEV_SHAPE, unpack_semantic_bev
from models.bev_planner.mode_contract import ModeIndex
from expert_dataset.riskentry_sidecar_adapter import (
    SidecarActorRecord,
    SidecarActorSnapshot,
    SidecarFrameCapture,
    SidecarRawEvent,
    SidecarRawFrame,
)
from expert_dataset.verify_riskentry_sidecar import verify_riskentry_sidecar_dataset
from expert_dataset.verify_joint_risk_bundle import verify_joint_risk_bundle


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "joint.yaml"
    path.write_text(
        f"""
dataset:
  name: joint_test
  sidecar_name: riskentry_actor_sidecar
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


def _sidecar(
    simulator_steps: int, *, terminal_event: str | None = None
) -> JointEpisodeSidecar:
    records = tuple(
        SidecarActorRecord(
            actor_index=index,
            actor_id=f"P{index}",
            source_object_id=f"agent{index}",
            actor_type="platoon",
            platoon_role=("leader", "middle", "rear")[index],
            first_seen_step=0,
            length_m=5.74,
            width_m=2.3,
        )
        for index in range(3)
    )
    captures = []
    for step in range(simulator_steps + 1):
        actors = tuple(
            SidecarActorSnapshot(
                actor_id=f"P{index}",
                source_object_id=f"agent{index}",
                actor_type="platoon",
                world_x_m=float(step + index * 10),
                world_y_m=0.0,
                heading_rad=0.0,
                velocity_x_mps=10.0,
                velocity_y_mps=0.0,
                length_m=5.74,
                width_m=2.3,
                acceleration_valid=step > 0,
                yaw_rate_valid=step > 0,
            )
            for index in range(3)
        )
        captures.append(
            SidecarFrameCapture(
                frame=SidecarRawFrame(step, step * 0.1, actors),
                events=(
                    ()
                    if terminal_event is None or step != simulator_steps
                    else (
                        SidecarRawEvent(
                            event_type=terminal_event,
                            step_index=step,
                            timestamp_s=step * 0.1,
                            actor_ids=("P0",),
                            terminal=True,
                        ),
                    )
                ),
            )
        )
    return JointEpisodeSidecar(
        captures=tuple(captures),
        actor_records=records,
        lane_records=(),
        key_actor_ids={},
    )


def test_load_config_is_strict_and_fingerprint_excludes_run_target(
    tmp_path: Path,
) -> None:
    config = runner.load_run_config(_write_config(tmp_path))
    assert config.dataset_root == tmp_path / "datasets/joint_test"
    assert config.sidecar_root == tmp_path / "datasets/riskentry_actor_sidecar"
    assert config.bundle_root == tmp_path / "datasets"
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


def test_formal_pilot_config_and_quota_clipping_are_strict() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_shared_formal_pilot15k_round13_97f.yaml")
    )
    assert config.resume is True
    assert config.target_joint_steps == 15_000
    assert tuple(config.formal_scenario_quotas.values()) == (3000,) * 5
    assert config.diagnostic_scenario_quotas is None
    with pytest.raises(ValueError, match="requires resume=true"):
        runner.replace(config, resume=False)

    rollout = JointEpisodeRollout(
        samples=tuple(_sample() for _ in range(5)),
        simulator_steps=8,
        rejected_joint_steps=0,
        failure_reason=None,
        terminated=False,
        truncated=True,
        sample_step_indices=(1, 2, 3, 4, 5),
    )
    selected, steps = runner._select_formal_quota_samples(rollout, remaining=3)
    assert len(selected) == 3
    assert steps == (1, 3, 5)


def test_resume_true_creates_fresh_bundle_but_rejects_partial_initialization(
    tmp_path: Path,
) -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_shared_formal_smoke_round13_97f.yaml")
    )
    config = runner.replace(
        config,
        bundle_root=tmp_path / "bundle",
        dataset_root=tmp_path / "bundle/platoon_joint_bev",
        sidecar_root=tmp_path / "bundle/riskentry_actor_sidecar",
    )
    assert runner._effective_resume_mode(config) is False

    config.dataset_root.mkdir(parents=True)
    (config.dataset_root / runner.JointBEVDatasetStore.CONTRACT_FILE).write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        runner.JointRiskBundleStorageError, match="partial component initialization"
    ):
        runner._effective_resume_mode(config)

    config.sidecar_root.mkdir(parents=True)
    (
        config.sidecar_root / runner.RiskEntrySidecarDatasetStore.CONTRACT_FILE
    ).write_text("{}\n", encoding="utf-8")
    (config.bundle_root / runner.JointRiskBundleIndex.MANIFEST_FILE).write_text(
        "{}\n", encoding="utf-8"
    )
    assert runner._effective_resume_mode(config) is True


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
        bundle_root=tmp_path / "bundle",
        dataset_root=tmp_path / "bundle/output",
        sidecar_root=tmp_path / "bundle/riskentry_actor_sidecar",
        target_joint_steps=1,
        max_episodes=1,
        max_episode_steps=11,
    )

    spawn_seeds = []

    class _Spawn:
        def set_episode_spawn_seed(self, seed):
            spawn_seeds.append(seed)

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
                    "spawn_manager": _Spawn(),
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
    reset_seeds = []

    def _fake_collect(env, *, max_steps, reset_seed=None):
        reset_seeds.append(reset_seed)
        return JointEpisodeRollout(
            samples=(_sample(),),
            simulator_steps=max_steps,
            rejected_joint_steps=0,
            failure_reason=None,
            terminated=False,
            truncated=True,
            sample_step_indices=(0,),
            sidecar=_sidecar(max_steps),
        )

    monkeypatch.setattr(
        runner,
        "collect_joint_episode",
        _fake_collect,
    )

    summary = runner.run_collection(config)

    assert summary["base"]["stored_episodes"] == 1
    assert summary["base"]["total_joint_samples"] == 1
    assert summary["bundle_attempts"] == 1
    assert reset_seeds == spawn_seeds
    assert len(reset_seeds) == 1
    assert fake_env.closed
    split = next(
        name
        for name, values in summary["base"]["splits"].items()
        if values["episodes"] == 1
    )
    assert summary["sidecar"]["splits"][split]["episodes"] == 1
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
    sidecar_report = verify_riskentry_sidecar_dataset(config.sidecar_root)
    assert sidecar_report["episodes"] == 1
    assert sidecar_report["raw_steps"] == 12
    assert (config.bundle_root / "bundle_episode_index.jsonl").is_file()
    bundle_report = verify_joint_risk_bundle(config.bundle_root)
    assert bundle_report["committed_base_episodes"] == 1
    assert bundle_report["committed_sidecar_episodes"] == 1


def test_dangerous_episode_is_sidecar_only_with_empty_base_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    config = runner.load_run_config(_write_config(tmp_path))
    config = runner.replace(
        config,
        bundle_root=tmp_path / "bundle",
        dataset_root=tmp_path / "bundle/platoon_joint_bev",
        sidecar_root=tmp_path / "bundle/riskentry_actor_sidecar",
        target_joint_steps=1,
        max_episodes=1,
        max_episode_steps=3,
    )

    class _Spawn:
        def set_episode_spawn_seed(self, seed):
            del seed

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
                {"global_config": {}, "spawn_manager": _Spawn()},
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
        lambda env, *, max_steps, reset_seed=None: JointEpisodeRollout(
            samples=(_sample(),),
            simulator_steps=max_steps,
            rejected_joint_steps=0,
            failure_reason="crash_vehicle:agent0",
            terminated=True,
            truncated=False,
            sample_step_indices=(1,),
            sidecar=_sidecar(max_steps, terminal_event="collision_vehicle"),
        ),
    )

    summary = runner.run_collection(config)

    assert summary["base"]["stored_episodes"] == 0
    assert summary["base"]["rejected_episodes"] == 1
    assert summary["bundle_attempts"] == 1
    report = verify_joint_risk_bundle(config.bundle_root)
    assert report["committed_base_episodes"] == 0
    assert report["committed_sidecar_episodes"] == 1
    assert report["sidecar_only_episodes"] == 1
    sidecar_episode = next(
        (config.sidecar_root / split / "episodes" / "episode_00000000")
        for split in ("train", "val", "test")
        if (config.sidecar_root / split / "episodes" / "episode_00000000").is_dir()
    )
    mapping = np.load(
        sidecar_episode / "base_sample_step_index.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    assert mapping.shape == (0,)
    assert fake_env.closed


@pytest.mark.parametrize("base_committed", [False, True])
def test_pending_prepared_sidecar_recovers_without_base_only_window(
    tmp_path: Path, base_committed: bool
) -> None:
    config = runner.load_run_config(_write_config(tmp_path))
    fingerprint = config.immutable_fingerprint()
    split = runner.EpisodeSplitConfig(
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, seed=17
    )
    spec = runner.JointEpisodeSpec(
        scenario_id="S5_hard_brake_lead",
        local_route="R1_entry_straight",
        spawn_seed=17,
        traffic_density=0.0,
        initial_speed_km_h=24.0,
    )
    rollout = JointEpisodeRollout(
        samples=(_sample(),),
        simulator_steps=2,
        rejected_joint_steps=0,
        failure_reason=None,
        terminated=False,
        truncated=False,
        sample_step_indices=(1,),
        scenario_summary={},
        sidecar=_sidecar(2),
    )
    base_root = tmp_path / "bundle/platoon_joint_bev"
    sidecar_root = tmp_path / "bundle/riskentry_actor_sidecar"
    base_store = runner.JointBEVDatasetStore(
        base_root,
        split_config=split,
        dataset_fingerprint=fingerprint,
        resume=False,
    )
    sidecar_store = runner.RiskEntrySidecarDatasetStore(
        sidecar_root,
        base_dataset_fingerprint=fingerprint,
        resume=False,
    )
    bundle = runner.JointRiskBundleIndex(
        tmp_path / "bundle",
        base_directory=base_root.name,
        sidecar_directory=sidecar_root.name,
        base_dataset_fingerprint=fingerprint,
        sidecar_dataset_fingerprint=sidecar_store.dataset_fingerprint,
        scenario_contract_sha256=runner.primary_scenario_contract()["sha256"],
        split_seed=17,
        resume=False,
    )
    bundle.begin_attempt(
        runner.BundleEpisodeAttempt(0, "train", spec.scenario_id, spec.local_route, 17)
    )
    runner._prepare_rollout_sidecar(
        sidecar_store,
        start=runner._sidecar_start(
            episode_index=0,
            split="train",
            spec=spec,
            rollout=rollout,
            base_dataset_fingerprint=fingerprint,
            decision_dt_s=0.1,
        ),
        rollout=rollout,
        base_sample_step_indices=(1,),
    )
    if base_committed:
        base_store.commit_episode(
            0,
            rollout.samples,
            {
                "scenario_id": spec.scenario_id,
                "local_route": spec.local_route,
                "spawn_seed": spec.spawn_seed,
                "selected_sample_steps": [1],
                "decision_dt_s": 0.1,
                "raw_timeline_length": 3,
                "sidecar_dataset_fingerprint": sidecar_store.dataset_fingerprint,
                "scenario_contract_sha256": runner.primary_scenario_contract()["sha256"],
            },
        )
    base_store.close()
    sidecar_store.close()
    bundle.close()

    with runner.JointBEVDatasetStore(
        base_root,
        split_config=split,
        dataset_fingerprint=fingerprint,
        resume=True,
    ) as resumed_base, runner.RiskEntrySidecarDatasetStore(
        sidecar_root,
        base_dataset_fingerprint=fingerprint,
        resume=True,
    ) as resumed_sidecar, runner.JointRiskBundleIndex(
        tmp_path / "bundle",
        base_directory=base_root.name,
        sidecar_directory=sidecar_root.name,
        base_dataset_fingerprint=fingerprint,
        sidecar_dataset_fingerprint=resumed_sidecar.dataset_fingerprint,
        scenario_contract_sha256=runner.primary_scenario_contract()["sha256"],
        split_seed=17,
        resume=True,
    ) as resumed_bundle:
        runner._recover_pending_bundle_attempt(
            resumed_bundle, resumed_base, resumed_sidecar
        )
        runner._validate_bundle_resume_state(
            resumed_bundle, resumed_base, resumed_sidecar
        )
        row = resumed_bundle.rows[0]
        assert row.sidecar_status == "committed"
        assert row.base_status == ("committed" if base_committed else "rejected")
        assert row.base_samples == (1 if base_committed else 0)

    report = verify_joint_risk_bundle(tmp_path / "bundle")
    assert report["committed_base_episodes"] == int(base_committed)
    assert report["committed_sidecar_episodes"] == 1
    assert report["sidecar_only_episodes"] == int(not base_committed)
