from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def _targeted_requirements(
    *,
    split_quotas: dict[str, int] | None = None,
    samples: int = 190,
    background_quotas: dict[int, int] | None = None,
) -> runner.TargetedSupplementRequirements:
    return runner.TargetedSupplementRequirements(
        scenario_id="S5_hard_brake_lead",
        behavior_category="temporary_formation_release_and_recovery",
        rule_maker_profile_id="balanced",
        accepted_episode_quotas=(
            {"train": 16, "val": 2, "test": 2}
            if split_quotas is None
            else split_quotas
        ),
        required_samples_per_episode=samples,
        max_simulator_attempts=200,
        initial_feasibility_attempts=10,
        minimum_lateral_range_m=2.5,
        maximum_return_error_m=0.5,
        accepted_background_count_quotas=(
            {3: 5, 4: 5, 5: 5, 6: 5}
            if background_quotas is None
            else background_quotas
        ),
        bootstrap_spawn_seeds=(23,),
    )


def _targeted_rollout(*, unsafe: bool = False) -> JointEpisodeRollout:
    samples = []
    y = np.zeros(190, dtype=np.float32)
    y[20:50] = np.linspace(0.1, 3.0, 30)
    y[50:100] = 3.0
    y[100:130] = np.linspace(2.9, 0.0, 30)
    for step in range(190):
        modes = np.full(3, int(ModeIndex.STOP), dtype=np.int64)
        if 20 <= step < 50:
            modes[:] = (
                int(ModeIndex.LEFT_LOW),
                int(ModeIndex.RIGHT_LOW),
                int(ModeIndex.LEFT_LOW),
            )
        elif 100 <= step < 130:
            modes[:] = (
                int(ModeIndex.RIGHT_LOW),
                int(ModeIndex.LEFT_LOW),
                int(ModeIndex.RIGHT_LOW),
            )
        pose = np.zeros((3, 3), dtype=np.float32)
        pose[:, 1] = (y[step], -y[step], y[step])
        samples.append(SimpleNamespace(gt_mode=modes, ego_pose_global=pose))
    return JointEpisodeRollout(
        samples=tuple(samples),
        simulator_steps=200,
        rejected_joint_steps=0,
        failure_reason=None,
        terminated=False,
        truncated=True,
        sample_step_indices=tuple(range(10, 200)),
        scenario_summary={
            "resolved_scenario_parameters": {
                "incidental_background_actor_count": 4,
            },
            "conflict_evidence": {
                "incidental_background_realized_count": 4,
                "observed_behavior_class": (
                    "temporary_formation_release_and_recovery"
                ),
                "mixed_direction_lane_change_completed": True,
                "reassembly_completed": True,
                "reassembly_to_initial_lane_completed": True,
                "hazard_cleared": True,
                "formation_recovered_after_hazard": True,
            },
        },
        sidecar=_sidecar(
            200,
            terminal_event="collision_vehicle" if unsafe else None,
        ),
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


def test_candidate_v3_config_binds_current_contract_and_s9_horizon() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v3_diagnostic10k_20260814.yaml")
    )
    assert config.scenario_contract_id == "candidate_v3"
    assert config.scenario_contract()["format"] == "bev_primary_s5_s9_contract_v2"
    assert config.episode_step_limit("S7_ego_merge_from_ramp") == 200
    assert config.episode_step_limit("S9_narrow_channel_negotiation") == 800


def test_targeted_supplement_config_freezes_all_production_gates() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v3_s5_release20.yaml")
    )
    requirements = config.targeted_supplement
    assert requirements is not None
    assert config.formal_scenario_quotas is None
    assert config.diagnostic_scenario_quotas is None
    assert requirements.rule_maker_profile_id == "balanced"
    assert requirements.accepted_episode_quotas == {
        "train": 16,
        "val": 2,
        "test": 2,
    }
    assert requirements.accepted_background_count_quotas == {
        3: 5,
        4: 5,
        5: 5,
        6: 5,
    }
    assert requirements.required_samples_per_episode == 190
    assert requirements.initial_feasibility_attempts == 10
    assert requirements.max_simulator_attempts == 200
    assert requirements.parallel_workers == 4
    assert requirements.bootstrap_spawn_seeds == ()
    assert requirements.initial_feasibility_min_accepted_episodes == 1
    assert config.immutable_fingerprint() == (
        "627b2a52b88842aaa4c27f6fa91e60255b8561b6271d30a08702cccf8415f9de"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.replace(
            config,
            formal_scenario_quotas={
                name: 760 for name, _ in runner.PRIMARY_S5_S9_SCENARIOS
            },
        )
    with pytest.raises(ValueError, match="balanced"):
        runner.replace(
            config,
            targeted_supplement=runner.replace(
                requirements, rule_maker_profile_id="brake_first"
            ),
        )


def test_targeted_physical_validator_requires_two_runs_recovery_and_safety() -> None:
    requirements = _targeted_requirements()
    evidence = runner._validate_targeted_rollout(
        _targeted_rollout(), requirements
    )
    assert evidence["lateral_mode_runs_by_role"] == [2, 2, 2]
    assert evidence["behavior_category"] == (
        "temporary_formation_release_and_recovery"
    )
    with pytest.raises(runner.JointCollectionError) as exc_info:
        runner._validate_targeted_rollout(
            _targeted_rollout(unsafe=True), requirements
        )
    assert exc_info.value.reason_code == "targeted_safety_rejected"


def test_targeted_initial_gate_counts_only_real_attempts_and_unlocks_on_success() -> None:
    requirements = _targeted_requirements()
    assert runner._targeted_hard_stop_reason(
        simulator_attempts=9,
        accepted_episodes=0,
        requirements=requirements,
    ) is None
    assert runner._targeted_hard_stop_reason(
        simulator_attempts=10,
        accepted_episodes=0,
        requirements=requirements,
    ) == "targeted_initial_feasibility_failed"
    assert runner._targeted_hard_stop_reason(
        simulator_attempts=10,
        accepted_episodes=1,
        requirements=requirements,
    ) is None
    assert runner._targeted_hard_stop_reason(
        simulator_attempts=200,
        accepted_episodes=19,
        requirements=requirements,
    ) == "targeted_max_simulator_attempts_exhausted"


def test_candidate_v4_config_freezes_eight_of_ten_gate_and_seed_window() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v4_s5_release20.yaml")
    )
    requirements = config.targeted_supplement
    assert requirements is not None
    assert config.scenario_contract_id == "candidate_v4"
    assert requirements.initial_feasibility_min_accepted_episodes == 8
    assert requirements.parallel_workers == 4
    assert requirements.bootstrap_spawn_seeds == tuple(range(82017, 82027))
    assert sum(seed % 10 < 8 for seed in requirements.bootstrap_spawn_seeds) == 8
    assert runner._targeted_hard_stop_reason(
        simulator_attempts=10,
        accepted_episodes=7,
        requirements=requirements,
    ) == "targeted_initial_feasibility_failed"
    assert runner._targeted_hard_stop_reason(
        simulator_attempts=10,
        accepted_episodes=8,
        requirements=requirements,
    ) is None

    rows = [
        SimpleNamespace(
            outcome="accepted" if seed % 10 < 8 else "sidecar_only",
            spawn_seed=seed,
            base_status="committed" if seed % 10 < 8 else "rejected",
        )
        for seed in requirements.bootstrap_spawn_seeds
    ]
    status = runner._targeted_initial_gate_status(
        bundle=SimpleNamespace(rows=rows),
        requirements=requirements,
        scenario_contract_id="candidate_v4",
    )
    assert status["passed"] is True
    rows[0].base_status = "rejected"
    rows[1].base_status = "committed"
    status = runner._targeted_initial_gate_status(
        bundle=SimpleNamespace(rows=rows),
        requirements=requirements,
        scenario_contract_id="candidate_v4",
    )
    assert status["accepted_episodes"] == 8
    assert status["passed"] is False


def test_targeted_parallel_batch_never_crosses_the_first_ten_gate() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v3_s5_release20.yaml")
    )
    requirements = config.targeted_supplement
    progress = runner._empty_targeted_progress(requirements)
    bundle = SimpleNamespace(rows=[])
    specs, entries = runner._plan_targeted_parallel_batch(
        config,
        start_episode_index=0,
        progress=progress,
        bundle=bundle,
        simulator_attempts=0,
    )
    assert len(specs) == len(entries) == 4
    assert len({spec.spawn_seed for spec in specs}) == 4
    specs, _ = runner._plan_targeted_parallel_batch(
        config,
        start_episode_index=8,
        progress=progress,
        bundle=bundle,
        simulator_attempts=8,
    )
    assert len(specs) == 2
    progress["accepted_episodes"] = 1
    specs, _ = runner._plan_targeted_parallel_batch(
        config,
        start_episode_index=8,
        progress=progress,
        bundle=bundle,
        simulator_attempts=8,
    )
    assert len(specs) == 2
    specs, _ = runner._plan_targeted_parallel_batch(
        config,
        start_episode_index=10,
        progress=progress,
        bundle=bundle,
        simulator_attempts=10,
    )
    assert len(specs) == 4


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


def test_candidate_v3_formal50k_freezes_diversity_contract() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v3_formal50k.yaml")
    )
    requirements = config.formal_diversity
    assert config.target_joint_steps == 50_000
    assert tuple(config.formal_scenario_quotas.values()) == (10_000,) * 5
    assert requirements is not None
    assert requirements.min_episodes_per_scenario == 50
    assert requirements.max_samples_per_episode == 200
    assert requirements.require_all_scenarios_in_each_split is True
    assert requirements.required_spawn_seeds == (17, 23, 31, 47, 59)
    assert requirements.required_incidental_background_actor_counts == (
        3,
        4,
        5,
        6,
    )
    assert set(requirements.required_behavior_categories) == set(
        config.formal_scenario_quotas
    )
    assert requirements.behavior_rule_maker_profiles == {
        "S5_hard_brake_lead": {
            "keep_emergency_braking": "brake_first",
            "temporary_formation_release_and_recovery": "balanced",
        }
    }
    assert config.episode_step_limit("S9_narrow_channel_negotiation") == 800

    forced = runner.sample_episode_spec_for_scenario(
        config,
        0,
        "S9_narrow_channel_negotiation",
        spawn_seed=17,
    )
    assert forced.spawn_seed == 17


def test_formal_diversity_capacity_reserves_the_last_sample() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v3_formal50k.yaml")
    )
    requirements = config.formal_diversity
    scenario_id = "S5_hard_brake_lead"
    row = {
        "joint_samples": 9_800,
        "episodes": 49,
        "spawn_seeds": set(requirements.required_spawn_seeds),
        "splits": set(runner.FORMAL_SPLITS),
        "incidental_background_actor_counts": set(
            requirements.required_incidental_background_actor_counts
        ),
        "behavior_categories": {"keep_emergency_braking"},
    }
    capacity = runner._formal_sample_capacity(
        scenario_id=scenario_id,
        remaining=200,
        row=row,
        requirements=requirements,
        split="train",
        spawn_seed=1001,
        coverage={
            "incidental_background_actor_count": 3,
            "behavior_category": "keep_emergency_braking",
        },
    )
    assert capacity == 199
    row["behavior_categories"].add(
        "temporary_formation_release_and_recovery"
    )
    assert runner._formal_sample_capacity(
        scenario_id=scenario_id,
        remaining=200,
        row=row,
        requirements=requirements,
        split="train",
        spawn_seed=1001,
        coverage={
            "incidental_background_actor_count": 3,
            "behavior_category": "keep_emergency_braking",
        },
    ) == 200


def test_formal_rule_profile_targets_the_first_missing_behavior() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v3_formal50k.yaml")
    )
    requirements = config.formal_diversity
    row = {"behavior_categories": set()}
    assert runner._formal_rule_maker_profile(
        scenario_id="S5_hard_brake_lead",
        row=row,
        requirements=requirements,
    ) == "brake_first"
    row["behavior_categories"].add("keep_emergency_braking")
    assert runner._formal_rule_maker_profile(
        scenario_id="S5_hard_brake_lead",
        row=row,
        requirements=requirements,
    ) == "balanced"
    assert runner._formal_rule_maker_profile(
        scenario_id="S7_ego_merge_from_ramp",
        row={"behavior_categories": set()},
        requirements=requirements,
    ) is None


def test_formal_coverage_requires_realized_background_and_behavior() -> None:
    summary = {
        "resolved_scenario_parameters": {
            "incidental_background_actor_count": 4,
        },
        "conflict_evidence": {
            "incidental_background_realized_count": 4,
            "observed_behavior_class": "keep_emergency_braking",
        },
    }
    assert runner._formal_episode_coverage(
        "S5_hard_brake_lead", summary
    ) == {
        "incidental_background_actor_count": 4,
        "behavior_category": "keep_emergency_braking",
    }
    summary["conflict_evidence"]["incidental_background_realized_count"] = 3
    with pytest.raises(runner.JointCollectionError) as exc_info:
        runner._formal_episode_coverage("S5_hard_brake_lead", summary)
    assert exc_info.value.reason_code == "formal_pilot_coverage_metadata_missing"


def test_formal_completion_rejects_missing_split_and_accepts_full_coverage() -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v3_formal50k.yaml")
    )
    requirements = config.formal_diversity
    progress = runner._empty_formal_progress(config.formal_scenario_quotas)
    for scenario_id, row in progress.items():
        row["joint_samples"] = 10_000
        row["episodes"] = 50
        row["spawn_seeds"].update(requirements.required_spawn_seeds)
        row["spawn_seeds"].update(range(1000, 1045))
        row["splits"].update(runner.FORMAL_SPLITS)
        row["incidental_background_actor_counts"].update((3, 4, 5, 6))
        row["behavior_categories"].update(
            requirements.required_behavior_categories[scenario_id]
        )
    runner._validate_formal_completion(
        progress, config.formal_scenario_quotas, requirements
    )

    progress["S9_narrow_channel_negotiation"]["splits"].remove("test")
    with pytest.raises(runner.JointCollectionError) as exc_info:
        runner._validate_formal_completion(
            progress, config.formal_scenario_quotas, requirements
        )
    assert exc_info.value.reason_code == "formal_pilot_split_coverage_incomplete"


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


def _targeted_test_config(
    tmp_path: Path,
    *,
    split_quotas: dict[str, int],
    max_episodes: int,
) -> runner.JointCollectionRunConfig:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v3_s5_release20.yaml")
    )
    requirements = _targeted_requirements(
        split_quotas=split_quotas,
        samples=1,
        background_quotas={4: sum(split_quotas.values())},
    )
    return runner.replace(
        config,
        bundle_root=tmp_path / "bundle",
        dataset_root=tmp_path / "bundle/platoon_joint_bev",
        sidecar_root=tmp_path / "bundle/riskentry_actor_sidecar",
        target_joint_steps=sum(split_quotas.values()),
        max_episodes=max_episodes,
        max_episode_steps=1,
        targeted_supplement=requirements,
    )


def _install_targeted_fake_env(monkeypatch, collect_fn):
    env_configs = []

    class _Spawn:
        def set_episode_spawn_seed(self, seed):
            del seed

    class _FakeEnv:
        def __init__(self, env_config):
            self.config = dict(env_config)
            env_configs.append(self.config)
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

        def set_runtime_scenario_route(self, scenario_id, local_route):
            self.config.update(
                {"scenario_id": scenario_id, "local_route": local_route}
            )

        def close(self):
            pass

    monkeypatch.setattr(runner, "SensorlessJointBEVPlatoonEnv", _FakeEnv)
    monkeypatch.setattr(runner, "collect_joint_episode", collect_fn)
    return env_configs


def _failed_targeted_rollout(max_steps: int) -> JointEpisodeRollout:
    return JointEpisodeRollout(
        samples=(_sample(),),
        simulator_steps=max_steps,
        rejected_joint_steps=0,
        failure_reason="trajectory_failed",
        terminated=False,
        truncated=True,
        sample_step_indices=(0,),
        scenario_summary={
            "resolved_scenario_parameters": {
                "incidental_background_actor_count": 4,
            },
            "conflict_evidence": {
                "incidental_background_realized_count": 4,
                "observed_behavior_class": "keep_emergency_braking",
            },
        },
        sidecar=_sidecar(max_steps),
    )


def _small_candidate_v4_success_rollout() -> JointEpisodeRollout:
    samples = []
    signed_y = (
        (0.0, 0.0, 0.0),
        (3.0, -3.0, 3.0),
        (3.0, -3.0, 3.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    for step in range(6):
        sample = _sample()
        modes = np.array(sample.gt_mode, copy=True)
        if step == 1:
            modes[:] = (
                int(ModeIndex.LEFT_LOW),
                int(ModeIndex.RIGHT_LOW),
                int(ModeIndex.LEFT_LOW),
            )
        elif step == 3:
            modes[:] = (
                int(ModeIndex.RIGHT_LOW),
                int(ModeIndex.LEFT_LOW),
                int(ModeIndex.RIGHT_LOW),
            )
        pose = np.array(sample.ego_pose_global, copy=True)
        pose[:, 1] = signed_y[step]
        valid_mask = np.array(sample.mode_valid_mask, copy=True)
        valid_mask[np.arange(3), modes] = True
        samples.append(
            runner.replace(
                sample,
                gt_mode=modes,
                ego_pose_global=pose,
                mode_valid_mask=valid_mask,
            )
        )
    return JointEpisodeRollout(
        samples=tuple(samples),
        simulator_steps=6,
        rejected_joint_steps=0,
        failure_reason=None,
        terminated=False,
        truncated=True,
        sample_step_indices=tuple(range(6)),
        scenario_summary={
            "resolved_scenario_parameters": {
                "incidental_background_actor_count": 4,
                "sampling_policy_id": "s5_release_enriched_80_v1",
                "target_background_condition_sampled": True,
                "left_relation": "ahead",
                "right_relation": "behind",
                "left_offset_m": 20.0,
                "right_offset_m": -15.0,
            },
            "conflict_evidence": {
                "incidental_background_realized_count": 4,
                "observed_behavior_class": (
                    "temporary_formation_release_and_recovery"
                ),
                "mixed_direction_lane_change_completed": True,
                "reassembly_completed": True,
                "reassembly_to_initial_lane_completed": True,
                "hazard_cleared": True,
                "formation_recovered_after_hazard": True,
                "s5_target_background_condition_realized": True,
                "s5_adjacent_background_realization": {
                    "left": {"realized_offset_m": 20.0},
                    "right": {"realized_offset_m": -15.0},
                },
            },
        },
        sidecar=_sidecar(6),
    )


def test_targeted_scheduler_skips_do_not_count_as_simulator_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    config = _targeted_test_config(
        tmp_path,
        split_quotas={"train": 0, "val": 1, "test": 0},
        max_episodes=5,
    )
    calls = []

    def _collect(env, *, max_steps, reset_seed=None):
        calls.append(reset_seed)
        return _failed_targeted_rollout(max_steps)

    env_configs = _install_targeted_fake_env(monkeypatch, _collect)
    with pytest.raises(runner.JointCollectionError) as exc_info:
        runner.run_collection(config)
    assert exc_info.value.reason_code == "targeted_split_quota_incomplete"
    rows = [
        json.loads(line)
        for line in (config.bundle_root / "bundle_episode_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["outcome"] for row in rows[:4]] == ["scheduler_skip"] * 4
    assert len(calls) == 1
    assert runner._targeted_simulator_attempts(
        type("Bundle", (), {"rows": [runner.BundleEpisodeResult.from_mapping(row) for row in rows]})()
    ) == 1
    assert env_configs[0]["rule_maker_profile_id"] == "balanced"


def test_targeted_first_ten_zero_success_stops_and_keeps_sidecars(
    tmp_path: Path, monkeypatch
) -> None:
    config = _targeted_test_config(
        tmp_path,
        split_quotas={"train": 1, "val": 0, "test": 0},
        max_episodes=100,
    )
    calls = []

    def _collect(env, *, max_steps, reset_seed=None):
        calls.append(reset_seed)
        return _failed_targeted_rollout(max_steps)

    _install_targeted_fake_env(monkeypatch, _collect)
    with pytest.raises(runner.JointCollectionError) as exc_info:
        runner.run_collection(config)
    assert exc_info.value.reason_code == "targeted_initial_feasibility_failed"
    assert len(calls) == 10
    assert len(calls) == len(set(calls))
    report = verify_joint_risk_bundle(
        config.bundle_root, scenario_contract_id="candidate_v3"
    )
    assert report["committed_base_episodes"] == 0
    assert report["committed_sidecar_episodes"] == 10
    assert report["sidecar_only_episodes"] == 10


def test_targeted_parallel_coordinator_stops_before_attempt_eleven(
    tmp_path: Path, monkeypatch
) -> None:
    config = _targeted_test_config(
        tmp_path,
        split_quotas={"train": 20, "val": 0, "test": 0},
        max_episodes=100,
    )
    config = runner.replace(
        config,
        targeted_supplement=runner.replace(
            config.targeted_supplement, parallel_workers=4
        ),
    )
    batch_sizes = []

    def _batch(config, specs):
        batch_sizes.append(len(specs))
        return tuple(
            ("ok", _failed_targeted_rollout(1), 0.1) for _ in specs
        )

    monkeypatch.setattr(runner, "_collect_targeted_batch", _batch)
    monkeypatch.setattr(
        runner,
        "SensorlessJointBEVPlatoonEnv",
        lambda config: pytest.fail("precollected parallel rollout created an env"),
    )
    with pytest.raises(runner.JointCollectionError) as exc_info:
        runner.run_collection(config)
    assert exc_info.value.reason_code == "targeted_initial_feasibility_failed"
    assert batch_sizes == [4, 4, 2]
    assert not (config.bundle_root / runner.TARGETED_BATCH_PENDING_FILE).exists()


def test_candidate_v4_stop_flag_passes_at_eight_and_resume_starts_at_attempt_eleven(
    tmp_path: Path, monkeypatch
) -> None:
    config = runner.load_run_config(
        Path("configs/dataset/data_collect_candidate_v4_s5_release20.yaml")
    )
    requirements = runner.replace(
        config.targeted_supplement,
        required_samples_per_episode=6,
        accepted_background_count_quotas={4: 20},
    )
    config = runner.replace(
        config,
        bundle_root=tmp_path / "candidate_v4_bundle",
        dataset_root=tmp_path / "candidate_v4_bundle/platoon_joint_bev",
        sidecar_root=(
            tmp_path / "candidate_v4_bundle/riskentry_actor_sidecar"
        ),
        target_joint_steps=120,
        max_episode_steps=6,
        targeted_supplement=requirements,
    )
    batch_sizes = []

    def _control_rollout():
        rollout = _failed_targeted_rollout(6)
        return runner.replace(
            rollout,
            scenario_summary={
                "resolved_scenario_parameters": {
                    "incidental_background_actor_count": 4,
                    "sampling_policy_id": "s5_release_enriched_80_v1",
                    "target_background_condition_sampled": False,
                    "left_relation": "ahead",
                    "right_relation": "ahead",
                    "left_offset_m": 12.0,
                    "right_offset_m": 14.0,
                },
                "conflict_evidence": {
                    "incidental_background_realized_count": 4,
                    "observed_behavior_class": "keep_emergency_braking",
                    "s5_target_background_condition_realized": False,
                    "s5_adjacent_background_realization": {},
                },
            },
            sidecar=_sidecar(6),
        )

    def _batch(_config, specs):
        batch_sizes.append(len(specs))
        return tuple(
            (
                "ok",
                (
                    _small_candidate_v4_success_rollout()
                    if spec.spawn_seed % 10 < 8
                    else _control_rollout()
                ),
                0.1,
            )
            for spec in specs
        )

    monkeypatch.setattr(runner, "_collect_targeted_batch", _batch)
    monkeypatch.setattr(
        runner,
        "SensorlessJointBEVPlatoonEnv",
        lambda env_config: pytest.fail("parallel rollout created a serial env"),
    )
    summary = runner.run_collection(config, stop_after_initial_gate=True)
    assert batch_sizes == [4, 4, 2]
    targeted = summary["targeted_supplement"]
    assert targeted["stopped_after_initial_gate"] is True
    assert targeted["initial_gate"]["passed"] is True
    assert targeted["observed"]["simulator_attempts"] == 10
    assert targeted["observed"]["accepted_episodes"] == 8
    assert not (config.bundle_root / runner.TARGETED_BATCH_PENDING_FILE).exists()

    from expert_dataset.finalize_s5_targeted_supplement import (
        audit_targeted_supplement,
    )

    audit = audit_targeted_supplement(config)
    assert audit["complete"] is False
    assert audit["initial_gate"]["passed"] is True
    assert audit["initial_gate"]["target_background_episodes_sampled"] == 8
    assert audit["initial_gate"]["target_background_episodes_realized"] == 8

    with pytest.raises(runner.JointCollectionError) as exc_info:
        runner.run_collection(runner.replace(config, max_episodes=11))
    assert exc_info.value.reason_code == "targeted_split_quota_incomplete"
    rows = [
        json.loads(line)
        for line in (config.bundle_root / "bundle_episode_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["outcome"] != "scheduler_skip"
    ]
    assert len(rows) == 11
    assert len({int(row["spawn_seed"]) for row in rows}) == 11


def test_targeted_parallel_pending_batch_recovers_as_actual_attempts(
    tmp_path: Path,
) -> None:
    config = _targeted_test_config(
        tmp_path,
        split_quotas={"train": 2, "val": 0, "test": 0},
        max_episodes=2,
    )
    fingerprint = config.immutable_fingerprint()
    with runner.JointBEVDatasetStore(
        config.dataset_root,
        split_config=config.split_config,
        dataset_fingerprint=fingerprint,
        resume=False,
    ) as base, runner.RiskEntrySidecarDatasetStore(
        config.sidecar_root,
        base_dataset_fingerprint=fingerprint,
        resume=False,
    ) as sidecar, runner.JointRiskBundleIndex(
        config.bundle_root,
        base_directory=config.dataset_root.name,
        sidecar_directory=config.sidecar_root.name,
        base_dataset_fingerprint=fingerprint,
        sidecar_dataset_fingerprint=sidecar.dataset_fingerprint,
        scenario_contract_sha256=config.scenario_contract()["sha256"],
        split_seed=17,
        resume=False,
    ) as bundle:
        specs, entries = runner._plan_targeted_parallel_batch(
            runner.replace(
                config,
                targeted_supplement=runner.replace(
                    config.targeted_supplement, parallel_workers=2
                ),
            ),
            start_episode_index=0,
            progress=runner._empty_targeted_progress(
                config.targeted_supplement
            ),
            bundle=bundle,
            simulator_attempts=0,
        )
        assert len(specs) == 2
        runner._write_targeted_batch_pending(config, entries)
        runner._recover_targeted_batch_pending(config, bundle, base)
        runner._validate_bundle_resume_state(bundle, base, sidecar)
        assert runner._targeted_simulator_attempts(bundle) == 2
        assert {row.outcome for row in bundle.rows} == {
            "interrupted_parallel_rollout"
        }
        assert not (
            config.bundle_root / runner.TARGETED_BATCH_PENDING_FILE
        ).exists()


def test_targeted_resume_restores_actual_attempt_count_and_unique_seeds(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def _collect(env, *, max_steps, reset_seed=None):
        calls.append(reset_seed)
        return _failed_targeted_rollout(max_steps)

    _install_targeted_fake_env(monkeypatch, _collect)
    first = _targeted_test_config(
        tmp_path,
        split_quotas={"train": 1, "val": 0, "test": 0},
        max_episodes=1,
    )
    with pytest.raises(runner.JointCollectionError):
        runner.run_collection(first)
    second = runner.replace(first, max_episodes=2)
    with pytest.raises(runner.JointCollectionError):
        runner.run_collection(second)
    assert calls[0] == 23
    assert len(calls) == 2
    assert len(set(calls)) == 2


def test_targeted_category_mismatch_is_sidecar_only(
    tmp_path: Path, monkeypatch
) -> None:
    config = _targeted_test_config(
        tmp_path,
        split_quotas={"train": 1, "val": 0, "test": 0},
        max_episodes=1,
    )

    def _collect(env, *, max_steps, reset_seed=None):
        rollout = _failed_targeted_rollout(max_steps)
        return runner.replace(rollout, failure_reason=None)

    _install_targeted_fake_env(monkeypatch, _collect)
    with pytest.raises(runner.JointCollectionError):
        runner.run_collection(config)
    report = verify_joint_risk_bundle(
        config.bundle_root, scenario_contract_id="candidate_v3"
    )
    assert report["committed_base_episodes"] == 0
    assert report["sidecar_only_episodes"] == 1
    row = json.loads(
        (config.bundle_root / "bundle_episode_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["base_rejection_reason"] == (
        "targeted_sustained_emergency_braking"
    )
    from expert_dataset.finalize_s5_targeted_supplement import (
        audit_targeted_supplement,
    )

    audit = audit_targeted_supplement(config)
    assert audit["complete"] is False
    assert audit["simulator_attempts"] == 1
    assert audit["behavior_category_counts"] == {
        "keep_emergency_braking": 1
    }
    assert audit["rejection_reason_counts"] == {
        "targeted_sustained_emergency_braking": 1
    }


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
