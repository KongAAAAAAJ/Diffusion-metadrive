from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

import evaluation.compare_grpo_stability as comparison
from evaluation.compare_grpo_stability import (
    GRPOStabilityComparisonError,
    compare_grpo_stability,
)
from models.bev_planner.joint_grpo import (
    JointGRPOConfig,
    JointGRPOPolicyUpdateConfig,
    joint_grpo_optimizer_contract,
    joint_grpo_optimizer_contract_sha256,
)
from models.bev_planner.joint_reward import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
)


SOURCE_SHA = "a" * 64
VALIDATION_STEPS = [20, 40, 60, 80, 100]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_config(arm: str, seed: int) -> dict[str, object]:
    update_epochs = comparison.ARM_UPDATE_EPOCHS[arm]
    policy_update = JointGRPOPolicyUpdateConfig(
        update_epochs=update_epochs,
        clip_epsilon_low=comparison.CLIP_EPSILON_LOW,
        clip_epsilon_high=comparison.CLIP_EPSILON_HIGH,
    )
    return {
        "format": comparison.ONLINE_CONFIG_FORMAT,
        "grpo_config": dict(comparison.EXPECTED_GRPO_CONFIG),
        "source_stage1_sha256": SOURCE_SHA,
        "reward_contract_sha256": "b" * 64,
        "reward_config_sha256": "c" * 64,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_application_contract": dict(
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT
        ),
        "trajectory_optimizer_sha256": "e" * 64,
        "scenario_contract_sha256": "f" * 64,
        "policy_update_contract": joint_grpo_optimizer_contract(policy_update),
        "policy_update_contract_sha256": (
            joint_grpo_optimizer_contract_sha256(policy_update)
        ),
        "rollout_collection_contract": (
            comparison._expected_rollout_collection_contract(200)
        ),
        "online_config": {
            "device": "cuda",
            "seed": seed,
            "group_size": comparison.GROUP_SIZE,
            "total_rollout_groups": comparison.TOTAL_ROLLOUT_GROUPS,
            "update_epochs": update_epochs,
            "clip_epsilon_low": comparison.CLIP_EPSILON_LOW,
            "clip_epsilon_high": comparison.CLIP_EPSILON_HIGH,
            "resume_checkpoint": None,
            "scenarios": [
                ["S5_hard_brake_lead", "R1_entry_straight"],
                ["S6_background_merge_in", "R6_mainline_merge_approach"],
                ["S7_ego_merge_from_ramp", "R7_merge_core"],
                ["S8_ego_exit_to_ramp", "R6_exit_to_ramp"],
                ["S9_narrow_channel_negotiation", "R8_narrow_channel"],
            ],
            "scenario_seeds": list(comparison.SCENARIO_SEEDS),
            "environment_steps_per_episode": 200,
            "rollout_groups_per_bucket_visit": (
                comparison.ROLLOUT_GROUPS_PER_BUCKET_VISIT
            ),
            "rollout_start_offset_max_steps": (
                comparison.ROLLOUT_START_OFFSET_MAX_STEPS
            ),
            "rollout_start_min_remaining_steps": (
                comparison.ROLLOUT_START_MIN_REMAINING_STEPS
            ),
            "pretrain_improvement_margin": (
                comparison.PRETRAIN_IMPROVEMENT_MARGIN
            ),
            "max_candidate_groups_per_state": (
                comparison.MAX_CANDIDATE_GROUPS_PER_STATE
            ),
            "max_attempted_groups_multiplier": (
                comparison.MAX_ATTEMPTED_GROUPS_MULTIPLIER
            ),
            "validation_interval_rollouts": (
                comparison.VALIDATION_INTERVAL_ROLLOUTS
            ),
            "advantage_vector_log_interval_rollouts": 20,
        },
    }


def _run_report(
    arm: str,
    *,
    wall_time_seconds: float = 12.5,
) -> dict[str, object]:
    update_epochs = comparison.ARM_UPDATE_EPOCHS[arm]
    policy_update = JointGRPOPolicyUpdateConfig(
        update_epochs=update_epochs,
        clip_epsilon_low=comparison.CLIP_EPSILON_LOW,
        clip_epsilon_high=comparison.CLIP_EPSILON_HIGH,
    )
    optimizer_steps = comparison.TOTAL_ROLLOUT_GROUPS * update_epochs
    bucket_counters = []
    bucket_count = len(comparison.SCENARIOS) * len(comparison.SCENARIO_SEEDS)
    base_target, remainder = divmod(
        comparison.TOTAL_ROLLOUT_GROUPS, bucket_count
    )
    buckets = (
        (scenario, scenario_seed)
        for scenario in comparison.SCENARIOS
        for scenario_seed in comparison.SCENARIO_SEEDS
    )
    for bucket_index, (scenario, scenario_seed) in enumerate(buckets):
        target_rollouts = base_target + int(bucket_index < remainder)
        bucket_counters.append(
            {
                "scenario": scenario[0],
                "route": scenario[1],
                "seed": scenario_seed,
                "target_accepted_rollouts": target_rollouts,
                "accepted_rollouts": target_rollouts,
                "attempted_rollouts": target_rollouts,
                "rejected_candidate_groups": 0,
                "rejected_no_pretrain_improvement": 0,
                "rejected_zero_reward_span": 0,
                "pretrain_fallback_steps": 0,
                "environment_episodes": 1,
                "optimizer_steps": target_rollouts * update_epochs,
            }
        )
    return {
        "format": comparison.ONLINE_REPORT_FORMAT,
        "training_status": "complete",
        "grpo_config": dict(comparison.EXPECTED_GRPO_CONFIG),
        "reward_application_contract": dict(
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT
        ),
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "policy_update_contract": joint_grpo_optimizer_contract(policy_update),
        "policy_update_contract_sha256": (
            joint_grpo_optimizer_contract_sha256(policy_update)
        ),
        "rollout_collection_contract": (
            comparison._expected_rollout_collection_contract(200)
        ),
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "checkpoint_round_trip": True,
        "accepted_rollout_groups": comparison.TOTAL_ROLLOUT_GROUPS,
        "accepted_rollout_groups_this_run": comparison.TOTAL_ROLLOUT_GROUPS,
        "target_accepted_rollout_groups": comparison.TOTAL_ROLLOUT_GROUPS,
        "attempted_rollout_groups": comparison.TOTAL_ROLLOUT_GROUPS,
        "attempted_rollout_groups_this_run": comparison.TOTAL_ROLLOUT_GROUPS,
        "max_attempted_rollout_groups": (
            comparison.TOTAL_ROLLOUT_GROUPS
            * comparison.MAX_ATTEMPTED_GROUPS_MULTIPLIER
        ),
        "rejected_candidate_groups": 0,
        "rejected_no_pretrain_improvement": 0,
        "rejected_zero_reward_span": 0,
        "pretrain_fallback_steps": 0,
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_this_run": optimizer_steps,
        "environment_steps": 130,
        "environment_episode_count": 10,
        "warmup_environment_steps": 30,
        "rollout_start_diagnostics_this_run": {
            "attempt_count": 10,
            "accepted_count": 10,
            "rejected_count": 0,
            "offset_min_steps": 0,
            "offset_mean_steps": 50.0,
            "offset_max_steps": 100,
        },
        "training_bucket_counters": bucket_counters,
        "training_plots": {
            "reward_x_axis": comparison.ACCEPTED_ROLLOUT_AXIS,
            "validation_reward_x_axis": comparison.ACCEPTED_ROLLOUT_AXIS,
            "optimizer_x_axis": comparison.ACCEPTED_ROLLOUT_AXIS,
        },
        "wall_time_seconds": wall_time_seconds,
        "validation_seeds": list(comparison.VALIDATION_SEEDS),
    }


def _write_reward_gain(
    tb_dir: Path,
    values: list[float],
    *,
    steps: list[int] = VALIDATION_STEPS,
    tag: str = comparison.VALIDATION_REWARD_GAIN_TAG,
    duplicate_first_step: bool = False,
) -> None:
    with SummaryWriter(log_dir=str(tb_dir)) as writer:
        for step, value in zip(steps, values):
            writer.add_scalar(tag, value, global_step=step)
        if duplicate_first_step:
            writer.add_scalar(tag, values[0] + 1.0, global_step=steps[0])


def _write_run(
    root: Path,
    arm: str,
    seed: int,
    values: list[float],
) -> Path:
    run_dir = root / f"{arm}_{seed}"
    _write_json(run_dir / "config.json", _run_config(arm, seed))
    _write_json(run_dir / "report.json", _run_report(arm))
    _write_reward_gain(run_dir / "tb", values)
    return run_dir


def _add_dynamic_sampling_activity(report: dict[str, object]) -> None:
    report["attempted_rollout_groups"] = 104
    report["attempted_rollout_groups_this_run"] = 104
    report["rejected_candidate_groups"] = 4
    report["rejected_no_pretrain_improvement"] = 3
    report["rejected_zero_reward_span"] = 1
    report["pretrain_fallback_steps"] = 1
    report["environment_steps"] = 131
    first_bucket = report["training_bucket_counters"][0]
    first_bucket["attempted_rollouts"] = 14
    first_bucket["rejected_candidate_groups"] = 4
    first_bucket["rejected_no_pretrain_improvement"] = 3
    first_bucket["rejected_zero_reward_span"] = 1
    first_bucket["pretrain_fallback_steps"] = 1


def _gain_series() -> dict[tuple[str, int], list[float]]:
    return {
        ("baseline", 17): [0.00, 0.40, 0.05, 0.45, 0.20],
        ("clipped", 17): [0.00, 0.10, 0.20, 0.30, 0.40],
        ("baseline", 23): [0.00, -0.25, 0.30, -0.10, 0.20],
        ("clipped", 23): [0.00, 0.08, 0.16, 0.24, 0.32],
        ("baseline", 42): [0.00, 0.05, 0.10, 0.15, 0.20],
        ("clipped", 42): [0.00, 0.30, -0.15, 0.25, 0.10],
    }


def _write_manifest(
    root: Path,
    *,
    gains: dict[tuple[str, int], list[float]] | None = None,
) -> Path:
    run_gains = gains or _gain_series()
    runs = []
    for arm in ("clipped", "baseline"):
        for seed in reversed(comparison.PAIRED_SEEDS):
            run_dir = _write_run(root, arm, seed, run_gains[(arm, seed)])
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "run_dir": str(run_dir.relative_to(root)),
                }
            )
    manifest = {
        "format": comparison.MANIFEST_FORMAT,
        "source_stage1_sha256": SOURCE_SHA,
        "paired_seeds": list(comparison.PAIRED_SEEDS),
        "baseline_update_epochs": 1,
        "clipped_update_epochs": 4,
        "group_size": comparison.GROUP_SIZE,
        "total_rollout_groups": comparison.TOTAL_ROLLOUT_GROUPS,
        "validation_interval_rollouts": (
            comparison.VALIDATION_INTERVAL_ROLLOUTS
        ),
        "optimizer_contract_version": comparison.OPTIMIZER_CONTRACT_VERSION,
        "application_contract_version": (
            comparison.APPLICATION_CONTRACT_VERSION
        ),
        "rollout_collection_contract_version": (
            comparison.ROLLOUT_COLLECTION_CONTRACT_VERSION
        ),
        "rollout_groups_per_bucket_visit": (
            comparison.ROLLOUT_GROUPS_PER_BUCKET_VISIT
        ),
        "rollout_start_offset_max_steps": (
            comparison.ROLLOUT_START_OFFSET_MAX_STEPS
        ),
        "rollout_start_min_remaining_steps": (
            comparison.ROLLOUT_START_MIN_REMAINING_STEPS
        ),
        "pretrain_improvement_margin": comparison.PRETRAIN_IMPROVEMENT_MARGIN,
        "max_candidate_groups_per_state": (
            comparison.MAX_CANDIDATE_GROUPS_PER_STATE
        ),
        "max_attempted_groups_multiplier": (
            comparison.MAX_ATTEMPTED_GROUPS_MULTIPLIER
        ),
        "clip_epsilon_low": comparison.CLIP_EPSILON_LOW,
        "clip_epsilon_high": comparison.CLIP_EPSILON_HIGH,
        "scenario_seeds": list(comparison.SCENARIO_SEEDS),
        "validation_seeds": list(comparison.VALIDATION_SEEDS),
        "runs": runs,
    }
    path = root / "manifest.json"
    _write_json(path, manifest)
    return path


def test_expected_grpo_config_matches_core_training_defaults() -> None:
    from train.train_bev_joint_grpo_online import (
        JointGRPOOnlineConfig,
        rollout_collection_contract,
    )

    assert comparison.EXPECTED_GRPO_CONFIG == dataclasses.asdict(
        JointGRPOConfig(group_size=comparison.GROUP_SIZE)
    )
    binding = comparison._run_binding(_run_config("baseline", 17))
    assert binding["rollout_collection_contract"] == (
        comparison._expected_rollout_collection_contract(200)
    )
    assert binding["rollout_start_offset_max_steps"] == 200
    assert binding["rollout_start_min_remaining_steps"] == 10
    assert comparison._expected_rollout_collection_contract(200) == (
        rollout_collection_contract(
            JointGRPOOnlineConfig(environment_steps_per_episode=200)
        )
    )


def test_compare_grpo_stability_emits_passing_report_and_png(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    output_dir = tmp_path / "comparison"

    report = compare_grpo_stability(manifest, output_dir)

    assert report["format"] == comparison.REPORT_FORMAT
    assert report["diagnostic_only"] is True
    assert report["eligible_for_formal_conclusions"] is False
    assert report["rollout_collection_contract_version"] == (
        comparison.ROLLOUT_COLLECTION_CONTRACT_VERSION
    )
    assert report["optimizer_contract_version"] == (
        comparison.OPTIMIZER_CONTRACT_VERSION
    )
    assert report["application_contract_version"] == (
        comparison.APPLICATION_CONTRACT_VERSION
    )
    assert report["rollout_start_offset_max_steps"] == 200
    assert report["rollout_start_min_remaining_steps"] == 10
    assert report["summary"]["volatility_reduced_pair_count"] == 2
    assert report["summary"]["volatility_gate_passed"] is True
    assert report["summary"]["final_mean_non_degraded"] is True
    assert report["summary"]["passed"] is True
    pairs = {value["seed"]: value for value in report["pairs"]}
    expected_baseline_std = float(
        np.std(np.diff(_gain_series()[("baseline", 17)]), ddof=0)
    )
    assert pairs[17]["baseline_adjacent_delta_std"] == pytest.approx(
        expected_baseline_std
    )
    assert pairs[17]["baseline_optimizer_steps"] == 100
    assert pairs[17]["clipped_optimizer_steps"] == 400
    assert pairs[17]["baseline_wall_time_seconds"] == 12.5
    assert pairs[17]["validation_accepted_rollout_groups"] == VALIDATION_STEPS
    assert pairs[17]["baseline_accepted_rollout_groups"] == 100
    assert pairs[17]["baseline_attempted_rollout_groups"] == 100
    assert pairs[17]["baseline_rejected_candidate_groups"] == 0
    assert pairs[17]["baseline_pretrain_fallback_steps"] == 0
    report_path = output_dir / "report.json"
    plot_path = output_dir / "validation_reward_gain_ab.png"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 1_000
    assert plot_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_compare_grpo_stability_reports_failed_acceptance_without_relabeling(
    tmp_path: Path,
) -> None:
    gains = _gain_series()
    gains[("clipped", 17)] = [0.0, 0.5, -0.5, 0.5, -0.2]
    gains[("clipped", 23)] = [0.0, 0.5, -0.5, 0.5, -0.2]
    gains[("clipped", 42)] = [0.0, 0.5, -0.5, 0.5, -0.2]

    report = compare_grpo_stability(
        _write_manifest(tmp_path, gains=gains), tmp_path / "comparison"
    )

    assert report["summary"]["volatility_gate_passed"] is False
    assert report["summary"]["final_mean_non_degraded"] is False
    assert report["summary"]["passed"] is False
    assert report["diagnostic_only"] is True


def test_compare_grpo_stability_preserves_dynamic_sampling_counters(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    report_path = tmp_path / "baseline_17" / "report.json"
    run_report = json.loads(report_path.read_text(encoding="utf-8"))
    _add_dynamic_sampling_activity(run_report)
    _write_json(report_path, run_report)

    report = compare_grpo_stability(manifest, tmp_path / "comparison")

    pair = next(value for value in report["pairs"] if value["seed"] == 17)
    assert pair["baseline_accepted_rollout_groups"] == 100
    assert pair["baseline_attempted_rollout_groups"] == 104
    assert pair["baseline_rejected_candidate_groups"] == 4
    assert pair["baseline_rejected_no_pretrain_improvement"] == 3
    assert pair["baseline_rejected_zero_reward_span"] == 1
    assert pair["baseline_pretrain_fallback_steps"] == 1


def test_compare_grpo_stability_rejects_incomplete_pair_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][-1] = dict(manifest["runs"][0])
    _write_json(manifest_path, manifest)

    with pytest.raises(GRPOStabilityComparisonError, match="duplicate run"):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_rejects_v2_manifest(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "stage2_grpo_stability_ab_manifest_v2"
    _write_json(manifest_path, manifest)

    with pytest.raises(GRPOStabilityComparisonError, match="manifest format mismatch"):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "rollout_collection_contract_version",
            "stage2_joint_grpo_persistent_episode_v2",
        ),
        ("optimizer_contract_version", "stage2_joint_grpo_optimizer_v2"),
        ("application_contract_version", "stage2_grpo_open_application_v1"),
        ("rollout_groups_per_bucket_visit", 5),
        ("rollout_start_offset_max_steps", 199),
        ("rollout_start_min_remaining_steps", 11),
        ("pretrain_improvement_margin", 0.0),
        ("max_candidate_groups_per_state", 4),
        ("max_attempted_groups_multiplier", 4),
    ],
)
def test_compare_grpo_stability_rejects_manifest_collection_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(
        GRPOStabilityComparisonError, match=rf"manifest {field} mismatch"
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_rejects_missing_v3_manifest_field(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("rollout_start_offset_max_steps")
    _write_json(manifest_path, manifest)

    with pytest.raises(GRPOStabilityComparisonError, match="fields mismatch"):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("artifact", "legacy_format", "message"),
    [
        ("config", "bev_joint_grpo_online_config_v6", "config format mismatch"),
        ("report", "bev_joint_grpo_online_report_v6", "report format mismatch"),
    ],
)
def test_compare_grpo_stability_rejects_v6_online_artifacts(
    tmp_path: Path,
    artifact: str,
    legacy_format: str,
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    artifact_path = tmp_path / "baseline_17" / f"{artifact}.json"
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value["format"] = legacy_format
    _write_json(artifact_path, value)

    with pytest.raises(GRPOStabilityComparisonError, match=message):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("update_epochs", 10, "update_epochs mismatch"),
        ("group_size", 8, "group_size mismatch"),
        (
            "rollout_groups_per_bucket_visit",
            5,
            "rollout_groups_per_bucket_visit mismatch",
        ),
        (
            "rollout_start_offset_max_steps",
            199,
            "rollout_start_offset_max_steps mismatch",
        ),
        (
            "rollout_start_min_remaining_steps",
            11,
            "rollout_start_min_remaining_steps mismatch",
        ),
        (
            "pretrain_improvement_margin",
            0.0,
            "pretrain_improvement_margin mismatch",
        ),
        (
            "max_candidate_groups_per_state",
            4,
            "max_candidate_groups_per_state mismatch",
        ),
        (
            "max_attempted_groups_multiplier",
            4,
            "max_attempted_groups_multiplier mismatch",
        ),
        ("resume_checkpoint", "/tmp/legacy.pt", "resume_checkpoint mismatch"),
    ],
)
def test_compare_grpo_stability_rejects_run_config_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    run_config_path = tmp_path / "baseline_17" / "config.json"
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    config["online_config"][field] = value
    _write_json(run_config_path, config)

    with pytest.raises(GRPOStabilityComparisonError, match=message):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("group_size", 4),
        ("mode_pg_weight", 0.5),
        ("trajectory_pg_weight", 0.5),
        ("bc_weight", 0.2),
        ("reference_kl_weight", 0.05),
        ("learning_rate", 2e-5),
        ("weight_decay", 0.0),
        ("max_grad_norm", 0.5),
    ],
)
def test_compare_grpo_stability_rejects_frozen_grpo_config_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    run_config_path = tmp_path / "baseline_17" / "config.json"
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    config["grpo_config"][field] = value
    _write_json(run_config_path, config)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match=rf"config grpo_config {field} mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_requires_complete_grpo_config(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    run_config_path = tmp_path / "baseline_17" / "config.json"
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    config["grpo_config"].pop("advantage_eps")
    _write_json(run_config_path, config)

    with pytest.raises(GRPOStabilityComparisonError, match="fields mismatch"):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_rejects_cross_run_fairness_drift(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    run_config_path = tmp_path / "clipped_23" / "config.json"
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    config["reward_config_sha256"] = "1" * 64
    _write_json(run_config_path, config)

    with pytest.raises(GRPOStabilityComparisonError, match="fairness binding"):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize("artifact", ["config", "report"])
def test_compare_grpo_stability_rejects_v1_application_contract(
    tmp_path: Path,
    artifact: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    artifact_path = tmp_path / "baseline_17" / f"{artifact}.json"
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value["reward_application_contract"]["version"] = (
        "stage2_grpo_open_application_v1"
    )
    _write_json(artifact_path, value)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match=rf"{artifact} application contract mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize("artifact", ["config", "report"])
def test_compare_grpo_stability_rejects_v2_optimizer_contract(
    tmp_path: Path,
    artifact: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    artifact_path = tmp_path / "baseline_17" / f"{artifact}.json"
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value["policy_update_contract"]["version"] = (
        "stage2_joint_grpo_optimizer_v2"
    )
    _write_json(artifact_path, value)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match=rf"{artifact} policy update contract mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize("artifact", ["config", "report"])
def test_compare_grpo_stability_rejects_advantage_normalization_drift(
    tmp_path: Path,
    artifact: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    artifact_path = tmp_path / "baseline_17" / f"{artifact}.json"
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value["policy_update_contract"]["advantage_normalization"] = (
        "group_mean_centered"
    )
    _write_json(artifact_path, value)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match=rf"{artifact} policy update contract mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize("artifact", ["config", "report"])
def test_compare_grpo_stability_rejects_optimizer_contract_sha_drift(
    tmp_path: Path,
    artifact: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    artifact_path = tmp_path / "baseline_17" / f"{artifact}.json"
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value["policy_update_contract_sha256"] = "0" * 64
    _write_json(artifact_path, value)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match=rf"{artifact} policy update contract mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_rejects_non_positive_environment_cap(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    config_path = tmp_path / "clipped_42" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["online_config"]["environment_steps_per_episode"] = 0
    _write_json(config_path, config)

    with pytest.raises(GRPOStabilityComparisonError, match="positive integer"):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_rejects_cross_run_environment_cap_drift(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    config_path = tmp_path / "clipped_23" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["online_config"]["environment_steps_per_episode"] = 201
    config["rollout_collection_contract"]["environment_steps_per_episode"] = 201
    _write_json(config_path, config)
    report_path = tmp_path / "clipped_23" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rollout_collection_contract"]["environment_steps_per_episode"] = 201
    _write_json(report_path, report)

    with pytest.raises(GRPOStabilityComparisonError, match="fairness binding"):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize("artifact", ["config", "report"])
def test_compare_grpo_stability_requires_rollout_collection_contract(
    tmp_path: Path,
    artifact: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    artifact_path = tmp_path / "baseline_17" / f"{artifact}.json"
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value.pop("rollout_collection_contract")
    _write_json(artifact_path, value)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match=rf"{artifact} rollout collection contract mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize("artifact", ["config", "report"])
def test_compare_grpo_stability_rejects_rollout_collection_contract_drift(
    tmp_path: Path,
    artifact: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    artifact_path = tmp_path / "baseline_17" / f"{artifact}.json"
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value["rollout_collection_contract"]["episode_reuse"] = "reset every group"
    _write_json(artifact_path, value)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match=rf"{artifact} rollout collection contract mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "stage2_joint_grpo_persistent_episode_v2"),
        ("rollout_start_offset_distribution", "exclusive_uniform_integer"),
        ("rollout_start_generator", "separate_start_generator"),
        ("validation_start", "randomized_after_ready"),
    ],
)
def test_compare_grpo_stability_rejects_random_start_contract_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    config_path = tmp_path / "baseline_17" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["rollout_collection_contract"][field] = value
    _write_json(config_path, config)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match="config rollout collection contract mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "was not found"),
        ("duplicate", "duplicate rollout step 20"),
        ("non_finite", "must be finite"),
        ("misaligned", "must use accepted-rollout steps"),
    ],
)
def test_compare_grpo_stability_rejects_invalid_reward_gain_events(
    tmp_path: Path,
    kind: str,
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    tb_dir = tmp_path / "clipped_42" / "tb"
    for event_file in tb_dir.glob("events.out.tfevents.*"):
        event_file.unlink()
    if kind == "missing":
        _write_reward_gain(tb_dir, [0.0] * 5, tag="validation/other")
    elif kind == "duplicate":
        _write_reward_gain(tb_dir, [0.0] * 5, duplicate_first_step=True)
    elif kind == "non_finite":
        _write_reward_gain(tb_dir, [0.0, np.nan, 0.0, 0.0, 0.0])
    else:
        _write_reward_gain(tb_dir, [0.0] * 4, steps=[20, 40, 60, 80])

    with pytest.raises(GRPOStabilityComparisonError, match=message):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_steps", 399, "optimizer-step count mismatch"),
        ("training_status", "incomplete", "training did not complete"),
        (
            "attempted_rollout_groups",
            99,
            "dynamic-sampling counters mismatch",
        ),
        (
            "max_attempted_rollout_groups",
            299,
            "dynamic-sampling counters mismatch",
        ),
        (
            "rejected_candidate_groups",
            1,
            "dynamic-sampling counters mismatch",
        ),
        (
            "pretrain_fallback_steps",
            1,
            "dynamic-sampling counters mismatch",
        ),
        ("wall_time_seconds", 0.0, "positive finite scalar"),
        ("validation_seeds", [31], "validation seeds mismatch"),
    ],
)
def test_compare_grpo_stability_rejects_invalid_run_report(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "clipped_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    _write_json(report_path, report)

    with pytest.raises(GRPOStabilityComparisonError, match=message):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    "field",
    ("reward_x_axis", "validation_reward_x_axis", "optimizer_x_axis"),
)
def test_compare_grpo_stability_rejects_nonaccepted_training_plot_axis(
    tmp_path: Path,
    field: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "clipped_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["training_plots"][field] = "absolute_optimizer_step"
    _write_json(report_path, report)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match="training plot axes must use accepted rollout groups",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "environment_episode_count",
            131,
            "environment_episode_count exceeds environment_steps",
        ),
        (
            "warmup_environment_steps",
            29,
            "warmup environment-step count mismatch",
        ),
    ],
)
def test_compare_grpo_stability_rejects_environment_counter_errors(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "clipped_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    _write_json(report_path, report)

    with pytest.raises(GRPOStabilityComparisonError, match=message):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"accepted_count": 0}, "accepted_count must be a positive integer"),
        (
            {"attempt_count": 11},
            "attempt_count must equal accepted_count plus rejected_count",
        ),
        (
            {"offset_min_steps": 51.0},
            "offset statistics must satisfy",
        ),
        (
            {"offset_max_steps": 201},
            "offset statistics must satisfy",
        ),
        (
            {"offset_mean_steps": float("nan")},
            "offset_mean_steps must be a finite scalar",
        ),
    ],
)
def test_compare_grpo_stability_rejects_invalid_start_diagnostics(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "clipped_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rollout_start_diagnostics_this_run"].update(updates)
    _write_json(report_path, report)

    with pytest.raises(GRPOStabilityComparisonError, match=message):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_requires_start_diagnostics(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "baseline_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("rollout_start_diagnostics_this_run")
    _write_json(report_path, report)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match="rollout_start_diagnostics_this_run must be an object",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_accepts_empty_environment_episodes(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    for arm in comparison.ARM_UPDATE_EPOCHS:
        for seed in comparison.PAIRED_SEEDS:
            report_path = tmp_path / f"{arm}_{seed}" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["environment_steps"] = 210
            report["environment_episode_count"] = 110
            report["warmup_environment_steps"] = 110
            for counter in report["training_bucket_counters"]:
                counter["environment_episodes"] = 11
            _write_json(report_path, report)

    report = compare_grpo_stability(manifest_path, tmp_path / "output")

    assert report["summary"]["passed"] is True


def test_compare_grpo_stability_rejects_unbalanced_bucket_targets(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "clipped_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["training_bucket_counters"][0]["target_accepted_rollouts"] = 9
    report["training_bucket_counters"][1]["target_accepted_rollouts"] = 11
    _write_json(report_path, report)

    with pytest.raises(
        GRPOStabilityComparisonError, match="target_accepted_rollouts mismatch"
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_rejects_incomplete_bucket_acceptance(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "clipped_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["training_bucket_counters"][0]["accepted_rollouts"] = 9
    _write_json(report_path, report)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match="accepted_rollouts must equal target_accepted_rollouts",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_rejects_bucket_dynamic_sampling_drift(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "clipped_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["training_bucket_counters"][0]["attempted_rollouts"] = 11
    _write_json(report_path, report)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match="attempted_rollouts must equal accepted plus rejected",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_rejects_report_grpo_config_drift(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "clipped_17" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["grpo_config"]["reference_kl_weight"] = 0.05
    _write_json(report_path, report)

    with pytest.raises(
        GRPOStabilityComparisonError,
        match="report grpo_config reference_kl_weight mismatch",
    ):
        compare_grpo_stability(manifest_path, tmp_path / "output")


def test_compare_grpo_stability_cli_contract() -> None:
    parser = comparison._build_parser()
    args = parser.parse_args(
        ["--manifest", "/tmp/ab.json", "--output-dir", "/tmp/ab-output"]
    )

    assert args.manifest == Path("/tmp/ab.json")
    assert args.output_dir == Path("/tmp/ab-output")
    help_text = parser.format_help()
    assert comparison.MANIFEST_FORMAT in help_text
    assert "report.json" in help_text
    assert "validation_reward_gain_ab.png" in help_text
