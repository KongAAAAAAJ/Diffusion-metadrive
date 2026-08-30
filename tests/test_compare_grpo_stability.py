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
from models.bev_planner.joint_grpo import JointGRPOConfig


SOURCE_SHA = "a" * 64
VALIDATION_STEPS = [20, 40, 60, 80, 100]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_config(arm: str, seed: int) -> dict[str, object]:
    update_epochs = comparison.ARM_UPDATE_EPOCHS[arm]
    return {
        "format": comparison.ONLINE_CONFIG_FORMAT,
        "grpo_config": dict(comparison.EXPECTED_GRPO_CONFIG),
        "source_stage1_sha256": SOURCE_SHA,
        "reward_contract_sha256": "b" * 64,
        "reward_config_sha256": "c" * 64,
        "reward_application_contract_sha256": "d" * 64,
        "trajectory_optimizer_sha256": "e" * 64,
        "scenario_contract_sha256": "f" * 64,
        "policy_update_contract": {
            "version": "stage2_joint_grpo_optimizer_v2",
            "update_epochs": update_epochs,
            "clip_epsilon": comparison.CLIP_EPSILON,
        },
        "online_config": {
            "device": "cuda",
            "seed": seed,
            "group_size": comparison.GROUP_SIZE,
            "total_rollout_groups": comparison.TOTAL_ROLLOUT_GROUPS,
            "update_epochs": update_epochs,
            "clip_epsilon": comparison.CLIP_EPSILON,
            "resume_checkpoint": None,
            "scenarios": [
                ["S5_hard_brake_lead", "R1_entry_straight"],
                ["S6_background_merge_in", "R6_mainline_merge_approach"],
                ["S7_ego_merge_from_ramp", "R7_merge_core"],
                ["S8_ego_exit_to_ramp", "R6_exit_to_ramp"],
                ["S9_narrow_channel_negotiation", "R8_narrow_channel"],
            ],
            "scenario_seeds": list(comparison.SCENARIO_SEEDS),
            "environment_steps_per_episode": 100,
            "validation_interval_rollouts": (
                comparison.VALIDATION_INTERVAL_ROLLOUTS
            ),
            "advantage_vector_log_interval_rollouts": 20,
        },
    }


def _run_report(
    arm: str,
    *,
    uninformative_rollouts: int = 0,
    wall_time_seconds: float = 12.5,
) -> dict[str, object]:
    optimizer_steps = (
        comparison.TOTAL_ROLLOUT_GROUPS - uninformative_rollouts
    ) * comparison.ARM_UPDATE_EPOCHS[arm]
    return {
        "format": comparison.ONLINE_REPORT_FORMAT,
        "grpo_config": dict(comparison.EXPECTED_GRPO_CONFIG),
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "checkpoint_round_trip": True,
        "sampled_rollouts": comparison.TOTAL_ROLLOUT_GROUPS,
        "sampled_rollouts_this_run": comparison.TOTAL_ROLLOUT_GROUPS,
        "target_rollout_groups": comparison.TOTAL_ROLLOUT_GROUPS,
        "uninformative_rollouts": uninformative_rollouts,
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_this_run": optimizer_steps,
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
        "clip_epsilon": comparison.CLIP_EPSILON,
        "scenario_seeds": list(comparison.SCENARIO_SEEDS),
        "validation_seeds": list(comparison.VALIDATION_SEEDS),
        "runs": runs,
    }
    path = root / "manifest.json"
    _write_json(path, manifest)
    return path


def test_expected_grpo_config_matches_core_training_defaults() -> None:
    assert comparison.EXPECTED_GRPO_CONFIG == dataclasses.asdict(
        JointGRPOConfig(group_size=comparison.GROUP_SIZE)
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
    assert pairs[17]["validation_rollout_groups"] == VALIDATION_STEPS
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


def test_compare_grpo_stability_rejects_incomplete_pair_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][-1] = dict(manifest["runs"][0])
    _write_json(manifest_path, manifest)

    with pytest.raises(GRPOStabilityComparisonError, match="duplicate run"):
        compare_grpo_stability(manifest_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("update_epochs", 2, "update_epochs mismatch"),
        ("group_size", 8, "group_size mismatch"),
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


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "was not found"),
        ("duplicate", "duplicate rollout step 20"),
        ("non_finite", "must be finite"),
        ("misaligned", "must use fresh-rollout steps"),
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
