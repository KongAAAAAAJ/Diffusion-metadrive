from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import evaluation.compare_grpo_stability as comparison


ARM = "single_step_fixed_scale_mode_residual"
COMMIT = "1" * 40
SOURCE_SHA = "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _policy_contract() -> dict[str, object]:
    return {
        "version": "stage2_joint_grpo_optimizer_v8",
        "ddim_path": comparison.ALIGNED_DDIM_PATH,
        "ppo_ratio_or_clipping": False,
        "advantage": (
            "collision_or_out=-1; otherwise fixed baseline-relative truncation"
        ),
        "reference_regularization": (
            "trajectory anchors cover every hard-valid optimizer-executable mode"
        ),
        "trainable_parameters": (
            "ten zero-initialized mode-specific output residual weights and biases"
        ),
    }


def _validation_history(gain: float) -> list[dict[str, float]]:
    return [
        {
            "accepted_update_state": float(state),
            "simulator_reward_gain": gain,
            "selected_reward_gain": gain + 0.1,
            "s7_out_delta": 0.0,
        }
        for state in range(20, 501, 20)
    ]


def _build_manifest(tmp_path: Path) -> Path:
    bank = tmp_path / "validation_bank.pt"
    bank.write_bytes(b"fixed-bank")
    bank_sha = hashlib.sha256(bank.read_bytes()).hexdigest()

    run27 = tmp_path / "run_27"
    _write_json(
        run27 / "report.json",
        {
            "training_status": "diagnostic_early_stop",
            "accepted_update_states": 240,
        },
    )
    runs = []
    for seed, gain in zip(comparison.PAIRED_SEEDS, (0.1, 0.2, -0.01)):
        run_dir = tmp_path / f"run_{seed}"
        _write_json(
            run_dir / "config.json",
            {
                "format": "bev_joint_grpo_online_config_v13",
                "implementation_commit": COMMIT,
                "source_stage1_sha256": SOURCE_SHA,
                "run_mode": "smoke",
                "online_config": {
                    "seed": seed,
                    "trajectories_per_mode": 48,
                    "total_rollout_groups": 500,
                    "validation_interval_rollouts": 20,
                },
                "policy_update_contract": _policy_contract(),
                "rollout_collection_contract": {
                    "version": "stage2_joint_grpo_persistent_episode_v7"
                },
            },
        )
        _write_json(
            run_dir / "report.json",
            {
                "format": "bev_joint_grpo_online_report_v13",
                "diagnostic_only": True,
                "training_status": "complete",
                "accepted_update_states": 500,
                "checkpoint_selection_status": "eligible",
                "best_checkpoint": str(run_dir / "best.pt"),
                "validation_selection_history": _validation_history(gain),
            },
        )
        runs.append({"arm": ARM, "seed": seed, "run_dir": str(run_dir)})

    manifest = {
        "format": comparison.MANIFEST_FORMAT,
        "diagnostic_only": True,
        "source_stage1_sha256": SOURCE_SHA,
        "paired_seeds": list(comparison.PAIRED_SEEDS),
        "scenario_seeds": [17, 23],
        "validation_seeds": [31, 47],
        "target_accepted_update_states": 500,
        "validation_interval_rollouts": 20,
        "validation_state_bank": str(bank),
        "validation_state_bank_sha256": bank_sha,
        "hyperparameters": {
            "trajectories_per_mode": 48,
            "learning_rate": 1e-5,
            "bc_weight": 0.1,
            "reference_kl_weight": 0.02,
            "max_grad_norm": 1.0,
            "advantage_scale": 1.0,
            "baseline_tolerance": 1e-6,
            "weight_decay": 0.0,
        },
        "arms": {
            ARM: {
                "commit": COMMIT,
                "online_config_format": "bev_joint_grpo_online_config_v13",
                "online_report_format": "bev_joint_grpo_online_report_v13",
                "ddim_path": comparison.ALIGNED_DDIM_PATH,
                "advantage": "fixed_scale_baseline_relative_safe_truncation",
                "policy_update": "single_step_on_policy",
                "anchor_scope": "all_valid_executable",
                "trainable_parameters": "mode_specific_output_residuals",
                "stability_guard": {
                    "post_update_reference_kl_max": 0.25,
                    "max_adapter_relative_drift": 0.02,
                },
            }
        },
        "historical_control_run27": {
            "run_dir": str(run27),
            "training_status": "diagnostic_early_stop",
            "accepted_update_states": 240,
        },
        "runs": runs,
    }
    path = tmp_path / "manifest.json"
    _write_json(path, manifest)
    return path


def test_v7_combination_diagnostic_passes_three_seed_gate(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    report = comparison.compare_grpo_stability(
        manifest, tmp_path / "comparison"
    )
    assert report["format"] == comparison.REPORT_FORMAT
    assert report["stable_arm_500_state_gate"]["passed"] is True
    assert report["stable_arm_500_state_gate"]["tail_nonnegative_seed_count"] == 2
    assert Path(report["plot"]).is_file()


def test_v7_rejects_any_stability_guard_failure(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    failed_dir = Path(value["runs"][0]["run_dir"])
    report_path = failed_dir / "report.json"
    failed = json.loads(report_path.read_text(encoding="utf-8"))
    failed.update(
        {
            "training_status": "stability_guard_rejected",
            "accepted_update_states": 0,
            "checkpoint_selection_status": "no_eligible_checkpoint",
            "best_checkpoint": None,
            "validation_selection_history": [],
        }
    )
    _write_json(report_path, failed)

    report = comparison.compare_grpo_stability(
        manifest, tmp_path / "comparison"
    )
    assert report["stable_arm_500_state_gate"]["any_stability_guard_rejection"]
    assert report["stable_arm_500_state_gate"]["passed"] is False


def test_v7_rejects_ppo_semantics(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    run_dir = Path(value["runs"][0]["run_dir"])
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["policy_update_contract"]["ppo_ratio_or_clipping"] = True
    _write_json(config_path, config)
    with pytest.raises(comparison.GRPOStabilityComparisonError, match="PPO"):
        comparison.compare_grpo_stability(manifest, tmp_path / "comparison")
