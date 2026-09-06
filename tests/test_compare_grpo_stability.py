from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import evaluation.compare_grpo_stability as comparison


SOURCE_SHA = "a" * 64
COMMITS = {
    "legacy_control": "1" * 40,
    "aligned_max": "2" * 40,
    "aligned_mean_full_valid": "3" * 40,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _arm_spec(arm: str) -> dict[str, object]:
    return {
        "commit": COMMITS[arm],
        "online_config_format": "bev_joint_grpo_online_config_v12",
        "online_report_format": "bev_joint_grpo_online_report_v12",
        "ddim_path": (
            comparison.ALIGNED_DDIM_PATH
            if arm != "legacy_control"
            else {"legacy": True}
        ),
        "gate": "mean" if arm == "aligned_mean_full_valid" else "max",
        "anchor_scope": (
            "all_valid_executable"
            if arm == "aligned_mean_full_valid"
            else "active_only"
        ),
    }


def _write_run(root: Path, arm: str, seed: int, base_gain: float) -> Path:
    run_dir = root / f"{arm}_{seed}"
    spec = _arm_spec(arm)
    config = {
        "format": spec["online_config_format"],
        "implementation_commit": COMMITS[arm],
        "source_stage1_sha256": SOURCE_SHA,
        "run_mode": "smoke",
        "online_config": {
            "seed": seed,
            "trajectories_per_mode": 48,
            "total_rollout_groups": 500,
            "update_epochs": 10,
            "clip_epsilon_low": 0.1,
            "clip_epsilon_high": 0.2,
            "validation_interval_rollouts": 20,
        },
        "policy_update_contract": {
            "activation_gate": f"{spec['gate']} reward gate",
            "reference_regularization": (
                "recompute for every hard-valid optimizer-executable mode"
                if spec["anchor_scope"] == "all_valid_executable"
                else "recompute for active modes only"
            ),
            "ddim_path": (
                comparison.ALIGNED_DDIM_PATH
                if arm != "legacy_control"
                else {"legacy": True}
            ),
        },
        "rollout_collection_contract": {
            "active_mode_gate": f"{spec['gate']} reward gate"
        },
    }
    history = [
        {
            "accepted_update_state": float(state),
            "simulator_reward_gain": base_gain + state / 10_000.0,
            "selected_reward_gain": base_gain + 0.1,
            "s7_out_delta": 0.0,
            "safety_eligible": 1.0,
        }
        for state in range(20, 501, 20)
    ]
    report = {
        "format": spec["online_report_format"],
        "training_status": "complete",
        "diagnostic_only": True,
        "accepted_update_states": 500,
        "checkpoint_selection_status": "eligible",
        "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "validation_selection_history": history,
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "report.json", report)
    return run_dir


def _manifest(tmp_path: Path) -> Path:
    bank = tmp_path / "validation_state_bank.pt"
    bank.write_bytes(b"fixed-state-bank")
    bases = {
        "legacy_control": -0.2,
        "aligned_max": -0.1,
        "aligned_mean_full_valid": 0.1,
    }
    runs = []
    for arm in comparison.ARMS:
        for seed in comparison.PAIRED_SEEDS:
            run_dir = _write_run(tmp_path, arm, seed, bases[arm])
            runs.append({"arm": arm, "seed": seed, "run_dir": str(run_dir)})
    value = {
        "format": comparison.MANIFEST_FORMAT,
        "diagnostic_only": True,
        "source_stage1_sha256": SOURCE_SHA,
        "paired_seeds": list(comparison.PAIRED_SEEDS),
        "scenario_seeds": list(comparison.SCENARIO_SEEDS),
        "validation_seeds": list(comparison.VALIDATION_SEEDS),
        "target_accepted_update_states": 500,
        "validation_interval_rollouts": 20,
        "validation_state_bank": str(bank),
        "validation_state_bank_sha256": hashlib.sha256(
            bank.read_bytes()
        ).hexdigest(),
        "hyperparameters": {
            "trajectories_per_mode": 48,
            "update_epochs": 10,
            "clip_epsilon_low": 0.1,
            "clip_epsilon_high": 0.2,
            "learning_rate": 1e-5,
            "bc_weight": 0.1,
            "reference_kl_weight": 0.02,
            "max_grad_norm": 1.0,
        },
        "arms": {arm: _arm_spec(arm) for arm in comparison.ARMS},
        "runs": runs,
    }
    path = tmp_path / "manifest.json"
    _write_json(path, value)
    return path


def test_three_arm_v6_comparison_passes_bounded_gate(tmp_path: Path) -> None:
    report = comparison.compare_grpo_stability(
        _manifest(tmp_path), tmp_path / "comparison"
    )
    assert report["format"] == comparison.REPORT_FORMAT
    assert report["diagnostic_only"] is True
    assert report["eligible_for_formal_conclusions"] is False
    assert report["stable_arm_500_state_gate"]["passed"] is True


def test_comparison_rejects_validation_bank_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    Path(value["validation_state_bank"]).write_bytes(b"changed")
    with pytest.raises(
        comparison.GRPOStabilityComparisonError,
        match="state bank digest",
    ):
        comparison.compare_grpo_stability(manifest, tmp_path / "comparison")


def test_comparison_rejects_aligned_max_mean_gate(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["arms"]["aligned_max"]["gate"] = "mean"
    _write_json(manifest, value)
    with pytest.raises(
        comparison.GRPOStabilityComparisonError,
        match="optimization semantics",
    ):
        comparison.compare_grpo_stability(manifest, tmp_path / "comparison")
