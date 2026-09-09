"""Stable reward/application contracts for vehicle-mode GRPO rewards."""

from __future__ import annotations

import hashlib
import json

VEHICLE_MODE_REWARD_CONTRACT = {
    "version": "stage2_vehicle_mode_reward_v1",
    "comparison_unit": "vehicle_mode",
    "candidate_shape": "[3,10,N,8,3]",
    "same_mode_pretrain_shape": "[3,10,8,3]",
    "teammate_context": "frozen Stage 1 argmax raw tau_d",
    "counterfactual": (
        "replace only the target vehicle trajectory; keep both teammates fixed"
    ),
    "reward_scope": (
        "target progress, comfort, road and out-of-drivable plus target-to-"
        "background and target-to-teammate gap, TTC, collision and clearance"
    ),
    "formation_component": False,
    "other_vehicle_self_events": False,
    "formula": (
        "+progress_weight*progress_score"
        "-gap_weight*gap_penalty"
        "-ttc_weight*ttc_penalty"
        "-road_weight*road_penalty"
        "-comfort_weight*comfort_penalty"
        "-collision_penalty*collision"
        "-out_of_drivable_penalty*out_of_drivable"
    ),
    "invalid_mode_values": "zero-filled and excluded by valid_mode_mask",
}
VEHICLE_MODE_REWARD_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        VEHICLE_MODE_REWARD_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

GRPO_OPEN_REWARD_APPLICATION_CONTRACT = {
    "version": "stage2_grpo_open_application_v6",
    "comparison_unit": "vehicle_mode",
    "policy_sample_domain": "all-mode raw tau_d",
    "policy_probability_domain": "per-vehicle per-mode DDIM trajectory transitions",
    "mode_policy_terms": False,
    "reward_input_domain": "tau_d",
    "teammate_reward_context": "frozen Stage 1 argmax raw tau_d",
    "same_mode_reference": "frozen Stage 1 raw tau_d for the target mode",
    "advantage_normalization": (
        "per-(vehicle,mode) mean-centered population-RMS standardization"
    ),
    "active_mode_condition": (
        "hard-valid, optimizer-executable and mean(candidate_reward) >= "
        "same-mode pretrain reward"
    ),
    "activation_margin": None,
    "activation_reward_span_threshold": None,
    "inactive_mode_loss_terms": "excluded from trajectory PG only",
    "anchor_scope": (
        "trajectory BC and trajectory KL cover every hard-valid "
        "optimizer-executable vehicle-mode"
    ),
    "sampled_candidate_execution": False,
    "sampled_candidate_optimizer_or_rule_maker": False,
    "live_state_retry": (
        "resample diffusion noise only when every vehicle-mode is inactive"
    ),
    "environment_execution_policy": (
        "always cached deterministic frozen Stage 1 argmax baseline"
    ),
    "execution_input_domain": "tau_cmd",
    "execution_transform": (
        "raw frozen baseline -> KinematicTrajectoryOptimizer -> "
        "RuleMaker finalize/safe-stop -> env.step"
    ),
    "baseline_execution_per_live_state": 1,
    "tracking_expansion_enabled": False,
    "calibration_required": False,
    "best_checkpoint_metric": (
        "validation/safety_constrained_vehicle_reward_gain_trailing3"
    ),
    "validation_reward_family": "vehicle_mode_counterfactual",
}
GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
