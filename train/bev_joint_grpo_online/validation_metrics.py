"""Vehicle-mode validation comparison and checkpoint-selection helpers."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .config import OnlineGRPOError
from .contracts import _checkpoint_file_sha256, _validated_selection_history


def _finalize_vehicle_mode_validation_metrics(
    *,
    vehicle_rewards: Sequence[float],
    macro_context_vehicle_rewards: Sequence[float],
    macro_context_pretrain_rewards: Sequence[float],
    paired_n48_reward_means: Sequence[float],
    paired_n48_frozen_reward_means: Sequence[float],
    paired_n48_gain_means: Sequence[float],
    vehicle_unsafe_count: int,
    vehicle_collision_count: int,
    vehicle_out_count: int,
    selected_vehicle_rewards: Sequence[float],
    selected_pretrain_vehicle_rewards: Sequence[float],
    selected_unsafe_count: int,
    selected_collision_count: int,
    selected_out_count: int,
    scenario_selected_rewards: Mapping[str, Sequence[float]],
    scenario_selected_pretrain_rewards: Mapping[str, Sequence[float]],
    scenario_road_penalties: Mapping[str, Sequence[float]],
    scenario_pretrain_road_penalties: Mapping[str, Sequence[float]],
    scenario_road_margins: Mapping[str, Sequence[float]],
    scenario_pretrain_road_margins: Mapping[str, Sequence[float]],
    scenario_selected_unsafe: Mapping[str, int],
    scenario_selected_collisions: Mapping[str, int],
    scenario_selected_outs: Mapping[str, int],
    role_rewards: Sequence[Sequence[float]],
    role_unsafe_counts: Sequence[int],
    role_collision_counts: Sequence[int],
    role_out_counts: Sequence[int],
    performance: Mapping[str, float],
    cache_hits: int,
    cache_misses: int,
) -> dict[str, float]:
    """Aggregate fixed-state vehicle-mode validation into scalar metrics."""

    if not vehicle_rewards:
        raise OnlineGRPOError("fixed validation produced no vehicle-mode rewards")

    metrics: dict[str, float] = {
        "validation/vehicle_reward_mean": float(np.mean(macro_context_vehicle_rewards)),
        "validation/vehicle_reward_flat_mean": float(np.mean(vehicle_rewards)),
        "validation/same_mode_pretrain_reward_mean": float(
            np.mean(macro_context_pretrain_rewards)
        ),
        "validation/vehicle_reward_gain": float(
            np.mean(macro_context_vehicle_rewards)
            - np.mean(macro_context_pretrain_rewards)
        ),
        "validation/paired_n48_vehicle_reward_mean": float(
            np.mean(paired_n48_reward_means)
        ),
        "validation/paired_n48_frozen_reward_mean": float(
            np.mean(paired_n48_frozen_reward_means)
        ),
        "validation/paired_n48_reward_gain": float(np.mean(paired_n48_gain_means)),
        "validation/vehicle_unsafe_count": float(vehicle_unsafe_count),
        "validation/vehicle_collision_count": float(vehicle_collision_count),
        "validation/vehicle_out_of_drivable_count": float(vehicle_out_count),
        "validation/selected_vehicle_reward_mean": float(
            np.mean(selected_vehicle_rewards)
        ),
        "validation/selected_pretrain_vehicle_reward_mean": float(
            np.mean(selected_pretrain_vehicle_rewards)
        ),
        "validation/selected_vehicle_reward_gain": float(
            np.mean(selected_vehicle_rewards)
            - np.mean(selected_pretrain_vehicle_rewards)
        ),
        "validation/selected_vehicle_unsafe_count": float(selected_unsafe_count),
        "validation/selected_vehicle_collision_count": float(selected_collision_count),
        "validation/selected_vehicle_out_of_drivable_count": float(selected_out_count),
    }

    for scenario_key in sorted(scenario_selected_rewards):
        current_selected = scenario_selected_rewards[scenario_key]
        frozen_selected = scenario_selected_pretrain_rewards[scenario_key]
        metrics.update(
            {
                f"validation/{scenario_key}/selected_vehicle_reward_mean": float(
                    np.mean(current_selected)
                ),
                f"validation/{scenario_key}/selected_vehicle_reward_gain": float(
                    np.mean(current_selected) - np.mean(frozen_selected)
                ),
                f"validation/{scenario_key}/road_penalty_mean": float(
                    np.mean(scenario_road_penalties[scenario_key])
                ),
                f"validation/{scenario_key}/pretrain_road_penalty_mean": float(
                    np.mean(scenario_pretrain_road_penalties[scenario_key])
                ),
                f"validation/{scenario_key}/minimum_road_margin_m": float(
                    np.min(scenario_road_margins[scenario_key])
                ),
                f"validation/{scenario_key}/pretrain_minimum_road_margin_m": float(
                    np.min(scenario_pretrain_road_margins[scenario_key])
                ),
                f"validation/{scenario_key}/selected_vehicle_unsafe_count": float(
                    scenario_selected_unsafe[scenario_key]
                ),
                f"validation/{scenario_key}/selected_vehicle_collision_count": float(
                    scenario_selected_collisions[scenario_key]
                ),
                f"validation/{scenario_key}/selected_vehicle_out_of_drivable_count": float(
                    scenario_selected_outs[scenario_key]
                ),
            }
        )

    for role in range(3):
        if not role_rewards[role]:
            raise OnlineGRPOError(
                f"fixed validation produced no valid modes for vehicle {role}"
            )
        metrics.update(
            {
                f"validation/vehicle_{role}_reward_mean": float(
                    np.mean(role_rewards[role])
                ),
                f"validation/vehicle_{role}_unsafe_count": float(
                    role_unsafe_counts[role]
                ),
                f"validation/vehicle_{role}_collision_count": float(
                    role_collision_counts[role]
                ),
                f"validation/vehicle_{role}_out_of_drivable_count": float(
                    role_out_counts[role]
                ),
            }
        )

    metrics.update({str(name): float(value) for name, value in performance.items()})
    metrics["perf/validation/frozen_cache_hits"] = float(cache_hits)
    metrics["perf/validation/frozen_cache_misses"] = float(cache_misses)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise OnlineGRPOError("fixed validation metrics must be finite")
    return metrics

def _validation_reward_comparison_metrics(
    current_validation: Mapping[str, object],
    pretrain_validation: Mapping[str, object],
) -> dict[str, float]:
    """Compare the active vehicle-mode validation against frozen Stage 1."""

    reward_tag = "validation/vehicle_reward_mean"
    if reward_tag not in current_validation:
        raise OnlineGRPOError(
            f"fixed validation is missing required metric {reward_tag}"
        )
    try:
        current_reward = float(current_validation[reward_tag])
        baseline_reward = float(pretrain_validation[reward_tag])
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError(
            "validation current and pretrain rewards must be finite scalars"
        ) from exc
    if not math.isfinite(current_reward) or not math.isfinite(baseline_reward):
        raise OnlineGRPOError(
            "validation current and pretrain rewards must be finite scalars"
        )

    metrics = {
        "validation/fixed_pretrain_vehicle_reward": baseline_reward,
        "validation/fixed_pretrain_vehicle_reward_gain": current_reward
        - baseline_reward,
    }
    comparison_tags = (
        "selected_vehicle_reward_mean",
        "vehicle_collision_count",
        "vehicle_out_of_drivable_count",
        "selected_vehicle_unsafe_count",
        "selected_vehicle_collision_count",
        "selected_vehicle_out_of_drivable_count",
        "S7/selected_vehicle_unsafe_count",
        "S7/selected_vehicle_collision_count",
        "S7/selected_vehicle_out_of_drivable_count",
        "S7/road_penalty_mean",
        "S7/minimum_road_margin_m",
    )
    for suffix in comparison_tags:
        tag = f"validation/{suffix}"
        if tag not in current_validation or tag not in pretrain_validation:
            continue
        current_value = float(current_validation[tag])
        frozen_value = float(pretrain_validation[tag])
        if not math.isfinite(current_value) or not math.isfinite(frozen_value):
            raise OnlineGRPOError(
                f"validation comparison metric {tag} must be finite"
            )
        metrics[f"validation/fixed_pretrain/{suffix}"] = frozen_value
        metrics[f"validation/gain/{suffix}"] = current_value - frozen_value

    aliases = {
        "selected_vehicle_reward_mean": "validation/selected_reward_gain",
        "S7/selected_vehicle_out_of_drivable_count": "validation/S7/out_delta",
        "S7/selected_vehicle_collision_count": "validation/S7/collision_delta",
        "S7/road_penalty_mean": "validation/S7/road_penalty_delta",
        "S7/minimum_road_margin_m": "validation/S7/road_margin_delta",
    }
    for suffix, alias in aliases.items():
        source = f"validation/gain/{suffix}"
        if source in metrics:
            metrics[alias] = metrics[source]

    metrics["validation/vehicle_reward_gain_vs_fixed_pretrain"] = (
        current_reward - baseline_reward
    )
    return metrics


def _validation_vehicle_reward(validation: Mapping[str, object]) -> float:
    """Return the active vehicle-mode validation reward as a finite scalar."""

    reward_tag = "validation/vehicle_reward_mean"
    if reward_tag not in validation:
        raise OnlineGRPOError(
            f"fixed validation is missing required metric {reward_tag}"
        )
    try:
        reward = float(validation[reward_tag])
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError(
            "validation vehicle reward must be a finite scalar"
        ) from exc
    if not math.isfinite(reward):
        raise OnlineGRPOError("validation vehicle reward must be a finite scalar")
    return reward


def _validation_is_safety_eligible(
    validation: Mapping[str, object], pretrain_validation: Mapping[str, object]
) -> bool:
    """Apply frozen-baseline safety constraints in the vehicle-mode reward domain."""

    constrained_tags = (
        "validation/selected_vehicle_unsafe_count",
        "validation/selected_vehicle_collision_count",
        "validation/selected_vehicle_out_of_drivable_count",
        "validation/S7/selected_vehicle_out_of_drivable_count",
    )
    for tag in constrained_tags:
        if tag not in validation or tag not in pretrain_validation:
            return False
        current = float(validation[tag])
        frozen = float(pretrain_validation[tag])
        if not math.isfinite(current) or not math.isfinite(frozen) or current > frozen:
            return False
    return True


def _resume_best_checkpoint_anchor(
    resume_checkpoint: Path, resume_payload: Mapping[str, object]
) -> tuple[Path | None, tuple[float, float] | None, list[dict[str, float]]]:
    """Resolve the safety-constrained best checkpoint and selection history."""

    history = _validated_selection_history(
        resume_payload.get("validation_selection_history")
    )
    raw_best_reward = resume_payload.get("best_validation_reward")
    raw_best_selected = resume_payload.get("best_selected_reward_gain")
    best_sha = resume_payload.get("best_checkpoint_sha256")
    if raw_best_reward is None:
        if raw_best_selected is not None or best_sha is not None:
            raise OnlineGRPOError("resume checkpoint absent best score conflicts")
        return None, None, history

    best_score = float(raw_best_reward), float(raw_best_selected)
    if best_sha is None:
        return Path(resume_checkpoint), best_score, history

    best_path = Path(resume_checkpoint).with_name("best.pt")
    if _checkpoint_file_sha256(best_path) != best_sha:
        raise OnlineGRPOError("resume best checkpoint SHA256 mismatch")
    try:
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OnlineGRPOError(
            f"unable to load resume best checkpoint: {best_path}"
        ) from exc
    if not isinstance(best_payload, Mapping):
        raise OnlineGRPOError("resume best checkpoint must be a mapping")

    binding_fields = (
        "schema_version",
        "format",
        "variant",
        "predecessor_condition",
        "grpo_config",
        "source_stage1_sha256",
        "run_mode",
        "reward_contract_version",
        "reward_contract_sha256",
        "reward_config_sha256",
        "reward_application_contract_sha256",
        "reward_input_domain",
        "training_candidate_domain",
        "environment_action_source",
        "execution_input_domain",
        "best_checkpoint_metric",
        "validation_reward_family",
        "tracking_expansion_enabled",
        "calibration_required",
        "policy_update_contract",
        "policy_update_contract_sha256",
        "rollout_collection_contract",
        "scenario_contract_sha256",
        "trajectory_optimizer_sha256",
    )
    if any(
        best_payload.get(field) != resume_payload.get(field)
        for field in binding_fields
    ):
        raise OnlineGRPOError("resume best checkpoint contract binding mismatch")

    raw_best_reward = best_payload.get("best_validation_reward")
    raw_best_selected = best_payload.get("best_selected_reward_gain")
    if (
        isinstance(raw_best_reward, bool)
        or not isinstance(raw_best_reward, (int, float))
        or not math.isfinite(float(raw_best_reward))
        or isinstance(raw_best_selected, bool)
        or not isinstance(raw_best_selected, (int, float))
        or not math.isfinite(float(raw_best_selected))
    ):
        raise OnlineGRPOError("resume best checkpoint score binding mismatch")
    if (
        best_payload.get("best_checkpoint_sha256") is not None
        or (float(raw_best_reward), float(raw_best_selected)) != best_score
    ):
        raise OnlineGRPOError("resume best checkpoint score binding mismatch")
    return best_path, best_score, history


__all__ = [
    "_finalize_vehicle_mode_validation_metrics",
    "_resume_best_checkpoint_anchor",
    "_validation_is_safety_eligible",
    "_validation_reward_comparison_metrics",
    "_validation_vehicle_reward",
]
