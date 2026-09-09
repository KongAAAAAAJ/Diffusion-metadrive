"""Fixed-state validation using only the active vehicle-mode reward."""
from __future__ import annotations

import dataclasses
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from models.bev_planner import KinematicTrajectoryOptimizer
from models.bev_planner.joint_reward import (
    VehicleModeCounterfactualReward,
    VehicleModePretrainRewardResult,
    VehicleModeRewardConfig,
    VehicleModeRewardResult,
)
from train.train_bev_diffusion_stage1 import planner_forward_from_batch

from .config import OnlineGRPOError
from .rollout import _diffusion_attempt_generators
from .validation_metrics import (
    _finalize_vehicle_mode_validation_metrics,
    _resume_best_checkpoint_anchor,
    _validation_is_safety_eligible,
    _validation_reward_comparison_metrics,
    _validation_vehicle_reward,
)
from .validation_state import (
    _load_grpo_validation_state_bank,
    _replay_fixed_validation_state,
    build_grpo_validation_state_bank,
)

@dataclass(frozen=True)
class _FixedValidationFrozenEntry:
    """Immutable frozen-policy work reused within one training process."""

    all_mode_trajectories: np.ndarray
    selected_trajectory: np.ndarray
    selected_mode: np.ndarray
    pretrain_reward: VehicleModePretrainRewardResult
    paired_candidates: np.ndarray
    paired_reward: VehicleModeRewardResult


@torch.no_grad()
def _fixed_vehicle_mode_validation(
    trainer: object,
    *,
    device: torch.device,
    reward_config: VehicleModeRewardConfig,
    scenarios: Sequence[tuple[str, str]],
    seeds: Sequence[int],
    validation_state_bank: Mapping[
        tuple[tuple[str, str], int], Mapping[str, object]
    ],
    frozen_cache: dict[
        tuple[tuple[str, str], int], _FixedValidationFrozenEntry
    ]
    | None = None,
) -> dict[str, float]:
    """Evaluate fixed states exclusively with the active vehicle-mode reward."""

    if not isinstance(reward_config, VehicleModeRewardConfig):
        raise OnlineGRPOError("validation reward_config must be VehicleModeRewardConfig")
    if reward_config.trajectories_per_mode != trainer.config.trajectories_per_mode:
        raise OnlineGRPOError(
            "validation reward trajectory count must match trainer configuration"
        )

    planner = trainer.planner
    vehicle_backend = VehicleModeCounterfactualReward(reward_config)
    single_vehicle_backend = VehicleModeCounterfactualReward(
        dataclasses.replace(reward_config, trajectories_per_mode=1)
    )
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    cache = {} if frozen_cache is None else frozen_cache

    performance: defaultdict[str, float] = defaultdict(float)
    validation_started = time.perf_counter()
    cache_hits = 0
    cache_misses = 0

    vehicle_rewards: list[float] = []
    macro_context_vehicle_rewards: list[float] = []
    macro_context_pretrain_rewards: list[float] = []
    paired_n48_reward_means: list[float] = []
    paired_n48_frozen_reward_means: list[float] = []
    paired_n48_gain_means: list[float] = []

    vehicle_unsafe_count = 0
    vehicle_collision_count = 0
    vehicle_out_count = 0
    selected_unsafe_count = 0
    selected_collision_count = 0
    selected_out_count = 0

    role_rewards: list[list[float]] = [[], [], []]
    role_unsafe_counts = [0, 0, 0]
    role_collision_counts = [0, 0, 0]
    role_out_counts = [0, 0, 0]

    selected_vehicle_rewards: list[float] = []
    selected_pretrain_vehicle_rewards: list[float] = []
    scenario_selected_rewards: dict[str, list[float]] = defaultdict(list)
    scenario_selected_pretrain_rewards: dict[str, list[float]] = defaultdict(list)
    scenario_road_penalties: dict[str, list[float]] = defaultdict(list)
    scenario_pretrain_road_penalties: dict[str, list[float]] = defaultdict(list)
    scenario_road_margins: dict[str, list[float]] = defaultdict(list)
    scenario_pretrain_road_margins: dict[str, list[float]] = defaultdict(list)
    scenario_selected_unsafe: dict[str, int] = defaultdict(int)
    scenario_selected_collisions: dict[str, int] = defaultdict(int)
    scenario_selected_outs: dict[str, int] = defaultdict(int)

    for scenario_index, scenario in enumerate(scenarios):
        for seed in seeds:
            cache_key = (tuple(scenario), int(seed))
            record = validation_state_bank.get(cache_key)
            if record is None:
                raise OnlineGRPOError(
                    "validation state bank is missing a scenario/seed record"
                )

            replay_started = time.perf_counter()
            env, values, _, execution_mask, batch, noise_bundle = (
                _replay_fixed_validation_state(
                    record,
                    scenario=tuple(scenario),
                    seed=int(seed),
                    device=device,
                    trajectory_optimizer=trajectory_optimizer,
                )
            )
            performance["perf/validation/state_replay_seconds"] += (
                time.perf_counter() - replay_started
            )

            try:
                current_inference_started = time.perf_counter()
                output = planner_forward_from_batch(
                    planner, batch, ddim_noise_bundle=noise_bundle
                )
                selected_modes = (
                    output["selected_mode"][0]
                    .detach()
                    .cpu()
                    .numpy()[None]
                    .astype(np.int64, copy=False)
                )
                current_all_modes = (
                    output["trajectory_candidates"][0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                performance["perf/validation/current_inference_seconds"] += (
                    time.perf_counter() - current_inference_started
                )

                cached = cache.get(cache_key)
                if cached is None:
                    cache_misses += 1
                    frozen_inference_started = time.perf_counter()
                    frozen = trainer.infer_frozen_pretrain_from_inputs(
                        batch, noise_bundle=noise_bundle
                    )
                    frozen_all_modes = (
                        frozen["all_mode_trajectories"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    frozen_argmax = (
                        frozen["selected_trajectory"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    frozen_selected_modes = (
                        frozen["selected_mode"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64, copy=False)
                    )
                    performance["perf/validation/frozen_inference_seconds"] += (
                        time.perf_counter() - frozen_inference_started
                    )
                    pretrain_reward_started = time.perf_counter()
                    geometry_context = vehicle_backend.build_geometry_context(
                        env, values, frozen_argmax
                    )
                    pretrain_local = vehicle_backend.score_pretrain(
                        env,
                        values,
                        frozen_all_modes,
                        frozen_argmax,
                        execution_mask,
                        geometry_context=geometry_context,
                    )
                    performance["perf/validation/pretrain_reward_seconds"] += (
                        time.perf_counter() - pretrain_reward_started
                    )
                else:
                    cache_hits += 1
                    frozen_all_modes = cached.all_mode_trajectories
                    frozen_argmax = cached.selected_trajectory
                    frozen_selected_modes = cached.selected_mode
                    pretrain_local = cached.pretrain_reward
                    geometry_context = vehicle_backend.build_geometry_context(
                        env, values, frozen_argmax
                    )

                paired_initial, paired_transition = _diffusion_attempt_generators(
                    device=device,
                    training_seed=10000019 + 1009 * int(seed),
                    live_state_index=scenario_index,
                    retry_index=0,
                )
                paired_sampling_started = time.perf_counter()
                if cached is None:
                    paired_rollout = trainer.sample_groups(
                        batch,
                        generator=paired_initial,
                        transition_generator=paired_transition,
                        noise_bundle_identity=(
                            10000019 + 1009 * int(seed),
                            int(scenario_index),
                            0,
                        ),
                    )
                    paired_current_candidates = (
                        paired_rollout.candidate_trajectories[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    paired_frozen_candidates = (
                        paired_rollout.frozen_candidate_trajectories[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                else:
                    paired_current_candidates = (
                        trainer.sample_current_groups(
                            batch,
                            generator=paired_initial,
                            transition_generator=paired_transition,
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    paired_frozen_candidates = cached.paired_candidates
                performance["perf/validation/paired_sampling_seconds"] += (
                    time.perf_counter() - paired_sampling_started
                )

                current_reward_started = time.perf_counter()
                paired_current = vehicle_backend.score_candidates(
                    env,
                    values,
                    paired_current_candidates,
                    frozen_argmax,
                    execution_mask,
                    pretrain_local,
                    geometry_context=geometry_context,
                )
                performance["perf/validation/current_n48_reward_seconds"] += (
                    time.perf_counter() - current_reward_started
                )

                if cached is None:
                    frozen_reward_started = time.perf_counter()
                    paired_frozen = vehicle_backend.score_candidates(
                        env,
                        values,
                        paired_frozen_candidates,
                        frozen_argmax,
                        execution_mask,
                        pretrain_local,
                        geometry_context=geometry_context,
                    )
                    performance["perf/validation/frozen_n48_reward_seconds"] += (
                        time.perf_counter() - frozen_reward_started
                    )
                    cached = _FixedValidationFrozenEntry(
                        all_mode_trajectories=np.array(
                            frozen_all_modes, dtype=np.float32, copy=True
                        ),
                        selected_trajectory=np.array(
                            frozen_argmax, dtype=np.float32, copy=True
                        ),
                        selected_mode=np.array(
                            frozen_selected_modes, dtype=np.int64, copy=True
                        ),
                        pretrain_reward=pretrain_local,
                        paired_candidates=np.array(
                            paired_frozen_candidates, dtype=np.float32, copy=True
                        ),
                        paired_reward=paired_frozen,
                    )
                    cache[cache_key] = cached
                else:
                    paired_frozen = cached.paired_reward

                paired_valid = np.asarray(
                    paired_current.valid_mode_mask, dtype=np.bool_
                )
                role_current = [
                    float(paired_current.rewards[role][paired_valid[role]].mean())
                    for role in range(3)
                ]
                role_frozen = [
                    float(paired_frozen.rewards[role][paired_valid[role]].mean())
                    for role in range(3)
                ]
                paired_n48_reward_means.append(float(np.mean(role_current)))
                paired_n48_frozen_reward_means.append(float(np.mean(role_frozen)))
                paired_n48_gain_means.append(
                    float(np.mean(np.asarray(role_current) - np.asarray(role_frozen)))
                )

                n1_reward_started = time.perf_counter()
                current_local = single_vehicle_backend.score_candidates(
                    env,
                    values,
                    current_all_modes[:, :, None],
                    frozen_argmax,
                    execution_mask,
                    pretrain_local,
                    geometry_context=geometry_context,
                    trajectories_per_mode=1,
                )
                performance["perf/validation/current_n1_reward_seconds"] += (
                    time.perf_counter() - n1_reward_started
                )

                valid = np.asarray(current_local.valid_mode_mask, dtype=np.bool_)
                scenario_key = str(scenario[0]).split("_", 1)[0]
                for role in range(3):
                    current_mode = int(selected_modes[0, role])
                    frozen_mode = int(frozen_selected_modes[role])
                    current_selected_reward = float(
                        current_local.rewards[role, current_mode, 0]
                    )
                    frozen_selected_reward = float(
                        pretrain_local.rewards[role, frozen_mode]
                    )
                    selected_vehicle_rewards.append(current_selected_reward)
                    selected_pretrain_vehicle_rewards.append(frozen_selected_reward)
                    scenario_selected_rewards[scenario_key].append(
                        current_selected_reward
                    )
                    scenario_selected_pretrain_rewards[scenario_key].append(
                        frozen_selected_reward
                    )
                    scenario_road_penalties[scenario_key].append(
                        float(current_local.components["road_penalty"][role, current_mode, 0])
                    )
                    scenario_pretrain_road_penalties[scenario_key].append(
                        float(pretrain_local.components["road_penalty"][role, frozen_mode])
                    )
                    scenario_road_margins[scenario_key].append(
                        float(
                            current_local.components["minimum_road_margin_m"][
                                role, current_mode, 0
                            ]
                        )
                    )
                    scenario_pretrain_road_margins[scenario_key].append(
                        float(
                            pretrain_local.components["minimum_road_margin_m"][
                                role, frozen_mode
                            ]
                        )
                    )

                    current_selected_unsafe = bool(
                        current_local.unsafe[role, current_mode, 0]
                    )
                    current_selected_collision = bool(
                        current_local.collision[role, current_mode, 0]
                    )
                    current_selected_out = bool(
                        current_local.out_of_drivable[role, current_mode, 0]
                    )
                    selected_unsafe_count += int(current_selected_unsafe)
                    selected_collision_count += int(current_selected_collision)
                    selected_out_count += int(current_selected_out)
                    scenario_selected_unsafe[scenario_key] += int(
                        current_selected_unsafe
                    )
                    scenario_selected_collisions[scenario_key] += int(
                        current_selected_collision
                    )
                    scenario_selected_outs[scenario_key] += int(current_selected_out)

                per_mode_reward = np.asarray(current_local.rewards).mean(axis=-1)
                per_mode_unsafe = np.asarray(current_local.unsafe)[..., 0]
                per_mode_collision = np.asarray(current_local.collision)[..., 0]
                per_mode_out = np.asarray(current_local.out_of_drivable)[..., 0]

                context_role_rewards = []
                context_role_pretrain_rewards = []
                for role in range(3):
                    role_valid = valid[role]
                    context_role_rewards.append(
                        float(per_mode_reward[role][role_valid].mean())
                    )
                    context_role_pretrain_rewards.append(
                        float(np.asarray(current_local.pretrain_rewards)[role][role_valid].mean())
                    )
                macro_context_vehicle_rewards.append(
                    float(np.mean(context_role_rewards))
                )
                macro_context_pretrain_rewards.append(
                    float(np.mean(context_role_pretrain_rewards))
                )

                vehicle_rewards.extend(per_mode_reward[valid].tolist())
                vehicle_unsafe_count += int(per_mode_unsafe[valid].sum())
                vehicle_collision_count += int(per_mode_collision[valid].sum())
                vehicle_out_count += int(per_mode_out[valid].sum())

                for role in range(3):
                    role_valid = valid[role]
                    role_rewards[role].extend(
                        per_mode_reward[role][role_valid].tolist()
                    )
                    role_unsafe_counts[role] += int(
                        per_mode_unsafe[role][role_valid].sum()
                    )
                    role_collision_counts[role] += int(
                        per_mode_collision[role][role_valid].sum()
                    )
                    role_out_counts[role] += int(
                        per_mode_out[role][role_valid].sum()
                    )
            finally:
                env.close()

    performance["perf/validation/total_seconds"] = (
        time.perf_counter() - validation_started
    )
    return _finalize_vehicle_mode_validation_metrics(
        vehicle_rewards=vehicle_rewards,
        macro_context_vehicle_rewards=macro_context_vehicle_rewards,
        macro_context_pretrain_rewards=macro_context_pretrain_rewards,
        paired_n48_reward_means=paired_n48_reward_means,
        paired_n48_frozen_reward_means=paired_n48_frozen_reward_means,
        paired_n48_gain_means=paired_n48_gain_means,
        vehicle_unsafe_count=vehicle_unsafe_count,
        vehicle_collision_count=vehicle_collision_count,
        vehicle_out_count=vehicle_out_count,
        selected_vehicle_rewards=selected_vehicle_rewards,
        selected_pretrain_vehicle_rewards=selected_pretrain_vehicle_rewards,
        selected_unsafe_count=selected_unsafe_count,
        selected_collision_count=selected_collision_count,
        selected_out_count=selected_out_count,
        scenario_selected_rewards=scenario_selected_rewards,
        scenario_selected_pretrain_rewards=scenario_selected_pretrain_rewards,
        scenario_road_penalties=scenario_road_penalties,
        scenario_pretrain_road_penalties=scenario_pretrain_road_penalties,
        scenario_road_margins=scenario_road_margins,
        scenario_pretrain_road_margins=scenario_pretrain_road_margins,
        scenario_selected_unsafe=scenario_selected_unsafe,
        scenario_selected_collisions=scenario_selected_collisions,
        scenario_selected_outs=scenario_selected_outs,
        role_rewards=role_rewards,
        role_unsafe_counts=role_unsafe_counts,
        role_collision_counts=role_collision_counts,
        role_out_counts=role_out_counts,
        performance=performance,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )

__all__ = [
    "_FixedValidationFrozenEntry",
    "_fixed_vehicle_mode_validation",
    "_load_grpo_validation_state_bank",
    "_resume_best_checkpoint_anchor",
    "_validation_is_safety_eligible",
    "_validation_reward_comparison_metrics",
    "_validation_vehicle_reward",
    "build_grpo_validation_state_bank",
]
