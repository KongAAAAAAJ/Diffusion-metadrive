"""Compare paired single-epoch and clipped multi-epoch GRPO diagnostics.

The input manifest is a JSON object with this exact shape (``run_dir`` may be
absolute or relative to the manifest):

.. code-block:: json

   {
     "format": "stage2_grpo_stability_ab_manifest_v5",
     "source_stage1_sha256": "<64 lowercase hex characters>",
     "paired_seeds": [17, 23, 42],
     "baseline_update_epochs": 1,
     "clipped_update_epochs": 4,
     "trajectories_per_mode": 48,
     "target_accepted_update_states": 100,
     "validation_interval_rollouts": 20,
     "optimizer_contract_version": "stage2_joint_grpo_optimizer_v5",
     "application_contract_version": "stage2_grpo_open_application_v3",
     "rollout_collection_contract_version": "stage2_joint_grpo_persistent_episode_v4",
     "rollout_groups_per_bucket_visit": 10,
     "rollout_start_offset_max_steps": 200,
     "rollout_start_min_remaining_steps": 10,
     "max_sampling_attempts_per_state": 3,
     "max_sampling_attempts_multiplier": 3,
     "clip_epsilon_low": 0.1,
     "clip_epsilon_high": 0.2,
     "scenario_seeds": [17, 23],
     "validation_seeds": [31, 47],
     "runs": [
       {"arm": "baseline", "seed": 17, "run_dir": "run_8"},
       {"arm": "clipped", "seed": 17, "run_dir": "run_9"},
       {"arm": "baseline", "seed": 23, "run_dir": "run_10"},
       {"arm": "clipped", "seed": 23, "run_dir": "run_11"},
       {"arm": "baseline", "seed": 42, "run_dir": "run_12"},
       {"arm": "clipped", "seed": 42, "run_dir": "run_13"}
     ]
   }

``runs`` must contain exactly one baseline and one clipped entry for each of
the three paired seeds (six entries total). The command validates each run's
v10 config/report, exact same-mode dynamic-sampling collection contract, the
complete frozen ``JointGRPOConfig``, and the ``validation/vehicle_reward_gain``
TensorBoard series, then writes ``report.json`` and
``validation_vehicle_reward_gain_ab.png``. Results remain diagnostic-only and do not
authorize formal conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from models.bev_planner.joint_grpo import (
    JointGRPOPolicyUpdateConfig,
    joint_grpo_optimizer_contract,
    joint_grpo_optimizer_contract_sha256,
)
from models.bev_planner.joint_reward import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    VEHICLE_MODE_REWARD_CONTRACT,
    VEHICLE_MODE_REWARD_CONTRACT_SHA256,
)


MANIFEST_FORMAT = "stage2_grpo_stability_ab_manifest_v5"
REPORT_FORMAT = "stage2_grpo_stability_ab_report_v5"
ONLINE_CONFIG_FORMAT = "bev_joint_grpo_online_config_v10"
ONLINE_REPORT_FORMAT = "bev_joint_grpo_online_report_v10"
APPLICATION_CONTRACT_VERSION = str(
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT["version"]
)
OPTIMIZER_CONTRACT_VERSION = "stage2_joint_grpo_optimizer_v5"
VALIDATION_REWARD_GAIN_TAG = "validation/vehicle_reward_gain"
PAIRED_SEEDS = (17, 23, 42)
ARM_UPDATE_EPOCHS = {"baseline": 1, "clipped": 4}
TRAJECTORIES_PER_MODE = 48
TARGET_ACCEPTED_UPDATE_STATES = 100
VALIDATION_INTERVAL_ROLLOUTS = 20
ROLLOUT_GROUPS_PER_BUCKET_VISIT = 10
ROLLOUT_START_OFFSET_MAX_STEPS = 200
ROLLOUT_START_MIN_REMAINING_STEPS = 10
MAX_SAMPLING_ATTEMPTS_PER_STATE = 3
MAX_SAMPLING_ATTEMPTS_MULTIPLIER = 3
ACCEPTED_UPDATE_AXIS = "absolute_accepted_update_state"
CLIP_EPSILON_LOW = 0.1
CLIP_EPSILON_HIGH = 0.2
SCENARIO_SEEDS = (17, 23)
VALIDATION_SEEDS = (31, 47)
SCENARIOS = (
    ("S5_hard_brake_lead", "R1_entry_straight"),
    ("S6_background_merge_in", "R6_mainline_merge_approach"),
    ("S7_ego_merge_from_ramp", "R7_merge_core"),
    ("S8_ego_exit_to_ramp", "R6_exit_to_ramp"),
    ("S9_narrow_channel_negotiation", "R8_narrow_channel"),
)
ROLLOUT_COLLECTION_CONTRACT_VERSION = (
    "stage2_joint_grpo_persistent_episode_v4"
)
EXPECTED_GRPO_CONFIG = {
    "trajectories_per_mode": TRAJECTORIES_PER_MODE,
    "initial_noise_timestep": 8,
    "denoise_steps": 4,
    "eta": 1.0,
    "trajectory_pg_weight": 1.0,
    "bc_weight": 0.1,
    "reference_kl_weight": 0.02,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "advantage_eps": 1e-6,
    "xy_beta_m": 1.0,
    "heading_beta_rad": 0.1,
    "heading_bc_weight": 0.2,
}


class GRPOStabilityComparisonError(ValueError):
    """Raised when a persisted clipped-GRPO A/B input is invalid."""


@dataclass(frozen=True)
class _RunSeries:
    arm: str
    seed: int
    run_dir: Path
    update_steps: np.ndarray
    reward_gain: np.ndarray
    optimizer_steps: int
    accepted_update_states: int
    sampling_attempts: int
    rejected_sampling_attempts: int
    exhausted_states: int
    baseline_execution_steps: int
    zero_signal_epochs: int
    wall_time_seconds: float

    @property
    def adjacent_delta_std(self) -> float:
        return float(np.std(np.diff(self.reward_gain), ddof=0))

    @property
    def final_reward_gain(self) -> float:
        return float(self.reward_gain[-1])


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GRPOStabilityComparisonError(
            f"unable to read {label}: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise GRPOStabilityComparisonError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise GRPOStabilityComparisonError(
            f"{label} fields mismatch: expected {sorted(expected)}"
        )


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GRPOStabilityComparisonError(f"{label} must be a positive integer")
    return int(value)


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GRPOStabilityComparisonError(
            f"{label} must be a non-negative integer"
        )
    return int(value)


def _finite_float(value: object, *, label: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive " if positive else ""
        raise GRPOStabilityComparisonError(
            f"{label} must be a {qualifier}finite scalar"
        )
    return float(value)


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
    ):
        raise GRPOStabilityComparisonError(f"{label} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GRPOStabilityComparisonError(
            f"{label} must be a SHA256 digest"
        ) from exc
    return value


def _validate_grpo_config(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GRPOStabilityComparisonError(f"{label} must be an object")
    _require_exact_keys(value, set(EXPECTED_GRPO_CONFIG), label=label)
    for name, expected in EXPECTED_GRPO_CONFIG.items():
        actual = value.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise GRPOStabilityComparisonError(
                f"{label} {name} mismatch: expected {expected!r}"
            )
    return {str(name): item for name, item in value.items()}


def _expected_rollout_collection_contract(
    environment_steps_per_episode: int,
) -> dict[str, object]:
    return {
        "version": ROLLOUT_COLLECTION_CONTRACT_VERSION,
        "budget_unit": "accepted_update_state",
        "attempt_budget_unit": "same_live_state_noise_resample",
        "pretrain_baseline_domain": "raw_tau_d",
        "pretrain_inference_per_live_state": 1,
        "comparison_unit": "vehicle_mode",
        "trajectories_per_mode": TRAJECTORIES_PER_MODE,
        "advantage_normalization": "within_vehicle_mode_centered_rms",
        "active_mode_gate": (
            "hard_valid_and_optimizer_executable_and_max_reward_gte_"
            "same_mode_frozen_pretrain_reward"
        ),
        "max_sampling_attempts_per_state": MAX_SAMPLING_ATTEMPTS_PER_STATE,
        "max_sampling_attempts_multiplier": (
            MAX_SAMPLING_ATTEMPTS_MULTIPLIER
        ),
        "retry_scope": "same_live_state",
        "partial_active_policy": "merge_all_active_modes_without_backfill",
        "retry_condition": "all_vehicle_modes_inactive",
        "sampled_candidate_execution": False,
        "environment_action": "cached_frozen_stage1_argmax_only",
        "bucket_order": "ordered_scenario_then_seed",
        "bucket_target_assignment": (
            "balanced_floor_with_remainder_to_lower_bucket_indices"
        ),
        "rollout_groups_per_bucket_visit": ROLLOUT_GROUPS_PER_BUCKET_VISIT,
        "environment_steps_per_episode": environment_steps_per_episode,
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
        "baseline_environment_steps_per_live_state": 1,
        "episode_reuse": "persistent_within_bucket_visit",
        "interrupted_visit_resume": "same_bucket_same_visit_progress",
        "checkpoint_boundary": "closed_environment_only",
        "active_environment_serialized": False,
        "rule_maker_commitment_scope": "live_environment_episode",
        "rollout_start_ready_gate": (
            "history_ready_and_primary_scenario_ready"
        ),
        "rollout_start_offset_distribution": "inclusive_uniform_integer",
        "rollout_start_generator": "shared_training_torch_generator",
        "rollout_start_offset_max_steps": ROLLOUT_START_OFFSET_MAX_STEPS,
        "rollout_start_min_remaining_steps": (
            ROLLOUT_START_MIN_REMAINING_STEPS
        ),
        "rollout_start_rng_draws": (
            "exactly_one_per_feasible_live_training_episode_including_zero_upper_bound"
        ),
        "rollout_start_upper_bound": (
            "min(configured_max,episode_cap_minus_t_ready_minus_min_remaining)"
        ),
        "scenario_window_close": {
            "S5_hard_brake_lead": (
                "conflict_evidence.formation_recovered_after_hazard"
            ),
            "S6_background_merge_in": (
                "conflict_evidence.formation_recovered_after_merge"
            ),
            "S7_ego_merge_from_ramp": (
                "route_completion.all_agents_entered_mainline_and_"
                "conflict_evidence.formation_recovered_after_merge"
            ),
            "S8_ego_exit_to_ramp": (
                "route_completion.all_agents_continued_on_exit_ramp_and_"
                "conflict_evidence.formation_recovered_on_ramp"
            ),
            "S9_narrow_channel_negotiation": (
                "route_completion.all_agents_returned_to_original_lane_and_"
                "conflict_evidence.formation_recovered_after_return"
            ),
        },
        "late_target_retry": (
            "same_bucket_same_visit_with_temporary_observed_last_open_upper_bound"
        ),
        "partial_visit_policy": (
            "retain_completed_rollouts_and_updates_then_new_episode_fresh_offset"
        ),
        "validation_start": "earliest_ready_without_random_offset",
    }


def _validate_rollout_collection_contract(
    value: object,
    *,
    expected: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or dict(value) != dict(expected)
    ):
        raise GRPOStabilityComparisonError(
            f"{label} rollout collection contract mismatch"
        )
    return {str(name): item for name, item in value.items()}


def _load_reward_gain(tb_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    if VALIDATION_REWARD_GAIN_TAG not in accumulator.Tags().get("scalars", ()):
        raise GRPOStabilityComparisonError(
            f"TensorBoard tag {VALIDATION_REWARD_GAIN_TAG!r} was not found"
        )
    events = accumulator.Scalars(VALIDATION_REWARD_GAIN_TAG)
    if not events:
        raise GRPOStabilityComparisonError(
            f"TensorBoard tag {VALIDATION_REWARD_GAIN_TAG!r} contains no events"
        )
    records: list[tuple[int, float]] = []
    observed_steps: set[int] = set()
    for event in events:
        step = int(event.step)
        if step in observed_steps:
            raise GRPOStabilityComparisonError(
                f"TensorBoard tag {VALIDATION_REWARD_GAIN_TAG!r} "
                f"has duplicate accepted-update step {step}"
            )
        observed_steps.add(step)
        value = float(event.value)
        if not math.isfinite(value):
            raise GRPOStabilityComparisonError(
                f"TensorBoard tag {VALIDATION_REWARD_GAIN_TAG!r} "
                f"step {step} must be finite"
            )
        records.append((step, value))
    records.sort(key=lambda item: item[0])
    steps = np.asarray([item[0] for item in records], dtype=np.int64)
    values = np.asarray([item[1] for item in records], dtype=np.float64)
    expected = np.arange(
        VALIDATION_INTERVAL_ROLLOUTS,
        TARGET_ACCEPTED_UPDATE_STATES + 1,
        VALIDATION_INTERVAL_ROLLOUTS,
        dtype=np.int64,
    )
    if not np.array_equal(steps, expected):
        raise GRPOStabilityComparisonError(
            "validation/vehicle_reward_gain must use accepted-update steps "
            f"{expected.tolist()}, got {steps.tolist()}"
        )
    return steps, values


def _validate_manifest(path: Path) -> tuple[str, tuple[Mapping[str, object], ...]]:
    manifest = _load_json(path, label="GRPO stability A/B manifest")
    _require_exact_keys(
        manifest,
        {
            "format",
            "source_stage1_sha256",
            "paired_seeds",
            "baseline_update_epochs",
            "clipped_update_epochs",
            "trajectories_per_mode",
            "target_accepted_update_states",
            "validation_interval_rollouts",
            "optimizer_contract_version",
            "application_contract_version",
            "rollout_collection_contract_version",
            "rollout_groups_per_bucket_visit",
            "rollout_start_offset_max_steps",
            "rollout_start_min_remaining_steps",
            "max_sampling_attempts_per_state",
            "max_sampling_attempts_multiplier",
            "clip_epsilon_low",
            "clip_epsilon_high",
            "scenario_seeds",
            "validation_seeds",
            "runs",
        },
        label="GRPO stability A/B manifest",
    )
    expected_values = {
        "format": MANIFEST_FORMAT,
        "paired_seeds": list(PAIRED_SEEDS),
        "baseline_update_epochs": ARM_UPDATE_EPOCHS["baseline"],
        "clipped_update_epochs": ARM_UPDATE_EPOCHS["clipped"],
        "trajectories_per_mode": TRAJECTORIES_PER_MODE,
        "target_accepted_update_states": TARGET_ACCEPTED_UPDATE_STATES,
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
        "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
        "application_contract_version": APPLICATION_CONTRACT_VERSION,
        "rollout_collection_contract_version": (
            ROLLOUT_COLLECTION_CONTRACT_VERSION
        ),
        "rollout_groups_per_bucket_visit": ROLLOUT_GROUPS_PER_BUCKET_VISIT,
        "rollout_start_offset_max_steps": ROLLOUT_START_OFFSET_MAX_STEPS,
        "rollout_start_min_remaining_steps": (
            ROLLOUT_START_MIN_REMAINING_STEPS
        ),
        "max_sampling_attempts_per_state": MAX_SAMPLING_ATTEMPTS_PER_STATE,
        "max_sampling_attempts_multiplier": (
            MAX_SAMPLING_ATTEMPTS_MULTIPLIER
        ),
        "clip_epsilon_low": CLIP_EPSILON_LOW,
        "clip_epsilon_high": CLIP_EPSILON_HIGH,
        "scenario_seeds": list(SCENARIO_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
    }
    for name, expected in expected_values.items():
        actual = manifest.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise GRPOStabilityComparisonError(f"manifest {name} mismatch")
    source_sha = _sha256(
        manifest.get("source_stage1_sha256"),
        label="manifest source_stage1_sha256",
    )
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 6:
        raise GRPOStabilityComparisonError("manifest runs must contain six entries")
    runs: list[Mapping[str, object]] = []
    observed: set[tuple[str, int]] = set()
    for index, value in enumerate(raw_runs):
        if not isinstance(value, Mapping):
            raise GRPOStabilityComparisonError(
                f"manifest runs[{index}] must be an object"
            )
        _require_exact_keys(value, {"arm", "seed", "run_dir"}, label=f"runs[{index}]")
        arm = value.get("arm")
        seed = value.get("seed")
        run_dir = value.get("run_dir")
        if not isinstance(arm, str) or arm not in ARM_UPDATE_EPOCHS:
            raise GRPOStabilityComparisonError(f"runs[{index}].arm is invalid")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in PAIRED_SEEDS
        ):
            raise GRPOStabilityComparisonError(f"runs[{index}].seed is invalid")
        if not isinstance(run_dir, str) or not run_dir:
            raise GRPOStabilityComparisonError(
                f"runs[{index}].run_dir must be a non-empty string"
            )
        identity = (str(arm), int(seed))
        if identity in observed:
            raise GRPOStabilityComparisonError(
                f"manifest contains duplicate run {identity}"
            )
        observed.add(identity)
        runs.append(value)
    expected_identities = {
        (arm, seed) for arm in ARM_UPDATE_EPOCHS for seed in PAIRED_SEEDS
    }
    if observed != expected_identities:
        raise GRPOStabilityComparisonError(
            "manifest must contain one baseline and clipped run for each paired seed"
        )
    return source_sha, tuple(runs)


def _run_binding(config: Mapping[str, object]) -> dict[str, object]:
    online = config.get("online_config")
    if not isinstance(online, Mapping):
        raise GRPOStabilityComparisonError("run config online_config is invalid")
    policy_update = config.get("policy_update_contract")
    if not isinstance(policy_update, Mapping):
        raise GRPOStabilityComparisonError(
            "run config policy_update_contract is invalid"
        )
    return {
        "grpo_config": config.get("grpo_config"),
        "source_stage1_sha256": config.get("source_stage1_sha256"),
        "reward_contract_version": config.get("reward_contract_version"),
        "reward_contract_sha256": config.get("reward_contract_sha256"),
        "reward_config_sha256": config.get("reward_config_sha256"),
        "reward_application_contract_sha256": config.get(
            "reward_application_contract_sha256"
        ),
        "reward_application_contract": config.get(
            "reward_application_contract"
        ),
        "trajectory_optimizer_sha256": config.get(
            "trajectory_optimizer_sha256"
        ),
        "scenario_contract_sha256": config.get("scenario_contract_sha256"),
        "policy_update_contract_common": {
            str(name): value
            for name, value in policy_update.items()
            if name != "update_epochs"
        },
        "rollout_collection_contract": config.get(
            "rollout_collection_contract"
        ),
        "device": online.get("device"),
        "scenarios": online.get("scenarios"),
        "scenario_seeds": online.get("scenario_seeds"),
        "environment_steps_per_episode": online.get(
            "environment_steps_per_episode"
        ),
        "trajectories_per_mode": online.get("trajectories_per_mode"),
        "total_rollout_groups": online.get("total_rollout_groups"),
        "clip_epsilon_low": online.get("clip_epsilon_low"),
        "clip_epsilon_high": online.get("clip_epsilon_high"),
        "validation_interval_rollouts": online.get(
            "validation_interval_rollouts"
        ),
        "rollout_groups_per_bucket_visit": online.get(
            "rollout_groups_per_bucket_visit"
        ),
        "rollout_start_offset_max_steps": online.get(
            "rollout_start_offset_max_steps"
        ),
        "rollout_start_min_remaining_steps": online.get(
            "rollout_start_min_remaining_steps"
        ),
        "max_sampling_attempts_per_state": online.get(
            "max_sampling_attempts_per_state"
        ),
        "max_sampling_attempts_multiplier": online.get(
            "max_sampling_attempts_multiplier"
        ),
        "environment_action_source": config.get("environment_action_source"),
        "best_checkpoint_metric": config.get("best_checkpoint_metric"),
        "training_candidate_domain": config.get("training_candidate_domain"),
        "joint_reward_role": config.get("joint_reward_role"),
    }


def _validate_training_bucket_counters(
    value: object,
    *,
    arm: str,
    seed: int,
    update_epochs: int,
    optimizer_steps: int,
    environment_episode_count: int,
    sampling_attempts: int,
    rejected_sampling_attempts: int,
    exhausted_states: int,
    baseline_execution_steps: int,
    zero_signal_epochs: int,
) -> None:
    label = f"{arm}/{seed} training_bucket_counters"
    expected_buckets = tuple(
        (scenario, scenario_seed)
        for scenario in SCENARIOS
        for scenario_seed in SCENARIO_SEEDS
    )
    if not isinstance(value, list) or len(value) != len(expected_buckets):
        raise GRPOStabilityComparisonError(
            f"{label} must contain {len(expected_buckets)} ordered buckets"
        )
    base_target, remainder = divmod(
        TARGET_ACCEPTED_UPDATE_STATES, len(expected_buckets)
    )
    accepted_total = 0
    sampling_total = 0
    rejected_total = 0
    exhausted_total = 0
    baseline_total = 0
    zero_signal_total = 0
    target_total = 0
    episode_total = 0
    optimizer_total = 0
    required_fields = {
        "scenario",
        "route",
        "seed",
        "target_accepted_updates",
        "accepted_update_states",
        "sampling_attempts",
        "rejected_sampling_attempts",
        "exhausted_states",
        "baseline_execution_steps",
        "zero_signal_epochs",
        "environment_episodes",
        "optimizer_steps",
    }
    for index, (counter, (scenario, scenario_seed)) in enumerate(
        zip(value, expected_buckets)
    ):
        counter_label = f"{label}[{index}]"
        if not isinstance(counter, Mapping) or not required_fields.issubset(counter):
            raise GRPOStabilityComparisonError(
                f"{counter_label} is missing required fields"
            )
        expected_target = base_target + int(index < remainder)
        if (
            counter.get("scenario") != scenario[0]
            or counter.get("route") != scenario[1]
            or counter.get("seed") != scenario_seed
        ):
            raise GRPOStabilityComparisonError(
                f"{counter_label} bucket identity mismatch"
            )
        target = _positive_int(
            counter.get("target_accepted_updates"),
            label=f"{counter_label}.target_accepted_updates",
        )
        accepted = _positive_int(
            counter.get("accepted_update_states"),
            label=f"{counter_label}.accepted_update_states",
        )
        sampled = _positive_int(
            counter.get("sampling_attempts"),
            label=f"{counter_label}.sampling_attempts",
        )
        rejected = _non_negative_int(
            counter.get("rejected_sampling_attempts"),
            label=f"{counter_label}.rejected_sampling_attempts",
        )
        exhausted = _non_negative_int(
            counter.get("exhausted_states"),
            label=f"{counter_label}.exhausted_states",
        )
        baseline_steps = _positive_int(
            counter.get("baseline_execution_steps"),
            label=f"{counter_label}.baseline_execution_steps",
        )
        zero_signal = _non_negative_int(
            counter.get("zero_signal_epochs"),
            label=f"{counter_label}.zero_signal_epochs",
        )
        episodes = _positive_int(
            counter.get("environment_episodes"),
            label=f"{counter_label}.environment_episodes",
        )
        bucket_optimizer_steps = _non_negative_int(
            counter.get("optimizer_steps"),
            label=f"{counter_label}.optimizer_steps",
        )
        if target != expected_target:
            raise GRPOStabilityComparisonError(
                f"{counter_label} target_accepted_updates mismatch"
            )
        if accepted != target:
            raise GRPOStabilityComparisonError(
                f"{counter_label} accepted_update_states must equal "
                "target_accepted_updates"
            )
        if sampled != accepted + rejected:
            raise GRPOStabilityComparisonError(
                f"{counter_label} sampling_attempts must equal accepted "
                "updates plus rejected sampling attempts"
            )
        if baseline_steps != accepted + exhausted:
            raise GRPOStabilityComparisonError(
                f"{counter_label} baseline_execution_steps must equal accepted "
                "updates plus exhausted states"
            )
        if rejected < exhausted * MAX_SAMPLING_ATTEMPTS_PER_STATE:
            raise GRPOStabilityComparisonError(
                f"{counter_label} exhausted states exceed retry evidence"
            )
        if zero_signal > accepted * update_epochs:
            raise GRPOStabilityComparisonError(
                f"{counter_label} zero_signal_epochs exceeds update epochs"
            )
        if bucket_optimizer_steps != accepted * update_epochs - zero_signal:
            raise GRPOStabilityComparisonError(
                f"{counter_label} optimizer_steps mismatch"
            )
        target_total += target
        accepted_total += accepted
        sampling_total += sampled
        rejected_total += rejected
        exhausted_total += exhausted
        baseline_total += baseline_steps
        zero_signal_total += zero_signal
        episode_total += episodes
        optimizer_total += bucket_optimizer_steps
    if (
        target_total != TARGET_ACCEPTED_UPDATE_STATES
        or accepted_total != TARGET_ACCEPTED_UPDATE_STATES
    ):
        raise GRPOStabilityComparisonError(f"{label} update-state totals mismatch")
    if (
        sampling_total != sampling_attempts
        or rejected_total != rejected_sampling_attempts
        or exhausted_total != exhausted_states
        or baseline_total != baseline_execution_steps
        or zero_signal_total != zero_signal_epochs
    ):
        raise GRPOStabilityComparisonError(
            f"{label} dynamic-sampling totals mismatch"
        )
    if episode_total != environment_episode_count:
        raise GRPOStabilityComparisonError(
            f"{label} environment episode total mismatch"
        )
    if optimizer_total != optimizer_steps:
        raise GRPOStabilityComparisonError(
            f"{label} optimizer-step total mismatch"
        )


def _validate_rollout_start_diagnostics(
    value: object,
    *,
    arm: str,
    seed: int,
) -> None:
    label = f"{arm}/{seed} rollout_start_diagnostics_this_run"
    if not isinstance(value, Mapping):
        raise GRPOStabilityComparisonError(f"{label} must be an object")
    _require_exact_keys(
        value,
        {
            "attempt_count",
            "accepted_count",
            "rejected_count",
            "offset_min_steps",
            "offset_mean_steps",
            "offset_max_steps",
        },
        label=label,
    )
    attempt_count = _non_negative_int(
        value.get("attempt_count"), label=f"{label}.attempt_count"
    )
    accepted_count = _positive_int(
        value.get("accepted_count"), label=f"{label}.accepted_count"
    )
    rejected_count = _non_negative_int(
        value.get("rejected_count"), label=f"{label}.rejected_count"
    )
    if attempt_count != accepted_count + rejected_count:
        raise GRPOStabilityComparisonError(
            f"{label} attempt_count must equal accepted_count plus rejected_count"
        )
    offset_min = _finite_float(
        value.get("offset_min_steps"), label=f"{label}.offset_min_steps"
    )
    offset_mean = _finite_float(
        value.get("offset_mean_steps"), label=f"{label}.offset_mean_steps"
    )
    offset_max = _finite_float(
        value.get("offset_max_steps"), label=f"{label}.offset_max_steps"
    )
    if not (
        0.0
        <= offset_min
        <= offset_mean
        <= offset_max
        <= float(ROLLOUT_START_OFFSET_MAX_STEPS)
    ):
        raise GRPOStabilityComparisonError(
            f"{label} offset statistics must satisfy "
            "0 <= min <= mean <= max <= rollout-start maximum"
        )


def _load_run(
    manifest_path: Path,
    entry: Mapping[str, object],
    *,
    source_sha: str,
    expected_binding: Mapping[str, object] | None,
) -> tuple[_RunSeries, dict[str, object]]:
    arm = str(entry["arm"])
    seed = int(entry["seed"])
    run_dir = Path(str(entry["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = manifest_path.parent / run_dir
    run_dir = run_dir.resolve()
    config = _load_json(run_dir / "config.json", label=f"{arm}/{seed} config")
    report = _load_json(run_dir / "report.json", label=f"{arm}/{seed} report")
    if config.get("format") != ONLINE_CONFIG_FORMAT:
        raise GRPOStabilityComparisonError(f"{arm}/{seed} config format mismatch")
    if report.get("format") != ONLINE_REPORT_FORMAT:
        raise GRPOStabilityComparisonError(f"{arm}/{seed} report format mismatch")
    config_grpo = _validate_grpo_config(
        config.get("grpo_config"), label=f"{arm}/{seed} config grpo_config"
    )
    report_grpo = _validate_grpo_config(
        report.get("grpo_config"), label=f"{arm}/{seed} report grpo_config"
    )
    if report_grpo != config_grpo:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} config/report grpo_config mismatch"
        )
    online = config.get("online_config")
    if not isinstance(online, Mapping):
        raise GRPOStabilityComparisonError(f"{arm}/{seed} online_config is invalid")
    expected_online = {
        "device": "cuda",
        "seed": seed,
        "trajectories_per_mode": TRAJECTORIES_PER_MODE,
        "total_rollout_groups": TARGET_ACCEPTED_UPDATE_STATES,
        "update_epochs": ARM_UPDATE_EPOCHS[arm],
        "clip_epsilon_low": CLIP_EPSILON_LOW,
        "clip_epsilon_high": CLIP_EPSILON_HIGH,
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
        "rollout_groups_per_bucket_visit": ROLLOUT_GROUPS_PER_BUCKET_VISIT,
        "rollout_start_offset_max_steps": ROLLOUT_START_OFFSET_MAX_STEPS,
        "rollout_start_min_remaining_steps": ROLLOUT_START_MIN_REMAINING_STEPS,
        "max_sampling_attempts_per_state": MAX_SAMPLING_ATTEMPTS_PER_STATE,
        "max_sampling_attempts_multiplier": (
            MAX_SAMPLING_ATTEMPTS_MULTIPLIER
        ),
        "scenarios": [list(value) for value in SCENARIOS],
        "scenario_seeds": list(SCENARIO_SEEDS),
        "resume_checkpoint": None,
    }
    for name, expected in expected_online.items():
        actual = online.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise GRPOStabilityComparisonError(
                f"{arm}/{seed} online_config {name} mismatch"
            )
    environment_steps_per_episode = _positive_int(
        online.get("environment_steps_per_episode"),
        label=f"{arm}/{seed} online_config environment_steps_per_episode",
    )
    expected_collection_contract = _expected_rollout_collection_contract(
        environment_steps_per_episode
    )
    _validate_rollout_collection_contract(
        config.get("rollout_collection_contract"),
        expected=expected_collection_contract,
        label=f"{arm}/{seed} config",
    )
    _validate_rollout_collection_contract(
        report.get("rollout_collection_contract"),
        expected=expected_collection_contract,
        label=f"{arm}/{seed} report",
    )
    if config.get("source_stage1_sha256") != source_sha:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} source Stage1 checkpoint mismatch"
        )
    config_reward_contract = config.get("reward_contract")
    if (
        not isinstance(config_reward_contract, Mapping)
        or dict(config_reward_contract) != VEHICLE_MODE_REWARD_CONTRACT
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} config vehicle-mode reward contract mismatch"
        )
    for artifact_name, artifact in (("config", config), ("report", report)):
        if (
            artifact.get("reward_contract_version")
            != VEHICLE_MODE_REWARD_CONTRACT["version"]
            or artifact.get("reward_contract_sha256")
            != VEHICLE_MODE_REWARD_CONTRACT_SHA256
        ):
            raise GRPOStabilityComparisonError(
                f"{arm}/{seed} {artifact_name} vehicle-mode reward contract mismatch"
            )
        if (
            artifact.get("training_candidate_domain")
            != "tau_d_all_vehicle_modes"
            or artifact.get("environment_action_source")
            != "cached_frozen_stage1_argmax"
            or artifact.get("best_checkpoint_metric")
            != "validation/vehicle_reward_mean"
            or artifact.get("joint_reward_role")
            != "historical_and_final_evaluation_only"
        ):
            raise GRPOStabilityComparisonError(
                f"{arm}/{seed} {artifact_name} vehicle-mode application fields mismatch"
            )
    for artifact_name, artifact in (("config", config), ("report", report)):
        application_contract = artifact.get("reward_application_contract")
        if (
            not isinstance(application_contract, Mapping)
            or dict(application_contract)
            != GRPO_OPEN_REWARD_APPLICATION_CONTRACT
            or artifact.get("reward_application_contract_sha256")
            != GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ):
            raise GRPOStabilityComparisonError(
                f"{arm}/{seed} {artifact_name} application contract mismatch"
            )
    expected_policy_update = JointGRPOPolicyUpdateConfig(
        update_epochs=ARM_UPDATE_EPOCHS[arm],
        clip_epsilon_low=CLIP_EPSILON_LOW,
        clip_epsilon_high=CLIP_EPSILON_HIGH,
    )
    expected_optimizer_contract = joint_grpo_optimizer_contract(
        expected_policy_update
    )
    expected_optimizer_sha256 = joint_grpo_optimizer_contract_sha256(
        expected_policy_update
    )
    for artifact_name, artifact in (("config", config), ("report", report)):
        contract = artifact.get("policy_update_contract")
        if (
            not isinstance(contract, Mapping)
            or dict(contract) != expected_optimizer_contract
            or artifact.get("policy_update_contract_sha256")
            != expected_optimizer_sha256
        ):
            raise GRPOStabilityComparisonError(
                f"{arm}/{seed} {artifact_name} policy update contract mismatch"
            )
    binding = _run_binding(config)
    if expected_binding is not None and binding != dict(expected_binding):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} fairness binding differs from the other runs"
        )

    if report.get("diagnostic_only") is not True or report.get(
        "eligible_for_formal_training"
    ) is not False:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} must remain diagnostic-only"
        )
    if report.get("training_status") != "complete":
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} training did not complete"
        )
    if report.get("checkpoint_round_trip") is not True:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} checkpoint round-trip did not pass"
        )
    training_plots = report.get("training_plots")
    if not isinstance(training_plots, Mapping) or any(
        training_plots.get(field) != ACCEPTED_UPDATE_AXIS
        for field in (
            "reward_x_axis",
            "validation_reward_x_axis",
            "optimizer_x_axis",
        )
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} training plot axes must use accepted update states"
        )
    accepted = _positive_int(
        report.get("accepted_update_states"),
        label=f"{arm}/{seed} accepted_update_states",
    )
    accepted_this_run = _positive_int(
        report.get("accepted_update_states_this_run"),
        label=f"{arm}/{seed} accepted_update_states_this_run",
    )
    optimizer_steps = _non_negative_int(
        report.get("optimizer_steps"), label=f"{arm}/{seed} optimizer_steps"
    )
    target_accepted = _positive_int(
        report.get("target_accepted_update_states"),
        label=f"{arm}/{seed} target_accepted_update_states",
    )
    sampling = _positive_int(
        report.get("sampling_attempts"),
        label=f"{arm}/{seed} sampling_attempts",
    )
    sampling_this_run = _positive_int(
        report.get("sampling_attempts_this_run"),
        label=f"{arm}/{seed} sampling_attempts_this_run",
    )
    max_sampling = _positive_int(
        report.get("max_sampling_attempts"),
        label=f"{arm}/{seed} max_sampling_attempts",
    )
    rejected = _non_negative_int(
        report.get("rejected_sampling_attempts"),
        label=f"{arm}/{seed} rejected_sampling_attempts",
    )
    exhausted = _non_negative_int(
        report.get("exhausted_states"),
        label=f"{arm}/{seed} exhausted_states",
    )
    baseline_steps = _positive_int(
        report.get("baseline_execution_steps"),
        label=f"{arm}/{seed} baseline_execution_steps",
    )
    zero_signal = _non_negative_int(
        report.get("zero_signal_epochs"),
        label=f"{arm}/{seed} zero_signal_epochs",
    )
    optimizer_steps_this_run = _non_negative_int(
        report.get("optimizer_steps_this_run"),
        label=f"{arm}/{seed} optimizer_steps_this_run",
    )
    if (
        accepted != TARGET_ACCEPTED_UPDATE_STATES
        or target_accepted != TARGET_ACCEPTED_UPDATE_STATES
        or accepted_this_run != TARGET_ACCEPTED_UPDATE_STATES
        or sampling_this_run != sampling
        or max_sampling
        != TARGET_ACCEPTED_UPDATE_STATES
        * MAX_SAMPLING_ATTEMPTS_MULTIPLIER
        or sampling > max_sampling
        or sampling != accepted + rejected
        or baseline_steps != accepted + exhausted
        or rejected < exhausted * MAX_SAMPLING_ATTEMPTS_PER_STATE
        or zero_signal > accepted * ARM_UPDATE_EPOCHS[arm]
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} dynamic-sampling counters mismatch"
        )
    expected_optimizer_steps = (
        accepted * ARM_UPDATE_EPOCHS[arm] - zero_signal
    )
    if (
        optimizer_steps != expected_optimizer_steps
        or optimizer_steps_this_run != expected_optimizer_steps
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} optimizer-step count mismatch"
        )
    if report.get("validation_seeds") != list(VALIDATION_SEEDS):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} validation seeds mismatch"
        )
    environment_steps = _positive_int(
        report.get("environment_steps"),
        label=f"{arm}/{seed} environment_steps",
    )
    environment_episode_count = _positive_int(
        report.get("environment_episode_count"),
        label=f"{arm}/{seed} environment_episode_count",
    )
    if environment_episode_count > environment_steps:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} environment_episode_count exceeds environment_steps"
        )
    warmup_environment_steps = _non_negative_int(
        report.get("warmup_environment_steps"),
        label=f"{arm}/{seed} warmup_environment_steps",
    )
    if warmup_environment_steps != environment_steps - baseline_steps:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} warmup environment-step count mismatch"
        )
    _validate_rollout_start_diagnostics(
        report.get("rollout_start_diagnostics_this_run"),
        arm=arm,
        seed=seed,
    )
    _validate_training_bucket_counters(
        report.get("training_bucket_counters"),
        arm=arm,
        seed=seed,
        update_epochs=ARM_UPDATE_EPOCHS[arm],
        optimizer_steps=optimizer_steps,
        environment_episode_count=environment_episode_count,
        sampling_attempts=sampling,
        rejected_sampling_attempts=rejected,
        exhausted_states=exhausted,
        baseline_execution_steps=baseline_steps,
        zero_signal_epochs=zero_signal,
    )
    wall_time_seconds = _finite_float(
        report.get("wall_time_seconds"),
        label=f"{arm}/{seed} wall_time_seconds",
        positive=True,
    )
    steps, values = _load_reward_gain(run_dir / "tb")
    return (
        _RunSeries(
            arm=arm,
            seed=seed,
            run_dir=run_dir,
            update_steps=steps,
            reward_gain=values,
            optimizer_steps=optimizer_steps,
            accepted_update_states=accepted,
            sampling_attempts=sampling,
            rejected_sampling_attempts=rejected,
            exhausted_states=exhausted,
            baseline_execution_steps=baseline_steps,
            zero_signal_epochs=zero_signal,
            wall_time_seconds=wall_time_seconds,
        ),
        binding,
    )


def _render_comparison_plot(
    runs: Mapping[tuple[str, int], _RunSeries], output_path: Path
) -> Path:
    fig, (gain_ax, stability_ax) = plt.subplots(2, 1, figsize=(10.0, 8.4))
    colors = {"baseline": "#4C78A8", "clipped": "#E45756"}
    for arm in ("baseline", "clipped"):
        arm_runs = [runs[(arm, seed)] for seed in PAIRED_SEEDS]
        stacked = np.stack([run.reward_gain for run in arm_runs], axis=0)
        for run in arm_runs:
            gain_ax.plot(
                run.update_steps,
                run.reward_gain,
                color=colors[arm],
                linewidth=1.0,
                alpha=0.32,
                label=f"{arm} seed {run.seed}",
            )
        gain_ax.plot(
            arm_runs[0].update_steps,
            stacked.mean(axis=0),
            color=colors[arm],
            linewidth=2.5,
            marker="o",
            label=f"{arm} mean",
        )
    gain_ax.axhline(0.0, color="#777777", linewidth=0.8)
    gain_ax.set_xlabel("Accepted update state")
    gain_ax.set_ylabel("Validation vehicle reward gain")
    gain_ax.set_title("Paired GRPO validation vehicle-reward gain")
    gain_ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.75)
    gain_ax.legend(frameon=False, fontsize=7.5, ncol=2)

    x = np.arange(len(PAIRED_SEEDS), dtype=np.float64)
    width = 0.34
    for offset, arm in ((-width / 2, "baseline"), (width / 2, "clipped")):
        stability_ax.bar(
            x + offset,
            [runs[(arm, seed)].adjacent_delta_std for seed in PAIRED_SEEDS],
            width=width,
            color=colors[arm],
            label=arm,
        )
    stability_ax.set_xticks(x, [str(seed) for seed in PAIRED_SEEDS])
    stability_ax.set_xlabel("Paired training seed")
    stability_ax.set_ylabel("Std. of adjacent vehicle-reward-gain deltas")
    stability_ax.set_title("Vehicle-reward-gain volatility by paired seed")
    stability_ax.grid(
        color="#D9D9D9", linewidth=0.8, alpha=0.75, axis="y"
    )
    stability_ax.legend(frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compare_grpo_stability(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    """Validate six paired runs and emit the frozen diagnostic A/B result."""

    manifest_path = Path(manifest_path).resolve()
    source_sha, entries = _validate_manifest(manifest_path)
    runs: dict[tuple[str, int], _RunSeries] = {}
    fairness_binding: dict[str, object] | None = None
    for entry in entries:
        run, binding = _load_run(
            manifest_path,
            entry,
            source_sha=source_sha,
            expected_binding=fairness_binding,
        )
        if fairness_binding is None:
            fairness_binding = binding
        runs[(run.arm, run.seed)] = run

    pairs: list[dict[str, object]] = []
    volatility_reduced_count = 0
    baseline_finals: list[float] = []
    clipped_finals: list[float] = []
    for seed in PAIRED_SEEDS:
        baseline = runs[("baseline", seed)]
        clipped = runs[("clipped", seed)]
        if not np.array_equal(baseline.update_steps, clipped.update_steps):
            raise GRPOStabilityComparisonError(
                f"paired seed {seed} validation update steps are not aligned"
            )
        reduced = clipped.adjacent_delta_std < baseline.adjacent_delta_std
        volatility_reduced_count += int(reduced)
        baseline_finals.append(baseline.final_reward_gain)
        clipped_finals.append(clipped.final_reward_gain)
        pairs.append(
            {
                "seed": seed,
                "validation_accepted_update_states": (
                    baseline.update_steps.tolist()
                ),
                "baseline_vehicle_reward_gain": baseline.reward_gain.tolist(),
                "clipped_vehicle_reward_gain": clipped.reward_gain.tolist(),
                "baseline_adjacent_delta_std": baseline.adjacent_delta_std,
                "clipped_adjacent_delta_std": clipped.adjacent_delta_std,
                "volatility_reduced": reduced,
                "baseline_final_vehicle_reward_gain": baseline.final_reward_gain,
                "clipped_final_vehicle_reward_gain": clipped.final_reward_gain,
                "final_vehicle_reward_gain_delta": (
                    clipped.final_reward_gain - baseline.final_reward_gain
                ),
                "baseline_optimizer_steps": baseline.optimizer_steps,
                "clipped_optimizer_steps": clipped.optimizer_steps,
                "baseline_accepted_update_states": (
                    baseline.accepted_update_states
                ),
                "clipped_accepted_update_states": (
                    clipped.accepted_update_states
                ),
                "baseline_sampling_attempts": (
                    baseline.sampling_attempts
                ),
                "clipped_sampling_attempts": (
                    clipped.sampling_attempts
                ),
                "baseline_rejected_sampling_attempts": (
                    baseline.rejected_sampling_attempts
                ),
                "clipped_rejected_sampling_attempts": (
                    clipped.rejected_sampling_attempts
                ),
                "baseline_exhausted_states": (
                    baseline.exhausted_states
                ),
                "clipped_exhausted_states": (
                    clipped.exhausted_states
                ),
                "baseline_baseline_execution_steps": (
                    baseline.baseline_execution_steps
                ),
                "clipped_baseline_execution_steps": (
                    clipped.baseline_execution_steps
                ),
                "baseline_zero_signal_epochs": (
                    baseline.zero_signal_epochs
                ),
                "clipped_zero_signal_epochs": (
                    clipped.zero_signal_epochs
                ),
                "baseline_wall_time_seconds": baseline.wall_time_seconds,
                "clipped_wall_time_seconds": clipped.wall_time_seconds,
            }
        )

    baseline_final_mean = float(np.mean(baseline_finals))
    clipped_final_mean = float(np.mean(clipped_finals))
    volatility_gate_passed = volatility_reduced_count >= 2
    final_mean_non_degraded = clipped_final_mean >= baseline_final_mean
    output_dir = Path(output_dir).resolve()
    plot_path = _render_comparison_plot(
        runs, output_dir / "validation_vehicle_reward_gain_ab.png"
    )
    report = {
        "format": REPORT_FORMAT,
        "diagnostic_only": True,
        "eligible_for_formal_conclusions": False,
        "manifest": str(manifest_path),
        "source_stage1_sha256": source_sha,
        "paired_seeds": list(PAIRED_SEEDS),
        "trajectories_per_mode": TRAJECTORIES_PER_MODE,
        "target_accepted_update_states_per_run": (
            TARGET_ACCEPTED_UPDATE_STATES
        ),
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
        "optimizer_contract_version": OPTIMIZER_CONTRACT_VERSION,
        "application_contract_version": APPLICATION_CONTRACT_VERSION,
        "rollout_collection_contract_version": (
            ROLLOUT_COLLECTION_CONTRACT_VERSION
        ),
        "rollout_groups_per_bucket_visit": ROLLOUT_GROUPS_PER_BUCKET_VISIT,
        "rollout_start_offset_max_steps": ROLLOUT_START_OFFSET_MAX_STEPS,
        "rollout_start_min_remaining_steps": (
            ROLLOUT_START_MIN_REMAINING_STEPS
        ),
        "max_sampling_attempts_per_state": MAX_SAMPLING_ATTEMPTS_PER_STATE,
        "max_sampling_attempts_multiplier": (
            MAX_SAMPLING_ATTEMPTS_MULTIPLIER
        ),
        "baseline_update_epochs": ARM_UPDATE_EPOCHS["baseline"],
        "clipped_update_epochs": ARM_UPDATE_EPOCHS["clipped"],
        "clip_epsilon_low": CLIP_EPSILON_LOW,
        "clip_epsilon_high": CLIP_EPSILON_HIGH,
        "pairs": pairs,
        "summary": {
            "volatility_reduced_pair_count": volatility_reduced_count,
            "required_volatility_reduced_pair_count": 2,
            "volatility_gate_passed": volatility_gate_passed,
            "baseline_final_vehicle_reward_gain_mean": baseline_final_mean,
            "clipped_final_vehicle_reward_gain_mean": clipped_final_mean,
            "final_vehicle_reward_gain_mean_delta": (
                clipped_final_mean - baseline_final_mean
            ),
            "final_mean_non_degraded": final_mean_non_degraded,
            "passed": volatility_gate_passed and final_mean_non_degraded,
        },
        "plot": str(plot_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compare the six-run Stage-2 clipped-GRPO A/B "
            "diagnostic. See the module docstring for the exact manifest schema."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            f"{MANIFEST_FORMAT} JSON; relative run_dir "
            "values resolve from this file"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "directory for report.json and "
            "validation_vehicle_reward_gain_ab.png"
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = compare_grpo_stability(args.manifest, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
