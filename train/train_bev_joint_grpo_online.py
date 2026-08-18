"""Online proxy-reward calibration and joint GRPO training for Variants A/B."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from evaluation.joint_simulator_branch import (
    JointEpisodeSpec,
    JointSimulatorBranchEvaluator,
    capture_joint_pose_global,
)
from evaluation.longitudinal_tracking_diagnostics import (
    audit_longitudinal_trajectory,
    build_longitudinal_tracking_report,
    summarize_longitudinal_control,
    summarize_trajectory_audits,
)
from evaluation.tracking_diagnostics import (
    aggregate_tracking_rows,
    classify_false_safe,
)
from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from models.bev_planner import (
    JointRewardConfig,
    JointTrajectoryProxyReward,
    KinematicTrajectoryOptimizer,
    KinematicTrajectoryOptimizerConfig,
    TrajectoryOptimizationError,
    TrajectoryOptimizationResult,
    calibrate_joint_rewards,
)
from models.bev_planner.mode_contract import ModeIndex
from train.bev_joint_grpo import (
    grpo_b_checkpoint_payload,
    grpo_checkpoint_payload,
    load_grpo_b_checkpoint,
    load_grpo_checkpoint,
    load_stage1_a_for_grpo,
    load_stage1_b_for_grpo,
    save_grpo_checkpoint,
)
from train.train_bev_diffusion_stage1 import planner_forward_from_batch
from scenarios.bev_round13_contract import (
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    INTERFACE_SMOKE_SCENARIOS,
    PRIMARY_S5_S9_SCENARIOS,
    BEVScenarioContractError,
    deterministic_initial_speed_km_h,
    primary_scenario_contract,
    validate_primary_scenario_contract,
)


AGENT_IDS = ("agent0", "agent1", "agent2")


class OnlineGRPOError(RuntimeError):
    """Raised when online reward calibration or training violates its contract."""


@dataclass(frozen=True)
class JointGRPOOnlineConfig:
    device: str = "cuda"
    seed: int = 17
    total_optimizer_steps: int = 5000
    calibration_report: Path | None = None
    resume_checkpoint: Path | None = None
    scenarios: tuple[tuple[str, str], ...] = PRIMARY_S5_S9_SCENARIOS
    scenario_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    environment_steps_per_episode: int = 100
    validation_interval_steps: int = 100

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise OnlineGRPOError("online GRPO device must be cpu or cuda")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise OnlineGRPOError("online GRPO seed must be an integer")
        for name in (
            "total_optimizer_steps",
            "environment_steps_per_episode",
            "validation_interval_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise OnlineGRPOError(f"{name} must be a positive integer")
        if not self.scenarios or any(
            len(value) != 2 or not value[0] or not value[1]
            for value in self.scenarios
        ):
            raise OnlineGRPOError("at least one scenario/route pair is required")
        try:
            primary_scenario_contract(self.scenarios)
        except BEVScenarioContractError as exc:
            raise OnlineGRPOError(str(exc)) from exc
        if not self.scenario_seeds or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.scenario_seeds
        ):
            raise OnlineGRPOError("scenario_seeds must contain integers")
        if self.calibration_report is not None:
            object.__setattr__(
                self, "calibration_report", Path(self.calibration_report)
            )
        if self.resume_checkpoint is not None:
            object.__setattr__(
                self, "resume_checkpoint", Path(self.resume_checkpoint)
            )


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise OnlineGRPOError("CUDA was requested but is unavailable")
    return torch.device(name)


def model_inputs_to_batch(
    values: object,
    device: torch.device,
    *,
    mode_valid_mask: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    if not hasattr(values, "as_dict"):
        raise OnlineGRPOError("online model inputs must expose as_dict()")
    result = {}
    for name, value in values.as_dict().items():
        array = np.asarray(
            mode_valid_mask
            if name == "mode_valid_mask" and mode_valid_mask is not None
            else value
        )
        if name == "mode_valid_mask" and (
            array.shape != (3, 10) or array.dtype != np.bool_
        ):
            raise OnlineGRPOError(
                "execution mode_valid_mask must be bool [3,10]"
            )
        result[name] = torch.from_numpy(np.array(array, copy=True)).unsqueeze(0).to(
            device
        )
    return result


def execution_mode_valid_mask(
    model_inputs: object,
    *,
    optimizer: KinematicTrajectoryOptimizer | None = None,
) -> np.ndarray:
    """Build the calibrated execution subset of the physical hard mask."""

    for name in ("coarse_trajectories", "ego_state", "mode_valid_mask"):
        if not hasattr(model_inputs, name):
            raise OnlineGRPOError(f"online model inputs are missing {name}")
    transform = optimizer or KinematicTrajectoryOptimizer()
    return transform.execution_mode_valid_mask(
        np.asarray(model_inputs.coarse_trajectories),
        np.asarray(model_inputs.ego_state)[:, 0],
        np.asarray(model_inputs.mode_valid_mask),
    )


def constant_velocity_actions(env: object) -> dict[str, np.ndarray]:
    actions = {}
    for agent_id in AGENT_IDS:
        vehicle = env.agents[agent_id]
        speed = max(0.0, float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6)
        trajectory = np.zeros((8, 3), dtype=np.float32)
        trajectory[:, 0] = speed * np.arange(1, 9, dtype=np.float32) * 0.5
        actions[agent_id] = trajectory
    return actions


def route_following_warmup_actions(
    env: object, builder: JointBEVSampleBuilder
) -> dict[str, np.ndarray]:
    """Label-free route-following warm-up for curved S5--S9 approaches."""
    if not builder.history_ready():
        return constant_velocity_actions(env)
    values = builder.build_model_inputs(env)
    actions = {}
    for role, agent_id in enumerate(AGENT_IDS):
        valid = np.asarray(values.mode_valid_mask[role], dtype=np.bool_)
        preferred = [
            int(ModeIndex.KEEP_HIGH),
            int(ModeIndex.KEEP_MEDIUM),
            int(ModeIndex.KEEP_LOW),
        ]
        mode = next((index for index in preferred if valid[index]), None)
        if mode is None:
            raise OnlineGRPOError(
                f"{agent_id} has no valid KEEP anchor during route warm-up"
            )
        actions[agent_id] = np.array(
            values.coarse_trajectories[role, mode],
            dtype=np.float32,
            copy=True,
        )
    return actions


def joint_trajectory_action(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(trajectories)
    if value.shape != (3, 8, 3) or not np.isfinite(value).all():
        raise OnlineGRPOError("selected online action must be finite [3,8,3]")
    return {
        agent_id: np.array(value[role], dtype=np.float32, copy=True, order="C")
        for role, agent_id in enumerate(AGENT_IDS)
    }


def optimize_selected_model_trajectories(
    model_inputs: object,
    raw_trajectories: np.ndarray,
    selected_modes: np.ndarray,
    *,
    optimizer: KinematicTrajectoryOptimizer | None = None,
) -> TrajectoryOptimizationResult:
    """Apply the execution transform after policy sampling.

    The caller retains ``raw_trajectories`` in the GRPO rollout, so DDIM
    replay/log-prob remains defined on the unmodified diffusion action.
    """

    for name in ("coarse_trajectories", "ego_state"):
        if not hasattr(model_inputs, name):
            raise OnlineGRPOError(f"online model inputs are missing {name}")
    transform = optimizer or KinematicTrajectoryOptimizer()
    return transform.optimize(
        raw_trajectories,
        np.asarray(model_inputs.coarse_trajectories),
        np.asarray(model_inputs.ego_state)[:, 0],
        selected_modes,
    )


def episode_has_ended(
    terminated: Mapping[str, object],
    truncated: Mapping[str, object],
    info: Mapping[str, object],
) -> bool:
    if bool(terminated.get("__all__", False)) or bool(
        truncated.get("__all__", False)
    ):
        return True
    for agent_id in AGENT_IDS:
        value = info.get(agent_id)
        if not isinstance(value, Mapping):
            continue
        if any(
            bool(value.get(name, False))
            for name in (
                "crash",
                "crash_vehicle",
                "crash_object",
                "crash_building",
                "crash_human",
                "out_of_road",
                "out_of_route",
            )
        ):
            return True
    return False


def _new_env(scenario: tuple[str, str], seed: int) -> object:
    env = SensorlessJointBEVPlatoonEnv(
        {
            "num_agents": 3,
            "traffic_density": 0.0,
            "initial_speed_km_h": deterministic_initial_speed_km_h(
                scenario[0], seed
            ),
            "start_seed": int(seed),
            "num_scenarios": 1,
        }
    )
    env.set_runtime_scenario_route(*scenario)
    spawn_manager = getattr(getattr(env, "engine", None), "spawn_manager", None)
    set_spawn_seed = getattr(spawn_manager, "set_episode_spawn_seed", None)
    if callable(set_spawn_seed):
        set_spawn_seed(int(seed))
    env.reset(seed=int(seed))
    return env


def _load_trainer(
    variant: str,
    source_checkpoint: Path,
    device: torch.device,
    *,
    allow_diagnostic_source: bool,
):
    loader = (
        load_stage1_a_for_grpo if variant == "A" else load_stage1_b_for_grpo
    )
    return loader(
        source_checkpoint,
        device=device,
        allow_diagnostic_source=allow_diagnostic_source,
    )


def _json_sha256(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OnlineGRPOError(f"invalid calibration report: {path}") from exc
    if not isinstance(payload, dict):
        raise OnlineGRPOError("calibration report must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_calibration_trajectory_optimizer_contract(
    calibration: Mapping[str, object],
) -> KinematicTrajectoryOptimizerConfig:
    raw_config = calibration.get("trajectory_optimizer_config")
    if not isinstance(raw_config, Mapping):
        raise OnlineGRPOError(
            "calibration trajectory optimizer contract mismatch"
        )
    try:
        parsed = KinematicTrajectoryOptimizerConfig(**dict(raw_config))
    except (TypeError, ValueError, TrajectoryOptimizationError) as exc:
        raise OnlineGRPOError(
            "calibration trajectory optimizer contract mismatch"
        ) from exc
    expected = KinematicTrajectoryOptimizerConfig()
    if (
        parsed != expected
        or calibration.get("trajectory_optimizer_sha256")
        != expected.sha256()
    ):
        raise OnlineGRPOError(
            "calibration trajectory optimizer contract mismatch"
        )
    return parsed


def _joint_rewards_are_informative(
    rewards: np.ndarray, *, minimum_span: float = 1e-6
) -> bool:
    """Whether a sampled group can carry non-zero signed GRPO credit."""

    values = np.asarray(rewards)
    if values.shape != (4,) or values.dtype not in (np.float32, np.float64):
        raise OnlineGRPOError("joint proxy rewards must be float [4]")
    if not np.isfinite(values).all():
        raise OnlineGRPOError("joint proxy rewards must be finite")
    if not math.isfinite(minimum_span) or minimum_span < 0.0:
        raise OnlineGRPOError(
            "minimum reward span must be finite and non-negative"
        )
    return float(np.ptp(values.astype(np.float64, copy=False))) > minimum_span


def _summarize_calibration_tracking(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], float | None, float | None, bool]:
    """Return an explicit failed tracking gate when no safe group exists."""

    if not rows:
        return (
            {
                "row_count": 0,
                "blocked_reason": "no_closed_loop_safe_group",
                "overall": None,
                "by_scenario_role": {},
                "recommended_envelope": {
                    "longitudinal_margin_m": None,
                    "lateral_margin_m": None,
                    "heading_margin_rad": None,
                    "raw_longitudinal_p99_m": None,
                    "raw_lateral_p99_m": None,
                    "raw_heading_p99_rad": None,
                    "within_controller_limits": False,
                },
            },
            None,
            None,
            False,
        )
    tracking = aggregate_tracking_rows(rows)
    tracking["row_count"] = len(rows)
    lateral_p95 = float(tracking["overall"]["lateral_m"]["p95"])
    heading_p95 = float(tracking["overall"]["heading_rad"]["p95"])
    return (
        tracking,
        lateral_p95,
        heading_p95,
        lateral_p95 <= 0.5 and heading_p95 <= 0.1,
    )


def _round_robin_training_buckets(
    scenarios: Sequence[tuple[str, str]], seeds: Sequence[int]
) -> tuple[tuple[tuple[str, str], int], ...]:
    buckets = tuple(
        ((str(scenario[0]), str(scenario[1])), int(seed))
        for scenario in scenarios
        for seed in seeds
    )
    if not buckets:
        raise OnlineGRPOError("online GRPO training schedule is empty")
    return buckets


def _scenario_summary(env: object) -> dict[str, object]:
    orchestrator = getattr(env, "_scenario_orchestrator", None)
    getter = getattr(orchestrator, "get_episode_summary", None)
    if not callable(getter):
        raise OnlineGRPOError("environment has no scenario realization summary")
    value = getter()
    if not isinstance(value, Mapping):
        raise OnlineGRPOError("scenario realization summary is invalid")
    return dict(value)


def _scenario_ready_for_primary_sampling(env: object) -> bool:
    summary = _scenario_summary(env)
    if not bool(summary.get("scenario_realized", False)):
        return False
    scenario_id = str(summary.get("scenario_id", ""))
    if scenario_id == "S5_hard_brake_lead":
        # The adjacent-lane support recipe is realized at reset, before the
        # actual three-second hard-brake event.  It must not make S5 eligible
        # for calibration on its own: require the non-support trigger and the
        # concrete brake profile installed by the hard-brake handler.
        notes = summary.get("scenario_notes", ())
        return bool(summary.get("scenario_triggered", False)) and (
            isinstance(notes, (list, tuple))
            and "lead_brake_profile" in notes
        )
    # S6/S8 background traffic is itself the evaluated hazard.  Sampling an
    # intermediate recipe state would let a new actor appear inside a 4-second
    # simulator branch although the proxy snapshot cannot contain it.  Other
    # primary scenarios contain optional/background coverage recipes whose
    # completion is not part of their hazard realization contract.
    if scenario_id in {
        "S6_background_merge_in",
        "S8_ego_exit_to_ramp",
    }:
        return bool(summary.get("scenario_recipes_complete", False))
    return True


def _replay_pose_error(
    scenario: tuple[str, str],
    seed: int,
    prefix: Sequence[Mapping[str, np.ndarray]],
    reference_pose: np.ndarray,
) -> tuple[float, float]:
    replay = _new_env(scenario, seed)
    try:
        for action in prefix:
            replay.step(action)
        pose = capture_joint_pose_global(replay)
    finally:
        replay.close()
    position = float(
        np.linalg.norm(pose[:, :2] - reference_pose[:, :2], axis=1).max()
    )
    heading = float(
        np.abs(
            np.arctan2(
                np.sin(pose[:, 2] - reference_pose[:, 2]),
                np.cos(pose[:, 2] - reference_pose[:, 2]),
            )
        ).max()
    )
    return position, heading


def run_s5_s9_preflight(
    output_path: Path,
    *,
    seeds: Sequence[int] = DEVELOPMENT_SEEDS,
    states_per_episode: int = 3,
    maximum_steps: int = 200,
) -> dict[str, object]:
    if tuple(int(value) for value in seeds) != DEVELOPMENT_SEEDS:
        raise OnlineGRPOError("preflight requires development seeds [17,23]")
    if states_per_episode != 3 or maximum_steps < 40:
        raise OnlineGRPOError(
            "preflight requires three states and at least 40 environment steps"
        )
    contract = primary_scenario_contract()
    episodes = []
    passed = True
    for scenario in PRIMARY_S5_S9_SCENARIOS:
        for seed in seeds:
            env = _new_env(scenario, int(seed))
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            prefix: list[dict[str, np.ndarray]] = []
            dt_s = simulator_decision_dt_s(env)
            collected = 0
            max_position_error = 0.0
            max_heading_error = 0.0
            failure = None
            last_summary: dict[str, object] = {}
            try:
                for step_index in range(maximum_steps):
                    builder.capture_state(env, step_index * dt_s)
                    last_summary = _scenario_summary(env)
                    if (
                        builder.history_ready()
                        and _scenario_ready_for_primary_sampling(env)
                    ):
                        values = builder.build_model_inputs(env)
                        fields = values.as_dict()
                        if set(fields) != {
                            "bev",
                            "ego_state",
                            "formation_relation_state",
                            "relation_valid_mask",
                            "agent_role",
                            "coarse_trajectories",
                            "mode_valid_mask",
                        }:
                            raise OnlineGRPOError(
                                "preflight online model-input schema mismatch"
                            )
                        pose = capture_joint_pose_global(env)
                        env.close()
                        env = None
                        position_error, heading_error = _replay_pose_error(
                            scenario, int(seed), prefix, pose
                        )
                        max_position_error = max(
                            max_position_error, position_error
                        )
                        max_heading_error = max(
                            max_heading_error, heading_error
                        )
                        if position_error > 0.01 or heading_error > 0.01:
                            raise OnlineGRPOError(
                                "preflight prefix replay exceeded 0.01m/0.01rad"
                            )
                        collected += 1
                        env = _new_env(scenario, int(seed))
                        builder = JointBEVSampleBuilder(AGENT_IDS)
                        builder.reset()
                        for replay_index, replay_action in enumerate(prefix):
                            builder.capture_state(env, replay_index * dt_s)
                            env.step(replay_action)
                        if collected >= states_per_episode:
                            break
                    action = route_following_warmup_actions(env, builder)
                    prefix.append(
                        {
                            name: np.array(value, copy=True)
                            for name, value in action.items()
                        }
                    )
                    _, _, terminated, truncated, info = env.step(action)
                    if episode_has_ended(terminated, truncated, info):
                        failure = "episode_ended_before_three_realized_states"
                        break
                if collected != states_per_episode:
                    passed = False
                    failure = failure or "insufficient_realized_states"
            except Exception as exc:
                passed = False
                failure = f"{type(exc).__name__}:{exc}"
            finally:
                if env is not None:
                    env.close()
            episodes.append(
                {
                    "scenario": scenario[0],
                    "route": scenario[1],
                    "seed": int(seed),
                    "states": collected,
                    "scenario_triggered": bool(
                        last_summary.get("scenario_triggered", False)
                    ),
                    "scenario_realized": bool(
                        last_summary.get("scenario_realized", False)
                    ),
                    "scenario_recipes_complete": bool(
                        last_summary.get("scenario_recipes_complete", False)
                    ),
                    "scenario_completed_recipe_count": int(
                        last_summary.get("scenario_completed_recipe_count", 0)
                    ),
                    "scenario_recipe_count": int(
                        last_summary.get("scenario_recipe_count", 0)
                    ),
                    "maximum_replay_position_error_m": max_position_error,
                    "maximum_replay_heading_error_rad": max_heading_error,
                    "failure": failure,
                    "passed": failure is None and collected == states_per_episode,
                }
            )
    report = {
        "format": "bev_s5_s9_preflight_v1",
        "diagnostic_only": True,
        "scenario_contract": contract,
        "seeds": [int(value) for value in seeds],
        "states_per_episode": states_per_episode,
        "passed": passed,
        "episodes": episodes,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def run_joint_reward_calibration(
    *,
    variant: Literal["A", "B"],
    source_checkpoint: Path,
    output_path: Path,
    device: str = "cuda",
    reward_config: JointRewardConfig | None = None,
    scenarios: Sequence[tuple[str, str]] = PRIMARY_S5_S9_SCENARIOS,
    seeds: Sequence[int] = HOLDOUT_SEEDS,
    states_per_episode: int = 3,
    calibration_phase: Literal["development", "holdout"] = "holdout",
) -> dict[str, object]:
    if variant not in ("A", "B"):
        raise OnlineGRPOError("calibration variant must be A or B")
    if states_per_episode <= 0:
        raise OnlineGRPOError("states_per_episode must be positive")
    if tuple((str(a), str(b)) for a, b in scenarios) != PRIMARY_S5_S9_SCENARIOS:
        raise OnlineGRPOError(
            "primary calibration requires the complete ordered S5--S9 set"
        )
    expected_seeds = (
        DEVELOPMENT_SEEDS
        if calibration_phase == "development"
        else HOLDOUT_SEEDS
        if calibration_phase == "holdout"
        else None
    )
    if expected_seeds is None:
        raise OnlineGRPOError(
            "calibration_phase must be development or holdout"
        )
    if tuple(int(value) for value in seeds) != expected_seeds:
        raise OnlineGRPOError(
            f"{calibration_phase} calibration requires seeds {list(expected_seeds)}"
        )
    contract = primary_scenario_contract(scenarios)
    torch_device = _device(device)
    trainer, source_payload, source_sha = _load_trainer(
        variant,
        Path(source_checkpoint),
        torch_device,
        allow_diagnostic_source=True,
    )
    reward_cfg = reward_config or JointRewardConfig()
    proxy_backend = JointTrajectoryProxyReward(reward_cfg)
    simulator_backend = JointSimulatorBranchEvaluator(reward_cfg)
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(17)
    proxy_rows = []
    simulator_rows = []
    proxy_bad_rows = []
    simulator_bad_rows = []
    details = []
    tracking_rows = []
    longitudinal_audit_rows = []
    longitudinal_state_reports = []
    longitudinal_states_without_safe_group = 0
    longitudinal_clean_errors = []
    longitudinal_target_speed_delta = []
    longitudinal_control_by_role = [
        {
            "target_delta": [],
            "actual_acceleration": [],
            "formation_increment": [],
            "saturation": [],
            "clean_longitudinal_error": [],
            "contamination": [],
        }
        for _ in range(3)
    ]
    false_safe_causes: dict[str, int] = {}

    for scenario in scenarios:
        for seed in seeds:
            env = _new_env(tuple(scenario), int(seed))
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            prefix: list[dict[str, np.ndarray]] = []
            dt_s = simulator_decision_dt_s(env)
            step_index = 0
            collected = 0
            try:
                while collected < states_per_episode:
                    builder.capture_state(env, step_index * dt_s)
                    if not (
                        builder.history_ready()
                        and _scenario_ready_for_primary_sampling(env)
                    ):
                        action = route_following_warmup_actions(env, builder)
                    else:
                        values = builder.build_model_inputs(env)
                        execution_mask = execution_mode_valid_mask(
                            values, optimizer=trajectory_optimizer
                        )
                        batch = model_inputs_to_batch(
                            values,
                            torch_device,
                            mode_valid_mask=execution_mask,
                        )
                        with torch.no_grad():
                            rollout = trainer.sample_groups(
                                batch, generator=generator
                            )
                        raw_candidates = (
                            rollout.selected_trajectories[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        sampled_modes = (
                            rollout.sampled_modes[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64, copy=False)
                        )
                        optimization = optimize_selected_model_trajectories(
                            values,
                            raw_candidates,
                            sampled_modes,
                            optimizer=trajectory_optimizer,
                        )
                        candidates = optimization.optimized_trajectories
                        proxy = proxy_backend.score(env, values, candidates)
                        spec = JointEpisodeSpec(
                            scenario_id=str(scenario[0]),
                            local_route=str(scenario[1]),
                            seed=int(seed),
                            reference_pose_global=capture_joint_pose_global(env),
                        )
                        # MetaDrive 0.4.3 owns one process-global Panda engine.
                        # Release the primary instance before creating fresh
                        # branch environments, then reconstruct it from the
                        # immutable seed+prefix below.
                        env.close()
                        env = None
                        simulator = simulator_backend.evaluate(
                            spec, prefix, candidates
                        )
                        proxy_rows.append(proxy.rewards.copy())
                        simulator_rows.append(simulator.reward.rewards.copy())
                        proxy_bad_rows.append(proxy.unsafe.copy())
                        simulator_bad_rows.append(simulator.reward.unsafe.copy())
                        dynamic_anchor_violations = []
                        for role in range(3):
                            valid_modes = np.flatnonzero(
                                values.mode_valid_mask[role]
                            )
                            for mode in valid_modes:
                                anchor_audit = audit_longitudinal_trajectory(
                                    values.coarse_trajectories[role, mode],
                                    float(values.ego_state[role, 0]),
                                )
                                longitudinal_audit_rows.append(
                                    {
                                        "source": "dynamic_anchor",
                                        "scenario": scenario[0],
                                        "role": role,
                                        "mode": int(mode),
                                        "audit": anchor_audit,
                                    }
                                )
                                if not anchor_audit.valid:
                                    dynamic_anchor_violations.append(
                                        {
                                            "role": role,
                                            "mode": int(mode),
                                            "violations": list(
                                                anchor_audit.violations
                                            ),
                                        }
                                    )
                        state_longitudinal_audits = []
                        for group in range(candidates.shape[0]):
                            group_audits = []
                            for role in range(3):
                                raw_audit = audit_longitudinal_trajectory(
                                    raw_candidates[group, role],
                                    float(values.ego_state[role, 0]),
                                )
                                longitudinal_audit_rows.append(
                                    {
                                        "source": f"diffusion_{variant}_raw",
                                        "scenario": scenario[0],
                                        "role": role,
                                        "mode": int(sampled_modes[group, role]),
                                        "audit": raw_audit,
                                    }
                                )
                                audit = audit_longitudinal_trajectory(
                                    candidates[group, role],
                                    float(values.ego_state[role, 0]),
                                )
                                longitudinal_audit_rows.append(
                                    {
                                        "source": f"diffusion_{variant}_optimized",
                                        "scenario": scenario[0],
                                        "role": role,
                                        "mode": int(
                                            sampled_modes[group, role]
                                        ),
                                        "audit": audit,
                                    }
                                )
                                group_audits.append(
                                    {
                                        "raw": raw_audit.as_dict(),
                                        "optimized": audit.as_dict(),
                                    }
                                )
                            state_longitudinal_audits.append(group_audits)
                        closed_loop_safe_groups = ~simulator.reward.unsafe
                        if bool(closed_loop_safe_groups.any()):
                            state_longitudinal_report = build_longitudinal_tracking_report(
                                simulator,
                                candidates,
                                stop_requested=(sampled_modes == 9),
                                tracking_group_mask=closed_loop_safe_groups,
                            )
                            longitudinal_state_reports.append(
                                state_longitudinal_report
                            )
                        else:
                            # Unsafe groups remain in calibration and
                            # false-safe attribution, but cannot establish a
                            # controller tracking envelope.
                            state_longitudinal_report = None
                            longitudinal_states_without_safe_group += 1
                        for group in range(candidates.shape[0]):
                            for role in range(3):
                                trace = simulator.tracking_traces[group][role]
                                if not trace["longitudinal_errors_m"]:
                                    continue
                                closed_loop_safe = not bool(
                                    simulator.reward.unsafe[group]
                                )
                                formation_enabled = np.asarray(
                                    trace["formation_constraint_enabled"],
                                    dtype=np.bool_,
                                )
                                trajectory_controlled = (
                                    np.ones(
                                        formation_enabled.shape, dtype=np.bool_
                                    )
                                    if role == 0
                                    else ~formation_enabled
                                )
                                lateral_clean = ~np.asarray(
                                    trace["lateral_heading_contaminated"],
                                    dtype=np.bool_,
                                )
                                longitudinal_mask = (
                                    trajectory_controlled
                                    & lateral_clean
                                    & closed_loop_safe
                                )
                                if closed_loop_safe:
                                    tracking_rows.append(
                                        {
                                            "scenario": scenario[0],
                                            "route": scenario[1],
                                            "seed": int(seed),
                                            "state_index": collected,
                                            "group": group,
                                            "role": role,
                                            "mode": int(sampled_modes[group, role]),
                                            # The safety envelope must retain
                                            # every real displacement of a
                                            # closed-loop-safe vehicle,
                                            # including bounded formation
                                            # correction.  Regime filtering is
                                            # only for controller-quality gates.
                                            "longitudinal_errors_m": trace[
                                                "longitudinal_errors_m"
                                            ],
                                            "lateral_errors_m": trace[
                                                "lateral_errors_m"
                                            ],
                                            "heading_errors_rad": trace[
                                                "heading_errors_rad"
                                            ],
                                        }
                                    )
                                longitudinal = np.abs(
                                    np.asarray(
                                        trace["longitudinal_errors_m"],
                                        dtype=np.float64,
                                    )
                                )
                                longitudinal_clean_errors.extend(
                                    longitudinal[longitudinal_mask].tolist()
                                )
                                longitudinal_target_speed_delta.extend(
                                    np.asarray(
                                        trace[
                                            "position_error_speed_increment_mps"
                                        ],
                                        dtype=np.float64,
                                    ).tolist()
                                )
                                role_control = (
                                    longitudinal_control_by_role[role]
                                )
                                role_control["target_delta"].extend(
                                    np.asarray(
                                        trace[
                                            "position_error_speed_increment_mps"
                                        ],
                                        dtype=np.float64,
                                    ).tolist()
                                )
                                role_control["actual_acceleration"].extend(
                                    np.asarray(
                                        trace[
                                            "actual_acceleration_mps2"
                                        ],
                                        dtype=np.float64,
                                    ).tolist()
                                )
                                role_control["formation_increment"].extend(
                                    np.asarray(
                                        trace[
                                            "formation_control_increment"
                                        ],
                                        dtype=np.float64,
                                    ).tolist()
                                )
                                role_control["saturation"].extend(
                                    np.asarray(
                                        trace["control_saturated"],
                                        dtype=np.bool_,
                                    ).tolist()
                                )
                                role_control[
                                    "clean_longitudinal_error"
                                ].extend(
                                    longitudinal[longitudinal_mask].tolist()
                                )
                                role_control["contamination"].extend(
                                    (~longitudinal_mask).tolist()
                                )
                        false_safe_indices = np.flatnonzero(
                            simulator.reward.unsafe & ~proxy.unsafe
                        )
                        false_safe_attribution = {}
                        for group in false_safe_indices:
                            cause = classify_false_safe(
                                simulator_collision=bool(
                                    simulator.reward.collision[group]
                                ),
                                simulator_out_of_drivable=bool(
                                    simulator.reward.out_of_drivable[group]
                                ),
                                failure_reasons=simulator.failure_reasons[group],
                                replay_position_error_m=float(
                                    simulator.replay_position_error_m[group]
                                ),
                                replay_heading_error_rad=float(
                                    simulator.replay_heading_error_rad[group]
                                ),
                                maximum_lateral_error_m=float(
                                    simulator.tracking_lateral_error_m[group].max()
                                ),
                                maximum_heading_error_rad=float(
                                    simulator.tracking_heading_error_rad[group].max()
                                ),
                                minimum_platoon_gap_m=float(
                                    simulator.minimum_platoon_gap_m[group]
                                ),
                                minimum_background_gap_m=float(
                                    simulator.minimum_background_gap_m[group]
                                ),
                            )
                            false_safe_attribution[str(int(group))] = cause
                            false_safe_causes[cause] = (
                                false_safe_causes.get(cause, 0) + 1
                            )
                        details.append(
                            {
                                "scenario": scenario[0],
                                "route": scenario[1],
                                "seed": int(seed),
                                "state_index": collected,
                                "proxy": proxy.rewards.tolist(),
                                "simulator": simulator.reward.rewards.tolist(),
                                "proxy_unsafe": proxy.unsafe.tolist(),
                                "simulator_unsafe": simulator.reward.unsafe.tolist(),
                                "simulator_collision": simulator.reward.collision.tolist(),
                                "simulator_out_of_drivable": simulator.reward.out_of_drivable.tolist(),
                                "false_safe_indices": false_safe_indices.tolist(),
                                "false_safe_attribution": false_safe_attribution,
                                "false_safe_trajectories": candidates[
                                    simulator.reward.unsafe & ~proxy.unsafe
                                ].tolist(),
                                "executed_steps": simulator.executed_steps.tolist(),
                                "minimum_platoon_gap_m": simulator.minimum_platoon_gap_m.tolist(),
                                "minimum_background_gap_m": simulator.minimum_background_gap_m.tolist(),
                                "proxy_minimum_platoon_gap_m": proxy.components[
                                    "minimum_platoon_gap_m"
                                ].tolist(),
                                "proxy_minimum_background_gap_m": proxy.components[
                                    "minimum_background_gap_m"
                                ].tolist(),
                                "failure_reasons": [
                                    list(value)
                                    for value in simulator.failure_reasons
                                ],
                                "component_error": {
                                    name: (
                                        proxy.components[name]
                                        - simulator.reward.components[name]
                                    ).tolist()
                                    for name in proxy.components
                                    if name in simulator.reward.components
                                },
                                "replay_position_error_m": simulator.replay_position_error_m.tolist(),
                                "replay_heading_error_rad": simulator.replay_heading_error_rad.tolist(),
                                "tracking_longitudinal_max_m": simulator.tracking_longitudinal_error_m.tolist(),
                                "tracking_lateral_max_m": simulator.tracking_lateral_error_m.tolist(),
                                "tracking_heading_max_rad": simulator.tracking_heading_error_rad.tolist(),
                                "reference_curvature_max_per_m": simulator.reference_curvature_max_per_m.tolist(),
                                "minimum_road_clearance_m": simulator.minimum_road_clearance_m.tolist(),
                                "tracking_traces": [
                                    [dict(role) for role in group]
                                    for group in simulator.tracking_traces
                                ],
                                "longitudinal_trajectory_audits": (
                                    state_longitudinal_audits
                                ),
                                "trajectory_optimization": {
                                    "config_sha256": optimization.config_sha256,
                                    "elapsed_ms": optimization.elapsed_ms,
                                    "raw_valid": optimization.raw_valid.tolist(),
                                    "optimized_valid": optimization.optimized_valid.tolist(),
                                    "intervention_ade_m": optimization.intervention_ade_m.tolist(),
                                    "intervention_fde_m": optimization.intervention_fde_m.tolist(),
                                    "retained_raw_fraction": optimization.retained_raw_fraction.tolist(),
                                    "profile_regularization": optimization.profile_regularization.tolist(),
                                    "predicted_max_positive_jerk_mps3": optimization.predicted_max_positive_jerk_mps3.tolist(),
                                    "predicted_max_brake_jerk_mps3": optimization.predicted_max_brake_jerk_mps3.tolist(),
                                    "predicted_command_acceleration_min_mps2": optimization.predicted_command_acceleration_min_mps2.tolist(),
                                    "predicted_command_acceleration_max_mps2": optimization.predicted_command_acceleration_max_mps2.tolist(),
                                    "terminal_arc_error_m": optimization.terminal_arc_error_m.tolist(),
                                    "brake_to_drive_transition_s": optimization.brake_to_drive_transition_s.tolist(),
                                    "raw_violations": [
                                        list(value)
                                        for value in optimization.raw_violations
                                    ],
                                },
                                "dynamic_anchor_violations": (
                                    dynamic_anchor_violations
                                ),
                                "longitudinal_tracking_report": (
                                    state_longitudinal_report.as_dict()
                                    if state_longitudinal_report is not None
                                    else {
                                        "excluded_reason": (
                                            "no_closed_loop_safe_group"
                                        )
                                    }
                                ),
                            }
                        )
                        action = joint_trajectory_action(
                            candidates[int(np.argmax(proxy.rewards))]
                        )
                        collected += 1
                        env = _new_env(tuple(scenario), int(seed))
                        builder = JointBEVSampleBuilder(AGENT_IDS)
                        builder.reset()
                        for replay_index, replay_action in enumerate(prefix):
                            builder.capture_state(env, replay_index * dt_s)
                            env.step(replay_action)
                        replay_pose = capture_joint_pose_global(env)
                        replay_position_error = float(
                            np.linalg.norm(
                                replay_pose[:, :2]
                                - spec.reference_pose_global[:, :2],
                                axis=1,
                            ).max()
                        )
                        replay_heading_error = float(
                            np.abs(
                                np.arctan2(
                                    np.sin(
                                        replay_pose[:, 2]
                                        - spec.reference_pose_global[:, 2]
                                    ),
                                    np.cos(
                                        replay_pose[:, 2]
                                        - spec.reference_pose_global[:, 2]
                                    ),
                                )
                            ).max()
                        )
                        if (
                            replay_position_error > 0.01
                            or replay_heading_error > 0.01
                        ):
                            raise OnlineGRPOError(
                                "primary environment reconstruction drifted "
                                "after branch evaluation"
                            )

                    prefix.append(
                        {
                            name: np.array(value, copy=True)
                            for name, value in action.items()
                        }
                    )
                    _, _, terminated, truncated, info = env.step(action)
                    step_index += 1
                    if episode_has_ended(terminated, truncated, info):
                        raise OnlineGRPOError(
                            "calibration episode ended before all states were collected"
                        )
                    if step_index > 100:
                        raise OnlineGRPOError(
                            "calibration history did not yield requested states"
                        )
            finally:
                if env is not None:
                    env.close()

    proxy_array = np.asarray(proxy_rows, dtype=np.float32)
    simulator_array = np.asarray(simulator_rows, dtype=np.float32)
    proxy_bad = np.asarray(proxy_bad_rows, dtype=np.bool_)
    simulator_bad = np.asarray(simulator_bad_rows, dtype=np.bool_)
    result = calibrate_joint_rewards(
        proxy_array,
        simulator_array,
        proxy_bad,
        simulator_bad,
        min_informative_groups=15,
    )
    tracking, lateral_p95, heading_p95, tracking_passed = (
        _summarize_calibration_tracking(tracking_rows)
    )
    longitudinal_audit = summarize_trajectory_audits(
        longitudinal_audit_rows
    )
    longitudinal_blockers = sorted(
        {
            blocker
            for state_report in longitudinal_state_reports
            for blocker in state_report.blockers
        }
    )
    if not longitudinal_state_reports:
        longitudinal_blockers.append("no_closed_loop_safe_group")
    longitudinal_clean_samples = sum(
        report.clean_sample_count for report in longitudinal_state_reports
    )
    longitudinal_contaminated_samples = sum(
        report.contaminated_sample_count
        for report in longitudinal_state_reports
    )
    clean_longitudinal_array = np.asarray(
        longitudinal_clean_errors, dtype=np.float64
    )
    target_speed_delta_array = np.asarray(
        longitudinal_target_speed_delta, dtype=np.float64
    )
    if longitudinal_state_reports:
        longitudinal_control_decomposition = summarize_longitudinal_control(
            longitudinal_control_by_role
        )
        maximum_state_longitudinal_p95 = max(
            report.longitudinal_error_p95_m
            for report in longitudinal_state_reports
        )
        maximum_state_longitudinal_p99 = max(
            report.longitudinal_error_p99_m
            for report in longitudinal_state_reports
        )
        maximum_target_speed_delta_p95 = max(
            report.target_reference_speed_delta_p95_mps
            for report in longitudinal_state_reports
        )
        maximum_continuous_saturation = max(
            report.maximum_continuous_saturation_s
            for report in longitudinal_state_reports
        )
        maximum_stop_terminal_speed = max(
            report.maximum_stop_terminal_speed_mps
            for report in longitudinal_state_reports
        )
    else:
        longitudinal_control_decomposition = {
            "blocked_reason": "no_closed_loop_safe_group"
        }
        maximum_state_longitudinal_p95 = None
        maximum_state_longitudinal_p99 = None
        maximum_target_speed_delta_p95 = None
        maximum_continuous_saturation = None
        maximum_stop_terminal_speed = None
    passed = bool(
        calibration_phase == "holdout"
        and result.passed
        and tracking_passed
        and not longitudinal_blockers
    )
    report: dict[str, object] = {
        "format": "bev_joint_reward_calibration_v2",
        "variant": variant,
        "calibration_phase": calibration_phase,
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "source_stage1_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_stage1_sha256": source_sha,
        "source_dataset_fingerprint": source_payload["dataset_fingerprint"],
        "reward_config": dataclasses.asdict(reward_cfg),
        "trajectory_optimizer_config": dataclasses.asdict(
            trajectory_optimizer.config
        ),
        "trajectory_optimizer_sha256": trajectory_optimizer.config.sha256(),
        "scenario_contract": contract,
        "scenario_contract_sha256": contract["sha256"],
        "scenarios": [list(value) for value in scenarios],
        "seeds": [int(value) for value in seeds],
        "states_per_episode": int(states_per_episode),
        "groups": len(proxy_rows),
        "mean_spearman": result.mean_spearman,
        "pairwise_agreement": result.pairwise_agreement,
        "informative_groups": result.informative_groups,
        "pairwise_comparisons": result.pairwise_comparisons,
        "false_safe_count": result.false_safe_count,
        "tracking": tracking,
        "tracking_gate": {
            "lateral_p95_m": lateral_p95,
            "heading_p95_rad": heading_p95,
            "passed": tracking_passed,
        },
        "longitudinal_diagnostics": {
            "trajectory_audit": longitudinal_audit,
            "clean_sample_count": longitudinal_clean_samples,
            "contaminated_sample_count": (
                longitudinal_contaminated_samples
            ),
            "states_without_closed_loop_safe_group": (
                longitudinal_states_without_safe_group
            ),
            "maximum_state_longitudinal_p95_m": (
                maximum_state_longitudinal_p95
            ),
            "overall_longitudinal_p95_m": (
                float(np.percentile(clean_longitudinal_array, 95))
                if clean_longitudinal_array.size
                else None
            ),
            "overall_longitudinal_p99_m": (
                float(np.percentile(clean_longitudinal_array, 99))
                if clean_longitudinal_array.size
                else None
            ),
            "overall_target_reference_speed_delta_p95_mps": (
                float(
                    np.percentile(
                        np.abs(target_speed_delta_array), 95
                    )
                )
                if target_speed_delta_array.size
                else None
            ),
            "control_decomposition": (
                longitudinal_control_decomposition
            ),
            "maximum_state_longitudinal_p99_m": (
                maximum_state_longitudinal_p99
            ),
            "maximum_target_reference_speed_delta_p95_mps": (
                maximum_target_speed_delta_p95
            ),
            "maximum_continuous_saturation_s": (
                maximum_continuous_saturation
            ),
            "maximum_stop_terminal_speed_mps": maximum_stop_terminal_speed,
            "blockers": longitudinal_blockers,
            "passed": not longitudinal_blockers,
        },
        "false_safe_causes": dict(sorted(false_safe_causes.items())),
        "blockers": longitudinal_blockers,
        "passed": passed,
        "details": details,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _next_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for child in output_root.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                indices.append(int(child.name[4:]))
            except ValueError:
                continue
    path = output_root / f"run_{max(indices, default=0) + 1}"
    path.mkdir()
    (path / "checkpoints").mkdir()
    return path


def _checkpoint_payload(
    *,
    variant: str,
    trainer: object,
    source_sha: str,
    source_payload: Mapping[str, object],
    metrics: Mapping[str, float],
    diagnostic_only: bool,
    run_mode: str,
    reward_config: JointRewardConfig,
    calibration_sha: str,
    calibration_gate_bypassed: bool,
    calibration_report_passed: bool,
    calibration_blockers: Sequence[str],
    scenario_contract_sha: str,
    scenario_seeds: Sequence[int],
    environment_steps: int,
) -> dict[str, object]:
    builder = (
        grpo_checkpoint_payload if variant == "A" else grpo_b_checkpoint_payload
    )
    payload = builder(
        trainer=trainer,
        source_stage1_sha256=source_sha,
        source_stage1_payload=source_payload,
        metrics=metrics,
        diagnostic_only=diagnostic_only,
    )
    payload.update(
        {
            "run_mode": run_mode,
            "reward_config": dataclasses.asdict(reward_config),
            "calibration_report_sha256": calibration_sha,
            "calibration_gate_bypassed": calibration_gate_bypassed,
            "calibration_report_passed": calibration_report_passed,
            "calibration_blockers": list(calibration_blockers),
            "scenario_contract_sha256": scenario_contract_sha,
            "scenario_seeds": [int(value) for value in scenario_seeds],
            "environment_steps": int(environment_steps),
            "trajectory_optimizer_config": dataclasses.asdict(
                KinematicTrajectoryOptimizerConfig()
            ),
            "trajectory_optimizer_sha256": (
                KinematicTrajectoryOptimizerConfig().sha256()
            ),
        }
    )
    return payload


def _validate_online_checkpoint_metadata(
    payload: Mapping[str, object],
    *,
    run_mode: str,
    reward_config: JointRewardConfig,
    calibration_sha: str,
    calibration_gate_bypassed: bool,
    calibration_report_passed: bool,
    calibration_blockers: Sequence[str],
    scenario_contract_sha: str,
    scenario_seeds: Sequence[int],
) -> None:
    expected = {
        "run_mode": run_mode,
        "reward_config": dataclasses.asdict(reward_config),
        "calibration_report_sha256": calibration_sha,
        "calibration_gate_bypassed": calibration_gate_bypassed,
        "calibration_report_passed": calibration_report_passed,
        "calibration_blockers": list(calibration_blockers),
        "scenario_contract_sha256": scenario_contract_sha,
        "scenario_seeds": [int(value) for value in scenario_seeds],
        "trajectory_optimizer_config": dataclasses.asdict(
            KinematicTrajectoryOptimizerConfig()
        ),
        "trajectory_optimizer_sha256": (
            KinematicTrajectoryOptimizerConfig().sha256()
        ),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise OnlineGRPOError(f"online GRPO checkpoint {name} mismatch")
    environment_steps = payload.get("environment_steps")
    if (
        isinstance(environment_steps, bool)
        or not isinstance(environment_steps, int)
        or environment_steps < 0
    ):
        raise OnlineGRPOError(
            "online GRPO checkpoint environment_steps is invalid"
        )


@torch.no_grad()
def _fixed_simulator_validation(
    planner: torch.nn.Module,
    *,
    device: torch.device,
    reward_config: JointRewardConfig,
    scenarios: Sequence[tuple[str, str]],
    seeds: Sequence[int],
) -> dict[str, float]:
    evaluator = JointSimulatorBranchEvaluator(reward_config)
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    rewards = []
    unsafe_count = 0
    collision_count = 0
    out_count = 0
    for scenario in scenarios:
        for seed in seeds:
            env = _new_env(tuple(scenario), int(seed))
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            prefix: list[dict[str, np.ndarray]] = []
            dt_s = simulator_decision_dt_s(env)
            try:
                for step_index in range(100):
                    builder.capture_state(env, step_index * dt_s)
                    if (
                        builder.history_ready()
                        and _scenario_ready_for_primary_sampling(env)
                    ):
                        break
                    action = route_following_warmup_actions(env, builder)
                    prefix.append(
                        {
                            name: np.array(value, copy=True)
                            for name, value in action.items()
                        }
                    )
                    _, _, terminated, truncated, info = env.step(action)
                    if episode_has_ended(terminated, truncated, info):
                        raise OnlineGRPOError(
                            "fixed validation ended during history warm-up"
                        )
                else:
                    raise OnlineGRPOError(
                        "fixed validation never reached a realized S5--S9 state"
                    )
                values = builder.build_model_inputs(env)
                execution_mask = execution_mode_valid_mask(
                    values, optimizer=trajectory_optimizer
                )
                batch = model_inputs_to_batch(
                    values,
                    device,
                    mode_valid_mask=execution_mask,
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed))
                noise = torch.randn(
                    (1, 3, 10, 8, 2),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                output = planner_forward_from_batch(
                    planner, batch, diffusion_noise=noise
                )
                raw_candidate = (
                    output["selected_trajectory"][0]
                    .detach()
                    .cpu()
                    .numpy()[None]
                )
                selected_modes = (
                    output["selected_mode"][0]
                    .detach()
                    .cpu()
                    .numpy()[None]
                )
                candidate = optimize_selected_model_trajectories(
                    values,
                    raw_candidate,
                    selected_modes,
                    optimizer=trajectory_optimizer,
                ).optimized_trajectories
                spec = JointEpisodeSpec(
                    scenario_id=str(scenario[0]),
                    local_route=str(scenario[1]),
                    seed=int(seed),
                    reference_pose_global=capture_joint_pose_global(env),
                )
            finally:
                env.close()
            branch = evaluator.evaluate(spec, prefix, candidate)
            rewards.append(float(branch.reward.rewards[0]))
            unsafe_count += int(branch.reward.unsafe[0])
            collision_count += int(branch.reward.collision[0])
            out_count += int(branch.reward.out_of_drivable[0])
    return {
        "validation/simulator_reward_mean": float(np.mean(rewards)),
        "validation/unsafe_count": float(unsafe_count),
        "validation/collision_count": float(collision_count),
        "validation/out_of_road_count": float(out_count),
    }


def _validation_reward_comparison_metrics(
    current_validation: Mapping[str, object],
    pretrain_reward: object,
) -> dict[str, float]:
    reward_tag = "validation/simulator_reward_mean"
    if reward_tag not in current_validation:
        raise OnlineGRPOError(
            f"fixed validation is missing required metric {reward_tag}"
        )
    try:
        current_reward = float(current_validation[reward_tag])
        baseline_reward = float(pretrain_reward)
    except (TypeError, ValueError) as exc:
        raise OnlineGRPOError(
            "validation current and pretrain rewards must be finite scalars"
        ) from exc
    if not math.isfinite(current_reward) or not math.isfinite(baseline_reward):
        raise OnlineGRPOError(
            "validation current and pretrain rewards must be finite scalars"
        )
    return {
        "validation/pretrain_reward": baseline_reward,
        "validation/reward_gain": current_reward - baseline_reward,
    }


def run_joint_grpo_training(
    config: JointGRPOOnlineConfig,
    *,
    variant: Literal["A", "B"],
    run_mode: Literal["formal", "smoke"],
    source_checkpoint: Path,
    output_root: Path,
    max_optimizer_steps: int | None = None,
    allow_failed_calibration_diagnostic: bool = False,
) -> dict[str, object]:
    if variant not in ("A", "B"):
        raise OnlineGRPOError("online GRPO variant must be A or B")
    if run_mode == "smoke":
        if (
            isinstance(max_optimizer_steps, bool)
            or not isinstance(max_optimizer_steps, int)
            or max_optimizer_steps <= 0
        ):
            raise OnlineGRPOError(
                "smoke mode requires a positive max_optimizer_steps"
            )
        target_steps = max_optimizer_steps
    elif run_mode == "formal":
        if max_optimizer_steps is not None:
            raise OnlineGRPOError("formal mode forbids a diagnostic step cap")
        target_steps = config.total_optimizer_steps
    else:
        raise OnlineGRPOError("run_mode must be formal or smoke")
    if allow_failed_calibration_diagnostic and (
        variant != "A" or run_mode != "smoke"
    ):
        raise OnlineGRPOError(
            "failed-calibration bypass is restricted to Variant A smoke"
        )
    if config.calibration_report is None:
        raise OnlineGRPOError("online GRPO requires a calibration report")
    calibration, calibration_sha = _json_sha256(config.calibration_report)
    if calibration.get("format") != "bev_joint_reward_calibration_v2":
        raise OnlineGRPOError("calibration report format mismatch")
    if calibration.get("variant") != variant:
        raise OnlineGRPOError("calibration report variant mismatch")
    calibration_report_passed = calibration.get("passed") is True
    if not calibration_report_passed and not allow_failed_calibration_diagnostic:
        raise OnlineGRPOError(
            "proxy calibration failed; online GRPO is intentionally blocked"
        )
    if calibration_report_passed and allow_failed_calibration_diagnostic:
        raise OnlineGRPOError(
            "failed-calibration bypass requires a failed calibration report"
        )
    calibration_gate_bypassed = bool(
        allow_failed_calibration_diagnostic and not calibration_report_passed
    )
    calibration_blockers = tuple(
        str(value) for value in calibration.get("blockers", ())
    )
    if calibration.get("calibration_phase") != "holdout":
        raise OnlineGRPOError("online GRPO requires an independent holdout calibration")
    try:
        scenario_contract_sha = validate_primary_scenario_contract(
            calibration.get("scenario_contract")
        )
    except BEVScenarioContractError as exc:
        raise OnlineGRPOError(str(exc)) from exc
    if calibration.get("scenario_contract_sha256") != scenario_contract_sha:
        raise OnlineGRPOError("calibration scenario contract hash mismatch")
    if tuple(tuple(value) for value in calibration.get("scenarios", ())) != (
        PRIMARY_S5_S9_SCENARIOS
    ):
        raise OnlineGRPOError("calibration does not cover the complete S5--S9 set")
    if tuple(calibration.get("seeds", ())) != HOLDOUT_SEEDS:
        raise OnlineGRPOError("calibration does not use holdout seeds [31,47]")
    _validate_calibration_trajectory_optimizer_contract(calibration)
    reward_config = JointRewardConfig(**dict(calibration["reward_config"]))
    torch_device = _device(config.device)
    trainer, source_payload, source_sha = _load_trainer(
        variant,
        Path(source_checkpoint),
        torch_device,
        allow_diagnostic_source=run_mode == "smoke",
    )
    if calibration.get("source_stage1_sha256") != source_sha:
        raise OnlineGRPOError("calibration report source checkpoint mismatch")
    if run_mode == "formal" and source_payload.get(
        "eligible_for_formal_training"
    ) is not True:
        raise OnlineGRPOError("formal GRPO requires an eligible Stage 1 source")
    pretrain_validation = _fixed_simulator_validation(
        trainer.planner,
        device=torch_device,
        reward_config=reward_config,
        scenarios=config.scenarios,
        seeds=HOLDOUT_SEEDS,
    )
    pretrain_reward = _validation_reward_comparison_metrics(
        pretrain_validation,
        pretrain_validation.get("validation/simulator_reward_mean"),
    )["validation/pretrain_reward"]

    checkpoint_loader = (
        load_grpo_checkpoint if variant == "A" else load_grpo_b_checkpoint
    )
    environment_steps = 0
    sampled_rollouts = 0
    uninformative_rollouts = 0
    last_metrics: dict[str, float] = {}
    if config.resume_checkpoint is not None:
        resume_payload = checkpoint_loader(
            config.resume_checkpoint,
            trainer,
            expected_source_stage1_sha256=source_sha,
        )
        _validate_online_checkpoint_metadata(
            resume_payload,
            run_mode=run_mode,
            reward_config=reward_config,
            calibration_sha=calibration_sha,
            calibration_gate_bypassed=calibration_gate_bypassed,
            calibration_report_passed=calibration_report_passed,
            calibration_blockers=calibration_blockers,
            scenario_contract_sha=scenario_contract_sha,
            scenario_seeds=config.scenario_seeds,
        )
        environment_steps = int(resume_payload["environment_steps"])
        last_metrics = {
            str(name): float(value)
            for name, value in resume_payload["metrics"].items()
        }
        if trainer.optimizer_step >= target_steps:
            raise OnlineGRPOError(
                "resume checkpoint already reached requested optimizer steps"
            )

    run_dir = _next_run_directory(Path(output_root))
    frozen = {
        "format": "bev_joint_grpo_online_config_v1",
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "calibration_gate_bypassed": calibration_gate_bypassed,
        "calibration_report_passed": calibration_report_passed,
        "calibration_blockers": list(calibration_blockers),
        "online_config": {
            **dataclasses.asdict(config),
            "calibration_report": str(config.calibration_report),
            "resume_checkpoint": (
                str(config.resume_checkpoint)
                if config.resume_checkpoint is not None
                else None
            ),
        },
        "reward_config": dataclasses.asdict(reward_config),
        "trajectory_optimizer_config": dataclasses.asdict(
            KinematicTrajectoryOptimizerConfig()
        ),
        "trajectory_optimizer_sha256": (
            KinematicTrajectoryOptimizerConfig().sha256()
        ),
        "calibration_report_sha256": calibration_sha,
        "scenario_contract": primary_scenario_contract(config.scenarios),
        "scenario_contract_sha256": scenario_contract_sha,
    }
    (run_dir / "config.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    proxy_backend = JointTrajectoryProxyReward(reward_config)
    trajectory_optimizer = KinematicTrajectoryOptimizer()
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(config.seed)
    training_buckets = _round_robin_training_buckets(
        config.scenarios, config.scenario_seeds
    )
    bucket_sample_counts = [0 for _ in training_buckets]
    bucket_update_counts = [0 for _ in training_buckets]
    consecutive_empty_episodes = 0
    best_key: tuple[float, float] | None = None
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    last_validated_step = -1

    try:
        while trainer.optimizer_step < target_steps:
            bucket_index = sampled_rollouts % len(training_buckets)
            scenario, seed = training_buckets[bucket_index]
            samples_at_episode_start = sampled_rollouts
            env = _new_env(scenario, seed)
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            dt_s = simulator_decision_dt_s(env)
            episode_step = 0
            try:
                while (
                    trainer.optimizer_step < target_steps
                    and sampled_rollouts == samples_at_episode_start
                    and episode_step < config.environment_steps_per_episode
                ):
                    builder.capture_state(env, episode_step * dt_s)
                    if not (
                        builder.history_ready()
                        and _scenario_ready_for_primary_sampling(env)
                    ):
                        action = route_following_warmup_actions(env, builder)
                    else:
                        values = builder.build_model_inputs(env)
                        execution_mask = execution_mode_valid_mask(
                            values, optimizer=trajectory_optimizer
                        )
                        batch = model_inputs_to_batch(
                            values,
                            torch_device,
                            mode_valid_mask=execution_mask,
                        )
                        rollout = trainer.sample_groups(
                            batch, generator=generator
                        )
                        raw_candidates = (
                            rollout.selected_trajectories[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        sampled_modes = (
                            rollout.sampled_modes[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64, copy=False)
                        )
                        try:
                            optimization = optimize_selected_model_trajectories(
                                values,
                                raw_candidates,
                                sampled_modes,
                                optimizer=trajectory_optimizer,
                            )
                        except TrajectoryOptimizationError:
                            np.savez_compressed(
                                run_dir / "trajectory_optimizer_failure.npz",
                                raw_trajectories=raw_candidates,
                                sampled_modes=sampled_modes,
                                coarse_trajectories=values.coarse_trajectories,
                                current_speeds_mps=values.ego_state[:, 0],
                                mode_valid_mask=values.mode_valid_mask,
                                environment_steps=np.asarray(
                                    [environment_steps], dtype=np.int64
                                ),
                                episode_steps=np.asarray(
                                    [episode_step], dtype=np.int64
                                ),
                            )
                            raise
                        candidates = optimization.optimized_trajectories
                        proxy = proxy_backend.score(env, values, candidates)
                        sampled_rollouts += 1
                        bucket_sample_counts[bucket_index] += 1
                        informative = _joint_rewards_are_informative(
                            proxy.rewards
                        )
                        if informative:
                            rewards = torch.from_numpy(
                                proxy.rewards.reshape(1, -1)
                            ).to(torch_device)
                            update = trainer.update(rollout, rewards)
                            last_metrics = update.loss.scalar_metrics()
                            last_metrics.update(
                                {
                                    "optimizer_step": float(
                                        trainer.optimizer_step
                                    ),
                                    "environment_steps": float(
                                        environment_steps
                                    ),
                                    "sampled_rollouts": float(sampled_rollouts),
                                    "uninformative_rollouts": float(
                                        uninformative_rollouts
                                    ),
                                    "training_bucket_index": float(bucket_index),
                                    "proxy_reward_mean": float(
                                        proxy.rewards.mean()
                                    ),
                                    "proxy_reward_max": float(
                                        proxy.rewards.max()
                                    ),
                                    "proxy_unsafe_rate": float(
                                        proxy.unsafe.mean()
                                    ),
                                    "gradient_total": float(
                                        update.total_gradient_norm
                                    ),
                                    "trajectory_optimizer_ms": float(
                                        optimization.elapsed_ms
                                    ),
                                    "trajectory_intervention_ade_m": float(
                                        optimization.intervention_ade_m.mean()
                                    ),
                                    "trajectory_intervention_fde_m": float(
                                        optimization.intervention_fde_m.mean()
                                    ),
                                    "raw_trajectory_valid_rate": float(
                                        optimization.raw_valid.mean()
                                    ),
                                    "trajectory_retained_raw_fraction": float(
                                        optimization.retained_raw_fraction.mean()
                                    ),
                                    "optimized_trajectory_valid_rate": float(
                                        optimization.optimized_valid.mean()
                                    ),
                                }
                            )
                            for name, value in update.gradient_norms.items():
                                last_metrics[f"gradient/{name}"] = float(value)
                            bucket_update_counts[bucket_index] += 1
                            for metric_name, metric_value in last_metrics.items():
                                writer.add_scalar(
                                    metric_name,
                                    metric_value,
                                    trainer.optimizer_step,
                                )
                            with metrics_path.open(
                                "a", encoding="utf-8"
                            ) as stream:
                                stream.write(
                                    json.dumps(last_metrics, sort_keys=True) + "\n"
                                )
                        else:
                            uninformative_rollouts += 1
                            writer.add_scalar(
                                "diagnostic/uninformative_rollouts",
                                float(uninformative_rollouts),
                                environment_steps,
                            )
                        action = joint_trajectory_action(
                            candidates[int(np.argmax(proxy.rewards))]
                        )

                    _, _, terminated, truncated, info = env.step(action)
                    environment_steps += 1
                    episode_step += 1
                    if episode_has_ended(terminated, truncated, info):
                        break
            finally:
                env.close()
            if sampled_rollouts == samples_at_episode_start:
                consecutive_empty_episodes += 1
                if consecutive_empty_episodes >= 3:
                    raise OnlineGRPOError(
                        "three consecutive episodes produced no online state "
                        f"rollout for training bucket {bucket_index}: "
                        f"{scenario[0]}/{scenario[1]} seed={seed}"
                    )
            else:
                consecutive_empty_episodes = 0
            if trainer.optimizer_step != last_validated_step and (
                trainer.optimizer_step == target_steps
                or (
                    trainer.optimizer_step > 0
                    and trainer.optimizer_step
                    % config.validation_interval_steps
                    == 0
                )
            ):
                validation = _fixed_simulator_validation(
                    trainer.planner,
                    device=torch_device,
                    reward_config=reward_config,
                    scenarios=config.scenarios,
                    seeds=HOLDOUT_SEEDS,
                )
                validation.update(
                    _validation_reward_comparison_metrics(
                        validation, pretrain_reward
                    )
                )
                last_metrics.update(validation)
                for metric_name, metric_value in validation.items():
                    writer.add_scalar(
                        metric_name, metric_value, trainer.optimizer_step
                    )
                validation_key = (
                    validation["validation/unsafe_count"],
                    -validation["validation/simulator_reward_mean"],
                )
                checkpoint = _checkpoint_payload(
                    variant=variant,
                    trainer=trainer,
                    source_sha=source_sha,
                    source_payload=source_payload,
                    metrics=last_metrics,
                    diagnostic_only=run_mode != "formal",
                    run_mode=run_mode,
                    reward_config=reward_config,
                    calibration_sha=calibration_sha,
                    calibration_gate_bypassed=calibration_gate_bypassed,
                    calibration_report_passed=calibration_report_passed,
                    calibration_blockers=calibration_blockers,
                    scenario_contract_sha=scenario_contract_sha,
                    scenario_seeds=config.scenario_seeds,
                    environment_steps=environment_steps,
                )
                save_grpo_checkpoint(last_path, checkpoint)
                if best_key is None or validation_key < best_key:
                    best_key = validation_key
                    save_grpo_checkpoint(best_path, checkpoint)
                last_validated_step = trainer.optimizer_step
    finally:
        writer.close()

    if best_key is None:
        raise OnlineGRPOError("online GRPO never completed fixed validation")
    payload = _checkpoint_payload(
        variant=variant,
        trainer=trainer,
        source_sha=source_sha,
        source_payload=source_payload,
        metrics=last_metrics,
        diagnostic_only=run_mode != "formal",
        run_mode=run_mode,
        reward_config=reward_config,
        calibration_sha=calibration_sha,
        calibration_gate_bypassed=calibration_gate_bypassed,
        calibration_report_passed=calibration_report_passed,
        calibration_blockers=calibration_blockers,
        scenario_contract_sha=scenario_contract_sha,
        scenario_seeds=config.scenario_seeds,
        environment_steps=environment_steps,
    )
    last_path = save_grpo_checkpoint(last_path, payload)

    restored, _, _ = _load_trainer(
        variant,
        Path(source_checkpoint),
        torch_device,
        allow_diagnostic_source=run_mode == "smoke",
    )
    loaded = checkpoint_loader(
        last_path,
        restored,
        expected_source_stage1_sha256=source_sha,
    )
    _validate_online_checkpoint_metadata(
        loaded,
        run_mode=run_mode,
        reward_config=reward_config,
        calibration_sha=calibration_sha,
        calibration_gate_bypassed=calibration_gate_bypassed,
        calibration_report_passed=calibration_report_passed,
        calibration_blockers=calibration_blockers,
        scenario_contract_sha=scenario_contract_sha,
        scenario_seeds=config.scenario_seeds,
    )

    report = {
        "format": "bev_joint_grpo_online_report_v1",
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "calibration_gate_bypassed": calibration_gate_bypassed,
        "calibration_report_passed": calibration_report_passed,
        "calibration_blockers": list(calibration_blockers),
        "optimizer_steps": trainer.optimizer_step,
        "environment_steps": environment_steps,
        "sampled_rollouts": sampled_rollouts,
        "uninformative_rollouts": uninformative_rollouts,
        "calibration_report_sha256": calibration_sha,
        "scenario_contract_sha256": scenario_contract_sha,
        "training_scenarios": [list(value) for value in config.scenarios],
        "training_seeds": [int(value) for value in config.scenario_seeds],
        "training_bucket_updates": [
            {
                "scenario": scenario[0],
                "route": scenario[1],
                "seed": seed,
                "sampled_rollouts": bucket_sample_counts[index],
                "optimizer_updates": bucket_update_counts[index],
            }
            for index, (scenario, seed) in enumerate(training_buckets)
        ],
        "validation_seeds": list(HOLDOUT_SEEDS),
        "last_checkpoint": str(last_path.resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "metrics": last_metrics,
        "checkpoint_round_trip": True,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _config_from_yaml(path: Path) -> JointGRPOOnlineConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise OnlineGRPOError(f"unable to read online GRPO config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise OnlineGRPOError("online GRPO YAML root must be a mapping")
    online = payload.get("online")
    if not isinstance(online, Mapping):
        raise OnlineGRPOError("online GRPO YAML requires an online mapping")
    scenarios = tuple(
        (str(value["scenario"]), str(value["route"]))
        for value in online.get("scenarios", ())
        if isinstance(value, Mapping)
    )
    return JointGRPOOnlineConfig(
        device=str(online.get("device", "cuda")),
        seed=int(online.get("seed", 17)),
        total_optimizer_steps=int(online.get("total_optimizer_steps", 5000)),
        calibration_report=(
            Path(str(online["calibration_report"]))
            if online.get("calibration_report")
            else None
        ),
        resume_checkpoint=(
            Path(str(online["resume_checkpoint"]))
            if online.get("resume_checkpoint")
            else None
        ),
        scenarios=scenarios or PRIMARY_S5_S9_SCENARIOS,
        scenario_seeds=tuple(
            int(value) for value in online.get("scenario_seeds", DEVELOPMENT_SEEDS)
        ),
        environment_steps_per_episode=int(
            online.get("environment_steps_per_episode", 100)
        ),
        validation_interval_steps=int(
            online.get("validation_interval_steps", 100)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--run-mode", choices=("formal", "smoke"), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument(
        "--allow-failed-calibration-diagnostic",
        action="store_true",
        help=(
            "Explicitly waive passed=true only for bounded Variant-A smoke; "
            "all other calibration bindings remain enforced."
        ),
    )
    arguments = parser.parse_args()
    config = _config_from_yaml(arguments.config)
    if arguments.calibration_report is not None:
        config = dataclasses.replace(
            config, calibration_report=arguments.calibration_report
        )
    report = run_joint_grpo_training(
        config,
        variant=arguments.variant,
        run_mode=arguments.run_mode,
        source_checkpoint=arguments.source_checkpoint,
        output_root=arguments.output_root,
        max_optimizer_steps=arguments.max_optimizer_steps,
        allow_failed_calibration_diagnostic=(
            arguments.allow_failed_calibration_diagnostic
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEVELOPMENT_SEEDS",
    "HOLDOUT_SEEDS",
    "INTERFACE_SMOKE_SCENARIOS",
    "PRIMARY_S5_S9_SCENARIOS",
    "JointGRPOOnlineConfig",
    "OnlineGRPOError",
    "constant_velocity_actions",
    "episode_has_ended",
    "joint_trajectory_action",
    "model_inputs_to_batch",
    "optimize_selected_model_trajectories",
    "run_joint_grpo_training",
    "run_joint_reward_calibration",
    "run_s5_s9_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
