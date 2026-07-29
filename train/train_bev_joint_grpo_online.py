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
from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from models.bev_planner import (
    JointRewardConfig,
    JointTrajectoryProxyReward,
    calibrate_joint_rewards,
)
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


AGENT_IDS = ("agent0", "agent1", "agent2")
DIAGNOSTIC_SCENARIOS = (
    ("S1_free_cruise_straight", "R3_mainline_straight"),
    ("S2_free_cruise_curve", "R2_entry_curve"),
    ("S3_straight_following", "R3_mainline_straight"),
    ("S4_curve_following", "R2_entry_curve"),
)
DIAGNOSTIC_SEEDS = (17, 23)


class OnlineGRPOError(RuntimeError):
    """Raised when online reward calibration or training violates its contract."""


@dataclass(frozen=True)
class JointGRPOOnlineConfig:
    device: str = "cuda"
    seed: int = 17
    total_optimizer_steps: int = 5000
    calibration_report: Path | None = None
    resume_checkpoint: Path | None = None
    scenarios: tuple[tuple[str, str], ...] = DIAGNOSTIC_SCENARIOS
    scenario_seeds: tuple[int, ...] = DIAGNOSTIC_SEEDS
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
    values: object, device: torch.device
) -> dict[str, torch.Tensor]:
    if not hasattr(values, "as_dict"):
        raise OnlineGRPOError("online model inputs must expose as_dict()")
    result = {}
    for name, value in values.as_dict().items():
        array = np.asarray(value)
        result[name] = torch.from_numpy(np.array(array, copy=True)).unsqueeze(0).to(
            device
        )
    return result


def constant_velocity_actions(env: object) -> dict[str, np.ndarray]:
    actions = {}
    for agent_id in AGENT_IDS:
        vehicle = env.agents[agent_id]
        speed = max(0.0, float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6)
        trajectory = np.zeros((8, 3), dtype=np.float32)
        trajectory[:, 0] = speed * np.arange(1, 9, dtype=np.float32) * 0.5
        actions[agent_id] = trajectory
    return actions


def joint_trajectory_action(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(trajectories)
    if value.shape != (3, 8, 3) or not np.isfinite(value).all():
        raise OnlineGRPOError("selected online action must be finite [3,8,3]")
    return {
        agent_id: np.array(value[role], dtype=np.float32, copy=True, order="C")
        for role, agent_id in enumerate(AGENT_IDS)
    }


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
            "start_seed": int(seed),
            "num_scenarios": 1,
        }
    )
    env.set_runtime_scenario_route(*scenario)
    env.reset()
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


def run_joint_reward_calibration(
    *,
    variant: Literal["A", "B"],
    source_checkpoint: Path,
    output_path: Path,
    device: str = "cuda",
    reward_config: JointRewardConfig | None = None,
    scenarios: Sequence[tuple[str, str]] = DIAGNOSTIC_SCENARIOS,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
    states_per_episode: int = 3,
) -> dict[str, object]:
    if variant not in ("A", "B"):
        raise OnlineGRPOError("calibration variant must be A or B")
    if states_per_episode <= 0:
        raise OnlineGRPOError("states_per_episode must be positive")
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
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(17)
    proxy_rows = []
    simulator_rows = []
    proxy_bad_rows = []
    simulator_bad_rows = []
    details = []

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
                    if not builder.history_ready():
                        action = constant_velocity_actions(env)
                    else:
                        values = builder.build_model_inputs(env)
                        batch = model_inputs_to_batch(values, torch_device)
                        with torch.no_grad():
                            rollout = trainer.sample_groups(
                                batch, generator=generator
                            )
                        candidates = (
                            rollout.selected_trajectories[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
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
                                "false_safe_indices": np.flatnonzero(
                                    simulator.reward.unsafe & ~proxy.unsafe
                                ).tolist(),
                                "false_safe_trajectories": candidates[
                                    simulator.reward.unsafe & ~proxy.unsafe
                                ].tolist(),
                                "executed_steps": simulator.executed_steps.tolist(),
                                "minimum_platoon_gap_m": simulator.minimum_platoon_gap_m.tolist(),
                                "minimum_background_gap_m": simulator.minimum_background_gap_m.tolist(),
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
                                },
                                "replay_position_error_m": simulator.replay_position_error_m.tolist(),
                                "replay_heading_error_rad": simulator.replay_heading_error_rad.tolist(),
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
        proxy_array, simulator_array, proxy_bad, simulator_bad
    )
    report: dict[str, object] = {
        "format": "bev_joint_reward_calibration_v1",
        "variant": variant,
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "source_stage1_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_stage1_sha256": source_sha,
        "source_dataset_fingerprint": source_payload["dataset_fingerprint"],
        "reward_config": dataclasses.asdict(reward_cfg),
        "scenarios": [list(value) for value in scenarios],
        "seeds": [int(value) for value in seeds],
        "states_per_episode": int(states_per_episode),
        "groups": len(proxy_rows),
        "mean_spearman": result.mean_spearman,
        "pairwise_agreement": result.pairwise_agreement,
        "informative_groups": result.informative_groups,
        "pairwise_comparisons": result.pairwise_comparisons,
        "false_safe_count": result.false_safe_count,
        "passed": result.passed,
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
            "scenario_seeds": [int(value) for value in scenario_seeds],
            "environment_steps": int(environment_steps),
        }
    )
    return payload


def _validate_online_checkpoint_metadata(
    payload: Mapping[str, object],
    *,
    run_mode: str,
    reward_config: JointRewardConfig,
    calibration_sha: str,
    scenario_seeds: Sequence[int],
) -> None:
    expected = {
        "run_mode": run_mode,
        "reward_config": dataclasses.asdict(reward_config),
        "calibration_report_sha256": calibration_sha,
        "scenario_seeds": [int(value) for value in scenario_seeds],
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
                for step_index in range(30):
                    builder.capture_state(env, step_index * dt_s)
                    if builder.history_ready():
                        break
                    action = constant_velocity_actions(env)
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
                        "fixed validation history was never ready"
                    )
                values = builder.build_model_inputs(env)
                batch = model_inputs_to_batch(values, device)
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
                candidate = (
                    output["selected_trajectory"][0]
                    .detach()
                    .cpu()
                    .numpy()[None]
                )
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


def run_joint_grpo_training(
    config: JointGRPOOnlineConfig,
    *,
    variant: Literal["A", "B"],
    run_mode: Literal["formal", "smoke"],
    source_checkpoint: Path,
    output_root: Path,
    max_optimizer_steps: int | None = None,
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
    if config.calibration_report is None:
        raise OnlineGRPOError("online GRPO requires a calibration report")
    calibration, calibration_sha = _json_sha256(config.calibration_report)
    if calibration.get("format") != "bev_joint_reward_calibration_v1":
        raise OnlineGRPOError("calibration report format mismatch")
    if calibration.get("variant") != variant:
        raise OnlineGRPOError("calibration report variant mismatch")
    if calibration.get("passed") is not True:
        raise OnlineGRPOError(
            "proxy calibration failed; online GRPO is intentionally blocked"
        )
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

    checkpoint_loader = (
        load_grpo_checkpoint if variant == "A" else load_grpo_b_checkpoint
    )
    environment_steps = 0
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
        "calibration_report_sha256": calibration_sha,
    }
    (run_dir / "config.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    proxy_backend = JointTrajectoryProxyReward(reward_config)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(config.seed)
    scenario_index = 0
    best_key: tuple[float, float] | None = None
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    last_validated_step = -1

    try:
        while trainer.optimizer_step < target_steps:
            scenario = config.scenarios[scenario_index % len(config.scenarios)]
            seed = config.scenario_seeds[
                scenario_index % len(config.scenario_seeds)
            ]
            scenario_index += 1
            env = _new_env(scenario, seed)
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            dt_s = simulator_decision_dt_s(env)
            episode_step = 0
            try:
                while (
                    trainer.optimizer_step < target_steps
                    and episode_step < config.environment_steps_per_episode
                ):
                    builder.capture_state(env, episode_step * dt_s)
                    if not builder.history_ready():
                        action = constant_velocity_actions(env)
                    else:
                        values = builder.build_model_inputs(env)
                        batch = model_inputs_to_batch(values, torch_device)
                        rollout = trainer.sample_groups(
                            batch, generator=generator
                        )
                        candidates = (
                            rollout.selected_trajectories[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        proxy = proxy_backend.score(env, values, candidates)
                        rewards = torch.from_numpy(
                            proxy.rewards.reshape(1, -1)
                        ).to(torch_device)
                        update = trainer.update(rollout, rewards)
                        last_metrics = update.loss.scalar_metrics()
                        last_metrics.update(
                            {
                                "optimizer_step": float(trainer.optimizer_step),
                                "environment_steps": float(environment_steps),
                                "proxy_reward_mean": float(
                                    proxy.rewards.mean()
                                ),
                                "proxy_reward_max": float(proxy.rewards.max()),
                                "proxy_unsafe_rate": float(proxy.unsafe.mean()),
                                "gradient_total": float(
                                    update.total_gradient_norm
                                ),
                            }
                        )
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
                    seeds=config.scenario_seeds,
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
        scenario_seeds=config.scenario_seeds,
    )

    report = {
        "format": "bev_joint_grpo_online_report_v1",
        "variant": variant,
        "run_mode": run_mode,
        "diagnostic_only": run_mode != "formal",
        "eligible_for_formal_training": run_mode == "formal",
        "optimizer_steps": trainer.optimizer_step,
        "environment_steps": environment_steps,
        "calibration_report_sha256": calibration_sha,
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
        scenarios=scenarios or DIAGNOSTIC_SCENARIOS,
        scenario_seeds=tuple(
            int(value) for value in online.get("scenario_seeds", DIAGNOSTIC_SEEDS)
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
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DIAGNOSTIC_SCENARIOS",
    "DIAGNOSTIC_SEEDS",
    "JointGRPOOnlineConfig",
    "OnlineGRPOError",
    "constant_velocity_actions",
    "episode_has_ended",
    "joint_trajectory_action",
    "model_inputs_to_batch",
    "run_joint_grpo_training",
    "run_joint_reward_calibration",
]


if __name__ == "__main__":
    raise SystemExit(main())
