"""Gate the exact frozen Stage-1 baseline execution path on S5--S9."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from expert_dataset.collect_joint_bev import (  # noqa: E402
    JointBEVSampleBuilder,
    simulator_decision_dt_s,
)
from models.bev_planner import (  # noqa: E402
    KinematicTrajectoryOptimizer,
    TrajectoryOptimizationError,
)
from models.bev_planner.mode_contract import ModeIndex  # noqa: E402
from scenarios.bev_round13_contract import (  # noqa: E402
    DEVELOPMENT_SEEDS,
    PRIMARY_S5_S9_SCENARIOS,
)
from train.bev_joint_grpo import load_stage1_a_for_grpo  # noqa: E402
from train.train_bev_joint_grpo_online import (  # noqa: E402
    AGENT_IDS,
    OnlineGRPOError,
    _condition_online_model_inputs,
    _new_env,
    _new_online_rule_maker,
    _scenario_ready_for_primary_sampling,
    _scenario_summary,
    episode_has_ended,
    execute_cached_frozen_baseline,
    execution_mode_valid_mask,
    model_inputs_to_batch,
    route_following_warmup_actions,
)


REPORT_FORMAT = "bev_frozen_stage1_baseline_gate_v1"
GATE_SCENARIOS = PRIMARY_S5_S9_SCENARIOS
GATE_SEEDS = DEVELOPMENT_SEEDS
BASELINE_STEPS_PER_BUCKET = 200
MAX_WARMUP_ENVIRONMENT_STEPS = 200

_COLLISION_KEYS = (
    "crash",
    "crash_vehicle",
    "crash_object",
    "crash_building",
    "crash_human",
    "crash_sidewalk",
)
_OUT_KEYS = ("out_of_road", "out_of_route")


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _agent_speeds_km_h(env: object) -> dict[str, float | None]:
    agents = getattr(env, "agents", {}) or {}
    return {
        agent_id: (
            float(getattr(agents[agent_id], "speed_km_h", 0.0))
            if agent_id in agents
            else None
        )
        for agent_id in AGENT_IDS
    }


def _agent_end_states(env: object) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    agents = getattr(env, "agents", {}) or {}
    for agent_id in AGENT_IDS:
        vehicle = agents.get(agent_id)
        if vehicle is None:
            result[agent_id] = {"present": False}
            continue
        position = np.asarray(getattr(vehicle, "position", (np.nan, np.nan)))
        result[agent_id] = {
            "present": True,
            "speed_km_h": float(getattr(vehicle, "speed_km_h", 0.0)),
            "position_xy": position[:2].astype(np.float64).tolist(),
            "heading_rad": float(getattr(vehicle, "heading_theta", 0.0)),
        }
    return result


def _agent_step_flags(info: object) -> dict[str, dict[str, bool]]:
    info_mapping = info if isinstance(info, Mapping) else {}
    result: dict[str, dict[str, bool]] = {}
    for agent_id in AGENT_IDS:
        value = info_mapping.get(agent_id, {})
        agent_info = value if isinstance(value, Mapping) else {}
        collision = any(bool(agent_info.get(key, False)) for key in _COLLISION_KEYS)
        out = any(bool(agent_info.get(key, False)) for key in _OUT_KEYS)
        result[agent_id] = {
            "collision": collision,
            "out_of_drivable": out,
            "unsafe": collision or out,
        }
    return result


def _all_flag(value: object) -> bool:
    return bool(value.get("__all__", False)) if isinstance(value, Mapping) else False


def _projection_audits(
    optimizer: KinematicTrajectoryOptimizer,
    model_inputs: object,
    raw_trajectories: np.ndarray | None,
    selected_modes: np.ndarray | None,
) -> dict[str, object]:
    coarse = np.asarray(model_inputs.coarse_trajectories)
    speeds = np.asarray(model_inputs.ego_state)[:, 0]
    stop_modes = np.full((3,), int(ModeIndex.STOP), dtype=np.int64)
    stop_raw = coarse[:, int(ModeIndex.STOP)]

    selected_raw: np.ndarray | None = None
    selected: np.ndarray | None = None
    if raw_trajectories is not None and selected_modes is not None:
        selected_raw = np.asarray(raw_trajectories)
        selected = np.asarray(selected_modes)
        if selected_raw.shape == (1, 3, 8, 3):
            selected_raw = selected_raw[0]
        if selected.shape == (1, 3):
            selected = selected[0]

    project_one = getattr(optimizer, "_project_one")

    def audit_set(
        trajectories: np.ndarray, modes: np.ndarray
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for role, agent_id in enumerate(AGENT_IDS):
            mode = int(modes[role])
            try:
                project_one(
                    trajectories[role],
                    coarse[role, mode],
                    float(speeds[role]),
                    mode,
                )
            except TrajectoryOptimizationError as exc:
                rows.append(
                    {
                        "role": role,
                        "agent_id": agent_id,
                        "mode": mode,
                        "speed_mps": float(speeds[role]),
                        "accepted": False,
                        "exception": str(exc),
                    }
                )
            else:
                rows.append(
                    {
                        "role": role,
                        "agent_id": agent_id,
                        "mode": mode,
                        "speed_mps": float(speeds[role]),
                        "accepted": True,
                        "exception": None,
                    }
                )
        return rows

    result: dict[str, object] = {"stop": audit_set(stop_raw, stop_modes)}
    if selected_raw is not None and selected is not None:
        result["selected"] = audit_set(selected_raw, selected)
    return result


def _failure_record(
    *,
    stage: str,
    error: BaseException,
    env: object,
    baseline_step: int,
    model_inputs: object | None = None,
    raw_trajectories: np.ndarray | None = None,
    selected_modes: np.ndarray | None = None,
    optimizer: KinematicTrajectoryOptimizer | None = None,
    condition: object | None = None,
) -> dict[str, object]:
    failure: dict[str, object] = {
        "stage": stage,
        "baseline_step": int(baseline_step),
        "exception_type": type(error).__name__,
        "exception": str(error),
        "agent_speeds_km_h": _agent_speeds_km_h(env),
        "selected_modes": (
            None
            if selected_modes is None
            else np.asarray(selected_modes, dtype=np.int64).reshape(-1).tolist()
        ),
    }
    if model_inputs is not None:
        failure["model_input_speeds_mps"] = (
            np.asarray(model_inputs.ego_state)[:, 0].astype(np.float64).tolist()
        )
        failure["hard_valid_modes"] = np.asarray(
            model_inputs.mode_valid_mask, dtype=np.bool_
        ).tolist()
    if condition is not None:
        failure["rule_condition"] = {
            "rule_actions": dict(getattr(condition, "rule_actions", {})),
            "is_commitment": bool(getattr(condition, "is_commitment", False)),
        }
    if model_inputs is not None and optimizer is not None:
        audits = _projection_audits(
            optimizer,
            model_inputs,
            raw_trajectories,
            selected_modes,
        )
        failure["projection_audits"] = audits
        failure["failed_roles"] = sorted(
            {
                int(row["role"])
                for rows in audits.values()
                for row in rows
                if not bool(row["accepted"])
            }
        )
    return failure


def _empty_bucket(scenario: tuple[str, str], seed: int) -> dict[str, object]:
    return {
        "scenario_id": scenario[0],
        "local_route": scenario[1],
        "seed": int(seed),
        "status": "running",
        "ready_state_reached": False,
        "counts": {
            "warmup_environment_steps": 0,
            "baseline_execution_steps": 0,
            "frozen_stage1_inference_calls": 0,
            "execution_mode_mask_calls": 0,
            "trajectory_optimizer_calls": 0,
            "rule_maker_finalize_calls": 0,
            "baseline_env_step_calls": 0,
        },
        "safety": {
            "unsafe_steps": 0,
            "collision_agent_steps": 0,
            "out_of_drivable_agent_steps": 0,
            "forced_safe_stop_steps": 0,
            "condition_failure_steps": 0,
            "per_vehicle": {
                agent_id: {
                    "unsafe_steps": 0,
                    "collision_steps": 0,
                    "out_of_drivable_steps": 0,
                    "minimum_speed_km_h": None,
                    "maximum_speed_km_h": None,
                }
                for agent_id in AGENT_IDS
            },
        },
        "selected_mode_counts": {
            agent_id: {str(mode): 0 for mode in range(10)} for agent_id in AGENT_IDS
        },
        "failure": None,
        "end_state": None,
    }


def _update_safety(
    bucket: dict[str, object],
    info: object,
    diagnostics: Mapping[str, int],
    *,
    speeds_km_h: Mapping[str, float | None],
) -> None:
    safety = bucket["safety"]
    assert isinstance(safety, dict)
    flags = _agent_step_flags(info)
    safety["unsafe_steps"] += int(any(row["unsafe"] for row in flags.values()))
    safety["collision_agent_steps"] += sum(
        int(row["collision"]) for row in flags.values()
    )
    safety["out_of_drivable_agent_steps"] += sum(
        int(row["out_of_drivable"]) for row in flags.values()
    )
    safety["forced_safe_stop_steps"] += int(diagnostics.get("forced_safe_stops", 0))
    safety["condition_failure_steps"] += int(
        diagnostics.get("condition_failures", 0)
    )
    per_vehicle = safety["per_vehicle"]
    assert isinstance(per_vehicle, dict)
    for agent_id in AGENT_IDS:
        row = per_vehicle[agent_id]
        assert isinstance(row, dict)
        row["unsafe_steps"] += int(flags[agent_id]["unsafe"])
        row["collision_steps"] += int(flags[agent_id]["collision"])
        row["out_of_drivable_steps"] += int(
            flags[agent_id]["out_of_drivable"]
        )
        speed = speeds_km_h[agent_id]
        if speed is None:
            continue
        current_min = row["minimum_speed_km_h"]
        current_max = row["maximum_speed_km_h"]
        row["minimum_speed_km_h"] = (
            speed if current_min is None else min(float(current_min), speed)
        )
        row["maximum_speed_km_h"] = (
            speed if current_max is None else max(float(current_max), speed)
        )


def _finish_bucket(
    bucket: dict[str, object],
    env: object,
    *,
    terminated: object,
    truncated: object,
    info: object,
) -> None:
    try:
        scenario = _scenario_summary(env)
    except OnlineGRPOError as exc:
        scenario = {"summary_error": str(exc)}
    bucket["end_state"] = {
        "terminated_all": _all_flag(terminated),
        "truncated_all": _all_flag(truncated),
        "episode_ended": episode_has_ended(
            terminated if isinstance(terminated, Mapping) else {},
            truncated if isinstance(truncated, Mapping) else {},
            info if isinstance(info, Mapping) else {},
        ),
        "agents": _agent_end_states(env),
        "last_step_flags": _agent_step_flags(info),
        "scenario_summary": _json_safe(scenario),
    }


def _run_bucket(
    trainer: object,
    *,
    scenario: tuple[str, str],
    seed: int,
    device: torch.device,
    optimizer: KinematicTrajectoryOptimizer,
    baseline_steps: int,
) -> dict[str, object]:
    bucket = _empty_bucket(scenario, seed)
    counts = bucket["counts"]
    assert isinstance(counts, dict)
    env = _new_env(scenario, seed)
    builder = JointBEVSampleBuilder(AGENT_IDS)
    builder.reset()
    rule_maker = _new_online_rule_maker(trainer.planner, env)
    dt_s = simulator_decision_dt_s(env)
    episode_step = 0
    terminated: object = {"__all__": False}
    truncated: object = {"__all__": False}
    info: object = {agent_id: {} for agent_id in AGENT_IDS}
    committed_execution_id: int | None = None
    committed_plan_actions: dict[str, int] | None = None
    ready_state_captured = False
    try:
        for _ in range(MAX_WARMUP_ENVIRONMENT_STEPS + 1):
            builder.capture_state(env, episode_step * dt_s)
            if builder.history_ready() and _scenario_ready_for_primary_sampling(env):
                ready_state_captured = True
                bucket["ready_state_reached"] = True
                break
            if counts["warmup_environment_steps"] >= MAX_WARMUP_ENVIRONMENT_STEPS:
                break
            action = route_following_warmup_actions(env, builder)
            step_result = env.step(action)
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                raise OnlineGRPOError("warm-up env.step must return a five-item tuple")
            _, _, terminated, truncated, info = step_result
            counts["warmup_environment_steps"] += 1
            episode_step += 1
            if episode_has_ended(terminated, truncated, info):
                bucket["status"] = "failed"
                bucket["failure"] = _failure_record(
                    stage="warmup_ended_before_ready",
                    error=OnlineGRPOError("episode ended before the ready state"),
                    env=env,
                    baseline_step=0,
                )
                _finish_bucket(
                    bucket,
                    env,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                return bucket
        if not ready_state_captured:
            bucket["status"] = "failed"
            bucket["failure"] = _failure_record(
                stage="warmup_ready_timeout",
                error=OnlineGRPOError("scenario never reached a ready state"),
                env=env,
                baseline_step=0,
            )
            _finish_bucket(
                bucket,
                env,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            return bucket

        for baseline_step in range(baseline_steps):
            if baseline_step > 0:
                builder.capture_state(env, episode_step * dt_s)
            values = builder.build_model_inputs(env)
            try:
                condition = _condition_online_model_inputs(
                    rule_maker,
                    env,
                    builder,
                    values,
                    committed_execution_id=committed_execution_id,
                    committed_plan_actions=committed_plan_actions,
                )
            except OnlineGRPOError as exc:
                bucket["status"] = "failed"
                bucket["failure"] = _failure_record(
                    stage="rule_maker_condition",
                    error=exc,
                    env=env,
                    baseline_step=baseline_step,
                    model_inputs=values,
                    optimizer=optimizer,
                )
                break
            values = condition.model_inputs
            counts["execution_mode_mask_calls"] += 1
            try:
                execution_mask = execution_mode_valid_mask(values, optimizer=optimizer)
            except TrajectoryOptimizationError as exc:
                bucket["status"] = "failed"
                bucket["failure"] = _failure_record(
                    stage="execution_mode_valid_mask",
                    error=exc,
                    env=env,
                    baseline_step=baseline_step,
                    model_inputs=values,
                    optimizer=optimizer,
                    condition=condition,
                )
                break
            batch = model_inputs_to_batch(
                values,
                device,
                mode_valid_mask=execution_mask,
            )
            frozen = trainer.infer_frozen_pretrain_from_inputs(batch)
            counts["frozen_stage1_inference_calls"] += 1
            raw = (
                frozen["selected_trajectory"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            modes = (
                frozen["selected_mode"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            if raw.shape != (1, 3, 8, 3) or modes.shape != (1, 3):
                raise OnlineGRPOError("frozen Stage-1 baseline output shape mismatch")
            if not bool(execution_mask[np.arange(3), modes[0]].all()):
                raise OnlineGRPOError("frozen Stage-1 selected a masked execution mode")
            for role, agent_id in enumerate(AGENT_IDS):
                mode_counts = bucket["selected_mode_counts"]
                assert isinstance(mode_counts, dict)
                agent_counts = mode_counts[agent_id]
                assert isinstance(agent_counts, dict)
                mode = int(modes[0, role])
                agent_counts[str(mode)] += 1
            pre_step_speeds = _agent_speeds_km_h(env)
            try:
                (
                    step_result,
                    _optimization,
                    diagnostics,
                    committed_execution_id,
                    committed_plan_actions,
                ) = execute_cached_frozen_baseline(
                    env=env,
                    rule_maker=rule_maker,
                    condition=condition,
                    scenario=scenario,
                    model_inputs=values,
                    frozen_raw_trajectories=raw,
                    frozen_selected_modes=modes,
                    optimizer=optimizer,
                )
            except (TrajectoryOptimizationError, OnlineGRPOError) as exc:
                bucket["status"] = "failed"
                bucket["failure"] = _failure_record(
                    stage="baseline_optimizer_or_rule_finalize",
                    error=exc,
                    env=env,
                    baseline_step=baseline_step,
                    model_inputs=values,
                    raw_trajectories=raw,
                    selected_modes=modes,
                    optimizer=optimizer,
                    condition=condition,
                )
                break
            counts["trajectory_optimizer_calls"] += 1
            counts["rule_maker_finalize_calls"] += 1
            counts["baseline_env_step_calls"] += 1
            counts["baseline_execution_steps"] += 1
            _, _, terminated, truncated, info = step_result
            _update_safety(
                bucket,
                info,
                diagnostics,
                speeds_km_h=pre_step_speeds,
            )
            episode_step += 1
            if episode_has_ended(terminated, truncated, info) and (
                baseline_step + 1 < baseline_steps
            ):
                bucket["status"] = "failed"
                bucket["failure"] = _failure_record(
                    stage="baseline_episode_ended_early",
                    error=OnlineGRPOError(
                        "episode ended before 200 frozen baseline execution steps"
                    ),
                    env=env,
                    baseline_step=baseline_step + 1,
                    model_inputs=values,
                    raw_trajectories=raw,
                    selected_modes=modes,
                    optimizer=optimizer,
                    condition=condition,
                )
                break

        if bucket["status"] == "running":
            bucket["status"] = "completed"
        _finish_bucket(
            bucket,
            env,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return bucket
    finally:
        env.close()


def run_frozen_baseline_gate(
    *,
    stage1_checkpoint: Path,
    output: Path,
    device: str = "cuda",
) -> dict[str, object]:
    """Run the fixed ten-bucket, 200-step frozen-baseline execution gate."""

    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise OnlineGRPOError("CUDA was requested but is unavailable")
    torch_device = torch.device(device)
    trainer, source_payload, source_sha256 = load_stage1_a_for_grpo(
        stage1_checkpoint,
        device=torch_device,
        allow_diagnostic_source=True,
    )
    optimizer = KinematicTrajectoryOptimizer()
    buckets: list[dict[str, object]] = []
    for scenario in GATE_SCENARIOS:
        for seed in GATE_SEEDS:
            bucket = _run_bucket(
                trainer,
                scenario=scenario,
                seed=int(seed),
                device=torch_device,
                optimizer=optimizer,
                baseline_steps=BASELINE_STEPS_PER_BUCKET,
            )
            buckets.append(bucket)
            if bucket["status"] != "completed":
                break
        if buckets[-1]["status"] != "completed":
            break

    expected_bucket_count = len(GATE_SCENARIOS) * len(GATE_SEEDS)
    expected_total_steps = expected_bucket_count * BASELINE_STEPS_PER_BUCKET
    count_rows = [row["counts"] for row in buckets]
    total_baseline_steps = sum(
        int(row["baseline_execution_steps"]) for row in count_rows
    )
    pipeline_count_match = all(
        int(row["baseline_execution_steps"])
        == int(row["frozen_stage1_inference_calls"])
        == int(row["execution_mode_mask_calls"])
        == int(row["trajectory_optimizer_calls"])
        == int(row["rule_maker_finalize_calls"])
        == int(row["baseline_env_step_calls"])
        for row in count_rows
    )
    no_collision_or_out = all(
        int(row["safety"]["collision_agent_steps"]) == 0
        and int(row["safety"]["out_of_drivable_agent_steps"]) == 0
        for row in buckets
    )
    gates = {
        "all_ten_buckets_completed": len(buckets) == expected_bucket_count
        and all(row["status"] == "completed" for row in buckets),
        "exactly_200_baseline_steps_per_bucket": len(buckets)
        == expected_bucket_count
        and all(
            int(row["counts"]["baseline_execution_steps"])
            == BASELINE_STEPS_PER_BUCKET
            for row in buckets
        ),
        "exact_total_baseline_steps": total_baseline_steps == expected_total_steps,
        "one_inference_mask_optimizer_finalize_and_step_per_baseline_state": (
            pipeline_count_match
        ),
        "no_collision_or_out_of_drivable": no_collision_or_out,
        "sampled_grpo_candidates_absent": True,
    }
    passed = all(gates.values())
    report: dict[str, object] = {
        "format": REPORT_FORMAT,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "source_stage1": {
            "variant": "A",
            "checkpoint": str(stage1_checkpoint.resolve()),
            "checkpoint_sha256": source_sha256,
            "run_mode": source_payload.get("run_mode"),
            "dataset_fingerprint": _json_safe(
                source_payload.get("dataset_fingerprint")
            ),
        },
        "execution_contract": {
            "scenarios": [list(value) for value in GATE_SCENARIOS],
            "seeds": list(GATE_SEEDS),
            "baseline_steps_per_bucket": BASELINE_STEPS_PER_BUCKET,
            "expected_bucket_count": expected_bucket_count,
            "expected_total_baseline_steps": expected_total_steps,
            "pipeline": [
                "warm history and realized scenario ready state",
                "execution_mode_valid_mask",
                "one deterministic frozen Stage-1 argmax inference",
                "trajectory optimizer",
                "RuleMaker finalize or safe-stop",
                "env.step",
            ],
            "grpo_sampler_called": False,
            "optimizer_thresholds_or_fallbacks_changed": False,
        },
        "totals": {
            "attempted_buckets": len(buckets),
            "baseline_execution_steps": total_baseline_steps,
            "warmup_environment_steps": sum(
                int(row["warmup_environment_steps"]) for row in count_rows
            ),
            "collision_agent_steps": sum(
                int(row["safety"]["collision_agent_steps"]) for row in buckets
            ),
            "out_of_drivable_agent_steps": sum(
                int(row["safety"]["out_of_drivable_agent_steps"])
                for row in buckets
            ),
            "forced_safe_stop_steps": sum(
                int(row["safety"]["forced_safe_stop_steps"]) for row in buckets
            ),
        },
        "gates": gates,
        "buckets": buckets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = run_frozen_baseline_gate(**vars(args))
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_STEPS_PER_BUCKET",
    "GATE_SCENARIOS",
    "GATE_SEEDS",
    "REPORT_FORMAT",
    "run_frozen_baseline_gate",
]
