"""Fixed validation state-bank capture, loading, and deterministic replay."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch

from expert_dataset.collect_joint_bev import JointBEVSampleBuilder, simulator_decision_dt_s
from models.bev_planner import (
    DDIMNoiseBundle,
    DDIMTransitionError,
    DEFAULT_DDIM_PATH,
    KinematicTrajectoryOptimizer,
)
from scenarios.bev_round13_contract import HOLDOUT_SEEDS, PRIMARY_S5_S9_SCENARIOS

from .config import AGENT_IDS, OnlineGRPOError
from .contracts import VALIDATION_STATE_BANK_FORMAT, _checkpoint_file_sha256
from .environment import (
    _condition_online_model_inputs,
    _new_env,
    _new_online_rule_maker,
    _scenario_ready_for_primary_sampling,
    _scenario_summary,
    episode_has_ended,
    execution_mode_valid_mask,
    model_inputs_to_batch,
    route_following_warmup_actions,
)

def build_grpo_validation_state_bank(
    path: Path,
    *,
    scenarios: Sequence[tuple[str, str]] = PRIMARY_S5_S9_SCENARIOS,
    seeds: Sequence[int] = HOLDOUT_SEEDS,
) -> dict[str, object]:
    """Capture the fixed S5--S9 validation states and common DDIM noises."""

    if tuple(tuple(value) for value in scenarios) != tuple(PRIMARY_S5_S9_SCENARIOS):
        raise OnlineGRPOError("validation state bank requires the fixed S5--S9 set")
    if tuple(int(value) for value in seeds) != tuple(HOLDOUT_SEEDS):
        raise OnlineGRPOError("validation state bank requires seeds 31/47")

    records: list[dict[str, object]] = []
    planner_identity = SimpleNamespace(config=SimpleNamespace(model_version="v2"))
    trajectory_optimizer = KinematicTrajectoryOptimizer()

    for scenario_index, scenario in enumerate(scenarios):
        for seed in seeds:
            env = _new_env(tuple(scenario), int(seed))
            builder = JointBEVSampleBuilder(AGENT_IDS)
            builder.reset()
            rule_maker = _new_online_rule_maker(planner_identity, env)
            prefix: list[dict[str, np.ndarray]] = []
            dt_s = simulator_decision_dt_s(env)
            try:
                for step_index in range(100):
                    builder.capture_state(env, step_index * dt_s)
                    if builder.history_ready() and _scenario_ready_for_primary_sampling(env):
                        break
                    action = route_following_warmup_actions(env, builder)
                    prefix.append(
                        {name: np.array(value, copy=True) for name, value in action.items()}
                    )
                    _, _, terminated, truncated, info = env.step(action)
                    if episode_has_ended(terminated, truncated, info):
                        raise OnlineGRPOError(
                            "validation state-bank warm-up ended before readiness"
                        )
                else:
                    raise OnlineGRPOError(
                        "validation state bank never reached a realized state"
                    )

                values = builder.build_model_inputs(env)
                condition = _condition_online_model_inputs(
                    rule_maker,
                    env,
                    builder,
                    values,
                    committed_execution_id=None,
                    committed_plan_actions=None,
                )
                values = condition.model_inputs
                scenario_summary = _scenario_summary(env)
                execution_mask = execution_mode_valid_mask(
                    values, optimizer=trajectory_optimizer
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(10000019 + 1009 * int(seed) + scenario_index)
                noise = DDIMNoiseBundle.sample(
                    (1, 3, 10, 8, 2),
                    device=torch.device("cpu"),
                    generator=generator,
                )
                records.append(
                    {
                        "scenario": tuple(scenario),
                        "seed": int(seed),
                        "prefix": prefix,
                        "model_inputs": {
                            name: np.array(value, copy=True)
                            for name, value in values.as_dict().items()
                        },
                        "execution_mode_valid_mask": np.array(execution_mask, copy=True),
                        "rule_actions": dict(condition.rule_actions),
                        "scenario_state_contract": {
                            "scenario_random_seed": scenario_summary.get(
                                "scenario_random_seed"
                            ),
                            "severity_bucket": scenario_summary.get("severity_bucket"),
                            "resolved_scenario_parameters": dict(
                                scenario_summary.get("resolved_scenario_parameters", {})
                            ),
                        },
                        "initial_noise": noise.initial_noise.cpu(),
                        "transition_noises": tuple(
                            value.cpu() for value in noise.transition_noises
                        ),
                    }
                )
            finally:
                env.close()

    payload: dict[str, object] = {
        "format": VALIDATION_STATE_BANK_FORMAT,
        "scenarios": [tuple(value) for value in scenarios],
        "seeds": [int(value) for value in seeds],
        "ddim_path": DEFAULT_DDIM_PATH.as_dict(),
        "records": records,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _load_grpo_validation_state_bank(
    path: Path,
    *,
    scenarios: Sequence[tuple[str, str]],
    seeds: Sequence[int],
) -> tuple[dict[tuple[tuple[str, str], int], Mapping[str, object]], str]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OnlineGRPOError(f"unable to load validation state bank: {path}") from exc
    if not isinstance(payload, Mapping):
        raise OnlineGRPOError("validation state bank must be an object")
    if (
        payload.get("format") != VALIDATION_STATE_BANK_FORMAT
        or payload.get("ddim_path") != DEFAULT_DDIM_PATH.as_dict()
        or payload.get("scenarios") != [tuple(value) for value in scenarios]
        or payload.get("seeds") != [int(value) for value in seeds]
    ):
        raise OnlineGRPOError("validation state bank contract mismatch")

    raw_records = payload.get("records")
    expected_count = len(scenarios) * len(seeds)
    if not isinstance(raw_records, list) or len(raw_records) != expected_count:
        raise OnlineGRPOError("validation state bank record count mismatch")

    records: dict[tuple[tuple[str, str], int], Mapping[str, object]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise OnlineGRPOError("validation state bank record is invalid")
        raw_scenario = record.get("scenario")
        if not isinstance(raw_scenario, (list, tuple)) or len(raw_scenario) != 2:
            raise OnlineGRPOError("validation state bank scenario is invalid")
        key = ((str(raw_scenario[0]), str(raw_scenario[1])), int(record["seed"]))
        if key in records:
            raise OnlineGRPOError("validation state bank has duplicate records")
        records[key] = record

    expected = {
        (tuple(scenario), int(seed)) for scenario in scenarios for seed in seeds
    }
    if set(records) != expected:
        raise OnlineGRPOError("validation state bank scenario/seed matrix mismatch")
    return records, _checkpoint_file_sha256(Path(path))


def _replay_fixed_validation_state(
    record: Mapping[str, object],
    *,
    scenario: tuple[str, str],
    seed: int,
    device: torch.device,
    trajectory_optimizer: KinematicTrajectoryOptimizer,
) -> tuple[object, object, object, np.ndarray, dict[str, Tensor], DDIMNoiseBundle]:
    env = _new_env(scenario, seed)
    builder = JointBEVSampleBuilder(AGENT_IDS)
    builder.reset()
    rule_maker = _new_online_rule_maker(
        SimpleNamespace(config=SimpleNamespace(model_version="v2")), env
    )
    raw_prefix = record.get("prefix")
    if not isinstance(raw_prefix, list):
        env.close()
        raise OnlineGRPOError("validation state bank prefix is invalid")

    dt_s = simulator_decision_dt_s(env)
    for step_index, raw_action in enumerate(raw_prefix):
        if not isinstance(raw_action, Mapping):
            env.close()
            raise OnlineGRPOError("validation state bank action is invalid")
        builder.capture_state(env, step_index * dt_s)
        action = {str(name): np.asarray(value).copy() for name, value in raw_action.items()}
        _, _, terminated, truncated, info = env.step(action)
        if episode_has_ended(terminated, truncated, info):
            env.close()
            raise OnlineGRPOError("fixed validation prefix ended unexpectedly")

    builder.capture_state(env, len(raw_prefix) * dt_s)
    values = builder.build_model_inputs(env)
    condition = _condition_online_model_inputs(
        rule_maker,
        env,
        builder,
        values,
        committed_execution_id=None,
        committed_plan_actions=None,
    )
    values = condition.model_inputs

    stored_inputs = record.get("model_inputs")
    if not isinstance(stored_inputs, Mapping):
        env.close()
        raise OnlineGRPOError("validation state bank inputs are invalid")
    current_inputs = values.as_dict()
    if set(stored_inputs) != set(current_inputs) or any(
        not np.array_equal(np.asarray(stored_inputs[name]), current_inputs[name])
        for name in current_inputs
    ):
        env.close()
        raise OnlineGRPOError("fixed validation replay changed model inputs")

    stored_rule_actions = record.get("rule_actions")
    if dict(condition.rule_actions) != dict(stored_rule_actions):
        env.close()
        raise OnlineGRPOError("fixed validation replay changed RuleMaker condition")

    summary = _scenario_summary(env)
    replayed_scenario_contract = {
        "scenario_random_seed": summary.get("scenario_random_seed"),
        "severity_bucket": summary.get("severity_bucket"),
        "resolved_scenario_parameters": dict(
            summary.get("resolved_scenario_parameters", {})
        ),
    }
    if replayed_scenario_contract != record.get("scenario_state_contract"):
        env.close()
        raise OnlineGRPOError(
            "fixed validation replay changed resolved scenario parameters"
        )

    execution_mask = execution_mode_valid_mask(values, optimizer=trajectory_optimizer)
    if not np.array_equal(
        execution_mask, np.asarray(record.get("execution_mode_valid_mask"))
    ):
        env.close()
        raise OnlineGRPOError("fixed validation replay changed executable modes")

    batch = model_inputs_to_batch(values, device, mode_valid_mask=execution_mask)
    try:
        bundle = DDIMNoiseBundle(
            initial_noise=record["initial_noise"].to(device=device),
            transition_noises=tuple(
                value.to(device=device) for value in record["transition_noises"]
            ),
        )
        bundle.validate(batch["coarse_trajectories"][..., :2].shape, device=device)
    except (KeyError, AttributeError, DDIMTransitionError) as exc:
        env.close()
        raise OnlineGRPOError("validation state bank noise is invalid") from exc

    return env, values, condition, execution_mask, batch, bundle


__all__ = [
    "build_grpo_validation_state_bank",
    "_load_grpo_validation_state_bank",
    "_replay_fixed_validation_state",
]
