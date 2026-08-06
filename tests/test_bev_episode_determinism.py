from __future__ import annotations

from pathlib import Path

from evaluation.audit_bev_expert_chain import (
    SensorlessJointBEVPlatoonEnv,
    _episode_spec,
    audit_episode,
)
from expert_dataset.run_joint_bev_collection import load_run_config


def _run_episode(scenario_id: str, route: str, seed: int, max_steps: int):
    config = load_run_config(
        Path("configs/dataset/data_collect_diagnostic64.yaml")
    )
    env_config = dict(config.env_config)
    env_config.update(start_seed=int(seed), num_scenarios=1)
    env = SensorlessJointBEVPlatoonEnv(env_config)
    try:
        return audit_episode(
            env,
            _episode_spec(scenario_id, route, int(seed)),
            max_steps=int(max_steps),
        )
    finally:
        env.close()


def _signature(result: dict) -> tuple:
    rows = []
    for step in result["steps"]:
        agent_state = tuple(
            (
                agent_id,
                tuple(round(float(value), 5) for value in state["position"]),
                round(float(state["heading"]), 6),
                round(float(state["speed_km_h"]), 5),
                tuple(state["lane_index"] or ()),
            )
            for agent_id, state in sorted(step.get("agents", {}).items())
        )
        rows.append(
            (
                int(step["step"]),
                tuple(sorted(step.get("rule", {}).get("best_actions", {}).items())),
                step.get("planner", {}).get("execution", {}).get(
                    "trajectory_source"
                ),
                step.get("failure_reason"),
                agent_state,
            )
        )
    return (
        result["failure_reason"],
        bool(result["scenario_summary"].get("scenario_triggered", False)),
        bool(result["scenario_summary"].get("scenario_realized", False)),
        tuple(rows),
    )


def test_same_seed_is_invariant_to_intervening_episode():
    baseline_b = _run_episode(
        "S5_hard_brake_lead", "R1_entry_straight", 23, 50
    )
    _run_episode("S5_hard_brake_lead", "R1_entry_straight", 17, 50)
    after_a_b = _run_episode(
        "S5_hard_brake_lead", "R1_entry_straight", 23, 50
    )

    assert _signature(after_a_b) == _signature(baseline_b)
