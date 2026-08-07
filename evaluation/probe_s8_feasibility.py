"""Probe S8 at the first production-planner failure without relaxing safety.

The probe reuses the exact failed simulator state and RuleMaker proposal batch.
It only densifies the legal longitudinal profile lattice; road footprint,
kinematics, 5 m/7 m gaps and dense OBB checks remain the production checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.audit_bev_expert_chain import (  # noqa: E402
    _compact_planner_debug,
    _compact_rule_debug,
    _episode_spec,
    _json_value,
)
from expert_dataset.collect_joint_bev import (  # noqa: E402
    JointBEVSampleBuilder,
    JointCollectionError,
    RulePlannerExpert,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from expert_dataset.run_joint_bev_collection import (  # noqa: E402
    _configure_episode,
    load_run_config,
)
from models.platoon_planner.platoon_normal_planner import (  # noqa: E402
    NormalPlannerNoFeasiblePlan,
    PlatoonNormalPlanner,
)


class _DenseS8FeasibilityPlanner(PlatoonNormalPlanner):
    """Diagnostic-only planner with denser legal longitudinal profiles."""

    def __init__(self) -> None:
        super().__init__(candidate_pool_size=24)

    def _candidate_accelerations(self, **kwargs) -> tuple[float, ...]:
        production = super()._candidate_accelerations(**kwargs)
        dense = np.arange(
            self.MIN_ACCEL_MPS2,
            self.MAX_ACCEL_MPS2 + 0.5,
            1.0,
            dtype=np.float64,
        )
        return tuple(sorted({*production, *(round(float(v), 3) for v in dense)}))

    def _acceleration_durations(self, acceleration_mps2: float) -> tuple[float, ...]:
        if float(acceleration_mps2) >= -0.25:
            return (1.0, 2.0, 3.0, 4.0)
        return tuple(float(value) for value in np.arange(0.5, 4.01, 0.5))

    def _recovery_accelerations(
        self,
        acceleration_mps2: float,
        acceleration_duration_s: float,
    ) -> tuple[float, ...]:
        if float(acceleration_duration_s) >= self.HORIZON_S:
            return (0.0,)
        return (-2.0, 0.0, 1.5, 3.0, 5.0)

    @staticmethod
    def _select_longitudinal_profiles(profiles, *, maximum=3):
        del maximum
        return PlatoonNormalPlanner._select_longitudinal_profiles(
            profiles,
            maximum=24,
        )


def run_probe(*, config_path: Path, seed: int, max_steps: int) -> dict:
    config = load_run_config(config_path)
    env_config = dict(config.env_config)
    env_config.update(start_seed=int(seed), num_scenarios=1)
    env = SensorlessJointBEVPlatoonEnv(env_config)
    spec = _episode_spec("S8_ego_exit_to_ramp", "R6_exit_to_ramp", int(seed))
    try:
        _configure_episode(env, spec)
        env.reset(seed=int(seed))
        agent_ids = ("agent0", "agent1", "agent2")
        expert = RulePlannerExpert(env, agent_ids)
        builder = JointBEVSampleBuilder(agent_ids)
        builder.reset()
        dt_s = simulator_decision_dt_s(env)
        trace: list[dict[str, object]] = []
        for step in range(int(max_steps)):
            builder.capture_state(env, step * dt_s)
            model_inputs = (
                builder.build_model_inputs(env) if builder.history_ready() else None
            )
            try:
                expert_step = expert.plan(env, model_inputs=model_inputs)
            except JointCollectionError as exc:
                production_debug = expert.planner.get_last_debug() or {}
                rule_debug = expert.rule_maker.get_last_debug() or {}
                batch = expert.rule_maker._outstanding_proposal_batch
                if exc.reason_code != "all_rule_proposals_infeasible" or batch is None:
                    return {
                        "status": "unexpected_failure",
                        "seed": int(seed),
                        "step": int(step),
                        "reason": exc.reason_code,
                        "rule": _compact_rule_debug(rule_debug),
                        "production": _compact_planner_debug(production_debug),
                        "trace_tail": trace[-16:],
                    }
                probe = _DenseS8FeasibilityPlanner()
                try:
                    result = probe.plan_ranked(env, batch.proposals)
                    probe_status = "feasible"
                    selected_rank = int(result.proposal_rank)
                except NormalPlannerNoFeasiblePlan:
                    probe_status = "infeasible_on_dense_lattice"
                    selected_rank = None
                return {
                    "status": probe_status,
                    "seed": int(seed),
                    "step": int(step),
                    "reason": exc.reason_code,
                    "selected_rank": selected_rank,
                    "rule": _compact_rule_debug(rule_debug),
                    "production": _compact_planner_debug(production_debug),
                    "dense_probe": _compact_planner_debug(
                        probe.get_last_debug() or {}
                    ),
                    "trace_tail": trace[-16:],
                }
            controller_debug = (
                expert.lqr_controller.get_last_debug()
                if expert.rule_maker.is_formation_locked
                else expert.pid_controller.get_last_debug()
            )
            trace.append(
                {
                    "step": int(step),
                    "controls": {
                        agent_id: np.asarray(control, dtype=np.float32).tolist()
                        for agent_id, control in expert_step.controls.items()
                    },
                    "trajectories_local": {
                        agent_id: np.asarray(trajectory, dtype=np.float32).tolist()
                        for agent_id, trajectory in (
                            expert_step.trajectories_local or {}
                        ).items()
                    },
                    "controller_debug": controller_debug,
                    "planner": _compact_planner_debug(
                        expert.planner.get_last_debug() or {}
                    ),
                    "rule": _compact_rule_debug(
                        expert.rule_maker.get_last_debug() or {}
                    ),
                    "agents": {
                        agent_id: {
                            "position": np.asarray(
                                env.agents[agent_id].position[:2],
                                dtype=np.float64,
                            ).tolist(),
                            "heading": float(env.agents[agent_id].heading_theta),
                            "speed_km_h": float(env.agents[agent_id].speed_km_h),
                            "steering": float(
                                getattr(env.agents[agent_id], "steering", 0.0)
                            ),
                        }
                        for agent_id in agent_ids
                    },
                }
            )
            env.low_level_step(dict(expert_step.controls))
        return {
            "status": "no_failure_within_horizon",
            "seed": int(seed),
            "max_steps": int(max_steps),
            "trace_tail": trace[-16:],
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset/data_collect_diagnostic64.yaml"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe(
        config_path=args.config,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=_json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
