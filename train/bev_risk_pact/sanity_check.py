"""Standalone gradient sanity check for Risk-PACT pilot.

Run from repository root:
    python -m train.bev_risk_pact.sanity_check
"""
from __future__ import annotations

import torch

from .config import RiskPACTConfig
from .constraint import RiskLevelSetConstraint
from .risk_field import DynamicGaussianRiskField
from .teacher import build_x0_pact_teacher


def main() -> None:
    torch.manual_seed(7)
    cfg = RiskPACTConfig(risk_threshold=0.35, teacher_step_m=0.20)
    field = DynamicGaussianRiskField(cfg)
    constraint = RiskLevelSetConstraint(cfg)

    # One role, one mode, 8 points. Ego moves along +x through a stationary actor.
    b, roles, modes, horizon, actors = 1, 1, 1, 8, 1
    x = torch.linspace(0.0, 14.0, horizon)
    y = torch.zeros_like(x)
    heading = torch.zeros_like(x)
    old = torch.stack((x, y, heading), dim=-1).reshape(b, roles, modes, horizon, 3)

    actor_state = torch.zeros((b, roles, actors, 8), dtype=torch.float32)
    actor_state[..., 0] = 8.0
    actor_state[..., 1] = 1.5
    actor_state[..., 2] = 1.0  # cos(yaw)
    actor_state[..., 6] = 4.8  # length
    actor_state[..., 7] = 2.0  # width
    valid = torch.ones((b, roles, actors), dtype=torch.bool)

    teacher = build_x0_pact_teacher(
        old,
        actor_state,
        valid,
        curriculum_scale=1.0,
        config=cfg,
        risk_field=field,
        constraint=constraint,
    )

    before = teacher.constraint.trajectory_risk.item()
    after_field = field.query(teacher.teacher_trajectory[..., :2], actor_state, valid)
    after = constraint(after_field.risk).trajectory_risk.item()
    print(f"risk_before={before:.6f}")
    print(f"risk_after ={after:.6f}")
    print(f"teacher_displacement_norm={teacher.displacement_norm.item():.6f} m")
    print(f"unsafe_or_near={bool(teacher.constraint.near_or_unsafe_mask.item())}")
    if teacher.constraint.near_or_unsafe_mask.item() and not after < before:
        raise SystemExit("sanity check failed: teacher step did not reduce risk")
    print("PASS: one x0-space PACT-lite step reduced the dynamic risk.")


if __name__ == "__main__":
    main()
