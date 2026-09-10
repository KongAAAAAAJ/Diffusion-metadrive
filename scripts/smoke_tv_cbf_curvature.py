"""Small CPU smoke test for the time-varying curvature CBF target."""
from __future__ import annotations

import torch

from models.bev_planner.time_varying_cbf import (
    curvature_cbf_safe_target,
    discrete_curvature_xy,
    time_varying_curvature_limit,
)
from models.bev_planner.ddim_transition import DEFAULT_DDIM_PATH


def main() -> int:
    print("DDIM timesteps:", DEFAULT_DDIM_PATH.timesteps)
    limits = [
        time_varying_curvature_limit(
            timestep=t,
            initial_timestep=DEFAULT_DDIM_PATH.initial_timestep,
            initial_limit_inv_m=0.20,
            final_limit_inv_m=0.08,
            schedule_power=1.0,
        )
        for t in DEFAULT_DDIM_PATH.timesteps
    ]
    print("curvature limits [1/m]:", [round(v, 5) for v in limits])

    straight = torch.stack(
        (torch.arange(8, dtype=torch.float32) * 5.0, torch.zeros(8)), dim=-1
    ).unsqueeze(0)
    straight_result = curvature_cbf_safe_target(
        straight, curvature_limit_inv_m=0.08, projection_passes=2
    )
    assert torch.allclose(straight_result.target_xy, straight)

    curved = torch.tensor(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 2.5],
            [15.0, 5.0],
            [20.0, 4.0],
            [25.0, 2.0],
            [30.0, 1.0],
            [35.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    before = float(discrete_curvature_xy(curved).abs().max())
    curved_result = curvature_cbf_safe_target(
        curved,
        curvature_limit_inv_m=0.08,
        projection_passes=4,
        max_correction_m=0.75,
    )
    after = float(discrete_curvature_xy(curved_result.target_xy).abs().max())
    print(f"curved path max |kappa|: {before:.5f} -> {after:.5f} 1/m")
    print(
        "target correction RMS [m]:",
        float(curved_result.correction_rms_m.mean()),
    )
    assert after <= before + 1e-6
    print("time-varying curvature CBF smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
