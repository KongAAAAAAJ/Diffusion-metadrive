"""Small CPU smoke test for the time-varying steering CBF target."""
from __future__ import annotations

import torch

from models.bev_planner.time_varying_cbf import (
    steering_angle_xy,
    steering_cbf_safe_target,
    time_varying_steering_limit_deg,
)
from models.bev_planner.ddim_transition import DEFAULT_DDIM_PATH


WHEELBASE_M = 5.6
MIN_SEGMENT_LENGTH_M = 0.2


def main() -> int:
    print("DDIM timesteps:", DEFAULT_DDIM_PATH.timesteps)
    limits = [
        time_varying_steering_limit_deg(
            timestep=t,
            initial_timestep=DEFAULT_DDIM_PATH.initial_timestep,
            initial_limit_deg=48.24,
            final_limit_deg=24.13,
            schedule_power=1.0,
        )
        for t in DEFAULT_DDIM_PATH.timesteps
    ]
    print("steering limits [deg]:", [round(v, 5) for v in limits])

    straight = torch.stack(
        (torch.arange(8, dtype=torch.float32) * 5.0, torch.zeros(8)), dim=-1
    ).unsqueeze(0)
    straight_result = steering_cbf_safe_target(
        straight,
        steering_limit_deg=24.13,
        wheelbase_m=WHEELBASE_M,
        min_segment_length_m=MIN_SEGMENT_LENGTH_M,
        projection_passes=2,
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
    before = float(
        torch.rad2deg(
            steering_angle_xy(
                curved,
                wheelbase_m=WHEELBASE_M,
                min_segment_length_m=MIN_SEGMENT_LENGTH_M,
            ).abs().max()
        )
    )
    curved_result = steering_cbf_safe_target(
        curved,
        steering_limit_deg=24.13,
        wheelbase_m=WHEELBASE_M,
        min_segment_length_m=MIN_SEGMENT_LENGTH_M,
        projection_passes=8,
        max_correction_m=0.75,
    )
    after = float(
        torch.rad2deg(
            steering_angle_xy(
                curved_result.target_xy,
                wheelbase_m=WHEELBASE_M,
                min_segment_length_m=MIN_SEGMENT_LENGTH_M,
            ).abs().max()
        )
    )
    print(f"curved path max |steering|: {before:.5f} -> {after:.5f} deg")
    print("target correction RMS [m]:", float(curved_result.correction_rms_m.mean()))
    print(
        "target degenerate segment fraction:",
        float(curved_result.target_degenerate_segment_fraction.mean()),
    )
    assert after <= before + 1e-6
    assert bool(torch.isfinite(curved_result.target_xy).all())

    # Explicitly test a near-overlapping segment. It must be masked rather than
    # producing singular curvature/steering gradients.
    degenerate = straight.clone()
    degenerate[:, 2] = degenerate[:, 1] + torch.tensor([0.01, 0.0])
    degenerate_result = steering_cbf_safe_target(
        degenerate,
        steering_limit_deg=24.13,
        wheelbase_m=WHEELBASE_M,
        min_segment_length_m=MIN_SEGMENT_LENGTH_M,
        projection_passes=2,
    )
    assert bool(torch.isfinite(degenerate_result.target_xy).all())
    assert float(degenerate_result.nominal_degenerate_segment_fraction.mean()) > 0.0

    print("time-varying steering CBF smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
