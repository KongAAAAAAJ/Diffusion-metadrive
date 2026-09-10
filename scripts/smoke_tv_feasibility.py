"""Dependency-light smoke test for time-varying steering feasibility losses."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "models" / "bev_planner" / "time_varying_feasibility.py"
DDIM_TIMESTEPS = (8, 5, 3, 0)
INITIAL_TIMESTEP = DDIM_TIMESTEPS[0]

# Load the lightweight feasibility module directly so this smoke test does not
# require the full BEV planner stack (e.g. timm/diffusers/MetaDrive).
spec = importlib.util.spec_from_file_location("_tv_feasibility_smoke", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to load feasibility module: {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
steering_feasibility_loss = module.steering_feasibility_loss
time_varying_limit = module.time_varying_limit


def _limits(timestep: int) -> tuple[float, float]:
    steering = time_varying_limit(
        timestep=timestep,
        initial_timestep=INITIAL_TIMESTEP,
        initial_limit=48.24,
        final_limit=24.13,
        schedule_power=1.0,
    )
    rate = time_varying_limit(
        timestep=timestep,
        initial_timestep=INITIAL_TIMESTEP,
        initial_limit=120.0,
        final_limit=60.0,
        schedule_power=1.0,
    )
    return steering, rate


def main() -> int:
    print("DDIM timesteps:", DDIM_TIMESTEPS)
    schedule = [_limits(t) for t in DDIM_TIMESTEPS]
    print("steering limits [deg]:", [round(v[0], 3) for v in schedule])
    print("steering-rate limits [deg/s]:", [round(v[1], 3) for v in schedule])
    assert schedule[0][0] > schedule[-1][0]
    assert schedule[0][1] > schedule[-1][1]

    straight = torch.stack(
        (torch.arange(1, 9, dtype=torch.float32), torch.zeros(8)), dim=-1
    ).requires_grad_(True)
    result = steering_feasibility_loss(
        straight,
        steering_limit_deg=24.13,
        steering_rate_limit_deg_s=60.0,
        wheelbase_m=5.6,
        trajectory_dt_s=0.5,
        min_segment_length_m=0.2,
    )
    assert float(result.steering_loss.detach()) == 0.0
    assert float(result.steering_rate_loss.detach()) == 0.0
    assert float(result.degenerate_segment_fraction) == 0.0
    print("straight steering loss:", float(result.steering_loss.detach()))
    print("straight steering-rate loss:", float(result.steering_rate_loss.detach()))

    zigzag = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.9],
            [3.0, -0.9],
            [4.0, 0.9],
            [5.0, -0.9],
            [6.0, 0.9],
            [7.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    result = steering_feasibility_loss(
        zigzag,
        steering_limit_deg=24.13,
        steering_rate_limit_deg_s=60.0,
        wheelbase_m=5.6,
        trajectory_dt_s=0.5,
        min_segment_length_m=0.2,
    )
    total = result.steering_loss + result.steering_rate_loss
    assert float(total.detach()) > 0.0
    total.backward()
    assert zigzag.grad is not None
    assert bool(torch.isfinite(zigzag.grad).all())
    assert float(torch.linalg.vector_norm(zigzag.grad)) > 0.0
    print("zigzag steering loss:", float(result.steering_loss.detach()))
    print("zigzag steering-rate loss:", float(result.steering_rate_loss.detach()))
    print(
        "zigzag max |steering| [deg]:",
        math.degrees(float(result.max_abs_steering_rad)),
    )
    print(
        "zigzag max |steering rate| [deg/s]:",
        math.degrees(float(result.max_abs_steering_rate_rad_s)),
    )
    print("zigzag gradient norm:", float(torch.linalg.vector_norm(zigzag.grad)))

    degenerate = zigzag.detach().clone()
    degenerate[3] = degenerate[2]
    degenerate.requires_grad_(True)
    result = steering_feasibility_loss(
        degenerate,
        steering_limit_deg=24.13,
        steering_rate_limit_deg_s=60.0,
        wheelbase_m=5.6,
        trajectory_dt_s=0.5,
        min_segment_length_m=0.2,
    )
    total = result.steering_loss + result.steering_rate_loss
    total.backward()
    assert bool(torch.isfinite(total))
    assert degenerate.grad is not None and bool(torch.isfinite(degenerate.grad).all())
    assert float(result.degenerate_segment_fraction) > 0.0
    print("degenerate segment fraction:", float(result.degenerate_segment_fraction))
    print("time-varying steering feasibility smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
