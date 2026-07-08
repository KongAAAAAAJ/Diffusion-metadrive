from __future__ import annotations

# !!!!还需验证稳定流形曲线画的对不对!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import math
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from models.platoon_planner.dynamics.truck_dynamics import (  # noqa: E402
    Dynamics,
    StabilityConfig,
    VehicleControl,
    VehicleParams,
    VehicleState,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "phase_planes"


@dataclass(frozen=True)
class Scenario:
    name: str
    state: VehicleState
    control: VehicleControl = field(default_factory=VehicleControl)
    expected_yaw_risk: Optional[str] = None
    expected_roll_risk: Optional[str] = None
    params: Optional[VehicleParams] = None
    config: Optional[StabilityConfig] = None
    yaw_beta_range: Optional[tuple[float, float]] = None
    yaw_rate_range: Optional[tuple[float, float]] = None
    roll_range: Optional[tuple[float, float]] = None
    roll_rate_range: Optional[tuple[float, float]] = None
    fixed_roll_ay: Optional[float] = None


def _finite_values(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _scenario_stability_label(yaw_result: dict, roll_result: dict) -> str:
    return "stable" if yaw_result["stable"] and roll_result["stable"] else "unstable"


def _run_scenario(scenario: Scenario) -> None:
    config = scenario.config or StabilityConfig()
    config = replace(config, equilibrium_grid_size=21, manifold_steps=300, manifold_dt=0.02)
    dynamics = Dynamics(params=scenario.params, config=config)
    yaw = dynamics.yaw_stability(scenario.state, scenario.control, method="engineering")
    roll = dynamics.roll_stability(scenario.state, scenario.control, method="engineering")
    yaw_manifold = dynamics.yaw_stability(scenario.state, scenario.control, method="manifold")
    roll_manifold = dynamics.roll_stability(scenario.state, scenario.control, method="manifold")

    energy_ratio = roll["roll_energy"] / max(roll["critical_energy"], dynamics.config.eps)
    engineering_stability = _scenario_stability_label(yaw, roll)
    manifold_stability = _scenario_stability_label(yaw_manifold, roll_manifold)
    print(f"\n[{scenario.name}]")
    print(
        "  scenario stability: "
        f"engineering={engineering_stability:<8} "
        f"manifold={manifold_stability:<8}"
    )
    print(
        "  condition: "
        f"vx={scenario.state.vx:.2f} m/s "
        f"steer={scenario.control.steer:+.3f} rad "
        f"mu={dynamics.params.friction_coeff:.2f} "
        f"tire={dynamics.config.tire_model}"
    )
    print(
        "  yaw : "
        f"risk={yaw['risk']:<12} "
        f"stable={yaw['stable']} "
        f"beta={yaw['beta']:+.4f} rad "
        f"yaw_rate={yaw['yaw_rate']:+.4f} rad/s "
        f"margin={yaw['margin']:+.4f}"
    )
    print(
        "  yaw(manifold): "
        f"risk={yaw_manifold['risk']:<12} "
        f"boundary={yaw_manifold['domain_status']:<22} "
        f"stability={yaw_manifold['stability_status']:<9} "
        f"source={yaw_manifold['decision_source']:<28} "
        f"stable_eq={yaw_manifold['stable_equilibrium']} "
        f"saddles={len(yaw_manifold['saddles'])} "
        f"branches={len(yaw_manifold['manifold_branches'])}"
    )
    print(
        "  roll: "
        f"risk={roll['risk']:<18} "
        f"stable={roll['stable']} "
        f"LTR={roll['ltr']:+.4f} "
        f"E/Ecrit={energy_ratio:.4f} "
        f"margin={roll['margin']:+.4f}"
    )
    print(
        "  roll(manifold): "
        f"risk={roll_manifold['risk']:<18} "
        f"boundary={roll_manifold['domain_status']:<22} "
        f"stability={roll_manifold['stability_status']:<9} "
        f"source={roll_manifold['decision_source']:<28} "
        f"stable_eq={roll_manifold['stable_equilibrium']} "
        f"saddles={len(roll_manifold['saddles'])} "
        f"branches={len(roll_manifold['manifold_branches'])}"
    )
    print(f"  yaw margins : {yaw_manifold['safety_margins']}")
    print(f"  roll margins: {roll_manifold['safety_margins']}")
    print(f"  Fz  : {roll['vertical_loads']}")
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in scenario.name)
    yaw_plot = dynamics.save_yaw_phase_plane_plot(
        scenario.state,
        scenario.control,
        output_path=OUTPUT_DIR / f"{safe_name}_yaw_phase_plane.png",
        title=f"{scenario.name} yaw phase plane",
        beta_range=scenario.yaw_beta_range,
        yaw_rate_range=scenario.yaw_rate_range,
    )
    roll_plot = dynamics.save_roll_phase_plane_plot(
        scenario.state,
        scenario.control,
        output_path=OUTPUT_DIR / f"{safe_name}_roll_phase_plane.png",
        title=f"{scenario.name} roll phase plane",
        roll_range=scenario.roll_range,
        roll_rate_range=scenario.roll_rate_range,
        fixed_ay=scenario.fixed_roll_ay,
    )
    print(f"  plots: {yaw_plot} | {roll_plot}")

    if scenario.expected_yaw_risk is not None:
        assert yaw["risk"] == scenario.expected_yaw_risk, (
            f"{scenario.name}: expected yaw risk {scenario.expected_yaw_risk}, "
            f"got {yaw['risk']}"
        )
    if scenario.expected_roll_risk is not None:
        assert roll["risk"] == scenario.expected_roll_risk, (
            f"{scenario.name}: expected roll risk {scenario.expected_roll_risk}, "
            f"got {roll['risk']}"
        )

    assert _finite_values(
        yaw["beta"],
        yaw["yaw_rate"],
        yaw["beta_dot"],
        yaw["yaw_rate_dot"],
        yaw["margin"],
        roll["ltr"],
        roll["roll_energy"],
        roll["critical_energy"],
        roll["margin"],
        yaw_manifold["margin"],
        roll_manifold["margin"],
    ), f"{scenario.name}: yaw/roll result contains NaN or Inf"


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="straight_stable",
            state=VehicleState(vx=15.0, vy=0.02, yaw_rate=0.005, roll=0.002, roll_rate=0.0),
            expected_yaw_risk="stable",
            expected_roll_risk="stable",
        ),
        Scenario(
            name="yaw_unstable_large_beta",
            state=VehicleState(vx=12.0, vy=4.0, yaw_rate=0.55),
            expected_yaw_risk="yaw_unstable",
            expected_roll_risk="stable",
            config=StabilityConfig(beta_limit=0.15, yaw_rate_limit=0.45),
        ),
        Scenario(
            name="static_roll_risk_high_ltr",
            state=VehicleState(vx=20.0, vy=0.0, yaw_rate=0.55, roll=0.02, roll_rate=0.0),
            expected_yaw_risk="stable",
            expected_roll_risk="static_roll_risk",
            params=VehicleParams(cg_height=1.8, track_width_front=1.8, track_width_rear=1.8),
            config=StabilityConfig(ltr_limit=0.70),
        ),
        Scenario(
            name="dynamic_roll_risk_high_roll_rate",
            state=VehicleState(vx=8.0, vy=0.0, yaw_rate=0.0, roll=0.04, roll_rate=3.0),
            expected_yaw_risk="stable",
            expected_roll_risk="dynamic_roll_risk",
            params=VehicleParams(cg_height=1.5, track_width_front=2.0, track_width_rear=2.0),
            config=StabilityConfig(ltr_limit=0.95, energy_margin=0.90),
        ),
        Scenario(
            name="low_speed_numeric_safety",
            state=VehicleState(vx=1e-5, vy=0.2, yaw_rate=0.1, roll=0.05, roll_rate=0.1),
        ),
        Scenario(
            name="aggressive_yaw_mid_mu_high_speed",
            state=VehicleState(vx=28.0, vy=1.2, yaw_rate=0.32),
            control=VehicleControl(steer=0.16),
            expected_yaw_risk="yaw_unstable",
            params=VehicleParams(friction_coeff=0.65),
            config=StabilityConfig(beta_limit=0.12, yaw_rate_limit=0.30, manifold_bounds_scale=3.5),
            yaw_beta_range=(-0.8, 0.8),
            yaw_rate_range=(-2.0, 2.0),
        ),
        Scenario(
            name="aggressive_yaw_low_mu_high_speed",
            state=VehicleState(vx=30.0, vy=1.5, yaw_rate=0.35),
            control=VehicleControl(steer=0.20),
            expected_yaw_risk="yaw_unstable",
            params=VehicleParams(friction_coeff=0.35),
            config=StabilityConfig(beta_limit=0.10, yaw_rate_limit=0.28, manifold_bounds_scale=4.0),
            yaw_beta_range=(-0.8, 0.8),
            yaw_rate_range=(-2.0, 2.0),
        ),
        Scenario(
            name="fixed_ay_soft_roll_scan_seed",
            state=VehicleState(vx=18.0, vy=0.0, yaw_rate=0.0, roll=0.08, roll_rate=0.4),
            params=VehicleParams(
                cg_height=1.95,
                sprung_cg_height=1.55,
                roll_stiffness=260000.0,
                track_width_front=1.9,
                track_width_rear=1.9,
            ),
            config=StabilityConfig(ltr_limit=0.80, energy_margin=0.85),
            roll_range=(-0.8, 0.8),
            roll_rate_range=(-6.0, 6.0),
            fixed_roll_ay=0.6 * 9.81,
        ),
    ]


def run_yaw_parameter_scan() -> None:
    vx_list = [22.0, 28.0, 32.0]
    steer_list = [0.10, 0.16, 0.22]
    mu_list = [0.35, 0.65]
    for vx in vx_list:
        for steer in steer_list:
            for mu in mu_list:
                scenario = Scenario(
                    name=f"scan_yaw_vx_{vx:.0f}_delta_{steer:.2f}_mu_{mu:.2f}",
                    state=VehicleState(vx=vx, vy=0.8, yaw_rate=0.18),
                    control=VehicleControl(steer=steer),
                    params=VehicleParams(friction_coeff=mu),
                    config=StabilityConfig(
                        beta_limit=0.12,
                        yaw_rate_limit=0.35,
                        equilibrium_grid_size=17,
                        manifold_steps=220,
                        manifold_dt=0.02,
                        manifold_bounds_scale=4.0,
                    ),
                    yaw_beta_range=(-0.8, 0.8),
                    yaw_rate_range=(-2.0, 2.0),
                )
                _run_scenario(scenario)


def run_fixed_ay_roll_scan() -> None:
    params = VehicleParams(
        cg_height=2.05,
        sprung_cg_height=1.65,
        roll_stiffness=240000.0,
        track_width_front=1.9,
        track_width_rear=1.9,
    )
    for ay_g in (0.2, 0.4, 0.6, 0.8):
        scenario = Scenario(
            name=f"scan_roll_fixed_ay_{ay_g:.1f}g",
            state=VehicleState(vx=18.0, vy=0.0, yaw_rate=0.0, roll=0.04, roll_rate=0.2),
            params=params,
            config=StabilityConfig(
                ltr_limit=0.80,
                energy_margin=0.85,
                equilibrium_grid_size=17,
                manifold_steps=220,
                manifold_dt=0.02,
            ),
            roll_range=(-0.9, 0.9),
            roll_rate_range=(-6.0, 6.0),
            fixed_roll_ay=ay_g * params.gravity,
        )
        _run_scenario(scenario)


def main() -> None:
    for scenario in build_scenarios():
        _run_scenario(scenario)
    run_yaw_parameter_scan()
    run_fixed_ay_roll_scan()
    print("\nAll Dynamics stability scenarios passed.")


if __name__ == "__main__":
    main()
