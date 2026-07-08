from __future__ import annotations

from pathlib import Path

import numpy as np

from models.platoon_planner.dynamics.truck_dynamics import (
    Dynamics,
    StabilityConfig,
    VehicleControl,
    VehicleParams,
    VehicleState,
)
from models.platoon_planner.dynamics._test_truck_dynamics import build_scenarios, _scenario_stability_label


def test_straight_small_disturbance_is_stable():
    dynamics = Dynamics()
    state = VehicleState(vx=15.0, vy=0.02, yaw_rate=0.005, roll=0.002, roll_rate=0.0)

    yaw = dynamics.yaw_stability(state)
    roll = dynamics.roll_stability(state)

    assert yaw["stable"] is True
    assert yaw["risk"] == "stable"
    assert roll["stable"] is True
    assert roll["risk"] == "stable"
    assert {"stable", "risk", "margin", "phase_norm"}.issubset(yaw)
    assert {"stable", "risk", "margin", "ltr", "vertical_loads"}.issubset(roll)


def test_scenario_stability_label_requires_yaw_and_roll_stable():
    assert _scenario_stability_label({"stable": True}, {"stable": True}) == "stable"
    assert _scenario_stability_label({"stable": False}, {"stable": True}) == "unstable"
    assert _scenario_stability_label({"stable": True}, {"stable": False}) == "unstable"


def test_manual_scenarios_include_nonzero_steer_controls():
    scenarios = build_scenarios()

    assert any(abs(scenario.control.steer) > 0.0 for scenario in scenarios)


def test_large_sideslip_or_yaw_rate_is_yaw_unstable():
    dynamics = Dynamics(config=StabilityConfig(beta_limit=0.15, yaw_rate_limit=0.45))
    state = VehicleState(vx=12.0, vy=4.0, yaw_rate=0.55)

    result = dynamics.yaw_stability(state)

    assert result["stable"] is False
    assert result["risk"] == "yaw_unstable"
    assert result["margin"] < 0.0
    assert abs(result["beta"]) > 0.15


def test_roll_stability_reports_static_ltr_risk_first():
    params = VehicleParams(cg_height=1.8, track_width_front=1.8, track_width_rear=1.8)
    dynamics = Dynamics(params=params, config=StabilityConfig(ltr_limit=0.70))
    state = VehicleState(vx=20.0, vy=0.0, yaw_rate=0.55, roll=0.02, roll_rate=0.0)

    result = dynamics.roll_stability(state)

    assert result["stable"] is False
    assert result["risk"] == "static_roll_risk"
    assert abs(result["ltr"]) > 0.70


def test_roll_stability_reports_dynamic_energy_risk_when_ltr_is_safe():
    params = VehicleParams(cg_height=1.5, track_width_front=2.0, track_width_rear=2.0)
    dynamics = Dynamics(params=params, config=StabilityConfig(ltr_limit=0.95, energy_margin=0.90))
    state = VehicleState(vx=8.0, vy=0.0, yaw_rate=0.0, roll=0.04, roll_rate=3.0)

    result = dynamics.roll_stability(state)

    assert result["stable"] is False
    assert result["risk"] == "dynamic_roll_risk"
    assert abs(result["ltr"]) < 0.95
    assert result["roll_energy"] >= 0.90 * result["critical_energy"]


def test_low_speed_state_does_not_produce_nan_or_inf():
    dynamics = Dynamics()
    state = VehicleState(vx=1e-5, vy=0.2, yaw_rate=0.1, roll=0.05, roll_rate=0.1)

    yaw = dynamics.yaw_stability(state)
    roll = dynamics.roll_stability(state)
    state_dot, aux = dynamics.state_derivative(state)

    values = [
        yaw["beta"],
        yaw["beta_dot"],
        yaw["yaw_rate_dot"],
        roll["ltr"],
        roll["roll_energy"],
        roll["critical_energy"],
        state_dot.vx,
        state_dot.vy,
        state_dot.yaw_rate,
        state_dot.roll_rate,
        aux["lateral_accel"],
    ]
    values.extend(state_dot.wheel_omegas)

    assert np.all(np.isfinite(values))


def test_default_tire_model_is_fiala_and_saturates_lateral_force():
    params = VehicleParams(friction_coeff=0.35)
    dynamics = Dynamics(params=params)
    state = VehicleState(vx=20.0, vy=8.0, yaw_rate=0.4)

    _, aux = dynamics.state_derivative(state)

    assert dynamics.config.tire_model == "fiala"
    for name, force in aux["tire_forces"].items():
        normal = max(aux["vertical_loads"][name], 0.0)
        force_norm = np.hypot(force["fx"], force["fy"])
        assert np.isfinite(force_norm)
        assert force_norm <= params.friction_coeff * normal + 1e-6
    assert any(abs(force["fy"]) > 0.9 * params.friction_coeff * max(aux["vertical_loads"][name], 0.0)
               for name, force in aux["tire_forces"].items())


def test_nonzero_steer_low_mu_raises_yaw_tire_utilization():
    params = VehicleParams(friction_coeff=0.35)
    dynamics = Dynamics(
        params=params,
        config=StabilityConfig(
            beta_limit=0.18,
            yaw_rate_limit=0.50,
            equilibrium_grid_size=13,
            manifold_steps=80,
            manifold_dt=0.02,
            manifold_bounds_scale=3.0,
        ),
    )
    state = VehicleState(vx=28.0, vy=1.5, yaw_rate=0.35)
    control = VehicleControl(steer=0.18)

    result = dynamics.yaw_stability(state, control=control, method="manifold")

    assert result["stability_status"] in {"warning", "unstable", "high_risk"}
    assert result["safety_margins"]["max_tire_utilization"] > 0.85


def test_roll_boundary_can_use_fixed_lateral_acceleration():
    dynamics = Dynamics()
    state = VehicleState(vx=0.0, vy=0.0, yaw_rate=0.0)
    fixed_ay = 0.6 * dynamics.params.gravity

    boundary = dynamics.compute_roll_stable_manifold_boundary(state, fixed_ay=fixed_ay)

    assert boundary["lateral_accel"] == fixed_ay
    assert boundary["bounds"][1] == (-6.0, 6.0)


def _fast_manifold_dynamics() -> Dynamics:
    return Dynamics(
        config=StabilityConfig(
            equilibrium_grid_size=13,
            manifold_steps=80,
            manifold_dt=0.02,
        )
    )


def test_yaw_manifold_method_returns_equilibrium_and_boundary_fields():
    dynamics = _fast_manifold_dynamics()
    state = VehicleState(vx=15.0, vy=0.02, yaw_rate=0.005, roll=0.002, roll_rate=0.0)

    result = dynamics.yaw_stability(state, method="manifold")

    assert result["method"] == "manifold"
    assert result["domain_status"] in {
        "manifold_boundary",
        "no_saddle_boundary",
        "no_stable_equilibrium",
    }
    assert "stable_equilibrium" in result
    assert "saddles" in result
    assert "manifold_branches" in result
    assert result["stability_status"] in {"stable", "warning", "unstable", "high_risk"}
    assert result["decision_source"] in {
        "manifold_boundary",
        "local_equilibrium_margin",
        "no_stable_equilibrium_safety",
    }
    assert "safety_margins" in result
    if result["domain_status"] == "no_saddle_boundary":
        assert result["stability_status"] == "stable"
        assert result["stable"] is True
    if result["stable_equilibrium"] is not None:
        assert np.all(np.isfinite(result["stable_equilibrium"]))
    for branch in result["manifold_branches"]:
        assert branch
        assert np.all(np.isfinite(np.asarray(branch, dtype=float)))


def test_roll_manifold_method_reports_stable_equilibrium_for_small_roll():
    dynamics = _fast_manifold_dynamics()
    state = VehicleState(vx=15.0, vy=0.02, yaw_rate=0.005, roll=0.002, roll_rate=0.0)

    result = dynamics.roll_stability(state, method="manifold")

    assert result["method"] == "manifold"
    assert result["domain_status"] in {
        "manifold_boundary",
        "no_saddle_boundary",
        "no_stable_equilibrium",
    }
    assert result["stable_equilibrium"] is not None
    assert np.all(np.isfinite(result["stable_equilibrium"]))
    assert "saddles" in result
    assert "manifold_branches" in result
    assert result["stability_status"] in {"stable", "warning", "unstable", "high_risk"}
    assert result["decision_source"] in {
        "manifold_boundary",
        "local_equilibrium_margin",
        "no_stable_equilibrium_safety",
    }
    assert "safety_margins" in result
    if result["domain_status"] == "no_saddle_boundary":
        assert result["stability_status"] == "stable"
        assert result["stable"] is True


def test_roll_manifold_method_flags_dynamic_roll_risk():
    params = VehicleParams(cg_height=1.5, track_width_front=2.0, track_width_rear=2.0)
    dynamics = Dynamics(
        params=params,
        config=StabilityConfig(
            ltr_limit=0.95,
            energy_margin=0.90,
            equilibrium_grid_size=13,
            manifold_steps=80,
            manifold_dt=0.02,
        ),
    )
    state = VehicleState(vx=8.0, vy=0.0, yaw_rate=0.0, roll=0.04, roll_rate=3.0)

    result = dynamics.roll_stability(state, method="manifold")

    assert result["method"] == "manifold"
    assert result["stable"] is False
    assert result["risk"] == "dynamic_roll_risk"
    assert "domain_status" in result
    assert result["stability_status"] in {"warning", "unstable", "high_risk"}
    assert "safety_margins" in result


def test_yaw_manifold_large_sideslip_uses_stability_status_not_boundary_status():
    dynamics = _fast_manifold_dynamics()
    dynamics.config = StabilityConfig(
        beta_limit=0.15,
        yaw_rate_limit=0.45,
        equilibrium_grid_size=13,
        manifold_steps=80,
        manifold_dt=0.02,
    )
    state = VehicleState(vx=12.0, vy=4.0, yaw_rate=0.55)

    result = dynamics.yaw_stability(state, method="manifold")

    assert result["method"] == "manifold"
    assert result["stability_status"] in {"warning", "unstable", "high_risk"}
    assert result["stable"] is False
    assert result["risk"] == "yaw_unstable"
    assert "domain_status" in result
    assert "decision_source" in result
    assert "distance_to_stable_equilibrium" in result["safety_margins"]


def test_yaw_manifold_no_stable_equilibrium_is_high_risk(monkeypatch):
    dynamics = _fast_manifold_dynamics()
    state = VehicleState(vx=15.0, vy=0.02, yaw_rate=0.005)

    def no_stable_boundary(_state, _control=None):
        return {
            "status": "no_stable_equilibrium",
            "stable_equilibria": [],
            "saddles": [],
            "classified_equilibria": [],
            "manifold_branches": [],
            "bounds": ((-0.3, 0.3), (-1.0, 1.0)),
            "vector_field": lambda beta, yaw_rate: (0.0, 0.0),
        }

    monkeypatch.setattr(dynamics, "compute_yaw_stable_manifold_boundary", no_stable_boundary)

    result = dynamics.yaw_stability(state, method="manifold")

    assert result["domain_status"] == "no_stable_equilibrium"
    assert result["stability_status"] == "high_risk"
    assert result["decision_source"] == "no_stable_equilibrium_safety"
    assert result["stable"] is False


def test_stable_manifold_boundary_helpers_return_expected_structure():
    dynamics = _fast_manifold_dynamics()
    state = VehicleState(vx=15.0, vy=0.02, yaw_rate=0.005, roll=0.002, roll_rate=0.0)

    yaw_boundary = dynamics.compute_yaw_stable_manifold_boundary(state)
    roll_boundary = dynamics.compute_roll_stable_manifold_boundary(state)

    assert {"status", "stable_equilibria", "saddles", "manifold_branches"}.issubset(yaw_boundary)
    assert {"status", "stable_equilibria", "saddles", "manifold_branches", "phi_crit"}.issubset(
        roll_boundary
    )
    assert yaw_boundary["status"] in {
        "manifold_boundary",
        "no_saddle_boundary",
        "no_stable_equilibrium",
    }
    assert roll_boundary["status"] in {
        "manifold_boundary",
        "no_saddle_boundary",
        "no_stable_equilibrium",
    }


def test_phase_plane_plot_helpers_save_nonempty_png(tmp_path: Path):
    dynamics = _fast_manifold_dynamics()
    state = VehicleState(vx=15.0, vy=0.02, yaw_rate=0.005, roll=0.002, roll_rate=0.0)
    yaw_path = tmp_path / "yaw_phase_plane.png"
    roll_path = tmp_path / "roll_phase_plane.png"

    saved_yaw = dynamics.save_yaw_phase_plane_plot(state, output_path=yaw_path)
    saved_roll = dynamics.save_roll_phase_plane_plot(state, output_path=roll_path)

    assert saved_yaw == yaw_path
    assert saved_roll == roll_path
    assert yaw_path.exists() and yaw_path.stat().st_size > 0
    assert roll_path.exists() and roll_path.stat().st_size > 0
    for path in (yaw_path, roll_path):
        assert path.with_suffix(".pdf").exists()
        assert path.with_suffix(".pdf").stat().st_size > 0
        assert path.with_suffix(".svg").exists()
        assert path.with_suffix(".svg").stat().st_size > 0
    roll_svg = roll_path.with_suffix(".svg").read_text(encoding="utf-8")
    assert "stability:" in roll_svg
    assert "status:" not in roll_svg
    assert "current -&gt; stable eq." in roll_svg or "current -> stable eq." in roll_svg
    assert "crit" in roll_svg


def test_trace_forward_trajectory_moves_toward_stable_equilibrium():
    dynamics = Dynamics(config=StabilityConfig(manifold_steps=20, manifold_dt=0.05))

    def vector_field(x: float, y: float) -> tuple[float, float]:
        return -x, -y

    trajectory = dynamics._trace_forward_trajectory(
        vector_field,
        initial=(0.5, -0.25),
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        equilibrium=(0.0, 0.0),
    )

    assert len(trajectory) > 2
    assert np.all(np.isfinite(np.asarray(trajectory, dtype=float)))
    assert dynamics._distance(trajectory[-1], (0.0, 0.0)) < dynamics._distance(
        trajectory[0], (0.0, 0.0)
    )


def test_trace_stable_manifold_expands_from_synthetic_saddle():
    dynamics = Dynamics(config=StabilityConfig(manifold_steps=12, manifold_dt=0.05))

    def vector_field(x: float, y: float) -> tuple[float, float]:
        return x, -y

    classified = dynamics._classified_equilibria(vector_field, [(0.0, 0.0)])
    branches = dynamics._branches_from_saddles(
        vector_field,
        [item for item in classified if item["kind"] == "saddle"],
        ((-1.0, 1.0), (-1.0, 1.0)),
    )

    assert classified[0]["kind"] == "saddle"
    assert len(branches) == 2
    assert all(len(branch) > 2 for branch in branches)
    assert np.all(np.isfinite(np.asarray(branches, dtype=float)))


def test_phase_plane_plot_draws_stable_manifold_in_orange(tmp_path: Path):
    dynamics = Dynamics(config=StabilityConfig(manifold_steps=12, manifold_dt=0.05))

    def vector_field(x: float, y: float) -> tuple[float, float]:
        return x, -y

    classified = dynamics._classified_equilibria(vector_field, [(0.0, 0.0)])
    saddles = [item for item in classified if item["kind"] == "saddle"]
    branches = dynamics._branches_from_saddles(vector_field, saddles, ((-1.0, 1.0), (-1.0, 1.0)))
    output_path = tmp_path / "synthetic_manifold.png"
    boundary = {
        "bounds": ((-1.0, 1.0), (-1.0, 1.0)),
        "vector_field": vector_field,
        "manifold_branches": branches,
        "stable_equilibria": [],
        "saddles": [item["point"] for item in saddles],
    }

    dynamics._save_phase_plane_plot(
        boundary,
        current=(0.2, 0.2),
        output_path=output_path,
        xlabel="x",
        ylabel="y",
        title="synthetic saddle",
    )

    svg = output_path.with_suffix(".svg").read_text(encoding="utf-8")
    assert "stable manifold" in svg
    assert "#f28e2b" in svg
