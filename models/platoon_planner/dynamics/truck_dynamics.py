from __future__ import annotations

"""
 - Engineering method: 用于实时计算稳定性判据
 - Manifold method: 用于离线分析，计算稳定性边界和相平面图
"""

import cmath
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping


WHEEL_ORDER = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class VehicleParams:
    """Truck parameters for the compact 8DOF model."""

    mass: float = 12000.0
    sprung_mass: float = 10000.0
    yaw_inertia: float = 52000.0
    roll_inertia: float = 18000.0
    lf: float = 2.2
    lr: float = 3.4
    track_width_front: float = 2.05
    track_width_rear: float = 2.05
    cg_height: float = 1.35
    sprung_cg_height: float = 1.10
    roll_stiffness: float = 520000.0
    roll_damping: float = 55000.0
    wheel_radius: float = 0.52
    wheel_inertia: float = 18.0
    cornering_stiffness_front: float = 130000.0
    cornering_stiffness_rear: float = 180000.0
    friction_coeff: float = 0.85
    gravity: float = 9.81

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @property
    def track_width(self) -> float:
        return 0.5 * (self.track_width_front + self.track_width_rear)

    @property
    def effective_roll_stiffness(self) -> float:
        return self.roll_stiffness - self.sprung_mass * self.gravity * self.sprung_cg_height


@dataclass(frozen=True)
class VehicleState:
    """8DOF state plus roll rate required by the second-order roll equation."""

    vx: float
    vy: float
    yaw_rate: float
    roll: float = 0.0
    roll_rate: float = 0.0
    wheel_omegas: tuple[float, float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0)
    )

    def __post_init__(self) -> None:
        if len(self.wheel_omegas) != 4:
            raise ValueError("wheel_omegas must contain fl, fr, rl, rr values")
        object.__setattr__(self, "wheel_omegas", tuple(float(v) for v in self.wheel_omegas))


@dataclass(frozen=True)
class VehicleControl:
    steer: float = 0.0
    wheel_torques: tuple[float, float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0)
    )

    def __post_init__(self) -> None:
        if len(self.wheel_torques) != 4:
            raise ValueError("wheel_torques must contain fl, fr, rl, rr values")
        object.__setattr__(self, "wheel_torques", tuple(float(v) for v in self.wheel_torques))


@dataclass(frozen=True)
class StabilityConfig:
    tire_model: str = "fiala"
    ltr_limit: float = 0.85
    energy_margin: float = 1.0
    beta_limit: float = 0.22
    yaw_rate_limit: float = 0.70
    beta_dot_limit: float = 1.25
    yaw_accel_limit: float = 2.5
    eps: float = 1e-6
    manifold_dt: float = 0.01
    manifold_steps: int = 1200
    manifold_eps: float = 1e-4
    manifold_bounds_scale: float = 1.5
    equilibrium_grid_size: int = 41
    equilibrium_tol: float = 1e-4
    jacobian_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.tire_model not in {"linear", "fiala"}:
            raise ValueError("tire_model must be 'linear' or 'fiala'")


class Dynamics:
    """Compact 8DOF truck dynamics and phase-plane stability checks."""

    def __init__(
        self,
        params: VehicleParams | None = None,
        config: StabilityConfig | None = None,
    ) -> None:
        self.params = params or VehicleParams()
        self.config = config or StabilityConfig()

    def state_derivative(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
    ) -> tuple[VehicleState, dict[str, Any]]:
        state = self._coerce_state(state)
        control = self._coerce_control(control)
        p = self.params

        kinematics = self._wheel_kinematics(state, control.steer)
        load_ay = state.vx * state.yaw_rate
        vertical_loads = self._vertical_loads(state, ax=0.0, ay=load_ay)
        tire_forces = self._tire_forces(kinematics, vertical_loads, state, control)

        front_fx = tire_forces["fl"]["fx"] + tire_forces["fr"]["fx"]
        front_fy = tire_forces["fl"]["fy"] + tire_forces["fr"]["fy"]
        rear_fx = tire_forces["rl"]["fx"] + tire_forces["rr"]["fx"]
        rear_fy = tire_forces["rl"]["fy"] + tire_forces["rr"]["fy"]
        steer = control.steer
        cos_delta = math.cos(steer)
        sin_delta = math.sin(steer)

        body_fx = front_fx * cos_delta - front_fy * sin_delta + rear_fx
        body_fy = front_fx * sin_delta + front_fy * cos_delta + rear_fy
        yaw_moment = self._yaw_moment(tire_forces, steer)

        vx_dot = state.yaw_rate * state.vy + body_fx / p.mass
        vy_dot = -state.yaw_rate * state.vx + body_fy / p.mass
        yaw_rate_dot = yaw_moment / p.yaw_inertia
        lateral_accel = vy_dot + state.vx * state.yaw_rate

        roll_accel = (
            p.sprung_mass * p.sprung_cg_height * lateral_accel
            + p.sprung_mass * p.gravity * p.sprung_cg_height * math.sin(state.roll)
            - p.roll_stiffness * state.roll
            - p.roll_damping * state.roll_rate
        ) / max(p.roll_inertia, self.config.eps)

        wheel_dot = tuple(
            (control.wheel_torques[idx] - p.wheel_radius * tire_forces[name]["fx"])
            / max(p.wheel_inertia, self.config.eps)
            for idx, name in enumerate(WHEEL_ORDER)
        )

        state_dot = VehicleState(
            vx=vx_dot,
            vy=vy_dot,
            yaw_rate=yaw_rate_dot,
            roll=state.roll_rate,
            roll_rate=roll_accel,
            wheel_omegas=wheel_dot,
        )
        aux = {
            "slip_angles": {name: tire_forces[name]["alpha"] for name in WHEEL_ORDER},
            "slip_ratios": {name: tire_forces[name]["slip_ratio"] for name in WHEEL_ORDER},
            "tire_forces": {
                name: {"fx": tire_forces[name]["fx"], "fy": tire_forces[name]["fy"]}
                for name in WHEEL_ORDER
            },
            "vertical_loads": vertical_loads,
            "yaw_moment": yaw_moment,
            "body_fx": body_fx,
            "body_fy": body_fy,
            "lateral_accel": lateral_accel,
            "load_lateral_accel": load_ay,
        }
        return state_dot, aux

    def yaw_stability(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
        method: str = "engineering",
    ) -> dict[str, Any]:
        if method == "manifold":
            return self._yaw_stability_manifold(state, control)
        if method != "engineering":
            raise ValueError("method must be 'engineering' or 'manifold'")
        return self._yaw_stability_engineering(state, control)

    def _yaw_stability_engineering(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._coerce_state(state)
        state_dot, _ = self.state_derivative(state, control)
        cfg = self.config

        beta = math.atan2(state.vy, self._signed_safe_vx(state.vx))
        speed_sq = max(state.vx * state.vx + state.vy * state.vy, cfg.eps)
        beta_dot = (state.vx * state_dot.vy - state.vy * state_dot.vx) / speed_sq
        beta_norm = abs(beta) / max(cfg.beta_limit, cfg.eps)
        yaw_norm = abs(state.yaw_rate) / max(cfg.yaw_rate_limit, cfg.eps)
        beta_dot_norm = abs(beta_dot) / max(cfg.beta_dot_limit, cfg.eps)
        yaw_accel_norm = abs(state_dot.yaw_rate) / max(cfg.yaw_accel_limit, cfg.eps)
        phase_norm = max(beta_norm, yaw_norm, 0.5 * beta_dot_norm, 0.5 * yaw_accel_norm)
        margin = 1.0 - phase_norm
        stable = margin >= 0.0

        return {
            "stable": stable,
            "risk": "stable" if stable else "yaw_unstable",
            "method": "engineering",
            "beta": beta,
            "yaw_rate": state.yaw_rate,
            "beta_dot": beta_dot,
            "yaw_rate_dot": state_dot.yaw_rate,
            "margin": margin,
            "phase_norm": phase_norm,
        }

    def roll_stability(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
        method: str = "engineering",
    ) -> dict[str, Any]:
        if method == "manifold":
            return self._roll_stability_manifold(state, control)
        if method != "engineering":
            raise ValueError("method must be 'engineering' or 'manifold'")
        return self._roll_stability_engineering(state, control)

    def _roll_stability_engineering(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._coerce_state(state)
        _, aux = self.state_derivative(state, control)
        vertical_loads = aux["vertical_loads"]
        ltr = self._ltr(vertical_loads)
        roll_energy, critical_energy, phi_crit = self._roll_energy(
            state.roll,
            state.roll_rate,
            aux["lateral_accel"],
        )

        static_margin = self.config.ltr_limit - abs(ltr)
        energy_limit = self.config.energy_margin * critical_energy
        dynamic_margin = 1.0 - roll_energy / max(energy_limit, self.config.eps)

        if static_margin < 0.0:
            stable = False
            risk = "static_roll_risk"
            margin = static_margin
        elif dynamic_margin < 0.0:
            stable = False
            risk = "dynamic_roll_risk"
            margin = dynamic_margin
        else:
            stable = True
            risk = "stable"
            margin = min(static_margin, dynamic_margin)

        return {
            "stable": stable,
            "risk": risk,
            "method": "engineering",
            "ltr": ltr,
            "roll": state.roll,
            "roll_rate": state.roll_rate,
            "roll_energy": roll_energy,
            "critical_energy": critical_energy,
            "margin": margin,
            "vertical_loads": vertical_loads,
            "phi_crit": phi_crit,
        }

    def _yaw_stability_manifold(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._coerce_state(state)
        control = self._coerce_control(control)
        engineering = self._yaw_stability_engineering(state, control)
        boundary = self.compute_yaw_stable_manifold_boundary(state, control)
        result = dict(engineering)
        result.update(
            {
                "method": "manifold",
                "domain_status": boundary["status"],
                "stable_equilibrium": self._first_or_none(boundary["stable_equilibria"]),
                "saddles": boundary["saddles"],
                "manifold_branches": boundary["manifold_branches"],
                "terminal_error": None,
            }
        )
        stable_eq = result["stable_equilibrium"]
        if stable_eq is not None:
            beta = result["beta"]
            yaw_rate = result["yaw_rate"]
            safety_check = self._finite_time_phase_safety_check(
                boundary["vector_field"], (beta, yaw_rate), tuple(stable_eq), boundary["bounds"]
            )
            result["terminal_error"] = safety_check["terminal_error"]
            result["terminal_state"] = safety_check["terminal_state"]
        else:
            safety_check = None
        return self._apply_yaw_manifold_decision(state, control, result, boundary, safety_check)

    def _roll_stability_manifold(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._coerce_state(state)
        control = self._coerce_control(control)
        engineering = self._roll_stability_engineering(state, control)
        boundary = self.compute_roll_stable_manifold_boundary(state, control)
        result = dict(engineering)
        result.update(
            {
                "method": "manifold",
                "domain_status": boundary["status"],
                "stable_equilibrium": self._first_or_none(boundary["stable_equilibria"]),
                "saddles": boundary["saddles"],
                "manifold_branches": boundary["manifold_branches"],
                "terminal_error": None,
            }
        )
        stable_eq = result["stable_equilibrium"]
        if stable_eq is not None:
            safety_check = self._finite_time_phase_safety_check(
                boundary["vector_field"],
                (state.roll, state.roll_rate),
                tuple(stable_eq),
                boundary["bounds"],
            )
            result["terminal_error"] = safety_check["terminal_error"]
            result["terminal_state"] = safety_check["terminal_state"]
        else:
            safety_check = None
        return self._apply_roll_manifold_decision(state, control, result, boundary, safety_check)

    def compute_yaw_stable_manifold_boundary(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
        beta_range: tuple[float, float] | None = None,
        yaw_rate_range: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        state = self._coerce_state(state)
        control = self._coerce_control(control)
        beta_limit = self.config.beta_limit * self.config.manifold_bounds_scale
        yaw_limit = self.config.yaw_rate_limit * self.config.manifold_bounds_scale
        beta_range = beta_range or (-beta_limit, beta_limit)
        yaw_rate_range = yaw_rate_range or (-yaw_limit, yaw_limit)
        bounds = (beta_range, yaw_rate_range)
        vx = max(abs(state.vx), self.config.eps)

        def vector_field(beta: float, yaw_rate: float) -> tuple[float, float]:
            return self._yaw_reduced_vector_field(beta, yaw_rate, vx, control)

        equilibria = self._find_equilibria_2d(vector_field, beta_range, yaw_rate_range)
        classified = self._classified_equilibria(vector_field, equilibria)
        stable = [item for item in classified if item["kind"] == "stable"]
        saddles = [item for item in classified if item["kind"] == "saddle"]
        branches = self._branches_from_saddles(vector_field, saddles, bounds)
        status = self._boundary_status(stable, saddles, branches)
        return {
            "status": status,
            "stable_equilibria": [item["point"] for item in stable],
            "saddles": [item["point"] for item in saddles],
            "classified_equilibria": classified,
            "manifold_branches": branches,
            "bounds": bounds,
            "vector_field": vector_field,
        }

    def compute_roll_stable_manifold_boundary(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
        roll_range: tuple[float, float] | None = None,
        roll_rate_range: tuple[float, float] | None = None,
        fixed_ay: float | None = None,
    ) -> dict[str, Any]:
        state = self._coerce_state(state)
        _, aux = self.state_derivative(state, control)
        ay = float(fixed_ay) if fixed_ay is not None else aux["lateral_accel"]
        phi_crit = math.atan(self.params.track_width / max(2.0 * self.params.cg_height, self.config.eps))
        roll_range = roll_range or (-1.5 * phi_crit, 1.5 * phi_crit)
        roll_rate_range = roll_rate_range or (-6.0, 6.0)
        bounds = (roll_range, roll_rate_range)

        def vector_field(roll: float, roll_rate: float) -> tuple[float, float]:
            return self._roll_reduced_vector_field(roll, roll_rate, ay)

        equilibria = self._find_equilibria_2d(vector_field, roll_range, roll_rate_range)
        classified = self._classified_equilibria(vector_field, equilibria)
        stable = [item for item in classified if item["kind"] == "stable"]
        saddles = [item for item in classified if item["kind"] == "saddle"]
        branches = self._branches_from_saddles(vector_field, saddles, bounds)
        status = self._boundary_status(stable, saddles, branches)
        return {
            "status": status,
            "stable_equilibria": [item["point"] for item in stable],
            "saddles": [item["point"] for item in saddles],
            "classified_equilibria": classified,
            "manifold_branches": branches,
            "bounds": bounds,
            "vector_field": vector_field,
            "phi_crit": phi_crit,
            "lateral_accel": ay,
        }

    def save_yaw_phase_plane_plot(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
        output_path: str | Path = "models/platoon_planner/dynamics/outputs/phase_planes/yaw_phase_plane.png",
        title: str | None = None,
        beta_range: tuple[float, float] | None = None,
        yaw_rate_range: tuple[float, float] | None = None,
    ) -> Path:
        state = self._coerce_state(state)
        control = self._coerce_control(control)
        boundary = self.compute_yaw_stable_manifold_boundary(
            state, control, beta_range=beta_range, yaw_rate_range=yaw_rate_range
        )
        stability = self._yaw_stability_manifold(state, control)
        beta = math.atan2(state.vy, self._signed_safe_vx(state.vx))
        stable_eq = self._first_or_none(boundary["stable_equilibria"])
        forward_trajectory = (
            self._trace_forward_trajectory(
                boundary["vector_field"],
                (beta, state.yaw_rate),
                boundary["bounds"],
                tuple(stable_eq),
            )
            if stable_eq is not None
            else None
        )
        return self._save_phase_plane_plot(
            boundary,
            current=(beta, state.yaw_rate),
            output_path=output_path,
            xlabel=r"$\beta$ [rad]",
            ylabel=r"$r$ [rad/s]",
            title=title or "Yaw beta-r phase plane",
            stability_status=stability.get("stability_status"),
            forward_trajectory=forward_trajectory,
        )

    def save_roll_phase_plane_plot(
        self,
        state: VehicleState | Mapping[str, Any],
        control: VehicleControl | Mapping[str, Any] | None = None,
        output_path: str | Path = "models/platoon_planner/dynamics/outputs/phase_planes/roll_phase_plane.png",
        title: str | None = None,
        roll_range: tuple[float, float] | None = None,
        roll_rate_range: tuple[float, float] | None = None,
        fixed_ay: float | None = None,
    ) -> Path:
        state = self._coerce_state(state)
        control = self._coerce_control(control)
        boundary = self.compute_roll_stable_manifold_boundary(
            state,
            control,
            roll_range=roll_range,
            roll_rate_range=roll_rate_range,
            fixed_ay=fixed_ay,
        )
        stability = self._roll_stability_manifold(state, control)
        stable_eq = self._first_or_none(boundary["stable_equilibria"])
        forward_trajectory = (
            self._trace_forward_trajectory(
                boundary["vector_field"],
                (state.roll, state.roll_rate),
                boundary["bounds"],
                tuple(stable_eq),
            )
            if stable_eq is not None
            else None
        )
        return self._save_phase_plane_plot(
            boundary,
            current=(state.roll, state.roll_rate),
            output_path=output_path,
            xlabel=r"$\phi$ [rad]",
            ylabel=r"$\dot{\phi}$ [rad/s]",
            title=title or "Roll phi-phi_dot phase plane",
            vlines=(-boundary["phi_crit"], boundary["phi_crit"]),
            stability_status=stability.get("stability_status"),
            forward_trajectory=forward_trajectory,
        )

    def _yaw_reduced_vector_field(
        self,
        beta: float,
        yaw_rate: float,
        vx: float,
        control: VehicleControl,
    ) -> tuple[float, float]:
        vx_signed = self._signed_safe_vx(vx)
        vy = vx_signed * math.tan(beta)
        wheel_omega = vx_signed / max(self.params.wheel_radius, self.config.eps)
        state = VehicleState(
            vx=vx_signed,
            vy=vy,
            yaw_rate=yaw_rate,
            roll=0.0,
            roll_rate=0.0,
            wheel_omegas=(wheel_omega, wheel_omega, wheel_omega, wheel_omega),
        )
        state_dot, _ = self.state_derivative(state, control)
        speed_sq = max(state.vx * state.vx + state.vy * state.vy, self.config.eps)
        beta_dot = (state.vx * state_dot.vy - state.vy * state_dot.vx) / speed_sq
        return beta_dot, state_dot.yaw_rate

    def _roll_reduced_vector_field(
        self,
        roll: float,
        roll_rate: float,
        ay: float,
    ) -> tuple[float, float]:
        p = self.params
        roll_accel = (
            p.sprung_mass * p.sprung_cg_height * ay * math.cos(roll)
            + p.sprung_mass * p.gravity * p.sprung_cg_height * math.sin(roll)
            - p.roll_damping * roll_rate
            - p.roll_stiffness * roll
        ) / max(p.roll_inertia, self.config.eps)
        return roll_rate, roll_accel

    def _find_equilibria_2d(
        self,
        vector_field: Any,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> list[tuple[float, float]]:
        n = max(5, int(self.config.equilibrium_grid_size))
        xs = self._linspace(x_range[0], x_range[1], n)
        ys = self._linspace(y_range[0], y_range[1], n)
        scored: list[tuple[float, tuple[float, float]]] = []
        for x in xs:
            for y in ys:
                fx, fy = vector_field(x, y)
                scored.append((math.hypot(fx, fy), (x, y)))
        scored.sort(key=lambda item: item[0])

        roots: list[tuple[float, float]] = []
        for _, seed in scored[: min(24, len(scored))]:
            root = self._newton_refine_2d(vector_field, seed, x_range, y_range)
            if root is None:
                continue
            residual = math.hypot(*vector_field(root[0], root[1]))
            if residual > max(self.config.equilibrium_tol, 1e-3):
                continue
            if not self._point_in_bounds(root, (x_range, y_range)):
                continue
            if all(self._distance(root, existing) > 5.0 * self.config.equilibrium_tol for existing in roots):
                roots.append(root)
        return roots

    def _newton_refine_2d(
        self,
        vector_field: Any,
        seed: tuple[float, float],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> tuple[float, float] | None:
        point = seed
        for _ in range(20):
            f0 = vector_field(point[0], point[1])
            if math.hypot(*f0) <= self.config.equilibrium_tol:
                return point
            jac = self._linearize_2d(vector_field, point)
            delta = self._solve_2x2(jac, (-f0[0], -f0[1]))
            if delta is None:
                return None
            improved = False
            base_norm = math.hypot(*f0)
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = (point[0] + scale * delta[0], point[1] + scale * delta[1])
                if not self._point_in_bounds(candidate, (x_range, y_range)):
                    continue
                candidate_norm = math.hypot(*vector_field(candidate[0], candidate[1]))
                if candidate_norm <= base_norm:
                    point = candidate
                    improved = True
                    break
            if not improved:
                return None
        return point

    def _classified_equilibria(self, vector_field: Any, roots: list[tuple[float, float]]) -> list[dict[str, Any]]:
        classified: list[dict[str, Any]] = []
        for root in roots:
            jac = self._linearize_2d(vector_field, root)
            eigenvalues = self._eigenvalues_2x2(jac)
            kind = self._classify_eigenvalues(eigenvalues)
            stable_vector = None
            if kind == "saddle":
                stable_vector = self._stable_eigenvector(jac, eigenvalues)
            classified.append(
                {
                    "point": root,
                    "jacobian": jac,
                    "eigenvalues": eigenvalues,
                    "kind": kind,
                    "stable_eigenvector": stable_vector,
                }
            )
        return classified

    def _linearize_2d(self, vector_field: Any, point: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
        h = self.config.jacobian_eps
        x, y = point
        fxp = vector_field(x + h, y)
        fxm = vector_field(x - h, y)
        fyp = vector_field(x, y + h)
        fym = vector_field(x, y - h)
        return (
            ((fxp[0] - fxm[0]) / (2.0 * h), (fyp[0] - fym[0]) / (2.0 * h)),
            ((fxp[1] - fxm[1]) / (2.0 * h), (fyp[1] - fym[1]) / (2.0 * h)),
        )

    def _eigenvalues_2x2(
        self,
        matrix: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[complex, complex]:
        a, b = matrix[0]
        c, d = matrix[1]
        trace = a + d
        det = a * d - b * c
        disc = cmath.sqrt(trace * trace - 4.0 * det)
        return (0.5 * (trace + disc), 0.5 * (trace - disc))

    def _classify_eigenvalues(self, eigenvalues: tuple[complex, complex]) -> str:
        real_parts = [value.real for value in eigenvalues]
        tol = 1e-7
        if all(value < -tol for value in real_parts):
            return "stable"
        if real_parts[0] * real_parts[1] < -tol:
            return "saddle"
        return "unstable"

    def _stable_eigenvector(
        self,
        matrix: tuple[tuple[float, float], tuple[float, float]],
        eigenvalues: tuple[complex, complex],
    ) -> tuple[float, float] | None:
        stable = min(eigenvalues, key=lambda value: value.real)
        if stable.real >= 0.0 or abs(stable.imag) > 1e-7:
            return None
        lam = stable.real
        a, b = matrix[0]
        c, d = matrix[1]
        if abs(b) + abs(lam - a) >= abs(c) + abs(lam - d):
            vec = (b, lam - a)
        else:
            vec = (lam - d, c)
        norm = math.hypot(vec[0], vec[1])
        if norm <= self.config.eps:
            return None
        return vec[0] / norm, vec[1] / norm

    def _branches_from_saddles(
        self,
        vector_field: Any,
        saddles: list[dict[str, Any]],
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> list[list[tuple[float, float]]]:
        branches: list[list[tuple[float, float]]] = []
        for saddle in saddles:
            vector = saddle.get("stable_eigenvector")
            if vector is None:
                continue
            branches.extend(self._trace_stable_manifold(vector_field, saddle["point"], vector, bounds))
        return branches

    def _trace_stable_manifold(
        self,
        vector_field: Any,
        saddle: tuple[float, float],
        stable_eigenvector: tuple[float, float],
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> list[list[tuple[float, float]]]:
        branches: list[list[tuple[float, float]]] = []
        for sign in (1.0, -1.0):
            point = (
                saddle[0] + sign * self.config.manifold_eps * stable_eigenvector[0],
                saddle[1] + sign * self.config.manifold_eps * stable_eigenvector[1],
            )
            branch = [saddle, point]
            for _ in range(max(1, int(self.config.manifold_steps))):
                point = self._rk4_step(vector_field, point, -self.config.manifold_dt)
                if not self._point_in_bounds(point, bounds):
                    break
                branch.append(point)
            if len(branch) > 2:
                branches.append(branch)
        return branches

    def _integrate_forward_to_equilibrium(
        self,
        vector_field: Any,
        initial: tuple[float, float],
        equilibrium: tuple[float, float],
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[tuple[float, float], float, bool]:
        point = initial
        escaped = False
        for _ in range(max(1, int(self.config.manifold_steps))):
            point = self._rk4_step(vector_field, point, self.config.manifold_dt)
            if not self._point_in_bounds(point, bounds):
                escaped = True
                break
            if self._distance(point, equilibrium) <= self.config.equilibrium_tol * 10.0:
                break
        return point, self._distance(point, equilibrium), escaped

    def _finite_time_phase_safety_check(
        self,
        vector_field: Any,
        initial: tuple[float, float],
        equilibrium: tuple[float, float],
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> dict[str, Any]:
        terminal, terminal_error, escaped = self._integrate_forward_to_equilibrium(
            vector_field, initial, equilibrium, bounds
        )
        convergence_tol = max(self.config.equilibrium_tol * 50.0, 1e-3)
        return {
            "terminal_state": terminal,
            "terminal_error": terminal_error,
            "escaped": escaped,
            "converged": (not escaped) and terminal_error <= convergence_tol,
            "convergence_tol": convergence_tol,
        }

    def _trace_forward_trajectory(
        self,
        vector_field: Any,
        initial: tuple[float, float],
        bounds: tuple[tuple[float, float], tuple[float, float]],
        equilibrium: tuple[float, float] | None = None,
    ) -> list[tuple[float, float]]:
        point = initial
        trajectory = [point]
        stop_tol = max(self.config.equilibrium_tol * 50.0, 1e-3)
        for _ in range(max(1, int(self.config.manifold_steps))):
            point = self._rk4_step(vector_field, point, self.config.manifold_dt)
            if not self._point_in_bounds(point, bounds):
                break
            trajectory.append(point)
            if equilibrium is not None and self._distance(point, equilibrium) <= stop_tol:
                break
        return trajectory

    def _apply_yaw_manifold_decision(
        self,
        state: VehicleState,
        control: VehicleControl,
        result: dict[str, Any],
        boundary: Mapping[str, Any],
        safety_check: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        stable_eq = result["stable_equilibrium"]
        point = (float(result["beta"]), float(result["yaw_rate"]))
        _, aux = self.state_derivative(state, control)
        max_tire_utilization = self._tire_utilization(aux)
        distance_norm = (
            self._normalized_phase_distance(
                point,
                tuple(stable_eq),
                (self.config.beta_limit, self.config.yaw_rate_limit),
            )
            if stable_eq is not None
            else math.inf
        )
        margins = {
            "engineering_margin": float(result["margin"]),
            "phase_norm": float(result["phase_norm"]),
            "distance_to_stable_equilibrium": distance_norm,
            "tire_utilization_margin": 1.0 - max_tire_utilization,
            "max_tire_utilization": max_tire_utilization,
            "terminal_error": math.inf,
            "escaped": True,
            "converged": False,
            "inside_manifold_boundary": None,
        }
        if safety_check is not None:
            margins.update(
                {
                    "terminal_error": float(safety_check["terminal_error"]),
                    "escaped": bool(safety_check["escaped"]),
                    "converged": bool(safety_check["converged"]),
                }
            )

        domain_status = result["domain_status"]
        if domain_status == "no_stable_equilibrium":
            result.update(
                {
                    "stable": False,
                    "risk": "yaw_unstable",
                    "stability_status": "high_risk",
                    "decision_source": "no_stable_equilibrium_safety",
                    "safety_margins": margins,
                }
            )
            return result

        inside = None
        if domain_status == "manifold_boundary":
            inside = self._point_inside_boundary(point, boundary["manifold_branches"])
            margins["inside_manifold_boundary"] = inside

        hard_violation = result["risk"] == "yaw_unstable" or max_tire_utilization > 1.05
        warning = (
            float(result["margin"]) < 0.15
            or distance_norm > 0.8
            or max_tire_utilization > 0.9
            or (safety_check is not None and not bool(safety_check["converged"]))
        )

        if domain_status == "manifold_boundary" and inside is not None:
            if (not inside) or hard_violation:
                stability_status = "unstable"
            elif warning:
                stability_status = "warning"
            else:
                stability_status = "stable"
            result["decision_source"] = "manifold_boundary"
        else:
            if hard_violation or (safety_check is not None and bool(safety_check["escaped"])):
                stability_status = "unstable"
            elif warning:
                stability_status = "warning"
            else:
                stability_status = "stable"
            result["decision_source"] = "local_equilibrium_margin"

        result["stability_status"] = stability_status
        result["stable"] = stability_status == "stable"
        result["risk"] = "stable" if result["stable"] else "yaw_unstable"
        if not result["stable"]:
            result["margin"] = min(float(result["margin"]), -0.01 if stability_status == "unstable" else 0.0)
        result["safety_margins"] = margins
        return result

    def _apply_roll_manifold_decision(
        self,
        state: VehicleState,
        control: VehicleControl,
        result: dict[str, Any],
        boundary: Mapping[str, Any],
        safety_check: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        stable_eq = result["stable_equilibrium"]
        point = (float(state.roll), float(state.roll_rate))
        _, aux = self.state_derivative(state, control)
        max_tire_utilization = self._tire_utilization(aux)
        energy_limit = self.config.energy_margin * max(float(result["critical_energy"]), self.config.eps)
        dynamic_margin = 1.0 - float(result["roll_energy"]) / max(energy_limit, self.config.eps)
        roll_angle_margin = float(result["phi_crit"]) - abs(float(state.roll))
        distance_norm = (
            self._normalized_phase_distance(point, tuple(stable_eq), (float(result["phi_crit"]), 4.0))
            if stable_eq is not None
            else math.inf
        )
        margins = {
            "ltr_margin": self.config.ltr_limit - abs(float(result["ltr"])),
            "roll_angle_margin": roll_angle_margin,
            "roll_energy_margin": dynamic_margin,
            "distance_to_stable_equilibrium": distance_norm,
            "tire_utilization_margin": 1.0 - max_tire_utilization,
            "max_tire_utilization": max_tire_utilization,
            "terminal_error": math.inf,
            "escaped": True,
            "converged": False,
            "inside_manifold_boundary": None,
        }
        if safety_check is not None:
            margins.update(
                {
                    "terminal_error": float(safety_check["terminal_error"]),
                    "escaped": bool(safety_check["escaped"]),
                    "converged": bool(safety_check["converged"]),
                }
            )

        domain_status = result["domain_status"]
        if domain_status == "no_stable_equilibrium":
            result.update(
                {
                    "stable": False,
                    "stability_status": "high_risk",
                    "decision_source": "no_stable_equilibrium_safety",
                    "safety_margins": margins,
                }
            )
            if result["risk"] == "stable":
                result["risk"] = "dynamic_roll_risk"
            return result

        inside = None
        if domain_status == "manifold_boundary":
            inside = self._point_inside_boundary(point, boundary["manifold_branches"])
            margins["inside_manifold_boundary"] = inside

        static_violation = margins["ltr_margin"] < 0.0
        dynamic_violation = dynamic_margin < 0.0 or roll_angle_margin <= 0.0
        hard_violation = static_violation or dynamic_violation or max_tire_utilization > 1.05
        warning = (
            margins["ltr_margin"] < 0.10
            or roll_angle_margin < 0.10 * max(float(result["phi_crit"]), self.config.eps)
            or dynamic_margin < 0.15
            or distance_norm > 0.8
            or max_tire_utilization > 0.9
            or (safety_check is not None and not bool(safety_check["converged"]))
        )

        if domain_status == "manifold_boundary" and inside is not None:
            if (not inside) or hard_violation:
                stability_status = "unstable"
            elif warning:
                stability_status = "warning"
            else:
                stability_status = "stable"
            result["decision_source"] = "manifold_boundary"
        else:
            if hard_violation or (safety_check is not None and bool(safety_check["escaped"])):
                stability_status = "unstable"
            elif warning:
                stability_status = "warning"
            else:
                stability_status = "stable"
            result["decision_source"] = "local_equilibrium_margin"

        result["stability_status"] = stability_status
        result["stable"] = stability_status == "stable"
        if static_violation:
            result["risk"] = "static_roll_risk"
        elif not result["stable"]:
            result["risk"] = "dynamic_roll_risk"
        else:
            result["risk"] = "stable"
        if not result["stable"]:
            result["margin"] = min(float(result["margin"]), -0.01 if stability_status == "unstable" else 0.0)
        result["safety_margins"] = margins
        return result

    def _tire_utilization(self, aux: Mapping[str, Any]) -> float:
        max_utilization = 0.0
        for name in WHEEL_ORDER:
            load = max(float(aux["vertical_loads"][name]), 0.0)
            limit = self.params.friction_coeff * load
            force = aux["tire_forces"][name]
            norm = math.hypot(float(force["fx"]), float(force["fy"]))
            if limit <= self.config.eps:
                utilization = math.inf if norm > self.config.eps else 0.0
            else:
                utilization = norm / limit
            max_utilization = max(max_utilization, utilization)
        return max_utilization

    def _normalized_phase_distance(
        self,
        point: tuple[float, float],
        equilibrium: tuple[float, float],
        limits: tuple[float, float],
    ) -> float:
        x_limit = max(abs(limits[0]), self.config.eps)
        y_limit = max(abs(limits[1]), self.config.eps)
        dx = (point[0] - equilibrium[0]) / x_limit
        dy = (point[1] - equilibrium[1]) / y_limit
        return math.hypot(dx, dy)

    def _rk4_step(self, vector_field: Any, point: tuple[float, float], dt: float) -> tuple[float, float]:
        def add(base: tuple[float, float], deriv: tuple[float, float], scale: float) -> tuple[float, float]:
            return base[0] + scale * deriv[0], base[1] + scale * deriv[1]

        k1 = vector_field(point[0], point[1])
        k2p = add(point, k1, 0.5 * dt)
        k2 = vector_field(k2p[0], k2p[1])
        k3p = add(point, k2, 0.5 * dt)
        k3 = vector_field(k3p[0], k3p[1])
        k4p = add(point, k3, dt)
        k4 = vector_field(k4p[0], k4p[1])
        return (
            point[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            point[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        )

    def _boundary_status(
        self,
        stable: list[dict[str, Any]],
        saddles: list[dict[str, Any]],
        branches: list[list[tuple[float, float]]],
    ) -> str:
        if not stable:
            return "no_stable_equilibrium"
        if not saddles:
            return "no_saddle_boundary"
        if not branches:
            return "no_saddle_boundary"
        return "manifold_boundary"

    def _point_inside_boundary(
        self,
        point: tuple[float, float],
        branches: list[list[tuple[float, float]]],
    ) -> bool | None:
        polygon: list[tuple[float, float]] = []
        for branch in branches:
            polygon.extend(branch)
        if len(polygon) < 3:
            return None
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i, pi in enumerate(polygon):
            pj = polygon[j]
            denom = pj[1] - pi[1]
            if abs(denom) <= self.config.eps:
                denom = self.config.eps if denom >= 0.0 else -self.config.eps
            if ((pi[1] > y) != (pj[1] > y)) and (
                x < (pj[0] - pi[0]) * (y - pi[1]) / denom + pi[0]
            ):
                inside = not inside
            j = i
        return inside

    def _save_phase_plane_plot(
        self,
        boundary: Mapping[str, Any],
        current: tuple[float, float],
        output_path: str | Path,
        xlabel: str,
        ylabel: str,
        title: str,
        vlines: tuple[float, float] | None = None,
        stability_status: str | None = None,
        forward_trajectory: list[tuple[float, float]] | None = None,
    ) -> Path:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rc = {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(4.8, 3.6))
            (x_range, y_range) = boundary["bounds"]
            xs = self._linspace(x_range[0], x_range[1], 35)
            ys = self._linspace(y_range[0], y_range[1], 35)
            stream_u: list[list[float]] = []
            stream_v: list[list[float]] = []
            vector_field = boundary["vector_field"]
            for y in ys:
                u_row: list[float] = []
                v_row: list[float] = []
                for x in xs:
                    u, v = vector_field(x, y)
                    norm = math.hypot(u, v)
                    if norm > self.config.eps:
                        u, v = u / norm, v / norm
                    u_row.append(u)
                    v_row.append(v)
                stream_u.append(u_row)
                stream_v.append(v_row)
            ax.streamplot(
                xs,
                ys,
                np.asarray(stream_u, dtype=float),
                np.asarray(stream_v, dtype=float),
                color="0.72",
                density=1.0,
                linewidth=0.55,
                arrowsize=0.7,
            )
            for branch in boundary["manifold_branches"]:
                if len(branch) < 2:
                    continue
                ax.plot(
                    [p[0] for p in branch],
                    [p[1] for p in branch],
                    color="#f28e2b",
                    linewidth=1.2,
                    label="stable manifold",
                )
            if forward_trajectory is not None and len(forward_trajectory) >= 2:
                ax.plot(
                    [p[0] for p in forward_trajectory],
                    [p[1] for p in forward_trajectory],
                    color="#c70a0a",
                    linewidth=2.0,
                    linestyle="-",
                    solid_capstyle="round",
                    label="current -> stable eq.",
                    zorder=3,
                )
            for point in boundary["stable_equilibria"]:
                ax.scatter(
                    point[0],
                    point[1],
                    color="#26864D",
                    marker="o",
                    s=32,
                    label="stable eq.",
                    zorder=4,
                )
            for point in boundary["saddles"]:
                ax.scatter(
                    point[0],
                    point[1],
                    color="#e2112c",
                    marker="x",
                    s=42,
                    label="saddle",
                    zorder=5,
                )
            ax.scatter(current[0], current[1], color="black", marker="*", s=58, label="current", zorder=6)
            if vlines is not None:
                y_top = y_range[1] - 0.06 * (y_range[1] - y_range[0])
                for value in vlines:
                    ax.axvline(value, color="#8c6d6d", linestyle="--", linewidth=0.9, alpha=0.9)
                    sign = "+" if value > 0 else "-"
                    ax.text(
                        value,
                        y_top,
                        rf"${sign}\phi_{{\mathrm{{crit}}}}$",
                        color="#5f5050",
                        fontsize=7,
                        ha="center",
                        va="top",
                        rotation=90,
                        backgroundcolor="white",
                    )
            if stability_status is not None:
                ax.text(
                    0.98,
                    0.97,
                    f"stability: {stability_status}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color="0.35",
                )
            ax.set_xlim(*x_range)
            ax.set_ylim(*y_range)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.18, linewidth=0.45)
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            if unique:
                ax.legend(unique.values(), unique.keys(), loc="best", frameon=False)
            fig.tight_layout()
            paths = self._phase_plane_output_paths(output)
            for path in paths:
                fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        return output

    def _phase_plane_output_paths(self, output: Path) -> list[Path]:
        suffix = output.suffix.lower()
        if suffix == ".png":
            return [output, output.with_suffix(".pdf"), output.with_suffix(".svg")]
        if suffix == ".pdf":
            return [output, output.with_suffix(".svg")]
        if suffix == ".svg":
            return [output, output.with_suffix(".pdf")]
        return [output, output.with_suffix(".pdf"), output.with_suffix(".svg")]

    def _linspace(self, start: float, stop: float, count: int) -> list[float]:
        if count <= 1:
            return [start]
        step = (stop - start) / float(count - 1)
        return [start + i * step for i in range(count)]

    def _solve_2x2(
        self,
        matrix: tuple[tuple[float, float], tuple[float, float]],
        rhs: tuple[float, float],
    ) -> tuple[float, float] | None:
        a, b = matrix[0]
        c, d = matrix[1]
        det = a * d - b * c
        if abs(det) <= self.config.eps:
            return None
        return ((rhs[0] * d - b * rhs[1]) / det, (a * rhs[1] - rhs[0] * c) / det)

    def _point_in_bounds(
        self,
        point: tuple[float, float],
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        return bounds[0][0] <= point[0] <= bounds[0][1] and bounds[1][0] <= point[1] <= bounds[1][1]

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _first_or_none(self, values: list[Any]) -> Any:
        return values[0] if values else None

    def _coerce_state(self, state: VehicleState | Mapping[str, Any]) -> VehicleState:
        if isinstance(state, VehicleState):
            return state
        return VehicleState(
            vx=float(state["vx"]),
            vy=float(state["vy"]),
            yaw_rate=float(state.get("yaw_rate", state.get("r", 0.0))),
            roll=float(state.get("roll", state.get("phi", 0.0))),
            roll_rate=float(state.get("roll_rate", state.get("phi_dot", 0.0))),
            wheel_omegas=tuple(state.get("wheel_omegas", (0.0, 0.0, 0.0, 0.0))),
        )

    def _coerce_control(
        self,
        control: VehicleControl | Mapping[str, Any] | None,
    ) -> VehicleControl:
        if control is None:
            return VehicleControl()
        if isinstance(control, VehicleControl):
            return control
        return VehicleControl(
            steer=float(control.get("steer", control.get("delta_f", 0.0))),
            wheel_torques=tuple(control.get("wheel_torques", (0.0, 0.0, 0.0, 0.0))),
        )

    def _signed_safe_vx(self, vx: float) -> float:
        eps = self.config.eps
        if abs(vx) < eps:
            return eps if vx >= 0.0 else -eps
        return vx

    def _wheel_kinematics(self, state: VehicleState, steer: float) -> dict[str, dict[str, float]]:
        p = self.params
        cos_delta = math.cos(steer)
        sin_delta = math.sin(steer)
        vx_fl = state.vx - 0.5 * p.track_width_front * state.yaw_rate
        vx_fr = state.vx + 0.5 * p.track_width_front * state.yaw_rate
        vx_rl = state.vx - 0.5 * p.track_width_rear * state.yaw_rate
        vx_rr = state.vx + 0.5 * p.track_width_rear * state.yaw_rate
        vy_f = state.vy + p.lf * state.yaw_rate
        vy_r = state.vy - p.lr * state.yaw_rate

        return {
            "fl": {
                "vx_tire": vx_fl * cos_delta + vy_f * sin_delta,
                "vy_tire": -vx_fl * sin_delta + vy_f * cos_delta,
            },
            "fr": {
                "vx_tire": vx_fr * cos_delta + vy_f * sin_delta,
                "vy_tire": -vx_fr * sin_delta + vy_f * cos_delta,
            },
            "rl": {"vx_tire": vx_rl, "vy_tire": vy_r},
            "rr": {"vx_tire": vx_rr, "vy_tire": vy_r},
        }

    def _vertical_loads(self, state: VehicleState, ax: float, ay: float) -> dict[str, float]:
        p = self.params
        wheelbase = max(p.wheelbase, self.config.eps)
        front_static = p.mass * p.gravity * p.lr / wheelbase
        rear_static = p.mass * p.gravity * p.lf / wheelbase
        longitudinal_transfer = p.mass * p.cg_height * ax / wheelbase
        front_axle = front_static - longitudinal_transfer
        rear_axle = rear_static + longitudinal_transfer

        roll_transfer_moment = (
            p.sprung_mass * p.sprung_cg_height * ay
            + p.roll_stiffness * state.roll
        )
        front_share = 0.5
        rear_share = 0.5
        front_transfer = front_share * roll_transfer_moment / max(p.track_width_front, self.config.eps)
        rear_transfer = rear_share * roll_transfer_moment / max(p.track_width_rear, self.config.eps)

        return {
            "fl": front_axle * 0.5 - front_transfer,
            "fr": front_axle * 0.5 + front_transfer,
            "rl": rear_axle * 0.5 - rear_transfer,
            "rr": rear_axle * 0.5 + rear_transfer,
        }

    def _tire_forces(
        self,
        kinematics: Mapping[str, Mapping[str, float]],
        vertical_loads: Mapping[str, float],
        state: VehicleState,
        control: VehicleControl,
    ) -> dict[str, dict[str, float]]:
        p = self.params
        forces: dict[str, dict[str, float]] = {}
        for idx, name in enumerate(WHEEL_ORDER):
            vx_tire = kinematics[name]["vx_tire"]
            vy_tire = kinematics[name]["vy_tire"]
            omega = state.wheel_omegas[idx]
            alpha = -math.atan2(vy_tire, self._signed_safe_vx(vx_tire))
            normal_load = max(vertical_loads[name], 0.0)
            cornering = (
                p.cornering_stiffness_front
                if name in {"fl", "fr"}
                else p.cornering_stiffness_rear
            )
            fx = control.wheel_torques[idx] / max(p.wheel_radius, self.config.eps)
            fy = self._lateral_tire_force(alpha, cornering, normal_load)
            fx, fy = self._saturate_friction(fx, fy, p.friction_coeff * normal_load)
            wheel_speed = p.wheel_radius * omega
            slip_den = max(abs(vx_tire), abs(wheel_speed), self.config.eps)
            slip_ratio = (wheel_speed - vx_tire) / slip_den
            forces[name] = {
                "fx": fx,
                "fy": fy,
                "alpha": alpha,
                "slip_ratio": slip_ratio,
            }
        return forces

    def _lateral_tire_force(self, alpha: float, cornering: float, normal_load: float) -> float:
        if self.config.tire_model == "linear":
            return cornering * alpha
        load_limit = self.params.friction_coeff * max(normal_load, 0.0)
        if load_limit <= self.config.eps:
            return 0.0
        tan_alpha = math.tan(alpha)
        abs_tan = abs(tan_alpha)
        critical_tan = 3.0 * load_limit / max(cornering, self.config.eps)
        if abs_tan >= critical_tan:
            return math.copysign(load_limit, tan_alpha)
        return (
            cornering * tan_alpha
            - (cornering * cornering / (3.0 * load_limit)) * abs_tan * tan_alpha
            + (cornering**3 / (27.0 * load_limit * load_limit)) * tan_alpha**3
        )

    def _saturate_friction(self, fx: float, fy: float, limit: float) -> tuple[float, float]:
        limit = max(limit, 0.0)
        norm = math.hypot(fx, fy)
        if norm <= limit or norm <= self.config.eps:
            return fx, fy
        scale = limit / norm
        return fx * scale, fy * scale

    def _yaw_moment(self, tire_forces: Mapping[str, Mapping[str, float]], steer: float) -> float:
        p = self.params
        cos_delta = math.cos(steer)
        sin_delta = math.sin(steer)
        front_fx = tire_forces["fl"]["fx"] + tire_forces["fr"]["fx"]
        front_fy = tire_forces["fl"]["fy"] + tire_forces["fr"]["fy"]
        rear_fy = tire_forces["rl"]["fy"] + tire_forces["rr"]["fy"]

        front_lateral_moment = p.lf * (front_fx * sin_delta + front_fy * cos_delta)
        rear_lateral_moment = -p.lr * rear_fy
        front_track_moment = 0.5 * p.track_width_front * (
            tire_forces["fr"]["fx"] * cos_delta
            - tire_forces["fr"]["fy"] * sin_delta
            - tire_forces["fl"]["fx"] * cos_delta
            + tire_forces["fl"]["fy"] * sin_delta
        )
        rear_track_moment = 0.5 * p.track_width_rear * (
            tire_forces["rr"]["fx"] - tire_forces["rl"]["fx"]
        )
        return front_lateral_moment + rear_lateral_moment + front_track_moment + rear_track_moment

    def _ltr(self, vertical_loads: Mapping[str, float]) -> float:
        right = vertical_loads["fr"] + vertical_loads["rr"]
        left = vertical_loads["fl"] + vertical_loads["rl"]
        return (right - left) / max(right + left, self.config.eps)

    def _roll_energy(self, roll: float, roll_rate: float, ay: float) -> tuple[float, float, float]:
        p = self.params
        k_eff = max(p.effective_roll_stiffness, self.config.eps)
        phi_crit = math.atan(p.track_width / max(2.0 * p.cg_height, self.config.eps))

        def energy(phi: float, phi_dot: float) -> float:
            return (
                0.5 * p.roll_inertia * phi_dot * phi_dot
                + 0.5 * k_eff * phi * phi
                - p.sprung_mass * p.sprung_cg_height * ay * phi
            )

        direction = 1.0 if roll >= 0.0 else -1.0
        critical_energy = energy(direction * phi_crit, 0.0)
        if critical_energy <= self.config.eps:
            critical_energy = 0.5 * k_eff * phi_crit * phi_crit
        return energy(roll, roll_rate), critical_energy, phi_crit
