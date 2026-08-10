from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chassis_execution.fixture import synthetic_command
from chassis_execution.online_context import (
    MetaDriveOnlineChassisContextBuilder,
    OnlineChassisContextError,
)


class _Lane:
    def local_coordinates(self, position):
        return float(position[0]), float(position[1])

    def heading_theta_at(self, _longitudinal):
        return 0.0


def _vehicle(x: float, speed: float, *, heading: float = 0.0):
    return SimpleNamespace(
        position=np.asarray([x, 0.0], dtype=np.float64),
        velocity=np.asarray(
            [speed * np.cos(heading), speed * np.sin(heading)], dtype=np.float64
        ),
        heading_theta=float(heading),
        steering=0.5,
        max_steering=40.0,
        throttle_brake=0.2,
        TIRE_RADIUS=0.4,
        lane=_Lane(),
    )


def _env(*, formation: bool = True):
    longitudinal = SimpleNamespace(
        _integral={"agent0": 0.1, "agent1": 0.2, "agent2": 0.3}
    )
    return SimpleNamespace(
        agents={
            "agent0": _vehicle(30.0, 6.0),
            "agent1": _vehicle(15.0, 5.0),
            "agent2": _vehicle(0.0, 4.0),
        },
        _trajectory_longitudinal_controller=longitudinal,
        _lateral_preview_pid_state={
            "agent0": (0.01, 0.02, True),
            "agent1": (0.03, 0.04, True),
            "agent2": (0.05, 0.06, True),
        },
        _last_longitudinal_control_debug={
            "agent0": {"original_arc_error_m": 0.1},
            "agent1": {"original_arc_error_m": 0.2},
            "agent2": {"original_arc_error_m": 0.3},
        },
        trajectory_formation_constraint_enabled=lambda: formation,
        _desired_center_spacing_m=lambda *_: 15.0,
    )


def _builder():
    command = synthetic_command(batch_size=1)
    return MetaDriveOnlineChassisContextBuilder(
        command.vehicle_condition[0],
        vehicle_condition_source="cf3:test:fingerprint",
    )


def test_online_context_maps_planar_state_controller_and_gaps() -> None:
    env = _env(formation=False)
    builder = _builder()
    first, diagnostics = builder.build(env, 0.0)
    initial = first.initial_state[0].numpy()
    context = first.controller_context[0].numpy()
    np.testing.assert_allclose(initial[:, 0], [6.0, 5.0, 4.0])
    np.testing.assert_allclose(initial[:, 5:7], 0.0)
    np.testing.assert_allclose(initial[:, 7], np.deg2rad(20.0))
    np.testing.assert_allclose(initial[:, 8], 0.2)
    np.testing.assert_allclose(initial[:, 9], [15.0, 12.5, 10.0])
    np.testing.assert_allclose(context[:, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(context[:, 1], [0.02, 0.04, 0.06])
    np.testing.assert_allclose(context[:, 2], [0.01, 0.03, 0.05])
    np.testing.assert_allclose(context[:, 4], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(context[:, 5], [0.0, 15.0, 15.0])
    np.testing.assert_allclose(context[:, 6], [0.0, 15.0, 15.0])
    np.testing.assert_allclose(context[:, 7], [0.0, 1.0, 1.0])
    assert first.controller_mode.tolist() == [[0, 0, 0]]
    assert diagnostics.planar_roll_assumption is True
    assert diagnostics.acceleration_from_history == (False, False, False)


def test_online_context_uses_strict_finite_differences() -> None:
    env = _env()
    builder = _builder()
    builder.build(env, 1.0)
    for vehicle in env.agents.values():
        vehicle.velocity = np.asarray([7.0, 0.0], dtype=np.float64)
        vehicle.heading_theta = 0.01
    second, diagnostics = builder.build(env, 1.1)
    values = second.initial_state[0].numpy()
    assert diagnostics.acceleration_from_history == (True, True, True)
    np.testing.assert_allclose(values[:, 4], 0.1, atol=1.0e-5)
    assert np.isfinite(values[:, 2:5]).all()
    assert second.controller_mode.tolist() == [[1, 1, 1]]
    with pytest.raises(OnlineChassisContextError, match="increase strictly"):
        builder.build(env, 1.1)


def test_online_context_rejects_missing_agents_and_bad_conditions() -> None:
    env = _env()
    env.agents.pop("agent2")
    with pytest.raises(OnlineChassisContextError, match="all three"):
        _builder().build(env, 0.0)
    command = synthetic_command(batch_size=1)
    bad = command.vehicle_condition[0].double()
    with pytest.raises(OnlineChassisContextError, match="float32"):
        MetaDriveOnlineChassisContextBuilder(
            bad, vehicle_condition_source="bad"
        )
