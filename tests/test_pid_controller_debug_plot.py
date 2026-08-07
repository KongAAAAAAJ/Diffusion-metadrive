from __future__ import annotations

import numpy as np
import pytest


class _FakeVehicle:
    def __init__(self, heading: float = 0.0, speed_km_h: float = 0.0) -> None:
        self.position = np.asarray([0.0, 0.0], dtype=np.float32)
        self.heading_theta = float(heading)
        self.speed_km_h = float(speed_km_h)
        self.heading = np.asarray(
            [np.cos(self.heading_theta), np.sin(self.heading_theta)],
            dtype=np.float32,
        )
        self.velocity = self.heading * (self.speed_km_h / 3.6)


def test_save_pid_debug_plot_saves_png_and_accumulates_history(tmp_path) -> None:
    from models.controller.controller_helper import (
        get_pid_debug_history,
        reset_pid_debug_history,
        save_pid_debug_plot,
    )

    reset_pid_debug_history()

    first_path = save_pid_debug_plot(
        agent_id="agent0",
        steering=0.3,
        actual_heading=0.10,
        heading_error=0.05,
        output_dir=tmp_path,
    )
    second_path = save_pid_debug_plot(
        agent_id="agent0",
        steering=-0.2,
        actual_heading=0.20,
        heading_error=0.05,
        output_dir=tmp_path,
    )

    assert first_path == second_path
    assert second_path.name == "pid_debug_agent0_latest.png"
    assert second_path.exists()

    history = get_pid_debug_history("agent0")
    assert history["steering"] == [0.3, -0.2]
    assert history["actual_heading"] == [0.10, 0.20]
    assert history["reference_heading"] == pytest.approx([0.15, 0.25])
    assert history["heading_error"] == [0.05, 0.05]


def test_pid_single_control_calls_debug_plot_for_valid_trajectory(monkeypatch) -> None:
    import models.controller.PIDController as pid_module

    calls: list[dict[str, float | str]] = []

    def _fake_save_pid_debug_plot(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(pid_module, "save_pid_debug_plot", _fake_save_pid_debug_plot)

    controller = pid_module.PIDTrajectoryController({"pid_dt": 0.5, "pid_lookahead_index": 1})
    trajectory_local = np.asarray(
        [
            [1.0, 0.0, 0.00],
            [2.0, 0.4, 0.07],
            [3.0, 0.6, 0.09],
            [4.0, 0.8, 0.10],
            [5.0, 1.0, 0.10],
            [6.0, 1.2, 0.10],
            [7.0, 1.4, 0.10],
            [8.0, 1.6, 0.10],
        ],
        dtype=np.float32,
    )

    action = controller._single_control("agent0", _FakeVehicle(heading=0.3, speed_km_h=12.0), trajectory_local)

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert -1.0 <= float(action[0]) <= 1.0
    assert -1.0 <= float(action[1]) <= 1.0
    assert len(calls) == 1
    assert calls[0]["agent_id"] == "agent0"
    assert calls[0]["actual_heading"] == 0.3
    assert calls[0]["heading_error"] == pytest.approx(float(trajectory_local[1, 2]))


def test_pid_single_control_empty_trajectory_does_not_plot(monkeypatch) -> None:
    import models.controller.PIDController as pid_module

    calls: list[dict] = []
    monkeypatch.setattr(pid_module, "save_pid_debug_plot", lambda **kwargs: calls.append(kwargs))

    controller = pid_module.PIDTrajectoryController()
    action = controller._single_control("agent0", _FakeVehicle(), np.zeros((0, 3), dtype=np.float32))

    np.testing.assert_array_equal(action, np.zeros((2,), dtype=np.float32))
    assert calls == []
