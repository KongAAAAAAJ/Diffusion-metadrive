from __future__ import annotations

import pytest

from evaluation.platoon_metrics import PlatoonMetrics


def test_platoon_metrics_exposes_required_methods():
    metrics = PlatoonMetrics()
    assert callable(metrics.start_episode)
    assert callable(metrics.update)
    assert callable(metrics.end_episode)
    assert callable(metrics.compute)


def test_compute_returns_five_float_keys():
    metrics = PlatoonMetrics()
    summary = metrics.compute()
    expected_keys = {
        "success_rate",
        "collision_rate",
        "formation_error",
        "recovery_time",
        "min_inter_vehicle_gap",
    }
    assert set(summary.keys()) == expected_keys
    assert all(isinstance(value, float) for value in summary.values())


def test_precision_for_two_known_episodes():
    metrics = PlatoonMetrics(formation_error_threshold=2.0)
    for success, collision, formation_error, min_gap in [
        (True, False, 1.0, 3.0),
        (False, True, 3.0, 2.0),
    ]:
        metrics.start_episode()
        metrics.update(
            {
                "agent0": {"arrive_dest": success, "crash": collision, "formation_error": formation_error, "min_gap": min_gap},
                "agent1": {"arrive_dest": success, "crash": collision, "formation_error": formation_error, "min_gap": min_gap},
                "agent2": {"arrive_dest": success, "crash": collision, "formation_error": formation_error, "min_gap": min_gap},
            }
        )
        metrics.end_episode()

    summary = metrics.compute()
    assert summary["success_rate"] == pytest.approx(0.5, abs=1e-3)
    assert summary["collision_rate"] == pytest.approx(0.5, abs=1e-3)
    assert summary["formation_error"] == pytest.approx(2.0, abs=1e-3)
    assert summary["min_inter_vehicle_gap"] == pytest.approx(2.0, abs=1e-3)


def test_empty_compute_returns_all_zero():
    summary = PlatoonMetrics().compute()
    assert summary == {
        "success_rate": 0.0,
        "collision_rate": 0.0,
        "formation_error": 0.0,
        "recovery_time": 0.0,
        "min_inter_vehicle_gap": 0.0,
    }


def test_recovery_time_is_measured_until_return_below_threshold():
    metrics = PlatoonMetrics(formation_error_threshold=2.0)
    metrics.start_episode()
    for _ in range(5):
        metrics.update(
            {
                "agent0": {"formation_error": 3.0, "min_gap": 5.0},
                "agent1": {"formation_error": 3.0, "min_gap": 5.0},
                "agent2": {"formation_error": 3.0, "min_gap": 5.0},
            }
        )
    metrics.update(
        {
            "agent0": {"formation_error": 1.0, "min_gap": 5.0},
            "agent1": {"formation_error": 1.0, "min_gap": 5.0},
            "agent2": {"formation_error": 1.0, "min_gap": 5.0},
        }
    )
    metrics.end_episode()
    assert metrics.compute()["recovery_time"] == pytest.approx(5.0, abs=1e-3)


def test_all_compute_values_are_float():
    metrics = PlatoonMetrics()
    metrics.start_episode()
    metrics.update(
        {
            "agent0": {"arrive_dest": True, "crash": False, "formation_error": 1.0, "min_gap": 4.0},
            "agent1": {"arrive_dest": True, "crash": False, "formation_error": 1.0, "min_gap": 4.0},
            "agent2": {"arrive_dest": True, "crash": False, "formation_error": 1.0, "min_gap": 4.0},
        }
    )
    metrics.end_episode()
    assert all(isinstance(value, float) for value in metrics.compute().values())
