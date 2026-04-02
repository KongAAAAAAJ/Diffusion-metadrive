from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from metadrive.exp_dataset.hierarchical_expert.rear_end_guard import RearEndGuardRegulator


def test_guard_returns_idm_acc_when_front_object_missing():
    guard = RearEndGuardRegulator()
    ego = SimpleNamespace(speed=10.0)

    assert guard.adjust_acceleration(ego=ego, front_obj=None, front_dist=100.0, idm_acc=0.2) == 0.2
    assert guard.last_diagnostics["active"] is False
    assert guard.last_diagnostics["ttc"] is None
    assert guard.last_diagnostics["gap"] == 100.0


def test_front_speed_uses_speed_attribute():
    guard = RearEndGuardRegulator()
    front_obj = SimpleNamespace(speed=8.5)

    assert guard._front_speed(front_obj) == 8.5


def test_front_speed_uses_speed_km_h_attribute():
    guard = RearEndGuardRegulator()
    front_obj = SimpleNamespace(speed_km_h=36.0)

    assert guard._front_speed(front_obj) == 10.0


def test_front_speed_uses_velocity_km_h_attribute():
    guard = RearEndGuardRegulator()
    front_obj = SimpleNamespace(velocity_km_h=np.asarray([0.0, 54.0], dtype=np.float64))

    assert guard._front_speed(front_obj) == 15.0


def test_no_closing_speed_preserves_idm_acc():
    guard = RearEndGuardRegulator()
    ego = SimpleNamespace(speed=8.0)
    front_obj = SimpleNamespace(speed=10.0)

    adjusted = guard.adjust_acceleration(ego=ego, front_obj=front_obj, front_dist=4.0, idm_acc=0.4)

    assert adjusted == 0.4
    assert guard.last_diagnostics["active"] is False


def test_soft_guard_returns_more_conservative_brake_than_idm():
    guard = RearEndGuardRegulator()
    ego = SimpleNamespace(speed=10.0)
    front_obj = SimpleNamespace(speed=8.0)

    adjusted = guard.adjust_acceleration(ego=ego, front_obj=front_obj, front_dist=15.0, idm_acc=0.5)

    assert adjusted == guard.SOFT_BRAKE
    assert guard.last_diagnostics["active"] is True
    assert guard.last_diagnostics["gap"] == 15.0
    assert guard.last_diagnostics["ttc"] == 7.5


def test_hard_guard_returns_more_conservative_brake_than_idm():
    guard = RearEndGuardRegulator()
    ego = SimpleNamespace(speed=12.0)
    front_obj = SimpleNamespace(speed=3.0)

    adjusted = guard.adjust_acceleration(ego=ego, front_obj=front_obj, front_dist=8.0, idm_acc=0.1)

    assert adjusted == guard.HARD_BRAKE
    assert guard.last_diagnostics["active"] is True
    assert adjusted < 0.1


def test_stronger_existing_idm_brake_is_preserved():
    guard = RearEndGuardRegulator()
    ego = SimpleNamespace(speed=12.0)
    front_obj = SimpleNamespace(speed=0.0)

    adjusted = guard.adjust_acceleration(ego=ego, front_obj=front_obj, front_dist=3.0, idm_acc=-5.0)

    assert adjusted == -5.0


def test_rate_limit_applies_across_sequential_calls():
    guard = RearEndGuardRegulator()
    ego = SimpleNamespace(speed=12.0)
    front_obj = SimpleNamespace(speed=0.0)

    first = guard.adjust_acceleration(ego=ego, front_obj=None, front_dist=100.0, idm_acc=0.5)
    second = guard.adjust_acceleration(ego=ego, front_obj=front_obj, front_dist=3.0, idm_acc=0.5)

    assert first == 0.5
    assert second == -0.5


def test_reset_clears_internal_rate_limit_state():
    guard = RearEndGuardRegulator()
    ego = SimpleNamespace(speed=12.0)
    front_obj = SimpleNamespace(speed=0.0)

    guard.adjust_acceleration(ego=ego, front_obj=None, front_dist=100.0, idm_acc=0.5)
    limited = guard.adjust_acceleration(ego=ego, front_obj=front_obj, front_dist=3.0, idm_acc=0.5)
    guard.reset()
    reset_output = guard.adjust_acceleration(ego=ego, front_obj=front_obj, front_dist=3.0, idm_acc=0.5)

    assert limited == -0.5
    assert reset_output == guard.HARD_BRAKE
