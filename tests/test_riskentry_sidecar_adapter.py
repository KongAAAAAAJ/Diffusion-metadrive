from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from expert_dataset.riskentry_sidecar_adapter import (
    MetaDriveRiskEntrySidecarAdapter,
    RiskEntrySidecarAdapterError,
)
from metadrive.type import MetaDriveType


class _Lane:
    def __init__(self, index=("A", "B", 0), *, width=3.5):
        self.index = index
        self.width = float(width)
        self.metadrive_type = MetaDriveType.LANE_SURFACE_STREET

    def local_coordinates(self, position):
        return float(position[0]), float(position[1])

    def heading_theta_at(self, longitudinal):  # noqa: ARG002
        return 0.0

    def width_at(self, longitudinal):  # noqa: ARG002
        return self.width


class _Vehicle:
    LENGTH = 5.74
    WIDTH = 2.3
    metadrive_type = MetaDriveType.VEHICLE

    def __init__(
        self,
        name: str,
        *,
        position=(0.0, 0.0),
        heading=0.0,
        velocity=(0.0, 0.0),
        lane=None,
        scenario_role=None,
    ):
        self.name = name
        self.position = np.asarray(position, dtype=np.float64)
        self.heading_theta = float(heading)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.lane = lane
        if scenario_role is not None:
            self.scenario_role = scenario_role


class _StaticObject:
    metadrive_type = MetaDriveType.TRAFFIC_OBJECT


class _Engine:
    def __init__(self, objects):
        self.objects = dict(objects)

    def get_objects(self):
        return self.objects


class _Orchestrator:
    def __init__(self):
        self.triggered = False
        self.realized = False
        self.trigger_step = None
        self.realized_step = None

    def get_episode_summary(self):
        return {
            "scenario_id": "S5_hard_brake_lead",
            "scenario_triggered": self.triggered,
            "scenario_realized": self.realized,
            "scenario_trigger_step": self.trigger_step,
            "scenario_realized_step": self.realized_step,
            "scenario_notes": ["fixture"],
        }


def _env(*, externals=()):
    lane = _Lane()
    agents = {
        "agent0": _Vehicle("ego0", position=(10.0, 0.0), velocity=(4.0, 0.0), lane=lane),
        "agent1": _Vehicle("ego1", position=(2.0, 0.0), velocity=(4.0, 0.0), lane=lane),
        "agent2": _Vehicle("ego2", position=(-6.0, 0.0), velocity=(4.0, 0.0), lane=lane),
    }
    objects = {vehicle.name: vehicle for vehicle in agents.values()}
    objects.update(externals)
    objects["road-cone"] = _StaticObject()
    return SimpleNamespace(
        agents=agents,
        engine=_Engine(objects),
        _scenario_orchestrator=_Orchestrator(),
    )


def _snapshot(capture, actor_id):
    return next(item for item in capture.frame.actors if item.actor_id == actor_id)


def test_registry_assigns_frozen_platoon_and_sorted_external_ids():
    lane = _Lane(("ramp", "merge", 0))
    external_b = _Vehicle(
        "traffic-b", position=(20.0, 3.5), velocity=(5.0, 0.0), lane=lane
    )
    external_a = _Vehicle(
        "traffic-a",
        position=(15.0, 3.5),
        velocity=(5.0, 0.0),
        lane=lane,
        scenario_role="hard_brake_lead",
    )
    env = _env(externals=(("z-key", external_b), ("a-key", external_a)))
    adapter = MetaDriveRiskEntrySidecarAdapter()

    capture = adapter.capture_frame(env, step_index=0, timestamp_s=0.0)

    assert [item.actor_id for item in adapter.actor_records] == [
        "P0",
        "P1",
        "P2",
        "V000",
        "V001",
    ]
    assert [item.source_object_id for item in adapter.actor_records] == [
        "agent0",
        "agent1",
        "agent2",
        "a-key",
        "z-key",
    ]
    assert [item.platoon_role for item in adapter.actor_records[:3]] == [
        "leader",
        "middle",
        "rear",
    ]
    assert [item.actor_id for item in capture.frame.actors] == [
        "P0",
        "P1",
        "P2",
        "V000",
        "V001",
    ]
    assert adapter.key_actor_ids == {"lead_braker": "V000"}
    assert len(adapter.lane_records) == 2
    assert all(item.acceleration_valid is False for item in capture.frame.actors)
    assert all(item.yaw_rate_valid is False for item in capture.frame.actors)


def test_world_acceleration_and_wrapped_yaw_rate_use_adjacent_frames():
    env = _env()
    env.agents["agent0"].heading_theta = math.pi - 0.02
    adapter = MetaDriveRiskEntrySidecarAdapter(decision_dt_s=0.1)
    first = adapter.capture_frame(env, step_index=0, timestamp_s=0.0)
    assert _snapshot(first, "P0").acceleration_valid is False

    env.agents["agent0"].velocity = np.asarray([4.2, -0.1])
    env.agents["agent0"].heading_theta = -math.pi + 0.03
    second = adapter.capture_frame(env, step_index=1, timestamp_s=0.1)
    state = _snapshot(second, "P0")

    assert state.acceleration_valid is True
    assert state.yaw_rate_valid is True
    assert state.acceleration_x_mps2 == pytest.approx(2.0)
    assert state.acceleration_y_mps2 == pytest.approx(-1.0)
    assert state.yaw_rate_radps == pytest.approx(0.5)


def test_lane_state_uses_explicit_mask_and_stable_lane_table():
    env = _env()
    env.agents["agent2"].lane = None
    adapter = MetaDriveRiskEntrySidecarAdapter()

    capture = adapter.capture_frame(env, step_index=0, timestamp_s=0.0)
    leader = _snapshot(capture, "P0")
    rear = _snapshot(capture, "P2")

    assert leader.lane_valid is True
    assert leader.lane_id == "L000"
    assert leader.lane_s_m == pytest.approx(10.0)
    assert leader.lane_lateral_m == pytest.approx(0.0)
    assert leader.lane_width_m == pytest.approx(3.5)
    assert rear.lane_valid is False
    assert rear.lane_id is None
    assert rear.lane_s_m == rear.lane_lateral_m == rear.lane_width_m == 0.0


def test_despawn_and_reappearance_preserve_actor_id_and_invalidate_derivative():
    traffic = _Vehicle(
        "traffic", position=(20.0, 3.5), velocity=(5.0, 0.0), lane=_Lane()
    )
    env = _env(externals=(("stable-registry-key", traffic),))
    adapter = MetaDriveRiskEntrySidecarAdapter()
    first = adapter.capture_frame(env, step_index=0, timestamp_s=0.0)
    assert _snapshot(first, "V000").actor_id == "V000"

    del env.engine.objects["stable-registry-key"]
    second = adapter.capture_frame(env, step_index=1, timestamp_s=0.1)
    despawns = [event for event in second.events if event.event_type == "actor_despawn"]
    assert len(despawns) == 1
    assert despawns[0].actor_ids == ("V000",)

    env.engine.objects["stable-registry-key"] = traffic
    traffic.position = np.asarray([21.0, 3.5])
    third = adapter.capture_frame(env, step_index=2, timestamp_s=0.2)
    state = _snapshot(third, "V000")
    assert state.actor_id == "V000"
    assert state.acceleration_valid is False
    assert len(adapter.actor_records) == 4


def test_transition_and_scenario_events_belong_to_result_state_and_are_deduplicated():
    env = _env()
    adapter = MetaDriveRiskEntrySidecarAdapter()
    adapter.capture_frame(env, step_index=0, timestamp_s=0.0)
    env._scenario_orchestrator.triggered = True
    env._scenario_orchestrator.realized = True
    env._scenario_orchestrator.trigger_step = 0
    env._scenario_orchestrator.realized_step = 0

    capture = adapter.capture_frame(
        env,
        step_index=1,
        timestamp_s=0.1,
        transition_info={
            "agent0": {"crash_vehicle": True},
            "agent1": {"crash_sidewalk": True, "out_of_road": True},
            "agent2": {},
        },
        terminated={"agent0": True, "agent1": True, "agent2": False, "__all__": True},
        truncated={"__all__": False},
    )
    by_type = {event.event_type: event for event in capture.events}

    assert set(by_type) == {
        "scenario_trigger",
        "scenario_realized",
        "collision_vehicle",
        "collision_sidewalk",
        "out_of_road",
        "terminated",
    }
    assert all(event.step_index == 1 for event in capture.events)
    assert all(event.timestamp_s == pytest.approx(0.1) for event in capture.events)
    assert by_type["collision_vehicle"].actor_ids == ("P0",)
    assert by_type["collision_vehicle"].terminal is True
    assert by_type["terminated"].actor_ids == ("P0", "P1")

    repeated = adapter.capture_frame(
        env,
        step_index=2,
        timestamp_s=0.2,
        transition_info={"agent0": {"crash_vehicle": True}},
        terminated={"__all__": True},
    )
    assert not repeated.events


def test_truncation_and_out_of_route_are_raw_events():
    env = _env()
    adapter = MetaDriveRiskEntrySidecarAdapter()
    adapter.capture_frame(env, step_index=0, timestamp_s=0.0)
    result = adapter.capture_frame(
        env,
        step_index=1,
        timestamp_s=0.1,
        transition_info={"agent2": {"out_of_route": True}},
        truncated={"agent0": False, "agent1": False, "agent2": True},
    )
    assert [event.event_type for event in result.events] == ["out_of_route", "truncated"]
    assert result.events[0].actor_ids == ("P2",)
    assert result.events[1].actor_ids == ("P2",)


def test_external_actor_collision_is_preserved_from_registry_state():
    traffic = _Vehicle("traffic", lane=_Lane())
    env = _env(externals=(("traffic-key", traffic),))
    adapter = MetaDriveRiskEntrySidecarAdapter()
    adapter.capture_frame(env, step_index=0, timestamp_s=0.0)
    traffic.crash_vehicle = True

    result = adapter.capture_frame(env, step_index=1, timestamp_s=0.1)

    collisions = [event for event in result.events if event.event_type == "collision_vehicle"]
    assert len(collisions) == 1
    assert collisions[0].actor_ids == ("V000",)
    assert collisions[0].terminal is False
    assert collisions[0].details["source_object_id"] == "traffic-key"


def test_adapter_rejects_timeline_gap_bad_timestamp_and_missing_initial_agent():
    env = _env()
    adapter = MetaDriveRiskEntrySidecarAdapter()
    adapter.capture_frame(env, step_index=0, timestamp_s=0.0)
    with pytest.raises(RiskEntrySidecarAdapterError, match="contiguous"):
        adapter.capture_frame(env, step_index=2, timestamp_s=0.2)

    adapter.reset()
    with pytest.raises(RiskEntrySidecarAdapterError, match="timestamp"):
        adapter.capture_frame(env, step_index=0, timestamp_s=0.1)

    adapter.reset()
    del env.agents["agent2"]
    with pytest.raises(RiskEntrySidecarAdapterError, match="missing platoon"):
        adapter.capture_frame(env, step_index=0, timestamp_s=0.0)


def test_adapter_rejects_duplicate_registry_alias_for_one_vehicle():
    traffic = _Vehicle("traffic", lane=_Lane())
    env = _env(externals=(("first", traffic), ("second", traffic)))
    adapter = MetaDriveRiskEntrySidecarAdapter()
    with pytest.raises(RiskEntrySidecarAdapterError, match="multiple registry keys"):
        adapter.capture_frame(env, step_index=0, timestamp_s=0.0)
