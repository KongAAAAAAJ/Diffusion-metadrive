"""Read-only MetaDrive adapter for the RiskEntry actor sidecar.

This module owns no persistence.  It turns one live simulator state boundary
into stable episode-local actor/lane records, actor snapshots, and raw events.
The writer and the joint-BEV collector integration intentionally remain
separate so a dangerous episode can later be retained by RiskEntry even when
the planning dataset rejects it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from expert_dataset.joint_risk_identity import PLATOON_AGENT_TO_ACTOR_ID
from metadrive.type import MetaDriveType


PLATOON_ROLES = {
    "agent0": "leader",
    "agent1": "middle",
    "agent2": "rear",
}
RAW_EVENT_TYPES = frozenset(
    {
        "scenario_trigger",
        "scenario_realized",
        "collision_vehicle",
        "collision_object",
        "collision_sidewalk",
        "out_of_road",
        "out_of_route",
        "terminated",
        "truncated",
        "actor_despawn",
    }
)


class RiskEntrySidecarAdapterError(ValueError):
    """Raised when live simulator state violates the frozen sidecar contract."""


@dataclass(frozen=True)
class SidecarActorRecord:
    actor_index: int
    actor_id: str
    source_object_id: str
    actor_type: str
    platoon_role: str | None
    first_seen_step: int
    length_m: float
    width_m: float


@dataclass(frozen=True)
class SidecarLaneRecord:
    lane_index: int
    lane_id: str
    source_lane_id: str
    lane_type: str


@dataclass(frozen=True)
class SidecarActorSnapshot:
    """Producer-side equivalent of RiskEntry ``ActorSnapshot``."""

    actor_id: str
    source_object_id: str
    actor_type: str
    world_x_m: float
    world_y_m: float
    heading_rad: float
    velocity_x_mps: float
    velocity_y_mps: float
    length_m: float
    width_m: float
    acceleration_x_mps2: float = 0.0
    acceleration_y_mps2: float = 0.0
    yaw_rate_radps: float = 0.0
    acceleration_valid: bool = False
    yaw_rate_valid: bool = False
    lane_id: str | None = None
    lane_s_m: float = 0.0
    lane_lateral_m: float = 0.0
    lane_heading_error_rad: float = 0.0
    lane_width_m: float = 0.0
    lane_valid: bool = False

    def __post_init__(self) -> None:
        values = (
            self.world_x_m,
            self.world_y_m,
            self.heading_rad,
            self.velocity_x_mps,
            self.velocity_y_mps,
            self.acceleration_x_mps2,
            self.acceleration_y_mps2,
            self.yaw_rate_radps,
            self.length_m,
            self.width_m,
            self.lane_s_m,
            self.lane_lateral_m,
            self.lane_heading_error_rad,
            self.lane_width_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise RiskEntrySidecarAdapterError("actor snapshot contains a non-finite value")
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise RiskEntrySidecarAdapterError("actor dimensions must be positive")
        if self.lane_valid and (self.lane_id is None or self.lane_width_m <= 0.0):
            raise RiskEntrySidecarAdapterError(
                "valid lane state requires lane_id and positive lane width"
            )


@dataclass(frozen=True)
class SidecarRawEvent:
    event_type: str
    step_index: int
    timestamp_s: float
    actor_ids: tuple[str, ...] = ()
    terminal: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in RAW_EVENT_TYPES:
            raise RiskEntrySidecarAdapterError(
                f"unsupported raw event type: {self.event_type!r}"
            )
        if self.step_index < 0 or not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise RiskEntrySidecarAdapterError("invalid event time")
        if len(set(self.actor_ids)) != len(self.actor_ids):
            raise RiskEntrySidecarAdapterError("event actor IDs must be unique")


@dataclass(frozen=True)
class SidecarRawFrame:
    step_index: int
    timestamp_s: float
    actors: tuple[SidecarActorSnapshot, ...]


@dataclass(frozen=True)
class SidecarFrameCapture:
    frame: SidecarRawFrame
    events: tuple[SidecarRawEvent, ...]


@dataclass(frozen=True)
class _PreviousActorState:
    step_index: int
    timestamp_s: float
    velocity: np.ndarray
    heading_rad: float


def _wrap_angle(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_source_id(value: Any, *, kind: str) -> str:
    if isinstance(value, str):
        result = value
    else:
        result = json.dumps(
            _json_scalar(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    if not result:
        raise RiskEntrySidecarAdapterError(f"empty {kind} registry identifier")
    return result


def _as_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception as exc:  # pragma: no cover - defensive against simulator extensions
        raise RiskEntrySidecarAdapterError(f"unable to read {name}") from exc
    if vector.size < size or not np.isfinite(vector[:size]).all():
        raise RiskEntrySidecarAdapterError(f"{name} must contain {size} finite values")
    return vector[:size].copy()


def _mapping_flag(value: bool | Mapping[str, Any] | None, agent_id: str) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get(agent_id, value.get("__all__", False)))
    return bool(value)


class MetaDriveRiskEntrySidecarAdapter:
    """Extract stable actor facts and raw events from one MetaDrive episode.

    ``capture_frame`` must be called at every state boundary, starting at zero.
    For action ``k``, pass the returned ``info/terminated/truncated`` to the
    capture of state ``k+1``.  This pins transition events to their result
    state and naturally supports the required terminal post-step snapshot.
    """

    def __init__(self, *, decision_dt_s: float = 0.1) -> None:
        if not math.isfinite(decision_dt_s) or decision_dt_s <= 0.0:
            raise RiskEntrySidecarAdapterError("decision_dt_s must be positive and finite")
        self.decision_dt_s = float(decision_dt_s)
        self.reset()

    def reset(self) -> None:
        self._actor_records_by_source: dict[str, SidecarActorRecord] = {}
        self._lane_records_by_source: dict[str, SidecarLaneRecord] = {}
        self._previous_actor_state: dict[str, _PreviousActorState] = {}
        self._previous_active_sources: set[str] = set()
        self._key_actor_ids: dict[str, str] = {}
        self._last_step_index: int | None = None
        self._scenario_triggered = False
        self._scenario_realized = False
        self._emitted_agent_events: set[tuple[str, str]] = set()
        self._terminated_emitted = False
        self._truncated_emitted = False

    @property
    def actor_records(self) -> tuple[SidecarActorRecord, ...]:
        return tuple(
            sorted(self._actor_records_by_source.values(), key=lambda item: item.actor_index)
        )

    @property
    def lane_records(self) -> tuple[SidecarLaneRecord, ...]:
        return tuple(
            sorted(self._lane_records_by_source.values(), key=lambda item: item.lane_index)
        )

    @property
    def key_actor_ids(self) -> Mapping[str, str]:
        return dict(sorted(self._key_actor_ids.items()))

    def capture_frame(
        self,
        env: Any,
        *,
        step_index: int,
        timestamp_s: float,
        transition_info: Mapping[str, Any] | None = None,
        terminated: bool | Mapping[str, Any] | None = None,
        truncated: bool | Mapping[str, Any] | None = None,
    ) -> SidecarFrameCapture:
        self._validate_frame_time(step_index, timestamp_s)
        discovered = self._discover_vehicles(env, step_index=step_index)
        active_sources = set(discovered)
        events: list[SidecarRawEvent] = []

        for source_object_id in sorted(self._previous_active_sources - active_sources):
            record = self._actor_records_by_source[source_object_id]
            events.append(
                SidecarRawEvent(
                    event_type="actor_despawn",
                    step_index=step_index,
                    timestamp_s=float(timestamp_s),
                    actor_ids=(record.actor_id,),
                    details={"source_object_id": source_object_id},
                )
            )

        snapshots = [
            self._snapshot_vehicle(
                source_object_id,
                vehicle,
                step_index=step_index,
                timestamp_s=float(timestamp_s),
            )
            for source_object_id, vehicle in discovered.items()
        ]
        snapshots.sort(
            key=lambda item: self._actor_records_by_source[item.source_object_id].actor_index
        )

        events.extend(self._scenario_events(env, step_index, float(timestamp_s)))
        events.extend(
            self._transition_events(
                env,
                discovered,
                transition_info or {},
                terminated=terminated,
                truncated=truncated,
                step_index=step_index,
                timestamp_s=float(timestamp_s),
            )
        )

        self._previous_active_sources = active_sources
        self._last_step_index = int(step_index)
        return SidecarFrameCapture(
            frame=SidecarRawFrame(
                step_index=int(step_index),
                timestamp_s=float(timestamp_s),
                actors=tuple(snapshots),
            ),
            events=tuple(events),
        )

    def _validate_frame_time(self, step_index: int, timestamp_s: float) -> None:
        if not isinstance(step_index, (int, np.integer)) or int(step_index) < 0:
            raise RiskEntrySidecarAdapterError("step_index must be a non-negative integer")
        expected_step = 0 if self._last_step_index is None else self._last_step_index + 1
        if int(step_index) != expected_step:
            raise RiskEntrySidecarAdapterError(
                f"raw timeline must be contiguous: expected step {expected_step}, got {step_index}"
            )
        expected_time = int(step_index) * self.decision_dt_s
        if not math.isfinite(timestamp_s) or not math.isclose(
            float(timestamp_s), expected_time, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RiskEntrySidecarAdapterError(
                f"timestamp must equal step_index * decision_dt_s ({expected_time})"
            )

    @staticmethod
    def _engine_objects(env: Any) -> Mapping[Any, Any]:
        engine = getattr(env, "engine", None)
        if engine is None:
            unwrapped = getattr(env, "unwrapped", None)
            engine = getattr(unwrapped, "engine", None)
        getter = getattr(engine, "get_objects", None)
        if getter is None:
            raise RiskEntrySidecarAdapterError("MetaDrive engine.get_objects() is unavailable")
        objects = getter()
        if not isinstance(objects, Mapping):
            raise RiskEntrySidecarAdapterError("MetaDrive object registry must be a mapping")
        return objects

    def _discover_vehicles(self, env: Any, *, step_index: int) -> dict[str, Any]:
        agents = getattr(env, "agents", None)
        if not isinstance(agents, Mapping):
            raise RiskEntrySidecarAdapterError("env.agents must be a mapping")
        if step_index == 0:
            missing = sorted(set(PLATOON_AGENT_TO_ACTOR_ID) - set(agents))
            if missing:
                raise RiskEntrySidecarAdapterError(
                    f"initial frame is missing platoon agents: {missing}"
                )

        discovered: dict[str, Any] = {}
        agent_objects: set[int] = set()
        for agent_id in PLATOON_AGENT_TO_ACTOR_ID:
            vehicle = agents.get(agent_id)
            if vehicle is None:
                continue
            source = str(agent_id)
            discovered[source] = vehicle
            agent_objects.add(id(vehicle))
            self._register_actor(
                source,
                vehicle,
                step_index=step_index,
                actor_id=PLATOON_AGENT_TO_ACTOR_ID[agent_id],
                actor_type="platoon",
                platoon_role=PLATOON_ROLES[agent_id],
            )

        pending_external: list[tuple[str, Any]] = []
        seen_object_ids: dict[int, str] = {}
        for registry_key, obj in self._engine_objects(env).items():
            if id(obj) in agent_objects or not self._is_vehicle(obj):
                continue
            source = _canonical_source_id(registry_key, kind="object")
            previous_source = seen_object_ids.setdefault(id(obj), source)
            if previous_source != source:
                raise RiskEntrySidecarAdapterError(
                    "one live vehicle appears under multiple registry keys"
                )
            if source in discovered:
                raise RiskEntrySidecarAdapterError(
                    f"object registry source ID collides with platoon source: {source!r}"
                )
            pending_external.append((source, obj))

        for source, vehicle in sorted(pending_external, key=lambda item: item[0]):
            if source in discovered:
                raise RiskEntrySidecarAdapterError(
                    f"duplicate object registry source ID: {source!r}"
                )
            discovered[source] = vehicle
            if source not in self._actor_records_by_source:
                actor_id = f"V{self._external_actor_count():03d}"
                self._register_actor(
                    source,
                    vehicle,
                    step_index=step_index,
                    actor_id=actor_id,
                    actor_type="external",
                    platoon_role=None,
                )
            self._record_scenario_role(source, vehicle)
        return dict(sorted(discovered.items()))

    @staticmethod
    def _is_vehicle(obj: Any) -> bool:
        object_type = getattr(obj, "metadrive_type", None)
        if object_type is not None:
            return MetaDriveType.is_vehicle(object_type)
        return all(
            hasattr(obj, name)
            for name in ("position", "heading_theta", "velocity", "LENGTH", "WIDTH")
        )

    def _external_actor_count(self) -> int:
        return sum(record.actor_type == "external" for record in self._actor_records_by_source.values())

    def _register_actor(
        self,
        source: str,
        vehicle: Any,
        *,
        step_index: int,
        actor_id: str,
        actor_type: str,
        platoon_role: str | None,
    ) -> None:
        length_m = self._positive_dimension(vehicle, "LENGTH")
        width_m = self._positive_dimension(vehicle, "WIDTH")
        existing = self._actor_records_by_source.get(source)
        if existing is not None:
            if existing.actor_id != actor_id or existing.actor_type != actor_type:
                raise RiskEntrySidecarAdapterError("unstable actor identity assignment")
            if not math.isclose(existing.length_m, length_m, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
                existing.width_m, width_m, rel_tol=0.0, abs_tol=1e-6
            ):
                raise RiskEntrySidecarAdapterError("actor dimensions changed during episode")
            return
        record = SidecarActorRecord(
            actor_index=len(self._actor_records_by_source),
            actor_id=actor_id,
            source_object_id=source,
            actor_type=actor_type,
            platoon_role=platoon_role,
            first_seen_step=int(step_index),
            length_m=length_m,
            width_m=width_m,
        )
        self._actor_records_by_source[source] = record

    @staticmethod
    def _positive_dimension(vehicle: Any, name: str) -> float:
        try:
            value = float(getattr(vehicle, name))
        except Exception as exc:
            raise RiskEntrySidecarAdapterError(f"vehicle {name} is unavailable") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise RiskEntrySidecarAdapterError(f"vehicle {name} must be positive and finite")
        return value

    def _record_scenario_role(self, source: str, vehicle: Any) -> None:
        role = getattr(vehicle, "scenario_role", None)
        if role is None:
            role = getattr(vehicle, "scenario_vehicle_role", None)
        if role is None:
            return
        role_name = str(role)
        aliases = {
            "hard_brake_lead": "lead_braker",
            "s6_merge_vehicle": "intruder",
            "injected_background": "intruder",
            "cut_in_adjacent_lead": "intruder",
        }
        key = aliases.get(role_name, role_name)
        actor_id = self._actor_records_by_source[source].actor_id
        previous = self._key_actor_ids.get(key)
        if previous is not None and previous != actor_id:
            raise RiskEntrySidecarAdapterError(
                f"scenario key actor role {key!r} maps to multiple actors"
            )
        self._key_actor_ids[key] = actor_id

    def _snapshot_vehicle(
        self,
        source: str,
        vehicle: Any,
        *,
        step_index: int,
        timestamp_s: float,
    ) -> SidecarActorSnapshot:
        record = self._actor_records_by_source[source]
        position = _as_vector(getattr(vehicle, "position", None), size=2, name="position")
        heading = _wrap_angle(float(getattr(vehicle, "heading_theta")))
        if not math.isfinite(heading):
            raise RiskEntrySidecarAdapterError("vehicle heading must be finite")
        velocity = self._world_velocity(vehicle, heading)

        acceleration = np.zeros(2, dtype=np.float64)
        yaw_rate = 0.0
        derivative_valid = False
        previous = self._previous_actor_state.get(source)
        if previous is not None and previous.step_index == step_index - 1:
            elapsed = timestamp_s - previous.timestamp_s
            if elapsed > 0.0 and math.isclose(
                elapsed, self.decision_dt_s, rel_tol=0.0, abs_tol=1e-6
            ):
                acceleration = (velocity - previous.velocity) / elapsed
                yaw_rate = _wrap_angle(heading - previous.heading_rad) / elapsed
                derivative_valid = bool(
                    np.isfinite(acceleration).all() and math.isfinite(yaw_rate)
                )
        self._previous_actor_state[source] = _PreviousActorState(
            step_index=int(step_index),
            timestamp_s=float(timestamp_s),
            velocity=velocity.copy(),
            heading_rad=heading,
        )

        lane_values = self._lane_state(vehicle, position, heading)
        return SidecarActorSnapshot(
            actor_id=record.actor_id,
            source_object_id=source,
            actor_type=record.actor_type,
            world_x_m=float(position[0]),
            world_y_m=float(position[1]),
            heading_rad=heading,
            velocity_x_mps=float(velocity[0]),
            velocity_y_mps=float(velocity[1]),
            length_m=record.length_m,
            width_m=record.width_m,
            acceleration_x_mps2=float(acceleration[0]) if derivative_valid else 0.0,
            acceleration_y_mps2=float(acceleration[1]) if derivative_valid else 0.0,
            yaw_rate_radps=float(yaw_rate) if derivative_valid else 0.0,
            acceleration_valid=derivative_valid,
            yaw_rate_valid=derivative_valid,
            lane_id=lane_values[0],
            lane_s_m=lane_values[1],
            lane_lateral_m=lane_values[2],
            lane_heading_error_rad=lane_values[3],
            lane_width_m=lane_values[4],
            lane_valid=lane_values[5],
        )

    @staticmethod
    def _world_velocity(vehicle: Any, heading: float) -> np.ndarray:
        try:
            return _as_vector(getattr(vehicle, "velocity"), size=2, name="world velocity")
        except (AttributeError, RiskEntrySidecarAdapterError):
            try:
                speed = float(getattr(vehicle, "speed"))
            except Exception as exc:
                raise RiskEntrySidecarAdapterError("vehicle world velocity is unavailable") from exc
            if not math.isfinite(speed) or speed < 0.0:
                raise RiskEntrySidecarAdapterError("vehicle speed must be finite and non-negative")
            return np.asarray(
                [speed * math.cos(heading), speed * math.sin(heading)], dtype=np.float64
            )

    def _lane_state(
        self, vehicle: Any, position: np.ndarray, heading: float
    ) -> tuple[str | None, float, float, float, float, bool]:
        lane = getattr(vehicle, "lane", None)
        if lane is None:
            return None, 0.0, 0.0, 0.0, 0.0, False
        try:
            source_lane_id = _canonical_source_id(
                getattr(lane, "index"), kind="lane"
            )
            lane_s, lateral = lane.local_coordinates(position)
            lane_heading = float(lane.heading_theta_at(float(lane_s)))
            lane_width = float(lane.width_at(float(lane_s)))
            values = np.asarray(
                [lane_s, lateral, lane_heading, lane_width], dtype=np.float64
            )
            if not np.isfinite(values).all() or lane_width <= 0.0:
                raise ValueError("invalid lane observation")
        except Exception:
            return None, 0.0, 0.0, 0.0, 0.0, False
        record = self._lane_records_by_source.get(source_lane_id)
        if record is None:
            record = SidecarLaneRecord(
                lane_index=len(self._lane_records_by_source),
                lane_id=f"L{len(self._lane_records_by_source):03d}",
                source_lane_id=source_lane_id,
                lane_type=self._classify_lane(lane),
            )
            self._lane_records_by_source[source_lane_id] = record
        return (
            record.lane_id,
            float(lane_s),
            float(lateral),
            _wrap_angle(heading - lane_heading),
            lane_width,
            True,
        )

    @staticmethod
    def _classify_lane(lane: Any) -> str:
        lane_index = getattr(lane, "index", None)
        text = " ".join(
            (
                str(lane_index).lower(),
                type(lane).__name__.lower(),
                str(getattr(lane, "metadrive_type", "")).lower(),
            )
        )
        if "shoulder" in text:
            return "shoulder"
        if "merge" in text or "ramp" in text or "connector" in text:
            return "merge"
        if "exit" in text:
            return "exit"
        if "intersection" in text or "junction" in text:
            return "intersection"
        if "mainline" in text or "freeway" in text:
            return "mainline"
        return "unknown"

    def _scenario_events(
        self, env: Any, step_index: int, timestamp_s: float
    ) -> list[SidecarRawEvent]:
        orchestrator = getattr(env, "_scenario_orchestrator", None)
        getter = getattr(orchestrator, "get_episode_summary", None)
        if getter is None:
            return []
        summary = getter()
        if not isinstance(summary, Mapping):
            raise RiskEntrySidecarAdapterError("scenario summary must be a mapping")
        events: list[SidecarRawEvent] = []
        triggered = bool(summary.get("scenario_triggered", False))
        realized = bool(summary.get("scenario_realized", False))
        if triggered and not self._scenario_triggered:
            events.append(
                SidecarRawEvent(
                    event_type="scenario_trigger",
                    step_index=step_index,
                    timestamp_s=timestamp_s,
                    details=self._scenario_event_details(summary, "scenario_trigger_step"),
                )
            )
        if realized and not self._scenario_realized:
            events.append(
                SidecarRawEvent(
                    event_type="scenario_realized",
                    step_index=step_index,
                    timestamp_s=timestamp_s,
                    details=self._scenario_event_details(summary, "scenario_realized_step"),
                )
            )
        self._scenario_triggered = self._scenario_triggered or triggered
        self._scenario_realized = self._scenario_realized or realized
        return events

    @staticmethod
    def _scenario_event_details(summary: Mapping[str, Any], step_key: str) -> dict[str, Any]:
        return {
            "scenario_id": _json_scalar(summary.get("scenario_id")),
            "reported_step": _json_scalar(summary.get(step_key)),
            "scenario_notes": _json_scalar(summary.get("scenario_notes", [])),
        }

    def _transition_events(
        self,
        env: Any,
        discovered: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        terminated: bool | Mapping[str, Any] | None,
        truncated: bool | Mapping[str, Any] | None,
        step_index: int,
        timestamp_s: float,
    ) -> list[SidecarRawEvent]:
        events: list[SidecarRawEvent] = []
        event_flags = (
            ("collision_vehicle", ("crash_vehicle",)),
            (
                "collision_object",
                ("crash_object", "crash_building", "crash_human"),
            ),
            ("collision_sidewalk", ("crash_sidewalk",)),
            ("out_of_road", ("out_of_road",)),
            ("out_of_route", ("out_of_route",)),
        )
        source_to_agent_id = {
            source: source if source in PLATOON_AGENT_TO_ACTOR_ID else None
            for source in discovered
        }
        for source, vehicle in discovered.items():
            record = self._actor_records_by_source[source]
            actor_id = record.actor_id
            agent_id = source_to_agent_id[source]
            agent_info = (
                info.get(agent_id, {})
                if agent_id is not None and isinstance(info, Mapping)
                else {}
            )
            if not isinstance(agent_info, Mapping):
                agent_info = {}
            raw_flags = {
                key: bool(agent_info.get(key, getattr(vehicle, key, False)))
                for _, keys in event_flags
                for key in keys
            }
            if agent_id is not None and not raw_flags["out_of_road"]:
                out_checker = getattr(env, "_agent_is_out_of_road", None)
                if callable(out_checker):
                    try:
                        raw_flags["out_of_road"] = bool(out_checker(agent_id))
                    except Exception:
                        pass
            if not raw_flags["out_of_road"] and getattr(vehicle, "on_lane", None) is False:
                raw_flags["out_of_road"] = True
            agent_terminal = agent_id is not None and (
                _mapping_flag(terminated, agent_id)
                or _mapping_flag(truncated, agent_id)
            )
            for event_type, keys in event_flags:
                if not any(raw_flags[key] for key in keys):
                    continue
                event_key = (event_type, actor_id)
                if event_key in self._emitted_agent_events:
                    continue
                self._emitted_agent_events.add(event_key)
                events.append(
                    SidecarRawEvent(
                        event_type=event_type,
                        step_index=step_index,
                        timestamp_s=timestamp_s,
                        actor_ids=(actor_id,),
                        terminal=agent_terminal,
                        details={
                            "source_agent_id": agent_id,
                            "source_object_id": source,
                            "raw_flags": {key: raw_flags[key] for key in keys},
                        },
                    )
                )

        any_terminated = any(
            _mapping_flag(terminated, agent_id) for agent_id in PLATOON_AGENT_TO_ACTOR_ID
        ) or (isinstance(terminated, Mapping) and bool(terminated.get("__all__", False)))
        any_truncated = any(
            _mapping_flag(truncated, agent_id) for agent_id in PLATOON_AGENT_TO_ACTOR_ID
        ) or (isinstance(truncated, Mapping) and bool(truncated.get("__all__", False)))
        if bool(terminated) and not isinstance(terminated, Mapping):
            any_terminated = True
        if bool(truncated) and not isinstance(truncated, Mapping):
            any_truncated = True
        if any_terminated and not self._terminated_emitted:
            self._terminated_emitted = True
            events.append(
                SidecarRawEvent(
                    event_type="terminated",
                    step_index=step_index,
                    timestamp_s=timestamp_s,
                    actor_ids=tuple(
                        actor_id
                        for agent_id, actor_id in PLATOON_AGENT_TO_ACTOR_ID.items()
                        if _mapping_flag(terminated, agent_id)
                    ),
                    terminal=True,
                    details={},
                )
            )
        if any_truncated and not self._truncated_emitted:
            self._truncated_emitted = True
            events.append(
                SidecarRawEvent(
                    event_type="truncated",
                    step_index=step_index,
                    timestamp_s=timestamp_s,
                    actor_ids=tuple(
                        actor_id
                        for agent_id, actor_id in PLATOON_AGENT_TO_ACTOR_ID.items()
                        if _mapping_flag(truncated, agent_id)
                    ),
                    terminal=True,
                    details={},
                )
            )
        return events


__all__ = [
    "MetaDriveRiskEntrySidecarAdapter",
    "PLATOON_ROLES",
    "RAW_EVENT_TYPES",
    "RiskEntrySidecarAdapterError",
    "SidecarActorRecord",
    "SidecarActorSnapshot",
    "SidecarFrameCapture",
    "SidecarLaneRecord",
    "SidecarRawEvent",
    "SidecarRawFrame",
]
