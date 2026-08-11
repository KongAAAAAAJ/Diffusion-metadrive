"""Deterministic constrained sampling for the candidate S5--S9 contract.

The sampler deliberately owns *correlated* logical parameters.  Scenario
recipes consume one resolved dictionary per episode instead of drawing every
field independently from MetaDrive's traffic RNG.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


SEVERITY_WEIGHTS: Mapping[str, float] = {
    "low": 0.3,
    "medium": 0.4,
    "high": 0.3,
}


def scenario_seed(spawn_seed: int, scenario_id: str, local_route: str) -> int:
    payload = f"{int(spawn_seed)}\0{scenario_id}\0{local_route}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _rng(spawn_seed: int, scenario_id: str, local_route: str) -> np.random.RandomState:
    return np.random.RandomState(scenario_seed(spawn_seed, scenario_id, local_route))


def _uniform(rng: np.random.RandomState, bounds) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _severity(rng: np.random.RandomState) -> str:
    draw = float(rng.uniform(0.0, 1.0))
    if draw < 0.3:
        return "low"
    if draw < 0.7:
        return "medium"
    return "high"


def resolve_s5_s9_parameters(
    *,
    spawn_seed: int,
    scenario_id: str,
    local_route: str,
    ego_initial_speed_km_h: float,
    decision_dt_s: float = 0.1,
) -> dict[str, object]:
    """Resolve one finite, JSON-compatible logical scenario parameter set."""

    rng = _rng(spawn_seed, scenario_id, local_route)
    severity = _severity(rng)
    result: dict[str, object] = {
        "scenario_seed": scenario_seed(spawn_seed, scenario_id, local_route),
        "severity_bucket": severity,
        "ego_initial_speed_km_h": float(ego_initial_speed_km_h),
    }

    if scenario_id == "S5_hard_brake_lead":
        # Four admissible patterns keep each side and relation non-degenerate
        # without defining discrete lane-availability variants.
        patterns = (
            ("ahead", "behind"),
            ("behind", "ahead"),
            ("ahead", "ahead"),
            ("behind", "behind"),
        )
        left_relation, right_relation = patterns[int(rng.randint(0, len(patterns)))]

        def offset(relation: str) -> float:
            bounds = (-20.0, -10.0) if relation == "behind" else (10.0, 22.0)
            return _uniform(rng, bounds)

        lead_delta = _uniform(rng, (-3.0, 3.0))
        result.update(
            brake_trigger_time_s=_uniform(rng, (2.5, 4.0)),
            lead_trigger_bumper_gap_m=_uniform(rng, (9.0, 15.0)),
            lead_speed_delta_from_ego_km_h=lead_delta,
            lead_approach_speed_km_h=float(
                np.clip(float(ego_initial_speed_km_h) + lead_delta, 0.0, 40.0)
            ),
            lead_brake_deceleration_mps2=_uniform(rng, (4.5, 7.0)),
            lead_target_speed_km_h=_uniform(rng, (0.0, 3.0)),
            left_relation=left_relation,
            right_relation=right_relation,
            left_offset_m=offset(left_relation),
            right_offset_m=offset(right_relation),
            left_speed_km_h=_uniform(rng, (18.0, 32.0)),
            right_speed_km_h=_uniform(rng, (18.0, 32.0)),
        )
    elif scenario_id == "S6_background_merge_in":
        # A separate digest bit avoids the all-odd fixed evaluation seeds
        # collapsing onto one target gap.
        digest = hashlib.sha256(
            f"{spawn_seed}\0{scenario_id}\0{local_route}\0target_gap".encode("utf-8")
        ).digest()
        target_gap = (
            "agent0-agent1" if int.from_bytes(digest[:4], "little") % 2 == 0
            else "agent1-agent2"
        )
        trigger_s = _uniform(rng, (0.0, 1.0))
        result.update(
            target_gap_id=target_gap,
            merge_actor_speed_km_h=_uniform(rng, (18.0, 27.0)),
            conflict_arrival_time_delta_s=_uniform(rng, (-0.5, 0.5)),
            predicted_conflict_ttc_s=_uniform(rng, (1.5, 3.5)),
            target_front_bumper_gap_m=_uniform(rng, (6.0, 10.0)),
            target_rear_bumper_gap_m=_uniform(rng, (6.0, 10.0)),
            trigger_time_s=trigger_s,
            merge_activation_step=int(round(trigger_s / max(decision_dt_s, 1e-6))),
            ego_distance_to_merge_point_m=_uniform(rng, (25.0, 50.0)),
        )
    elif scenario_id == "S7_ego_merge_from_ramp":
        actor_count = {"low": 4, "medium": 5, "high": 6}[severity]
        gap_bounds = {
            "low": (60.0, 70.0),
            "medium": (52.0, 60.0),
            "high": (45.0, 52.0),
        }[severity]
        result.update(
            actor_count=actor_count,
            optional_adjacent_actor_count=actor_count - 4,
            usable_mainline_gap_m=_uniform(rng, gap_bounds),
            mainline_headway_s=_uniform(rng, (1.0, 2.0)),
            ego_distance_to_merge_point_m=_uniform(rng, (25.0, 55.0)),
            critical_front_arrival_delta_s=_uniform(rng, (-1.0, 0.0)),
            critical_rear_arrival_delta_s=_uniform(rng, (0.0, 1.0)),
            initial_platoon_bumper_gap_m=_uniform(rng, (12.0, 20.0)),
            mainline_actor_speeds_km_h=[
                _uniform(rng, (20.0, 31.0)) for _ in range(actor_count)
            ],
            expected_behavior=(
                "pass_first" if severity == "low" else "yield_then_merge"
            ),
        )
    elif scenario_id == "S8_ego_exit_to_ramp":
        result.update(
            ego_distance_to_diverge_m=_uniform(rng, (50.0, 90.0)),
            mandatory_lane_change_remaining_distance_m=_uniform(rng, (30.0, 60.0)),
            exit_lane_front_actor_speed_km_h=_uniform(rng, (16.0, 23.0)),
            exit_lane_rear_actor_speed_km_h=_uniform(rng, (20.0, 28.0)),
            usable_exit_lane_gap_m=_uniform(rng, (45.0, 75.0)),
        )
    elif scenario_id == "S9_narrow_channel_negotiation":
        # Solve the blocker gap from a jointly feasible TTC/relative-speed
        # tuple.  Independent draws can request a negative blocker speed and
        # silently violate the 2--4 s TTC contract after clipping.
        maximum_blocker_speed = min(
            8.0, max(float(ego_initial_speed_km_h) - 13.5, 0.0)
        )
        blocker_speed = _uniform(rng, (0.0, maximum_blocker_speed))
        closing_speed_mps = max(
            (float(ego_initial_speed_km_h) - blocker_speed) / 3.6, 1e-3
        )
        minimum_ttc = max(2.0, 15.0 / closing_speed_mps)
        desired_ttc = _uniform(rng, (minimum_ttc, 4.0))
        blocker_gap = float(
            np.clip(closing_speed_mps * desired_ttc, 15.0, 28.0)
        )
        actual_ttc = blocker_gap / closing_speed_mps
        result.update(
            bypass_side="left",
            source_lane_id=1,
            bypass_lane_id=0,
            blocker_speed_km_h=blocker_speed,
            agent0_to_blocker_bumper_gap_m=blocker_gap,
            bypass_constraint_actor_speed_km_h=_uniform(rng, (12.0, 22.0)),
            # The actor remains inside the specified range but is correlated
            # with the three-car sweep: placing it near +20 m and faster than
            # ego leaves the target-lane window open instead of overlapping
            # agent1/agent2 at negative offsets.
            bypass_constraint_actor_relative_offset_m=_uniform(rng, (18.0, 20.0)),
            usable_bypass_gap_m=_uniform(rng, (45.0, 70.0)),
            ego_distance_to_narrow_entry_m=_uniform(rng, (25.0, 50.0)),
            predicted_blocker_ttc_s=actual_ttc,
            latest_lane_change_completion_before_blocker_m=_uniform(rng, (8.0, 12.0)),
        )
    else:
        return {}

    return result


__all__ = [
    "SEVERITY_WEIGHTS",
    "resolve_s5_s9_parameters",
    "scenario_seed",
]
