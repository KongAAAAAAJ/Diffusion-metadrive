from __future__ import annotations

from typing import Iterable, Mapping


def _as_float(mapping: Mapping, key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(mapping: Mapping, key: str, default: bool = False) -> bool:
    return bool(mapping.get(key, default))


def _has_terminal_collision(info: Mapping) -> bool:
    collision_keys = ("crash_vehicle", "crash_object", "crash_building", "crash_human")
    known_crash_keys = collision_keys + ("crash_sidewalk",)
    if any(key in info for key in known_crash_keys):
        return any(_as_bool(info, key, False) for key in collision_keys)
    return _as_bool(info, "crash", False)


def _clip_reward(value: float, config: Mapping, key: str = "reward_clip") -> float:
    clip = _as_float(config, key, 0.0)
    if clip <= 0.0:
        return float(value)
    return float(max(-clip, min(clip, value)))


def compute_step_reward(info: dict, config: dict) -> float:
    info = info or {}
    config = config or {}

    delta_s_max = max(_as_float(config, "delta_s_max", 5.0), 1e-6)
    d_norm = max(_as_float(config, "d_norm", 10.0), 1e-6)
    d_safe = max(_as_float(config, "d_safe", 8.0), 1e-6)

    w_progress = _as_float(config, "w_progress", _as_float(config, "w_prog", 1.0))
    w_formation = _as_float(config, "w_formation", _as_float(config, "w_form", 0.5))
    w_safety = _as_float(config, "w_safety", _as_float(config, "w_safe", 0.3))
    w_collision = _as_float(config, "w_collision", _as_float(config, "w_coll", 10.0))
    w_road = _as_float(config, "w_road", 5.0)
    w_comfort = _as_float(config, "w_comfort", _as_float(config, "w_comf", 0.1))
    arrive_bonus = _as_float(config, "arrive_bonus", 20.0)
    recovery_bonus = _as_float(config, "recovery_bonus", 5.0)

    progress = _as_float(info, "progress", 0.0)
    formation_error = abs(_as_float(info, "formation_error", 0.0))
    min_gap = _as_float(info, "min_gap", d_safe)
    jerk = abs(_as_float(info, "jerk", 0.0))
    delta_steering = abs(_as_float(info, "delta_steering", 0.0))

    r_progress = progress / delta_s_max
    r_formation = -(formation_error / d_norm)
    r_safety = -max(0.0, d_safe - min_gap) / d_safe
    collision = _has_terminal_collision(info)
    r_collision = -w_collision if collision else 0.0
    r_road = -w_road if _as_bool(info, "out_of_road", False) else 0.0
    r_comfort = -(0.5 * jerk + 0.5 * delta_steering)

    reward = (
        w_progress * r_progress    
        + w_formation * r_formation 
        + w_safety * r_safety     
        + r_collision       
        + r_road      
        + w_comfort * r_comfort   
    )

    if _as_bool(info, "all_arrive_dest", False):
        reward += arrive_bonus
    if _as_bool(info, "formation_recovered", False):
        reward += recovery_bonus

    if collision:
        reward = min(reward, -w_collision)
    if _as_bool(info, "out_of_road", False):
        reward = min(reward, -w_road)

    return _clip_reward(float(reward), config)


def compute_trajectory_reward(step_infos: Iterable[dict], config: dict) -> float:
    return float(sum(compute_step_reward(info, config) for info in (step_infos or [])))



def compute_team_reward(per_agent_step_infos: Mapping[str, Iterable[dict]], config: dict) -> float:
    config = config or {}
    agent_ids = list(per_agent_step_infos.keys())
    if not agent_ids:
        return 0.0

    w_formation = _as_float(config, "w_team_formation", 0.5)
    w_safety = _as_float(config, "w_team_safety", 1.0)
    w_efficiency = _as_float(config, "w_team_efficiency", 0.3)
    collision_penalty = _as_float(config, "w_team_collision", 10.0)
    d_norm = max(_as_float(config, "d_norm", 10.0), 1e-6)
    delta_s_max = max(_as_float(config, "delta_s_max", 5.0), 1e-6)

    total_formation_error = 0.0
    total_progress = 0.0
    count = 0
    crash_count = 0
    num_agents = len(agent_ids)
    for step_infos in per_agent_step_infos.values():
        agent_crashed = False
        for info in step_infos:
            total_formation_error += abs(_as_float(info, "formation_error", 0.0))
            total_progress += max(_as_float(info, "progress", 0.0), 0.0)
            agent_crashed = agent_crashed or _has_terminal_collision(info)
            count += 1
        if agent_crashed:
            crash_count += 1

    avg_formation_error = total_formation_error / max(count, 1)
    avg_progress = total_progress / max(len(agent_ids), 1)
    formation_term = -(avg_formation_error / d_norm)
    safety_term = -collision_penalty * (crash_count / max(num_agents, 1))
    efficiency_term = avg_progress / delta_s_max
    reward = w_formation * formation_term + w_safety * safety_term + w_efficiency * efficiency_term
    return _clip_reward(float(reward), config)
