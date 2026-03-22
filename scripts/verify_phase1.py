from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from envs.platoon_env import PlatoonEnv
from metadrive.policy.idm_policy import IDMPolicy


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase 1 platoon environment with IDM control.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", type=int, choices=(0, 1), default=1)
    parser.add_argument("--top-down", dest="top_down", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args(list(argv) if argv is not None else None)


def _build_idm_actions(env: PlatoonEnv, idm_policies: dict[str, object]) -> dict[str, np.ndarray]:
    actions = {}
    for agent_id in env._agent_ids:  # keep deterministic platoon ordering
        policy = idm_policies.get(agent_id)
        if policy is None:
            actions[agent_id] = np.zeros((2,), dtype=np.float32)
            continue
        try:
            actions[agent_id] = np.asarray(policy.act(), dtype=np.float32)
        except Exception:
            actions[agent_id] = np.zeros((2,), dtype=np.float32)
    return actions


def _build_idm_policy(vehicle, target_speed_km_h: float) -> IDMPolicy:
    policy = IDMPolicy(vehicle, random_seed=0)
    # Tune IDM for dense-platoon following on the default Phase 1 map with background traffic.
    policy.TIME_WANTED = 0.05
    policy.DISTANCE_WANTED = 0.5
    policy.NORMAL_SPEED = 18.0
    policy.target_speed = 18.0
    policy.ACC_FACTOR = 0.8
    policy.DEACC_FACTOR = -5.0
    return policy


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    env = PlatoonEnv(
        {
            "use_render": bool(args.render),
            "enable_idm_lane_change": False,
        }
    )

    try:
        idm_policies: dict[str, object] = {}
        completed_episodes = 0
        episode_steps = 0
        episode_formation_errors: list[float] = []
        episode_min_gap = float("inf")
        episode_collision = False
        episode_out_of_road = False
        episode_records: list[dict[str, float]] = []
        obs = env.reset()
        env.config["enable_idm_lane_change"] = False
        env.engine.global_config["enable_idm_lane_change"] = False
        for agent_id in env._agent_ids:
            vehicle = env.agents.get(agent_id)
            if vehicle is not None:
                idm_policies[agent_id] = _build_idm_policy(vehicle, env.platoon_config.target_speed_km_h)

        while completed_episodes < args.episodes:
            for agent_id in env._agent_ids:
                if agent_id not in idm_policies and env.agents.get(agent_id) is not None:
                    idm_policies[agent_id] = _build_idm_policy(env.agents[agent_id], env.platoon_config.target_speed_km_h)
            obs, reward, terminated, truncated, info = env.step(_build_idm_actions(env, idm_policies))
            episode_steps += 1
            if info:
                episode_formation_errors.extend(float(agent_info.get("formation_error", 0.0)) for agent_info in info.values())
                positive_gaps = [
                    float(agent_info.get("min_gap", float("inf")))
                    for agent_info in info.values()
                    if float(agent_info.get("min_gap", 0.0)) > 0.0
                ]
                if positive_gaps:
                    episode_min_gap = min(episode_min_gap, min(positive_gaps))
                episode_collision = episode_collision or any(
                    bool(
                        agent_info.get("crash", False)
                        or agent_info.get("crash_vehicle", False)
                        or agent_info.get("crash_object", False)
                        or agent_info.get("crash_building", False)
                    )
                    for agent_info in info.values()
                )
                episode_out_of_road = episode_out_of_road or any(
                    bool(agent_info.get("out_of_road", False)) for agent_info in info.values()
                )
            if args.top_down:
                env.render(mode="top_down")
            if any(agent_info.get("max_step", False) for agent_info in info.values()):
                for agent_id in env._agent_ids:
                    terminated[agent_id] = True
                terminated["__all__"] = True
                truncated["__all__"] = False
            if episode_steps >= args.max_steps:
                for agent_id in env._agent_ids:
                    terminated[agent_id] = True
                terminated["__all__"] = True
                truncated["__all__"] = False
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                completed_episodes += 1
                episode_info = info if info else env._last_info
                # In MetaDrive MA env, arrived agents are removed from env.agents before the episode fully terminates.
                # Treat "all agents completed without collision/out_of_road" as a successful platoon episode.
                episode_success = (len(env.agents) == 0) and (not episode_collision) and (not episode_out_of_road)
                episode_formation_error = float(np.mean(episode_formation_errors)) if episode_formation_errors else 0.0
                episode_min_gap_value = 0.0 if not np.isfinite(episode_min_gap) else float(episode_min_gap)
                episode_records.append(
                    {
                        "success": float(episode_success),
                        "collision": float(episode_collision),
                        "formation_error": episode_formation_error,
                        "min_gap": episode_min_gap_value,
                    }
                )
                print(
                    f"episode={completed_episodes} "
                    f"success={int(episode_success)} "
                    f"collision={int(episode_collision)} "
                    f"formation_error={episode_formation_error:.3f} "
                    f"min_gap={episode_min_gap_value:.3f}"
                )
                if completed_episodes >= args.episodes:
                    break
                obs = env.reset()
                episode_steps = 0
                episode_formation_errors = []
                episode_min_gap = float("inf")
                episode_collision = False
                episode_out_of_road = False
                env.config["enable_idm_lane_change"] = False
                env.engine.global_config["enable_idm_lane_change"] = False
                idm_policies = {}
                for agent_id in env._agent_ids:
                    vehicle = env.agents.get(agent_id)
                    if vehicle is not None:
                        idm_policies[agent_id] = _build_idm_policy(vehicle, env.platoon_config.target_speed_km_h)

        if episode_records:
            summary = {
                "success_rate": float(np.mean([record["success"] for record in episode_records])),
                "collision_rate": float(np.mean([record["collision"] for record in episode_records])),
                "formation_error": float(np.mean([record["formation_error"] for record in episode_records])),
                "recovery_time": float(env.get_platoon_metrics().get("recovery_time", 0.0)),
                "min_inter_vehicle_gap": float(min(record["min_gap"] for record in episode_records)),
            }
        else:
            summary = env.get_platoon_metrics()
        print(
            "summary: "
            f"success_rate={summary['success_rate']:.3f} "
            f"collision_rate={summary['collision_rate']:.3f} "
            f"formation_error={summary['formation_error']:.3f} "
            f"recovery_time={summary['recovery_time']:.3f} "
            f"min_inter_vehicle_gap={summary['min_inter_vehicle_gap']:.3f}"
        )
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
