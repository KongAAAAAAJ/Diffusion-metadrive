from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO/IDM expert in DatasetCollectEnv for quick inspection.")
    parser.add_argument("--expert-type", type=str, default="ppo", choices=("ppo", "idm"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", type=int, choices=(0, 1), default=1)
    parser.add_argument("--print-obs-summary", type=int, choices=(0, 1), default=1)
    parser.add_argument("--print-trajectory-debug", type=int, choices=(0, 1), default=0)
    return parser.parse_args(list(argv) if argv is not None else None)


def _print_observation_summary(obs_dict) -> None:
    print("Agents:", list(obs_dict.keys()))
    obs_sample = list(obs_dict.values())[0]
    if isinstance(obs_sample, dict):
        print("Obs keys:", list(obs_sample.keys()))
        for key, value in obs_sample.items():
            print(f"  '{key}' shape: {getattr(value, 'shape', None)}")
    else:
        print("Obs shape:", getattr(obs_sample, "shape", None))


def _build_debug_context(vehicle, collect_expert_module):
    reference_lane = collect_expert_module._select_reference_lane(vehicle)
    reference_lane_index = collect_expert_module._safe_lane_ordinal(getattr(reference_lane, "index", None))
    front_distance, front_speed = collect_expert_module._extract_front_object_state(vehicle, reference_lane)
    current_ref_lanes = getattr(vehicle.navigation, "current_ref_lanes", None)
    next_ref_lanes = getattr(vehicle.navigation, "next_ref_lanes", None)
    return collect_expert_module.TrajectoryCorrectionContext(
        current_lane_index=reference_lane_index,
        future_lane_indices=(reference_lane_index,) * 8 if reference_lane_index is not None else (None,) * 8,
        current_ref_lane_count=max(len(current_ref_lanes) if current_ref_lanes else 1, 1),
        next_ref_lane_count=(len(next_ref_lanes) if next_ref_lanes is not None else None),
        lane_width=max(float(getattr(reference_lane, "width", getattr(vehicle.lane, "width", 4.0))), 1.0),
        front_object_distance=front_distance,
        ego_speed_km_h=float(vehicle.speed_km_h),
        front_object_speed_km_h=front_speed,
    )


def _print_trajectory_debug(vehicle, collect_expert_module) -> None:
    reference_lane = collect_expert_module._select_reference_lane(vehicle)
    current_pose = collect_expert_module.pose_to_array(vehicle)
    future_offsets = tuple(5 * (idx + 1) for idx in range(8))
    raw_trajectory = np.stack(
        [
            collect_expert_module.world_future_to_local(
                current_pose,
                collect_expert_module._pose_world_to_array(
                    reference_lane.position(min(reference_lane.length, offset * 1.5), 0.0),
                    reference_lane.heading_theta_at(min(reference_lane.length, offset * 1.5)),
                ),
            )
            for offset in future_offsets
        ],
        axis=0,
    ).astype(np.float32)
    context = _build_debug_context(vehicle, collect_expert_module)
    mode = collect_expert_module.classify_trajectory_mode(raw_trajectory, context)
    corrected, _ = collect_expert_module.correct_trajectory_geometry(
        raw_trajectory,
        mode,
        reference_trajectory=raw_trajectory,
        attraction_strength=collect_expert_module.ExpertCollectorConfig.centerline_attraction_strength,
        smoothing_strength=collect_expert_module.ExpertCollectorConfig.smoothing_strength,
    )
    print(
        "[trajectory_debug] "
        f"mode={mode.name.lower()} "
        f"raw_final_y={float(raw_trajectory[-1, 1]):+.3f} "
        f"corrected_final_y={float(corrected[-1, 1]):+.3f}"
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    from metadrive.envs.diffusion_envs.base_multi_env import DatasetCollectEnv
    from metadrive.examples.ppo_expert import expert as ppo_expert
    from metadrive.exp_dataset import collect_expert as collect_expert_module
    from metadrive.policy.idm_policy import IDMPolicy

    env = DatasetCollectEnv()
    obs_dict, _ = env.reset()
    if bool(args.print_obs_summary):
        _print_observation_summary(obs_dict)

    ep_reward = 0.0
    ep_count = 0
    step = 0
    idm_policy = None

    while ep_count < args.episodes:
        actions = {}
        for agent_id, vehicle in env.agents.items():
            if args.expert_type == "ppo":
                actions[agent_id] = ppo_expert(vehicle, deterministic=True)
                if bool(args.print_trajectory_debug):
                    _print_trajectory_debug(vehicle, collect_expert_module)
            else:
                if idm_policy is None:
                    idm_policy = IDMPolicy(vehicle, random_seed=0)
                actions[agent_id] = idm_policy.act()

        obs_dict, reward, terminated, truncated, info = env.step(actions)
        if bool(args.render):
            env.render(mode="top_down", text={"step": step, "ep": ep_count, "expert": args.expert_type})
        ep_reward += sum(reward.values())
        step += 1

        if terminated["__all__"] or truncated["__all__"]:
            ep_count += 1
            agent_info = list(info.values())[0] if info else {}
            print(
                f"[Episode {ep_count}] steps={step}, "
                f"reward={ep_reward:.2f}, "
                f"arrive_dest={agent_info.get('arrive_dest', False)}, "
                f"crash={agent_info.get('crash', False)}, "
                f"out_of_road={agent_info.get('out_of_road', False)}"
            )
            ep_reward = 0.0
            step = 0
            if ep_count >= args.episodes:
                break
            obs_dict, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
