from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO/IDM expert in DatasetCollectEnv for quick inspection.")
    parser.add_argument("--expert-type", type=str, default="idm", choices=("ppo", "idm"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", type=int, choices=(0, 1), default=1)
    parser.add_argument("--print-obs-summary", type=int, choices=(0, 1), default=1)
    parser.add_argument("--print-trajectory-debug", type=int, choices=(0, 1), default=0)
    parser.add_argument("--print-idm-debug", type=int, choices=(0, 1), default=0)
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



def _print_idm_debug(step: int, vehicle, expert_policy) -> None:
    base_policy = expert_policy._ensure_policy() if hasattr(expert_policy, "_ensure_policy") else expert_policy
    target_lane = getattr(base_policy, "routing_target_lane", None)
    current_lane_index = getattr(vehicle, "lane_index", None)
    target_lane_index = None if target_lane is None else getattr(target_lane, "index", None)
    current_ref_lanes = getattr(vehicle.navigation, "current_ref_lanes", None) or []
    next_ref_lanes = getattr(vehicle.navigation, "next_ref_lanes", None) or []

    target_long = None
    target_lat = None
    target_heading = None
    steer = None
    if target_lane is not None:
        target_long, target_lat = target_lane.local_coordinates(vehicle.position)
        target_heading = target_lane.heading_theta_at(target_long + 1)
    action = getattr(base_policy, "action_info", {}).get("action")
    if action is not None and len(action) >= 1:
        steer = float(action[0])

    print(
        "[idm_debug] "
        f"step={step} "
        f"veh_lane={current_lane_index} "
        f"target_lane={target_lane_index} "
        f"current_ref={[lane.index for lane in current_ref_lanes]} "
        f"next_ref={[lane.index for lane in next_ref_lanes]} "
        f"target_long={None if target_long is None else round(float(target_long), 3)} "
        f"target_lat={None if target_lat is None else round(float(target_lat), 3)} "
        f"target_heading={None if target_heading is None else round(float(target_heading), 3)} "
        f"speed_kmh={float(vehicle.speed_km_h):.3f} "
        f"steer={None if steer is None else round(steer, 4)} "
        f"yellow={bool(getattr(vehicle, 'on_yellow_continuous_line', False))} "
        f"on_lane={bool(getattr(vehicle, 'on_lane', True))}"
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    from metadrive.envs.diffusion_envs.base_multi_env import DatasetCollectEnv
    from metadrive.examples.ppo_expert import expert as ppo_expert
    from metadrive.exp_dataset import collect_expert as collect_expert_module
    from metadrive.exp_dataset.expert_idm_policy import ExpertIDMPolicy

    # !暂时降低RGB相机分辨率以提升渲染效率
    from metadrive.component.sensors.rgb_camera import RGBCamera
    env = DatasetCollectEnv(config={
        "traffic_density": 0.1,
        "use_render": bool(args.render),
        "sensors": dict(
            rgb_camera=(RGBCamera, 40, 25),  # 320x180 RGB camera
        ),
    })
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
                    idm_policy = ExpertIDMPolicy(vehicle, random_seed=0)
                actions[agent_id] = idm_policy.act()
                if bool(args.print_idm_debug):
                    _print_idm_debug(step, vehicle, idm_policy)

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
            idm_policy = None
            if ep_count >= args.episodes:
                break
            obs_dict, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
