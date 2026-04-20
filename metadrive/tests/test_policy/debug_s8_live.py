#!/usr/bin/env python
"""Live simulation test: run S8 scenario and track mode_context at each step."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np

HYBRID_MAP = [
    {"block_id": "s0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 300.0},
    {"block_id": "c0", "id": "C", "parent_block_id": "s0", "parent_socket_index": 0, "length": 100.0, "radius": 40.0, "angle": 150.0, "dir": 1},
    {"block_id": "c1", "id": "C", "parent_block_id": "c0", "parent_socket_index": 0, "length": 100.0, "radius": 40.0, "angle": 180.0, "dir": 0},
    {"block_id": "g0", "id": "G", "parent_block_id": "c1", "parent_socket_index": 0, "length": 100.0, "extension_length": 30.0},
    {"block_id": "s_main0", "id": "S", "parent_block_id": "g0", "parent_socket_index": 0, "length": 200.0},
    {"block_id": "s_ramp0", "id": "s", "parent_block_id": "g0", "parent_socket_index": 1, "length": 60.0},
    {"block_id": "c0_ramp0", "id": "c", "parent_block_id": "s_ramp0", "parent_socket_index": 0, "length": 50.0, "radius": 50.0, "angle": 50.0, "dir": 1},
]

ROUTE_BLOCK_IDS = ("g0", "s_ramp0", "c0_ramp0")


def main():
    from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv
    from metadrive.policy.diffusion_policy.mode_context import build_mode_context_from_vehicle
    from metadrive.policy.diffusion_policy.mode_trajectory_generator import ModeTrajectoryGenerator
    from metadrive.policy.diffusion_policy.mode_definitions import MODE_SLOTS

    env_config = {
        "use_render": False,
        "num_scenarios": 1,
        "use_hybrid_map": True,
        "hybrid_map_blocks_config": HYBRID_MAP,
        "map": 5,
        "start_seed": 59,
        "ego_main_route_block_ids": ROUTE_BLOCK_IDS,
        "traffic_density": 0.0,
        "num_agents": 1,
        "crash_done": False,
        "out_of_road_done": False,
        "image_observation": False,
    }

    env = BaseMultiEnv(env_config)
    try:
        obs, info = env.reset()
        agent_id = list(obs.keys())[0]
        current_map = env.current_map

        # Run N steps with IDM policy
        max_steps = 300
        for step in range(max_steps):
            vehicle = env.agents.get(agent_id)
            if vehicle is None:
                print(f"Step {step}: vehicle removed")
                break

            lane = getattr(vehicle, 'lane', None)
            lane_idx = getattr(lane, 'index', None) if lane else None
            pos = vehicle.position
            heading = getattr(vehicle, 'heading_theta', 0.0)

            try:
                ctx = build_mode_context_from_vehicle(
                    vehicle, current_map=current_map,
                    ego_main_route_block_ids=list(ROUTE_BLOCK_IDS),
                )
            except Exception as e:
                print(f"Step {step}: ERROR {e}")
                action = {agent_id: np.array([0.0, 0.0])}
                obs, reward, terminated, truncated, info = env.step(action)
                continue

            # Only print if there's a branch detected
            has_br = ctx.has_left_branch or ctx.has_right_branch
            # Also print on specific lane segments for visibility
            lane_str = f"{lane_idx[0]}->{lane_idx[1]}:{lane_idx[2]}" if lane_idx else "?"

            # Key lanes to watch
            interesting = False
            if lane_idx is not None:
                s = str(lane_idx[0]) + str(lane_idx[1])
                if any(kw in s for kw in ['4G1_0_', '4G1_4_', '6s0_0_', '7c0_0_']):
                    interesting = True

            if has_br or interesting:
                print(f"Step {step:3d} lane={lane_str} pos=[{pos[0]:.1f},{pos[1]:.1f}] "
                      f"left_br={ctx.has_left_branch} right_br={ctx.has_right_branch}")
                if ctx.left_branch_polyline is not None:
                    lbr = ctx.left_branch_polyline
                    print(f"         left_br: [{lbr[0,0]:.1f},{lbr[0,1]:.1f}] -> [{lbr[-1,0]:.1f},{lbr[-1,1]:.1f}]")
                if ctx.right_branch_polyline is not None:
                    rbr = ctx.right_branch_polyline
                    print(f"         right_br: [{rbr[0,0]:.1f},{rbr[0,1]:.1f}] -> [{rbr[-1,0]:.1f},{rbr[-1,1]:.1f}]")

            # Use IDM policy if available, else random gentle steer
            action = {agent_id: np.array([0.0, 0.3])}  # gentle acceleration
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception:
                break

            done_val = terminated.get(agent_id, False) or truncated.get(agent_id, False)
            if done_val:
                print(f"Step {step}: done")
                break
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
