#!/usr/bin/env python
"""Live simulation debug: S5 hard_brake_lead (KL collision) and S6 merge_in (divergence filter)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
from metadrive.exp_dataset.expert_idm_policy import ExpertIDMPolicy
from metadrive.exp_dataset.scenario_definitions import get_scenario_definition, SCENARIO_BY_ID
from metadrive.exp_dataset.route_definitions import ROUTE_BY_NAME, get_route_blocks


def run_scenario(scenario_id, route_name, max_steps=400):
    from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv
    from metadrive.policy.diffusion_policy.mode_context import build_mode_context_from_vehicle
    from metadrive.policy.diffusion_policy.mode_trajectory_generator import ModeTrajectoryGenerator

    sdef = get_scenario_definition(scenario_id)
    route_block_ids = tuple(get_route_blocks(route_name))

    env_config = {
        "use_render": False,
        "num_scenarios": 1,
        "use_hybrid_map": True,
        "map": 5,
        "start_seed": 59,
        "ego_main_route_block_ids": route_block_ids,
        "traffic_density": 0.15,
        "num_agents": 1,
        "crash_done": False,
        "out_of_road_done": False,
        "image_observation": False,
    }
    if sdef.ego_spawn_lane_preference and sdef.ego_spawn_lane_preference == "rightmost":
        env_config["random_spawn_lane_index"] = False

    env = BaseMultiEnv(env_config)
    gen = ModeTrajectoryGenerator()

    try:
        obs, info = env.reset()
        agent_id = list(obs.keys())[0]
        vehicle = env.agents[agent_id]

        idm_policy = ExpertIDMPolicy(control_object=vehicle, random_seed=59)

        print(f"\n{'='*60}")
        print(f"  {scenario_id} / {route_name}")
        print(f"{'='*60}")

        current_map = env.current_map

        for step in range(max_steps):

            lane = vehicle.lane
            lane_idx = getattr(lane, 'index', None)
            pos = vehicle.position

            if step % 10 == 0 or step < 5:
                try:
                    ctx = build_mode_context_from_vehicle(
                        vehicle, current_map=current_map,
                        ego_main_route_block_ids=route_block_ids,
                    )
                    traj_out = gen.generate(ctx)
                except Exception as e:
                    print(f"[step={step}] ERROR: {e}")
                    continue

                # KL validity
                kl_high_valid = traj_out.mode_valid_mask[0]
                kl_med_valid = traj_out.mode_valid_mask[1]
                kl_low_valid = traj_out.mode_valid_mask[2]
                # LC validity
                lc_l_valid = any(traj_out.mode_valid_mask[3:6])
                lc_r_valid = any(traj_out.mode_valid_mask[6:9])
                estop_valid = traj_out.mode_valid_mask[9]

                # Front distance from ctx
                front_d = ctx.front_object_distance

                print(f"[step={step:3d}] lane={lane_idx} speed={ctx.ego_speed_mps:.1f}m/s "
                      f"front={front_d:.1f}m "
                      f"KL=[H:{int(kl_high_valid)} M:{int(kl_med_valid)} L:{int(kl_low_valid)}] "
                      f"LC_L={int(lc_l_valid)} LC_R={int(lc_r_valid)} ESTOP={int(estop_valid)} "
                      f"left_br={ctx.has_left_branch} right_br={ctx.has_right_branch}")

                # Issue 2: when front_d is close (<20m) and ego is fast, KL_HIGH should detect collision
                if 0 < front_d < 25.0 and ctx.ego_speed_mps > 5.0:
                    if kl_high_valid:
                        print(f"  *** WARNING: KL_HIGH valid despite close front={front_d:.1f}m at speed={ctx.ego_speed_mps:.1f}")
                    else:
                        print(f"  *** OK: KL_HIGH correctly invalid with front={front_d:.1f}m at speed={ctx.ego_speed_mps:.1f}")

                # Issue 3: S6 divergence filter - right LC should be invalid for merge-out ramps
                if scenario_id == "S6_background_merge_in" and ctx.has_right_branch:
                    # Convert branch midpoint to world to identify the lane
                    rbp = ctx.right_branch_polyline
                    heading = getattr(vehicle, 'heading_theta', 0.0)
                    cos_h, sin_h = np.cos(heading), np.sin(heading)
                    mid_l = rbp[len(rbp)//2]
                    mid_wx = pos[0] + cos_h * mid_l[0] - sin_h * mid_l[1]
                    mid_wy = pos[1] + sin_h * mid_l[0] + cos_h * mid_l[1]
                    rn = current_map.road_network
                    best_lid, best_d = "?", float('inf')
                    for sn in rn.graph:
                        for en in rn.graph[sn]:
                            for rl in rn.graph[sn][en]:
                                try:
                                    lo, la = rl.local_coordinates(np.array([mid_wx, mid_wy]))
                                    if 0 <= lo <= rl.length and abs(la) < best_d:
                                        best_d = abs(la)
                                        best_lid = f"{sn}->{en} idx={getattr(rl,'index',None)}"
                                except:
                                    pass
                    # Also print polyline lat profile
                    lats = rbp[:, 1]
                    print(f"  *** S6 RIGHT BRANCH: nearest={best_lid} lat_d={best_d:.2f}")
                    print(f"      polyline lat: [{lats[0]:.2f}...{lats[len(lats)//2]:.2f}...{lats[-1]:.2f}]")

            action = idm_policy.act()
            obs, reward, term, trunc, info = env.step({agent_id: action})

            if any(term.values()) or any(trunc.values()):
                print(f"[step={step}] Episode ended.")
                break

    finally:
        try:
            env.close()
        except:
            pass


if __name__ == "__main__":
    # Issue 2: S5 hard_brake_lead - KL collision with front vehicle
    run_scenario("S5_hard_brake_lead", "R3_mainline_straight", max_steps=300)

    # Issue 3: S6 background_merge_in - divergence filter
    run_scenario("S6_background_merge_in", "R6_mainline_merge_approach", max_steps=300)
