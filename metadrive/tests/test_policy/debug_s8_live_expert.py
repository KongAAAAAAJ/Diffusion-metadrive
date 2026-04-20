#!/usr/bin/env python
"""Live simulation debug: S8 exit-to-ramp with expert IDM policy.

Drives the vehicle through the complete scenario and logs mode context
at every step, focusing on deacc_lane and ramp positions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
from metadrive.exp_dataset.expert_idm_policy import ExpertIDMConfig, ExpertIDMPolicy

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


def _debug_branch_source(vehicle, current_map, env, route_block_ids):
    """Replay branch resolution logic with verbose output to identify which road is selected."""
    from metadrive.policy.diffusion_policy.mode_context import (
        _get_route_blocks, _get_all_positive_roads_from_block,
        _sample_lane_polyline, _sample_lane_polyline_from_vehicle_position,
        _world_to_local_xy, _build_reachable_nodes, _build_diverged_branch_starts,
        _is_grand_predecessor,
    )

    road_network = current_map.road_network
    graph = road_network.graph
    current_lane = vehicle.lane
    cur_lane_idx = getattr(current_lane, 'index', None)
    cur_start = str(cur_lane_idx[0]) if cur_lane_idx else None
    cur_end = str(cur_lane_idx[1]) if cur_lane_idx else None

    current_pose = np.asarray(
        [float(vehicle.position[0]), float(vehicle.position[1]), float(vehicle.heading_theta)],
        dtype=np.float32,
    )

    forward_reachable = _build_reachable_nodes(graph, cur_end, max_hops=6) if cur_end else set()
    ancestor_nodes, diverged_starts = _build_diverged_branch_starts(graph, cur_start, max_ancestor_hops=6) if cur_start else (set(), set())

    print(f"\n    DEBUG BRANCH SOURCE: cur={cur_start}->{cur_end}")
    print(f"    forward_reachable={sorted(forward_reachable)}")
    print(f"    ancestor_nodes={sorted(ancestor_nodes)}")
    print(f"    diverged_starts={sorted(diverged_starts)}")

    route_blocks = _get_route_blocks(current_map, route_block_ids)
    for block in route_blocks:
        block_roads = _get_all_positive_roads_from_block(block)
        bid = getattr(block, 'graph_block_id', '?')
        for road in block_roads:
            road_start = str(getattr(road, 'start_node', ''))
            road_end = str(getattr(road, 'end_node', ''))

            # Track filter reasons
            skip_reason = None
            if cur_start and road_start == cur_start and road_end == cur_end:
                skip_reason = "same_road"
            elif cur_end and road_start == cur_end:
                skip_reason = "fwd_continuation"
            elif cur_start and road_end == cur_start:
                skip_reason = "predecessor"
            elif cur_start and road_start == cur_start:
                skip_reason = "sibling"
            elif road_start in forward_reachable:
                skip_reason = "fwd_reachable"
            elif cur_start and cur_end and _is_grand_predecessor(graph, road_start, road_end, cur_start, cur_end):
                skip_reason = "grand_predecessor"
            elif road_start in diverged_starts:
                skip_reason = "diverged_start"
            elif road_start in ancestor_nodes and road_end not in ancestor_nodes and road_end != cur_start and road_end != cur_end:
                skip_reason = "ancestor_to_non_ancestor"

            if skip_reason:
                print(f"    [{bid}] {road_start}->{road_end}: SKIPPED ({skip_reason})")
            else:
                # This road passes filters - check lateral
                try:
                    lanes = graph[road_start][road_end]
                    best_lat = float('inf')
                    for lane in lanes:
                        poly_w = _sample_lane_polyline(lane, num_points=5, longitudinal_step_m=2.0)
                        poly_l = _world_to_local_xy(current_pose, poly_w)
                        lat = float(np.mean(poly_l[:, 1]))
                        if abs(lat) < abs(best_lat):
                            best_lat = lat
                    # Full polyline
                    best_lane = lanes[0]  # simplified
                    for lane in lanes:
                        poly_w = _sample_lane_polyline(lane, num_points=5, longitudinal_step_m=2.0)
                        poly_l = _world_to_local_xy(current_pose, poly_w)
                        lt = float(np.mean(poly_l[:, 1]))
                        if abs(lt) == abs(best_lat):
                            best_lane = lane
                    poly_w = _sample_lane_polyline_from_vehicle_position(
                        best_lane, current_pose[:2], num_points=30, longitudinal_step_m=2.0)
                    poly_l = _world_to_local_xy(current_pose, poly_w)
                    lat_hint = float(np.mean(poly_l[:, 1]))
                    print(f"    [{bid}] {road_start}->{road_end}: PASSES (best_lat={best_lat:.2f}, "
                          f"full_lat_hint={lat_hint:.2f}, first_x={poly_l[0,0]:.2f})")
                except Exception as e:
                    print(f"    [{bid}] {road_start}->{road_end}: PASSES but error: {e}")


# Key road segments we want to track
DEACC_LANE_KEY = ("4G1_0_", "4G0_0_")
RAMP_KEYS = [
    ("4G0_0_", "4G1_1_"),
    ("4G1_1_", "4G1_2_"),
    ("4G1_2_", "4G1_3_"),
    ("4G1_3_", "4G1_4_"),
    ("4G1_4_", "6s0_0_"),
    ("6s0_0_", "7c0_0_"),
    ("7c0_0_", "7c0_1_"),
]


def main():
    from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv
    from metadrive.policy.diffusion_policy.mode_context import build_mode_context_from_vehicle

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
        vehicle = env.agents[agent_id]

        # Build expert IDM policy
        idm_policy = ExpertIDMPolicy(
            control_object=vehicle,
            random_seed=59,
        )

        print("=" * 80)
        print("LIVE S8 SIMULATION - Expert IDM Policy")
        print("=" * 80)

        rn = env.current_map.road_network
        current_map = env.current_map

        # Print checkpoints
        nav = vehicle.navigation
        print(f"Route checkpoints: {nav.checkpoints}")
        print(f"Current ref lanes: {[getattr(l, 'index', None) for l in nav.current_ref_lanes]}")
        if nav.next_ref_lanes:
            print(f"Next ref lanes: {[getattr(l, 'index', None) for l in nav.next_ref_lanes]}")

        deacc_logged = False
        ramp_logged = set()

        for step in range(500):
            # Get current vehicle state
            lane = vehicle.lane
            lane_idx = getattr(lane, 'index', None)
            pos = vehicle.position
            heading = getattr(vehicle, 'heading_theta', 0.0)

            # Check if we're on an interesting lane
            is_deacc = lane_idx is not None and lane_idx[0] == DEACC_LANE_KEY[0] and lane_idx[1] == DEACC_LANE_KEY[1]
            is_ramp = lane_idx is not None and any(
                lane_idx[0] == k[0] and lane_idx[1] == k[1] for k in RAMP_KEYS
            )
            is_dec_road = lane_idx is not None and lane_idx[0] == '3C0_1_' and lane_idx[1] == '4G0_0_'

            # Log at interesting positions
            should_log = (
                (is_deacc and step % 5 == 0) or
                (is_ramp and step % 10 == 0) or
                (is_dec_road and step % 5 == 0) or
                step == 0
            )

            if should_log:
                # Get navigation state
                cur_ref_idxs = [getattr(l, 'index', None) for l in nav.current_ref_lanes]
                next_ref_idxs = [getattr(l, 'index', None) for l in (nav.next_ref_lanes or [])]

                try:
                    ctx = build_mode_context_from_vehicle(
                        vehicle, current_map=current_map,
                        ego_main_route_block_ids=ROUTE_BLOCK_IDS,
                    )
                except Exception as e:
                    print(f"[step={step}] ERROR building mode context: {e}")
                    continue

                # Only log interesting events for deacc/ramp/highway
                if is_deacc or is_ramp or is_dec_road:
                    long, lat = lane.local_coordinates(pos)
                    print(f"\n[step={step}] lane_idx={lane_idx} long={long:.1f} lat={lat:.1f} "
                          f"pos=[{pos[0]:.1f},{pos[1]:.1f}] heading={heading:.4f}")
                    print(f"  cur_ref={cur_ref_idxs}")
                    print(f"  next_ref={next_ref_idxs}")
                    print(f"  left_adj={ctx.has_left_adjacent} right_adj={ctx.has_right_adjacent}")
                    print(f"  left_br={ctx.has_left_branch} right_br={ctx.has_right_branch}")

                    for name, poly in [("left_br", ctx.left_branch_polyline),
                                       ("right_br", ctx.right_branch_polyline)]:
                        if poly is not None:
                            mean_lat = float(np.mean(poly[:, 1]))
                            print(f"  {name}: [{poly[0,0]:.2f},{poly[0,1]:.2f}] -> [{poly[-1,0]:.2f},{poly[-1,1]:.2f}] mean_lat={mean_lat:.2f}")
                            # Convert midpoint to world and find nearest road lane
                            cos_h = np.cos(heading)
                            sin_h = np.sin(heading)
                            mid_local = poly[len(poly)//2]
                            mid_wx = pos[0] + cos_h * mid_local[0] - sin_h * mid_local[1]
                            mid_wy = pos[1] + sin_h * mid_local[0] + cos_h * mid_local[1]
                            best_lid = "?"
                            best_d = float('inf')
                            for sn in rn.graph:
                                for en in rn.graph[sn]:
                                    for rl in rn.graph[sn][en]:
                                        try:
                                            lo, la = rl.local_coordinates(np.array([mid_wx, mid_wy]))
                                            if 0 <= lo <= rl.length and abs(la) < best_d:
                                                best_d = abs(la)
                                                best_lid = f"{getattr(rl,'index',None)}"
                                        except:
                                            pass
                            print(f"    -> world mid=[{mid_wx:.2f},{mid_wy:.2f}] nearest_lane={best_lid} lat={best_d:.2f}")

                    if is_deacc and (ctx.has_left_branch or ctx.has_right_branch):
                        print(f"  *** DEACC_LANE HAS BRANCH: left={ctx.has_left_branch} right={ctx.has_right_branch}")
                        if ctx.left_branch_polyline is not None:
                            ep = ctx.left_branch_polyline[-1]
                            print(f"      left_br endpoint lateral = {ep[1]:.2f} (expect ~3.50 for adjacent)")
                        # Generate trajectories and check LC_L_H world endpoint
                        from metadrive.policy.diffusion_policy.mode_trajectory_generator import ModeTrajectoryGenerator
                        gen = ModeTrajectoryGenerator()
                        traj_out = gen.generate(ctx)
                        # LC_L_H is slot index 3
                        if traj_out.mode_valid_mask[3]:
                            lc_l_h = traj_out.coarse_trajectories[3]  # (8,2) vehicle-local
                            cos_h = np.cos(heading)
                            sin_h = np.sin(heading)
                            ep_local = lc_l_h[-1]
                            ep_wx = pos[0] + cos_h * ep_local[0] - sin_h * ep_local[1]
                            ep_wy = pos[1] + sin_h * ep_local[0] + cos_h * ep_local[1]
                            # Find nearest lane for world endpoint
                            best_lid2 = "?"
                            best_d2 = float('inf')
                            for sn in rn.graph:
                                for en in rn.graph[sn]:
                                    for rl in rn.graph[sn][en]:
                                        try:
                                            lo2, la2 = rl.local_coordinates(np.array([ep_wx, ep_wy]))
                                            if 0 <= lo2 <= rl.length and abs(la2) < best_d2:
                                                best_d2 = abs(la2)
                                                best_lid2 = f"{getattr(rl,'index',None)}"
                                        except:
                                            pass
                            print(f"      LC_L_H traj endpoint: local=[{ep_local[0]:.2f},{ep_local[1]:.2f}] "
                                  f"world=[{ep_wx:.2f},{ep_wy:.2f}] nearest={best_lid2} lat={best_d2:.2f}")

                    if is_ramp and (ctx.has_left_branch or ctx.has_right_branch):
                        ramp_key = (lane_idx[0], lane_idx[1])
                        if ramp_key not in ramp_logged:
                            print(f"  *** RAMP HAS BRANCH (SHOULD NOT): left={ctx.has_left_branch} right={ctx.has_right_branch}")
                            ramp_logged.add(ramp_key)
                            _debug_branch_source(vehicle, current_map, env, ROUTE_BLOCK_IDS)

                    if is_dec_road:
                        print(f"  *** HIGHWAY: left_adj={ctx.has_left_adjacent} right_adj={ctx.has_right_adjacent} "
                              f"left_br={ctx.has_left_branch} right_br={ctx.has_right_branch}")
                        if not ctx.has_right_branch and not ctx.has_right_adjacent:
                            print(f"      NO RIGHT DETECTED — deacc_lane should be right_branch")
                            _debug_branch_source(vehicle, current_map, env, ROUTE_BLOCK_IDS)

            # Step with IDM policy action
            action = idm_policy.act()
            obs, reward, term, trunc, info = env.step({agent_id: action})

            if any(term.values()) or any(trunc.values()):
                print(f"\n[step={step}] Episode ended. term={term} trunc={trunc}")
                break

        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()