#!/usr/bin/env python
"""Diagnostic script: dump G-block geometry for S8_ego_exit_to_ramp."""
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


def dump_road_network(env):
    current_map = env.current_map
    road_network = current_map.road_network
    graph = road_network.graph

    print("=" * 80)
    print("ROAD NETWORK GRAPH")
    print("=" * 80)
    for start_node in sorted(graph.keys(), key=str):
        for end_node in sorted(graph[start_node].keys(), key=str):
            lanes = graph[start_node][end_node]
            print(f"  {start_node} -> {end_node}: {len(lanes)} lane(s)")
            for i, lane in enumerate(lanes):
                lane_start = np.asarray(lane.position(0.0, 0.0))
                lane_end = np.asarray(lane.position(float(lane.length), 0.0))
                width = float(getattr(lane, 'width', getattr(lane, '_width', -1)))
                try:
                    heading = float(lane.heading_theta_at(0.0))
                except Exception:
                    heading = float('nan')
                idx = getattr(lane, 'index', None)
                ltype = type(lane).__name__
                print(f"    [{i}] idx={idx} type={ltype} len={lane.length:.2f} w={width:.2f} "
                      f"start=[{lane_start[0]:.2f},{lane_start[1]:.2f}] "
                      f"end=[{lane_end[0]:.2f},{lane_end[1]:.2f}] "
                      f"heading={heading:.4f}")

    print("\n" + "=" * 80)
    print("BLOCKS")
    print("=" * 80)
    for block in current_map.blocks:
        bid = getattr(block, 'graph_block_id', '?')
        block_net = getattr(block, 'block_network', None)
        roads_strs = []
        if block_net is not None:
            pos_lanes = block_net.get_positive_lanes() or []
            seen = set()
            for lg in pos_lanes:
                if not lg:
                    continue
                idx = getattr(lg[0], 'index', None)
                if idx is not None:
                    key = (str(idx[0]), str(idx[1]))
                    if key not in seen:
                        seen.add(key)
                        roads_strs.append(f"{idx[0]}->{idx[1]}({len(lg)}lanes)")
        print(f"  {bid}: {type(block).__name__} roads={roads_strs}")


def dump_mode_context(vehicle, current_map, step_label=""):
    from models.diffusion.mode_context import build_mode_context_from_vehicle

    route_block_ids = ["g0", "s_ramp0", "c0_ramp0"]

    print(f"\n--- MODE CONTEXT {step_label} ---")
    lane = getattr(vehicle, 'lane', None)
    if lane is not None:
        idx = getattr(lane, 'index', None)
        pos = vehicle.position
        print(f"  lane.index={idx}  pos=[{pos[0]:.2f},{pos[1]:.2f}]")
    nav = getattr(vehicle, 'navigation', None)
    if nav is not None:
        cur_ref = getattr(nav, 'current_ref_lanes', None)
        next_ref = getattr(nav, 'next_ref_lanes', None)
        print(f"  cur_ref={[getattr(l,'index',None) for l in (cur_ref or [])]}")
        print(f"  next_ref={[getattr(l,'index',None) for l in (next_ref or [])]}")

    try:
        ctx = build_mode_context_from_vehicle(
            vehicle, current_map=current_map,
            ego_main_route_block_ids=route_block_ids,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print(f"  left_adj={ctx.has_left_adjacent} right_adj={ctx.has_right_adjacent} "
          f"left_br={ctx.has_left_branch} right_br={ctx.has_right_branch}")

    for name, poly in [("cur_lane", ctx.current_lane_polyline),
                       ("left_lane", ctx.left_lane_polyline),
                       ("right_lane", ctx.right_lane_polyline),
                       ("left_br", ctx.left_branch_polyline),
                       ("right_br", ctx.right_branch_polyline)]:
        if poly is not None:
            print(f"  {name}: [{poly[0,0]:.2f},{poly[0,1]:.2f}] -> [{poly[-1,0]:.2f},{poly[-1,1]:.2f}]")
        else:
            print(f"  {name}: None")

    from models.diffusion.mode_trajectory_generator import ModeTrajectoryGenerator
    from models.diffusion.mode_definitions import MODE_SLOTS
    gen = ModeTrajectoryGenerator()
    out = gen.generate(ctx)
    for slot in MODE_SLOTS:
        traj = out.coarse_trajectories[slot.index]
        ep = traj[-1]
        v = "V" if out.mode_valid_mask[slot.index] else "X"
        print(f"  [{v}] {slot.name:20s} ep=[{ep[0]:.2f},{ep[1]:.2f}]")


class FakeNavigation:
    def __init__(self, current_ref_lanes, next_ref_lanes=None):
        self.current_ref_lanes = current_ref_lanes
        self.next_ref_lanes = next_ref_lanes


class FakeVehicle:
    def __init__(self, position, heading_theta, lane, speed_km_h=50.0, lane_index=None):
        self.position = np.asarray(position, dtype=np.float32)
        self.heading_theta = float(heading_theta)
        self.lane = lane
        self.speed_km_h = float(speed_km_h)
        self.lane_index = lane_index or getattr(lane, 'index', None)
        self.navigation = FakeNavigation(
            current_ref_lanes=[lane],
            next_ref_lanes=None,
        )


def test_specific_positions(env):
    """Test mode_context at specific lane positions relevant to S8 problems."""
    from models.diffusion.mode_context import build_mode_context_from_vehicle

    current_map = env.current_map
    rn = current_map.road_network
    route_block_ids = ["g0", "s_ramp0", "c0_ramp0"]

    # Get key lanes
    dec_road_lanes = rn.graph['3C0_1_']['4G0_0_']  # 3 lanes
    deacc_lanes = rn.graph['4G1_0_']['4G0_0_']     # 1 lane
    extend_lanes = rn.graph['4G0_0_']['4G0_1_']    # 3 lanes
    bend1_lanes = rn.graph['4G0_0_']['4G1_1_']     # 1 lane
    connect_lanes = rn.graph['4G1_1_']['4G1_2_']   # 1 lane
    s_ramp0_lanes = rn.graph['4G1_4_']['6s0_0_']   # 1 lane
    c0_ramp0_curve_lanes = rn.graph['6s0_0_']['7c0_0_']  # 1 lane
    c0_ramp0_straight_lanes = rn.graph['7c0_0_']['7c0_1_']  # 1 lane

    tests = [
        # P1: Vehicle on dec_road lane 2 (rightmost), mid-way
        ("P1: dec_road lane_2, long=65", dec_road_lanes[2], 65.0),
        # P1: Vehicle on dec_road lane 2, near end
        ("P1: dec_road lane_2, long=120", dec_road_lanes[2], 120.0),
        # P2: Vehicle on deacc_lane, near start
        ("P2: deacc_lane, long=10", deacc_lanes[0], 10.0),
        # P2: Vehicle on deacc_lane, mid-way
        ("P2: deacc_lane, long=50", deacc_lanes[0], 50.0),
        # P2: Vehicle on deacc_lane, near end
        ("P2: deacc_lane, long=90", deacc_lanes[0], 90.0),
        # P3: Vehicle on bend_1 (just entered ramp)
        ("P3: bend_1, long=3", bend1_lanes[0], 3.0),
        # P3: Vehicle on connect (after bend_1)
        ("P3: connect, long=10", connect_lanes[0], 10.0),
        # P5: Vehicle on s_ramp0 (after g0 block)
        ("P5: s_ramp0, long=10", s_ramp0_lanes[0], 10.0),
        ("P5: s_ramp0, long=30", s_ramp0_lanes[0], 30.0),
        ("P5: s_ramp0, long=50", s_ramp0_lanes[0], 50.0),
        ("P5: s_ramp0, long=58", s_ramp0_lanes[0], 58.0),
        # P6: Vehicle on c0_ramp0 curve
        ("P6: c0_ramp0_curve, long=5", c0_ramp0_curve_lanes[0], 5.0),
        ("P6: c0_ramp0_curve, long=20", c0_ramp0_curve_lanes[0], 20.0),
        ("P6: c0_ramp0_curve, long=40", c0_ramp0_curve_lanes[0], 40.0),
        # P6: Vehicle on c0_ramp0 straight
        ("P6: c0_ramp0_straight, long=10", c0_ramp0_straight_lanes[0], 10.0),
    ]

    for label, lane, longitudinal in tests:
        pos = np.asarray(lane.position(longitudinal, 0.0), dtype=np.float32)
        try:
            heading = float(lane.heading_theta_at(longitudinal))
        except Exception:
            heading = float(lane.heading_theta_at(0.0))

        v = FakeVehicle(pos, heading, lane, speed_km_h=50.0)

        print(f"\n{'='*60}")
        print(f"TEST: {label}")
        print(f"  lane.index={getattr(lane, 'index', None)}")
        print(f"  pos=[{pos[0]:.2f},{pos[1]:.2f}] heading={heading:.4f}")

        try:
            ctx = build_mode_context_from_vehicle(
                v, current_map=current_map,
                ego_main_route_block_ids=route_block_ids,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        print(f"  left_adj={ctx.has_left_adjacent} right_adj={ctx.has_right_adjacent}")
        print(f"  left_br={ctx.has_left_branch} right_br={ctx.has_right_branch}")

        for name, poly in [("cur_lane", ctx.current_lane_polyline),
                           ("left_lane", ctx.left_lane_polyline),
                           ("right_lane", ctx.right_lane_polyline),
                           ("left_br", ctx.left_branch_polyline),
                           ("right_br", ctx.right_branch_polyline)]:
            if poly is not None:
                # Show local coords
                print(f"  {name}: [{poly[0,0]:.2f},{poly[0,1]:.2f}] -> [{poly[-1,0]:.2f},{poly[-1,1]:.2f}]")
                # For branches, also show world endpoint to identify which lane
                if "br" in name:
                    cos_h = np.cos(heading)
                    sin_h = np.sin(heading)
                    mid_local = poly[len(poly)//2]
                    mid_wx = pos[0] + cos_h * mid_local[0] - sin_h * mid_local[1]
                    mid_wy = pos[1] + sin_h * mid_local[0] + cos_h * mid_local[1]
                    # Find which lane this midpoint is closest to
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
            else:
                print(f"  {name}: None")

        from models.diffusion.mode_trajectory_generator import ModeTrajectoryGenerator
        from models.diffusion.mode_definitions import MODE_SLOTS
        gen = ModeTrajectoryGenerator()
        out = gen.generate(ctx)
        for slot in MODE_SLOTS:
            traj = out.coarse_trajectories[slot.index]
            ep = traj[-1]
            v_flag = "V" if out.mode_valid_mask[slot.index] else "X"
            print(f"  [{v_flag}] {slot.name:20s} ep=[{ep[0]:.2f},{ep[1]:.2f}]")

        # Check P4: convert endpoint back to world coords and verify on road
        for slot in MODE_SLOTS:
            if not out.mode_valid_mask[slot.index]:
                continue
            traj = out.coarse_trajectories[slot.index]
            ep_local = traj[-1]
            # Convert local to world
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            ep_world_x = pos[0] + cos_h * ep_local[0] - sin_h * ep_local[1]
            ep_world_y = pos[1] + sin_h * ep_local[0] + cos_h * ep_local[1]
            ep_world = np.array([ep_world_x, ep_world_y])

            # Check if endpoint is on any nearby road
            on_road = False
            best_road_name = "???"
            best_lat = float('inf')
            for sn in rn.graph:
                for en in rn.graph[sn]:
                    for road_lane in rn.graph[sn][en]:
                        try:
                            long_coord, lat_coord = road_lane.local_coordinates(ep_world)
                            if 0 <= long_coord <= road_lane.length + 1.0 and abs(lat_coord) < abs(best_lat):
                                best_lat = lat_coord
                                best_road_name = f"{sn}->{en}"
                                half_w = float(getattr(road_lane, 'width', 3.5)) / 2
                                if abs(lat_coord) <= half_w + 0.5:
                                    on_road = True
                        except Exception:
                            pass

            if not on_road:
                print(f"  *** P4 VIOLATION: {slot.name} endpoint world=[{ep_world_x:.2f},{ep_world_y:.2f}] "
                      f"nearest road={best_road_name} lat={best_lat:.2f}")


def main():
    from envs.diffusion_envs.base_multi_env import BaseMultiEnv

    env_config = {
        "use_render": False,
        "num_scenarios": 1,
        "use_hybrid_map": True,
        "hybrid_map_blocks_config": HYBRID_MAP,
        "map": 5,
        "start_seed": 59,
        "ego_main_route_block_ids": ("g0", "s_ramp0", "c0_ramp0"),
        "traffic_density": 0.0,
        "num_agents": 1,
        "crash_done": False,
        "out_of_road_done": False,
        "image_observation": False,
    }

    env = BaseMultiEnv(env_config)
    try:
        obs, info = env.reset()
        dump_road_network(env)
        test_specific_positions(env)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
