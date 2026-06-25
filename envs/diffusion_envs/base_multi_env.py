from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from envs.diffusion_envs.custom_hybrid_map import MAHybridPGMapManager
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.diff_obs.top_down_state_obs_multi_channel import TopDownLidarStateObservation, DatasetCollectObservation
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.utils import Config
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.engine.engine_utils import initialize_global_config
from models.diffusion.transfuser_config import build_transfuser_config, transfuser_config_to_dict
from routes.route_definitions import get_route_blocks, get_required_preset, ROUTE_BY_NAME
import numpy as np


# 主线固定路线：
DEFAULT_MAIN_ROUTE_BLOCK_IDS = (
    "s0",
    "c0",
    "c1",
    "g0",
    "s_main0",
    "x0",
    "s_main1",
    "c2",
    "g1",
    "c3",
    "merge0",
    "s_main2",
    "split0",
    "c4",
)

DEFAULT_RAMP_MERGE_ROUTE_BLOCK_IDS = (
    "s0",
    "c0",
    "c1",
    "g0",
    "s_ramp0",
    "c0_ramp0",
    "s_ramp1",
    "c1_ramp0",
    "h_ramp0",
    "g1",
    "c3",
    "merge0",
    "s_main2",
    "split0",
    "c4",
)

DEFAULT_ROUTE_PRESET = "ramp_merge"
ROUTE_PRESET_BLOCK_IDS = {
    "mainline": DEFAULT_MAIN_ROUTE_BLOCK_IDS,
    "ramp_merge": DEFAULT_RAMP_MERGE_ROUTE_BLOCK_IDS,
}


DEFAULT_HYBRID_MAP_CONFIG = [
    {"block_id": "s0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 300.0},
    {"block_id": "c0", "id": "C", "parent_block_id": "s0", "parent_socket_index": 0, "length": 100.0, "radius": 40.0, "angle": 150.0, "dir": 1},
    {"block_id": "c1", "id": "C", "parent_block_id": "c0", "parent_socket_index": 0, "length": 100.0, "radius": 40.0, "angle": 180.0, "dir": 0},
    {"block_id": "g0", "id": "G", "parent_block_id": "c1", "parent_socket_index": 0, "length": 100.0, "extension_length": 30.0},
    {"block_id": "s_main0", "id": "S", "parent_block_id": "g0", "parent_socket_index": 0, "length": 200.0},
    {"block_id": "x0", "id": "X", "parent_block_id": "s_main0", "parent_socket_index": 0, "radius": 60.0, "change_lane_num": 0, "decrease_increase": 0},
    {"block_id": "s_main1", "id": "S", "parent_block_id": "x0", "parent_socket_index": 0, "length": 150.0},
    {"block_id": "c2", "id": "C", "parent_block_id": "s_main1", "parent_socket_index": 0, "length": 100.0, "radius": 55.0, "angle": 90.0, "dir": 1},
    {"block_id": "g1", "id": "g", "parent_block_id": "c2", "parent_socket_index": 0, "length": 100.0, "extension_length": 80.0},
    {"block_id": "c3", "id": "C", "parent_block_id": "g1", "parent_socket_index": 0, "length": 100.0, "radius": 35.0, "angle": 30.0, "dir": 1},
    {"block_id": "merge0", "id": "y", "parent_block_id": "c3", "parent_socket_index": 0, "length": 100.0, "lane_num": 2},
    {"block_id": "s_main2", "id": "S", "parent_block_id": "merge0", "parent_socket_index": 0, "length": 20.0},
    {"block_id": "split0", "id": "Y", "parent_block_id": "s_main2", "parent_socket_index": 0, "length": 200.0, "lane_num": 2},
    {"block_id": "c4", "id": "C", "parent_block_id": "split0", "parent_socket_index": 0, "length": 150.0, "radius": 45.0, "angle": 60.0, "dir": 1},
    {"block_id": "s_ramp0", "id": "s", "parent_block_id": "g0", "parent_socket_index": 1, "length": 60.0},
    {"block_id": "c0_ramp0", "id": "c", "parent_block_id": "s_ramp0", "parent_socket_index": 0, "length": 50.0, "radius": 50.0, "angle": 50.0, "dir": 1},
    {"block_id": "s_ramp1", "id": "s", "parent_block_id": "c0_ramp0", "parent_socket_index": 0, "length": 60.0},
    {"block_id": "c1_ramp0", "id": "c", "parent_block_id": "s_ramp1", "parent_socket_index": 0, "length": 50.0, "radius": 80.0, "angle": 130.0, "dir": 1},
    {
        "block_id": "h_ramp0",
        "id": "H",
        "parent_block_id": "c1_ramp0",
        "parent_socket_index": 0,
        "secondary_parent_block_id": "g1",
        "secondary_parent_socket_index": 1,
    },
]


class BaseMultiEnv(MultiAgentMetaDrive):
    def __init__(self, config=None):
        normalized_config = self._normalize_route_config(config)
        super().__init__(normalized_config)

    @classmethod
    def _resolve_route_block_ids(cls, route_preset: str):
        try:
            return ROUTE_PRESET_BLOCK_IDS[route_preset]
        except KeyError as exc:
            valid = ", ".join(sorted(ROUTE_PRESET_BLOCK_IDS))
            raise ValueError(f"Unknown route_preset '{route_preset}'. Expected one of: {valid}.") from exc

    @classmethod
    def _normalize_route_config(cls, config):
        normalized = {} if config is None else dict(config)
        # local_route takes highest precedence: auto-derive blocks and preset
        local_route = normalized.get("local_route")
        if local_route and local_route in ROUTE_BY_NAME:
            normalized["ego_main_route_block_ids"] = get_route_blocks(local_route)
            normalized.setdefault("route_preset", get_required_preset(local_route))
        elif "route_preset" in normalized:
            normalized["ego_main_route_block_ids"] = cls._resolve_route_block_ids(normalized["route_preset"])
        return normalized

    @staticmethod
    def default_config() -> Config:
        config = MultiAgentMetaDrive.default_config()
        config.update(
            dict(
                # observation
                agent_observation=LidarStateObservation,

                # scenario
                start_seed=1,
                num_scenarios=1,      # 地图池大小，从(seed, seed+num_scenarios)中随机选取seed生成地图
                map=5,      # 随机指定Block数量
                use_hybrid_map=True,         # True → 使用 MAHybridMap(指定 block 配置列表)
                hybrid_map_blocks_config=DEFAULT_HYBRID_MAP_CONFIG,  # block 配置列表

                horizon=2000,
                force_seed_spawn_manager=True,  # 车辆生成点seed是否与全局seed绑定（可复现性）
                ego_spawn_mode="main_route_only",
                route_preset=DEFAULT_ROUTE_PRESET,
                # 片段式路线 (scenario_id / local_route)
                # 设置 local_route 后，ego_main_route_block_ids 和 route_preset 自动推导
                scenario_id=None,   # e.g. "S5_hard_brake_lead"
                local_route=None,   # e.g. "R3_mainline_straight"
                # 手动指定 ego 走的主线（graph_block_id 顺序列表）。
                # 起点 = 第一个 block_id 的第一条正向 respawn road，
                # 终点 = 最后一个 block_id 的第一条正向 respawn road 的 end_node。
                # 取代了原先从 socket 自动走到尽头的实现。
                ego_main_route_block_ids=DEFAULT_MAIN_ROUTE_BLOCK_IDS,
                ego_spawn_buffer_mode="traffic_gap",
                ego_spawn_buffer_scale=1.0,
                platoon_fixed_route_spawn=False,
                platoon_spawn_gap_m=10.0,
                platoon_spawn_tail_buffer_m=6.0,
                platoon_spawn_front_buffer_m=8.0,
                initial_speed_km_h=25.0,
                traffic_spawn_min_gap_ahead=12.0,
                traffic_spawn_min_gap_behind=8.0,
                traffic_spawn_lane_relaxation=True,
                use_render=True,
                camera_height=30,  # [m] 观察视角高度

                # traffic
                traffic_density=0.06,
                traffic_mode=TrafficMode.Hybrid,  # Respawn, Trigger, Basic, Hybrid
                random_traffic=True,
                traffic_target_speed=(18.0, 25.0),  # [km/h] 按 seed 可复现地为每个 episode 采样目标车速
                accident_prob=0.,  # 在reset()时，生成一个静态事故/施工场景的概率

                # Agent
                random_spawn_lane_index=True,
                num_agents=1,
                vehicle_config=dict(
                    vehicle_model="xl",  # 卡车（需在 vehicle_config 子字典中指定才生效）
                    destination=None,
                ),
                crash_done=True,  # agent碰撞后移除，所有agent都移除后，episode结束
                out_of_road_done=True,
                delay_done=25,   # [s] 移除前冷却时间
                allow_respawn=False,  # agent被移除后不允许重生

                )
        )
        return config

    def setup_engine(self):
        super(BaseMultiEnv, self).setup_engine()
        # 用 MAHybridPGMapManager 替换默认的 PGMapManager，
        # 在 use_hybrid_map=False 时其行为与 PGMapManager 完全一致。
        self.engine.update_manager("map_manager", MAHybridPGMapManager())
        from envs.diffusion_envs.route_spawn_manager import RouteAwareSpawnManager
        from envs.diffusion_envs.route_traffic_manager import RouteAwareTrafficManager

        self.engine.update_manager("spawn_manager", RouteAwareSpawnManager())
        self.engine.update_manager("traffic_manager", RouteAwareTrafficManager())
        # Ensure object_manager is always present so scenario recipes can spawn
        # static objects (e.g. TrafficBarrier) regardless of accident_prob.
        if "object_manager" not in self.engine._managers:
            from metadrive.manager.object_manager import TrafficObjectManager
            self.engine.register_manager("object_manager", TrafficObjectManager())

    @staticmethod
    def _get_candidate_drivable_lanes(vehicle):
        navigation = getattr(vehicle, "navigation", None)
        current_ref_lanes = getattr(navigation, "current_ref_lanes", None) or []
        candidates = list(current_ref_lanes)
        current_lane = getattr(vehicle, "lane", None)
        if current_lane is not None:
            candidates.append(current_lane)

        lanes = []
        seen = set()
        for lane in candidates:
            lane_id = id(lane)
            if lane is None or lane_id in seen:
                continue
            seen.add(lane_id)
            lanes.append(lane)
        return lanes

    def _is_out_of_road(self, vehicle):
        if self.config.get("out_of_route_done", False) and getattr(vehicle, "out_of_route", False):
            return True

        candidate_lanes = self._get_candidate_drivable_lanes(vehicle)
        if not candidate_lanes:
            return not getattr(vehicle, "on_lane", False)

        for point in getattr(vehicle, "bounding_box", []):
            if any(lane.point_on_lane(point) for lane in candidate_lanes):
                return False
        return True

# 与 BaseMultiEnv 行为完全相同，但使用 TopDownLidarStateObservation 作为 observation
class TopDownStateMultiEnv(BaseMultiEnv):
    @staticmethod
    def default_config() -> Config:
        config = BaseMultiEnv.default_config()
        config.update(
            dict(
                agent_observation=TopDownLidarStateObservation,
                use_render=True,
            )
        )
        return config
    

# 与 BaseMultiEnv 行为完全相同，但使用多模态传感器作为 Obsercation，与 Transfuser 对齐
class DatasetCollectEnv(BaseMultiEnv):
    def __init__(self, config=None):
        config = {} if config is None else dict(config)
        enable_cuda_image = config.get("image_on_cuda", False)
        num_agents = config.get("num_agents", self.default_config()["num_agents"])
        assert not enable_cuda_image or num_agents == 1, "CUDA image collection only supports num_agents == 1"

        if enable_cuda_image:
            init_config = dict(config)
            init_config["image_on_cuda"] = False
            super().__init__(init_config)
            # 绕过基类中image_on_data的检查
            self.config["image_on_cuda"] = True
            initialize_global_config(self.config)
            self.agent_manager = self._get_agent_manager()
        else:
            super().__init__(config)

    @staticmethod
    def default_config() -> Config:
        config = BaseMultiEnv.default_config()
        config.update(
            dict(
                agent_observation=DatasetCollectObservation,
                image_observation=True,
                use_render=True,
                sensors=dict(
                    rgb_camera=(RGBCamera, 320, 180),
                ),
                dataset_camera_views={},
                transfuser_config=transfuser_config_to_dict(build_transfuser_config("small")),
                transfuser_checkpoint_path=None,
                transfuser_policy_device="cpu",
                transfuser_target_speed_km_h=30.0,
                transfuser_lookahead_index=2,
                transfuser_controller_type="stabilized",
            )
        )
        return config
    
    

if __name__ == "__main__":
    env = DatasetCollectEnv({"use_render": True, "num_agents": 1})
    obs_dict, _ = env.reset()
    print("Agents:", list(obs_dict.keys()))
    obs_sample = list(obs_dict.values())[0]
    if isinstance(obs_sample, dict):
        print("Obs keys:", list(obs_sample.keys()))
        for key, value in obs_sample.items():
            print(f"  {key}: shape={getattr(value, 'shape', None)}, dtype={getattr(value, 'dtype', None)}")

    if hasattr(env, "switch_to_third_person_view"):
        env.switch_to_third_person_view()

    episode = 0
    step = 0
    try:
        while episode < 3:
            actions = {
                agent_id: env.action_space[agent_id].sample()
                for agent_id in env.agents.keys()
            }
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            env.render(
                text={
                    "episode": episode,
                    "step": step,
                    "agents": len(env.agents),
                }
            )
            step += 1

            if terminated["__all__"] or truncated["__all__"]:
                agent_info = list(info.values())[0] if info else {}
                print(
                    f"[Episode {episode}] steps={step}, "
                    f"reward={sum(reward.values()):.2f}, "
                    f"arrive_dest={agent_info.get('arrive_dest', False)}, "
                    f"crash={agent_info.get('crash', False)}, "
                    f"out_of_road={agent_info.get('out_of_road', False)}"
                )
                episode += 1
                step = 0
                obs_dict, _ = env.reset()
                if hasattr(env, "switch_to_third_person_view"):
                    env.switch_to_third_person_view()
    finally:
        env.close()
