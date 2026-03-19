from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.envs.diffusion_envs.custom_hybrid_map import MAHybridPGMapManager
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.diff_obs.top_down_state_obs_multi_channel import TopDownLidarStateObservation, DatasetCollectObservation
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.utils import Config
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.engine.engine_utils import initialize_global_config
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config, transfuser_config_to_dict


class BaseMultiEnv(MultiAgentMetaDrive):
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
                use_hybrid_map=True,         # True → 使用 MAHybridMap(指定 block 类型序列)
                hybrid_map_sequence="SSXCOCSS",    # block 序列字符串，如 "SXC"、"SSXCS"
                
                horizon=2000,
                force_seed_spawn_manager=True,  # 车辆生成点seed是否与全局seed绑定（可复现性）
                use_render=True,
                camera_height=30,  # [m] 观察视角高度

                # traffic
                traffic_density=0.06,
                traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
                random_traffic=True,
                accident_prob=0.,  # 在reset()时，生成一个静态事故/施工场景的概率

                # Agent
                random_spawn_lane_index=True,
                num_agents=1,
                vehicle_config=dict(
                    vehicle_model="xl",  # 卡车（需在 vehicle_config 子字典中指定才生效）
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
    

# 用于数据收集的环境，observation包含ego_state、lidar、rgb等多模态信息
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
