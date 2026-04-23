from __future__ import annotations

from typing import Dict, Mapping, Optional
import copy

import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - fallback for older setups
    import gym  # type: ignore

from maps.map_presets import DEFAULT_HYBRID_MAP_CONFIG  # noqa: E402

try:  # pragma: no cover - exercised only in real MetaDrive runtime
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv
    from metadrive.obs.diff_obs.top_down_state_obs_multi_channel import DatasetCollectObservation
    from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
    from metadrive.policy.diffusion_policy.transfuser_features import observation_to_features

    _METADRIVE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - import errors are surfaced at runtime
    RGBCamera = None  # type: ignore
    FirstPGBlock = None  # type: ignore
    _METADRIVE_IMPORT_ERROR = exc
    BaseMultiEnv = object  # type: ignore
    DatasetCollectObservation = object  # type: ignore


from evaluation.platoon_metrics import PlatoonMetrics
from scenarios.hazard_scenarios import get_hazard_scenario_config


def _wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _world_to_ego(ego_pose: np.ndarray, other_pose: np.ndarray) -> np.ndarray:
    dx = float(other_pose[0] - ego_pose[0])
    dy = float(other_pose[1] - ego_pose[1])
    heading = float(ego_pose[2])
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    return np.asarray(
        [
            cos_h * dx + sin_h * dy,
            -sin_h * dx + cos_h * dy,
            _wrap_to_pi(float(other_pose[2]) - heading),
        ],
        dtype=np.float32,
    )


def _ego_to_world(ego_pose: np.ndarray, local_pose: np.ndarray) -> np.ndarray:
    heading = float(ego_pose[2])
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    x_local = float(local_pose[0])
    y_local = float(local_pose[1])
    return np.asarray(
        [
            float(ego_pose[0]) + cos_h * x_local - sin_h * y_local,
            float(ego_pose[1]) + sin_h * x_local + cos_h * y_local,
            _wrap_to_pi(float(ego_pose[2]) + float(local_pose[2])),
        ],
        dtype=np.float32,
    )


def obs_to_tensor(observation: Mapping[str, np.ndarray], device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
    tensor_obs: dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        tensor_obs[key] = tensor
    return tensor_obs


class PlatoonEnvConfig:
    def __init__(
        self,
        num_agents: int = 3,
        use_render: bool = False,
        allow_respawn: bool = False,
        use_hybrid_map: bool = True,
        hybrid_map_blocks_config: Optional[list[dict[str, object]]] = None,
        num_scenarios: int = 1,
        traffic_density: float = 0.04,
        horizon: int = 1000,
        initial_speed_km_h: float = 25.0,
        target_speed_km_h: float = 25.0,
        headway_time_s: float = 0.5,
        trajectory_dt: float = 0.5,
        vehicle_length_m: float = 5.74,
        formation_error_threshold: float = 2.0,
        observation_mode: str = "lidar_state",
        scenario_id: Optional[str] = None,
        local_route: Optional[str] = None,
    ) -> None:
        self.num_agents = int(num_agents)
        self.use_render = bool(use_render)
        self.allow_respawn = bool(allow_respawn)
        self.use_hybrid_map = bool(use_hybrid_map)
        if hybrid_map_blocks_config is None and self.use_hybrid_map:
            self.hybrid_map_blocks_config = copy.deepcopy(list(DEFAULT_HYBRID_MAP_CONFIG))
        elif hybrid_map_blocks_config is None:
            self.hybrid_map_blocks_config = None
        else:
            self.hybrid_map_blocks_config = copy.deepcopy(hybrid_map_blocks_config)
        self.num_scenarios = int(num_scenarios)
        self.traffic_density = float(traffic_density)
        self.horizon = int(horizon)
        self.initial_speed_km_h = float(initial_speed_km_h)
        self.target_speed_km_h = float(target_speed_km_h)
        self.headway_time_s = float(headway_time_s)
        self.trajectory_dt = float(trajectory_dt)
        self.vehicle_length_m = float(vehicle_length_m)
        self.formation_error_threshold = float(formation_error_threshold)
        self.observation_mode = str(observation_mode)
        self.scenario_id = scenario_id
        self.local_route = local_route


class PlatoonEnv(BaseMultiEnv):
    """Phase 1 platoon environment with trajectory-mode and low-level action compatibility."""

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        if _METADRIVE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "PlatoonEnv requires the full MetaDrive runtime. "
                "Please install/run the AGENTS.md environment dependencies before using this environment."
            ) from _METADRIVE_IMPORT_ERROR
        merged = self._resolve_hazard_scenario(self._merge_config(config))
        self._env_overrides = self._extract_env_overrides(merged)
        self._runtime_flags = self._extract_runtime_flags(merged)
        self.platoon_config = PlatoonEnvConfig(**self._extract_platoon_config(merged))
        self._metrics = PlatoonMetrics(formation_error_threshold=self.platoon_config.formation_error_threshold)
        self._last_info: dict[str, dict] = {}
        self._agent_ids = [f"agent{i}" for i in range(self.platoon_config.num_agents)]
        self._last_actions: dict[str, np.ndarray] = {}
        self._last_progress_refs: dict[str, tuple[object, float, np.ndarray]] = {}
        self._multimodal_config = build_transfuser_config("base")
        self._scenario_orchestrator = None
        self._scenario_step_count = 0
        super().__init__(config=self._build_metadrive_config())

    @classmethod
    def _merge_config(cls, config: Optional[Mapping[str, object]]) -> dict:
        defaults = PlatoonEnvConfig().__dict__.copy()
        if config:
            defaults.update(dict(config))
        return defaults

    @staticmethod
    def _resolve_hazard_scenario(config: Mapping[str, object]) -> dict[str, object]:
        resolved = dict(config)
        scenario_name = resolved.get("hazard_scenario", None)
        scenario_config = get_hazard_scenario_config(scenario_name)
        if scenario_config is not None:
            resolved.update(dict(scenario_config.get("env_overrides", {})))
            # Propagate scenario_id / local_route from hazard config only when the
            # caller has not already set them explicitly.
            for field in ("scenario_id", "local_route"):
                if resolved.get(field) is None and scenario_config.get(field) is not None:
                    resolved[field] = scenario_config[field]
        return resolved

    @staticmethod
    def _extract_platoon_config(config: Mapping[str, object]) -> dict[str, object]:
        keys = {
            "num_agents",
            "use_render",
            "allow_respawn",
            "use_hybrid_map",
            "hybrid_map_blocks_config",
            "num_scenarios",
            "traffic_density",
            "horizon",
            "initial_speed_km_h",
            "target_speed_km_h",
            "headway_time_s",
            "trajectory_dt",
            "vehicle_length_m",
            "formation_error_threshold",
            "observation_mode",
            "scenario_id",
            "local_route",
        }
        return {key: config[key] for key in keys if key in config}

    @staticmethod
    def _extract_env_overrides(config: Mapping[str, object]) -> dict[str, object]:
        platoon_keys = {
            "num_agents",
            "use_render",
            "allow_respawn",
            "use_hybrid_map",
            "hybrid_map_blocks_config",
            "num_scenarios",
            "traffic_density",
            "horizon",
            "initial_speed_km_h",
            "target_speed_km_h",
            "headway_time_s",
            "trajectory_dt",
            "vehicle_length_m",
            "formation_error_threshold",
            "observation_mode",
            # scenario_id / local_route are intentionally excluded here so they pass
            # through to the MetaDrive config via _build_metadrive_config explicitly.
        }
        runtime_only_keys = {"enable_idm_lane_change", "hazard_scenario", "scenario_id", "local_route"}
        return {key: value for key, value in config.items() if key not in platoon_keys and key not in runtime_only_keys}

    @staticmethod
    def _extract_runtime_flags(config: Mapping[str, object]) -> dict[str, object]:
        runtime_only_keys = {"enable_idm_lane_change", "hazard_scenario"}
        return {key: value for key, value in config.items() if key in runtime_only_keys}

    def _build_metadrive_config(self) -> dict:
        speed_m_s = self.platoon_config.initial_speed_km_h / 3.6
        gap_m = self._desired_center_spacing_m()
        lane_index = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
        lead_long = 20.0
        agent_configs = {
            f"agent{i}": {
                "spawn_lane_index": lane_index,
                "spawn_longitude": lead_long - i * gap_m,
                "spawn_lateral": 0.0,
                "spawn_velocity": (speed_m_s, 0.0),
                "spawn_velocity_car_frame": True,
            }
            for i in range(self.platoon_config.num_agents)
        }
        # BaseMultiEnv already defaults: use_hybrid_map=True,
        # traffic_mode=TrafficMode.Trigger, agent_observation=LidarStateObservation.
        # Only override what differs from those defaults.
        cfg = {
            "num_agents": self.platoon_config.num_agents,
            "allow_respawn": self.platoon_config.allow_respawn,
            "use_hybrid_map": self.platoon_config.use_hybrid_map,
            "hybrid_map_blocks_config": self.platoon_config.hybrid_map_blocks_config,
            "num_scenarios": self.platoon_config.num_scenarios,
            "use_render": self.platoon_config.use_render,
            "traffic_density": self.platoon_config.traffic_density,
            "force_seed_spawn_manager": True,
            "random_spawn_lane_index": False,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,
            "agent_configs": agent_configs,
            "horizon": self.platoon_config.horizon,
            **self._build_observation_config(),
            **self._env_overrides,
        }
        # Pass scenario_id / local_route to MetaDrive; base_multi_env._normalize_route_config
        # will derive ego_main_route_block_ids and route_preset from local_route automatically.
        if self.platoon_config.scenario_id is not None:
            cfg["scenario_id"] = self.platoon_config.scenario_id
        if self.platoon_config.local_route is not None:
            cfg["local_route"] = self.platoon_config.local_route
        return cfg

    def _build_observation_config(self) -> dict[str, object]:
        if self.platoon_config.observation_mode != "multimodal":
            return {}
        return {
            "agent_observation": DatasetCollectObservation,
            "image_observation": True,
            "sensors": {
                "rgb_camera": (RGBCamera, 320, 180),
            },
        }

    @property
    def action_space(self):
        if self.config.get("control_mode", "physics") == "teleport":
            return super().action_space
        single = gym.spaces.Box(low=-100.0, high=100.0, shape=(8, 3), dtype=np.float32)
        return gym.spaces.Dict({agent_id: single for agent_id in self._agent_ids})

    def _reposition_platoon_on_route(self) -> None:
        """Teleport the platoon onto the first road of the configured local_route.

        This is needed when local_route points to a segment other than the map entry
        (e.g. R3_mainline_straight starts at s_main0, not FirstPGBlock.NODE_1).
        The vehicles are initially spawned at NODE_1/NODE_2 (the safe default), then
        immediately repositioned after reset() so the episode starts in the correct
        road segment.
        """
        if not self.platoon_config.local_route:
            return
        spawn_manager = getattr(self.engine, "spawn_manager", None)
        current_map = getattr(self.engine, "current_map", None)
        if spawn_manager is None or current_map is None:
            return
        try:
            route_roads = spawn_manager.get_main_route_spawn_roads(current_map)
        except Exception:
            return
        if not route_roads:
            return
        road = route_roads[0]
        try:
            lanes = current_map.road_network.graph[road.start_node][road.end_node]
        except (KeyError, AttributeError, TypeError):
            return
        if not lanes:
            return

        speed_m_s = self.platoon_config.initial_speed_km_h / 3.6
        gap_m = self._desired_center_spacing_m()
        lead_long = 20.0
        for i, agent_id in enumerate(self._agent_ids):
            vehicle = self.agents.get(agent_id)
            if vehicle is None:
                continue
            long = max(lead_long - i * gap_m, 2.0)
            lane_idx = 0
            lane = lanes[lane_idx]
            pos = lane.position(long, 0.0)
            heading = lane.heading_theta_at(long)
            vehicle.set_position(pos)
            vehicle.set_heading_theta(heading)
            # Re-apply initial speed along lane heading
            vehicle.set_velocity(
                (speed_m_s * np.cos(heading), speed_m_s * np.sin(heading)),
                in_local_frame=False,
            )

    def _setup_scenario_orchestrator(self) -> None:
        """Initialise PlatoonScenarioOrchestrator when scenario_id + local_route are both set."""
        self._scenario_orchestrator = None
        self._scenario_step_count = 0
        scenario_id = self.platoon_config.scenario_id
        local_route = self.platoon_config.local_route
        if not scenario_id or not local_route:
            return
        try:
            from scenarios.definitions import get_scenario_definition, SCENARIO_BY_ID
            from scenarios.platoon_orchestrator import PlatoonScenarioOrchestrator
            if scenario_id not in SCENARIO_BY_ID:
                return
            defn = get_scenario_definition(scenario_id)
            if local_route not in defn.trigger_by_local_route:
                return
            self._scenario_orchestrator = PlatoonScenarioOrchestrator(
                defn, local_route, self._agent_ids
            )
            self._scenario_orchestrator.reset(self, self._agent_ids[0])
        except Exception:
            self._scenario_orchestrator = None

    def reset(self):
        self._metrics.start_episode()
        obs, _ = super().reset()

        if "enable_idm_lane_change" in self._runtime_flags:
            self.config["enable_idm_lane_change"] = bool(self._runtime_flags["enable_idm_lane_change"])
            self.engine.global_config["enable_idm_lane_change"] = bool(self._runtime_flags["enable_idm_lane_change"])

        # Teleport to route segment if local_route is set
        self._reposition_platoon_on_route()

        # Initialise scenario orchestrator (hazard injection)
        self._setup_scenario_orchestrator()

        self._last_actions = {
            agent_id: np.zeros((2,), dtype=np.float32) for agent_id in self._agent_ids
        }
        self._last_progress_refs = {
            agent_id: self._capture_progress_reference(agent_id) for agent_id in self._agent_ids
        }
        self._last_info = {}

        return self._augment_observations(obs)

    def _augment_observations(self, obs: Mapping[str, object]) -> dict[str, dict]:
        ret = {}
        for agent_id, agent_obs in obs.items():
            ret[agent_id] = self._format_agent_observation(agent_id, agent_obs)
        return ret

    def _format_agent_observation(self, agent_id: str, agent_obs: object) -> dict[str, np.ndarray]:
        if self.platoon_config.observation_mode == "multimodal":
            if not isinstance(agent_obs, dict):
                raise TypeError("Multimodal PlatoonEnv expects dict observations from DatasetCollectObservation.")
            features = observation_to_features(agent_obs, self._multimodal_config)
            return {
                "camera": features["camera_feature"].cpu().numpy().astype(np.float32, copy=False),
                "lidar": features["lidar_feature"].cpu().numpy().astype(np.float32, copy=False),
                "status": features["status_feature"].cpu().numpy().astype(np.float32, copy=False),
                "formation_relation_state": self.get_formation_relation_state(agent_id),
            }

        if isinstance(agent_obs, dict):
            augmented = dict(agent_obs)
        else:
            augmented = {"obs": agent_obs}
        augmented["formation_relation_state"] = self.get_formation_relation_state(agent_id)
        return augmented

    def step(self, actions: Dict[str, np.ndarray]):
        if self.config.get("control_mode", "physics") == "teleport":
            obs, reward, terminated, truncated, info = super().step(actions)
            info = self._build_info_dict("teleport", actions=actions, base_info=info)
            terminated, truncated = self._enforce_platoon_episode_end(terminated, truncated, info)
            obs = self._augment_observations(obs)
            self._metrics.update(info)
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                self._metrics.end_episode()
            if info:
                self._last_info = info
            return obs, reward, terminated, truncated, info

        control_mode = self._infer_control_mode(actions)
        if control_mode == "trajectory":
            low_level_actions = {
                agent_id: self.trajectory_to_control(agent_id, np.asarray(action, dtype=np.float32))
                for agent_id, action in actions.items()
            }
        else:
            low_level_actions = {
                agent_id: np.asarray(action, dtype=np.float32).reshape(2,)
                for agent_id, action in actions.items()
            }
        return self.low_level_step(low_level_actions, control_mode=control_mode)

    def _infer_control_mode(self, actions: Mapping[str, np.ndarray]) -> str:
        if self.config.get("control_mode", "physics") == "teleport":
            return "teleport"
        first_action = np.asarray(next(iter(actions.values())))
        if first_action.shape == (8, 3):
            return "trajectory"
        if first_action.shape == (2,):
            return "low_level"
        raise ValueError(f"Unsupported action shape for PlatoonEnv: {first_action.shape}")

    def low_level_step(self, actions: Dict[str, np.ndarray], control_mode: str = "low_level"):
        # Tick ScenarioOrchestrator before the physics step (mirrors collect_expert behaviour)
        if getattr(self, "_scenario_orchestrator", None) is not None:
            lead_agent_id = self._agent_ids[0]
            self._scenario_step_count = getattr(self, "_scenario_step_count", 0) + 1
            self._scenario_orchestrator.before_step(self, lead_agent_id, self._scenario_step_count)

        obs, reward, terminated, truncated, info = super().step(actions)
        info = self._build_info_dict(control_mode, actions=actions, base_info=info)
        terminated, truncated = self._enforce_platoon_episode_end(terminated, truncated, info)
        obs = self._augment_observations(obs)
        self._metrics.update(info)
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            self._metrics.end_episode()
        if info:
            self._last_info = info
        return obs, reward, terminated, truncated, info

    def _enforce_platoon_episode_end(
        self,
        terminated: Mapping[str, bool],
        truncated: Mapping[str, bool],
        info: Mapping[str, Mapping[str, object]],
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        terminated = dict(terminated)
        truncated = dict(truncated)
        active_ids = [agent_id for agent_id in self._agent_ids if agent_id in info]
        if not active_ids:
            terminated["__all__"] = True
            truncated["__all__"] = False
            return terminated, truncated
        any_failure = any(
            bool(
                info[agent_id].get("crash", False)
                or info[agent_id].get("crash_vehicle", False)
                or info[agent_id].get("crash_object", False)
                or info[agent_id].get("crash_building", False)
                or info[agent_id].get("out_of_road", False)
            )
            for agent_id in active_ids
        )
        if any_failure:
            for agent_id in active_ids:
                terminated[agent_id] = True
            terminated["__all__"] = True
            truncated["__all__"] = False
            return terminated, truncated

        terminated["__all__"] = bool(terminated.get("__all__", False)) or any(
            bool(terminated.get(agent_id, False)) for agent_id in active_ids
        )
        truncated["__all__"] = bool(truncated.get("__all__", False)) or all(
            bool(truncated.get(agent_id, False)) for agent_id in active_ids
        )
        return terminated, truncated

    def _capture_progress_reference(self, agent_id: str) -> tuple[object, float, np.ndarray]:
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            return (None, 0.0, np.zeros((2,), dtype=np.float32))
        lane = getattr(vehicle, "lane", None)
        if lane is not None:
            longitudinal = float(lane.local_coordinates(vehicle.position)[0])
        else:
            longitudinal = 0.0
        return (lane, longitudinal, np.asarray(vehicle.position[:2], dtype=np.float32))

    def _compute_progress(self, agent_id: str) -> float:
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            return 0.0
        previous_lane, previous_s, previous_xy = self._last_progress_refs.get(
            agent_id, (None, 0.0, np.asarray(vehicle.position[:2], dtype=np.float32))
        )
        lane = getattr(vehicle, "lane", None)
        current_xy = np.asarray(vehicle.position[:2], dtype=np.float32)
        if lane is not None and previous_lane is lane:
            current_s = float(lane.local_coordinates(vehicle.position)[0])
            progress = current_s - float(previous_s)
        else:
            progress = float(np.linalg.norm(current_xy - previous_xy))
            current_s = float(lane.local_coordinates(vehicle.position)[0]) if lane is not None else 0.0
        self._last_progress_refs[agent_id] = (lane, current_s, current_xy)
        return float(progress)

    def _build_info_dict(
        self,
        control_mode: str,
        actions: Mapping[str, np.ndarray],
        base_info: Optional[Mapping[str, dict]] = None,
    ) -> dict[str, dict]:
        info = {}
        min_gap = self._compute_min_gap()
        active_ids = list(self.agents.keys())
        for agent_id in active_ids:
            agent_info = dict((base_info or {}).get(agent_id, {}))
            raw_action = np.asarray(actions.get(agent_id, np.zeros((2,), dtype=np.float32)), dtype=np.float32)
            if raw_action.shape == (2,):
                current_action = raw_action
            else:
                current_action = np.zeros((2,), dtype=np.float32)
            previous_action = self._last_actions.get(agent_id, np.zeros((2,), dtype=np.float32))
            delta_steering = float(current_action[0] - previous_action[0])
            jerk = float(current_action[1] - previous_action[1])
            agent_info["formation_relation_state"] = self.get_formation_relation_state(agent_id)
            agent_info["formation_error"] = self._compute_agent_formation_error(agent_id)
            agent_info["min_gap"] = min_gap
            agent_info["control_mode"] = control_mode
            agent_info["crash"] = bool(agent_info.get("crash", False))
            agent_info["arrive_dest"] = bool(agent_info.get("arrive_dest", False))
            agent_info["out_of_road"] = bool(agent_info.get("out_of_road", False))
            agent_info["progress"] = float(self._compute_progress(agent_id))
            agent_info["jerk"] = jerk
            agent_info["delta_steering"] = delta_steering
            agent_info["speed_km_h"] = float(self._agent_speed_km_h(agent_id))
            self._last_actions[agent_id] = current_action
            info[agent_id] = agent_info
        return info

    def _agent_pose(self, agent_id: str) -> Optional[np.ndarray]:
        """Returns [x, y, heading] or None if the agent is no longer active."""
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            return None
        return np.asarray([vehicle.position[0], vehicle.position[1], vehicle.heading_theta], dtype=np.float32)

    def _agent_speed_km_h(self, agent_id: str) -> float:
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            return 0.0
        return float(vehicle.speed_km_h)

    def _agent_velocity_ms(self, agent_id: str) -> np.ndarray:
        """Returns [vx, vy] in world frame (m/s). Falls back to heading-aligned speed."""
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            return np.zeros(2, dtype=np.float32)
        velocity = getattr(vehicle, "velocity", None)
        if velocity is not None:
            return np.asarray(velocity[:2], dtype=np.float32)
        speed = self._agent_speed_km_h(agent_id) / 3.6
        heading = float(vehicle.heading_theta)
        return np.asarray([speed * np.cos(heading), speed * np.sin(heading)], dtype=np.float32)

    def _vehicle_length_m(self, agent_id: Optional[str] = None) -> float:
        if agent_id is None:
            return float(self.platoon_config.vehicle_length_m)
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            return float(self.platoon_config.vehicle_length_m)
        return float(getattr(vehicle, "LENGTH", self.platoon_config.vehicle_length_m))

    def _desired_center_spacing_m(self, ego_id: Optional[str] = None, other_id: Optional[str] = None) -> float:
        speed_m_s = self.platoon_config.initial_speed_km_h / 3.6
        ego_length = self._vehicle_length_m(ego_id)
        other_length = self._vehicle_length_m(other_id)
        return 0.5 * (ego_length + other_length) + speed_m_s * self.platoon_config.headway_time_s

    def get_formation_relation_state(self, agent_id: str) -> np.ndarray:
        ego_pose = self._agent_pose(agent_id)
        ego_idx = self._agent_ids.index(agent_id)
        relation = []
        for other_id in self._agent_ids:
            if other_id == agent_id:
                continue
            other_pose = self._agent_pose(other_id)
            # Pad zeros if ego or neighbor is no longer active
            if ego_pose is None or other_pose is None:
                relation.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            other_idx = self._agent_ids.index(other_id)
            local = _world_to_ego(ego_pose, other_pose)
            desired_gap = self._desired_center_spacing_m(agent_id, other_id)
            relation.extend(
                [
                    float(local[0]),
                    float(local[1]),
                    float(local[2]),
                    float(self._agent_speed_km_h(other_id) - self._agent_speed_km_h(agent_id)),
                    float((ego_idx - other_idx) * desired_gap),
                    0.0,
                ]
            )
        while len(relation) < 12:
            relation.append(0.0)
        return np.asarray(relation[:12], dtype=np.float32)

    def _compute_agent_formation_error(self, agent_id: str) -> float:
        """Returns the mean absolute error to the desired position relative to neighbors."""
        ego_pose = self._agent_pose(agent_id)
        if ego_pose is None:
            return 0.0
        ego_idx = self._agent_ids.index(agent_id)
        errors = []
        for neighbor_offset in (-1, 1):
            other_idx = ego_idx + neighbor_offset
            if other_idx < 0 or other_idx >= len(self._agent_ids):
                continue
            other_id = self._agent_ids[other_idx]
            other_pose = self._agent_pose(other_id)
            if other_pose is None:
                continue
            desired_gap = self._desired_center_spacing_m(agent_id, other_id)
            actual_gap = float(np.linalg.norm(other_pose[:2] - ego_pose[:2]))
            errors.append(abs(actual_gap - desired_gap))
        if not errors:
            return 0.0
        return float(np.mean(errors, dtype=np.float32))

    def _compute_min_gap(self) -> float:
        poses = [self._agent_pose(aid) for aid in self._agent_ids]
        poses = [p for p in poses if p is not None]  # skip done agents
        min_gap = float("inf")
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                gap = float(np.linalg.norm(poses[i][:2] - poses[j][:2]))
                min_gap = min(min_gap, gap)
        return 0.0 if not np.isfinite(min_gap) else min_gap

    def _lateral_pd(self, trajectory: np.ndarray) -> float:
        lookahead_idx = min(2, len(trajectory) - 1)
        waypoint = trajectory[lookahead_idx]
        x = max(float(waypoint[0]), 1e-3)
        y = float(waypoint[1])
        heading = float(waypoint[2]) if trajectory.shape[1] > 2 else 0.0
        steering_angle = float(np.arctan2(y, x))
        steering = 0.85 * steering_angle + 0.35 * _wrap_to_pi(heading)
        return float(np.clip(steering, -1.0, 1.0))

    def _solve_lqr_gain(self, dt: float = 0.1) -> np.ndarray:
        a = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
        b = np.asarray([[0.0], [dt]], dtype=np.float32)
        q = np.diag([1.2, 0.8]).astype(np.float32)
        r = np.asarray([[0.5]], dtype=np.float32)
        p = q.copy()
        for _ in range(50):
            bt_p = b.T @ p
            inv = np.linalg.inv(r + bt_p @ b)
            p = q + a.T @ p @ a - a.T @ p @ b @ inv @ bt_p @ a
        k = np.linalg.inv(r + b.T @ p @ b) @ (b.T @ p @ a)
        return np.asarray(k, dtype=np.float32)

    def _longitudinal_lqr(self, agent_id: str) -> float:
        ego_idx = self._agent_ids.index(agent_id)
        current_speed = self._agent_speed_km_h(agent_id) / 3.6
        target_speed = self.platoon_config.target_speed_km_h / 3.6

        def _speed_tracking_accel() -> float:
            speed_error = current_speed - target_speed
            accel = -0.35 * speed_error
            return float(np.clip(accel, -1.0, 1.0))

        if ego_idx == 0:
            return _speed_tracking_accel()

        front_id = self._agent_ids[ego_idx - 1]
        ego_pose = self._agent_pose(agent_id)
        front_pose = self._agent_pose(front_id)
        if ego_pose is None or front_pose is None:
            return _speed_tracking_accel()

        front_local = _world_to_ego(ego_pose, front_pose)
        actual_gap = float(front_local[0])
        desired_gap = self._desired_center_spacing_m(agent_id, front_id)
        distance_error = desired_gap - actual_gap
        speed_error = current_speed - target_speed
        state = np.asarray([distance_error, speed_error], dtype=np.float32)
        gain = self._solve_lqr_gain()
        accel = -float((gain @ state.reshape(-1, 1)).item())
        return float(np.clip(accel, -1.0, 1.0))

    def trajectory_to_control(self, agent_id: str, trajectory: np.ndarray) -> np.ndarray:
        if trajectory.shape != (8, 3):
            raise ValueError(f"Expected trajectory shape (8, 3), got {trajectory.shape}")
        steering = self._lateral_pd(trajectory)
        throttle = self._longitudinal_lqr(agent_id)
        return np.asarray([steering, throttle], dtype=np.float32)

    def get_platoon_metrics(self) -> dict:
        return self._metrics.compute()

    def evaluate_trajectory_group(self, agent_id: str, trajectories: np.ndarray) -> dict:
        if not self.agents:
            raise RuntimeError("PlatoonEnv.evaluate_trajectory_group() requires env.reset() before use.")

        traj_array = np.asarray(trajectories, dtype=np.float32)
        if traj_array.ndim != 3 or traj_array.shape[1:] != (8, 3):
            raise ValueError(f"Expected trajectories shape [G, 8, 3], got {traj_array.shape}")

        ego_pose = self._agent_pose(agent_id)
        if ego_pose is None:
            raise RuntimeError(f"Agent {agent_id} is no longer active; cannot evaluate trajectories.")

        other_poses_t0: dict[str, np.ndarray] = {}
        other_velocities: dict[str, np.ndarray] = {}
        for other_id in self._agent_ids:
            if other_id == agent_id:
                continue
            pose = self._agent_pose(other_id)
            if pose is None:
                continue
            other_poses_t0[other_id] = pose
            other_velocities[other_id] = self._agent_velocity_ms(other_id)
        current_relation = self.get_formation_relation_state(agent_id).copy()

        step_infos: list[list[dict]] = []
        crash_flags: list[bool] = []
        out_flags: list[bool] = []

        for trajectory in traj_array:
            per_step: list[dict] = []

            # 转到世界坐标
            trajectory_world = np.stack([_ego_to_world(ego_pose, pose) for pose in trajectory], axis=0)
            
            crash = False
            out_of_road = False
            min_gap_episode = float("inf")
            prev_heading = float(trajectory[0, 2])
            prev_turn = 0.0

            for step_idx, world_pose in enumerate(trajectory_world):
                local_pose = trajectory[step_idx]
                t = float(step_idx + 1) * float(getattr(self.platoon_config, "trajectory_dt", 0.5))
                other_poses_at_t: dict[str, np.ndarray] = {}
                for other_id, pose_t0 in other_poses_t0.items():
                    vel = other_velocities[other_id]
                    other_poses_at_t[other_id] = np.asarray(
                        [pose_t0[0] + vel[0] * t, pose_t0[1] + vel[1] * t, pose_t0[2]],
                        dtype=np.float32,
                    )

                progress = float(max(local_pose[0], 0.0))
                formation_error = self._surrogate_formation_error(agent_id, world_pose, other_poses_at_t)
                min_gap = self._surrogate_min_gap(world_pose, other_poses_at_t)
                min_gap_episode = min(min_gap_episode, min_gap)
                lateral_abs = self._surrogate_lateral_offset(agent_id, world_pose)
                crash = crash or bool(min_gap < self._vehicle_length_m(agent_id) * 0.5)
                out_of_road = out_of_road or bool(lateral_abs > self._road_half_width(agent_id))

                heading = float(local_pose[2])
                delta_steering = _wrap_to_pi(heading - prev_heading)
                jerk = float(delta_steering - prev_turn)
                prev_heading = heading
                prev_turn = delta_steering

                per_step.append(
                    {
                        "progress": progress,  # *进度奖励
                        "formation_error": float(formation_error),  # *队形奖励
                        "min_gap": float(min_gap),  # *agents之间的最小距离
                        "jerk": float(jerk),  # *steering jerk
                        "delta_steering": float(delta_steering),  # *steering change
                        "crash": bool(crash),  # *是否碰撞
                        "out_of_road": bool(out_of_road),  # *是否出界
                    }
                )

            if np.isfinite(min_gap_episode):
                for item in per_step:
                    item["min_gap"] = float(min_gap_episode)

            step_infos.append(per_step)
            crash_flags.append(bool(crash))
            out_flags.append(bool(out_of_road))

        # Ensure surrogate evaluation does not mutate env state.
        assert np.allclose(current_relation, self.get_formation_relation_state(agent_id))
        return {
            "step_infos": step_infos,
            "crash_flags": crash_flags,
            "out_of_road_flags": out_flags,
        }

    def get_state(self) -> dict:
        vehicle_states = {}
        for agent_id in self._agent_ids:
            vehicle = self.agents.get(agent_id)
            if vehicle is None:
                continue
            steering = None
            if hasattr(vehicle, "steering"):
                try:
                    steering = float(vehicle.steering)
                except Exception:
                    steering = None
            vehicle_states[agent_id] = {
                "object_name": str(getattr(vehicle, "name", agent_id)),
                "position": np.asarray(vehicle.position, dtype=np.float64).copy(),
                "heading": float(vehicle.heading_theta),
                "velocity": np.asarray(getattr(vehicle, "velocity", (0.0, 0.0)), dtype=np.float64).copy(),
                "steering": steering,
            }

        traffic_states = {}
        try:
            if hasattr(self, "engine") and self.engine is not None:
                traffic_manager = getattr(self.engine, "traffic_manager", None)
                traffic_vehicles = getattr(traffic_manager, "traffic_vehicles", None) if traffic_manager is not None else None
                if traffic_vehicles is not None:
                    for vehicle in traffic_vehicles:
                        vehicle_name = getattr(vehicle, "name", None)
                        if not vehicle_name:
                            continue
                        traffic_states[str(vehicle_name)] = {
                            "position": np.asarray(getattr(vehicle, "position", (0.0, 0.0)), dtype=np.float64).copy(),
                            "heading": float(getattr(vehicle, "heading_theta", 0.0)),
                            "velocity": np.asarray(getattr(vehicle, "velocity", (0.0, 0.0)), dtype=np.float64).copy(),
                        }
        except Exception:
            traffic_states = {}

        last_actions = {}
        for key, value in getattr(self, "_last_actions", {}).items():
            if isinstance(value, np.ndarray):
                last_actions[key] = value.copy()
            else:
                last_actions[key] = value

        last_progress = {}
        for key, value in getattr(self, "_last_progress_refs", {}).items():
            lane, longitudinal, position = value
            last_progress[key] = {
                "lane": lane,
                "longitudinal": float(longitudinal),
                "position": np.asarray(position, dtype=np.float64).copy(),
            }

        last_info = {}
        for key, value in getattr(self, "_last_info", {}).items():
            last_info[key] = dict(value)

        return {
            "vehicle_states": vehicle_states,
            "traffic_states": traffic_states,
            "_last_actions": last_actions,
            "_last_progress_refs": last_progress,
            "_last_info": last_info,
        }

    def set_state(self, state: dict) -> None:
        def _raise_restore_error(agent_id: str, missing_api: str) -> None:
            raise RuntimeError(f"Cannot restore state for {agent_id}: no supported {missing_api} API")

        def _restore_position(agent_id: str, vehicle, position: np.ndarray) -> None:
            if hasattr(vehicle, "set_position"):
                vehicle.set_position(position.tolist())
                return
            origin = getattr(vehicle, "origin", None)
            if origin is not None and hasattr(origin, "setPos"):
                z_value = 0.0
                if hasattr(origin, "getPos"):
                    try:
                        z_value = float(origin.getPos()[2])
                    except Exception:
                        z_value = 0.0
                origin.setPos(float(position[0]), float(position[1]), z_value)
                return
            _raise_restore_error(agent_id, "position")

        def _restore_heading(agent_id: str, vehicle, heading: float) -> None:
            if hasattr(vehicle, "set_heading_theta"):
                vehicle.set_heading_theta(heading)
                return
            origin = getattr(vehicle, "origin", None)
            if origin is not None and hasattr(origin, "setH"):
                origin.setH(np.degrees(heading))
                return
            _raise_restore_error(agent_id, "heading")

        def _restore_velocity(agent_id: str, vehicle, velocity: np.ndarray) -> None:
            if hasattr(vehicle, "set_velocity"):
                vehicle.set_velocity(velocity.tolist())
                return
            chassis = getattr(vehicle, "chassis", None)
            node = chassis.node() if chassis is not None and hasattr(chassis, "node") else None
            if node is not None and hasattr(node, "setLinearVelocity"):
                vector = None
                try:
                    from panda3d.core import Vec3  # pragma: no cover
                    vector = Vec3(float(velocity[0]), float(velocity[1]), 0.0)
                except Exception:
                    vector = (float(velocity[0]), float(velocity[1]), 0.0)
                node.setLinearVelocity(vector)
                return
            _raise_restore_error(agent_id, "velocity")

        vehicle_states = state.get("vehicle_states", {})
        agent_manager = getattr(self, "agent_manager", None)
        available_agent_objects = {}
        if agent_manager is not None:
            available_agent_objects.update(getattr(agent_manager, "_active_objects", {}))
            for object_name, payload in getattr(agent_manager, "_dying_objects", {}).items():
                if isinstance(payload, (list, tuple)) and payload:
                    available_agent_objects[object_name] = payload[0]
            # Fallback: retrieve from episode_created_agents for fully terminated agents
            episode_created = getattr(agent_manager, "episode_created_agents", None)
            if episode_created is not None:
                for agent_id, vehicle in episode_created.items():
                    obj_name = str(getattr(vehicle, "name", agent_id))
                    if obj_name not in available_agent_objects:
                        available_agent_objects[obj_name] = vehicle

        restored_active_objects = {}
        restored_agent_to_object = {}
        restored_object_to_agent = {}
        for agent_id, vehicle_state in vehicle_states.items():
            snapshot_object_name = str(vehicle_state.get("object_name", agent_id))
            vehicle = self.agents.get(agent_id)
            if vehicle is None:
                vehicle = available_agent_objects.get(snapshot_object_name)
            if vehicle is None:
                continue
            position = np.asarray(vehicle_state.get("position", vehicle.position), dtype=np.float64)
            heading = float(vehicle_state.get("heading", vehicle.heading_theta))
            velocity = np.asarray(vehicle_state.get("velocity", getattr(vehicle, "velocity", (0.0, 0.0))), dtype=np.float64)
            steering = vehicle_state.get("steering")

            _restore_position(agent_id, vehicle, position)
            _restore_heading(agent_id, vehicle, heading)
            _restore_velocity(agent_id, vehicle, velocity)

            if steering is not None and hasattr(vehicle, "steering"):
                try:
                    vehicle.steering = float(steering)
                except Exception:
                    pass
            if hasattr(vehicle, "set_static"):
                try:
                    vehicle.set_static(False)
                except Exception:
                    pass

            current_object_name = str(getattr(vehicle, "name", snapshot_object_name))
            restored_active_objects[current_object_name] = vehicle
            restored_agent_to_object[agent_id] = current_object_name
            restored_object_to_agent[current_object_name] = agent_id

        if agent_manager is not None and restored_active_objects:
            agent_manager._active_objects = dict(restored_active_objects)
            agent_manager._dying_objects = {}
            agent_manager._agent_to_object = dict(restored_agent_to_object)
            agent_manager._object_to_agent = dict(restored_object_to_agent)
            if hasattr(agent_manager, "_agents_finished_this_frame"):
                agent_manager._agents_finished_this_frame = {}

        # Reset episode termination flags so env.step() works after restoring
        # a state snapshot taken before agents terminated.
        if hasattr(self, "dones") and isinstance(self.dones, dict):
            for key in list(self.dones.keys()):
                if key == "__all__" or key in vehicle_states:
                    self.dones[key] = False

        traffic_states = state.get("traffic_states", {})
        if traffic_states:
            try:
                traffic_manager = getattr(self.engine, "traffic_manager", None) if hasattr(self, "engine") and self.engine is not None else None
                traffic_vehicles = getattr(traffic_manager, "traffic_vehicles", None) if traffic_manager is not None else None
                name_to_vehicle = {str(getattr(vehicle, "name", "")): vehicle for vehicle in (traffic_vehicles or [])}
                for vehicle_name, vehicle_state in traffic_states.items():
                    traffic_vehicle = name_to_vehicle.get(str(vehicle_name))
                    if traffic_vehicle is None:
                        continue
                    try:
                        position = np.asarray(vehicle_state.get("position", getattr(traffic_vehicle, "position", (0.0, 0.0))), dtype=np.float64)
                        heading = float(vehicle_state.get("heading", getattr(traffic_vehicle, "heading_theta", 0.0)))
                        velocity = np.asarray(vehicle_state.get("velocity", getattr(traffic_vehicle, "velocity", (0.0, 0.0))), dtype=np.float64)
                        _restore_position(str(vehicle_name), traffic_vehicle, position)
                        _restore_heading(str(vehicle_name), traffic_vehicle, heading)
                        _restore_velocity(str(vehicle_name), traffic_vehicle, velocity)
                    except Exception:
                        continue
            except Exception:
                pass

        restored_actions = {}
        for key, value in state.get("_last_actions", {}).items():
            if isinstance(value, np.ndarray):
                restored_actions[key] = value.copy()
            else:
                restored_actions[key] = np.asarray(value, dtype=np.float32).copy() if value is not None else value
        self._last_actions = restored_actions

        restored_progress = {}
        for key, value in state.get("_last_progress_refs", {}).items():
            if isinstance(value, dict):
                restored_progress[key] = (
                    value.get("lane"),
                    float(value.get("longitudinal", 0.0)),
                    np.asarray(value.get("position", np.zeros((2,), dtype=np.float32)), dtype=np.float32).copy(),
                )
            else:
                lane, longitudinal, position = value
                restored_progress[key] = (lane, float(longitudinal), np.asarray(position, dtype=np.float32).copy())
        self._last_progress_refs = restored_progress
        self._last_info = {key: dict(value) for key, value in state.get("_last_info", {}).items()}

    def get_current_obs(self) -> dict[str, dict]:
        raw_obs = {}
        observations = getattr(self, "observations", {})
        for agent_id in self._agent_ids:
            vehicle = self.agents.get(agent_id)
            if vehicle is None:
                continue
            obs_instance = observations.get(agent_id) if isinstance(observations, dict) else None
            if obs_instance is not None and hasattr(obs_instance, "observe"):
                raw_obs[agent_id] = obs_instance.observe(vehicle)
            else:
                raw_obs[agent_id] = self._last_info.get(agent_id, {}).get("raw_obs", {"obs": np.zeros((1,), dtype=np.float32)})
        return self._augment_observations(raw_obs)

    def _road_half_width(self, agent_id: str) -> float:
        vehicle = self.agents.get(agent_id)
        lane = getattr(vehicle, "lane", None) if vehicle is not None else None
        if lane is None:
            return 3.5
        width = getattr(lane, "width", None)
        if callable(getattr(lane, "width_at", None)):
            try:
                width = float(lane.width_at(lane.local_coordinates(vehicle.position)[0]))
            except Exception:
                width = width
        if width is None:
            width = 7.0
        return max(float(width) * 0.5 + 0.2, 3.5)

    def _surrogate_lateral_offset(self, agent_id: str, world_pose: np.ndarray) -> float:
        vehicle = self.agents.get(agent_id)
        lane = getattr(vehicle, "lane", None) if vehicle is not None else None
        if lane is None:
            return abs(float(world_pose[1]))
        try:
            _, lateral = lane.local_coordinates(world_pose[:2])
            return abs(float(lateral))
        except Exception:
            return abs(float(world_pose[1]))

    def _surrogate_min_gap(self, ego_world_pose: np.ndarray, other_poses: Mapping[str, np.ndarray]) -> float:
        if not other_poses:
            return 100.0
        gaps = [float(np.linalg.norm(ego_world_pose[:2] - other_pose[:2])) for other_pose in other_poses.values()]
        return float(min(gaps))

    def _surrogate_formation_error(
        self,
        agent_id: str,
        ego_world_pose: np.ndarray,
        other_poses: Mapping[str, np.ndarray],
    ) -> float:
        ego_idx = self._agent_ids.index(agent_id)
        errors = []
        for other_id, other_pose in other_poses.items():
            other_idx = self._agent_ids.index(other_id)
            desired_gap = self._desired_center_spacing_m(agent_id, other_id)
            actual_gap = float(np.linalg.norm(ego_world_pose[:2] - other_pose[:2]))
            lateral_gap = abs(float(_world_to_ego(ego_world_pose, other_pose)[1]))
            spacing_error = abs(actual_gap - desired_gap)
            if abs(other_idx - ego_idx) == 1:
                errors.append(spacing_error + 0.2 * lateral_gap)
            else:
                errors.append(0.5 * spacing_error + 0.1 * lateral_gap)
        if not errors:
            return 0.0
        return float(np.mean(errors, dtype=np.float32))
