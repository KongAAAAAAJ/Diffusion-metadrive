from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - fallback for older setups
    import gym  # type: ignore

try:  # pragma: no cover - exercised only in real MetaDrive runtime
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv

    _METADRIVE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - import errors are surfaced at runtime
    FirstPGBlock = None  # type: ignore
    _METADRIVE_IMPORT_ERROR = exc
    BaseMultiEnv = object  # type: ignore


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


class PlatoonEnvConfig:
    def __init__(
        self,
        num_agents: int = 3,
        use_render: bool = False,
        allow_respawn: bool = False,
        use_hybrid_map: bool = True,
        hybrid_map_sequence: str = "SSXCOCSS",
        num_scenarios: int = 1,
        traffic_density: float = 0.04,
        horizon: int = 1000,
        initial_speed_km_h: float = 25.0,
        target_speed_km_h: float = 25.0,
        headway_time_s: float = 0.5,
        vehicle_length_m: float = 5.74,
        formation_error_threshold: float = 2.0,
    ) -> None:
        self.num_agents = int(num_agents)
        self.use_render = bool(use_render)
        self.allow_respawn = bool(allow_respawn)
        self.use_hybrid_map = bool(use_hybrid_map)
        self.hybrid_map_sequence = str(hybrid_map_sequence)
        self.num_scenarios = int(num_scenarios)
        self.traffic_density = float(traffic_density)
        self.horizon = int(horizon)
        self.initial_speed_km_h = float(initial_speed_km_h)
        self.target_speed_km_h = float(target_speed_km_h)
        self.headway_time_s = float(headway_time_s)
        self.vehicle_length_m = float(vehicle_length_m)
        self.formation_error_threshold = float(formation_error_threshold)


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
        return resolved

    @staticmethod
    def _extract_platoon_config(config: Mapping[str, object]) -> dict[str, object]:
        keys = {
            "num_agents",
            "use_render",
            "allow_respawn",
            "use_hybrid_map",
            "hybrid_map_sequence",
            "num_scenarios",
            "traffic_density",
            "horizon",
            "initial_speed_km_h",
            "target_speed_km_h",
            "headway_time_s",
            "vehicle_length_m",
            "formation_error_threshold",
        }
        return {key: config[key] for key in keys if key in config}

    @staticmethod
    def _extract_env_overrides(config: Mapping[str, object]) -> dict[str, object]:
        platoon_keys = {
            "num_agents",
            "use_render",
            "allow_respawn",
            "use_hybrid_map",
            "hybrid_map_sequence",
            "num_scenarios",
            "traffic_density",
            "horizon",
            "initial_speed_km_h",
            "target_speed_km_h",
            "headway_time_s",
            "vehicle_length_m",
            "formation_error_threshold",
        }
        runtime_only_keys = {"enable_idm_lane_change", "hazard_scenario"}
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
        # BaseMultiEnv already defaults: use_hybrid_map=True, hybrid_map_sequence="SSXCOCSS",
        # traffic_mode=TrafficMode.Trigger, agent_observation=LidarStateObservation.
        # Only override what differs from those defaults.
        return {
            "num_agents": self.platoon_config.num_agents,
            "allow_respawn": self.platoon_config.allow_respawn,
            "use_hybrid_map": self.platoon_config.use_hybrid_map,
            "hybrid_map_sequence": self.platoon_config.hybrid_map_sequence,
            "num_scenarios": self.platoon_config.num_scenarios,
            "use_render": self.platoon_config.use_render,
            "traffic_density": self.platoon_config.traffic_density,
            "force_seed_spawn_manager": True,
            "random_spawn_lane_index": False,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,
            "agent_configs": agent_configs,
            "horizon": self.platoon_config.horizon,
            **self._env_overrides,
        }

    @property
    def action_space(self):
        single = gym.spaces.Box(low=-100.0, high=100.0, shape=(8, 3), dtype=np.float32)
        return gym.spaces.Dict({agent_id: single for agent_id in self._agent_ids})

    def reset(self):
        self._metrics.start_episode()
        obs, _ = super().reset()
        if "enable_idm_lane_change" in self._runtime_flags:
            self.config["enable_idm_lane_change"] = bool(self._runtime_flags["enable_idm_lane_change"])
            self.engine.global_config["enable_idm_lane_change"] = bool(self._runtime_flags["enable_idm_lane_change"])
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
            # LidarStateObservation returns np.ndarray; DatasetCollectObservation returns dict.
            if isinstance(agent_obs, dict):
                augmented = dict(agent_obs)
            else:
                augmented = {"obs": agent_obs}
            augmented["formation_relation_state"] = self.get_formation_relation_state(agent_id)
            ret[agent_id] = augmented
        return ret

    def step(self, actions: Dict[str, np.ndarray]):
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
        first_action = np.asarray(next(iter(actions.values())))
        if first_action.shape == (8, 3):
            return "trajectory"
        if first_action.shape == (2,):
            return "low_level"
        raise ValueError(f"Unsupported action shape for PlatoonEnv: {first_action.shape}")

    def low_level_step(self, actions: Dict[str, np.ndarray], control_mode: str = "low_level"):
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
            current_action = np.asarray(actions.get(agent_id, np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(2,)
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
        if ego_idx == 0:
            speed_error = current_speed - target_speed
            accel = -0.35 * speed_error
            return float(np.clip(accel, -1.0, 1.0))

        front_id = self._agent_ids[ego_idx - 1]
        ego_pose = self._agent_pose(agent_id)
        front_pose = self._agent_pose(front_id)
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
