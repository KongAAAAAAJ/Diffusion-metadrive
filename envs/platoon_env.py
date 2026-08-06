from __future__ import annotations

from typing import Dict, Mapping, Optional
import copy
import math

import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - fallback for older setups
    import gym  # type: ignore

try:  # pragma: no cover - exercised only in real MetaDrive runtime
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from envs.diffusion_envs.base_multi_env import BaseMultiEnv, DEFAULT_HYBRID_MAP_CONFIG
    from metadrive.obs.diff_obs.top_down_state_obs_multi_channel import DatasetCollectObservation
    from models.diffusion.transfuser_config import build_transfuser_config
    from models.diffusion.transfuser_features import observation_to_features

    _METADRIVE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - import errors are surfaced at runtime
    RGBCamera = None  # type: ignore
    FirstPGBlock = None  # type: ignore
    _METADRIVE_IMPORT_ERROR = exc
    BaseMultiEnv = object  # type: ignore
    DEFAULT_HYBRID_MAP_CONFIG = ()  # type: ignore
    DatasetCollectObservation = object  # type: ignore


from evaluation.platoon_metrics import PlatoonMetrics
from models.controller.longitudinal_reference import (
    LongitudinalCascadeController,
    LongitudinalTrackingReference,
    trajectory_to_longitudinal_reference,
)
from routes.route_definitions import ROUTE_BY_NAME, get_required_preset, get_route_blocks
from scenarios.definitions import SCENARIO_BY_ID


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
        initial_speed_km_h: float = 80.0,
        target_speed_km_h: float = 90.0,
        headway_time_s: float = 0.5,
        trajectory_dt: float = 0.5,
        vehicle_length_m: float = 5.74,
        formation_error_threshold: float = 2.0,
        observation_mode: str = "lidar_state",
        scenario_id: Optional[str] = None,
        local_route: Optional[str] = None,
        platoon_reward_enabled: bool = True,
        platoon_w_safety: float = 2.0,
        platoon_w_formation: float = 2.0,
        platoon_w_efficiency: float = 0.5,
        platoon_w_comfort: float = 0.1,
        platoon_collision_penalty: float = 10.0,
        platoon_out_of_road_penalty: float = 5.0,
        platoon_d_safe: float = 8.0,
        platoon_d_norm: float = 10.0,
        platoon_delta_s_max: float = 5.0,
        platoon_reward_clip: float = 20.0,
        lqr_lat_q1: float = 1.0,
        lqr_lat_q2: float = 1.0,
        lqr_lat_r: float = 0.1,
        preview_pid_kp: float = 1.6,
        preview_pid_ki: float = 0.05,
        preview_pid_kd: float = 0.12,
        preview_pid_integral_limit: float = 1.0,
        preview_lookahead_time_s: float = 0.6,
        preview_lookahead_min_m: float = 3.0,
        preview_lookahead_max_m: float = 8.0,
        preview_heading_weight: float = 0.5,
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
        self.platoon_reward_enabled = bool(platoon_reward_enabled)
        self.platoon_w_safety = float(platoon_w_safety)
        self.platoon_w_formation = float(platoon_w_formation)
        self.platoon_w_efficiency = float(platoon_w_efficiency)
        self.platoon_w_comfort = float(platoon_w_comfort)
        self.platoon_collision_penalty = float(platoon_collision_penalty)
        self.platoon_out_of_road_penalty = float(platoon_out_of_road_penalty)
        self.platoon_d_safe = float(platoon_d_safe)
        self.platoon_d_norm = float(platoon_d_norm)
        self.platoon_delta_s_max = float(platoon_delta_s_max)
        self.platoon_reward_clip = float(platoon_reward_clip)
        self.lqr_lat_q1 = float(lqr_lat_q1)
        self.lqr_lat_q2 = float(lqr_lat_q2)
        self.lqr_lat_r = float(lqr_lat_r)
        self.preview_pid_kp = float(preview_pid_kp)
        self.preview_pid_ki = float(preview_pid_ki)
        self.preview_pid_kd = float(preview_pid_kd)
        self.preview_pid_integral_limit = float(preview_pid_integral_limit)
        self.preview_lookahead_time_s = float(preview_lookahead_time_s)
        self.preview_lookahead_min_m = float(preview_lookahead_min_m)
        self.preview_lookahead_max_m = float(preview_lookahead_max_m)
        self.preview_heading_weight = float(preview_heading_weight)


class PlatoonEnv(BaseMultiEnv):
    """Phase 1 platoon environment with trajectory-mode and low-level action compatibility."""

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        if _METADRIVE_IMPORT_ERROR is not None:
            raise RuntimeError(
                "PlatoonEnv requires the full MetaDrive runtime. "
                "Please install/run the AGENTS.md environment dependencies before using this environment."
            ) from _METADRIVE_IMPORT_ERROR
        explicit_config_keys = set(dict(config or {}).keys())
        self._explicit_config_keys = frozenset(explicit_config_keys)
        removed_alias_key = "hazard_" + "scenario"
        if removed_alias_key in explicit_config_keys:
            raise ValueError(
                "The legacy scenario alias config key was removed. "
                "Select scenarios with scenario_id/local_route or scenario_ids instead."
            )
        merged = self._merge_config(config)
        merged = self._apply_scenario_definition_defaults(merged, explicit_config_keys)
        self._env_overrides = self._extract_env_overrides(merged)
        self._runtime_flags = self._extract_runtime_flags(merged)
        self.platoon_config = PlatoonEnvConfig(**self._extract_platoon_config(merged))
        self._metrics = PlatoonMetrics(formation_error_threshold=self.platoon_config.formation_error_threshold)
        self._last_info: dict[str, dict] = {}
        self._agent_ids = [f"agent{i}" for i in range(self.platoon_config.num_agents)]
        self._agent_roles: dict[str, str] = {
            agent_id: ("leader" if i == 0 else "follower")
            for i, agent_id in enumerate(self._agent_ids)
        }
        self._last_actions: dict[str, np.ndarray] = {}
        self._last_progress_refs: dict[str, tuple[object, float, np.ndarray]] = {}
        self._multimodal_config = build_transfuser_config("base")
        self._scenario_orchestrator = None
        self._scenario_step_count = 0
        self._pending_low_level_actions: dict[str, np.ndarray] = {}
        self._platoon_reward_cache: Optional[dict[str, object]] = None
        self._pending_step_trajectories: dict[str, np.ndarray] = {}   # (8,3) local ego frame
        self._pending_step_mode_groups: dict[str, str] = {}            # "keep"/"left"/"right"/"stop"
        self._pending_step_all_candidates: dict[str, np.ndarray] = {}  # (num_modes,8,3) local ego frame
        self._pending_step_mode_valid_masks: dict[str, np.ndarray] = {}   # (num_modes,) bool
        self._trajectory_reward_cache: Optional[dict[str, object]] = None
        self._lateral_preview_pid_state: dict[str, tuple[float, float, bool]] = {}
        self._trajectory_longitudinal_controller = LongitudinalCascadeController(
            dt_s=float(merged.get("physics_world_step_size", 0.02))
            * int(merged.get("decision_repeat", 5))
        )
        super().__init__(config=self._build_metadrive_config())
        self._install_platoon_runtime_config()

    def _cfg(self, key: str, default=None):
        config = getattr(self, "config", None)
        if config is not None:
            has_key = False
            try:
                has_key = key in config
            except Exception:
                has_key = False
            try:
                if has_key:
                    return config.get(key)
            except Exception:
                pass
            try:
                if has_key:
                    return config[key]
            except Exception:
                pass
            if hasattr(config, key):
                return getattr(config, key)
        platoon_config = getattr(self, "platoon_config", None)
        if platoon_config is not None and hasattr(platoon_config, key):
            return getattr(platoon_config, key)
        return default

    def _cfg_float(self, key: str, default: float) -> float:
        return float(self._cfg(key, default))

    def _cfg_int(self, key: str, default: int) -> int:
        return int(self._cfg(key, default))

    def _cfg_bool(self, key: str, default: bool = False) -> bool:
        return bool(self._cfg(key, default))

    @property
    def agent_roles(self) -> dict[str, str]:
        """Mapping from agent_id to role: 'leader' (agent0) or 'follower'."""
        return dict(self._agent_roles)

    def get_agent_role(self, agent_id: str) -> str:
        """Return 'leader' or 'follower' for the given agent_id."""
        return self._agent_roles.get(agent_id, "follower")

    def apply_dynamic_roles(self, roles: Mapping[str, str]) -> None:
        """Update per-agent roles for the current step.

        Unknown agent ids are ignored. agent0 remains leader unless explicitly
        omitted from roles, in which case its current role is preserved.
        """
        if not roles:
            return
        for agent_id in self._agent_ids:
            if agent_id in roles:
                role = str(roles[agent_id]).strip().lower()
                self._agent_roles[agent_id] = "follower" if role == "follower" else "leader"

    def _platoon_runtime_config(self) -> dict[str, object]:
        return {
            "initial_speed_km_h": self.platoon_config.initial_speed_km_h,
            "target_speed_km_h": self.platoon_config.target_speed_km_h,
            "headway_time_s": self.platoon_config.headway_time_s,
            "trajectory_dt": self.platoon_config.trajectory_dt,
            "vehicle_length_m": self.platoon_config.vehicle_length_m,
            "formation_error_threshold": self.platoon_config.formation_error_threshold,
            "observation_mode": self.platoon_config.observation_mode,
            "platoon_reward_enabled": self.platoon_config.platoon_reward_enabled,
            "platoon_w_safety": self.platoon_config.platoon_w_safety,
            "platoon_w_formation": self.platoon_config.platoon_w_formation,
            "platoon_w_efficiency": self.platoon_config.platoon_w_efficiency,
            "platoon_w_comfort": self.platoon_config.platoon_w_comfort,
            "platoon_collision_penalty": self.platoon_config.platoon_collision_penalty,
            "platoon_out_of_road_penalty": self.platoon_config.platoon_out_of_road_penalty,
            "platoon_d_safe": self.platoon_config.platoon_d_safe,
            "platoon_d_norm": self.platoon_config.platoon_d_norm,
            "platoon_delta_s_max": self.platoon_config.platoon_delta_s_max,
            "platoon_reward_clip": self.platoon_config.platoon_reward_clip,
            "lqr_lat_q1": self.platoon_config.lqr_lat_q1,
            "lqr_lat_q2": self.platoon_config.lqr_lat_q2,
            "lqr_lat_r": self.platoon_config.lqr_lat_r,
            "preview_pid_kp": self.platoon_config.preview_pid_kp,
            "preview_pid_ki": self.platoon_config.preview_pid_ki,
            "preview_pid_kd": self.platoon_config.preview_pid_kd,
            "preview_pid_integral_limit": self.platoon_config.preview_pid_integral_limit,
            "preview_lookahead_time_s": self.platoon_config.preview_lookahead_time_s,
            "preview_lookahead_min_m": self.platoon_config.preview_lookahead_min_m,
            "preview_lookahead_max_m": self.platoon_config.preview_lookahead_max_m,
            "preview_heading_weight": self.platoon_config.preview_heading_weight,
        }

    def _install_platoon_runtime_config(self) -> None:
        updates = self._platoon_runtime_config()
        self.config.update(updates)
        engine = getattr(self, "engine", None)
        global_config = getattr(engine, "global_config", None)
        if global_config is not None:
            global_config.update(updates)

    def set_runtime_scenario_route(self, scenario_id: str, local_route: str) -> None:
        """Switch the scenario route before reset and invalidate stale destinations."""
        scenario_id = str(scenario_id)
        local_route = str(local_route)
        scenario = SCENARIO_BY_ID.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")
        if local_route not in ROUTE_BY_NAME:
            raise ValueError(f"Unknown local_route: {local_route}")
        if local_route not in scenario.allowed_local_routes:
            raise ValueError(f"Scenario {scenario_id} does not allow local route {local_route}")

        updates: dict[str, object] = {
            "scenario_id": scenario_id,
            "local_route": local_route,
            "ego_main_route_block_ids": get_route_blocks(local_route),
            "route_preset": get_required_preset(local_route),
        }
        if "initial_speed_km_h" not in self._explicit_config_keys:
            scenario_speed = getattr(scenario, "ego_initial_speed_km_h", None)
            if scenario_speed is not None:
                updates["initial_speed_km_h"] = self._sample_scenario_float(scenario_speed)
        if "traffic_density" not in self._explicit_config_keys:
            traffic_density = getattr(scenario, "override_traffic_density", None)
            if traffic_density is not None:
                updates["traffic_density"] = float(traffic_density)
        updates.update(dict(getattr(scenario, "env_overrides", None) or {}))

        self.platoon_config.scenario_id = scenario_id
        self.platoon_config.local_route = local_route
        if "initial_speed_km_h" in updates:
            self.platoon_config.initial_speed_km_h = float(updates["initial_speed_km_h"])
        if "traffic_density" in updates:
            self.platoon_config.traffic_density = float(updates["traffic_density"])
        self.config.update(updates)

        engine = getattr(self, "engine", None)
        global_config = getattr(engine, "global_config", None)
        if global_config is not None:
            global_config.update(updates)

        config_owners = [self.config]
        if global_config is not None and global_config is not self.config:
            config_owners.append(global_config)
        for config_owner in config_owners:
            for agent_config in (config_owner.get("agent_configs", {}) or {}).values():
                agent_config["destination"] = None

    @classmethod
    def _merge_config(cls, config: Optional[Mapping[str, object]]) -> dict:
        defaults = PlatoonEnvConfig().__dict__.copy()
        if config:
            defaults.update(dict(config))
        return defaults

    @staticmethod
    def _apply_scenario_definition_defaults(
        config: Mapping[str, object],
        explicit_config_keys: set[str],
    ) -> dict[str, object]:
        resolved = dict(config)
        scenario_id = resolved.get("scenario_id")
        if not scenario_id:
            return resolved
        scenario = SCENARIO_BY_ID.get(str(scenario_id))
        if scenario is None:
            return resolved
        scenario_initial_speed = getattr(scenario, "ego_initial_speed_km_h", None)
        if scenario_initial_speed is not None and "initial_speed_km_h" not in explicit_config_keys:
            resolved["initial_speed_km_h"] = PlatoonEnv._float_or_range_midpoint(scenario_initial_speed)
        scenario_traffic_density = getattr(scenario, "override_traffic_density", None)
        if scenario_traffic_density is not None and "traffic_density" not in explicit_config_keys:
            resolved["traffic_density"] = float(scenario_traffic_density)
        resolved.update(dict(getattr(scenario, "env_overrides", None) or {}))
        return resolved

    def _sample_scenario_float(self, value) -> float:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            low, high = value
            rng = self._scenario_rng()
            if rng is not None and hasattr(rng, "uniform"):
                return float(rng.uniform(float(low), float(high)))
            return float((float(low) + float(high)) * 0.5)
        return float(value)

    def _scenario_rng(self):
        engine = getattr(self, "engine", None)
        traffic_manager = getattr(engine, "traffic_manager", None)
        rng = getattr(traffic_manager, "np_random", None)
        if rng is not None:
            return rng
        rng = getattr(self, "np_random", None)
        if rng is not None:
            return rng
        return getattr(engine, "np_random", None)

    @staticmethod
    def _float_or_range_midpoint(value) -> float:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            low, high = value
            return float((float(low) + float(high)) * 0.5)
        return float(value)

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
            "platoon_reward_enabled",
            "platoon_w_safety",
            "platoon_w_formation",
            "platoon_w_efficiency",
            "platoon_w_comfort",
            "platoon_collision_penalty",
            "platoon_out_of_road_penalty",
            "platoon_d_safe",
            "platoon_d_norm",
            "platoon_delta_s_max",
            "platoon_reward_clip",
            "lqr_lat_q1",
            "lqr_lat_q2",
            "lqr_lat_r",
            "preview_pid_kp",
            "preview_pid_ki",
            "preview_pid_kd",
            "preview_pid_integral_limit",
            "preview_lookahead_time_s",
            "preview_lookahead_min_m",
            "preview_lookahead_max_m",
            "preview_heading_weight",
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
            "platoon_reward_enabled",
            "platoon_w_safety",
            "platoon_w_formation",
            "platoon_w_efficiency",
            "platoon_w_comfort",
            "platoon_collision_penalty",
            "platoon_out_of_road_penalty",
            "platoon_d_safe",
            "platoon_d_norm",
            "platoon_delta_s_max",
            "platoon_reward_clip",
            "lqr_lat_q1",
            "lqr_lat_q2",
            "lqr_lat_r",
            "preview_pid_kp",
            "preview_pid_ki",
            "preview_pid_kd",
            "preview_pid_integral_limit",
            "preview_lookahead_time_s",
            "preview_lookahead_min_m",
            "preview_lookahead_max_m",
            "preview_heading_weight",
            # scenario_id / local_route are intentionally excluded here so they pass
            # through to the MetaDrive config via _build_metadrive_config explicitly.
        }
        runtime_only_keys = {"enable_idm_lane_change", "scenario_id", "local_route"}
        return {key: value for key, value in config.items() if key not in platoon_keys and key not in runtime_only_keys}

    @staticmethod
    def _extract_runtime_flags(config: Mapping[str, object]) -> dict[str, object]:
        runtime_only_keys = {"enable_idm_lane_change"}
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
            "platoon_fixed_route_spawn": True,
            "platoon_spawn_gap_m": float(gap_m),
            "platoon_spawn_tail_buffer_m": 6.0,
            "platoon_spawn_front_buffer_m": 8.0,
            "initial_speed_km_h": self.platoon_config.initial_speed_km_h,
            "horizon": self.platoon_config.horizon,
            "traffic_target_speed": (60.0, 90.0),  # km/h
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
        if not self._cfg("local_route", None):
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
        fixed_agent0_config = None
        fixed_lane_index = ()
        if self._cfg_bool("platoon_fixed_route_spawn", False):
            fixed_agent0_config = (
                getattr(getattr(self, "engine", None), "global_config", {})
                .get("agent_configs", {})
                .get("agent0")
            )
            if fixed_agent0_config is not None:
                fixed_lane_index = tuple(fixed_agent0_config.get("spawn_lane_index", ()))
        road = route_roads[0]
        road_start = road.start_node
        road_end = road.end_node
        if len(fixed_lane_index) == 3:
            road_start, road_end = fixed_lane_index[:2]
        try:
            lanes = current_map.road_network.graph[road_start][road_end]
        except (KeyError, AttributeError, TypeError):
            return
        if not lanes:
            return

        speed_m_s = self._cfg_float("initial_speed_km_h", 25.0) / 3.6
        gap_m = self._desired_center_spacing_m()
        lane_idx = 0
        if fixed_agent0_config is not None:
            if len(fixed_lane_index) == 3 and tuple(fixed_lane_index[:2]) == (road_start, road_end):
                lane_idx = int(fixed_lane_index[2])
        lane_idx = max(0, min(lane_idx, len(lanes) - 1))
        lane = lanes[lane_idx]
        if fixed_agent0_config is not None:
            lead_long = float(fixed_agent0_config.get("spawn_longitude", self._select_route_spawn_lead_long(lane, gap_m)))
        else:
            lead_long = self._select_route_spawn_lead_long(lane, gap_m)
        for i, agent_id in enumerate(self._agent_ids):
            vehicle = self.agents.get(agent_id)
            if vehicle is None:
                continue
            long = self._route_spawn_vehicle_longitude(lead_long, gap_m, i)
            pos = lane.position(long, 0.0)
            heading = lane.heading_theta_at(long)
            vehicle.set_position(pos)
            vehicle.set_heading_theta(heading)
            # Re-apply initial speed along lane heading
            vehicle.set_velocity(
                (speed_m_s * np.cos(heading), speed_m_s * np.sin(heading)),
                in_local_frame=False,
            )
            # Refresh navigation so vehicle.lane / current_ref_lanes reflect the
            # new physical position on the route road (c2, s_main0, etc.) rather
            # than the original NODE_1/NODE_2 spawn lane.  Without this,
            # IDMPolicy.move_to_next_road() sets routing_target_lane to the stale
            # NODE_1→NODE_2 lane and steering_control() computes headings against
            # that wrong lane at the teleported position → vehicles steer in the
            # opposite direction on curves (observed as left-turn on S6 right curve).
            nav = getattr(vehicle, "navigation", None)
            if nav is not None:
                try:
                    nav.update_localization(vehicle)
                except Exception:
                    pass

    def _route_spawn_min_tail_buffer_m(self) -> float:
        return self._cfg_float("platoon_spawn_tail_buffer_m", 6.0)

    def _route_spawn_min_front_buffer_m(self) -> float:
        return self._cfg_float("platoon_spawn_front_buffer_m", 8.0)

    @staticmethod
    def _route_spawn_reference_lead_long_m() -> float:
        return 20.0

    def _route_spawn_lead_long_bounds(self, lane_length: float, gap_m: float) -> tuple[float, float]:
        platoon_span = gap_m * max(len(self._agent_ids) - 1, 0)
        min_lead = platoon_span + self._route_spawn_min_tail_buffer_m()
        max_lead = max(float(lane_length) - self._route_spawn_min_front_buffer_m(), min_lead)
        return float(min_lead), float(max_lead)

    def _route_spawn_vehicle_longitude(self, lead_long: float, gap_m: float, vehicle_index: int) -> float:
        platoon_long = float(lead_long) - float(vehicle_index) * float(gap_m)
        return max(platoon_long, 2.0)

    def _route_spawn_clearance_score(self, lane, lead_long: float, gap_m: float) -> float:
        traffic_manager = getattr(getattr(self, "engine", None), "traffic_manager", None)
        traffic_vehicles = list(getattr(traffic_manager, "_traffic_vehicles", []) or [])
        if not traffic_vehicles:
            return 0.0

        score = float("inf")
        for i, _ in enumerate(self._agent_ids):
            long = self._route_spawn_vehicle_longitude(lead_long, gap_m, i)
            pos = np.asarray(lane.position(long, 0.0), dtype=np.float32)
            for vehicle in traffic_vehicles:
                traffic_pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0)), dtype=np.float32)
                dist = float(np.linalg.norm(pos - traffic_pos))
                if dist < score:
                    score = dist
        if score == float("inf"):
            return 0.0
        return score

    def _select_route_spawn_lead_long(self, lane, gap_m: float) -> float:
        lane_length = float(getattr(lane, "length", 0.0))
        min_lead, max_lead = self._route_spawn_lead_long_bounds(lane_length, gap_m)
        reference = self._route_spawn_reference_lead_long_m()
        base_lead = float(np.clip(reference, min_lead, max_lead))
        if self._cfg_bool("platoon_fixed_route_spawn", False):
            return float(min(min_lead, max_lead))
        if max_lead <= min_lead + 1e-3:
            return base_lead

        candidates = {base_lead, min_lead, max_lead}
        for value in np.linspace(min_lead, max_lead, num=7):
            candidates.add(float(value))

        best_lead = base_lead
        best_score = None
        for candidate in sorted(candidates):
            clearance = self._route_spawn_clearance_score(lane, candidate, gap_m)
            # Prefer higher-clearance placements while staying close to the nominal
            # start point when several candidates are similarly safe.
            score = clearance - 0.15 * abs(candidate - base_lead)
            if best_score is None or score > best_score:
                best_score = score
                best_lead = candidate
        return float(best_lead)

    def _setup_scenario_orchestrator(self) -> None:
        """Initialise PlatoonScenarioOrchestrator when scenario_id + local_route are both set."""
        self._scenario_orchestrator = None
        self._scenario_step_count = 0
        scenario_id = self.config.scenario_id
        local_route = self.config.local_route
        
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

    def reset(self, seed: Optional[int] = None):
        self._metrics.start_episode()
        self._platoon_reward_cache = None
        self._pending_low_level_actions = {}
        obs, _ = super().reset(seed=seed)

        if "enable_idm_lane_change" in self._runtime_flags:
            self.config["enable_idm_lane_change"] = bool(self._runtime_flags["enable_idm_lane_change"])
            self.engine.global_config["enable_idm_lane_change"] = bool(self._runtime_flags["enable_idm_lane_change"])

        # Teleport to route segment if local_route is set
        self._reposition_platoon_on_route()
        self._clear_traffic_vehicles_near_platoon()

        # Initialise scenario orchestrator (hazard injection)
        self._setup_scenario_orchestrator()

        self._last_actions = {
            agent_id: np.zeros((2,), dtype=np.float32) for agent_id in self._agent_ids
        }
        self._last_progress_refs = {
            agent_id: self._capture_progress_reference(agent_id) for agent_id in self._agent_ids
        }
        self._last_info = {}
        self._platoon_reward_cache = None
        self._trajectory_reward_cache = None
        self._pending_low_level_actions = {}
        self._pending_step_trajectories = {}
        self._pending_step_mode_groups = {}
        self._pending_step_all_candidates = {}
        self._pending_step_mode_valid_masks = {}
        self._lateral_preview_pid_state = {}
        self._trajectory_longitudinal_controller.reset()

        return self._augment_observations(obs)

    def _augment_observations(self, obs: Mapping[str, object]) -> dict[str, dict]:
        ret = {}
        for agent_id, agent_obs in obs.items():
            ret[agent_id] = self._format_agent_observation(agent_id, agent_obs)
        return ret

    def _format_agent_observation(self, agent_id: str, agent_obs: object) -> dict[str, np.ndarray]:
        if self._cfg("observation_mode", "lidar_state") == "multimodal":
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
            self._pending_low_level_actions = {
                agent_id: (
                    np.asarray(action, dtype=np.float32).reshape(2,)
                    if np.asarray(action).shape == (2,)
                    else np.zeros((2,), dtype=np.float32)
                )
                for agent_id, action in actions.items()
            }
            self._platoon_reward_cache = None
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
            # Store trajectory (local ego frame, 8×3) for road_topo_reward computation.
            self._pending_step_trajectories = {
                agent_id: np.asarray(action, dtype=np.float32)
                for agent_id, action in actions.items()
            }
            low_level_actions = {
                agent_id: self.trajectory_to_control(agent_id, np.asarray(action, dtype=np.float32))
                for agent_id, action in actions.items()
            }
        else:
            # Do NOT clear _pending_step_trajectories here: ModeSelectionSB3Env may have
            # pre-populated it (with (8,3) trajectories) before calling base_env.step().
            # Only reset if nothing was externally provided.
            if not self._pending_step_trajectories:
                self._pending_step_trajectories = {}
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
        self._pending_low_level_actions = {
            agent_id: np.asarray(action, dtype=np.float32).reshape(2,)
            for agent_id, action in actions.items()
        }
        self._platoon_reward_cache = None
        self._trajectory_reward_cache = None
        # Tick ScenarioOrchestrator before the physics step (mirrors collect_expert behaviour)
        if getattr(self, "_scenario_orchestrator", None) is not None:
            lead_agent_id = self._agent_ids[0]
            self._scenario_step_count = getattr(self, "_scenario_step_count", 0) + 1
            self._scenario_orchestrator.before_step(self, lead_agent_id, self._scenario_step_count)

        obs, reward, terminated, truncated, info = super().step(actions)
        # self._clear_traffic_vehicles_near_platoon()  # 只在reset时清除一次，避免step中频繁操作影响性能
        info = self._build_info_dict(control_mode, actions=actions, base_info=info)
        terminated, truncated = self._enforce_platoon_episode_end(terminated, truncated, info)

        obs = self._augment_observations(obs)
        self._metrics.update(info)
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            self._metrics.end_episode()
        if info:
            self._last_info = info
        return obs, reward, terminated, truncated, info

    def reward_function(self, vehicle_id: str):
        """Team-level platoon reward shared by all agents.

        BaseEnv calls this once per active agent in a step. The first call builds
        and caches the team reward so progress and comfort terms are consumed only
        once; later calls return the same scalar with per-agent diagnostics.
        """
        if not self._cfg_bool("platoon_reward_enabled", True):
            return super().reward_function(vehicle_id)
        platoon_cache = self._get_platoon_reward_cache()  # 状态的platoon reward
        traj_cache = self._get_trajectory_reward_cache()  # 动作轨迹的road topology reward
        form_cache = self._build_traj_form_reward()  # 动作轨迹的form reward
        w_topo = self._cfg_float("platoon_w_topo_reward", 0.0)  # 3.0
        w_form = self._cfg_float("platoon_w_traj_form_reward", 0.0)  
        reward = (float(platoon_cache.get("reward", 0.0))
                  + w_topo * float(traj_cache.get("reward", 0.0))
                  + w_form * float(form_cache.get("reward", 0.0)))
        
        # # 打印platoon_cache和traj_cache的内容
        # print(f"Platoon reward: {float(platoon_cache.get('reward', 0.0))}")
        # print(f"Step Trajectory reward: {w_topo * float(traj_cache.get('reward', 0.0))}")
        # # 打印每个agent的reward_topo
        # _pa = traj_cache.get("per_agent", {})
        # print(f"agent0 topo reward: {_pa.get('agent0', {}).get('road_topo_reward', 'N/A')}")
        # print(f"agent1 topo reward: {_pa.get('agent1', {}).get('road_topo_reward', 'N/A')}")
        # print(f"agent2 topo reward: {_pa.get('agent2', {}).get('road_topo_reward', 'N/A')}")


        per_agent_platoon = platoon_cache.get("per_agent", {})
        per_agent_traj = traj_cache.get("per_agent", {})
        per_agent_form = form_cache.get("per_agent", {})
        info = dict(per_agent_platoon.get(vehicle_id, {})) if isinstance(per_agent_platoon, Mapping) else {}
        info.update(per_agent_traj.get(vehicle_id, {}) if isinstance(per_agent_traj, Mapping) else {})
        info.update(per_agent_form.get(vehicle_id, {}) if isinstance(per_agent_form, Mapping) else {})
        info["reward_topo"] = float(traj_cache.get("reward", 0.0))
        info["reward_traj_form"] = float(form_cache.get("reward", 0.0))
        return reward, info

    def _get_platoon_reward_cache(self) -> dict[str, object]:
        cache = getattr(self, "_platoon_reward_cache", None)
        if cache is None:
            cache = self._build_platoon_reward_cache()
            self._platoon_reward_cache = cache
        return cache

    def _get_trajectory_reward_cache(self) -> dict[str, object]:
        cache = getattr(self, "_trajectory_reward_cache", None)
        if cache is None:
            cache = self._build_trajectory_reward_cache()
            self._trajectory_reward_cache = cache
        return cache

    # ------------------------------------------------------------------
    # Formation consistency reward (semantic mode-group comparison)
    # ------------------------------------------------------------------

    @staticmethod
    def _lateral_group(mode_group: str) -> str:
        """Map mode group to lateral destination: 'stay' | 'left' | 'right'."""
        if mode_group in ("left",):
            return "left"
        if mode_group in ("right",):
            return "right"
        return "stay"   # keep / stop / unknown → stays in current lane

    def _build_traj_form_reward(self) -> dict[str, object]:
        """Evaluate whether consecutive vehicles choose mode trajectories that
        keep them in the same lane, preserving platoon formation.

        For each consecutive pair (agent[i-1], agent[i]):
          same lateral destination → 1.0, different → 0.0
        Team reward = mean over all follower pairs, ∈ [0, 1].

        Per-agent: each follower gets its own pair score; the lead (agent0)
        gets the team mean (it contributes to all pairs indirectly).
        """
        agent_ids = [aid for aid in self._agent_ids if aid in self.agents]
        if len(agent_ids) < 2:
            return {"reward": 0.0, "per_agent": {
                aid: {"traj_form_reward": 0.0, "traj_form_pair_match": None}
                for aid in agent_ids
            }}

        mode_groups = getattr(self, "_pending_step_mode_groups", {}) or {}

        pair_scores: list[float] = []
        follower_scores: dict[str, float] = {}

        for i in range(1, len(agent_ids)):
            lead_id = agent_ids[i - 1]
            follower_id = agent_ids[i]
            lead_lat = self._lateral_group(mode_groups.get(lead_id, "keep"))
            follower_lat = self._lateral_group(mode_groups.get(follower_id, "keep"))
            match = float(lead_lat == follower_lat)
            pair_scores.append(match)
            follower_scores[follower_id] = match

        team_reward = float(np.mean(pair_scores)) if pair_scores else 0.0

        per_agent: dict[str, dict] = {}
        for aid in agent_ids:
            if aid == agent_ids[0]:
                # Lead: no pair above it; report team mean as its formation reward
                per_agent[aid] = {"traj_form_reward": team_reward, "traj_form_pair_match": None}
            else:
                score = follower_scores.get(aid, 0.0)
                per_agent[aid] = {"traj_form_reward": score, "traj_form_pair_match": bool(score > 0.5)}

        return {"reward": team_reward, "per_agent": per_agent}

    # ------------------------------------------------------------------
    # Trajectory quality reward
    # ------------------------------------------------------------------

    @staticmethod
    def _get_adjacent_lane(vehicle, side: str):
        """Return the adjacent lane object ('left' or 'right') or None."""
        lane = getattr(vehicle, "lane", None)
        if lane is None:
            return None
        lane_idx = getattr(lane, "index", None)
        if lane_idx is None or len(lane_idx) < 3:
            return None
        try:
            engine = getattr(vehicle, "engine", None)
            current_map = getattr(engine, "current_map", None) if engine is not None else None
            road_network = getattr(current_map, "road_network", None) if current_map is not None else None
            if road_network is None:
                return None
            lanes_in_road = road_network.graph[lane_idx[0]][lane_idx[1]]
            current_idx = int(lane_idx[2])
            # MetaDrive convention: index 0 = leftmost, higher = rightmore.
            # 'left' → lower index; 'right' → higher index.
            target_idx = current_idx - 1 if side == "left" else current_idx + 1
            if 0 <= target_idx < len(lanes_in_road):
                return lanes_in_road[target_idx]
        except Exception:
            pass
        return None

    def _build_trajectory_reward_cache(self) -> dict[str, object]:
        """Evaluate how closely each selected trajectory follows its target lane centre.

        road_topo_reward[agent] = max(0, 1 - avg_lateral_deviation / norm_m)
        norm_m: keep_lane=2 m, lane_change=5 m (configurable)
        Team reward = mean over active agents, ∈ [0, 1].
        """
        agent_ids = [aid for aid in self._agent_ids if aid in self.agents]
        if not agent_ids:
            return {"reward": 0.0, "per_agent": {}}

        _norm_keep = max(self._cfg_float("platoon_topo_reward_norm_keep_m", 4.0), 1e-3)
        _norm_lc   = max(self._cfg_float("platoon_topo_reward_norm_lc_m",   12.0), 1e-3)

        topo_scores: dict[str, float] = {}
        per_agent: dict[str, dict] = {}

        trajs = getattr(self, "_pending_step_trajectories", {}) or {}
        mode_groups = getattr(self, "_pending_step_mode_groups", {}) or {}
        all_cands = getattr(self, "_pending_step_all_candidates", {}) or {}
        valid_masks = getattr(self, "_pending_step_mode_valid_masks", {}) or {}

        neutral_score = 0

        for agent_id in agent_ids:
            traj = trajs.get(agent_id)
            if traj is None:
                topo_scores[agent_id] = neutral_score   # no trajectory info: neutral score
                per_agent[agent_id] = {"road_topo_reward": neutral_score, "topo_avg_deviation_m": float("nan")}
                continue

            traj = np.asarray(traj, dtype=np.float32)
            if traj.ndim != 2 or traj.shape[0] < 1 or traj.shape[1] < 2:
                topo_scores[agent_id] = neutral_score
                per_agent[agent_id] = {"road_topo_reward": neutral_score, "topo_avg_deviation_m": float("nan")}
                #!
                print(f"Agent {agent_id}: Invalid trajectory shape {traj.shape}, assigning neutral topo score.")
                continue

            vehicle = self.agents.get(agent_id)
            if vehicle is None:
                topo_scores[agent_id] = neutral_score
                per_agent[agent_id] = {"road_topo_reward": neutral_score, "topo_avg_deviation_m": float("nan")}
                #!
                print(f"Agent {agent_id}: Vehicle not found, assigning neutral topo score.")
                continue

            mode_group = mode_groups.get(agent_id, "keep")

            # Determine target lane for deviation measurement.
            if mode_group in ("keep", "stop"):
                target_lane = getattr(vehicle, "lane", None)
            else:
                target_lane = self._get_adjacent_lane(vehicle, mode_group)
                if target_lane is None:
                    # Adjacent lane not found → fall back to current lane.
                    target_lane = getattr(vehicle, "lane", None)

            if target_lane is None:
                topo_scores[agent_id] = neutral_score
                per_agent[agent_id] = {"road_topo_reward": neutral_score, "topo_avg_deviation_m": float("nan")}
                #!
                print(f"Agent {agent_id}: Target lane not found, assigning neutral topo score.")
                continue

            # Convert trajectory from ego-local frame to world frame.
            ego_pos = np.asarray(vehicle.position[:2], dtype=np.float64)
            ego_hdg = float(vehicle.heading_theta)
            cos_h, sin_h = float(np.cos(ego_hdg)), float(np.sin(ego_hdg))
            deviations: list[float] = []
            for pt in traj:
                lx, ly = float(pt[0]), float(pt[1])
                wx = ego_pos[0] + cos_h * lx - sin_h * ly
                wy = ego_pos[1] + sin_h * lx + cos_h * ly
                try:
                    _, lat = target_lane.local_coordinates([wx, wy])
                    deviations.append(abs(float(lat)))
                except Exception:
                    pass

            if not deviations:
                topo_scores[agent_id] = neutral_score
                per_agent[agent_id] = {"road_topo_reward": neutral_score, "topo_avg_deviation_m": float("nan")}
                #!
                print(f"Agent {agent_id}: No valid trajectory points for deviation calculation, assigning neutral topo score.")
                continue

            avg_dev = float(np.mean(deviations))
            norm_m = _norm_keep if mode_group in ("keep", "stop") else _norm_lc
            score = float(max(0.0, 1.0 - avg_dev / norm_m))
            topo_scores[agent_id] = score
            per_agent[agent_id] = {
                "road_topo_reward": score,
                "topo_avg_deviation_m": avg_dev,
                "topo_mode_group": mode_group,
            }

            # ── Debug: compute topo reward for every candidate trajectory ────────────────
            if agent_id in all_cands and vehicle is not None:
                cand_all = np.asarray(all_cands[agent_id], dtype=np.float32)  # (M, 8, 3)
                vmask = valid_masks.get(agent_id)  # (M,) bool or None
                cand_rewards: list[float] = []
                for m_i in range(cand_all.shape[0]):
                    # Skip invalid modes — they won't be selected so comparing is meaningless
                    if vmask is not None and not bool(vmask[m_i]):
                        cand_rewards.append(float("nan"))
                        continue
                    m_devs: list[float] = []
                    for pt in cand_all[m_i]:
                        lx, ly = float(pt[0]), float(pt[1])
                        wx = ego_pos[0] + cos_h * lx - sin_h * ly
                        wy = ego_pos[1] + sin_h * lx + cos_h * ly
                        try:
                            _, lat = target_lane.local_coordinates([wx, wy])
                            m_devs.append(abs(float(lat)))
                        except Exception:
                            pass
                    if m_devs:
                        m_avg = float(np.mean(m_devs))
                        m_score = float(max(0.0, 1.0 - m_avg / norm_m))
                    else:
                        m_score = 0.0
                    cand_rewards.append(m_score)
                # Rank among valid modes only: 0 = actor selected the best-topo candidate
                valid_pairs = [(i, r) for i, r in enumerate(cand_rewards) if not (r != r)]  # skip nan
                sorted_valid = sorted(valid_pairs, key=lambda t: -t[1])
                selected_rank = next(
                    (rank for rank, (i, r) in enumerate(sorted_valid)
                     if abs(r - score) < 1e-4),
                    -1,
                )
                per_agent[agent_id]["debug_candidate_topo_rewards"] = cand_rewards
                per_agent[agent_id]["debug_selected_topo_rank"] = selected_rank

        _reward = float(np.mean(list(topo_scores.values()))) if topo_scores else 0.0
        return {"reward": _reward, "per_agent": per_agent}

    # ------------------------------------------------------------------
    # Reward sub-functions (one per component)
    # ------------------------------------------------------------------

    @staticmethod
    def _r_safety(crash_count: int, out_count: int, n: int) -> float:
        """Safety: collision penalty + out-of-road penalty, normalised by N."""
        n = max(n, 1)
        return -100.0 * crash_count / n - 50.0 * out_count / n
        # return - 50.0 * out_count / n  # !去掉碰撞惩罚，专注于超出道路的惩罚

    @staticmethod
    def _r_spacing(d: float, d_exp: float) -> float:
        """Spacing reward for one follower (bumper-to-bumper gap d, target d_exp).

        d ∈ (d_exp-5, d_exp+5]  → r = 5 / |d - d_exp|   (peaks near target)
        d ∈ (15, 30]            → r = (30 - d) / 15      (decays to 0 at 30 m)
        d ∈ (0,  5)             → r = -10 / d + 3        (strongly negative)
        else                    → r = 0
        """
        diff = abs(d - d_exp)
        if diff <= 5.0:
            return 5.0 / max(diff, 0.5)        # cap at 10.0 when diff→0
        if 15.0 < d <= 30.0:
            return (30.0 - d) / 15.0
        if 0.0 < d < 5.0:
            return -10.0 / max(d, 0.1) + 3.0
        return 0.0

    @staticmethod
    def _r_speed(v: float, v_target: float) -> float:
        """Speed reward for one vehicle (v and v_target in km/h).

        |v - v_target| ≤ 5           → r = 5 / |v - v_target|  (peaks at v_target)
        15 < v < v_target - 5  (slow) → r = v/10 - (v_target-15)/10
        v <= 15                (very slow) → r = v - (v_target-15)/10 - 13.5
        v > v_target + 5       (fast) → r = -v/(v_target+5) + 2
        else                          → r = 0
        """
        diff = abs(v - v_target)
        if diff <= 5.0:
            return 5.0 / max(diff, 0.5)        # cap at 10.0 when diff→0
        if 15.0 < v < v_target - 5.0:
            return v / 10.0 - (v_target - 15.0) / 10.0
        if v <= 15.0:
            return v - (v_target - 15.0) / 10.0 - 13.5
        if v > v_target + 5.0:
            return -v / max(v_target + 5.0, 1e-6) + 2.0
        return 0.0

    @staticmethod
    def _r_progress(s: float, s_max: float) -> float:
        """Progress (proximity) reward: 15 * s / s_max."""
        return 1.0 * s / max(s_max, 1e-6)

    @staticmethod
    def _r_comfort(j: float, delta: float) -> float:
        """Comfort reward based on throttle jerk j and steering change delta.

        r = (20 - |j|) / 20 + (10 - |delta|) / 10
        """
        return (20.0 - abs(j)) / 20.0 + (10.0 - abs(delta)) / 10.0

    # ------------------------------------------------------------------
    # Gap helper (bumper-to-bumper distance between two consecutive agents)
    # ------------------------------------------------------------------

    def _bumper_gap(self, ego_id: str, front_id: str) -> float:
        """Bumper-to-bumper longitudinal gap (m) between ego and the vehicle ahead."""
        ego_pose = self._agent_pose(ego_id)
        front_pose = self._agent_pose(front_id)
        if ego_pose is None or front_pose is None:
            return float("inf")
        center_dist = float(np.linalg.norm(front_pose[:2] - ego_pose[:2]))
        half_lengths = 0.5 * (self._vehicle_length_m(ego_id) + self._vehicle_length_m(front_id))
        return max(center_dist - half_lengths, 0.0)

    # ------------------------------------------------------------------
    # Main cache builder
    # ------------------------------------------------------------------

    def _build_platoon_reward_cache(self) -> dict[str, object]:
        agent_ids = [aid for aid in self._agent_ids if aid in self.agents]
        if not agent_ids:
            return {"reward": 0.0, "per_agent": {}}

        delta_s_max = max(self._cfg_float("platoon_delta_s_max", 5.0), 1e-6)
        d_exp       = max(self._cfg_float("platoon_d_exp", 10.0), 1e-6)
        target_speed_global = max(self._cfg_float("target_speed_km_h", 25.0), 1e-6)
        lead_id     = self._agent_ids[0]
        v_lead      = self._agent_speed_km_h(lead_id) if lead_id in self.agents else target_speed_global

        # ── per-agent data collection ────────────────────────────────
        pending_actions = getattr(self, "_pending_low_level_actions", {}) or {}
        progress_by_agent: dict[str, float] = {}
        gap_by_agent:      dict[str, float] = {}   # bumper-to-bumper to front (followers only)
        speed_by_agent:    dict[str, float] = {}
        jerk_by_agent:     dict[str, float] = {}
        delta_by_agent:    dict[str, float] = {}
        crash_by_agent:    dict[str, bool]  = {}
        out_by_agent:      dict[str, bool]  = {}

        for agent_id in agent_ids:
            cur = np.asarray(
                pending_actions.get(agent_id, np.zeros(2, dtype=np.float32)), dtype=np.float32
            ).reshape(2)
            prev = self._last_actions.get(agent_id, np.zeros(2, dtype=np.float32))

            progress_by_agent[agent_id] = float(self._compute_progress(agent_id))
            speed_by_agent[agent_id]    = float(self._agent_speed_km_h(agent_id))
            delta_by_agent[agent_id]    = float(cur[0] - prev[0])
            jerk_by_agent[agent_id]     = float(cur[1] - prev[1])
            crash_by_agent[agent_id]    = self._agent_has_terminal_collision(agent_id)
            out_by_agent[agent_id]      = self._agent_is_out_of_road(agent_id)

            ego_idx = self._agent_ids.index(agent_id)
            if ego_idx > 0:
                front_id = self._agent_ids[ego_idx - 1]
                gap_by_agent[agent_id] = self._bumper_gap(agent_id, front_id)

        num_agents  = max(len(agent_ids), 1)
        crash_count = int(sum(crash_by_agent.values()))
        out_count   = int(sum(out_by_agent.values()))

        # ── 1. Safety ────────────────────────────────────────────────
        reward_safety = self._r_safety(crash_count, out_count, num_agents)

        # ── 2. Spacing (followers only, averaged) ────────────────────
        spacing_scores = [
            self._r_spacing(gap_by_agent[aid], d_exp)
            for aid in agent_ids
            if aid in gap_by_agent
        ]
        reward_spacing = float(np.mean(spacing_scores)) if spacing_scores else 0.0

        # ── 3. Speed (leader tracks target_speed; followers track v_lead) ──
        def _v_target(aid: str) -> float:
            return target_speed_global if self._agent_ids.index(aid) == 0 else v_lead

        speed_scores = [self._r_speed(speed_by_agent[aid], _v_target(aid)) for aid in agent_ids]
        reward_speed = float(np.mean(speed_scores)) if speed_scores else 0.0

        # ── 4. Progress (proximity) ──────────────────────────────────
        progress_scores = [self._r_progress(progress_by_agent[aid], delta_s_max) for aid in agent_ids]
        reward_progress = float(np.mean(progress_scores)) if progress_scores else 0.0

        # ── 5. Comfort ───────────────────────────────────────────────
        comfort_scores = [
            self._r_comfort(jerk_by_agent[aid], delta_by_agent[aid]) for aid in agent_ids
        ]
        reward_comfort = float(np.mean(comfort_scores)) if comfort_scores else 0.0
        reward_comfort = 0.0  # !暂时去掉舒适度奖励，专注于安全、车距、速度和进度

        reward = reward_safety + reward_spacing + reward_speed + reward_progress + reward_comfort
        clip = self._cfg_float("platoon_reward_clip", 150.0)  
        reward = float(np.clip(reward, -clip, clip)) if clip > 0.0 else float(reward)
        # # 归一化奖励到 [-1, 1] 区间（假设最大绝对值不超过 clip）
        # reward = float(reward / clip) if clip > 0.0 else float(reward)

        per_agent: dict[str, dict[str, object]] = {}
        for agent_id in agent_ids:
            per_agent[agent_id] = {
                "platoon_reward":    reward,
                "reward_safety":     float(reward_safety),
                "reward_spacing":    float(reward_spacing),
                "reward_speed":      float(reward_speed),
                "reward_progress":   float(reward_progress),
                "reward_comfort":    float(reward_comfort),
                "team_crash_count":  int(crash_count),
                "team_out_count":    int(out_count),
                "bumper_gap":        float(gap_by_agent.get(agent_id, float("inf"))),
                "speed_km_h":        float(speed_by_agent[agent_id]),
                "progress":          float(progress_by_agent[agent_id]),
                "jerk":              float(jerk_by_agent[agent_id]),
                "delta_steering":    float(delta_by_agent[agent_id]),
            }
        return {"reward": reward, "per_agent": per_agent}

    def _agent_has_terminal_collision(self, agent_id: str) -> bool:
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            return False
        return bool(
            getattr(vehicle, "crash_vehicle", False)
            or getattr(vehicle, "crash_object", False)
            or getattr(vehicle, "crash_building", False)
            or getattr(vehicle, "crash_human", False)
        )

    def _agent_is_out_of_road(self, agent_id: str) -> bool:
        vehicle = self.agents.get(agent_id)
        if vehicle is None:
            return False
        try:
            return bool(self._is_out_of_road(vehicle))
        except Exception:
            return bool(getattr(vehicle, "out_of_road", False) or (not getattr(vehicle, "on_lane", True)))

    def _clear_traffic_vehicles_near_platoon(self) -> None:
        """Remove background traffic that overlaps or spawns too close to any platoon agent."""
        traffic_manager = getattr(getattr(self, "engine", None), "traffic_manager", None)
        traffic_vehicles = list(getattr(traffic_manager, "_traffic_vehicles", []) or [])
        if traffic_manager is None or not traffic_vehicles:
            return
        agents = getattr(self, "agents", {}) or {}
        active_agents = [agents[agent_id] for agent_id in self._agent_ids if agent_id in agents]
        if not active_agents:
            return
        min_clearance = float(self.config.get("traffic_spawn_min_agent_clearance_m", 20.0))
        to_remove = []
        for traffic_vehicle in traffic_vehicles:
            if bool(getattr(traffic_vehicle, "scenario_managed_vehicle", False)):
                continue
            try:
                traffic_pos = np.asarray(getattr(traffic_vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
                too_close = any(
                    float(np.linalg.norm(
                        traffic_pos - np.asarray(getattr(agent, "position", (0.0, 0.0))[:2], dtype=np.float32)
                    )) < min_clearance
                    for agent in active_agents
                )
                if too_close:
                    to_remove.append(traffic_vehicle)
            except Exception:
                continue
        if not to_remove:
            return
        ids_to_remove = [vehicle.id for vehicle in to_remove if getattr(vehicle, "id", None) is not None]
        if ids_to_remove:
            traffic_manager.clear_objects(ids_to_remove)
        for vehicle in to_remove:
            try:
                traffic_manager._traffic_vehicles.remove(vehicle)
            except ValueError:
                pass

    def _enforce_platoon_episode_end(
        self,
        terminated: Mapping[str, bool],
        truncated: Mapping[str, bool],
        info: Mapping[str, Mapping[str, object]],
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        terminated = dict(terminated)
        truncated = dict(truncated)
        info_ids = [agent_id for agent_id in self._agent_ids if agent_id in info]
        if not info_ids:
            terminated["__all__"] = True
            truncated["__all__"] = False
            return terminated, truncated
        failure_ids = [
            agent_id
            for agent_id in info_ids
            if bool(
                info[agent_id].get("crash_vehicle", False)
                or info[agent_id].get("crash_object", False)
                or info[agent_id].get("crash_building", False)
                or info[agent_id].get("crash_human", False)
                or info[agent_id].get("out_of_road", False)
            )
        ]
        if failure_ids:
            for agent_id in failure_ids:
                terminated[agent_id] = True

        terminated["__all__"] = bool(terminated.get("__all__", False)) or any(
            bool(terminated.get(agent_id, False)) for agent_id in self._agent_ids
        )
        truncated["__all__"] = bool(truncated.get("__all__", False)) or all(
            bool(truncated.get(agent_id, False)) for agent_id in self._agent_ids
        )
        if failure_ids:
            truncated["__all__"] = False
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
        base_info = base_info or {}
        info = {
            agent_id: dict(base_info[agent_id])
            for agent_id in self._agent_ids
            if agent_id in base_info
        }
        scenario_summary = {}
        scenario_orchestrator = getattr(self, "_scenario_orchestrator", None)
        if scenario_orchestrator is not None and hasattr(scenario_orchestrator, "get_episode_summary"):
            scenario_summary = dict(scenario_orchestrator.get_episode_summary())
        reward_cache = self._platoon_reward_cache if self._cfg_bool("platoon_reward_enabled", True) else None
        reward_per_agent = (
            reward_cache.get("per_agent", {})
            if isinstance(reward_cache, Mapping)
            else {}
        )
        min_gap = self._compute_min_gap()
        active_ids = [agent_id for agent_id in self._agent_ids if agent_id in self.agents]
        for agent_id in active_ids:
            agent_info = dict(info.get(agent_id, {}))
            raw_action = np.asarray(actions.get(agent_id, np.zeros((2,), dtype=np.float32)), dtype=np.float32)
            if raw_action.shape == (2,):
                current_action = raw_action
            else:
                current_action = np.zeros((2,), dtype=np.float32)
            previous_action = self._last_actions.get(agent_id, np.zeros((2,), dtype=np.float32))
            cached_agent_reward = (
                dict(reward_per_agent.get(agent_id, {}))
                if isinstance(reward_per_agent, Mapping)
                else {}
            )
            delta_steering = float(cached_agent_reward.get("delta_steering", current_action[0] - previous_action[0]))
            jerk = float(cached_agent_reward.get("jerk", current_action[1] - previous_action[1]))
            agent_info["role"] = self.get_agent_role(agent_id)
            agent_info["formation_relation_state"] = self.get_formation_relation_state(agent_id)
            if "formation_error" in cached_agent_reward:
                formation_error = cached_agent_reward["formation_error"]
            else:
                formation_error = self._compute_agent_formation_error(agent_id)
            agent_info["formation_error"] = float(formation_error)
            agent_info["min_gap"] = float(cached_agent_reward.get("min_gap", min_gap))
            agent_info["control_mode"] = control_mode
            agent_info["crash"] = bool(agent_info.get("crash", False))
            agent_info["arrive_dest"] = bool(agent_info.get("arrive_dest", False))
            agent_info["out_of_road"] = bool(agent_info.get("out_of_road", False))
            if "progress" in cached_agent_reward:
                progress = cached_agent_reward["progress"]
            else:
                progress = self._compute_progress(agent_id)
            agent_info["progress"] = float(progress)
            agent_info["jerk"] = jerk
            agent_info["delta_steering"] = delta_steering
            if "speed_km_h" in cached_agent_reward:
                speed_km_h = cached_agent_reward["speed_km_h"]
            else:
                speed_km_h = self._agent_speed_km_h(agent_id)
            agent_info["speed_km_h"] = float(speed_km_h)
            for reward_key in (
                "platoon_reward",
                "reward_safety",
                "reward_formation",
                "reward_efficiency",
                "reward_comfort",
                "team_min_gap",
                "team_mean_formation_error",
                "team_mean_progress",
                "team_crash_count",
                "team_out_of_road_count",
            ):
                if reward_key in cached_agent_reward:
                    agent_info[reward_key] = cached_agent_reward[reward_key]
            self._last_actions[agent_id] = current_action
            info[agent_id] = agent_info
        if scenario_summary:
            for agent_info in info.values():
                agent_info.update(scenario_summary)
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
        return self._cfg_float("vehicle_length_m", 5.74)

    def _desired_center_spacing_m(self, ego_id: Optional[str] = None, other_id: Optional[str] = None) -> float:
        speed_m_s = self._cfg_float("initial_speed_km_h", 25.0) / 3.6
        ego_length = self._vehicle_length_m(ego_id)
        other_length = self._vehicle_length_m(other_id)
        # return 0.5 * (ego_length + other_length) + speed_m_s * self._cfg_float("headway_time_s", 0.5)
        return 0.5 * (ego_length + other_length) + 10  # !暂时设置成固定的

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

    def _lateral_preview_pid(self, agent_id: str, trajectory: np.ndarray) -> float:
        """Track an ego-local preview point with one bounded PID controller."""
        vehicle = (getattr(self, "agents", {}) or {}).get(agent_id)
        if vehicle is None:
            raise ValueError(f"Unknown or inactive agent: {agent_id}")
        speed_mps = max(0.0, self._agent_speed_km_h(agent_id) / 3.6)
        lookahead_min = self._cfg_float("preview_lookahead_min_m", 3.0)
        lookahead_max = self._cfg_float("preview_lookahead_max_m", 8.0)
        if lookahead_min <= 0.0 or lookahead_max < lookahead_min:
            raise ValueError("preview lookahead bounds are invalid")
        lookahead_m = float(
            np.clip(
                self._cfg_float("preview_lookahead_time_s", 0.6) * speed_mps,
                lookahead_min,
                lookahead_max,
            )
        )
        path_xy = np.concatenate(
            (
                np.zeros((1, 2), dtype=np.float64),
                trajectory[:, :2].astype(np.float64, copy=False),
            ),
            axis=0,
        )
        segment = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(segment)))
        query = min(lookahead_m, float(arc[-1]))
        if query <= 1.0e-6:
            return 0.0
        x = float(np.interp(query, arc, path_xy[:, 0]))
        y = float(np.interp(query, arc, path_xy[:, 1]))
        headings = np.concatenate(
            ([0.0], np.unwrap(trajectory[:, 2].astype(np.float64)))
        )
        heading = float(np.interp(query, arc, headings))
        if not np.isfinite([x, y, heading]).all():
            raise ValueError("Preview point is not finite")

        bearing_error = math.atan2(y, max(x, 1.0e-3))
        heading_error = _wrap_to_pi(heading)
        error = _wrap_to_pi(
            bearing_error
            + self._cfg_float("preview_heading_weight", 0.5) * heading_error
        )
        dt_s = self._cfg_float("physics_world_step_size", 0.02) * self._cfg_int(
            "decision_repeat", 5
        )
        if dt_s <= 0.0:
            raise ValueError("controller decision timestep must be positive")
        kp = self._cfg_float("preview_pid_kp", 1.6)
        ki = self._cfg_float("preview_pid_ki", 0.05)
        kd = self._cfg_float("preview_pid_kd", 0.12)
        integral_limit = self._cfg_float("preview_pid_integral_limit", 1.0)
        if (
            not np.isfinite([kp, ki, kd, integral_limit]).all()
            or min(kp, ki, kd) < 0.0
            or integral_limit <= 0.0
        ):
            raise ValueError("preview PID gains are invalid")

        integral, previous_error, initialized = self._lateral_preview_pid_state.get(
            agent_id, (0.0, 0.0, False)
        )
        candidate_integral = float(
            np.clip(integral + error * dt_s, -integral_limit, integral_limit)
        )
        derivative = (error - previous_error) / dt_s if initialized else 0.0
        raw = kp * error + ki * candidate_integral + kd * derivative
        if abs(raw) > 1.0 and np.sign(raw) == np.sign(error):
            candidate_integral = integral
            raw = kp * error + ki * candidate_integral + kd * derivative
        self._lateral_preview_pid_state[agent_id] = (
            candidate_integral,
            error,
            True,
        )
        return float(np.clip(raw, -1.0, 1.0))

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

    @staticmethod
    def _trajectory_target_speed_mps(trajectory: np.ndarray) -> float:
        trajectory = np.asarray(trajectory)
        if trajectory.shape != (8, 3):
            raise ValueError(
                f"Expected trajectory shape (8, 3), got {trajectory.shape}"
            )
        if not np.issubdtype(trajectory.dtype, np.floating) or not np.isfinite(
            trajectory
        ).all():
            raise ValueError("Trajectory must contain finite floating-point values")
        # This helper is only valid for a trajectory rooted at the current ego
        # pose.  Fixed-world references must pass an explicit longitudinal
        # profile and may not encode position lag as extra target speed.
        target_speed = float(np.linalg.norm(trajectory[0, :2]) / 0.5)
        path = np.concatenate(
            (
                np.zeros((1, 3), dtype=np.float64),
                trajectory.astype(np.float64, copy=False),
            ),
            axis=0,
        )
        distance = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        heading_delta = np.abs(
            np.arctan2(
                np.sin(np.diff(path[:, 2])),
                np.cos(np.diff(path[:, 2])),
            )
        )
        curvature = heading_delta / np.maximum(distance, 1.0e-3)
        maximum_curvature = float(curvature.max(initial=0.0))
        curve_speed_limit = (
            math.sqrt(6.0 / maximum_curvature)
            if maximum_curvature > 1.0e-6
            else float("inf")
        )
        return float(
            np.clip(min(target_speed, curve_speed_limit), 0.0, 100.0 / 3.6)
        )

    def _longitudinal_lqr(
        self, agent_id: str, trajectory_target_speed_mps: float
    ) -> float:
        ego_idx = self._agent_ids.index(agent_id)
        current_speed = self._agent_speed_km_h(agent_id) / 3.6
        target_speed = float(trajectory_target_speed_mps)
        if not np.isfinite(target_speed) or target_speed < 0.0:
            raise ValueError(
                "trajectory_target_speed_mps must be finite and non-negative"
            )

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

    def trajectory_reference_to_control(
        self,
        agent_id: str,
        trajectory: np.ndarray,
        longitudinal_reference: LongitudinalTrackingReference,
    ) -> np.ndarray:
        trajectory = np.asarray(trajectory)
        if trajectory.shape != (8, 3) or not np.issubdtype(
            trajectory.dtype, np.floating
        ) or not np.isfinite(trajectory).all():
            raise ValueError("trajectory must be finite floating-point [8,3]")
        steering = self._lateral_preview_pid(agent_id, trajectory)
        ego_idx = self._agent_ids.index(agent_id)
        current_speed = self._agent_speed_km_h(agent_id) / 3.6
        gap_acceleration = 0.0
        if ego_idx > 0:
            front_id = self._agent_ids[ego_idx - 1]
            ego_pose = self._agent_pose(agent_id)
            front_pose = self._agent_pose(front_id)
            if ego_pose is not None and front_pose is not None:
                actual_gap = float(_world_to_ego(ego_pose, front_pose)[0])
                desired_gap = self._desired_center_spacing_m(agent_id, front_id)
                front_speed = self._agent_speed_km_h(front_id) / 3.6
                gap_acceleration = float(
                    np.clip(
                        0.15 * (actual_gap - desired_gap)
                        + 0.20 * (front_speed - current_speed),
                        -1.0,
                        1.0,
                    )
                )
        longitudinal_controller = getattr(
            self, "_trajectory_longitudinal_controller", None
        )
        if longitudinal_controller is None:
            longitudinal_controller = LongitudinalCascadeController(
                dt_s=self._cfg_float("physics_world_step_size", 0.02)
                * self._cfg_int("decision_repeat", 5)
            )
            self._trajectory_longitudinal_controller = longitudinal_controller
        throttle, _ = longitudinal_controller.compute(
            agent_id,
            current_speed,
            longitudinal_reference,
            gap_acceleration_mps2=gap_acceleration,
        )
        return np.asarray([steering, throttle], dtype=np.float32)

    def trajectory_to_control(self, agent_id: str, trajectory: np.ndarray) -> np.ndarray:
        trajectory = np.asarray(trajectory)
        current_speed = self._agent_speed_km_h(agent_id) / 3.6
        reference = trajectory_to_longitudinal_reference(
            trajectory,
            current_speed,
            source="online_trajectory",
        )
        return self.trajectory_reference_to_control(
            agent_id, trajectory, reference
        )

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
                t = float(step_idx + 1) * self._cfg_float("trajectory_dt", 0.5)
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
