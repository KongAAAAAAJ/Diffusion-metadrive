from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym  # type: ignore

from metadrive.policy.diffusion_policy.selected_mode_guidance import apply_selected_mode_guidance
from metadrive.policy.diffusion_policy.transfuser_policy import compute_trajectory_control


class ModeSelectionSB3Env(gym.Env):
    """Joint CTDE Gymnasium env for selecting one trajectory mode per vehicle."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__()
        self.config = self._with_selected_scenario(dict(config or {}))
        self.num_agents = int(self.config.get("num_agents", 3))
        self.base_env = self._build_base_env()
        self.planner = self._build_planner()
        self._agent_ids = [f"agent{i}" for i in range(self.num_agents)]
        self._last_raw_obs: dict[str, Mapping[str, Any]] = {}
        self._last_export: dict[str, Any] | None = None
        self._last_planner_batch: dict[str, dict] | None = None
        self._last_obs: dict[str, np.ndarray] | None = None
        self._num_modes = int(self.config.get("num_modes", 1))
        self._relation_state_dim = int(self.config.get("relation_state_dim", 12))
        self._global_state_dim = int(self.config.get("global_state_dim", self.num_agents * 20))
        debug_log_path = self.config.get("debug_log_path")
        self._debug_log_path = Path(debug_log_path) if debug_log_path else None
        self._step_count = 0
        self.action_space = gym.spaces.MultiDiscrete([self._num_modes] * self.num_agents)
        self.observation_space = self._make_observation_space()
        self._lookahead_index = int(self.config.get("lookahead_index", 2))
        self._target_speed_km_h = float(self.config.get("target_speed_km_h", 30.0))
        self._controller_type = str(self.config.get("controller_type", "stabilized"))
        # trajectory_source: "diffusion" (default) → use diffusion planner candidates
        #                     "coarse"             → skip diffusion, use coarse kinematic trajectory
        self._trajectory_source = str(self.config.get("trajectory_source", "diffusion"))
        # use_action_mask: True (default) → MaskablePPO enforces mode_valid_mask
        #                  False          → all modes selectable; invalid slots filled with keep-lane fallback
        self._use_action_mask = bool(self.config.get("use_action_mask", True))

    def _build_base_env(self):
        if self.config.get("base_env") is not None:
            return self.config["base_env"]
        factory = self.config.get("base_env_factory")
        if callable(factory):
            return factory(self.config)
        from envs.platoon_env import PlatoonEnv

        filtered = {
            key: value
            for key, value in self.config.items()
            if key
            not in {
                "base_env",
                "base_env_factory",
                "planner",
                "planner_factory",
                "num_modes",
                "relation_state_dim",
                "global_state_dim",
                "planner_device",
                "lookahead_index",
                "target_speed_km_h",
                "controller_type",
                "debug_dynamic_anchor_errors",
                "scenario_ids",
                "scenario_index",
                "local_route_index",
                "debug_log_path",
                "trajectory_source",
                "use_action_mask",
                "seed_offset",
            }
        }
        return PlatoonEnv(filtered)

    @staticmethod
    def _with_selected_scenario(config: dict[str, Any]) -> dict[str, Any]:
        scenario_ids = list(config.get("scenario_ids") or [])
        if not scenario_ids:
            return config
        try:
            from scenarios.definitions import SCENARIO_BY_ID
        except Exception as exc:  # pragma: no cover - import should be available in project runtime
            raise RuntimeError("scenario_ids require scenarios.definitions.SCENARIO_BY_ID") from exc

        index = (int(config.get("scenario_index", 0)) + int(config.get("seed_offset", 0))) % len(scenario_ids)
        scenario_id = str(scenario_ids[index])
        if scenario_id not in SCENARIO_BY_ID:
            raise ValueError(f"Unknown scenario_id in mode-selection config: {scenario_id}")
        scenario = SCENARIO_BY_ID[scenario_id]
        local_routes = tuple(scenario.allowed_local_routes)
        if not local_routes:
            raise ValueError(f"Scenario {scenario_id} has no allowed local routes.")
        route_index = int(config.get("local_route_index", 0)) % len(local_routes)
        config = dict(config)
        config.setdefault("scenario_id", scenario_id)
        config.setdefault("local_route", local_routes[route_index])
        return config

    def _build_planner(self):
        if self.config.get("planner") is not None:
            return self.config["planner"]
        factory = self.config.get("planner_factory")
        if callable(factory):
            return factory(self.config)
        raise RuntimeError("ModeSelectionSB3Env requires planner or planner_factory.")

    def _make_observation_space(self):
        return gym.spaces.Dict(
            {
                "agent_relation_states": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents, self._relation_state_dim),
                    dtype=np.float32,
                ),
                "trajectory_candidates": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents, self._num_modes, 8, 3),
                    dtype=np.float32,
                ),
                "agent_mode_masks": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents, self._num_modes),
                    dtype=np.bool_,
                ),
                "pretrained_logits": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents, self._num_modes),
                    dtype=np.float32,
                ),
                "global_state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._global_state_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        result = self.base_env.reset()
        self._step_count = 0
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
        else:
            raw_obs, info = result, {}
        self._last_raw_obs = self._normalize_raw_obs(raw_obs or {}, previous_obs=None)
        obs = self._refresh_mode_export()
        return obs, dict(info or {})

    def step(self, action):
        if self._last_export is None:
            raise RuntimeError("ModeSelectionSB3Env.step() called before reset().")
        if self._last_planner_batch is None:
            raise RuntimeError("ModeSelectionSB3Env.step() missing cached planner batch.")
        step_export = self._last_export
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.shape[0] != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} mode actions, got {action_arr.shape[0]}.")
        masks = np.asarray(step_export["mode_valid_mask"], dtype=bool)
        agent_ids = list(step_export["agent_ids"])
        initial_candidates = np.asarray(step_export["trajectory_candidates"], dtype=np.float32)

        # If any controlled agent has been removed from the environment (crash / out-of-road
        # during delay_done cooldown), the episode must have already been terminal.  Force an
        # immediate done rather than proceeding with stale / invalid actions.
        live_agents = getattr(self.base_env, "agents", {})
        missing = [aid for aid in agent_ids if aid not in live_agents]
        if missing:
            obs = self._last_obs if self._last_obs is not None else self._refresh_mode_export()
            return obs, 0.0, True, False, {"force_done": True, "missing_agents": missing}

        if self._use_action_mask:
            for idx, (agent_id, mode_idx) in enumerate(zip(agent_ids, action_arr)):
                if mode_idx < 0 or mode_idx >= masks.shape[1] or not bool(masks[idx, mode_idx]):
                    raise ValueError(f"invalid mode action for {agent_id}: {int(mode_idx)}")
        planner_batch = {
            agent_id: dict(self._last_planner_batch[agent_id])
            for agent_id in agent_ids
            if agent_id in self._last_planner_batch
        }
        coarse_by_agent = {
            agent_id: np.asarray(planner_batch[agent_id]["coarse_trajectories"], dtype=np.float32)
            for agent_id in agent_ids
            if planner_batch.get(agent_id, {}).get("coarse_trajectories") is not None
        }
        vehicle_state_before = self._vehicle_states(agent_ids)
        vehicle_position_before = {
            agent_id: state["position"] for agent_id, state in vehicle_state_before.items()
        }

        if self._trajectory_source == "coarse":
            # Skip diffusion model inference entirely; execute the coarse kinematic trajectory.
            guidance_metadata = {}
            # Build dummy candidates array from coarse trajectories (padded to (8,3))
            num_modes_val = next(iter(coarse_by_agent.values())).shape[0] if coarse_by_agent else self._num_modes
            candidates = np.zeros((len(agent_ids), num_modes_val, 8, 3), dtype=np.float32)
            for idx, agent_id in enumerate(agent_ids):
                if agent_id in coarse_by_agent:
                    coarse = coarse_by_agent[agent_id]  # (num_modes, 8, 2)
                    candidates[idx, :coarse.shape[0], :, :2] = coarse
                    # Compute heading from consecutive xy differences for the (8, 2) coarse trajectory
                    for m_i in range(coarse.shape[0]):
                        xy = coarse[m_i]  # (8, 2)
                        headings = np.arctan2(
                            np.diff(xy[:, 1], prepend=xy[0, 1]),
                            np.diff(xy[:, 0], prepend=xy[0, 0]),
                        )
                        headings[0] = headings[1] if len(headings) > 1 else 0.0
                        candidates[idx, m_i, :, 2] = headings
            masks = np.asarray(step_export["mode_valid_mask"], dtype=bool)
        else:
            # Default: use guided diffusion planner output.
            guidance_metadata = apply_selected_mode_guidance(
                planner_batch,
                agent_ids=agent_ids,
                selected_modes=[int(value) for value in action_arr.tolist()],
                coarse_by_agent=coarse_by_agent,
                config=getattr(self.planner, "config", None),
            )
            guided_export = self.planner.export_mode_selection(planner_batch)
            guided_agent_ids = list(guided_export["agent_ids"])
            if guided_agent_ids != agent_ids:
                raise RuntimeError(f"Guided export agent order changed: {guided_agent_ids} != {agent_ids}")
            candidates = np.asarray(guided_export["trajectory_candidates"], dtype=np.float32)
            masks = np.asarray(guided_export["mode_valid_mask"], dtype=bool)

        # initial_candidates: pre-guidance diffusion output (used for logging only)
        # For coarse mode, use the coarse-derived candidates as both initial and final.
        if self._trajectory_source == "coarse":
            initial_candidates = candidates.copy()
        trajectories: dict[str, np.ndarray] = {}
        low_level_actions: dict[str, np.ndarray] = {}
        controller_debug: dict[str, dict[str, float]] = {}
        for idx, (agent_id, mode_idx) in enumerate(zip(agent_ids, action_arr)):
            trajectory = np.asarray(candidates[idx, mode_idx], dtype=np.float32)
            if trajectory.shape != (8, 3):
                raise ValueError(f"selected trajectory for {agent_id} must have shape (8,3), got {trajectory.shape}")
            trajectories[agent_id] = trajectory
            vehicle = getattr(self.base_env, "agents", {}).get(agent_id)
            current_speed_km_h = float(getattr(vehicle, "speed_km_h", 0.0)) if vehicle is not None else 0.0
            action_2d, debug = compute_trajectory_control(
                trajectory=trajectory,
                lookahead_index=self._lookahead_index,
                current_speed_km_h=current_speed_km_h,
                target_speed_km_h=self._target_speed_km_h,
                controller_type=self._controller_type,
            )
            low_level_actions[agent_id] = action_2d
            controller_debug[agent_id] = debug

        # Store selected trajectories + semantic mode groups so that
        # PlatoonEnv._build_trajectory_reward_cache() can compute road_topo_reward
        # without needing to re-infer the mode group from trajectory shape.
        if hasattr(self.base_env, "_pending_step_trajectories"):
            self.base_env._pending_step_trajectories = dict(trajectories)
        if hasattr(self.base_env, "_pending_step_all_candidates"):
            self.base_env._pending_step_all_candidates = {
                agent_id: np.asarray(candidates[idx], dtype=np.float32)
                for idx, agent_id in enumerate(agent_ids)
            }
        if hasattr(self.base_env, "_pending_step_mode_valid_masks"):
            self.base_env._pending_step_mode_valid_masks = {
                agent_id: np.asarray(masks[idx], dtype=bool)
                for idx, agent_id in enumerate(agent_ids)
            }
        if hasattr(self.base_env, "_pending_step_mode_groups"):
            planner_config = getattr(self.planner, "config", None)
            _mode_groups: dict[str, str] = {}
            if planner_config is not None:
                from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots
                _slots = build_mode_slots(
                    keep_lane_count=planner_config.mode_keep_lane_count,
                    lane_change_left_count=planner_config.mode_lane_change_left_count,
                    lane_change_right_count=planner_config.mode_lane_change_right_count,
                    emergency_stop_count=planner_config.mode_emergency_stop_count,
                )
                _slot_map = {s.index: s for s in _slots}
                for agent_id, mode_idx in zip(agent_ids, action_arr):
                    slot = _slot_map.get(int(mode_idx))
                    if slot is None:
                        _mode_groups[agent_id] = "keep"
                    elif slot.semantic_group == "STOP":
                        _mode_groups[agent_id] = "stop"
                    elif slot.lateral_direction == "left":
                        _mode_groups[agent_id] = "left"
                    elif slot.lateral_direction == "right":
                        _mode_groups[agent_id] = "right"
                    else:
                        _mode_groups[agent_id] = "keep"
            self.base_env._pending_step_mode_groups = _mode_groups

        result = self.base_env.step(low_level_actions)
        if len(result) == 5:
            raw_obs, env_reward, terminated, truncated, info = result
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        else:
            raw_obs, env_reward, done_dict, info = result
            terminated = done_dict
            truncated = {agent_id: False for agent_id in done_dict}
            done = bool(done_dict.get("__all__", False))
        info = dict(info or {})
        missing_agent_ids = [agent_id for agent_id in self._agent_ids if agent_id not in (raw_obs or {})]
        if missing_agent_ids:
            terminated = dict(terminated)
            truncated = dict(truncated)
            for agent_id in self._agent_ids:
                terminated[agent_id] = True
            terminated["__all__"] = True
            truncated["__all__"] = False
            done = True
            info["missing_agent_ids"] = list(missing_agent_ids)
        vehicle_state_after = self._vehicle_states(agent_ids)
        vehicle_position_after = {
            agent_id: state["position"] for agent_id, state in vehicle_state_after.items()
        }
        env_reward = dict(env_reward or {})
        reward_values = [float(env_reward.get(agent_id, 0.0)) for agent_id in agent_ids]
        scalar_reward = float(np.mean(reward_values)) if reward_values else 0.0

        self._last_raw_obs = self._normalize_raw_obs(raw_obs or {}, previous_obs=self._last_raw_obs)
        obs = self._refresh_mode_export() if self._last_raw_obs else self._last_obs
        pretrained_argmax = np.asarray(step_export["pretrained_argmax_mode"], dtype=np.int64)
        termination_flags = {
            **{agent_id: bool(terminated.get(agent_id, False)) for agent_id in agent_ids},
            "__all__": bool(terminated.get("__all__", False)),
        }
        truncation_flags = {
            **{agent_id: bool(truncated.get(agent_id, False)) for agent_id in agent_ids},
            "__all__": bool(truncated.get("__all__", False)),
        }
        safety_flags = {}
        base_crash_flags = {}
        for agent_id in agent_ids:
            agent_info = dict(info.get(agent_id, {}))
            base_crash_flags[agent_id] = {
                "crash": bool(agent_info.get("crash", False)),
                "crash_vehicle": bool(agent_info.get("crash_vehicle", False)),
                "crash_human": bool(agent_info.get("crash_human", False)),
                "crash_object": bool(agent_info.get("crash_object", False)),
                "crash_building": bool(agent_info.get("crash_building", False)),
                "crash_sidewalk": bool(agent_info.get("crash_sidewalk", False)),
                "out_of_road": bool(agent_info.get("out_of_road", False)),
            }
            safety_flags[agent_id] = {
                "crash": bool(agent_info.get("crash", False)),
                "terminal_crash": bool(
                    agent_info.get("crash_vehicle", False)
                    or agent_info.get("crash_human", False)
                    or agent_info.get("crash_object", False)
                    or agent_info.get("crash_building", False)
                ),
                "out_of_road": bool(agent_info.get("out_of_road", False)),
            }
        info.update(
            {
                "step": int(self._step_count),
                "executed_mode": [int(x) for x in action_arr.tolist()],
                "pretrained_argmax_mode": [int(x) for x in pretrained_argmax.tolist()],
                "invalid_mode_rate": 0.0,
                "raw_cls_logits": np.asarray(step_export["raw_cls_logits"], dtype=float).tolist(),
                "guided_raw_cls_logits": np.asarray(
                    guided_export["raw_cls_logits"] if self._trajectory_source != "coarse" else step_export["raw_cls_logits"],
                    dtype=float,
                ).tolist(),
                "masked_cls_logits": np.asarray(
                    guided_export["masked_cls_logits"] if self._trajectory_source != "coarse" else step_export["masked_cls_logits"],
                    dtype=float,
                ).tolist(),
                "mode_valid_mask": masks.astype(bool).tolist(),
                "initial_selected_trajectory_endpoint": [
                    initial_candidates[idx, int(action_arr[idx]), -1, :2].astype(float).tolist()
                    for idx in range(len(agent_ids))
                ],
                "selected_trajectory_endpoint": [
                    trajectories[agent_id][-1, :2].astype(float).tolist() for agent_id in agent_ids
                ],
                "target_point_after": {
                    agent_id: guidance_metadata.get(agent_id, {}).get("target_point_after")
                    for agent_id in agent_ids
                },
                "selected_coarse_endpoint": {
                    agent_id: guidance_metadata.get(agent_id, {}).get("selected_coarse_endpoint")
                    for agent_id in agent_ids
                },
                "selected_low_level_action": {
                    agent_id: low_level_actions[agent_id].astype(float).tolist() for agent_id in agent_ids
                },
                "controller_debug": controller_debug,
                "candidate_endpoints": candidates[:, :, -1, :2].astype(float).tolist(),
                "reward": scalar_reward,
                "terminated": bool(terminated.get("__all__", False)),
                "truncated": bool(truncated.get("__all__", False)),
                "termination_flags": termination_flags,
                "truncation_flags": truncation_flags,
                "safety_flags": safety_flags,
                "base_crash_flags": base_crash_flags,
                "vehicle_position_before": vehicle_position_before,
                "vehicle_position_after": vehicle_position_after,
                "vehicle_state_before": vehicle_state_before,
                "vehicle_state_after": vehicle_state_after,
            }
        )
        self._write_debug_log(info)
        self._step_count += 1
        return obs, scalar_reward, done, False, info

    def _write_debug_log(self, info: Mapping[str, Any]) -> None:
        if self._debug_log_path is None:
            return
        self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: info[key]
            for key in (
                "step",
                "executed_mode",
                "pretrained_argmax_mode",
                "invalid_mode_rate",
                "raw_cls_logits",
                "guided_raw_cls_logits",
                "masked_cls_logits",
                "mode_valid_mask",
                "initial_selected_trajectory_endpoint",
                "selected_trajectory_endpoint",
                "target_point_after",
                "selected_coarse_endpoint",
                "selected_low_level_action",
                "controller_debug",
                "candidate_endpoints",
                "reward",
                "terminated",
                "truncated",
                "termination_flags",
                "truncation_flags",
                "safety_flags",
                "base_crash_flags",
                "vehicle_position_before",
                "vehicle_position_after",
                "vehicle_state_before",
                "vehicle_state_after",
            )
            if key in info
        }
        with self._debug_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def action_masks(self) -> np.ndarray:
        if not self._use_action_mask:
            return np.ones((self.num_agents * self._num_modes,), dtype=bool)
        if self._last_export is None:
            return np.ones((self.num_agents * self._num_modes,), dtype=bool)
        return np.asarray(self._last_export["mode_valid_mask"], dtype=bool).reshape(-1)

    def _refresh_mode_export(self) -> dict[str, np.ndarray]:
        planner_batch = {
            agent_id: self._build_planner_sample(agent_id, self._last_raw_obs[agent_id])
            for agent_id in self._agent_ids
            if agent_id in self._last_raw_obs
        }
        export = self.planner.export_mode_selection(planner_batch)
        self._last_planner_batch = planner_batch
        self._last_export = export
        self._num_modes = int(np.asarray(export["mode_valid_mask"]).shape[1])
        self.action_space = gym.spaces.MultiDiscrete([self._num_modes] * self.num_agents)
        self.observation_space = self._make_observation_space()
        obs = {
            "agent_relation_states": self._build_agent_relation_states(self._last_raw_obs, list(export["agent_ids"])),
            "trajectory_candidates": np.asarray(export["trajectory_candidates"], dtype=np.float32),
            "agent_mode_masks": np.asarray(export["mode_valid_mask"], dtype=bool),
            "pretrained_logits": np.asarray(export["raw_cls_logits"], dtype=np.float32),
            "global_state": self._build_global_state(self._last_raw_obs),
        }
        self._last_obs = obs
        return obs

    def _build_agent_relation_states(
        self,
        obs: Mapping[str, Mapping[str, Any]],
        agent_ids: list[str],
    ) -> np.ndarray:
        rows = []
        for agent_id in agent_ids:
            relation = np.asarray(obs[agent_id].get("formation_relation_state", []), dtype=np.float32).reshape(-1)
            if relation.size < self._relation_state_dim:
                relation = np.pad(relation, (0, self._relation_state_dim - relation.size))
            rows.append(relation[: self._relation_state_dim])
        if len(rows) < self.num_agents:
            rows.extend([np.zeros((self._relation_state_dim,), dtype=np.float32) for _ in range(self.num_agents - len(rows))])
        return np.stack(rows[: self.num_agents], axis=0).astype(np.float32)

    def _normalize_raw_obs(
        self,
        raw_obs: Mapping[str, Mapping[str, Any]],
        previous_obs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> dict[str, Mapping[str, Any]]:
        normalized: dict[str, Mapping[str, Any]] = {}
        previous_obs = previous_obs or {}
        for agent_id in self._agent_ids:
            if agent_id in raw_obs:
                normalized[agent_id] = raw_obs[agent_id]
            elif agent_id in previous_obs:
                normalized[agent_id] = previous_obs[agent_id]
            else:
                raise KeyError(f"ModeSelectionSB3Env missing initial observation for {agent_id}")
        return normalized

    def _build_global_state(self, obs: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
        parts = []
        for agent_id in sorted(obs):
            item = obs[agent_id]
            parts.append(np.asarray(item.get("status", []), dtype=np.float32).reshape(-1))
            parts.append(np.asarray(item.get("formation_relation_state", []), dtype=np.float32).reshape(-1))
        state = np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
        if state.size < self._global_state_dim:
            state = np.pad(state, (0, self._global_state_dim - state.size))
        return state[: self._global_state_dim].astype(np.float32)

    def _vehicle_states(self, agent_ids: list[str]) -> dict[str, dict[str, float | list[float]]]:
        states: dict[str, dict[str, float | list[float]]] = {}
        agents = getattr(self.base_env, "agents", {})
        for agent_id in agent_ids:
            vehicle = agents.get(agent_id)
            if vehicle is None:
                continue
            position = np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2], dtype=np.float64)
            states[agent_id] = {
                "position": position.astype(float).tolist(),
                "heading": float(getattr(vehicle, "heading_theta", getattr(vehicle, "heading", 0.0))),
                "length": float(getattr(vehicle, "LENGTH", 0.0)),
                "width": float(getattr(vehicle, "WIDTH", 0.0)),
            }
        return states

    def _build_planner_sample(self, agent_id: str, obs: Mapping[str, Any]) -> dict[str, Any]:
        keys = ("camera", "lidar", "status", "formation_relation_state")
        missing = [key for key in keys if key not in obs]
        if missing:
            raise KeyError(f"ModeSelectionSB3Env observation missing keys: {missing}")
        sample = {key: obs[key] for key in keys}
        vehicle = getattr(self.base_env, "agents", {}).get(agent_id)
        if vehicle is not None:
            sample.update(self._build_dynamic_mode_features(vehicle))
        else:
            # Vehicle removed (crash/out-of-road): inject fallback so coarse_trajectories
            # is always present in _last_planner_batch → prevents KeyError in next step.
            sample.update(self._build_dynamic_mode_features_fallback())
        return sample

    def _build_dynamic_mode_features_fallback(self) -> dict[str, np.ndarray]:
        from metadrive.policy.diffusion_policy.mode_definitions import mode_slot_count
        planner_config = getattr(self.planner, "config", None)
        if planner_config is None:
            return {}
        num_slots = mode_slot_count(
            planner_config.mode_keep_lane_count,
            planner_config.mode_lane_change_left_count,
            planner_config.mode_lane_change_right_count,
            planner_config.mode_emergency_stop_count,
        )
        return {
            "coarse_trajectories": np.zeros((num_slots, 8, 2), dtype=np.float32),
            "mode_valid_mask": np.ones((num_slots,), dtype=bool),   # all-True: any action passes mask check
        }

    def _build_dynamic_mode_features(self, vehicle) -> dict[str, np.ndarray]:
        from metadrive.policy.diffusion_policy.mode_context import build_mode_context_from_vehicle
        from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots, mode_slot_count
        from metadrive.policy.diffusion_policy.mode_trajectory_generator import ModeTrajectoryGenerator

        planner_config = getattr(self.planner, "config", None)
        if planner_config is None:
            return {}
        mode_slots = build_mode_slots(
            keep_lane_count=planner_config.mode_keep_lane_count,
            lane_change_left_count=planner_config.mode_lane_change_left_count,
            lane_change_right_count=planner_config.mode_lane_change_right_count,
            emergency_stop_count=planner_config.mode_emergency_stop_count,
        )
        num_slots = mode_slot_count(
            planner_config.mode_keep_lane_count,
            planner_config.mode_lane_change_left_count,
            planner_config.mode_lane_change_right_count,
            planner_config.mode_emergency_stop_count,
        )
        try:
            current_map = getattr(getattr(vehicle, "engine", None), "current_map", None)
            ctx = build_mode_context_from_vehicle(vehicle, current_map=current_map)
            generator = ModeTrajectoryGenerator(
                keep_lane_high_speed_mps=planner_config.mode_keep_high_speed_mps,
                keep_lane_medium_speed_mps=planner_config.mode_keep_medium_speed_mps,
                keep_lane_low_speed_mps=planner_config.mode_keep_low_speed_mps,
                emergency_decel_mps2=planner_config.mode_emergency_decel_mps2,
                keep_lane_level_count=planner_config.mode_keep_lane_count,
                lane_change_left_level_count=planner_config.mode_lane_change_left_count,
                lane_change_right_level_count=planner_config.mode_lane_change_right_count,
                emergency_stop_level_count=planner_config.mode_emergency_stop_count,
                mode_slots=mode_slots,
            )
            output = generator.generate(ctx, generate_all=not self._use_action_mask)
            return {
                "coarse_trajectories": np.asarray(output.coarse_trajectories, dtype=np.float32),
                "mode_valid_mask": np.asarray(output.mode_valid_mask, dtype=bool),
            }
        except Exception as exc:
            if bool(self.config.get("debug_dynamic_anchor_errors", False)):
                print(f"[ModeSelectionSB3Env] dynamic anchor generation failed for {getattr(vehicle, 'name', '?')}: {exc}")
            return {
                "coarse_trajectories": np.zeros((num_slots, 8, 2), dtype=np.float32),
                "mode_valid_mask": np.zeros((num_slots,), dtype=bool),
            }
