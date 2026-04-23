from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym  # type: ignore

try:  # pragma: no cover - ray is optional for local unit tests
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except Exception:  # pragma: no cover
    class MultiAgentEnv:  # type: ignore
        pass

from evaluation.co_preference_reward import compute_preference_reward
from evaluation.reward_terms import compute_step_reward, compute_team_reward
from models.co_preference.geometry import (
    TOPOLOGY_CURRENT,
    TOPOLOGY_NAMES,
    build_topology_mask,
    map_preference_to_target_point,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return float(arr[0]) if arr.size else float(default)
    except Exception:
        return float(default)


class CoPreferencePlatoonEnv(MultiAgentEnv):
    """RLlib-style wrapper that maps co-preference actions to planner target points."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        super().__init__()
        self.config = dict(config or {})
        self.num_agents = int(self.config.get("num_agents", 3))
        self.reward_config = dict(self.config.get("reward_config", {}))
        self.preference_reward_config = dict(self.config.get("preference_reward_config", {}))
        self._agent_ids = [f"agent{i}" for i in range(self.num_agents)]
        self.base_env = self._build_base_env()
        self.planner = self._build_planner()
        self._last_obs: dict[str, dict[str, Any]] = {}
        self._previous_targets: dict[str, np.ndarray] = {}
        self.action_space = self._make_action_space()

    def _make_action_space(self):
        single = gym.spaces.Dict(
            {
                "topology_choice": gym.spaces.Discrete(len(TOPOLOGY_NAMES)),
                "s": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )
        return gym.spaces.Dict({agent_id: single for agent_id in self._agent_ids})

    def _build_base_env(self):
        if self.config.get("base_env") is not None:
            return self.config["base_env"]
        factory = self.config.get("base_env_factory")
        if callable(factory):
            return factory(self.config)
        from envs.platoon_env import PlatoonEnv

        env_config = {
            key: value
            for key, value in self.config.items()
            if key
            not in {
                "base_env",
                "base_env_factory",
                "planner",
                "planner_factory",
                "reward_config",
                "preference_reward_config",
            }
        }
        return PlatoonEnv(env_config)

    def _build_planner(self):
        if self.config.get("planner") is not None:
            return self.config["planner"]
        factory = self.config.get("planner_factory")
        if callable(factory):
            return factory(self.config)
        raise RuntimeError("CoPreferencePlatoonEnv requires planner or planner_factory.")

    def reset(self, *, seed=None, options=None):  # noqa: D401 - RLlib compatibility
        result = self.base_env.reset()
        info = {}
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
        self._last_obs = dict(obs or {})
        self._previous_targets = {}
        augmented = self._augment_observations(self._last_obs)
        if self.config.get("rllib_reset_compat", False):
            return augmented
        return augmented if not info else (augmented, info)

    def step(self, actions: Mapping[str, Any]):
        if not self._last_obs:
            raise RuntimeError("CoPreferencePlatoonEnv.step() called before reset().")
        preference_points: dict[str, np.ndarray] = {}
        selected_polylines: dict[str, np.ndarray] = {}
        co_info: dict[str, dict[str, Any]] = {}
        planner_batch: dict[str, dict[str, Any]] = {}

        for agent_id, obs in self._last_obs.items():
            action = actions.get(agent_id, {"topology_choice": TOPOLOGY_CURRENT, "s": 0.5})
            topology_choice, s = self._parse_action(action)
            polylines = self._extract_polylines(agent_id, obs)
            target_point, debug = map_preference_to_target_point(topology_choice, s, polylines)
            preference_points[agent_id] = target_point
            selected_polylines[agent_id] = np.asarray(polylines[TOPOLOGY_NAMES[int(debug["co_preference_choice"])]], dtype=np.float32)
            co_info[agent_id] = debug
            planner_batch[agent_id] = self._build_planner_sample(obs)

        trajectories = self.planner.forward_with_preference(planner_batch, preference_points)
        result = self.base_env.step(trajectories)
        if len(result) == 5:
            raw_obs, reward, terminated, truncated, info = result
        else:
            raw_obs, reward, done, info = result
            terminated = dict(done)
            truncated = {agent_id: False for agent_id in terminated}
            truncated["__all__"] = False

        info = dict(info or {})
        reward = dict(reward or {})
        for agent_id, target_point in preference_points.items():
            agent_info = dict(info.get(agent_id, {}))
            pref_reward, pref_terms = compute_preference_reward(
                target_point=target_point,
                selected_polyline=selected_polylines[agent_id],
                previous_target_point=self._previous_targets.get(agent_id),
                **self.preference_reward_config,
            )
            agent_info.update(co_info[agent_id])
            agent_info.update(pref_terms)
            agent_info["co_preference_reward"] = float(pref_reward)
            local_reward = compute_step_reward(agent_info, self.reward_config)
            reward[agent_id] = float(reward.get(agent_id, 0.0)) + local_reward + pref_reward
            info[agent_id] = agent_info
            self._previous_targets[agent_id] = np.asarray(target_point, dtype=np.float32)

        team_reward = compute_team_reward({agent_id: [info[agent_id]] for agent_id in info}, self.reward_config)
        for agent_id in reward:
            reward[agent_id] = float(reward[agent_id]) + float(team_reward)
            info.setdefault(agent_id, {})["co_preference_team_reward"] = float(team_reward)

        self._last_obs = dict(raw_obs or {})
        next_obs = self._augment_observations(self._last_obs)
        if self.config.get("rllib_step_compat", False):
            done = dict(terminated)
            done["__all__"] = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            return next_obs, reward, done, info
        return next_obs, reward, terminated, truncated, info

    def _augment_observations(self, obs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        ret: dict[str, dict[str, Any]] = {}
        for agent_id, agent_obs in obs.items():
            item = dict(agent_obs)
            item["co_preference_topology_mask"] = build_topology_mask(self._extract_polylines(agent_id, item)).astype(np.float32)
            ret[agent_id] = item
        return ret

    def _parse_action(self, action: Any) -> tuple[int, float]:
        if isinstance(action, Mapping):
            return int(_as_float(action.get("topology_choice"), TOPOLOGY_CURRENT)), _as_float(action.get("s"), 0.5)
        arr = np.asarray(action).reshape(-1)
        if arr.size >= 2:
            return int(arr[0]), float(arr[1])
        if arr.size == 1:
            return int(arr[0]), 0.5
        return TOPOLOGY_CURRENT, 0.5

    def _extract_polylines(self, agent_id: str, obs: Mapping[str, Any]) -> dict[str, Any]:
        if "topology_polylines" in obs:
            return dict(obs["topology_polylines"])
        if "current_lane_polyline" in obs:
            return {
                "current": obs.get("current_lane_polyline"),
                "left": obs.get("left_lane_polyline"),
                "right": obs.get("right_lane_polyline"),
                "branch": obs.get("branch_polyline", obs.get("right_branch_polyline", obs.get("left_branch_polyline"))),
            }
        vehicle = getattr(self.base_env, "agents", {}).get(agent_id) if hasattr(self.base_env, "agents") else None
        if vehicle is not None:
            try:
                from metadrive.policy.diffusion_policy.mode_context import build_mode_context_from_vehicle

                ctx = build_mode_context_from_vehicle(
                    vehicle,
                    getattr(getattr(self.base_env, "engine", None), "current_map", None),
                    getattr(getattr(self.base_env, "config", {}), "get", lambda *_: None)("scenario_id"),
                    getattr(getattr(self.base_env, "config", {}), "get", lambda *_: None)("local_route"),
                    getattr(getattr(self.base_env, "config", {}), "get", lambda *_: None)("ego_main_route_block_ids"),
                )
                return {
                    "current": ctx.current_lane_polyline,
                    "left": ctx.left_lane_polyline,
                    "right": ctx.right_lane_polyline,
                    "branch": ctx.right_branch_polyline if ctx.right_branch_polyline is not None else ctx.left_branch_polyline,
                }
            except Exception:
                pass
        return {
            "current": np.asarray([[0.0, 0.0], [20.0, 0.0]], dtype=np.float32),
            "left": None,
            "right": None,
            "branch": None,
        }

    def _build_planner_sample(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        key_map = {
            "camera": "camera",
            "lidar": "lidar",
            "status": "status",
            "formation_relation_state": "formation_relation_state",
        }
        sample = {}
        for target_key, source_key in key_map.items():
            if source_key not in obs:
                raise KeyError(f"CoPreferencePlatoonEnv observation is missing {source_key!r}.")
            sample[target_key] = obs[source_key]
        return sample
