from __future__ import annotations

from typing import Mapping

import numpy as np

if not hasattr(np, "bool8"):  # pragma: no cover - ray 2.4 expects this legacy alias
    np.bool8 = np.bool_

try:  # pragma: no cover - ray is optional in local dev
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except Exception:  # pragma: no cover
    DefaultCallbacks = object  # type: ignore


def _collect_episode_infos(episode, *, base_env=None, env_index: int | None = None) -> list[dict]:
    # 收集每个智能体的 info 字典
    infos: list[dict] = []
    if base_env is not None and env_index is not None:
        getter = getattr(base_env, "get_sub_environments", None)
        if callable(getter):
            try:
                sub_envs = getter()
            except TypeError:
                sub_envs = []
            if 0 <= env_index < len(sub_envs):
                sub_env = sub_envs[env_index]
                info_getter = getattr(sub_env, "get_last_step_infos", None)
                if callable(info_getter):
                    step_infos = info_getter()
                    if isinstance(step_infos, Mapping):
                        infos.extend([info for info in step_infos.values() if info])
                        if infos:
                            return infos

    raw_mapping = getattr(episode, "_agent_to_last_info", None)
    if isinstance(raw_mapping, Mapping):
        infos.extend([info for info in raw_mapping.values() if info])
    if infos:
        return infos

    agent_ids = []
    for attr in ("agent_rewards", "_agent_to_last_obs", "_agent_to_last_action"):
        value = getattr(episode, attr, None)
        if isinstance(value, Mapping):
            agent_ids = list(value.keys())
            break
    if not agent_ids:
        last = getattr(episode, "last_info_for", None)
        if callable(last):
            try:
                info = last()
            except TypeError:
                info = None
            if info:
                return [info]
        return []

    last_info_for = getattr(episode, "last_info_for", None)
    if not callable(last_info_for):
        return infos
    normalized_agent_ids = []
    seen = set()
    for agent_id in agent_ids:
        normalized = agent_id[0] if isinstance(agent_id, tuple) and agent_id else agent_id
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_agent_ids.append(normalized)
    for agent_id in normalized_agent_ids:
        try:
            info = last_info_for(agent_id)
        except TypeError:
            info = None
        if info:
            infos.append(info)
    return infos


def _finite_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


class PlatoonFormationCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs,
    ):
        episode.user_data["formation_errors"] = []
        episode.user_data["crashes"] = []
        episode.user_data["team_rewards"] = []
        episode.user_data["selected_intents"] = []

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        infos = _collect_episode_infos(episode, base_env=base_env, env_index=env_index)
        for info in infos:
            episode.user_data["formation_errors"].append(float(info.get("formation_error", 0.0)))
            episode.user_data["crashes"].append(float(info.get("crash", False)))
            if "team_reward" in info:
                episode.user_data["team_rewards"].append(float(info.get("team_reward", 0.0)))
            elif "selector_reward" in info:
                episode.user_data["team_rewards"].append(float(info.get("selector_reward", 0.0)))
            if "selected_intent" in info:
                episode.user_data["selected_intents"].append(int(info["selected_intent"]))

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        intents = episode.user_data.get("selected_intents", [])
        formation_errors = episode.user_data.get("formation_errors", [])
        crashes = episode.user_data.get("crashes", [])
        team_rewards = episode.user_data.get("team_rewards", [])

        episode.custom_metrics["formation_error_mean"] = _finite_mean(formation_errors)
        episode.custom_metrics["crash_rate"] = _finite_mean(crashes)
        episode.custom_metrics["team_reward_mean"] = _finite_mean(team_rewards)

        if intents:
            max_intent = int(max(intents))
            counts = np.bincount(np.asarray(intents, dtype=np.int64), minlength=max_intent + 1)
            probs = counts / max(counts.sum(), 1)
            entropy = float(-(probs * np.log(probs + 1e-8)).sum() / np.log(len(counts) + 1e-8))
            episode.custom_metrics["intent_entropy"] = entropy
            for idx, count in enumerate(counts):
                episode.custom_metrics[f"intent_usage_{idx}"] = float(count / max(counts.sum(), 1))
        else:
            episode.custom_metrics["intent_entropy"] = 0.0
