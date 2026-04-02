from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

if not hasattr(np, "bool8"):  # pragma: no cover - ray 2.4 expects this legacy alias
    np.bool8 = np.bool_

try:  # pragma: no cover - optional for pure numpy unit tests
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import gym  # type: ignore
except Exception:  # pragma: no cover - fallback for newer setups
    import gymnasium as gym

# Patch gym seeding for NumPy 2.0 + Ray 2.2.0 compatibility.
# Ray worker processes import this module when the env is created, so the patch
# is applied in every worker regardless of PYTHONPATH / sitecustomize.py.
try:
    from gym.utils import seeding as _gym_seeding

    _orig_gen_ctor = getattr(_gym_seeding, "_generator_ctor", None)
    if callable(_orig_gen_ctor):
        def _compat_gen_ctor(bit_generator="MT19937"):  # pragma: no cover
            if isinstance(bit_generator, np.random.BitGenerator):
                return np.random.Generator(bit_generator)
            try:
                return _orig_gen_ctor(bit_generator)
            except (ValueError, TypeError):
                return np.random.default_rng()
        _gym_seeding._generator_ctor = _compat_gen_ctor
except Exception:  # pragma: no cover - gym may not be present
    pass

try:  # pragma: no cover - ray is optional for local unit tests
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except Exception:  # pragma: no cover
    class MultiAgentEnv:  # type: ignore
        pass

from evaluation.reward_terms import compute_step_reward, compute_team_reward


def _wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _flatten_numeric(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32)
        return arr.reshape(-1)
    if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        return np.asarray([float(value)], dtype=np.float32)
    if isinstance(value, Mapping):
        chunks = []
        for key in sorted(value.keys()):
            chunks.append(_flatten_numeric(value[key]))
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=0)
    if isinstance(value, (list, tuple)):
        chunks = [_flatten_numeric(item) for item in value]
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=0)
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return np.zeros((0,), dtype=np.float32)
    return arr.reshape(-1)


def _to_torch_batch(value: Any, device: Optional["torch.device"] = None) -> Any:
    if torch is None:
        return value
    if isinstance(value, Mapping):
        return {key: _to_torch_batch(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_torch_batch(item, device) for item in value)
    if isinstance(value, np.ndarray):
        tensor = torch.as_tensor(value)
        return tensor.to(device) if device is not None else tensor
    if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        tensor = torch.as_tensor(value)
        return tensor.to(device) if device is not None else tensor
    return value


def _pad_or_trim(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    if values.shape[0] >= dim:
        return values[:dim].astype(np.float32, copy=False)
    padded = np.zeros((dim,), dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded


def _as_numpy_float32(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _infer_default_mode_embedding_dim(config: Mapping[str, object]) -> int:
    model_size = str(config.get("model_size", "small"))
    ego_fut_mode = int(config.get("ego_fut_mode", config.get("num_modes", config.get("K", 8))))
    try:
        from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config

        tf_config = build_transfuser_config(model_size, ego_fut_mode=ego_fut_mode)
        return int(tf_config.tf_d_model)
    except Exception:
        return int(config.get("candidate_summary_dim", 6))


def _build_default_candidate_generator(config: Mapping[str, object]):
    """构建一个默认的多车扩散规划器，加载单车预训练权重，完成多车权重迁移和冻结，仅用于 selector 阶段的候选轨迹生成"""
    pretrained_ckpt = str(config.get("pretrained_ckpt", "") or "").strip()
    if not pretrained_ckpt:
        raise RuntimeError(
            "SelectorPlatoonEnv requires pretrained_ckpt when no candidate_generator "
            "or candidate_generator_factory is injected."
        )

    anchor_path = str(
        config.get("anchor_path")
        or config.get("plan_anchor_path")
        or "metadrive/exp_dataset/anchors.npy"
    )
    model_size = str(config.get("model_size", "small"))
    ego_fut_mode = int(config.get("ego_fut_mode", config.get("num_modes", config.get("K", 8))))
    num_agents = int(config.get("num_agents", 3))
    planner_device = str(config.get("planner_device", "cpu"))
    if planner_device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        planner_device = "cpu"

    from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
    from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
    from models.platoon.weight_migration import migrate_single_to_platoon

    tf_config = build_transfuser_config(
        model_size,
        plan_anchor_path=anchor_path,
        ego_fut_mode=ego_fut_mode,
    )
    planner = PlatoonDiffusionPlanner(config=tf_config, num_vehicles=num_agents)
    planner = migrate_single_to_platoon(pretrained_ckpt, planner)
    planner.freeze_for_selector()
    if torch is not None:
        planner = planner.to(torch.device(planner_device))
    return planner


class SelectorPlatoonEnv(MultiAgentEnv):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        super().__init__()
        self.config = dict(config or {})
        self.num_agents = int(self.config.get("num_agents", 3))
        self.K = int(self.config.get("K", 9))
        self.agent_context_dim = int(self.config.get("agent_context_dim", 32))
        self.candidate_summary_dim = int(self.config.get("candidate_summary_dim", 6))
        default_mode_embedding_dim = self.candidate_summary_dim
        if (
            "mode_embedding_dim" not in self.config
            and self.config.get("candidate_generator") is None
            and not callable(self.config.get("candidate_generator_factory"))
        ):
            default_mode_embedding_dim = _infer_default_mode_embedding_dim(self.config)
        self.mode_embedding_dim = int(
            self.config.get("mode_embedding_dim", default_mode_embedding_dim)
        )
        self.lambda_local = float(self.config.get("lambda_local", 0.7))
        self.lambda_team = float(self.config.get("lambda_team", 0.3))

        self.reward_config = dict(self.config.get("reward_config", {}))

        self.rllib_reset_compat = bool(self.config.get("rllib_reset_compat", False))
        self.rllib_step_compat = bool(self.config.get("rllib_step_compat", False))

        self._agent_ids = [f"agent{i}" for i in range(self.num_agents)]
        self._selector_keys = {
            "base_env",
            "base_env_factory",
            "candidate_generator",
            "candidate_generator_factory",
            "pretrained_ckpt",
            "anchor_path",
            "model_size",
            "ego_fut_mode",
            "num_modes",
            "K",
            "planner_device",
            "agent_context_dim",
            "candidate_summary_dim",
            "mode_embedding_dim",
            "rllib_reset_compat",
            "rllib_step_compat",
            "lambda_local",
            "lambda_team",
            "reward_config",
        }
        self.base_env = self._build_base_env(self.config)
        self.generator = self._build_candidate_generator(self.config)
        self._planner_device = None
        if torch is not None and hasattr(self.generator, "parameters"):
            try:
                self._planner_device = next(self.generator.parameters()).device
            except (StopIteration, TypeError):
                self._planner_device = None
        self._candidate_cache: dict[str, dict[str, np.ndarray]] = {}
        self._last_raw_obs: dict[str, dict[str, Any]] = {}
        self._last_step_info: dict[str, dict[str, Any]] = {}
        self._episode_step = 0
        self.single_observation_space = self._build_single_observation_space()
        self.observation_space = self.single_observation_space
        self.single_action_space = gym.spaces.Discrete(self.K)
        self.action_space = self.single_action_space

    def _build_base_env(self, config: Mapping[str, object]):
        # 如果配置里直接提供了 base_env，直接使用
        base_env = config.get("base_env")
        if base_env is not None:
            return base_env
        factory = config.get("base_env_factory")

        if callable(factory):
            return factory(config)
        try:
            # *默认使用 PlatoonEnv 作为base_env
            from envs.platoon_env import PlatoonEnv
        except Exception as exc:  # pragma: no cover - exercised only when MetaDrive is missing
            raise RuntimeError(
                "SelectorPlatoonEnv requires PlatoonEnv unless a base_env/base_env_factory is injected."
            ) from exc
        env_config = {key: value for key, value in config.items() if key not in self._selector_keys}
        return PlatoonEnv(env_config)

    def _build_candidate_generator(self, config: Mapping[str, object]):
        generator = config.get("candidate_generator")
        if generator is not None:
            return generator
        factory = config.get("candidate_generator_factory")
        if callable(factory):
            return factory(config)
        return _build_default_candidate_generator(config)

    def _build_single_observation_space(self):
        return gym.spaces.Dict(
            {
                "agent_context": gym.spaces.Box(            # planner提供的agent上下文信息
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.agent_context_dim,),
                    dtype=np.float32,
                ),
                "formation_relation_state": gym.spaces.Box(     # planner提供的编队关系状态
                    low=-np.inf,
                    high=np.inf,
                    shape=(12,),
                    dtype=np.float32,
                ),
                "mode_embeddings": gym.spaces.Box(      # planner提供的候选轨迹的模式嵌入（如有）
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.K, self.mode_embedding_dim),
                    dtype=np.float32,
                ),
                "candidate_summary": gym.spaces.Box(     # planner提供的候选轨迹摘要信息
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.K, self.candidate_summary_dim),
                    dtype=np.float32,
                ),
                "global_state": gym.spaces.Box(  # 全局状态信息
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents * (self.agent_context_dim + 12 + 2 * self.candidate_summary_dim),),
                    dtype=np.float32,
                ),
                "action_mask": gym.spaces.Box(      # 可选的动作掩码，当前实现中始终为全1
                    low=0.0,
                    high=1.0,
                    shape=(self.K,),
                    dtype=np.float32,
                ),
            }
        )

    def _extract_agent_context(self, agent_obs: Mapping[str, Any]) -> np.ndarray:
        if "agent_context" in agent_obs:
            return _pad_or_trim(_flatten_numeric(agent_obs["agent_context"]), self.agent_context_dim)
        if "status" in agent_obs:
            return _pad_or_trim(_flatten_numeric(agent_obs["status"]), self.agent_context_dim)
        if "obs" in agent_obs:
            return _pad_or_trim(_flatten_numeric(agent_obs["obs"]), self.agent_context_dim)
        return _pad_or_trim(_flatten_numeric(agent_obs), self.agent_context_dim)

    def _build_candidate_summary(self, traj: np.ndarray) -> np.ndarray:
        """将轨迹转为特征"""
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError(f"Expected a trajectory of shape (T, 3), got {traj.shape}")
        dx = np.diff(traj[:, 0], prepend=traj[0:1, 0]).astype(np.float32)
        dy = np.diff(traj[:, 1], prepend=traj[0:1, 1]).astype(np.float32)
        path_length = float(np.sum(np.sqrt(dx * dx + dy * dy)))
        features = np.asarray(
            [
                float(traj[-1, 0]),
                float(traj[-1, 1]),
                _wrap_to_pi(float(traj[-1, 2])),
                path_length,
                float(np.mean(np.abs(traj[:, 1]))),
                float(np.max(np.abs(traj[:, 1]))),
                float(np.mean(dx)),
                float(np.mean(dy)),
                float(np.std(traj[:, 2])),
            ],
            dtype=np.float32,
        )
        return _pad_or_trim(features, self.candidate_summary_dim)

    def _build_candidate_payload(self, raw_obs: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
        """"""
        if hasattr(self.generator, "forward_selector"):
            return self._build_candidate_payload_batched(raw_obs) 
        
        candidate_payloads: dict[str, dict[str, np.ndarray]] = {}
        for agent_id in self._agent_ids:
            agent_obs = raw_obs.get(agent_id)
            if agent_obs is None:
                continue
            payload = self._generate_candidate_payload(agent_id, agent_obs)
            trajs = payload["trajs"]
            summaries = payload["candidate_summary"]
            mode_embeddings = payload["mode_embeddings"]
            candidate_payloads[agent_id] = {
                "trajs": trajs.astype(np.float32, copy=False),
                "candidate_summary": summaries,
                "mode_embeddings": mode_embeddings.astype(np.float32, copy=False),
                "agent_context": self._extract_agent_context(agent_obs),
                "formation_relation_state": _pad_or_trim(
                    _flatten_numeric(agent_obs.get("formation_relation_state", np.zeros((12,), dtype=np.float32))),
                    12,
                ),
            }
        return candidate_payloads

    def _build_candidate_payload_batched(self, raw_obs: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
        """把 base_env 给出的原始多车观测 raw_obs，转换成 selector 后续构造 observation 所需的中间表示 payload"""
        active_agent_ids = [agent_id for agent_id in self._agent_ids if raw_obs.get(agent_id) is not None]
        if not active_agent_ids:
            return {}

        batch = {
            agent_id: _to_torch_batch(dict(raw_obs[agent_id]), self._planner_device)
            for agent_id in active_agent_ids
        }
        outputs = self.generator.forward_selector(batch)  # 调用 planner 的 forward_selector() 方法，得到所有 agent 的候选轨迹等信息
        if not isinstance(outputs, Mapping):
            raise RuntimeError("forward_selector() must return a mapping keyed by agent_id.")

        candidate_payloads: dict[str, dict[str, np.ndarray]] = {}
        for agent_id in active_agent_ids:
            if agent_id not in outputs:
                raise KeyError(f"Planner forward_selector output is missing payload for active agent {agent_id!r}.")
            payload = outputs[agent_id]
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"Planner payload for {agent_id!r} must be a mapping, got {type(payload)!r}.")
            trajs = payload.get("trajectory_candidates", payload.get("trajectory"))
            if trajs is None:
                raise ValueError(f"Planner payload for {agent_id} is missing trajectory candidates.")
            trajs = self._normalize_candidate_trajs(trajs)  # 候选轨迹归一化

            candidate_summary = np.stack(
                [self._build_candidate_summary(traj) for traj in trajs], axis=0
            ).astype(np.float32)
            mode_embeddings = self._normalize_mode_embeddings(
                payload.get("trajectory_mode_embedding"),
                candidate_summary,
            )
            agent_obs = raw_obs[agent_id]
            candidate_payloads[agent_id] = {
                "trajs": trajs.astype(np.float32, copy=False),
                "candidate_summary": candidate_summary,
                "mode_embeddings": mode_embeddings.astype(np.float32, copy=False),
                "agent_context": self._extract_agent_context(agent_obs),
                "formation_relation_state": _pad_or_trim(
                    _flatten_numeric(
                        agent_obs.get(
                            "formation_relation_state",
                            np.zeros((12,), dtype=np.float32),
                        )
                    ),
                    12,
                ),
            }
        return candidate_payloads

    def _normalize_candidate_trajs(self, trajs: Any) -> np.ndarray:
        trajs = _as_numpy_float32(trajs)
        if trajs.ndim != 3 or trajs.shape[1:] != (8, 3):
            raise ValueError(f"Expected candidate trajectories of shape (K, 8, 3), got {trajs.shape}")
        if trajs.shape[0] == self.K:
            return trajs
        if trajs.shape[0] == 1:
            return np.repeat(trajs, self.K, axis=0)
        if trajs.shape[0] > self.K:
            return trajs[: self.K].astype(np.float32, copy=False)
        raise ValueError(f"Expected at least {self.K} candidate trajectories, got {trajs.shape[0]}")

    def _normalize_mode_embeddings(self, mode_embeddings: Any, candidate_summary: np.ndarray) -> np.ndarray:
        if mode_embeddings is None:
            flat = _pad_or_trim(candidate_summary.reshape(-1), self.K * self.mode_embedding_dim)
            return flat.reshape(self.K, self.mode_embedding_dim)
        
        mode_embeddings = _as_numpy_float32(mode_embeddings)
        if mode_embeddings.ndim != 2:
            raise ValueError(f"Expected mode embeddings of shape (K, D), got {mode_embeddings.shape}")
        if mode_embeddings.shape[0] == 1:
            mode_embeddings = np.repeat(mode_embeddings, self.K, axis=0)
        elif mode_embeddings.shape[0] < self.K:
            raise ValueError(f"Expected at least {self.K} mode embeddings, got {mode_embeddings.shape[0]}")
        elif mode_embeddings.shape[0] > self.K:
            mode_embeddings = mode_embeddings[: self.K]
        flat = _pad_or_trim(mode_embeddings.reshape(-1), self.K * self.mode_embedding_dim)
        return flat.reshape(self.K, self.mode_embedding_dim)

    def _generate_candidate_payload(self, agent_id: str, agent_obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        generator = self.generator
        if hasattr(generator, "generate"):
            try:
                trajs = generator.generate(agent_id=agent_id, agent_obs=agent_obs, base_env=self.base_env, k=self.K)
            except TypeError:
                try:
                    trajs = generator.generate(agent_id, agent_obs, self.base_env, self.K)
                except TypeError:
                    try:
                        trajs = generator.generate(agent_obs)
                    except TypeError:
                        trajs = generator.generate(agent_obs, self.K)
            trajs = self._normalize_candidate_trajs(trajs)
            candidate_summary = np.stack([self._build_candidate_summary(traj) for traj in trajs], axis=0).astype(np.float32)
            return {
                "trajs": trajs,
                "candidate_summary": candidate_summary,
                "mode_embeddings": self._normalize_mode_embeddings(None, candidate_summary), 
            }
        if callable(generator):
            trajs = self._normalize_candidate_trajs(
                generator(agent_id=agent_id, agent_obs=agent_obs, base_env=self.base_env, k=self.K)
            )
            candidate_summary = np.stack([self._build_candidate_summary(traj) for traj in trajs], axis=0).astype(np.float32)
            return {
                "trajs": trajs,
                "candidate_summary": candidate_summary,
                "mode_embeddings": self._normalize_mode_embeddings(None, candidate_summary),
            }
        raise RuntimeError(
            "SelectorPlatoonEnv candidate generator must provide forward_selector(), "
            "generate(), or be a callable that returns (K, 8, 3) trajectories."
        )

    def _compose_global_state(self, payloads: Mapping[str, Mapping[str, np.ndarray]]) -> np.ndarray:
        segments: list[np.ndarray] = []
        zero_context = np.zeros((self.agent_context_dim,), dtype=np.float32)
        zero_relation = np.zeros((12,), dtype=np.float32)
        zero_summary = np.zeros((self.candidate_summary_dim,), dtype=np.float32)
        for agent_id in self._agent_ids:
            payload = payloads.get(agent_id)
            if payload is None:
                segments.append(np.concatenate([zero_context, zero_relation, zero_summary, zero_summary], axis=0))
                continue
            candidate_summary = np.asarray(payload["candidate_summary"], dtype=np.float32)
            summary_mean = candidate_summary.mean(axis=0) if candidate_summary.size else zero_summary
            summary_std = candidate_summary.std(axis=0) if candidate_summary.size else zero_summary
            segments.append(
                np.concatenate(
                    [
                        _pad_or_trim(payload["agent_context"], self.agent_context_dim),
                        _pad_or_trim(payload["formation_relation_state"], 12),
                        _pad_or_trim(summary_mean, self.candidate_summary_dim),
                        _pad_or_trim(summary_std, self.candidate_summary_dim),
                    ],
                    axis=0,
                )
            )
        return np.concatenate(segments, axis=0).astype(np.float32, copy=False)

    def _build_selector_obs(self, raw_obs: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
        payloads = self._build_candidate_payload(raw_obs)
        global_state = self._compose_global_state(payloads)

        selector_obs: dict[str, dict[str, np.ndarray]] = {}
        for agent_id, payload in payloads.items():
            selector_obs[agent_id] = {
                "agent_context": _pad_or_trim(payload["agent_context"], self.agent_context_dim),
                "formation_relation_state": _pad_or_trim(payload["formation_relation_state"], 12),
                "mode_embeddings": np.asarray(payload["mode_embeddings"], dtype=np.float32),
                "candidate_summary": np.asarray(payload["candidate_summary"], dtype=np.float32),
                "global_state": global_state.copy(),
                "action_mask": np.ones((self.K,), dtype=np.float32),
            }
        self._candidate_cache = payloads
        self._last_raw_obs = {agent_id: dict(agent_obs) for agent_id, agent_obs in raw_obs.items()}
        return selector_obs

    def _unpack_reset(self, result):
        if isinstance(result, tuple) and len(result) >= 2:
            obs, info = result[0], result[1]
            return obs, info
        return result, {}

    def _unpack_step(self, result):
        if not isinstance(result, tuple):
            raise TypeError("Base env step must return a tuple")
        if len(result) == 5:
            return result
        if len(result) == 4:
            obs, reward, terminated, info = result
            truncated = {agent_id: False for agent_id in terminated if agent_id != "__all__"}
            truncated["__all__"] = False
            if isinstance(terminated, bool):
                terminated = {"__all__": bool(terminated)}
            return obs, reward, terminated, truncated, info
        raise ValueError(f"Unsupported base env step return signature with {len(result)} items")

    def reset(self, *, seed=None, options=None):
        del options
        self._episode_step = 0
        self._last_step_info = {}
        if seed is not None and hasattr(self.base_env, "reset"):
            try:
                result = self.base_env.reset(seed=seed)
            except TypeError:
                result = self.base_env.reset()
        else:
            result = self.base_env.reset()

        raw_obs, info = self._unpack_reset(result)
        raw_obs = dict(raw_obs or {})
        selector_obs = self._build_selector_obs(raw_obs)
        info = info if isinstance(info, Mapping) else {}
        info_dict = {
            agent_id: {
                "valid_candidates": self.K,
                "candidate_cache_hit": False,
                "episode_step": self._episode_step,
            }
            for agent_id in selector_obs.keys()
        }
        for agent_id, payload in selector_obs.items():
            info_dict[agent_id].update(
                {
                    "formation_relation_state": payload["formation_relation_state"].copy(),
                    "control_mode": "trajectory",
                    "intent_valid": True,
                }
            )
            if agent_id in info and isinstance(info[agent_id], Mapping):
                info_dict[agent_id].update(dict(info[agent_id]))
        if self.rllib_reset_compat:
            return selector_obs
        return selector_obs, info_dict

    def step(self, action_dict: Mapping[str, int]):
        if not self._candidate_cache:
            raise RuntimeError("SelectorPlatoonEnv.step() called before reset().")

        prev_candidate_cache = {
            agent_id: {
                key: np.asarray(value, dtype=np.float32).copy() if isinstance(value, np.ndarray) else value
                for key, value in payload.items()
            }
            for agent_id, payload in self._candidate_cache.items()
        }
        traj_actions = {}
        step_info: dict[str, dict[str, Any]] = {}
        active_agent_ids = [agent_id for agent_id in self._agent_ids if agent_id in self._candidate_cache]
        for agent_id in active_agent_ids:
            if agent_id not in action_dict:
                raise KeyError(f"Missing selector action for {agent_id}")
            intent = int(action_dict[agent_id])
            if not (0 <= intent < self.K):
                raise ValueError(f"Invalid intent {intent} for {agent_id}; expected [0, {self.K - 1}]")
            trajs = np.asarray(self._candidate_cache[agent_id]["trajs"], dtype=np.float32)
            traj_actions[agent_id] = trajs[intent].copy()
            step_info[agent_id] = {
                "selected_intent": int(intent),
                "intent_valid": True,
                "candidate_cache_hit": True,
                "control_mode": "trajectory",
            }

        step_result = self.base_env.step(traj_actions)
        raw_obs, _, terminated, truncated, base_info = self._unpack_step(step_result)
        raw_obs = dict(raw_obs or {})
        base_info = dict(base_info or {})

        selector_obs = self._build_selector_obs(raw_obs) if raw_obs else {}
        removed_agents = [agent_id for agent_id in active_agent_ids if agent_id not in raw_obs]
        team_failure = bool(removed_agents)
        shared_team_reward = compute_team_reward(
            {agent_id: [base_info.get(agent_id, {})] for agent_id in active_agent_ids},
            self.reward_config,
        )

        reward: dict[str, float] = {}
        for agent_id in active_agent_ids:
            local_reward = compute_step_reward(base_info.get(agent_id, {}), self.reward_config)
            reward[agent_id] = float(self.lambda_local * local_reward + self.lambda_team * shared_team_reward)
            step_info[agent_id].update(
                {
                    "selector_reward": reward[agent_id],
                    "team_reward": float(shared_team_reward),
                    "formation_relation_state": np.asarray(
                        base_info.get(agent_id, {}).get(
                            "formation_relation_state",
                            prev_candidate_cache[agent_id]["formation_relation_state"],
                        ),
                        dtype=np.float32,
                    ).copy(),
                    "formation_error": float(base_info.get(agent_id, {}).get("formation_error", 0.0)),
                    "progress": float(base_info.get(agent_id, {}).get("progress", 0.0)),
                    "jerk": float(base_info.get(agent_id, {}).get("jerk", 0.0)),
                    "delta_steering": float(base_info.get(agent_id, {}).get("delta_steering", 0.0)),
                    "speed_km_h": float(base_info.get(agent_id, {}).get("speed_km_h", 0.0)),
                    "crash": bool(base_info.get(agent_id, {}).get("crash", False)),
                    "out_of_road": bool(base_info.get(agent_id, {}).get("out_of_road", False)),
                    "min_gap": float(base_info.get(agent_id, {}).get("min_gap", 0.0)),
                }
            )
            if step_info[agent_id]["crash"] or step_info[agent_id]["out_of_road"]:
                team_failure = True

        terminated = dict(terminated or {})
        truncated = dict(truncated or {})
        for agent_id in active_agent_ids:
            terminated.setdefault(agent_id, False)
            truncated.setdefault(agent_id, False)
            if bool(terminated.get(agent_id, False)) or bool(truncated.get(agent_id, False)):
                team_failure = True
        if team_failure:
            for agent_id in active_agent_ids:
                terminated[agent_id] = True
                truncated[agent_id] = False
        terminated["__all__"] = bool(terminated.get("__all__", False))
        truncated["__all__"] = bool(truncated.get("__all__", False))
        if team_failure:
            terminated["__all__"] = True
            truncated["__all__"] = False

        returned_info = {
            agent_id: dict(step_info[agent_id])
            for agent_id in selector_obs.keys()
            if agent_id in step_info
        }

        self._episode_step += 1
        self._last_raw_obs = {agent_id: dict(agent_obs) for agent_id, agent_obs in raw_obs.items()}
        self._last_step_info = {agent_id: dict(info) for agent_id, info in step_info.items()}
        if self.rllib_step_compat:
            # RLlib <=2.4 old-gym API: step returns (obs, rew, done, info)
            done = {k: bool(terminated.get(k, False) or truncated.get(k, False)) for k in terminated}
            done["__all__"] = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            return selector_obs, reward, done, returned_info
        return selector_obs, reward, terminated, truncated, returned_info

    def close(self) -> None:
        close = getattr(self.base_env, "close", None)
        if callable(close):
            close()

    def render(self):
        render = getattr(self.base_env, "render", None)
        if callable(render):
            return render()
        return None

    def get_state(self) -> dict:
        getter = getattr(self.base_env, "get_state", None)
        if callable(getter):
            return getter()
        return {}

    def set_state(self, state: dict) -> None:
        setter = getattr(self.base_env, "set_state", None)
        if callable(setter):
            setter(state)
        self._candidate_cache = {}
        self._last_raw_obs = {}

    def get_current_obs(self) -> dict[str, dict[str, np.ndarray]]:
        getter = getattr(self.base_env, "get_current_obs", None)
        if callable(getter):
            raw_obs = getter()
            if isinstance(raw_obs, tuple) and len(raw_obs) >= 1:
                raw_obs = raw_obs[0]
            return self._build_selector_obs(dict(raw_obs or {}))
        return {}

    def get_last_step_infos(self) -> dict[str, dict[str, Any]]:
        return {agent_id: dict(info) for agent_id, info in self._last_step_info.items()}
