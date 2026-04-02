from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .intent_selector import IntentSelectorActor, IntentSelectorCritic

if not hasattr(np, "bool8"):  # pragma: no cover - ray 2.4 expects this legacy alias
    np.bool8 = np.bool_

try:  # pragma: no cover - ray is optional in the local dev environment
    from ray.rllib.models import ModelV2
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover
    ModelV2 = object  # type: ignore
    TorchModelV2 = nn.Module  # type: ignore
    _RAY_AVAILABLE = False


def _flatten_tensor(value: Tensor) -> Tensor:
    if value.ndim <= 1:
        return value
    return value.reshape(value.shape[0], -1)


def _as_tensor(value: object, device: torch.device | None = None) -> Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _infer_dim(value: object) -> int:
    if hasattr(value, "shape"):
        shape = tuple(int(dim) for dim in getattr(value, "shape"))
        if not shape:
            return 1
        return int(np.prod(shape))
    return int(np.prod(np.asarray(value).shape))


def _space_dim(space) -> int:
    from collections.abc import Mapping as _Mapping

    if hasattr(space, "spaces") and isinstance(space.spaces, dict):
        return sum(_space_dim(subspace) for subspace in space.spaces.values())
    if hasattr(space, "shape") and getattr(space, "shape") is not None:
        return int(np.prod(space.shape))
    if hasattr(space, "n"):
        return int(space.n)
    if isinstance(space, _Mapping):
        return sum(_space_dim(subspace) for subspace in space.values())
    raise TypeError(f"Unsupported observation space: {space!r}")


def _concat_obs_fields(obs: Mapping[str, object], keys: Sequence[str]) -> Tensor:
    tensors: list[Tensor] = []
    for key in keys:
        if key not in obs:
            continue
        tensor = _as_tensor(obs[key])
        tensors.append(_flatten_tensor(tensor))
    if not tensors:
        raise ValueError(f"No selector observation fields found in keys={list(keys)}")
    return torch.cat(tensors, dim=-1)


def _reshape_obs_field(key: str, tensor: Tensor, *, num_modes: int, mode_embedding_dim: int, summary_dim: int) -> Tensor:
    if key == "mode_embeddings":
        return tensor.reshape(tensor.shape[0], num_modes, mode_embedding_dim)
    if key == "candidate_summary":
        return tensor.reshape(tensor.shape[0], num_modes, summary_dim)
    return tensor


class _IntentSelectorRLlibModelBase:
    def _require_ray(self) -> None:
        if not _RAY_AVAILABLE:
            raise RuntimeError(
                "RLlib is not available in this environment. "
                "Install ray[rllib]==2.4.0 before training the selector MAPPO policy."
            )


if _RAY_AVAILABLE:
    class IntentSelectorRLlibModel(TorchModelV2, nn.Module, _IntentSelectorRLlibModelBase):
        """TorchModelV2 adapter for shared-actor + centralized-critic MAPPO."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            self._require_ray()
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            nn.Module.__init__(self)

            custom_config = dict(model_config.get("custom_model_config", {}))
            original_obs_space = getattr(obs_space, "original_space", obs_space)
            self.actor_obs_keys = tuple(
                custom_config.get(
                    "actor_obs_keys",
                    ("agent_context", "formation_relation_state", "mode_embeddings", "candidate_summary", "action_mask"),
                )
            )
            self.critic_obs_key = str(custom_config.get("critic_obs_key", "global_state"))
            hidden = custom_config.get("hidden", (256, 256))

            if hasattr(action_space, "n"):
                self.num_modes = int(action_space.n)
            else:
                self.num_modes = int(custom_config.get("num_modes", _space_dim(action_space)))

            if hasattr(original_obs_space, "spaces") and isinstance(original_obs_space.spaces, dict):
                if self.critic_obs_key not in original_obs_space.spaces:
                    raise KeyError(f"critic_obs_key={self.critic_obs_key!r} missing from observation space")
                self.obs_component_dims = {
                    key: _space_dim(space) for key, space in original_obs_space.spaces.items()
                }
                self.obs_component_order = tuple(original_obs_space.spaces.keys())
                self.agent_context_dim = _space_dim(original_obs_space.spaces["agent_context"])
                self.relation_dim = _space_dim(original_obs_space.spaces["formation_relation_state"])
                mode_space = original_obs_space.spaces["mode_embeddings"]
                summary_space = original_obs_space.spaces["candidate_summary"]
                self.mode_embedding_dim = int(mode_space.shape[-1])
                self.summary_dim = int(summary_space.shape[-1])
                self.critic_obs_dim = _space_dim(original_obs_space.spaces[self.critic_obs_key])
            else:
                raise TypeError("IntentSelectorRLlibModel requires dict observation spaces.")

            self.actor = IntentSelectorActor(
                agent_context_dim=self.agent_context_dim,
                relation_dim=self.relation_dim,
                mode_embedding_dim=self.mode_embedding_dim,
                summary_dim=self.summary_dim,
                num_modes=self.num_modes,
                hidden=hidden,
            )
            self.critic = IntentSelectorCritic(self.critic_obs_dim, hidden=hidden)
            self._last_global_state: Tensor | None = None

        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs"]
            if not isinstance(obs, Mapping):
                flat_obs = _as_tensor(obs)
                parsed_obs: dict[str, Tensor] = {}
                cursor = 0
                for key in self.obs_component_order:
                    dim = int(self.obs_component_dims[key])
                    parsed_obs[key] = flat_obs[:, cursor : cursor + dim]
                    cursor += dim
                obs = {
                    key: _reshape_obs_field(
                        key,
                        value,
                        num_modes=self.num_modes,
                        mode_embedding_dim=self.mode_embedding_dim,
                        summary_dim=self.summary_dim,
                    )
                    for key, value in parsed_obs.items()
                }
            agent_context = _as_tensor(obs["agent_context"])
            relation = _as_tensor(obs["formation_relation_state"], device=agent_context.device)
            mode_embeddings = _as_tensor(obs["mode_embeddings"], device=agent_context.device)
            candidate_summary = _as_tensor(obs["candidate_summary"], device=agent_context.device)
            global_state = _as_tensor(obs[self.critic_obs_key], device=agent_context.device)
            self._last_global_state = global_state

            logits = self.actor(
                agent_context=agent_context,
                formation_relation_state=relation,
                mode_embeddings=mode_embeddings,
                candidate_summary=candidate_summary,
            )
            action_mask = obs.get("action_mask")
            if action_mask is not None:
                mask = _as_tensor(action_mask, device=logits.device)
                logits = logits + torch.log(mask.clamp(min=1e-6))
            return logits, state

        def value_function(self):
            if self._last_global_state is None:
                raise RuntimeError("value_function() called before forward(); global state cache is empty.")
            return self.critic(self._last_global_state)
else:
    class IntentSelectorRLlibModel(nn.Module, _IntentSelectorRLlibModelBase):
        """Import-friendly fallback when ray is unavailable."""

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._require_ray()
