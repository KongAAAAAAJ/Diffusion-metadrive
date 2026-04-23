from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn

from .model import CoPreferenceModel

if not hasattr(np, "bool8"):  # pragma: no cover - ray 2.x compatibility
    np.bool8 = np.bool_

try:  # pragma: no cover - ray is optional in local tests
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover
    TorchModelV2 = object  # type: ignore
    _RAY_AVAILABLE = False


def _as_tensor(value, device=None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class _CoPreferenceRLlibModelBase:
    def _require_ray(self) -> None:
        if not _RAY_AVAILABLE:
            raise RuntimeError("Install ray[rllib] before running co-preference PPO training.")


if _RAY_AVAILABLE:
    class CoPreferenceRLlibModel(TorchModelV2, nn.Module, _CoPreferenceRLlibModelBase):
        """RLlib adapter producing topology logits plus a continuous s action head."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            self._require_ray()
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            nn.Module.__init__(self)
            custom = dict(model_config.get("custom_model_config", {}))
            status_dim = int(custom.get("status_dim", 8))
            relation_dim = int(custom.get("relation_dim", 12))
            hidden = custom.get("hidden", (256, 256))
            self.policy = CoPreferenceModel(status_dim=status_dim, relation_dim=relation_dim, hidden=hidden)
            self.value_head = nn.Sequential(
                nn.Linear(status_dim + relation_dim, int(custom.get("value_hidden", 256))),
                nn.ReLU(),
                nn.Linear(int(custom.get("value_hidden", 256)), 1),
            )
            self._last_value_input: torch.Tensor | None = None

        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs"]
            if not isinstance(obs, Mapping):
                raise TypeError("CoPreferenceRLlibModel expects dict observations.")
            status = _as_tensor(obs["status"], device=next(self.parameters()).device)
            relation = _as_tensor(obs["formation_relation_state"], device=status.device)
            mask = obs.get("co_preference_topology_mask")
            mask_tensor = _as_tensor(mask, device=status.device).bool() if mask is not None else None
            out = self.policy(
                status_feature=status,
                formation_relation_state=relation,
                topology_mask=mask_tensor,
            )
            self._last_value_input = torch.cat([status, relation], dim=-1)
            # RLlib's default Torch distributions consume a flat vector.  The
            # first four dimensions are categorical logits; the final dimension
            # is a bounded preference scalar in [0, 1].
            return torch.cat([out["topology_logits"], out["s"].unsqueeze(-1)], dim=-1), state

        def value_function(self):
            if self._last_value_input is None:
                raise RuntimeError("value_function() called before forward().")
            return self.value_head(self._last_value_input).squeeze(-1)
else:
    class CoPreferenceRLlibModel(nn.Module, _CoPreferenceRLlibModelBase):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._require_ray()
