from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .platoon_diffusion_planner import PlatoonDiffusionPlanner


def _extract_single_vehicle_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(ckpt_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("_transfuser_model.") for key in state_dict):
        return {
            key.removeprefix("_transfuser_model."): value
            for key, value in state_dict.items()
            if key.startswith("_transfuser_model.")
        }
    if any(key.startswith("agent._transfuser_model.") for key in state_dict):
        return {
            key.removeprefix("agent._transfuser_model."): value
            for key, value in state_dict.items()
            if key.startswith("agent._transfuser_model.")
        }
    return state_dict


def migrate_single_to_platoon(ckpt_path: str, model: PlatoonDiffusionPlanner) -> PlatoonDiffusionPlanner:
    single_state = _extract_single_vehicle_state_dict(ckpt_path)
    target_state = model.model.state_dict()
    compatible = {}
    for key, value in single_state.items():
        if key == "_status_encoding.weight":
            continue
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            compatible[key] = value

    model.model.load_state_dict(compatible, strict=False)

    if "_status_encoding.weight" in single_state:
        with torch.no_grad():
            nn.init.kaiming_uniform_(model.model._status_encoding.weight[:, 8:], a=5 ** 0.5)
            model.model._status_encoding.weight[:, :8].copy_(single_state["_status_encoding.weight"])
            if "_status_encoding.bias" in single_state:
                model.model._status_encoding.bias.copy_(single_state["_status_encoding.bias"])

    return model
