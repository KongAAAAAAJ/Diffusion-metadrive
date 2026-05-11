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


def is_platoon_grpo_checkpoint(ckpt_path: str) -> bool:
    """Return True if the checkpoint was saved by the GRPO trainer (PlatoonDiffusionPlanner format)."""
    ckpt = torch.load(Path(ckpt_path), map_location="cpu")
    model_state = ckpt.get("model_state", {})
    return bool(model_state) and any(k.startswith("model.") for k in model_state)


def load_platoon_grpo_checkpoint(ckpt_path: str, model: PlatoonDiffusionPlanner) -> PlatoonDiffusionPlanner:
    """Load a full PlatoonDiffusionPlanner checkpoint saved by the GRPO trainer.

    Unlike migrate_single_to_platoon, this loads the complete platoon state
    (including relation_encoder and the fine-tuned plan_cls_branch) in one shot.
    Missing or shape-mismatched keys are silently skipped (strict=False).
    """
    ckpt = torch.load(Path(ckpt_path), map_location="cpu")
    model_state: dict[str, torch.Tensor] = ckpt["model_state"]
    result = model.load_state_dict(model_state, strict=False)
    missing = [k for k in result.missing_keys if "relation_encoder" not in k]
    if missing:
        print(f"[weight_migration] load_platoon_grpo_checkpoint: {len(missing)} missing keys (first 5: {missing[:5]})")
    return model


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
        single_w = single_state["_status_encoding.weight"]
        single_dim = single_w.shape[1]  # single-vehicle status dim (e.g. 19)
        with torch.no_grad():
            nn.init.kaiming_uniform_(model.model._status_encoding.weight[:, single_dim:], a=5 ** 0.5)
            model.model._status_encoding.weight[:, :single_dim].copy_(single_w)
            if "_status_encoding.bias" in single_state:
                model.model._status_encoding.bias.copy_(single_state["_status_encoding.bias"])

    return model
