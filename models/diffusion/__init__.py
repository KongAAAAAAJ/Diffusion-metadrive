"""MetaDrive TransFuser package with lazy optional training imports."""

from __future__ import annotations

from typing import Any


__all__ = ["TransfuserAgent", "TransfuserConfig", "TransfuserPolicy"]


def __getattr__(name: str) -> Any:
    if name == "TransfuserConfig":
        from models.diffusion.transfuser_config import TransfuserConfig

        return TransfuserConfig
    if name == "TransfuserPolicy":
        from models.diffusion.transfuser_policy import TransfuserPolicy

        return TransfuserPolicy
    if name == "TransfuserAgent":
        try:
            from models.diffusion.transfuser_agent import TransfuserAgent
        except ModuleNotFoundError:
            return None
        return TransfuserAgent
    raise AttributeError(name)
