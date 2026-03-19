"""MetaDrive TransFuser package."""

from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from metadrive.policy.diffusion_policy.transfuser_policy import TransfuserPolicy

try:
    from metadrive.policy.diffusion_policy.transfuser_agent import TransfuserAgent
except ModuleNotFoundError:
    TransfuserAgent = None

__all__ = ["TransfuserAgent", "TransfuserConfig", "TransfuserPolicy"]
