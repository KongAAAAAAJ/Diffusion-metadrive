"""MetaDrive TransFuser package."""

from models.diffusion.transfuser_config import TransfuserConfig
from models.diffusion.transfuser_policy import TransfuserPolicy

try:
    from models.diffusion.transfuser_agent import TransfuserAgent
except ModuleNotFoundError:
    TransfuserAgent = None

__all__ = ["TransfuserAgent", "TransfuserConfig", "TransfuserPolicy"]
