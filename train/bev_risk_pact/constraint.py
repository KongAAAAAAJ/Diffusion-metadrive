from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import RiskPACTConfig


@dataclass(frozen=True)
class RiskConstraintResult:
    trajectory_risk: Tensor
    violation: Tensor
    safe_mask: Tensor
    near_or_unsafe_mask: Tensor
    temporal_weights: Tensor


class RiskLevelSetConstraint:
    """Convert per-step risk into a smooth trajectory-level level-set constraint."""

    def __init__(self, config: RiskPACTConfig | None = None) -> None:
        self.config = config or RiskPACTConfig()

    def __call__(self, risk: Tensor) -> RiskConstraintResult:
        if risk.ndim < 1:
            raise ValueError("risk must have a horizon axis")
        if not risk.is_floating_point() or not bool(torch.isfinite(risk).all()):
            raise ValueError("risk must be finite floating point")
        beta = float(self.config.temporal_softmax_beta)
        temporal_weights = torch.softmax(beta * risk, dim=-1)
        trajectory_risk = (temporal_weights * risk).sum(dim=-1)
        threshold = float(self.config.risk_threshold)
        temperature = float(self.config.violation_temperature)
        raw = (trajectory_risk - threshold) / temperature
        violation = temperature * F.softplus(raw)
        safe_cutoff = threshold - float(self.config.safe_margin)
        safe_mask = trajectory_risk <= safe_cutoff
        near_or_unsafe = ~safe_mask
        # Exact zero supervision for clearly safe trajectories.  This prevents
        # the pilot from learning "safer than necessary" conservative behavior.
        violation = torch.where(safe_mask, torch.zeros_like(violation), violation)
        return RiskConstraintResult(
            trajectory_risk=trajectory_risk,
            violation=violation,
            safe_mask=safe_mask,
            near_or_unsafe_mask=near_or_unsafe,
            temporal_weights=temporal_weights,
        )
