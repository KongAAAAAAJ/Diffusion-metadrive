"""Risk-field PACT pilot for ChassisFusion."""

from .config import RiskPACTConfig, RiskPACTVisualizationConfig
from .constraint import RiskConstraintResult, RiskLevelSetConstraint
from .curriculum import (
    RiskPACTCurriculumState,
    risk_pact_curriculum_scale,
    risk_pact_curriculum_state,
)
from .risk_field import DynamicGaussianRiskField, RiskFieldResult
from .loss import PACTLiteDistillationLossResult, pact_lite_distillation_loss
from .teacher import PACTTeacherResult, build_x0_pact_teacher
from .training_visualizer import (
    RiskPACTTrainingVisualizer,
    RiskPACTVisualizationEvent,
)

__all__ = [
    "RiskPACTConfig",
    "RiskPACTVisualizationConfig",
    "RiskConstraintResult",
    "RiskLevelSetConstraint",
    "RiskPACTCurriculumState",
    "risk_pact_curriculum_scale",
    "risk_pact_curriculum_state",
    "DynamicGaussianRiskField",
    "RiskFieldResult",
    "PACTLiteDistillationLossResult",
    "pact_lite_distillation_loss",
    "PACTTeacherResult",
    "build_x0_pact_teacher",
    "RiskPACTTrainingVisualizer",
    "RiskPACTVisualizationEvent",
]
