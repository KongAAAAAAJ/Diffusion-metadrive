from .platoon_normal_planner import (
    CommittedTrajectoryError,
    JointTrajectoryExecutor,
    JointTrajectoryExecutionPlan,
    NormalPlannerKinematicError,
    PlatoonNormalPlanner,
    RolledJointTrajectory,
    TrajectoryExecutionSpec,
)

try:
    from ._relation_encoder import RelationEncoder
    from .platoon_diffusion_planner import PlatoonDiffusionPlanner
    from ._weight_migration import migrate_single_to_platoon
except ModuleNotFoundError:
    RelationEncoder = None
    PlatoonDiffusionPlanner = None
    migrate_single_to_platoon = None

__all__ = [
    "NormalPlannerKinematicError",
    "CommittedTrajectoryError",
    "JointTrajectoryExecutor",
    "JointTrajectoryExecutionPlan",
    "RelationEncoder",
    "PlatoonDiffusionPlanner",
    "PlatoonNormalPlanner",
    "RolledJointTrajectory",
    "TrajectoryExecutionSpec",
    "migrate_single_to_platoon",
]
