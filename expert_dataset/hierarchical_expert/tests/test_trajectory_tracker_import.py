from importlib import import_module

from metadrive.policy.idm_policy import IDMPolicy


def test_pure_pursuit_tracker_imports_with_idm_steering_limit():
    module = import_module("expert_dataset.hierarchical_expert.trajectory_tracker")

    assert module.PurePursuitTracker.MAX_STEERING == IDMPolicy.MAX_STEERING_ANGLE
