import inspect

from metadrive.exp_dataset import collect_expert


def test_collect_expert_uses_base_idm_policy_signature():
    signature = inspect.signature(collect_expert.Expert.__init__)

    assert "control_object" in signature.parameters
    assert "random_seed" in signature.parameters
    assert "style_profile" not in signature.parameters
