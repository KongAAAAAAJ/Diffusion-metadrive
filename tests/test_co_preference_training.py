import torch

from models.co_preference.model import CoPreferenceModel
from train.train_co_preference import imitation_loss


def test_imitation_loss_uses_teacher_topology_and_s_labels() -> None:
    model = CoPreferenceModel(status_dim=8, relation_dim=12, hidden=(16,))
    batch = {
        "status_feature": torch.zeros((4, 8), dtype=torch.float32),
        "formation_relation_state": torch.zeros((4, 12), dtype=torch.float32),
        "topology_mask": torch.ones((4, 4), dtype=torch.bool),
        "teacher_topology_choice": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "teacher_s": torch.tensor([0.1, 0.3, 0.7, 0.9], dtype=torch.float32),
    }

    loss, metrics = imitation_loss(model, batch)

    assert loss.ndim == 0
    assert loss.requires_grad
    assert metrics["topology_loss"] > 0.0
    assert metrics["s_loss"] >= 0.0
