from __future__ import annotations

import torch
from torch import nn

from models.platoon_planner._relation_encoder import RelationEncoder


def test_relation_encoder_contract_and_stability():
    encoder = RelationEncoder()
    assert isinstance(encoder, nn.Module)

    input_single = torch.randn(1, 12)
    output_single = encoder(input_single)
    assert tuple(output_single.shape) == (1, 12)

    input_batch = torch.randn(8, 12)
    output_batch = encoder(input_batch)
    assert tuple(output_batch.shape) == (8, 12)

    parameter_count = sum(param.numel() for param in encoder.parameters())
    assert parameter_count < 10000
    assert any(isinstance(module, nn.LayerNorm) for module in encoder.modules())

    for _ in range(3):
        out = encoder(torch.randn(4, 12))
        assert not torch.isnan(out).any()
