import torch

from models.occupancy import (
    OccupancyPredictor,
    build_reachable_mask,
    ego_local_to_world_xy,
    heatmap_matching,
    world_to_ego_local_xy,
)


def test_predictor_outputs_feasible_occupancy_heatmap_and_identity_gate():
    predictor = OccupancyPredictor(traj_points=8, status_dim=19, grid_h=32, grid_w=32)
    target_line = torch.stack(
        [torch.linspace(0.0, 14.0, 8), torch.zeros(8)],
        dim=-1,
    ).unsqueeze(0)
    status = torch.zeros(1, 19)
    lane_decision = torch.zeros(1, 1)
    base_mask = torch.zeros(1, 32, 32)
    base_mask[:, 12:20, 8:24] = 1.0

    out = predictor(
        target_line,
        status,
        lane_decision,
        base_feasible_mask=base_mask,
    )

    assert "feasible_occupancy_heatmap" in out
    assert out["feasible_occupancy_heatmap"].shape == (1, 32, 32)
    assert torch.allclose(out["feasible_occupancy_heatmap"], base_mask, atol=1e-6)


def test_heatmap_matching_rewards_feasible_candidate_over_offroad_candidate():
    heatmap = torch.zeros(1, 32, 32)
    # Ego-local grid default: lon [-2, 30], lat [-10, 10].
    # This band is centred near y=0 and covers forward positions.
    heatmap[:, 8:24, 13:19] = 1.0

    good = torch.stack([torch.linspace(0.0, 14.0, 8), torch.zeros(8), torch.zeros(8)], dim=-1)
    bad = torch.stack([torch.linspace(0.0, 14.0, 8), torch.full((8,), 8.0), torch.zeros(8)], dim=-1)
    candidates = torch.stack([good, bad], dim=0).unsqueeze(0)

    features = heatmap_matching(candidates, heatmap, grid_h=32, grid_w=32)

    assert features.shape == (1, 2, 6)
    assert features[0, 0, 0] > features[0, 1, 0]
    assert features[0, 0, 3] < features[0, 1, 3]


def test_dynamic_occupancy_overlap_penalizes_candidate():
    heatmap = torch.ones(1, 32, 32)
    dynamic_mask = torch.zeros(1, 32, 32)
    dynamic_mask[:, 8:24, 13:19] = 1.0

    blocked = torch.stack([torch.linspace(0.0, 14.0, 8), torch.zeros(8), torch.zeros(8)], dim=-1)
    clear = torch.stack([torch.linspace(0.0, 14.0, 8), torch.full((8,), 8.0), torch.zeros(8)], dim=-1)
    candidates = torch.stack([blocked, clear], dim=0).unsqueeze(0)

    features = heatmap_matching(
        candidates,
        heatmap,
        dynamic_occupancy_mask=dynamic_mask,
        grid_h=32,
        grid_w=32,
    )

    assert features[0, 0, 4] > features[0, 1, 4]


def test_predictor_accepts_platoon_status_with_extra_relation_feature():
    predictor = OccupancyPredictor(traj_points=8, status_dim=19, grid_h=16, grid_w=16)
    target_line = torch.zeros(1, 8, 2)
    platoon_status = torch.zeros(1, 20)
    out = predictor(target_line, platoon_status, torch.zeros(1, 1))

    assert out["feasible_occupancy_heatmap"].shape == (1, 16, 16)


def test_world_local_coordinate_roundtrip_for_occupancy_inputs():
    pose = torch.tensor([10.0, -2.0, 0.3])
    local = torch.tensor([[0.0, 0.0], [5.0, 1.0], [12.0, -2.0]])

    world = ego_local_to_world_xy(local, pose)
    recovered = world_to_ego_local_xy(world, pose)

    assert torch.allclose(recovered, local, atol=1e-5)


def test_reachable_mask_grows_with_speed():
    slow = build_reachable_mask(1, torch.tensor([1.0]), grid_h=32, grid_w=32)
    fast = build_reachable_mask(1, torch.tensor([8.0]), grid_h=32, grid_w=32)

    assert fast.sum() > slow.sum()
