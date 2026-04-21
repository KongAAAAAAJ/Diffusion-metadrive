from typing import Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from metadrive.policy.diffusion_policy.transfuser_features import BoundingBox2DIndex


def _zero_like_loss(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
    reference = predictions.get("trajectory")
    if reference is None:
        return torch.tensor(0.0, dtype=torch.float32)
    return reference.new_zeros(())


def _select_topology_supervised_trajectory(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    trajectory = predictions.get("trajectory")
    candidates = predictions.get("trajectory_candidates_train")
    mode_labels = targets.get("hierarchical_mode_label")
    if trajectory is None:
        return None
    if candidates is None or mode_labels is None:
        return trajectory

    mode_labels = mode_labels.to(device=candidates.device, dtype=torch.long).view(-1)
    batch_size = candidates.shape[0]
    valid_mask = torch.logical_and(mode_labels >= 0, mode_labels < candidates.shape[1])
    if mode_labels.shape[0] != batch_size:
        return trajectory
    safe_labels = mode_labels.clamp(min=0, max=candidates.shape[1] - 1)
    gather_idx = safe_labels[:, None, None, None].expand(-1, 1, candidates.shape[2], candidates.shape[3])
    gathered = torch.gather(candidates, 1, gather_idx).squeeze(1)
    if torch.all(valid_mask):
        return gathered
    trajectory = trajectory.to(device=gathered.device, dtype=gathered.dtype)
    valid_mask = valid_mask[:, None, None]
    return torch.where(valid_mask, gathered, trajectory)


def _nearest_polyline_segment_metrics(
    points_xy: torch.Tensor,
    topology_polyline: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seg_start = topology_polyline[:, :-1, :]
    seg_end = topology_polyline[:, 1:, :]
    seg_vec = seg_end - seg_start
    seg_len_sq = (seg_vec * seg_vec).sum(dim=-1)
    valid_seg_mask = seg_len_sq > 1e-6

    point_expanded = points_xy[:, :, None, :]
    start_expanded = seg_start[:, None, :, :]
    vec_expanded = seg_vec[:, None, :, :]
    len_sq_expanded = seg_len_sq[:, None, :]

    projection = ((point_expanded - start_expanded) * vec_expanded).sum(dim=-1) / len_sq_expanded.clamp_min(1e-6)
    projection = projection.clamp(0.0, 1.0)
    closest = start_expanded + projection[..., None] * vec_expanded
    distances = torch.linalg.norm(point_expanded - closest, dim=-1)
    distances = torch.where(valid_seg_mask[:, None, :], distances, torch.full_like(distances, float("inf")))

    nearest_idx = distances.argmin(dim=-1)
    batch_idx = torch.arange(points_xy.shape[0], device=points_xy.device)[:, None]
    closest_distance = distances[batch_idx, torch.arange(points_xy.shape[1], device=points_xy.device)[None, :], nearest_idx]
    nearest_tangent = seg_vec[batch_idx, nearest_idx]
    nearest_tangent = F.normalize(nearest_tangent, dim=-1, eps=1e-6)
    has_valid_segment = valid_seg_mask.any(dim=-1, keepdim=True).expand(-1, points_xy.shape[1])
    return closest_distance, nearest_tangent, has_valid_segment


def _topology_consistency_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: TransfuserConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topology_polyline = targets.get("topology_polyline")
    trajectory = _select_topology_supervised_trajectory(predictions, targets)
    if topology_polyline is None or trajectory is None:
        zero = _zero_like_loss(predictions)
        return zero, zero, zero

    topology_polyline = topology_polyline.to(device=trajectory.device, dtype=trajectory.dtype)
    trajectory_xy = trajectory[..., :2]
    if topology_polyline.ndim != 3 or topology_polyline.shape[1] < 2:
        zero = _zero_like_loss(predictions)
        return zero, zero, zero

    point_distances, nearest_tangent, point_valid_mask = _nearest_polyline_segment_metrics(trajectory_xy, topology_polyline)
    corridor_penalty = F.relu(point_distances - float(config.corridor_half_width_m))
    if torch.any(point_valid_mask):
        corridor_loss = corridor_penalty[point_valid_mask].mean()
    else:
        corridor_loss = _zero_like_loss(predictions)

    trajectory_dirs = trajectory_xy[:, 1:, :] - trajectory_xy[:, :-1, :]
    dir_norm = torch.linalg.norm(trajectory_dirs, dim=-1)
    dir_valid_mask = torch.logical_and(point_valid_mask[:, :-1], dir_norm > 1e-6)
    normalized_dirs = F.normalize(trajectory_dirs, dim=-1, eps=1e-6)
    cosine = (normalized_dirs * nearest_tangent[:, :-1, :]).sum(dim=-1).clamp(-1.0, 1.0)
    lane_direction_penalty = 1.0 - cosine
    if torch.any(dir_valid_mask):
        lane_direction_loss = lane_direction_penalty[dir_valid_mask].mean()
    else:
        lane_direction_loss = _zero_like_loss(predictions)

    topology_loss = float(config.topology_weight) * (
        float(config.lane_direction_weight) * lane_direction_loss
        + float(config.corridor_weight) * corridor_loss
    )
    return topology_loss, lane_direction_loss, corridor_loss


def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    # import ipdb; ipdb.set_trace()
    if "trajectory_loss" in predictions:
        trajectory_loss = predictions["trajectory_loss"]
    else:
        trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(
        predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
    )
    if 'diffusion_loss' in predictions:
        diffusion_loss = predictions['diffusion_loss']
    else:
        diffusion_loss = 0
    topology_loss, lane_direction_loss, corridor_loss = _topology_consistency_loss(
        predictions=predictions,
        targets=targets,
        config=config,
    )
    loss = (
        config.trajectory_weight * trajectory_loss
        + config.diff_loss_weight * diffusion_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
        + topology_loss
    )
    loss_dict = {
        'loss': loss,
        'trajectory_loss': config.trajectory_weight*trajectory_loss,
        'diffusion_loss': config.diff_loss_weight*diffusion_loss,
        'agent_class_loss': config.agent_class_weight*agent_class_loss,
        'agent_box_loss': config.agent_box_weight*agent_box_loss,
        'bev_semantic_loss': config.bev_semantic_weight*bev_semantic_loss,
        'topology_loss': topology_loss,
        'lane_direction_loss': float(config.topology_weight) * float(config.lane_direction_weight) * lane_direction_loss,
        'corridor_loss': float(config.topology_weight) * float(config.corridor_weight) * corridor_loss,
    }
    if "trajectory_loss_dict" in predictions:
        trajectory_loss_dict = predictions["trajectory_loss_dict"]
        loss_dict.update(trajectory_loss_dict)
    # import ipdb; ipdb.set_trace()
    return loss_dict


def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
