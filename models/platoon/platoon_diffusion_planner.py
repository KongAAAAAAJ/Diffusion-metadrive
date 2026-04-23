from __future__ import annotations

import copy
from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from metadrive.policy.diffusion_policy.transfuser_model_v2 import V2TransfuserModel
from metadrive.policy.diffusion_policy.modules.blocks import gen_sineembed_for_position

from .relation_encoder import RelationEncoder


class PlatoonDiffusionPlanner(nn.Module):
    def __init__(self, config: TransfuserConfig, num_vehicles: int = 3):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.num_vehicles = int(num_vehicles)
        self.inference_seed = 0
        self.model = V2TransfuserModel(self.config)
        self.relation_encoder = RelationEncoder(12, 64, 12)
        self._expand_status_encoding()

    def _expand_status_encoding(self) -> None:
        old_layer = self.model._status_encoding
        in_features = int(self.config.status_feature_dim) + 12
        expanded = nn.Linear(in_features, old_layer.out_features)
        nn.init.kaiming_uniform_(expanded.weight, a=5 ** 0.5)
        with torch.no_grad():
            expanded.weight[:, : old_layer.in_features].copy_(old_layer.weight)
            expanded.bias.copy_(old_layer.bias)
        self.model._status_encoding = expanded

    @staticmethod
    def _ensure_batch_dim(value: Tensor) -> Tensor:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 1:
            return value.unsqueeze(0)
        if value.ndim == 3:
            return value.unsqueeze(0)
        return value

    def _build_model_inputs(
        self, batch: Mapping[str, Mapping[str, Tensor]]
    ) -> tuple[list[str], Dict[str, Tensor], Dict[str, dict]]:
        agent_ids = list(batch.keys())
        camera = []
        lidar = []
        status = []
        relation = []
        agent_contexts: Dict[str, dict] = {}
        for agent_id in agent_ids:
            sample = batch[agent_id]
            camera_tensor = self._ensure_batch_dim(sample["camera"])
            lidar_tensor = self._ensure_batch_dim(sample["lidar"])
            status_tensor = self._ensure_batch_dim(sample["status"])
            relation_tensor = self._ensure_batch_dim(sample["formation_relation_state"])
            camera.append(camera_tensor)
            lidar.append(lidar_tensor)
            status.append(status_tensor)
            relation.append(relation_tensor)
            agent_contexts[agent_id] = {
                "camera": camera_tensor,
                "lidar": lidar_tensor,
                "status": status_tensor,
                "formation_relation_state": relation_tensor,
            }

        camera_feature = torch.cat(camera, dim=0).float()
        lidar_feature = torch.cat(lidar, dim=0).float()
        status_feature = torch.cat(status, dim=0).float()
        relation_feature = torch.cat(relation, dim=0).float()
        relation_embedding = self.relation_encoder(relation_feature)
        fused_status = torch.cat([status_feature, relation_embedding], dim=-1)

        model_inputs = {
            "camera_feature": camera_feature,
            "lidar_feature": lidar_feature,
            "status_feature": fused_status,
        }
        return agent_ids, model_inputs, agent_contexts

    def forward(self, batch: Mapping[str, Mapping[str, Tensor]]) -> Dict[str, Tensor]:
        if not batch:
            return {}
        agent_ids, model_inputs, _ = self._build_model_inputs(batch)
        outputs = self._forward_model(model_inputs)
        trajectories = outputs["trajectory"]
        return {agent_id: trajectories[idx] for idx, agent_id in enumerate(agent_ids)}

    def forward_with_preference(
        self,
        batch: Mapping[str, Mapping[str, Tensor]],
        preference_points: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        if not batch:
            return {}
        agent_ids, model_inputs, _ = self._build_model_inputs(batch)
        points = []
        for agent_id in agent_ids:
            if agent_id not in preference_points:
                raise KeyError(f"Missing co-preference target point for {agent_id!r}.")
            point = self._ensure_batch_dim(torch.as_tensor(preference_points[agent_id])).float()
            points.append(point[:, :2])
        model_inputs["preference_point"] = torch.cat(points, dim=0).to(model_inputs["status_feature"].device)
        outputs = self._forward_model(model_inputs)
        trajectories = outputs["trajectory"]
        return {agent_id: trajectories[idx] for idx, agent_id in enumerate(agent_ids)}

    def freeze_for_co_preference(self) -> "PlatoonDiffusionPlanner":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def _forward_model(self, model_inputs: Mapping[str, Tensor], return_multimodal: bool = False) -> Dict[str, Tensor]:
        if self.training:
            if return_multimodal:
                return self.model.infer_multimodal(model_inputs)
            return self.model(model_inputs)

        devices = []
        if model_inputs["camera_feature"].is_cuda:
            devices = [model_inputs["camera_feature"].device]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.inference_seed)
            if devices:
                torch.cuda.manual_seed_all(self.inference_seed)
            if return_multimodal:
                return self.model.infer_multimodal(model_inputs)
            return self.model(model_inputs)

    # ------------------------------------------------------------------
    # RL helpers: backbone context extraction + per-step denoising
    # ------------------------------------------------------------------

    def extract_rl_context(self, batch: Mapping[str, Mapping[str, Tensor]]) -> Tuple[Dict[str, dict], list]:
        """Run backbone + transformer for all agents.

        Returns
        -------
        contexts : dict[agent_id -> context_dict]
            Each context_dict contains the tensors needed for per-step denoising:
            ego_query, agents_query, cross_bev_feature, bev_spatial_shape,
            status_encoding_token  (all with batch_size=1 per agent).
        agent_ids : list[str]
            Ordered list of agent IDs from batch.
        """
        agent_ids = list(batch.keys())
        m = self.model

        # Collect per-agent inputs (same aggregation as forward())
        contexts: Dict[str, dict] = {}
        for agent_id in agent_ids:
            sample = batch[agent_id]
            camera = self._ensure_batch_dim(sample["camera"]).float()          # [1, 3, H, W]
            lidar = self._ensure_batch_dim(sample["lidar"]).float()            # [1, 1, H, W]
            status = self._ensure_batch_dim(sample["status"]).float()          # [1, 8]
            relation = self._ensure_batch_dim(sample["formation_relation_state"]).float()  # [1, 12]

            relation_emb = self.relation_encoder(relation)
            fused_status = torch.cat([status, relation_emb], dim=-1)           # [1, 20]

            bs = fused_status.shape[0]  # always 1
            bev_upscale, bev_feat, _ = m._backbone(camera, lidar)
            bev_spatial_shape = bev_upscale.shape[2:]
            concat_bev_shape = bev_feat.shape[2:]

            bev_flat = m._bev_downscale(bev_feat).flatten(-2, -1).permute(0, 2, 1)
            status_enc = m._status_encoding(fused_status)           # [1, tf_d_model]

            keyval = torch.cat([bev_flat, status_enc[:, None]], dim=1)
            keyval = keyval + m._keyval_embedding.weight[None, ...]

            concat_cross = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
                bs, -1, concat_bev_shape[0], concat_bev_shape[1])
            concat_cross = F.interpolate(concat_cross, size=bev_spatial_shape,
                                         mode="bilinear", align_corners=False)
            cross_bev = torch.cat([concat_cross, bev_upscale], dim=1)
            cross_bev = m.bev_proj(cross_bev.flatten(-2, -1).permute(0, 2, 1))
            cross_bev = cross_bev.permute(0, 2, 1).contiguous().view(
                bs, -1, bev_spatial_shape[0], bev_spatial_shape[1])

            query = m._query_embedding.weight[None, ...].repeat(bs, 1, 1)
            query_out = m._tf_decoder(query, keyval)
            ego_query, agents_query = query_out.split(m._query_splits, dim=1)

            contexts[agent_id] = {
                "ego_query": ego_query,                        # [1, 1, D]
                "agents_query": agents_query,                  # [1, N_bbox, D]
                "cross_bev_feature": cross_bev,               # [1, C, H, W]
                "bev_spatial_shape": bev_spatial_shape,
                "status_encoding_token": status_enc[:, None], # [1, 1, D]
            }
        return contexts, agent_ids

    def predict_denoised_traj(
        self,
        noisy_traj_norm: Tensor,   # [B, num_modes, 8, 2]  normalized xy
        timestep: Tensor,           # [B] int64
        context: dict,
    ) -> Tensor:                    # [B, num_modes, 8, 2]  predicted denoised (normalized)
        """Single denoising step via the trajectory head's diff_decoder.

        The context must have been produced by ``extract_rl_context`` for the
        same agent and then repeated along dim-0 to match B.
        """
        th = self.model._trajectory_head
        bs, num_modes = noisy_traj_norm.shape[:2]

        # FIX (Problem F): clamp to [-1, 1] before denorm, matching reference
        # forward_train_rl line 826: x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
        noisy_traj_norm = torch.clamp(noisy_traj_norm, min=-1, max=1)
        noisy_pts = th.denorm_odo(noisy_traj_norm)                      # physical space
        pos_embed = gen_sineembed_for_position(noisy_pts, hidden_dim=64)
        traj_feat = th.plan_anchor_encoder(pos_embed.flatten(-2))       # [B, num_modes, D]
        traj_feat = traj_feat.view(bs, num_modes, -1)

        time_embed = th.time_mlp(timestep).view(bs, 1, -1)             # [B, 1, D]

        poses_reg_list, _ = th.diff_decoder(
            traj_feat,
            noisy_pts,
            context["cross_bev_feature"],
            context["bev_spatial_shape"],
            context["agents_query"],
            context["ego_query"],
            time_embed,
            context["status_encoding_token"],
            None,  # global_img
        )
        x_start = poses_reg_list[-1][..., :2]                          # [B, num_modes, 8, 2]
        return th.norm_odo(x_start)[..., :2]                           # normalized xy only
