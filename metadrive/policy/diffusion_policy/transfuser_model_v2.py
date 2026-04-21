from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from metadrive.policy.diffusion_policy.transfuser_backbone import TransfuserBackbone
from metadrive.policy.diffusion_policy.transfuser_features import BoundingBox2DIndex
from diffusers.schedulers import DDIMScheduler
from metadrive.policy.diffusion_policy.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from metadrive.policy.diffusion_policy.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from metadrive.policy.diffusion_policy.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union


class StateSE2Index:
    X = 0
    Y = 1
    HEADING = 2

    
class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]   # *1用于预测自车轨迹, num_bounding_boxes用于预测周车轨迹

        self._config = config
        self._backbone = TransfuserBackbone(config)  # *感知主干网络，主要用来提取camera和lidar的特征，并将它们融合成BEV特征

        self._keyval_embedding = nn.Embedding(
            config.lidar_vert_anchors * config.lidar_horz_anchors + 1, config.tf_d_model
        )
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)  # *ego feature (1) + agent feature (num_bounding_boxes) --> 映射到256*256维

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(self._backbone.num_features, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(config.status_feature_dim, config.tf_d_model)  # *将状态信息（19D ego_state + navi）映射成与Transformer输入维度相同的特征
        self._target_point_mlp = nn.Sequential(
            nn.Linear(2, config.tf_d_model),
            nn.ReLU(),
            nn.Linear(config.tf_d_model, config.target_point_dim),
            nn.ReLU(),
        )
        self._target_line_mlp = nn.Sequential(
            nn.Linear(2 * config.target_line_num_points, config.tf_d_model),
            nn.ReLU(),
            nn.Linear(config.tf_d_model, config.target_point_dim),
            nn.ReLU(),
        )

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )   # *语义分割

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)  # *Transformer解码器

        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )   # *周车轨迹预测头

        self._trajectory_head = TrajectoryHead(
            num_poses=int(config.trajectory_sampling.time_horizon // config.trajectory_sampling.interval_length),
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )   # *自车轨迹预测头

        self.bev_proj = nn.Sequential(
            *linear_relu_ln(config.tf_d_model, 1, 1, config.tf_d_model + config.bev_features_channels),
        )


    def _encode_target_guidance(self, features: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self._config.target_guidance_type == "line":
            target_line: Optional[torch.Tensor] = features.get("target_line")
            if target_line is None:
                target_point: Optional[torch.Tensor] = features.get("target_point")
                if target_point is None:
                    return None
                ratios = torch.linspace(
                    0.0,
                    1.0,
                    self._config.target_line_num_points,
                    dtype=target_point.dtype,
                    device=target_point.device,
                ).view(1, -1, 1)
                target_line = ratios * target_point[:, None, :2]
            if target_line.shape[1] != self._config.target_line_num_points:
                target_line = F.interpolate(
                    target_line.permute(0, 2, 1),
                    size=self._config.target_line_num_points,
                    mode="linear",
                    align_corners=True,
                ).permute(0, 2, 1)
            target_line = target_line.reshape(target_line.shape[0], -1)
            return self._target_line_mlp(target_line)

        target_point: Optional[torch.Tensor] = features.get("target_point")
        return self._target_point_mlp(target_point) if target_point is not None else None


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        # *单样本 shape=[3, 256, 1024]，batch 后 shape=[B, 3, 256, 1024]
        camera_feature: torch.Tensor = features["camera_feature"]  

        # *单样本 shape=[1, 256, 256]，batch 后 shape=[B, 1, 256, 256]
        lidar_feature: torch.Tensor = features["lidar_feature"]

        # *单样本 shape=[19]，batch 后 shape=[B, 19]
        status_feature: torch.Tensor = features["status_feature"]  # *导航命令[left, straight, right, lane_follow/other] + 车速[vx,vy] + 加速度[ax,ay]
        batch_size = status_feature.shape[0]
        target_point_embed = self._encode_target_guidance(features)

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)  # *bev_feature是下采样后的BEV特征(512*512)，
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)  # *将BEV特征压由512维压缩到256维，并展平为(batch_size, 256, num_bev_pixels)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)  # 8*8 --> 256*256 

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)  # *query是可学习的参数向量，在训练过程中进行优化
        query_out = self._tf_decoder(query, keyval)  # *Transformer解码器的输入是query和keyval，其中keyval包含了BEV特征和状态编码，输出是经过多层Transformer解码器处理后的特征表示

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)  # *bev特征图进行语义分割，生成每个像素的语义类别概率图
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)  # *将query分割成ego query和agent query

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        coarse_trajectories = features.get("coarse_trajectories")
        mode_valid_mask = features.get("mode_valid_mask")

        trajectory = self._trajectory_head(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding[:, None],
            targets=targets,
            global_img=None,
            target_point_embed=target_point_embed,
            coarse_trajectories=coarse_trajectories,
            mode_valid_mask=mode_valid_mask,
        )
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

    def infer_multimodal(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module inference pass exposing selector-facing multimodal outputs."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        batch_size = status_feature.shape[0]
        target_point_embed = self._encode_target_guidance(features)

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1]
        )
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode="bilinear", align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1]
        )
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        coarse_trajectories = features.get("coarse_trajectories")
        mode_valid_mask = features.get("mode_valid_mask")

        trajectory = self._trajectory_head.infer_multimodal(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding[:, None],
            global_img=None,
            target_point_embed=target_point_embed,
            coarse_trajectories=coarse_trajectories,
            mode_valid_mask=mode_valid_mask,
        )
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        target_point_dim=0,
        if_zeroinit_reg=True,
        trajectory_reg_decoder_type="mlp",
        trajectory_gru_hidden_dim=None,
        trajectory_gru_use_mode_embedding=False,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims    # 特征嵌入的维度
        self.ego_fut_ts = ego_fut_ts    # 预测的未来时间步数 8
        self.ego_fut_mode = ego_fut_mode    # 预测的未来轨迹模式数 20
        self.target_point_dim = target_point_dim
        self.trajectory_reg_decoder_type = trajectory_reg_decoder_type
        self.trajectory_gru_hidden_dim = trajectory_gru_hidden_dim or embed_dims
        self.trajectory_gru_use_mode_embedding = trajectory_gru_use_mode_embedding
        guided_dim = embed_dims + target_point_dim
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, input_dims=guided_dim),
            nn.Linear(embed_dims, 1),
        )
        if self.trajectory_reg_decoder_type == "mlp":
            self.plan_reg_branch = nn.Sequential(
                nn.Linear(guided_dim, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, ego_fut_ts * 3),
            )
        elif self.trajectory_reg_decoder_type == "gru":
            gru_input_dim = 2 + 3 + target_point_dim
            self.hidden_init = nn.Linear(embed_dims, self.trajectory_gru_hidden_dim)
            self.reg_gru_cell = nn.GRUCell(gru_input_dim, self.trajectory_gru_hidden_dim)
            self.delta_head = nn.Linear(self.trajectory_gru_hidden_dim, 3)
        else:
            raise ValueError(
                f"Unsupported trajectory_reg_decoder_type: {self.trajectory_reg_decoder_type!r}"
            )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg and self.trajectory_reg_decoder_type == "mlp":
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)
        if self.if_zeroinit_reg and self.trajectory_reg_decoder_type == "gru":
            nn.init.constant_(self.delta_head.weight, 0)
            nn.init.constant_(self.delta_head.bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
        target_point_embed=None,
        noisy_traj_points=None,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape
        base_traj_feature = traj_feature
        if self.target_point_dim > 0:
            if target_point_embed is None:
                target_point_embed = traj_feature.new_zeros((bs, self.target_point_dim))
            target_point_embed = target_point_embed.unsqueeze(1).expand(-1, ego_fut_mode, -1)
            traj_feature = torch.cat([traj_feature, target_point_embed], dim=-1)

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)  # *轨迹分类分支，输出每个模式的概率
        if self.trajectory_reg_decoder_type == "mlp":
            traj_delta = self.plan_reg_branch(traj_feature)  # *轨迹回归分支，输出每个模式的轨迹
            plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)
        else:
            if noisy_traj_points is None:
                raise ValueError("noisy_traj_points must be provided when trajectory_reg_decoder_type='gru'")
            plan_reg = self._forward_gru(
                traj_feature=base_traj_feature,
                noisy_traj_points=noisy_traj_points,
                target_point_embed=target_point_embed,
            )

        return plan_reg, plan_cls

    def _forward_gru(
        self,
        traj_feature,
        noisy_traj_points,
        target_point_embed=None,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape
        hidden = self.hidden_init(traj_feature.reshape(bs * ego_fut_mode, -1))
        if target_point_embed is None or self.target_point_dim <= 0:
            target_point_flat = hidden.new_zeros((bs * ego_fut_mode, 0))
        else:
            target_point_flat = target_point_embed.reshape(bs * ego_fut_mode, -1)

        anchors = noisy_traj_points.reshape(bs * ego_fut_mode, self.ego_fut_ts, 2)
        prev_point = noisy_traj_points.new_zeros((bs * ego_fut_mode, 3))
        deltas = []
        for t in range(self.ego_fut_ts):
            anchor_t = anchors[:, t, :]
            gru_input = [anchor_t, prev_point]
            if target_point_flat.shape[-1] > 0:
                gru_input.append(target_point_flat)
            gru_input = torch.cat(gru_input, dim=-1)
            hidden = self.reg_gru_cell(gru_input, hidden)
            delta_t = self.delta_head(hidden)
            current_point = torch.cat([anchor_t + delta_t[:, :2], delta_t[:, 2:3]], dim=-1)
            prev_point = current_point
            deltas.append(delta_t.unsqueeze(1))

        return torch.cat(deltas, dim=1).reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)

class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=config.tf_d_model,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model, config.tf_d_model)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=config.ego_fut_mode,
            target_point_dim=config.target_point_dim,
            trajectory_reg_decoder_type=config.trajectory_reg_decoder_type,
            trajectory_gru_hidden_dim=config.trajectory_gru_hidden_dim,
            trajectory_gru_use_mode_embedding=config.trajectory_gru_use_mode_embedding,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                target_point_embed=None,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading  
        # *self.task_decoder预测的是噪声残差，即去噪轨迹点与加噪轨迹点的差值，最终轨迹点=加噪轨迹点+残差
        poses_reg, poses_cls = self.task_decoder(
            traj_feature,
            target_point_embed=target_point_embed,
            noisy_traj_points=noisy_traj_points,
        ) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                target_point_embed=None,
                global_img=None,
                return_traj_feature: bool = False):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                target_point_embed,
                global_img,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        if return_traj_feature:
            return poses_reg_list, poses_cls_list, traj_feature
        return poses_reg_list, poses_cls_list

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = config.ego_fut_mode
        self.use_dynamic_anchors = config.use_dynamic_anchors

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        if self.use_dynamic_anchors and not plan_anchor_path:
            plan_anchor = np.zeros((self.ego_fut_mode, num_poses, 2), dtype=np.float32)
        elif self.use_dynamic_anchors:
            try:
                plan_anchor = np.load(plan_anchor_path)[:self.ego_fut_mode]
            except FileNotFoundError:
                plan_anchor = np.zeros((self.ego_fut_mode, num_poses, 2), dtype=np.float32)
        else:
            plan_anchor = np.load(plan_anchor_path)[:self.ego_fut_mode]

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, num_poses * 64),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, config.trajectory_decoder_layers)

        self.loss_computer = LossComputer(config)
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def _get_anchors(
        self, bs: int, device: torch.device, coarse_trajectories: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return (bs, ego_fut_mode, T, 2) anchor tensor.

        Uses dynamic coarse_trajectories when use_dynamic_anchors=True and they are provided;
        otherwise falls back to the frozen plan_anchor parameter.
        """
        if self.use_dynamic_anchors and coarse_trajectories is not None:
            return coarse_trajectories.to(device=device, dtype=torch.float32)
        return self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)

    def forward(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
        return_candidates: bool = False,
        target_point_embed=None,
        coarse_trajectories=None,
        mode_valid_mask=None,
    ) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                targets,
                global_img,
                target_point_embed=target_point_embed,
                coarse_trajectories=coarse_trajectories,
                mode_valid_mask=mode_valid_mask,
            )
        else:
            return self.forward_test(
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                global_img,
                return_candidates=return_candidates,
                target_point_embed=target_point_embed,
                coarse_trajectories=coarse_trajectories,
                mode_valid_mask=mode_valid_mask,
            )

    def infer_multimodal(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img=None,
        target_point_embed=None,
        coarse_trajectories=None,
        mode_valid_mask=None,
    ) -> Dict[str, torch.Tensor]:
        """Return multimodal trajectory candidates before selector sampling."""
        return self.forward_test(
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            global_img,
            return_candidates=True,
            target_point_embed=target_point_embed,
            coarse_trajectories=coarse_trajectories,
            mode_valid_mask=mode_valid_mask,
        )


    def forward_train(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
        target_point_embed=None,
        coarse_trajectories=None,
        mode_valid_mask=None,
    ) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor (or dynamic coarse trajectories)
        plan_anchor = self._get_anchors(bs, device, coarse_trajectories)
        odo_info_fut = self.norm_odo(plan_anchor)
        timesteps = torch.randint(
            0, 50,
            (bs,), device=device
        )
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(     # 加噪轨迹点
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs,1,-1)


        # 4. begin the stacked decoder  预测去噪轨迹点
        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature,
            noisy_traj_points,
            bev_feature,
            bev_spatial_shape,
            agents_query,
            ego_query,
            time_embed,
            status_encoding,
            target_point_embed,
            global_img,
        )

        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor,
                                                 mode_valid_mask=mode_valid_mask)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        return {
            "trajectory": best_reg,
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
            "trajectory_candidates_train": poses_reg_list[-1],
            "trajectory_mode_logits_train": poses_cls_list[-1],
        }

    def forward_test(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        return_candidates: bool = False,
        target_point_embed=None,
        coarse_trajectories=None,
        mode_valid_mask=None,
    ) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        # 1. add truncated noise to the plan anchor (or dynamic coarse trajectories)
        plan_anchor = self._get_anchors(bs, device, coarse_trajectories)
        img = self.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        noisy_trajs = self.denorm_odo(img)
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            if return_candidates:
                poses_reg_list, poses_cls_list, traj_feature = self.diff_decoder(
                    traj_feature,
                    noisy_traj_points,
                    bev_feature,
                    bev_spatial_shape,
                    agents_query,
                    ego_query,
                    time_embed,
                    status_encoding,
                    target_point_embed,
                    global_img,
                    return_traj_feature=True,
                )
            else:
                poses_reg_list, poses_cls_list = self.diff_decoder(
                    traj_feature,
                    noisy_traj_points,
                    bev_feature,
                    bev_spatial_shape,
                    agents_query,
                    ego_query,
                    time_embed,
                    status_encoding,
                    target_point_embed,
                    global_img,
                )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample

        # 选择分类分数最高的模式作为最终预测结果（对 invalid slot 屏蔽）
        if mode_valid_mask is not None:
            masked_cls = poses_cls.masked_fill(~mode_valid_mask.to(device=poses_cls.device), float('-inf'))
            best_mode_idx = masked_cls.argmax(dim=-1)
        else:
            best_mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = best_mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        return {
            "trajectory": best_reg,
            "trajectory_mode_idx": best_mode_idx,
            "trajectory_mode_logits": poses_cls,
            **(
                {
                    "trajectory_candidates": poses_reg,
                    "trajectory_mode_embedding": traj_feature,
                }
                if return_candidates
                else {}
            ),
        }
