from dataclasses import dataclass, field
from dataclasses import asdict as dataclass_asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class TrajectorySampling:
    time_horizon: float = 4.0
    interval_length: float = 0.5

    @property
    def num_poses(self) -> int:
        if self.interval_length <= 0:
            raise ValueError("interval_length must be > 0")
        return int(round(self.time_horizon / self.interval_length))


@dataclass
class TransfuserConfig:
    """MetaDrive-native TransFuser config without navsim/nuplan dependencies."""

    trajectory_sampling: TrajectorySampling = field(default_factory=TrajectorySampling)

    model_size: str = "small"
    image_architecture: str = "resnet18"
    lidar_architecture: str = "resnet18"
    bkb_path: str = "/media/kong/Elements_SE/ckpts/resnet34.a1_in1k/pytorch_model.bin"
    plan_anchor_path: str = "metadrive/exp_dataset/metadrive_anchors.npy"

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    lidar_max_distance: float = 50.0  # ?用途

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False
    lidar_seq_len: int = 1

    camera_width: int = 768
    camera_height: int = 256
    raw_camera_width: int = 320  # ?用途
    raw_camera_height: int = 180  # ?用途
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = 8
    img_horz_anchors: int = 24
    lidar_vert_anchors: int = 8
    lidar_horz_anchors: int = 8

    block_exp: int = 4
    n_layer: int = 1  # Number of transformer layers used in the vision backbone
    n_head: int = 4
    n_scale: int = 4
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0

    perspective_downsample_factor: int = 1
    transformer_decoder_join: bool = True
    detect_boxes: bool = True
    use_bev_semantic: bool = True
    use_semantic: bool = False
    use_depth: bool = False
    add_features: bool = True

    tf_d_model: int = 128
    tf_d_ffn: int = 512
    tf_num_layers: int = 2
    tf_num_head: int = 4
    tf_dropout: float = 0.0

    num_bounding_boxes: int = 16
    ego_fut_mode: int = 10
    trajectory_decoder_layers: int = 1
    trajectory_reg_decoder_type: str = "mlp"
    trajectory_gru_hidden_dim: Optional[int] = None
    trajectory_gru_use_mode_embedding: bool = False

    # dynamic anchor / mode config
    use_dynamic_anchors: bool = True
    mode_keep_high_speed_mps: float = 13.0    # ~47 km/h
    mode_keep_medium_speed_mps: float = 8.0   # ~29 km/h
    mode_keep_low_speed_mps: float = 3.0      # ~11 km/h
    mode_emergency_decel_mps2: float = 4.5
    mode_lane_change_min_gap_m: float = 12.0

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    use_ema: bool = False

    # !BEV 语义映射分类
    bev_semantic_classes: Dict[int, Tuple[str, str]] = field(
        default_factory=lambda: {
            1: ("road", "#666666"),
            2: ("traffic", "#2f6df6"),
            3: ("ego_history", "#f4a261"),
        }
    )
    bev_background_color: str = "#000000"

    bev_pixel_width: int = 256  # lidar_resolution_width
    bev_pixel_height: int = 128  # lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes: int = 4
    bev_features_channels: int = 32
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    status_feature_dim: int = 19  # full ego_state (9D) + navigation info (10D)
    target_point_dim: int = 32
    target_point_min_forward_distance_m: float = 3.0
    target_point_prediction_horizon_s: float = 4.0
    target_point_max_reachable_accel_mps2: float = 1.5
    target_point_max_reachable_jerk_mps3: float = 2.0
    target_point_front_safe_gap_m: float = 10.0
    target_point_non_front_overlap_buffer_m: float = 1.0
    target_point_vehicle_prediction_use_constant_accel: bool = True

    weight_decay: float = 1e-4
    lr_steps: Tuple[int, ...] = (70,)
    optimizer_type: str = "AdamW"
    scheduler_type: str = "WarmupCosLR"
    cfg_lr_mult: float = 0.5
    opt_paramwise_cfg: Dict = field(
        default_factory=lambda: {"name": {"image_encoder": {"lr_mult": 0.5}}}  # cfg_lr_mult
    )

    batch_size: int = 16  # 每个训练批次的数据量，决定一次前向/反向传播处理多少样本
    num_workers: int = 8  # 24 线程机器上的安全起点，继续增大需实测吞吐。
    persistent_workers: bool = True
    prefetch_factor: int = 2
    pin_memory: Optional[bool] = None
    cache_shards_in_memory: bool = False
    max_epochs: int = 100  # 最大训练轮数，整个数据集会被训练100次。
    min_lr: float = 1e-6  # 学习率的下限，通常用于学习率调度器，防止学习率过低。
    warmup_epochs: int = 3  # 学习率预热的轮数，在训练初期逐渐增加学习率以稳定训练过程。
    precision: str = "auto"
    check_val_every_n_epoch: int = 1
    val_visualization_interval: int = 1
    enable_val_visualization: bool = False

    dataset_root: str = "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max(abs(value) for value in values)

    def __post_init__(self) -> None:
        self.img_vert_anchors = self.camera_height // 32
        self.img_horz_anchors = self.camera_width // 32
        self.lidar_vert_anchors = self.lidar_resolution_height // 32
        self.lidar_horz_anchors = self.lidar_resolution_width // 32
        if self.trajectory_reg_decoder_type not in ("mlp", "gru"):
            raise ValueError(
                f"trajectory_reg_decoder_type must be 'mlp' or 'gru', got {self.trajectory_reg_decoder_type!r}"
            )
        if self.trajectory_gru_hidden_dim is None:
            self.trajectory_gru_hidden_dim = self.tf_d_model


def build_transfuser_config(model_size: str = "small", **overrides) -> TransfuserConfig:
    model_size = model_size.lower()
    presets = {
        "small": dict(
            model_size="small",
            image_architecture="resnet18",
            lidar_architecture="resnet18",
            bkb_path="/media/kong/Elements_SE/ckpts/resnet18.a1_in1k/pytorch_model.bin",
            camera_width=768,
            camera_height=256,
            tf_d_model=128,
            tf_d_ffn=512,
            tf_num_layers=2,
            tf_num_head=4,
            n_layer=1,
            n_head=4,
            num_bounding_boxes=16,
            ego_fut_mode=10,
            trajectory_decoder_layers=1,
            bev_features_channels=32,
        ),
        "base": dict(
            model_size="base",
            image_architecture="resnet34",
            lidar_architecture="resnet34",
            bkb_path="/media/kong/Elements_SE/ckpts/resnet34.a1_in1k/pytorch_model.bin",
            camera_width=1024,
            camera_height=256,
            tf_d_model=256,
            tf_d_ffn=1024,
            tf_num_layers=3,
            tf_num_head=8,
            n_layer=2,
            n_head=4,
            num_bounding_boxes=30,
            ego_fut_mode=20,
            trajectory_decoder_layers=2,
            bev_features_channels=64,
        ),
    }
    if model_size not in presets:
        raise ValueError(f"Unsupported model_size: {model_size}")
    config_kwargs = dict(presets[model_size])
    config_kwargs.update(overrides)
    return TransfuserConfig(**config_kwargs)


def transfuser_config_to_dict(config: TransfuserConfig) -> Dict:
    config_dict = dataclass_asdict(config)
    config_dict["bev_semantic_classes"] = {
        str(key): value for key, value in config_dict["bev_semantic_classes"].items()
    }
    return config_dict
