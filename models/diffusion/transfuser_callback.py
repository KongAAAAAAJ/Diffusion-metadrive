from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils

from models.diffusion.transfuser_config import TransfuserConfig
from models.diffusion.transfuser_features import BoundingBox2DIndex


class TransfuserCallback(pl.Callback):
    """MetaDrive visualization callback for TransFuser."""

    def __init__(
        self,
        config: TransfuserConfig,
        num_plots: int = 2,
        num_rows: int = 1,
        num_columns: int = 2,
    ) -> None:
        self._config = config
        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns
        self._bev_colors = {
            0: _hex_to_rgb(config.bev_background_color),
            1: _hex_to_rgb(config.bev_semantic_classes[1][1]),
            2: _hex_to_rgb(config.bev_semantic_classes[2][1]),
            3: _hex_to_rgb(config.bev_semantic_classes[3][1]),
        }

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        # 每个验证轮结束后自动可视化模型预测效果，并将图像添加到 TensorBoard 中
        if (
            not self._config.enable_val_visualization
            or trainer.logger is None
            or not trainer.val_dataloaders
        ):
            return
        interval = max(self._config.val_visualization_interval, 1)
        current_epoch = trainer.current_epoch + 1
        if current_epoch % interval != 0:
            return

        device = lightning_module.device
        dataloader = trainer.val_dataloaders
        if isinstance(dataloader, list):
            dataloader = dataloader[0]
        batch = next(iter(dataloader))
        features, targets = batch
        features = dict_to_device(features, device)
        targets = dict_to_device(targets, device)

        with torch.no_grad():
            predictions = lightning_module(features, targets)

        grid = self._visualize_model(
            dict_to_device(features, "cpu"),
            dict_to_device(targets, "cpu"),
            dict_to_device(predictions, "cpu"),
        )
        trainer.logger.experiment.add_image("val_plot", grid, global_step=trainer.current_epoch)

    def _visualize_model(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # 将模型输入、预测和目标以图片形式拼接，便于训练过程中直观对比和调试模型效果。
        camera = features["camera_feature"].permute(0, 2, 3, 1).numpy()
        lidar_map = features["lidar_feature"].squeeze(1).numpy()
        bev_gt = targets["bev_semantic_map"].numpy()
        bev_pred = predictions["bev_semantic_map"].argmax(1).numpy()
        agent_labels = targets["agent_labels"].numpy()
        agent_states = targets["agent_states"].numpy()
        pred_agent_labels = predictions["agent_labels"].sigmoid().numpy() > 0.5
        pred_agent_states = predictions["agent_states"].numpy()
        trajectory = targets["trajectory"].numpy()
        pred_trajectory = predictions["trajectory"].numpy()

        plots = []
        total = min(camera.shape[0], self._num_rows * self._num_columns)
        for sample_idx in range(total):
            plot = render_open_loop_prediction(
                features=features,
                targets=targets,
                predictions=predictions,
                config=self._config,
                sample_idx=sample_idx,
            )
            plots.append(torch.from_numpy(plot).permute(2, 0, 1))

        return vutils.make_grid(plots, normalize=False, nrow=2)


def dict_to_device(data: Dict[str, torch.Tensor], device: Union[torch.device, str]) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _camera_to_hwc(camera: Any) -> np.ndarray:
    camera = _tensor_to_numpy(camera)
    if camera.ndim == 3 and camera.shape[0] in (1, 3, 4) and camera.shape[-1] not in (1, 3, 4):
        camera = np.transpose(camera, (1, 2, 0))
    return camera


def semantic_map_to_rgb(semantic_map: np.ndarray, color_map: Dict[int, tuple[int, int, int]]) -> np.ndarray:
    # 将语义分割标签图（整数类别图）转换为彩色 RGB 图像，便于可视化
    rgb_map = np.zeros((*semantic_map.shape, 3), dtype=np.uint8)
    for label, color in color_map.items():
        rgb_map[semantic_map == label] = color
    return rgb_map


def lidar_map_to_rgb(
    lidar_map: np.ndarray,
    agent_states: Optional[np.ndarray],
    pred_agent_states: Optional[np.ndarray],
    trajectory: Optional[np.ndarray],
    pred_trajectory: np.ndarray,
    config: TransfuserConfig,
    anchor_trajectory: Optional[np.ndarray] = None,
    all_anchors: Optional[np.ndarray] = None,
) -> np.ndarray:
    gt_color, pred_color, anchor_color = (0, 255, 0), (255, 0, 0), (255, 191, 0)
    point_size = 3
    height, width = lidar_map.shape[:2]

    def coords_to_pixel(coords):
        pixel_center = np.array([[height / 2.0, width / 2.0]])
        coords_idcs = (coords / config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)

    rgb_map = (lidar_map * 255).astype(np.uint8)
    rgb_map = 255 - rgb_map[..., None].repeat(3, axis=-1)

    for color, boxes in ((gt_color, agent_states), (pred_color, pred_agent_states)):
        if boxes is None:
            continue
        for agent_state in boxes:
            x, y, heading, length, width_box = agent_state
            corners = np.array(
                [
                    [length / 2, width_box / 2],
                    [length / 2, -width_box / 2],
                    [-length / 2, -width_box / 2],
                    [-length / 2, width_box / 2],
                ],
                dtype=np.float32,
            )
            rot = np.array(
                [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]],
                dtype=np.float32,
            )
            corners = corners @ rot.T + np.array([x, y], dtype=np.float32)
            corners = coords_to_pixel(corners).reshape((-1, 1, 2))
            corners = np.flip(corners, axis=-1)
            cv2.polylines(rgb_map, [corners], isClosed=True, color=color, thickness=2)

    for color, traj in ((gt_color, trajectory), (pred_color, pred_trajectory)):
        if traj is None:
            continue
        points = coords_to_pixel(traj[:, :2])
        for x, y in points:
            cv2.circle(rgb_map, (y, x), point_size, color, -1)

    if all_anchors is not None:
        for anchor in all_anchors:
            anchor_points = coords_to_pixel(anchor[:, :2])
            for x, y in anchor_points:
                cv2.circle(rgb_map, (y, x), 1, (90, 90, 90), -1)

    if anchor_trajectory is not None:
        anchor_points = coords_to_pixel(anchor_trajectory[:, :2])
        for x, y in anchor_points:
            cv2.circle(rgb_map, (y, x), point_size, anchor_color, -1)

    return rgb_map[::-1, ::-1]


def _render_transfuser_prediction_panel(
    *,
    camera: np.ndarray,
    lidar_map: np.ndarray,
    pred_trajectory: np.ndarray,
    config: TransfuserConfig,
    bev_current: Optional[np.ndarray] = None,
    gt_trajectory: Optional[np.ndarray] = None,
    gt_bev_semantic_map: Optional[np.ndarray] = None,
    pred_bev_semantic_map: Optional[np.ndarray] = None,
    gt_agent_states: Optional[np.ndarray] = None,
    gt_agent_labels: Optional[np.ndarray] = None,
    pred_agent_states: Optional[np.ndarray] = None,
    pred_agent_labels: Optional[np.ndarray] = None,
    mode_idx: Optional[int] = None,
    anchors: Optional[np.ndarray] = None,
    overlay_all_anchors: bool = False,
    metadata_text: Optional[list[str]] = None,
    current_panel_label: str = "BEV Current",
) -> np.ndarray:
    bev_colors = {
        0: _hex_to_rgb(config.bev_background_color),
        1: _hex_to_rgb(config.bev_semantic_classes[1][1]),
        2: _hex_to_rgb(config.bev_semantic_classes[2][1]),
        3: _hex_to_rgb(config.bev_semantic_classes[3][1]),
    }
    plot = np.zeros((512, 1024, 3), dtype=np.uint8)
    cam = np.clip(_camera_to_hwc(camera) * 255.0, 0.0, 255.0).astype(np.uint8)
    plot[:256, :1024] = add_panel_label(
        cv2.resize(cam, (1024, 256), interpolation=cv2.INTER_LINEAR), "Cameras"
    )

    if gt_bev_semantic_map is not None:
        bev_gt_rgb = semantic_map_to_rgb(_tensor_to_numpy(gt_bev_semantic_map), bev_colors)
    else:
        bev_gt_rgb = np.zeros((128, 256, 3), dtype=np.uint8)
        bev_gt_rgb[:] = 12
    plot[256:, :256] = add_panel_label(
        cv2.resize(bev_gt_rgb, (256, 256), interpolation=cv2.INTER_NEAREST),
        "BEV GT" if gt_bev_semantic_map is not None else "BEV N/A",
    )

    if pred_bev_semantic_map is not None:
        bev_pred = _tensor_to_numpy(pred_bev_semantic_map)
        if bev_pred.ndim == 3:
            bev_pred = bev_pred.argmax(0)
        bev_current_rgb = semantic_map_to_rgb(bev_pred, bev_colors)
    elif bev_current is not None:
        bev_current_rgb = semantic_map_to_rgb(_tensor_to_numpy(bev_current), bev_colors)
    else:
        bev_current_rgb = np.zeros((128, 256, 3), dtype=np.uint8)
        bev_current_rgb[:] = 12
    plot[256:, 256:512] = add_panel_label(
        cv2.resize(bev_current_rgb, (256, 256), interpolation=cv2.INTER_NEAREST),
        current_panel_label,
    )

    selected_anchor = None
    if anchors is not None and mode_idx is not None and 0 <= mode_idx < len(anchors):
        selected_anchor = anchors[mode_idx]

    gt_boxes = None
    if gt_agent_states is not None:
        gt_boxes = _tensor_to_numpy(gt_agent_states)
        if gt_agent_labels is not None:
            gt_boxes = gt_boxes[_tensor_to_numpy(gt_agent_labels).astype(bool)]
    pred_boxes = None
    if pred_agent_states is not None:
        pred_boxes = _tensor_to_numpy(pred_agent_states)
        if pred_agent_labels is not None:
            pred_boxes = pred_boxes[_tensor_to_numpy(pred_agent_labels).astype(bool)]

    lidar_rgb = lidar_map_to_rgb(
        _tensor_to_numpy(lidar_map).squeeze(0),
        gt_boxes,
        pred_boxes,
        None if gt_trajectory is None else _tensor_to_numpy(gt_trajectory),
        _tensor_to_numpy(pred_trajectory),
        config,
        anchor_trajectory=selected_anchor,
        all_anchors=anchors if overlay_all_anchors else None,
    )
    plot[256:, 512:768] = add_panel_label(
        cv2.resize(lidar_rgb, (256, 256), interpolation=cv2.INTER_NEAREST),
        "Lidar/Traj",
    )

    anchor_panel = np.zeros((256, 256, 3), dtype=np.uint8)
    anchor_panel[:] = 12
    if selected_anchor is not None:
        anchor_panel = cv2.resize(
            lidar_map_to_rgb(
                np.zeros_like(_tensor_to_numpy(lidar_map).squeeze(0)),
                None,
                None,
                None if gt_trajectory is None else _tensor_to_numpy(gt_trajectory),
                _tensor_to_numpy(pred_trajectory),
                config,
                anchor_trajectory=selected_anchor,
            ),
            (256, 256),
            interpolation=cv2.INTER_NEAREST,
        )
    anchor_panel = add_panel_label(
        anchor_panel,
        f"Anchor mode={mode_idx}" if mode_idx is not None else "Anchor",
    )
    plot[256:, 768:1024] = anchor_panel

    lines = [] if metadata_text is None else list(metadata_text)
    pred_final_y = float(_tensor_to_numpy(pred_trajectory)[-1, 1])
    if gt_trajectory is not None:
        gt_final_y = float(_tensor_to_numpy(gt_trajectory)[-1, 1])
        lines.insert(0, f"mode={mode_idx} pred_y={pred_final_y:+.2f} gt_y={gt_final_y:+.2f}")
    elif mode_idx is not None:
        lines.insert(0, f"mode={mode_idx} pred_y={pred_final_y:+.2f}")
    if lines:
        line_height = 22
        overlay_height = 10 + line_height * len(lines)
        cv2.rectangle(plot, (10, 212), (520, 212 + overlay_height), (0, 0, 0), thickness=-1)
        for idx, line in enumerate(lines):
            cv2.putText(
                plot,
                line,
                (18, 232 + idx * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return plot


def render_open_loop_prediction(
    features: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: TransfuserConfig,
    sample_idx: int = 0,
    anchors: Optional[np.ndarray] = None,
    overlay_all_anchors: bool = False,
    metadata_text: Optional[list[str]] = None,
) -> np.ndarray:
    camera = features["camera_feature"][sample_idx].permute(1, 2, 0).numpy()
    lidar_map = features["lidar_feature"][sample_idx].numpy()
    bev_gt = targets["bev_semantic_map"][sample_idx].numpy()
    bev_pred = predictions["bev_semantic_map"][sample_idx].argmax(0).numpy()
    agent_labels = targets["agent_labels"][sample_idx].numpy()
    agent_states = targets["agent_states"][sample_idx].numpy()
    pred_agent_labels = predictions["agent_labels"][sample_idx].sigmoid().numpy() > 0.5
    pred_agent_states = predictions["agent_states"][sample_idx].numpy()
    trajectory = targets["trajectory"][sample_idx].numpy()
    pred_trajectory = predictions["trajectory"][sample_idx].numpy()
    mode_idx = None
    if "trajectory_mode_idx" in predictions:
        mode_idx = int(predictions["trajectory_mode_idx"][sample_idx].item())
    return _render_transfuser_prediction_panel(
        camera=camera,
        lidar_map=lidar_map,
        pred_trajectory=pred_trajectory,
        config=config,
        gt_trajectory=trajectory,
        gt_bev_semantic_map=bev_gt,
        pred_bev_semantic_map=bev_pred,
        gt_agent_states=agent_states,
        gt_agent_labels=agent_labels,
        pred_agent_states=pred_agent_states,
        pred_agent_labels=pred_agent_labels,
        mode_idx=mode_idx,
        anchors=anchors,
        overlay_all_anchors=overlay_all_anchors,
        metadata_text=metadata_text,
        current_panel_label="BEV Pred",
    )


def render_closed_loop_prediction(
    *,
    features: Dict[str, Any],
    predictions: Dict[str, Any],
    config: TransfuserConfig,
    anchors: Optional[np.ndarray] = None,
    overlay_all_anchors: bool = False,
    metadata_text: Optional[list[str]] = None,
) -> np.ndarray:
    mode_idx = predictions.get("trajectory_mode_idx")
    if hasattr(mode_idx, "item"):
        mode_idx = int(mode_idx.item())
    elif mode_idx is not None:
        mode_idx = int(mode_idx)
    return _render_transfuser_prediction_panel(
        camera=features["camera_feature"],
        lidar_map=features["lidar_feature"],
        pred_trajectory=predictions["trajectory"],
        config=config,
        bev_current=None,
        gt_trajectory=None,
        gt_bev_semantic_map=None,
        pred_bev_semantic_map=None,
        gt_agent_states=None,
        gt_agent_labels=None,
        pred_agent_states=None,
        pred_agent_labels=None,
        mode_idx=mode_idx,
        anchors=anchors,
        overlay_all_anchors=overlay_all_anchors,
        metadata_text=metadata_text,
        current_panel_label="BEV Current",
    )


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    text_width = max(96, 12 + len(label) * 11)
    cv2.rectangle(labeled, (6, 6), (text_width, 30), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled
