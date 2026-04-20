from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

from metadrive.policy.diffusion_policy.mode_definitions import MODE_SLOTS, ModeSlot


_COLOR_BY_MODE_NAME = {
    "KEEP_LANE_HIGH": (245, 158, 11),
    "KEEP_LANE_MEDIUM": (234, 88, 12),
    "KEEP_LANE_LOW": (217, 70, 239),
    "LANE_CHANGE_LEFT_HIGH": (22, 163, 74),
    "LANE_CHANGE_LEFT_MEDIUM": (34, 197, 94),
    "LANE_CHANGE_LEFT_LOW": (134, 239, 172),
    "LANE_CHANGE_RIGHT_HIGH": (13, 148, 136),
    "LANE_CHANGE_RIGHT_MEDIUM": (16, 185, 129),
    "LANE_CHANGE_RIGHT_LOW": (153, 246, 228),
    "EMERGENCY_STOP": (239, 68, 68),
}
_INVALID_COLOR = (170, 170, 170)
_RECOMMENDED_COLOR = (255, 255, 255)


@dataclass
class ModeOverlayRenderContext:
    frame: np.ndarray
    ego_world_position: np.ndarray
    ego_heading_rad: float
    camera_position: tuple[float, float]
    heading_up: bool
    screen_size: tuple[int, int]
    scaling: float
    scenario_id: str
    local_route: str
    front_object_distance: float
    left_lane_gap: float
    right_lane_gap: float
    has_left_branch: bool
    has_right_branch: bool
    recommended_mode_index: int | None = None
    world_to_screen_projector: Callable[[np.ndarray], np.ndarray] | None = None


def _mode_short_name(slot: ModeSlot) -> str:
    mapping = {
        "KEEP_LANE_HIGH": "KL_H",
        "KEEP_LANE_MEDIUM": "KL_M",
        "KEEP_LANE_LOW": "KL_L",
        "LANE_CHANGE_LEFT_HIGH": "LC_L_H",
        "LANE_CHANGE_LEFT_MEDIUM": "LC_L_M",
        "LANE_CHANGE_LEFT_LOW": "LC_L_L",
        "LANE_CHANGE_RIGHT_HIGH": "LC_R_H",
        "LANE_CHANGE_RIGHT_MEDIUM": "LC_R_M",
        "LANE_CHANGE_RIGHT_LOW": "LC_R_L",
        "EMERGENCY_STOP": "STOP",
    }
    return mapping.get(slot.name, slot.name)


def pick_recommended_mode(valid_mask: np.ndarray) -> int | None:
    priority = (4, 7, 3, 6, 5, 8, 1, 0, 2, 9)
    for index in priority:
        if index < len(valid_mask) and bool(valid_mask[index]):
            return int(index)
    return None


def _local_to_world(trajectory_xy: np.ndarray, ego_world_position: np.ndarray, ego_heading_rad: float) -> np.ndarray:
    trajectory_xy = np.asarray(trajectory_xy, dtype=np.float32)
    ego_world_position = np.asarray(ego_world_position, dtype=np.float32)
    cos_h = np.cos(float(ego_heading_rad))
    sin_h = np.sin(float(ego_heading_rad))
    world_x = ego_world_position[0] + cos_h * trajectory_xy[:, 0] - sin_h * trajectory_xy[:, 1]
    world_y = ego_world_position[1] + sin_h * trajectory_xy[:, 0] + cos_h * trajectory_xy[:, 1]
    return np.stack([world_x, world_y], axis=1).astype(np.float32, copy=False)


def _world_to_screen(
    world_xy: np.ndarray,
    camera_position: tuple[float, float],
    screen_size: tuple[int, int],
    scaling: float,
    heading_up: bool,
    ego_world_position: np.ndarray,
    ego_heading_rad: float,
) -> np.ndarray:
    world_xy = np.asarray(world_xy, dtype=np.float32)
    cam = np.asarray(camera_position, dtype=np.float32)
    screen_center = np.asarray([screen_size[0] / 2.0, screen_size[1] / 2.0], dtype=np.float32)
    rel_x = world_xy[:, 0] - cam[0]
    rel_y = world_xy[:, 1] - cam[1]
    screen = np.stack(
        [
            screen_center[0] + rel_x * float(scaling),
            screen_center[1] - rel_y * float(scaling),
        ],
        axis=1,
    )
    if not heading_up:
        return screen.astype(np.float32, copy=False)

    anchor = np.asarray(
        [
            screen_center[0] + (float(ego_world_position[0]) - cam[0]) * float(scaling),
            screen_center[1] - (float(ego_world_position[1]) - cam[1]) * float(scaling),
        ],
        dtype=np.float32,
    )
    rel = screen - anchor
    rotation = np.deg2rad(-np.rad2deg(float(ego_heading_rad)) + 90.0)
    cos_t = np.cos(rotation)
    sin_t = np.sin(rotation)
    rotated = np.stack(
        [
            cos_t * rel[:, 0] + sin_t * rel[:, 1],
            -sin_t * rel[:, 0] + cos_t * rel[:, 1],
        ],
        axis=1,
    )
    return (anchor + rotated).astype(np.float32, copy=False)


def _draw_mode_table(image: np.ndarray, valid_mask: np.ndarray, render_context: ModeOverlayRenderContext) -> None:
    x = 14
    y = 20
    cv2.rectangle(image, (8, 8), (355, 222), (247, 248, 250), thickness=-1)
    cv2.rectangle(image, (8, 8), (355, 222), (210, 214, 220), thickness=1)
    cv2.putText(
        image,
        f"{render_context.scenario_id} | {render_context.local_route}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    y += 18
    context_text = (
        f"front={render_context.front_object_distance:.1f}m "
        f"lgap={render_context.left_lane_gap:.1f} "
        f"rgap={render_context.right_lane_gap:.1f}"
    )
    cv2.putText(image, context_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
    y += 16
    branch_text = f"left_branch={int(render_context.has_left_branch)} right_branch={int(render_context.has_right_branch)}"
    cv2.putText(image, branch_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
    y += 18
    for slot in MODE_SLOTS:
        is_valid = bool(valid_mask[slot.index])
        state = "valid" if is_valid else "invalid"
        color = _INVALID_COLOR if not is_valid else _COLOR_BY_MODE_NAME.get(slot.name, (255, 255, 255))
        cv2.putText(
            image,
            f"{slot.index}: {slot.name} [{slot.speed_profile.name}] {state}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 16


def overlay_mode_trajectories_on_frame(
    render_context: ModeOverlayRenderContext,
    coarse_trajectories: np.ndarray,
    mode_valid_mask: np.ndarray,
    include_invalid: bool = True,
) -> np.ndarray:
    image = np.asarray(render_context.frame).copy()
    for slot, trajectory_xy in zip(MODE_SLOTS, np.asarray(coarse_trajectories, dtype=np.float32)):
        is_valid = bool(mode_valid_mask[slot.index])
        if not is_valid and not include_invalid:
            continue
        trajectory_xy = np.asarray(trajectory_xy, dtype=np.float32)
        if trajectory_xy.ndim != 2 or trajectory_xy.shape[1] != 2:
            continue
        # Anchor every visualization at the ego origin so the drawn trajectory always
        # starts from the rendered ego position even when the generator outputs only
        # future samples.
        anchored_local_xy = np.concatenate(
            [np.zeros((1, 2), dtype=np.float32), trajectory_xy],
            axis=0,
        )
        world_xy = _local_to_world(anchored_local_xy, render_context.ego_world_position, render_context.ego_heading_rad)
        if render_context.world_to_screen_projector is not None:
            screen_xy = np.asarray(
                [render_context.world_to_screen_projector(point) for point in world_xy],
                dtype=np.float32,
            )
        else:
            screen_xy = _world_to_screen(
                world_xy,
                render_context.camera_position,
                render_context.screen_size,
                render_context.scaling,
                render_context.heading_up,
                render_context.ego_world_position,
                render_context.ego_heading_rad,
            )
        polyline = np.round(screen_xy).astype(np.int32).reshape(-1, 1, 2)
        color = _COLOR_BY_MODE_NAME.get(slot.name, (255, 255, 255))
        if not is_valid:
            color = _INVALID_COLOR
        thickness = 3 if render_context.recommended_mode_index == slot.index else 2
        cv2.polylines(image, [polyline], False, color, thickness=thickness, lineType=cv2.LINE_AA)
        start_point = tuple(int(value) for value in np.round(screen_xy[0]).astype(np.int32))
        cv2.circle(image, start_point, 3 if is_valid else 2, color, thickness=-1, lineType=cv2.LINE_AA)
        end_point = tuple(int(value) for value in np.round(screen_xy[-1]).astype(np.int32))
        label_color = _RECOMMENDED_COLOR if render_context.recommended_mode_index == slot.index else color
        cv2.putText(
            image,
            _mode_short_name(slot),
            (end_point[0] + 4, end_point[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            label_color,
            1,
            cv2.LINE_AA,
        )

    _draw_mode_table(image, np.asarray(mode_valid_mask, dtype=bool), render_context)
    return image
