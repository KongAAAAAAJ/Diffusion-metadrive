from __future__ import annotations

import numpy as np

from models.diffusion.mode_definitions import MODE_SLOTS
from models.diffusion.mode_visualization import (
    ModeOverlayRenderContext,
    _world_to_screen,
    overlay_mode_trajectories_on_frame,
)


def test_overlay_mode_trajectories_draws_lines_and_legend():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    coarse = np.zeros((len(MODE_SLOTS), 8, 2), dtype=np.float32)
    for mode_idx in range(len(MODE_SLOTS)):
        coarse[mode_idx, :, 0] = np.linspace(0.0, 15.0 + mode_idx, 8, dtype=np.float32)
        coarse[mode_idx, :, 1] = mode_idx - 3.0
    valid_mask = np.asarray([True, True, False, True, False, True, True, True, False, True], dtype=bool)

    render_context = ModeOverlayRenderContext(
        frame=frame,
        ego_world_position=np.asarray([0.0, 0.0], dtype=np.float32),
        ego_heading_rad=0.0,
        camera_position=(0.0, 0.0),
        heading_up=False,
        screen_size=(256, 256),
        scaling=4.0,
        scenario_id="S8_ego_exit_to_ramp",
        local_route="R6_exit_to_ramp",
        front_object_distance=18.0,
        left_lane_gap=20.0,
        right_lane_gap=12.0,
        has_left_branch=False,
        has_right_branch=True,
        recommended_mode_index=7,
    )

    image = overlay_mode_trajectories_on_frame(render_context, coarse, valid_mask)

    assert image.shape == frame.shape
    assert np.count_nonzero(image) > 0
    assert not np.array_equal(image, frame)


def test_overlay_mode_trajectories_anchor_starts_at_ego_center_and_uses_distinct_mode_colors():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    coarse = np.zeros((len(MODE_SLOTS), 8, 2), dtype=np.float32)
    for mode_idx in range(len(MODE_SLOTS)):
        coarse[mode_idx, :, 0] = np.linspace(2.0, 16.0, 8, dtype=np.float32)
        coarse[mode_idx, :, 1] = 0.25 * mode_idx
    valid_mask = np.ones((len(MODE_SLOTS),), dtype=bool)

    render_context = ModeOverlayRenderContext(
        frame=frame,
        ego_world_position=np.asarray([0.0, 0.0], dtype=np.float32),
        ego_heading_rad=0.0,
        camera_position=(0.0, 0.0),
        heading_up=False,
        screen_size=(256, 256),
        scaling=4.0,
        scenario_id="S1_free_cruise_straight",
        local_route="R1_entry_straight",
        front_object_distance=18.0,
        left_lane_gap=20.0,
        right_lane_gap=12.0,
        has_left_branch=False,
        has_right_branch=False,
        recommended_mode_index=None,
    )

    image = overlay_mode_trajectories_on_frame(render_context, coarse, valid_mask)

    center_patch = image[124:133, 124:133]
    assert np.count_nonzero(center_patch) > 0

    non_black_colors = {
        tuple(int(channel) for channel in pixel)
        for pixel in image.reshape(-1, 3)
        if any(int(channel) != 0 for channel in pixel)
    }
    assert len(non_black_colors) >= 10


def test_world_to_screen_heading_up_projects_forward_to_top_of_screen():
    world_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)

    screen_xy = _world_to_screen(
        world_xy,
        camera_position=(0.0, 0.0),
        screen_size=(256, 256),
        scaling=4.0,
        heading_up=True,
        ego_world_position=np.asarray([0.0, 0.0], dtype=np.float32),
        ego_heading_rad=0.0,
    )

    assert np.allclose(screen_xy[0], np.asarray([128.0, 128.0], dtype=np.float32))
    assert float(screen_xy[1, 1]) < float(screen_xy[0, 1])
    assert abs(float(screen_xy[1, 0]) - float(screen_xy[0, 0])) < 1e-4
