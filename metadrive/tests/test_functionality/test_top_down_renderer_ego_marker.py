from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import metadrive.engine.top_down_renderer as top_down_renderer_module
from metadrive.engine.top_down_renderer import TopDownRenderer
from metadrive.obs.top_down_obs_impl import WorldSurface


def test_draw_tracked_agent_highlight_marks_surface():
    surface = WorldSurface((200, 200), 0, top_down_renderer_module.pygame.Surface((200, 200)))
    surface.fill((255, 255, 255))
    surface.move_display_window_to(np.asarray([0.0, 0.0], dtype=np.float32))
    vehicle = SimpleNamespace(
        position=np.asarray([10.0, 10.0], dtype=np.float32),
        heading_theta=0.0,
        top_down_width=1.8,
        top_down_length=4.5,
    )

    TopDownRenderer._draw_tracked_agent_highlight(surface, vehicle)

    pixels = np.array(top_down_renderer_module.pygame.surfarray.array3d(surface))
    assert np.any(np.any(pixels != 255, axis=2))


def test_draw_tracked_agent_highlight_marks_plain_screen_surface():
    surface = top_down_renderer_module.pygame.Surface((200, 200))
    surface.fill((255, 255, 255))
    vehicle = SimpleNamespace(
        position=np.asarray([10.0, 10.0], dtype=np.float32),
        heading_theta=0.0,
        top_down_width=1.8,
        top_down_length=4.5,
    )

    TopDownRenderer._draw_tracked_agent_highlight(surface, vehicle, screen_position=np.asarray([100.0, 100.0]))

    pixels = np.array(top_down_renderer_module.pygame.surfarray.array3d(surface))
    assert np.any(np.any(pixels != 255, axis=2))


def test_draw_warning_marker_marks_surface_in_red():
    surface = WorldSurface((200, 200), 0, top_down_renderer_module.pygame.Surface((200, 200)))
    surface.fill((255, 255, 255))
    surface.move_display_window_to(np.asarray([0.0, 0.0], dtype=np.float32))
    vehicle = SimpleNamespace(
        position=np.asarray([10.0, 10.0], dtype=np.float32),
        top_down_width=1.8,
        top_down_length=4.5,
        scenario_warning_marker="!",
    )

    TopDownRenderer._draw_warning_marker(surface, vehicle)

    pixels = np.array(top_down_renderer_module.pygame.surfarray.array3d(surface))
    assert np.any((pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 120) & (pixels[:, :, 2] < 120))
