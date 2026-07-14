"""tools/topdown_view.py

Top-down rendering helpers for platoon scenarios.

Public API
----------
build_render_kwargs(camera_position, heading_up)  ->  dict
overlay_platoon_labels(frame, env, agent_ids)      ->  np.ndarray
overlay_rule_maker_debug(frame, env, debug)        ->  np.ndarray
overlay_planning_debug(frame, env, debug)          ->  np.ndarray
capture_topdown_frame(env, lead_id, heading_up, agent_ids, rule_maker_debug, planning_debug) -> np.ndarray | None

All overlay functions accept an RGB uint8 ndarray and return a new one;
the input is never mutated.  They are pygame-based and safe to call from
any process that has a display-free pygame initialisation (use_render=False).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ── Screen / film geometry ────────────────────────────────────────────────────
TOPDOWN_SCREEN_SIZE: int = 800
TOPDOWN_FILM_SIZE: int = 3000

# ── Label colours (ego vehicle tags) ─────────────────────────────────────────
LABEL_BG     = (255, 247, 237)
LABEL_FG     = (154,  52,  18)
LABEL_BORDER = (251, 146,  60)

# ── Scenario warning marker (background traffic) ─────────────────────────────
SCENARIO_MARKER_BG     = (220,  38,  38)
SCENARIO_MARKER_FG     = (254, 90, 79)
SCENARIO_MARKER_BORDER = (127,  29,  29)

# ── Leader triangle ───────────────────────────────────────────────────────────
LEADER_TRIANGLE_FILL   = ( 34, 197,  94)
LEADER_TRIANGLE_BORDER = ( 21, 128,  61)

# ── Rule-maker trajectory colours  (R,G,B,A) ─────────────────────────────────
RULE_CANDIDATE_COLORS: dict[int, tuple] = {
    -1: ( 59, 130, 246, 110),   # left  — dim blue
     0: (245, 158,  11, 110),   # keep  — dim amber
     1: (168,  85, 247, 110),   # right — dim purple
}
RULE_SELECTED_COLORS: dict[int, tuple] = {
    -1: ( 37,  99, 235, 220),   # left  — bright blue
     0: (217, 119,   6, 220),   # keep  — bright amber
     1: (147,  51, 234, 220),   # right — bright purple
}

PLANNING_CANDIDATE_COLOR = ( 20, 184, 166, 105)
PLANNING_SELECTED_COLOR  = ( 13, 148, 136, 240)

# ── Formation lock badge ──────────────────────────────────────────────────────
LOCK_BADGE_LOCKED   = (220,  38,  38, 210)   # red
LOCK_BADGE_UNLOCKED = ( 22, 163,  74, 210)   # green
LOCK_BADGE_FG       = (255, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_camera_offset(renderer, screen_w: int, screen_h: int) -> Optional[np.ndarray]:
    cam_pos = getattr(renderer, "position", None)
    if cam_pos is None:
        tracked = getattr(renderer, "current_track_agent", None)
        if tracked is None:
            return None
        cam_pos = tracked.position
    try:
        frame_px = renderer._frame_canvas.pos2pix(float(cam_pos[0]), float(cam_pos[1]))
        return np.asarray(
            [frame_px[0] - screen_w / 2, frame_px[1] - screen_h / 2], dtype=np.float32
        )
    except Exception:
        return None


def _draw_dashed_polyline(
    surface,
    color: tuple,
    points: list[tuple[int, int]],
    width: int,
) -> None:
    import pygame

    if len(points) < 2:
        return
    for idx in range(len(points) - 1):
        if idx % 2 == 0:
            pygame.draw.line(surface, color, points[idx], points[idx + 1], width=max(1, int(width)))


def _draw_numpy_marker(frame: np.ndarray, x: int, y: int, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - 2), min(w, x + 3)
    y0, y1 = max(0, y - 2), min(h, y + 3)
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1, :3] = np.asarray(color[:3], dtype=np.uint8)


def _draw_numpy_polyline(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    result = frame.copy()
    for x, y in points:
        _draw_numpy_marker(result, int(x), int(y), color)
    return result


def _draw_world_trajectories(
    frame: np.ndarray,
    env,
    debug: Optional[dict],
    *,
    selected_color: tuple,
    candidate_color: tuple,
    color_by_action: bool = False,
) -> np.ndarray:
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is None or not debug:
        return frame

    try:
        import pygame
    except ModuleNotFoundError:
        screen_w, screen_h = frame.shape[1], frame.shape[0]
        off = _get_camera_offset(renderer, screen_w, screen_h)
        if off is None:
            return frame
        result = frame.copy()
        for agent_id in debug.get("agent_ids", []) or []:
            for candidate in (debug.get("candidates_by_agent", {}) or {}).get(agent_id, []) or []:
                trajectory_world = candidate.get("trajectory_world")
                if not trajectory_world:
                    continue
                points: list[tuple[int, int]] = []
                for point in trajectory_world:
                    try:
                        screen_pos = renderer._world_to_screen_position(np.asarray(point, dtype=np.float32)[:2], off)
                    except Exception:
                        screen_pos = None
                    if screen_pos is not None:
                        points.append((int(screen_pos[0]), int(screen_pos[1])))
                result = _draw_numpy_polyline(
                    result,
                    points,
                    (13, 148, 136) if bool(candidate.get("selected", False)) else (20, 184, 166),
                )
        return result

    if not pygame.get_init():
        pygame.init()

    screen_w, screen_h = frame.shape[1], frame.shape[0]
    off = _get_camera_offset(renderer, screen_w, screen_h)
    if off is None:
        return frame

    overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
    for agent_id in debug.get("agent_ids", []) or []:
        for candidate in (debug.get("candidates_by_agent", {}) or {}).get(agent_id, []) or []:
            trajectory_world = candidate.get("trajectory_world")
            if not trajectory_world:
                continue
            action = int(candidate.get("action", 0))
            selected = bool(candidate.get("selected", False))
            if color_by_action:
                rgba = (
                    RULE_SELECTED_COLORS.get(action, selected_color)
                    if selected
                    else RULE_CANDIDATE_COLORS.get(action, candidate_color)
                )
            else:
                rgba = selected_color if selected else candidate_color
            width = 5 if selected else 2
            screen_points: list[tuple[int, int]] = []
            for point in trajectory_world:
                try:
                    screen_pos = renderer._world_to_screen_position(
                        np.asarray(point, dtype=np.float32)[:2], off
                    )
                except Exception:
                    screen_pos = None
                if screen_pos is None:
                    continue
                screen_points.append((int(screen_pos[0]), int(screen_pos[1])))
            if len(screen_points) < 2:
                continue
            if selected:
                pygame.draw.lines(overlay, rgba, False, screen_points, width)
            else:
                _draw_dashed_polyline(overlay, rgba, screen_points, width)

    base = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    base.blit(overlay, (0, 0))
    return pygame.surfarray.array3d(base).swapaxes(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_render_kwargs(
    camera_position: Optional[tuple[float, float]] = None,
    heading_up: bool = False,
    screen_size: Optional[int] = None,
    film_size: Optional[int] = None,
) -> dict:
    """Return kwargs for env.render() in top-down mode."""
    ss = int(screen_size) if screen_size is not None else TOPDOWN_SCREEN_SIZE
    fs = int(film_size) if film_size is not None else TOPDOWN_FILM_SIZE
    kwargs: dict = {
        "mode": "top_down",
        "window": False,
        "screen_size": (ss, ss),
        "film_size": (fs, fs),
        "target_agent_heading_up": heading_up,
    }
    if camera_position is not None:
        kwargs["camera_position"] = camera_position
    return kwargs


def overlay_platoon_labels(
    frame: np.ndarray,
    env,
    agent_ids: list[str],
) -> np.ndarray:
    """Draw ego1/ego2/... labels and leader triangle at each platoon vehicle.

    Args:
        frame: RGB uint8 ndarray (H, W, 3).
        env: PlatoonEnv (or compatible) instance.
        agent_ids: Ordered list of agent IDs to label.

    Returns:
        New RGB uint8 ndarray with overlays applied.
    """
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is None:
        return frame

    agents = getattr(env, "agents", {}) or {}
    screen_w, screen_h = frame.shape[1], frame.shape[0]

    off = _get_camera_offset(renderer, screen_w, screen_h)
    if off is None:
        return frame

    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    traffic_vehicles = list(getattr(traffic_manager, "_traffic_vehicles", []) or [])

    try:
        import pygame
    except ModuleNotFoundError:
        result = frame.copy()
        for vehicle in traffic_vehicles:
            if not getattr(vehicle, "scenario_warning_marker", None):
                continue
            try:
                screen_pos = renderer._world_to_screen_position(vehicle.position, off)
            except Exception:
                screen_pos = None
            if screen_pos is not None:
                _draw_numpy_marker(result, int(screen_pos[0]), int(screen_pos[1]) - 16, SCENARIO_MARKER_BG)
        for agent_id in agent_ids:
            vehicle = agents.get(agent_id)
            if vehicle is None:
                continue
            try:
                screen_pos = renderer._world_to_screen_position(vehicle.position, off)
            except Exception:
                screen_pos = None
            if screen_pos is not None:
                role = "follower"
                get_agent_role = getattr(env, "get_agent_role", None)
                if callable(get_agent_role):
                    try:
                        role = str(get_agent_role(agent_id))
                    except Exception:
                        role = "follower"
                if role == "leader":
                    _draw_numpy_marker(result, int(screen_pos[0]), int(screen_pos[1]) - 8, LEADER_TRIANGLE_FILL)
                _draw_numpy_marker(result, int(screen_pos[0]) + 10, int(screen_pos[1]) - 12, LABEL_BORDER)
        return result

    if not pygame.font.get_init():
        pygame.font.init()
    if not pygame.get_init():
        pygame.init()
    font = pygame.font.Font(None, 16)

    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    # !背景车标注符号
    # for vehicle in traffic_vehicles:
    #     marker = getattr(vehicle, "scenario_warning_marker", None)
    #     if not marker:
    #         continue
    #     try:
    #         screen_pos = renderer._world_to_screen_position(vehicle.position, off)
    #     except Exception:
    #         continue
    #     if screen_pos is None:
    #         continue
    #     sx, sy = int(screen_pos[0]), int(screen_pos[1])
    #     if sx < -40 or sx > screen_w + 40 or sy < -40 or sy > screen_h + 40:
    #         continue
    #     # text_surf = font.render(str(marker), True, SCENARIO_MARKER_FG, SCENARIO_MARKER_BG)
    #     text_surf = font.render(str(marker), True, SCENARIO_MARKER_FG)
    #     text_rect = text_surf.get_rect()
    #     text_rect.center = (sx, sy - 16)
    #     padded_rect = text_rect.inflate(10, 6)
    #     # pygame.draw.rect(surf, SCENARIO_MARKER_BG, padded_rect, border_radius=6)  # 画红色圆角背景框
    #     # pygame.draw.rect(surf, SCENARIO_MARKER_BORDER, padded_rect, width=1, border_radius=6)  # 画深红色边框
    #     surf.blit(text_surf, text_rect)

    # Per-vehicle ego labels
    for i, agent_id in enumerate(agent_ids):
        vehicle = agents.get(agent_id)
        if vehicle is None:
            continue
        try:
            screen_pos = renderer._world_to_screen_position(vehicle.position, off)
        except Exception:
            continue
        if screen_pos is None:
            continue
        sx, sy = int(screen_pos[0]), int(screen_pos[1])
        if sx < -40 or sx > screen_w + 40 or sy < -40 or sy > screen_h + 40:
            continue

        role = "follower"
        get_agent_role = getattr(env, "get_agent_role", None)
        if callable(get_agent_role):
            try:
                role = str(get_agent_role(agent_id))
            except Exception:
                pass
        elif isinstance(getattr(env, "agent_roles", None), dict):
            role = str(getattr(env, "agent_roles", {}).get(agent_id, "follower"))

        # ego编队主从车标记
        if role == "leader":
            pygame.draw.circle(surf, (59, 130, 246), (sx, sy), 2, width=2)
        elif role == "follower":
            pygame.draw.circle(surf, (34, 197, 94), (sx, sy), 2, width=2)

        # 绘制ego标记
        # label = f"ego{i + 1}"
        # text_surf = font.render(label, True, LABEL_FG, LABEL_BG)
        # text_rect = text_surf.get_rect()
        # text_rect.center = (sx + 10, sy - 12)
        # padded_rect = text_rect.inflate(8, 4)
        # pygame.draw.rect(surf, LABEL_BG, padded_rect, border_radius=6)
        # pygame.draw.rect(surf, LABEL_BORDER, padded_rect, width=1, border_radius=6)
        # surf.blit(text_surf, text_rect)

    return pygame.surfarray.array3d(surf).swapaxes(0, 1)


def overlay_rule_maker_debug(
    frame: np.ndarray,
    env,
    rule_maker_debug: Optional[dict],
) -> np.ndarray:
    """Draw rule-maker candidate trajectories and formation lock badge.

    Candidate trajectories are drawn as dashed polylines; the selected
    trajectory is drawn as a solid line.  A formation-lock badge is drawn
    in the top-left corner when ``rule_maker_debug["formation_locked"]``
    is present.

    Args:
        frame: RGB uint8 ndarray (H, W, 3).
        env: PlatoonEnv (or compatible) instance — used to access the renderer.
        rule_maker_debug: Dict returned by ``MultiAgentRuleMaker.get_last_debug()``,
            or ``None`` / empty to skip rendering.

    Returns:
        New RGB uint8 ndarray with overlays applied.
    """
    if not rule_maker_debug:
        return frame

    # ── Candidate / selected trajectory polylines ─────────────────────────────
    # frame = _draw_world_trajectories(
    #     frame,
    #     env,
    #     rule_maker_debug,
    #     selected_color=(255, 255, 255, 220),
    #     candidate_color=(255, 255, 255, 110),
    #     color_by_action=True,
    # )

    # ── Formation lock badge — top-left corner ────────────────────────────────
    formation_locked = rule_maker_debug.get("formation_locked")
    if formation_locked is None:
        return frame

    import pygame
    base = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    if formation_locked is not None:
        if not pygame.font.get_init():
            pygame.font.init()
        badge_font = pygame.font.Font(None, 20)
        locked = bool(formation_locked)
        risk_triggered = bool(rule_maker_debug.get("risk_triggered", False))
        if locked:
            badge_text = "\U0001f512 LOCKED"
            bg_rgba = LOCK_BADGE_LOCKED
        elif risk_triggered:
            badge_text = "\U0001f513 UNLOCKED (risk)"
            bg_rgba = LOCK_BADGE_UNLOCKED
        else:
            badge_text = "\U0001f513 UNLOCKED"
            bg_rgba = LOCK_BADGE_UNLOCKED
        text_surf = badge_font.render(badge_text, True, LOCK_BADGE_FG)
        text_rect = text_surf.get_rect()
        padded = text_rect.inflate(14, 8)
        padded.topleft = (8, 8)
        badge_surf = pygame.Surface(padded.size, pygame.SRCALPHA)
        badge_surf.fill(bg_rgba)
        base.blit(badge_surf, padded.topleft)
        text_rect.center = padded.center
        base.blit(text_surf, text_rect)

    # ── Discrete decision badges — top-right corner ───────────────────────────
    best_actions = rule_maker_debug.get("best_actions") or {}
    agent_ids = rule_maker_debug.get("agent_ids") or []
    if best_actions and agent_ids:
        if not pygame.font.get_init():
            pygame.font.init()
        decision_font = pygame.font.Font(None, 20)
        _ACTION_LABEL = {-1: "← LEFT", 0: "KEEP", 1: "RIGHT →"}
        _ACTION_BG: dict[int, tuple] = {
            -1: ( 37,  99, 235, 210),
            0:  (217, 119,   6, 210),
            1:  (147,  51, 234, 210),
        }
        screen_w = base.get_width()
        y_cursor = 8
        for i, agent_id in enumerate(agent_ids):
            if agent_id not in best_actions:
                continue
            action = int(best_actions[agent_id])
            label = f"ego{i + 1}: {_ACTION_LABEL.get(action, str(action))}"
            bg_rgba = _ACTION_BG.get(action, (100, 100, 100, 210))
            text_surf = decision_font.render(label, True, LOCK_BADGE_FG)
            text_rect = text_surf.get_rect()
            padded = text_rect.inflate(14, 8)
            padded.topright = (screen_w - 8, y_cursor)
            badge_surf = pygame.Surface(padded.size, pygame.SRCALPHA)
            badge_surf.fill(bg_rgba)
            base.blit(badge_surf, padded.topleft)
            text_rect.center = padded.center
            base.blit(text_surf, text_rect)
            y_cursor += padded.height + 4

    return pygame.surfarray.array3d(base).swapaxes(0, 1)


def overlay_planning_debug(
    frame: np.ndarray,
    env,
    planning_debug: Optional[dict],
) -> np.ndarray:
    """Draw planner candidate trajectories and highlight the selected trajectory."""
    return _draw_world_trajectories(
        frame,
        env,
        planning_debug,
        selected_color=PLANNING_SELECTED_COLOR,
        candidate_color=PLANNING_CANDIDATE_COLOR,
        color_by_action=False,
    )


def capture_topdown_frame(
    env,
    lead_id: str,
    heading_up: bool,
    agent_ids: list[str],
    rule_maker_debug: Optional[dict] = None,
    planning_debug: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Capture one top-down frame with all platoon overlays applied.

    Args:
        env: PlatoonEnv instance.
        lead_id: Agent ID whose position is used as the camera anchor.
        heading_up: If True, rotate view so the ego heading points up.
        agent_ids: All platoon agent IDs for label drawing.
        rule_maker_debug: Optional debug dict from MultiAgentRuleMaker;
            if provided, trajectory overlays and the lock badge are drawn.

    Returns:
        RGB uint8 ndarray, or ``None`` if the frame could not be captured.
    """
    agents = getattr(env, "agents", {}) or {}
    vehicle = agents.get(lead_id)
    if vehicle is None:
        return None
    cam_pos = (float(vehicle.position[0]), float(vehicle.position[1]))

    renderer = getattr(env, "top_down_renderer", None)
    if renderer is not None:
        renderer.position = cam_pos

    try:
        if renderer is None:
            frame = env.render(**build_render_kwargs(cam_pos, heading_up))
            renderer = getattr(env, "top_down_renderer", None)
            if renderer is not None:
                renderer.position = cam_pos
        else:
            frame = env.render(**build_render_kwargs(heading_up=heading_up))
        if frame is None:
            return None
        frame = np.asarray(frame)
    except Exception:
        return None

    frame = overlay_rule_maker_debug(frame, env, rule_maker_debug) 
    frame = overlay_planning_debug(frame, env, planning_debug)
    return overlay_platoon_labels(frame, env, agent_ids)
