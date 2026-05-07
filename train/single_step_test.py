"""Manual-mode single-step closed-loop test for PlatoonDiffusionPlanner.

This tool does not load or call a PPO/SB3 policy. It loads a full diffusion
checkpoint only to export multimodal trajectory candidates, then executes
human-specified mode indices in the closed-loop platoon environment.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import torch

from metadrive.policy.diffusion_policy.test_transfuser_policy import (
    _apply_episode_route_config,
    _build_platoon_planner_batch,
    _build_step_trajectory_plot_path,
    _capture_2d_topdown_frame,
    _capture_step_plot_render_context,
    _extract_road_topology,
    _mode_slot_names_from_config,
    _platoon_agent_final_info,
    _record_step_visualization,
    _resolve_episode_scenario_route,
    _save_step_trajectory_plot,
)
from metadrive.policy.diffusion_policy.transfuser_config import (
    build_transfuser_config,
    diffusion_model_config_to_overrides,
    load_diffusion_model_config,
)
from metadrive.policy.diffusion_policy.transfuser_policy import compute_trajectory_control


DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/single_step_manual_test")
DEFAULT_MODEL_CONFIG_PATH = "configs/diffusion/model.yaml"
DEFAULT_CHECKPOINT = "/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt"
TARGET_POINT_POLICIES = ("expert", "selected_endpoint")


def resolve_device(device: str) -> str:
    requested = str(device or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def parse_manual_modes(value: str, num_agents: int) -> list[int]:
    raw = [item.strip() for item in str(value).split(",") if item.strip()]
    if not raw:
        raise ValueError("--manual-modes must contain at least one integer mode index.")
    try:
        modes = [int(item) for item in raw]
    except ValueError as exc:
        raise ValueError(f"--manual-modes must be comma-separated integers, got: {value}") from exc
    if len(modes) == 1:
        return modes * int(num_agents)
    if len(modes) != int(num_agents):
        raise ValueError(
            f"--manual-modes must contain one value or exactly {num_agents} values, got {len(modes)}: {value}"
        )
    return modes


def validate_manual_modes(agent_ids: list[str], manual_modes: list[int], mode_valid_mask: np.ndarray) -> None:
    mask = np.asarray(mode_valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mode_valid_mask must have shape [N,M], got {mask.shape}")
    if len(agent_ids) != len(manual_modes):
        raise ValueError(f"Expected {len(agent_ids)} manual modes, got {len(manual_modes)}")
    for idx, (agent_id, mode_idx) in enumerate(zip(agent_ids, manual_modes)):
        if mode_idx < 0 or mode_idx >= mask.shape[1] or not bool(mask[idx, mode_idx]):
            valid = np.flatnonzero(mask[idx]).astype(int).tolist()
            raise ValueError(f"invalid manual mode {mode_idx} for {agent_id}; valid modes: {valid}")


def _coerce_point(value) -> list[float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None
    return [float(arr[0]), float(arr[1])]


def _apply_selected_endpoint_target_policy(
    planner_batch: dict[str, dict[str, Any]],
    *,
    agent_ids: list[str],
    manual_modes: list[int],
    coarse_by_agent: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for agent_id, mode_idx in zip(agent_ids, manual_modes):
        if agent_id not in planner_batch:
            continue
        if agent_id not in coarse_by_agent:
            raise ValueError(f"No coarse_trajectories available for {agent_id}; cannot override target point.")
        coarse = np.asarray(coarse_by_agent[agent_id], dtype=np.float32)
        endpoint = np.asarray(coarse[int(mode_idx), -1, :2], dtype=np.float32)
        before = _coerce_point(planner_batch[agent_id].get("target_point"))
        planner_batch[agent_id]["target_point"] = endpoint.copy()
        planner_batch[agent_id]["preference_point"] = endpoint.copy()
        metadata[agent_id] = {
            "target_point_before": before,
            "target_point_after": endpoint.astype(float).tolist(),
            "selected_coarse_endpoint": endpoint.astype(float).tolist(),
        }
    return metadata


def _build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "num_agents": int(args.num_agents),
        "observation_mode": "multimodal",
        "use_render": bool(args.use_render),
        "show_policy_mark": False,
        "start_seed": int(args.start_seed),
        "num_scenarios": 1,
        "traffic_density": float(args.traffic_density),
        "image_on_cuda": bool(args.image_on_cuda),
        "allow_respawn": False,
    }
    if args.scenario_id:
        cfg["scenario_id"] = args.scenario_id
    if args.local_route:
        from routes.route_definitions import get_required_preset, get_route_blocks

        cfg["local_route"] = args.local_route
        cfg["route_preset"] = get_required_preset(args.local_route)
        cfg["ego_main_route_block_ids"] = list(get_route_blocks(args.local_route))
    return cfg


def _build_platoon_planner(checkpoint_path: str, args: argparse.Namespace, model_config: dict[str, Any]):
    from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
    from models.platoon.weight_migration import migrate_single_to_platoon

    model_size = str(model_config.get("model_size", "small"))
    transfuser_config = build_transfuser_config(
        model_size,
        **diffusion_model_config_to_overrides(model_config),
    )
    planner = PlatoonDiffusionPlanner(transfuser_config, num_vehicles=int(args.num_agents))
    planner = migrate_single_to_platoon(checkpoint_path, planner)
    planner.eval()
    for parameter in planner.parameters():
        parameter.requires_grad_(False)
    return planner.to(torch.device(resolve_device(args.planner_device))), transfuser_config


def _attach_dynamic_mode_features(planner_batch: dict[str, dict[str, Any]], env, config) -> None:
    from metadrive.policy.diffusion_policy.test_transfuser_policy import _build_dynamic_mode_features_for_vehicle
    from metadrive.policy.diffusion_policy.transfuser_features import compute_target_point

    for agent_id, sample in planner_batch.items():
        vehicle = env.agents.get(agent_id)
        if vehicle is None:
            continue
        sample.update(_build_dynamic_mode_features_for_vehicle(vehicle, config))
        target_point = compute_target_point(vehicle, config)
        sample["target_point"] = target_point.detach().cpu().numpy().astype(np.float32)
        sample["preference_point"] = np.asarray(sample["target_point"], dtype=np.float32).copy()


def _save_topdown_frame(env, output_dir: Path, step_idx: int) -> str | None:
    import cv2

    frame = _capture_2d_topdown_frame(env)
    if frame is None:
        return None
    path = output_dir / "topdown_frames" / f"step_{step_idx:05d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    return str(path)


def _make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^run_(\d+)$")
    existing = [
        int(match.group(1))
        for item in output_root.iterdir()
        if item.is_dir() and (match := pattern.match(item.name))
    ]
    run_dir = output_root / f"run_{max(existing) + 1 if existing else 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_manual_single_step(args: argparse.Namespace) -> Path:
    from envs.platoon_env import PlatoonEnv

    output_dir = Path(args.output_dir) if args.output_dir else _make_run_dir(Path(args.output_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_diffusion_model_config(args.model_config_path)
    env = PlatoonEnv(_build_env_config(args))
    planner, transfuser_config = _build_platoon_planner(args.checkpoint, args, model_config)
    mode_slot_names = _mode_slot_names_from_config(transfuser_config)
    manual_modes = parse_manual_modes(args.manual_modes, int(args.num_agents))
    road_boundaries = _extract_road_topology(env)

    rng = np.random.RandomState(int(args.start_seed))
    route_selection = _resolve_episode_scenario_route(args.scenario_id, rng, fixed_route=args.local_route)
    _apply_episode_route_config(env, route_selection)

    print(f"[single_step] checkpoint          : {args.checkpoint}", flush=True)
    print(f"[single_step] scenario/local_route: {route_selection.scenario_id} / {route_selection.local_route}", flush=True)
    print(f"[single_step] manual_modes        : {manual_modes}", flush=True)
    print(f"[single_step] target_point_policy : {args.target_point_policy}", flush=True)
    print(f"[single_step] output_dir          : {output_dir}", flush=True)

    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, _info = result
    else:
        obs, _info = result, {}

    debug_steps: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    step_idx = 0
    try:
        while not done and step_idx < int(args.max_steps):
            active_agent_ids = [agent_id for agent_id in [f"agent{i}" for i in range(int(args.num_agents))] if agent_id in obs]
            planner_batch = _build_platoon_planner_batch(obs, active_agent_ids)
            _attach_dynamic_mode_features(planner_batch, env, transfuser_config)
            with torch.no_grad():
                export = planner.export_mode_selection(planner_batch)

            exported_ids = list(export["agent_ids"])
            selected_modes = [manual_modes[int(agent_id.replace("agent", ""))] for agent_id in exported_ids]
            candidates = np.asarray(export["trajectory_candidates"], dtype=np.float32)
            raw_logits = np.asarray(export["raw_cls_logits"], dtype=np.float32)
            masked_logits = np.asarray(export["masked_cls_logits"], dtype=np.float32)
            mode_valid_mask = np.asarray(export["mode_valid_mask"], dtype=bool)
            validate_manual_modes(exported_ids, selected_modes, mode_valid_mask)

            coarse_by_agent = {
                agent_id: np.asarray(planner_batch[agent_id]["coarse_trajectories"], dtype=np.float32)
                for agent_id in exported_ids
                if "coarse_trajectories" in planner_batch[agent_id]
            }
            target_metadata = {
                agent_id: {
                    "target_point_before": _coerce_point(planner_batch[agent_id].get("target_point")),
                    "target_point_after": _coerce_point(planner_batch[agent_id].get("target_point")),
                    "selected_coarse_endpoint": (
                        np.asarray(coarse_by_agent[agent_id], dtype=np.float32)[mode_idx, -1, :2]
                        .astype(float)
                        .tolist()
                        if agent_id in coarse_by_agent
                        else None
                    ),
                }
                for agent_id, mode_idx in zip(exported_ids, selected_modes)
            }
            if args.target_point_policy == "selected_endpoint":
                target_metadata = _apply_selected_endpoint_target_policy(
                    planner_batch,
                    agent_ids=exported_ids,
                    manual_modes=selected_modes,
                    coarse_by_agent=coarse_by_agent,
                )
                with torch.no_grad():
                    export = planner.export_mode_selection(planner_batch)
                candidates = np.asarray(export["trajectory_candidates"], dtype=np.float32)
                raw_logits = np.asarray(export["raw_cls_logits"], dtype=np.float32)
                masked_logits = np.asarray(export["masked_cls_logits"], dtype=np.float32)
                mode_valid_mask = np.asarray(export["mode_valid_mask"], dtype=bool)
                validate_manual_modes(exported_ids, selected_modes, mode_valid_mask)

            step_plot_frame, step_plot_projector = _capture_step_plot_render_context(env, enabled=bool(args.save_trajectory_plot))
            _save_topdown_frame(env, output_dir, step_idx)

            pre_step_state = {}
            for agent_id in exported_ids:
                vehicle = env.agents.get(agent_id)
                if vehicle is None:
                    continue
                pre_step_state[agent_id] = {
                    "vehicle": vehicle,
                    "xy": np.asarray(vehicle.position[:2], dtype=np.float64),
                    "heading": float(getattr(vehicle, "heading_theta", 0.0)),
                }

            low_level_actions: dict[str, np.ndarray] = {}
            final_info_by_agent: dict[str, dict[str, Any]] = {}
            for agent_index, agent_id in enumerate(exported_ids):
                mode_idx = int(selected_modes[agent_index])
                trajectory = np.asarray(candidates[agent_index, mode_idx], dtype=np.float32)
                vehicle = env.agents.get(agent_id)
                current_speed_km_h = float(getattr(vehicle, "speed_km_h", 0.0)) if vehicle is not None else 0.0
                action, controller_debug = compute_trajectory_control(
                    trajectory=trajectory,
                    lookahead_index=int(args.lookahead_index),
                    current_speed_km_h=current_speed_km_h,
                    target_speed_km_h=float(args.target_speed_km_h),
                    controller_type=str(args.controller_type),
                )
                low_level_actions[agent_id] = action
                final_info = _platoon_agent_final_info(
                    agent_id=agent_id,
                    agent_index=agent_index,
                    agent_obs=obs[agent_id],
                    base_info={},
                    trajectory=trajectory,
                    candidates=candidates,
                    masked_logits=masked_logits,
                    raw_logits=raw_logits,
                    mode_valid_mask=mode_valid_mask,
                    coarse_trajectories=planner_batch[agent_id].get("coarse_trajectories"),
                    mode_idx=mode_idx,
                    mode_slot_names=mode_slot_names,
                    controller_debug=controller_debug,
                )
                final_info.update(target_metadata.get(agent_id, {}))
                if final_info.get("target_point_after") is not None:
                    final_info["target_point"] = np.asarray(final_info["target_point_after"], dtype=np.float32)
                final_info_by_agent[agent_id] = final_info

            step_result = env.step(low_level_actions)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            else:
                obs, reward, done_dict, info = step_result
                terminated = done_dict
                truncated = {agent_id: False for agent_id in done_dict}
                done = bool(done_dict.get("__all__", False))

            env_rewards = {agent_id: float((reward or {}).get(agent_id, 0.0)) for agent_id in exported_ids}
            total_reward += float(np.mean(list(env_rewards.values()))) if env_rewards else 0.0
            for agent_id in exported_ids:
                if agent_id in final_info_by_agent:
                    final_info_by_agent[agent_id]["env_reward"] = env_rewards.get(agent_id)
                    final_info_by_agent[agent_id].update(dict((info or {}).get(agent_id, {})))
                    if final_info_by_agent[agent_id].get("target_point_after") is not None:
                        final_info_by_agent[agent_id]["target_point"] = np.asarray(
                            final_info_by_agent[agent_id]["target_point_after"],
                            dtype=np.float32,
                        )

            if bool(args.save_trajectory_plot):
                step_records = []
                for agent_id in exported_ids:
                    state = pre_step_state.get(agent_id)
                    final_info = final_info_by_agent.get(agent_id)
                    if state is None or final_info is None:
                        continue
                    _record_step_visualization(
                        ego_before_step=state["vehicle"],
                        ego_xy_before_step=state["xy"],
                        ego_heading_before_step=state["heading"],
                        final_info=final_info,
                        episode_length=step_idx + 1,
                        save_trajectory_plot=True,
                        actual_positions=[],
                        planned_trajectories=[],
                        multimodal_trajectories=[],
                        step_plot_records=step_records,
                        topdown_frame=step_plot_frame,
                        world_to_screen_projector=step_plot_projector,
                        agent_label=f"EGO{exported_ids.index(agent_id) + 1}",
                    )
                if step_records:
                    primary = step_records[0]
                    primary.peer_records = step_records[1:]
                    _save_step_trajectory_plot(
                        step_record=primary,
                        road_boundaries=road_boundaries,
                        output_path=_build_step_trajectory_plot_path(output_dir, 0, step_idx + 1),
                        episode_idx=0,
                        show_topology_polyline=False,
                    )

            debug_steps.append(
                {
                    "step": int(step_idx + 1),
                    "target_point_policy": args.target_point_policy,
                    "manual_modes": {agent_id: int(mode) for agent_id, mode in zip(exported_ids, selected_modes)},
                    "manual_mode_names": {
                        agent_id: (
                            mode_slot_names[int(mode)]
                            if 0 <= int(mode) < len(mode_slot_names)
                            else f"mode_{int(mode)}"
                        )
                        for agent_id, mode in zip(exported_ids, selected_modes)
                    },
                    "mode_valid_mask": mode_valid_mask.astype(bool).tolist(),
                    "target_metadata": target_metadata,
                    "selected_candidate_endpoint": {
                        agent_id: candidates[idx, int(selected_modes[idx]), -1, :2].astype(float).tolist()
                        for idx, agent_id in enumerate(exported_ids)
                    },
                    "env_reward": env_rewards,
                    "terminated": dict(terminated or {}),
                    "truncated": dict(truncated or {}),
                }
            )
            step_idx += 1
    finally:
        env.close()

    summary = {
        "checkpoint": args.checkpoint,
        "scenario_id": route_selection.scenario_id,
        "local_route": route_selection.local_route,
        "manual_modes": manual_modes,
        "target_point_policy": args.target_point_policy,
        "num_steps": int(step_idx),
        "total_reward": float(total_reward),
        "steps": debug_steps,
    }
    debug_path = output_dir / "single_step_debug.json"
    debug_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[single_step] debug saved: {debug_path}", flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual mode closed-loop single-step test.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config-path", default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--manual-modes", default="0")
    parser.add_argument("--target-point-policy", choices=TARGET_POINT_POLICIES, default="expert")
    parser.add_argument("--scenario-id", default="S1_free_cruise_straight")
    parser.add_argument("--local-route", default="")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--traffic-density", type=float, default=0.06)
    parser.add_argument("--use-render", type=int, choices=(0, 1), default=0)
    parser.add_argument("--image-on-cuda", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save-trajectory-plot", type=int, choices=(0, 1), default=1)
    parser.add_argument("--lookahead-index", type=int, default=2)
    parser.add_argument("--target-speed-km-h", type=float, default=30.0)
    parser.add_argument("--controller-type", default="stabilized")
    parser.add_argument("--planner-device", default="auto")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_manual_single_step(args)
    print(f"[single_step] done. outputs: {run_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
