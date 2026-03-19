from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch, time
from metadrive.envs.diffusion_envs.base_multi_env import DatasetCollectEnv
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config, transfuser_config_to_dict
from metadrive.policy.diffusion_policy.transfuser_policy import TransfuserPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop evaluation for MetaDrive TransFuser.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained TransFuser checkpoint.")
    parser.add_argument("--model-size", type=str, default="auto")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--image-on-cuda", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--target-speed-km-h", type=float, default=30.0)
    parser.add_argument("--lookahead-index", type=int, default=2)
    parser.add_argument("--controller-type", type=str, default="stabilized")
    parser.add_argument("--print-trajectory-debug", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-camera-interval", type=int, default=0)  # 默认关闭相机图片保存
    parser.add_argument("--camera-output-dir", type=str, default="/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/closed_loop/cameras")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--traffic-density", type=float, default=0.06)
    parser.add_argument("--plan-anchor-path", type=str, default="metadrive/exp_dataset/metadrive_anchors.npy")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _extract_state_dict(checkpoint_obj):
    return checkpoint_obj.get("state_dict", checkpoint_obj)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def save_triplet_cameras(observation: dict, output_dir: Path, episode_idx: int, step_idx: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for obs_key, suffix in (("rgb_left", "left"), ("rgb_front", "front"), ("rgb_right", "right")):
        if obs_key not in observation:
            continue
        image = _to_uint8_image(observation[obs_key])
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_path = output_dir / f"ep{episode_idx:03d}_step{step_idx:05d}_{suffix}.png"
        cv2.imwrite(str(image_path), image_bgr)


def infer_model_size_from_checkpoint(checkpoint_path: Path) -> str:
# 自动推断模型规模（small/base）
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    for key in (
        "_transfuser_model._query_embedding.weight",
        "agent._transfuser_model._query_embedding.weight",
        "_query_embedding.weight",
    ):
        if key in state_dict:
            query_embedding = state_dict[key]
            if tuple(query_embedding.shape) == (31, 256):
                return "base"
            if tuple(query_embedding.shape) == (17, 128):
                return "small"
    raise ValueError(
        f"Unable to infer model size from checkpoint: {checkpoint_path}. "
        "Expected one of the known _query_embedding shapes for small/base."
    )


def build_env_config(args, resolved_model_size: str):
    transfuser_config = build_transfuser_config(resolved_model_size)
    if args.plan_anchor_path:
        transfuser_config.plan_anchor_path = args.plan_anchor_path

    return {
        "use_render": args.render,
        "num_agents": 1,
        "start_seed": args.start_seed,
        "num_scenarios": args.num_scenarios,
        "traffic_density": args.traffic_density,
        "agent_policy": TransfuserPolicy,
        "show_policy_mark": False,
        "image_on_cuda": bool(args.image_on_cuda),

        "transfuser_checkpoint_path": args.checkpoint,
        "transfuser_policy_device": resolve_device(args.device),
        "transfuser_target_speed_km_h": args.target_speed_km_h,
        "transfuser_lookahead_index": args.lookahead_index,
        "transfuser_controller_type": args.controller_type,
        "transfuser_config": transfuser_config_to_dict(transfuser_config),
    }


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    resolved_model_size = infer_model_size_from_checkpoint(checkpoint_path) if args.model_size == "auto" else args.model_size
    resolved_device = resolve_device(args.device)
    print(f"[test] checkpoint={checkpoint_path}")
    print(f"[test] model_size={resolved_model_size}")
    print(f"[test] device={resolved_device}")
    print(f"[test] controller_type={args.controller_type}")
    print(f"[test] image_on_cuda={bool(args.image_on_cuda)}")
    if args.save_camera_interval > 0:
        print(f"[test] save_camera_interval={args.save_camera_interval} camera_output_dir={args.camera_output_dir}")

    env = DatasetCollectEnv(build_env_config(args, resolved_model_size))
    summary = {
        "success": 0,
        "crash": 0,
        "out_of_road": 0,
        "episode_reward": [],
        "episode_length": [],
        "lookahead_y": [],
        "lookahead_heading": [],
        "steering": [],
        "mode_idx": [],
    }

    try:
        for episode_idx in range(args.episodes):
            obs, info = env.reset()
            if bool(args.render) and hasattr(env, "switch_to_third_person_view"):
                env.switch_to_third_person_view()

            done = False
            episode_reward = 0.0
            episode_length = 0
            final_info = {}

            while not done:
                # External actions are ignored when agent_policy is a closed-loop policy.
                dummy_actions = {
                    agent_id: np.zeros(2, dtype=np.float32)
                    for agent_id in env.agents.keys()
                }
                t_start = time.time()
                obs, reward, terminated, truncated, info = env.step(dummy_actions)
                t_end = time.time()
                print(f"[episode={episode_idx} step={episode_length}] step_time={t_end - t_start:.4f}s")
                agent_id = next(iter(reward.keys()))
                episode_reward += float(reward[agent_id])
                episode_length += 1
                final_info = info.get(agent_id, {})
                done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))

                if args.save_camera_interval > 0:
                    agent_obs = obs.get(agent_id)
                    if agent_obs is not None and episode_length % args.save_camera_interval == 0:
                        save_triplet_cameras(
                            observation=agent_obs,
                            output_dir=Path(args.camera_output_dir),
                            episode_idx=episode_idx,
                            step_idx=episode_length,
                        )

                controller_debug = final_info.get("controller_debug")
                if controller_debug:
                    summary["lookahead_y"].append(float(controller_debug.get("waypoint_y", 0.0)))
                    summary["lookahead_heading"].append(float(controller_debug.get("waypoint_heading", 0.0)))
                    summary["steering"].append(float(controller_debug.get("steering", 0.0)))
                    mode_idx = final_info.get("trajectory_mode_idx")
                    if mode_idx is not None:
                        summary["mode_idx"].append(int(mode_idx))
                    if bool(args.print_trajectory_debug):
                        mode_text = f" mode={mode_idx}" if mode_idx is not None else ""
                        print(
                            f"[analysis episode={episode_idx} step={episode_length}] "
                            f"lookahead_y={controller_debug.get('waypoint_y', 0.0):+.3f} "
                            f"heading={controller_debug.get('waypoint_heading', 0.0):+.3f} "
                            f"steering={controller_debug.get('steering', 0.0):+.3f}"
                            f"{mode_text}"
                        )

                if bool(args.render):
                    env.render(
                        text={
                            "episode": episode_idx,
                            "step": episode_length,
                            "reward": f"{episode_reward:.2f}",
                        }
                    )

            summary["success"] += int(bool(final_info.get("arrive_dest", False)))
            summary["crash"] += int(bool(final_info.get("crash_vehicle", False) or final_info.get("crash", False)))
            summary["out_of_road"] += int(bool(final_info.get("out_of_road", False)))
            summary["episode_reward"].append(episode_reward)
            summary["episode_length"].append(episode_length)
            print(
                f"[episode={episode_idx}] reward={episode_reward:.2f} length={episode_length} "
                f"success={final_info.get('arrive_dest', False)} crash={final_info.get('crash', False)} "
                f"out_of_road={final_info.get('out_of_road', False)}"
            )
    finally:
        env.close()

    num_episodes = max(args.episodes, 1)
    mode_hist = dict(sorted(Counter(summary["mode_idx"]).items()))
    print(
        "summary: "
        f"success_rate={summary['success'] / num_episodes:.3f} "
        f"crash_rate={summary['crash'] / num_episodes:.3f} "
        f"out_of_road_rate={summary['out_of_road'] / num_episodes:.3f} "
        f"avg_reward={float(np.mean(summary['episode_reward'])):.2f} "
        f"avg_length={float(np.mean(summary['episode_length'])):.1f}"
    )
    if summary["lookahead_y"]:
        lookahead_y = np.asarray(summary["lookahead_y"], dtype=np.float32)
        lookahead_heading = np.asarray(summary["lookahead_heading"], dtype=np.float32)
        steering = np.asarray(summary["steering"], dtype=np.float32)
        print(
            "[analysis_summary] "
            f"mean_lookahead_y={float(lookahead_y.mean()):+.4f} "
            f"rightward_fraction={float((lookahead_y > 0).mean()):.3f} "
            f"mean_heading={float(lookahead_heading.mean()):+.4f} "
            f"mean_steering={float(steering.mean()):+.4f} "
            f"mode_hist={mode_hist}"
        )


if __name__ == "__main__":
    main()
