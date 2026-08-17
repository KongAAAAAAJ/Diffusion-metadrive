"""Diagnostic metric tables and BEV-only visual artifacts for model evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from models.bev_planner.mode_contract import MODE_NAMES


AGENT_IDS = ("agent0", "agent1", "agent2")


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mode_name(index: int) -> str:
    return MODE_NAMES[index] if 0 <= index < len(MODE_NAMES) else f"MODE_{index}"


def _lateral_family(index: int) -> str:
    if index in (0, 1, 2):
        return "keep"
    if index in (3, 4, 5):
        return "left"
    if index in (6, 7, 8):
        return "right"
    return "stop"


def _semantic_bev_rgb(bev: np.ndarray) -> np.ndarray:
    """Convert an arbitrary CxHxW semantic raster to a stable RGB diagnostic."""

    value = np.asarray(bev)
    if value.ndim != 3:
        raise ValueError(f"semantic BEV must be CxHxW, got {value.shape}")
    normalized = value.astype(np.float32)
    if normalized.max(initial=0.0) > 1.0:
        normalized /= 255.0
    rgb = np.zeros((*normalized.shape[1:], 3), dtype=np.float32)
    palette = np.asarray(
        (
            (0.30, 0.30, 0.30),
            (0.95, 0.95, 0.95),
            (0.15, 0.55, 0.95),
            (0.95, 0.25, 0.15),
            (0.25, 0.85, 0.35),
            (0.90, 0.70, 0.15),
            (0.65, 0.30, 0.90),
            (0.15, 0.80, 0.80),
        ),
        dtype=np.float32,
    )
    for channel in range(normalized.shape[0]):
        mask = np.clip(normalized[channel], 0.0, 1.0)[..., None]
        color = palette[channel % len(palette)]
        rgb = np.maximum(rgb, mask * color)
    return np.asarray(np.clip(rgb * 255.0, 0.0, 255.0), dtype=np.uint8)


def _figure_rgb(fig: object) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        height, width, 4
    )[..., :3].copy()


def _write_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import mediapy

    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), list(frames), fps=fps)


def _world_from_local(pose: np.ndarray, local_xy: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    local_xy = np.asarray(local_xy, dtype=np.float64)
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return local_xy @ rotation.T + pose[:2]


def _local_from_world(pose: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    delta = np.asarray(world_xy, dtype=np.float64) - pose[:2]
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    return np.asarray(
        (cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]),
        dtype=np.float64,
    )


class OpenLoopArtifactCollector:
    """Collect legacy-equivalent per-role open-loop metrics and visual evidence."""

    def __init__(self, root: Path, *, save_visualizations: bool = True) -> None:
        self.root = Path(root)
        self.save_visualizations = bool(save_visualizations)
        self.records: list[dict[str, object]] = []
        self.joint_samples = 0

    def add_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        output: Mapping[str, torch.Tensor],
    ) -> None:
        cpu_batch = {name: value.detach().cpu().numpy() for name, value in batch.items()}
        cpu_output = {
            name: value.detach().cpu().numpy()
            for name, value in output.items()
            if isinstance(value, torch.Tensor)
        }
        batch_size = int(cpu_batch["bev"].shape[0])
        for item in range(batch_size):
            joint_index = self.joint_samples
            selected_modes = cpu_output["selected_mode"][item]
            logits = cpu_output["mode_logits"][item]
            selected = cpu_output["selected_trajectory"][item]
            candidates = cpu_output["trajectory_candidates"][item]
            expert = cpu_batch["expert_trajectory"][item]
            gt_modes = cpu_batch["gt_mode"][item]
            for role, agent_id in enumerate(AGENT_IDS):
                pred = np.asarray(selected[role], dtype=np.float64)
                gt = np.asarray(expert[role], dtype=np.float64)
                distances = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1)
                pred_mode = int(selected_modes[role])
                gt_mode = int(gt_modes[role])
                top = np.argsort(logits[role])[::-1][:3]
                self.records.append(
                    {
                        "sample_index": joint_index,
                        "agent_id": agent_id,
                        "pred_mode_idx": pred_mode,
                        "gt_mode_idx": gt_mode,
                        "pred_mode_name": _mode_name(pred_mode),
                        "gt_mode_name": _mode_name(gt_mode),
                        "pred_final_xy": pred[-1, :2].tolist(),
                        "gt_final_xy": gt[-1, :2].tolist(),
                        "pred_trajectory_xy": pred[:, :2].tolist(),
                        "gt_trajectory_xy": gt[:, :2].tolist(),
                        "ade": float(distances.mean()),
                        "fde": float(distances[-1]),
                        "signed_final_y_error": float(pred[-1, 1] - gt[-1, 1]),
                        "trajectory_l1": float(np.abs(pred - gt).mean()),
                        "trajectory_final_l2": float(
                            np.linalg.norm(pred[-1, :2] - gt[-1, :2])
                        ),
                        "topk_mode_logits": [
                            {"mode_idx": int(index), "logit": float(logits[role, index])}
                            for index in top
                        ],
                    }
                )
            if self.save_visualizations:
                self._save_joint_images(
                    joint_index,
                    cpu_batch["bev"][item],
                    candidates,
                    selected,
                    expert,
                    selected_modes,
                    gt_modes,
                )
            self.joint_samples += 1

    def _save_joint_images(
        self,
        sample_index: int,
        bev: np.ndarray,
        candidates: np.ndarray,
        selected: np.ndarray,
        expert: np.ndarray,
        selected_modes: np.ndarray,
        gt_modes: np.ndarray,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for role, axis in enumerate(axes):
            axis.imshow(_semantic_bev_rgb(bev[role]))
            axis.set_title(
                f"{AGENT_IDS[role]} pred={_mode_name(int(selected_modes[role]))}\n"
                f"gt={_mode_name(int(gt_modes[role]))}"
            )
            axis.axis("off")
        fig.suptitle(f"Stage1 open-loop semantic BEV sample {sample_index}")
        fig.tight_layout()
        _write_png(self.root / "images" / f"sample_{sample_index:05d}.png", _figure_rgb(fig))
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for role, axis in enumerate(axes):
            for mode in range(candidates.shape[1]):
                path = candidates[role, mode]
                axis.plot(path[:, 0], path[:, 1], color="#94A3B8", alpha=0.35)
            axis.plot(selected[role, :, 0], selected[role, :, 1], "-o", label="selected")
            axis.plot(expert[role, :, 0], expert[role, :, 1], "-o", label="expert")
            axis.set_title(AGENT_IDS[role])
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.suptitle(f"Stage1 open-loop trajectories sample {sample_index}")
        fig.tight_layout()
        _write_png(
            self.root / "trajectory_plots" / f"sample_{sample_index:05d}.png",
            _figure_rgb(fig),
        )
        plt.close(fig)

    def finalize(self) -> dict[str, object]:
        if not self.records:
            raise ValueError("open-loop artifact collection produced no records")
        metrics = summarize_open_loop_records(self.records)
        _json_dump(self.root / "open_loop_samples.json", self.records)
        self.root.mkdir(parents=True, exist_ok=True)
        csv_path = self.root / "open_loop_samples.csv"
        scalar_fields = (
            "sample_index",
            "agent_id",
            "pred_mode_idx",
            "gt_mode_idx",
            "pred_mode_name",
            "gt_mode_name",
            "ade",
            "fde",
            "signed_final_y_error",
            "trajectory_l1",
            "trajectory_final_l2",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_fields)
            writer.writeheader()
            writer.writerows(
                {name: record[name] for name in scalar_fields} for record in self.records
            )
        summary = {
            "joint_samples": self.joint_samples,
            "role_samples": len(self.records),
            "metrics": metrics,
            "artifacts": {
                "samples_json": str(self.root / "open_loop_samples.json"),
                "samples_csv": str(csv_path),
                "images": str(self.root / "images") if self.save_visualizations else None,
                "trajectory_plots": (
                    str(self.root / "trajectory_plots")
                    if self.save_visualizations
                    else None
                ),
            },
        }
        _json_dump(self.root / "open_loop_summary.json", summary)
        return summary


def summarize_open_loop_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    pred_modes = [int(record["pred_mode_idx"]) for record in records]
    gt_modes = [int(record["gt_mode_idx"]) for record in records]
    confusion: dict[str, dict[str, int]] = defaultdict(dict)
    per_slot = {}
    for slot in range(len(MODE_NAMES)):
        indices = [index for index, value in enumerate(gt_modes) if value == slot]
        correct = sum(pred_modes[index] == slot for index in indices)
        per_slot[MODE_NAMES[slot]] = {
            "accuracy": correct / len(indices) if indices else 0.0,
            "correct": correct,
            "total": len(indices),
        }
    for gt_mode, pred_mode in zip(gt_modes, pred_modes):
        row = confusion.setdefault(_mode_name(gt_mode), {})
        row[_mode_name(pred_mode)] = row.get(_mode_name(pred_mode), 0) + 1
    pred_y = np.asarray([record["pred_final_xy"][1] for record in records], dtype=np.float64)
    gt_y = np.asarray([record["gt_final_xy"][1] for record in records], dtype=np.float64)
    metric = lambda name: float(np.mean([float(record[name]) for record in records]))
    mode_matches = sum(first == second for first, second in zip(pred_modes, gt_modes))
    lateral_matches = sum(
        _lateral_family(first) == _lateral_family(second)
        for first, second in zip(pred_modes, gt_modes)
    )
    pred_hist = {str(key): value for key, value in sorted(Counter(pred_modes).items())}
    gt_hist = {str(key): value for key, value in sorted(Counter(gt_modes).items())}
    mode_final_y_mean = {
        str(mode): float(np.mean([pred_y[index] for index, value in enumerate(pred_modes) if value == mode]))
        for mode in sorted(set(pred_modes))
    }
    return {
        "num_samples": len(records),
        "trajectory_l1": metric("trajectory_l1"),
        "trajectory_final_l2": metric("trajectory_final_l2"),
        "ade_mean": metric("ade"),
        "fde_mean": metric("fde"),
        "signed_final_y_error": metric("signed_final_y_error"),
        "mean_pred_final_y": float(pred_y.mean()),
        "mean_gt_final_y": float(gt_y.mean()),
        "rightward_bias_fraction": float((pred_y > 0.0).mean()),
        "mode_hist": pred_hist,
        "mode_final_y_mean": mode_final_y_mean,
        "trajectory_mode_hist": gt_hist,
        "pred_mode_hist": pred_hist,
        "gt_mode_hist": gt_hist,
        "mode_accuracy": mode_matches / len(records),
        "mode_match_count": mode_matches,
        "mode_total_count": len(records),
        "lateral_accuracy": lateral_matches / len(records),
        "lateral_match_count": lateral_matches,
        "per_slot_accuracy": per_slot,
        "mode_confusion": confusion,
    }


@dataclass
class _EpisodeArtifacts:
    frames_topdown: list[np.ndarray] = field(default_factory=list)
    frames_bev: list[np.ndarray] = field(default_factory=list)
    positions: dict[str, list[list[float]]] = field(
        default_factory=lambda: {agent_id: [] for agent_id in AGENT_IDS}
    )
    control_records: list[dict[str, object]] = field(default_factory=list)
    coordinate_records: list[dict[str, object]] = field(default_factory=list)
    step_records: list[dict[str, object]] = field(default_factory=list)


class ClosedLoopArtifactWriter:
    """Write BEV-only equivalents of the legacy closed-loop evidence bundle."""

    def __init__(
        self,
        root: Path,
        model_id: str,
        *,
        video_fps: int = 10,
        frame_interval: int = 1,
    ) -> None:
        self.root = Path(root) / model_id
        self.video_fps = int(video_fps)
        self.frame_interval = int(frame_interval)
        self.episodes: dict[str, _EpisodeArtifacts] = {}

    @staticmethod
    def episode_id(scenario: Sequence[str], seed: int) -> str:
        return f"{scenario[0]}__{scenario[1]}__seed_{int(seed)}"

    def start_episode(self, scenario: Sequence[str], seed: int) -> str:
        episode_id = self.episode_id(scenario, seed)
        self.episodes[episode_id] = _EpisodeArtifacts()
        return episode_id

    def record_step(
        self,
        episode_id: str,
        *,
        step_index: int,
        dt_s: float,
        pre_poses: np.ndarray,
        post_poses: np.ndarray,
        trajectories: np.ndarray | None,
        selected_modes: Sequence[int] | None,
        rewards: Mapping[str, float],
        controls: Mapping[str, np.ndarray] | None,
        bev: np.ndarray | None,
    ) -> None:
        episode = self.episodes[episode_id]
        pre = np.asarray(pre_poses, dtype=np.float64)
        post = np.asarray(post_poses, dtype=np.float64)
        planned_world: list[np.ndarray | None] = []
        step_records = []
        for role, agent_id in enumerate(AGENT_IDS):
            episode.positions[agent_id].append(post[role, :2].tolist())
            planned = None
            reference_local = np.zeros(2, dtype=np.float64)
            if trajectories is not None:
                local = np.asarray(trajectories[role, :, :2], dtype=np.float64)
                planned = _world_from_local(pre[role], local)
                reference_local = local[0] * min(max(float(dt_s) / 0.5, 0.0), 1.0)
            planned_world.append(planned)
            actual_local = _local_from_world(pre[role], post[role, :2])
            error = actual_local - reference_local
            control = None if controls is None else np.asarray(controls.get(agent_id, ()), dtype=np.float64)
            actual_speed_km_h = float(
                np.linalg.norm(post[role, :2] - pre[role, :2])
                / max(float(dt_s), 1.0e-6)
                * 3.6
            )
            target_speed_km_h = float(
                np.linalg.norm(reference_local) / max(float(dt_s), 1.0e-6) * 3.6
            )
            prior = next(
                (
                    value
                    for value in reversed(episode.control_records)
                    if value["agent_id"] == agent_id
                ),
                None,
            )
            record = {
                "episode_id": episode_id,
                "step_idx": int(step_index),
                "agent_id": agent_id,
                "lateral_error_m": float(error[1]),
                "longitudinal_error_m": float(error[0]),
                "abs_lateral_error_m": float(abs(error[1])),
                "abs_longitudinal_error_m": float(abs(error[0])),
                "reference_world_x": float(_world_from_local(pre[role], reference_local)[0]),
                "reference_world_y": float(_world_from_local(pre[role], reference_local)[1]),
                "actual_world_x": float(post[role, 0]),
                "actual_world_y": float(post[role, 1]),
                "speed_km_h": actual_speed_km_h,
                "target_speed_km_h": target_speed_km_h,
                "speed_error_km_h": actual_speed_km_h - target_speed_km_h,
                "acceleration_mps2": (
                    (actual_speed_km_h - float(prior["speed_km_h"]))
                    / 3.6
                    / max(float(dt_s), 1.0e-6)
                    if prior is not None
                    else None
                ),
                "steering": float(control[0]) if control is not None and control.size >= 2 else None,
                "throttle": float(control[1]) if control is not None and control.size >= 2 else None,
            }
            episode.control_records.append(record)
            step_records.append(
                {
                    "agent_id": agent_id,
                    "pre_pose": pre[role].tolist(),
                    "post_pose": post[role].tolist(),
                    "selected_mode": (
                        int(selected_modes[role]) if selected_modes is not None else None
                    ),
                    "reward": float(rewards.get(agent_id, 0.0)),
                    "planned_trajectory_local": (
                        np.asarray(trajectories[role]).tolist()
                        if trajectories is not None
                        else None
                    ),
                    "planned_trajectory_world": planned.tolist() if planned is not None else None,
                }
            )
            episode.coordinate_records.append(
                {
                    "episode_id": episode_id,
                    "step_idx": int(step_index),
                    "agent_id": agent_id,
                    "ego_pose_global": pre[role].tolist(),
                    "planned_local_xy": (
                        np.asarray(trajectories[role, :, :2]).tolist()
                        if trajectories is not None
                        else None
                    ),
                    "planned_world_xy": planned.tolist() if planned is not None else None,
                    "actual_post_world_xy": post[role, :2].tolist(),
                }
            )
        step_payload = {
            "episode_id": episode_id,
            "step_idx": int(step_index),
            "dt_s": float(dt_s),
            "agents": step_records,
        }
        episode.step_records.append(step_payload)
        _json_dump(
            self.root / "trajectory_data" / episode_id / f"step_{step_index:05d}.json",
            step_payload,
        )
        if step_index % self.frame_interval == 0:
            topdown = self._render_topdown(episode_id, step_index, planned_world, selected_modes)
            episode.frames_topdown.append(topdown)
            _write_png(
                self.root / "combined_traj_frames" / episode_id / f"step_{step_index:05d}.png",
                topdown,
            )
            if bev is not None:
                bev_frame = self._render_bev(episode_id, step_index, bev, selected_modes)
                episode.frames_bev.append(bev_frame)
                _write_png(
                    self.root / "step_images" / episode_id / f"step_{step_index:05d}.png",
                    bev_frame,
                )

    def _render_topdown(
        self,
        episode_id: str,
        step_index: int,
        planned_world: Sequence[np.ndarray | None],
        selected_modes: Sequence[int] | None,
    ) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episode = self.episodes[episode_id]
        colors = ("#2563EB", "#059669", "#DC2626")
        fig, axis = plt.subplots(figsize=(8, 8))
        for role, agent_id in enumerate(AGENT_IDS):
            positions = np.asarray(episode.positions[agent_id], dtype=np.float64)
            if positions.size:
                axis.plot(positions[:, 0], positions[:, 1], "-o", color=colors[role], label=f"{agent_id} actual")
            planned = planned_world[role]
            if planned is not None:
                label = f"{agent_id} plan"
                if selected_modes is not None:
                    label += f" ({_mode_name(int(selected_modes[role]))})"
                axis.plot(planned[:, 0], planned[:, 1], "--", color=colors[role], alpha=0.8, label=label)
        axis.set_title(f"{episode_id} step {step_index}: planned vs actual")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
        fig.tight_layout()
        image = _figure_rgb(fig)
        plt.close(fig)
        return image

    @staticmethod
    def _render_bev(
        episode_id: str,
        step_index: int,
        bev: np.ndarray,
        selected_modes: Sequence[int] | None,
    ) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for role, axis in enumerate(axes):
            axis.imshow(_semantic_bev_rgb(bev[role]))
            mode = "warmup" if selected_modes is None else _mode_name(int(selected_modes[role]))
            axis.set_title(f"{AGENT_IDS[role]}: {mode}")
            axis.axis("off")
        fig.suptitle(f"{episode_id} step {step_index}: semantic BEV")
        fig.tight_layout()
        image = _figure_rgb(fig)
        plt.close(fig)
        return image

    def finish_episode(self, episode_id: str) -> None:
        episode = self.episodes[episode_id]
        audit_path = self.root / "coordinate_audit" / f"{episode_id}.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as handle:
            for record in episode.coordinate_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        _write_video(
            self.root / "videos_2d" / f"{episode_id}.mp4",
            episode.frames_topdown,
            self.video_fps,
        )
        _write_video(
            self.root / "videos_semantic_bev" / f"{episode_id}.mp4",
            episode.frames_bev,
            self.video_fps,
        )
        self._save_control_plots(episode_id, episode.control_records)

    def _save_control_plots(
        self, episode_id: str, records: Sequence[Mapping[str, object]]
    ) -> None:
        if not records:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output = self.root / "control_errors"
        output.mkdir(parents=True, exist_ok=True)
        for metric, filename, ylabel in (
            ("lateral_error_m", "lat_error", "lateral error (m)"),
            ("longitudinal_error_m", "lon_error", "longitudinal error (m)"),
            ("speed_error_km_h", "speed_error", "speed error (km/h)"),
        ):
            fig, axis = plt.subplots(figsize=(10, 4))
            for agent_id in AGENT_IDS:
                subset = [record for record in records if record["agent_id"] == agent_id]
                axis.plot(
                    [record["step_idx"] for record in subset],
                    [record[metric] for record in subset],
                    label=agent_id,
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(f"{episode_id}: {ylabel}")
            axis.set_xlabel("step")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend()
            fig.tight_layout()
            fig.savefig(output / f"{episode_id}_{filename}.png", dpi=150)
            plt.close(fig)

        fig, axis = plt.subplots(figsize=(10, 4))
        for agent_id in AGENT_IDS:
            subset = [record for record in records if record["agent_id"] == agent_id]
            axis.plot([record["step_idx"] for record in subset], [record["throttle"] for record in subset], label=f"{agent_id} throttle")
            axis.plot([record["step_idx"] for record in subset], [record["steering"] for record in subset], "--", label=f"{agent_id} steer")
        axis.set_title(f"{episode_id}: control commands")
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{episode_id}_control.png", dpi=150)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(8, 8))
        for agent_id in AGENT_IDS:
            subset = [record for record in records if record["agent_id"] == agent_id]
            axis.plot(
                [record["reference_world_x"] for record in subset],
                [record["reference_world_y"] for record in subset],
                "--",
                label=f"{agent_id} reference",
            )
            axis.plot(
                [record["actual_world_x"] for record in subset],
                [record["actual_world_y"] for record in subset],
                "-",
                label=f"{agent_id} actual",
            )
        axis.set_title(f"{episode_id}: time-aligned reference vs actual")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{episode_id}_trajectory_compare.png", dpi=150)
        plt.close(fig)

    def finalize(self) -> dict[str, object]:
        records = [record for episode in self.episodes.values() for record in episode.control_records]
        step_records_path = self.root / "step_records.jsonl"
        step_records_path.parent.mkdir(parents=True, exist_ok=True)
        with step_records_path.open("w", encoding="utf-8") as handle:
            for episode in self.episodes.values():
                for record in episode.step_records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        lateral = np.asarray([record["abs_lateral_error_m"] for record in records], dtype=np.float64)
        longitudinal = np.asarray([record["abs_longitudinal_error_m"] for record in records], dtype=np.float64)
        summary = {
            "definition": (
                "Actual post-step pose is transformed into the pre-step ego frame and "
                "compared with the selected trajectory interpolated at the control interval."
            ),
            "num_role_steps": len(records),
            "mean_abs_lateral_error_m": float(lateral.mean()) if lateral.size else None,
            "max_abs_lateral_error_m": float(lateral.max()) if lateral.size else None,
            "p95_abs_lateral_error_m": float(np.percentile(lateral, 95)) if lateral.size else None,
            "mean_abs_longitudinal_error_m": float(longitudinal.mean()) if longitudinal.size else None,
            "max_abs_longitudinal_error_m": float(longitudinal.max()) if longitudinal.size else None,
            "p95_abs_longitudinal_error_m": float(np.percentile(longitudinal, 95)) if longitudinal.size else None,
            "episodes": {
                episode_id: {"records": episode.control_records}
                for episode_id, episode in sorted(self.episodes.items())
            },
        }
        _json_dump(self.root / "control_errors" / "control_error_summary.json", summary)
        return {
            "root": str(self.root),
            "control_error_summary": str(
                self.root / "control_errors" / "control_error_summary.json"
            ),
            "trajectory_data": str(self.root / "trajectory_data"),
            "step_records_jsonl": str(step_records_path),
            "coordinate_audit": str(self.root / "coordinate_audit"),
            "combined_trajectory_frames": str(self.root / "combined_traj_frames"),
            "semantic_bev_images": str(self.root / "step_images"),
            "topdown_videos": str(self.root / "videos_2d"),
            "semantic_bev_videos": str(self.root / "videos_semantic_bev"),
            "control_error_metrics": {key: value for key, value in summary.items() if key != "episodes"},
        }
