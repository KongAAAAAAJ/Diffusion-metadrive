from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from metadrive.policy.base_policy import BasePolicy
from models.diffusion.transfuser_config import TransfuserConfig, TrajectorySampling
from models.diffusion.transfuser_features import (
    LaneDecision,
    decide_lane_change_for_vehicle,
    observation_to_features,
)
from models.diffusion.transfuser_model_v2 import V2TransfuserModel


def _merge_transfuser_config(overrides: Optional[Dict[str, Any]]) -> TransfuserConfig:
    if overrides is None:
        return TransfuserConfig()
    merged = dict(overrides)
    if "trajectory_sampling" in merged and isinstance(merged["trajectory_sampling"], Mapping):
        merged["trajectory_sampling"] = TrajectorySampling(**dict(merged["trajectory_sampling"]))
    if "bev_semantic_classes" in merged and isinstance(merged["bev_semantic_classes"], Mapping):
        merged["bev_semantic_classes"] = {
            int(key): value for key, value in dict(merged["bev_semantic_classes"]).items()
        }
    return TransfuserConfig(**merged)


def _wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _trajectory_reference_speed_sequence_km_h(
    trajectory: np.ndarray,
    interval_s: float = 0.5,
) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if trajectory.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)

    interval_s = max(float(interval_s), 1e-3)
    segment_distances = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    segment_speeds_mps = segment_distances / interval_s
    speed_sequence_mps = np.concatenate(
        [segment_speeds_mps, segment_speeds_mps[-1:].copy()],
        axis=0,
    )
    return (speed_sequence_mps * 3.6).astype(np.float32, copy=False)


def compute_trajectory_control(
    trajectory: np.ndarray,
    lookahead_index: int,
    current_speed_km_h: float,
    target_speed_km_h: float,
    controller_type: str = "stabilized",
) -> tuple[np.ndarray, Dict[str, float]]:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    reference_speed_sequence_km_h = _trajectory_reference_speed_sequence_km_h(trajectory)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0:
        action = np.asarray([0.0, 0.0], dtype=np.float32)
        return action, {
            "controller_type": controller_type,
            "lookahead_index": 0,
            "waypoint_x": 0.0,
            "waypoint_y": 0.0,
            "waypoint_heading": 0.0,
            "steering": 0.0,
            "throttle": 0.0,
            "trajectory_target_speed_km_h": 0.0,
            "target_speed_km_h": float(target_speed_km_h),
            "speed_error": float(target_speed_km_h - current_speed_km_h),
        }

    valid_idx = min(max(int(lookahead_index), 0), max(len(trajectory) - 1, 0))
    waypoint = trajectory[valid_idx]
    x = float(waypoint[0])
    y = float(waypoint[1])
    heading = float(waypoint[2]) if waypoint.shape[0] > 2 else 0.0
    forward = max(x, 1e-3)

    steering_angle = float(np.arctan2(y, forward))
    wrapped_heading = _wrap_to_pi(heading)
    if controller_type == "legacy":
        steering_pre_clip = 1.5 * steering_angle + 0.2 * wrapped_heading
    else:
        if abs(y) < 0.35 and abs(wrapped_heading) < 0.04:
            steering_angle = 0.0
            wrapped_heading = 0.0
        speed_scale = float(np.clip(20.0 / max(current_speed_km_h, 5.0), 0.55, 1.15))
        steering_pre_clip = speed_scale * (0.85 * steering_angle + 0.35 * wrapped_heading)

    steering = float(np.clip(steering_pre_clip, -1.0, 1.0))
    trajectory_target_speed_km_h = (
        float(reference_speed_sequence_km_h[min(valid_idx, len(reference_speed_sequence_km_h) - 1)])
        if len(reference_speed_sequence_km_h) > 0
        else 0.0
    )
    effective_target_speed_km_h = min(float(target_speed_km_h), trajectory_target_speed_km_h)
    speed_error = float(effective_target_speed_km_h - current_speed_km_h)
    throttle = float(np.clip(speed_error / 10.0, -1.0, 1.0))
    action = np.asarray([steering, throttle], dtype=np.float32)
    debug = {
        "controller_type": controller_type,
        "lookahead_index": valid_idx,
        "waypoint_x": x,
        "waypoint_y": y,
        "waypoint_heading": heading,
        "steering_angle": steering_angle,
        "heading_term": wrapped_heading,
        "steering_pre_clip": float(steering_pre_clip),
        "steering": steering,
        "throttle": throttle,
        "trajectory_target_speed_km_h": float(trajectory_target_speed_km_h),
        "target_speed_km_h": float(effective_target_speed_km_h),
        "speed_error": speed_error,
    }
    return action, debug


class TransfuserPolicy(BasePolicy):
    """Closed-loop MetaDrive policy backed by a trained TransFuser checkpoint."""

    DEBUG_MARK_COLOR = (72, 201, 176, 255)

    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        global_config = self.engine.global_config
        self._model_config = _merge_transfuser_config(global_config.get("transfuser_config"))
        self._checkpoint_path = global_config.get("transfuser_checkpoint_path", None)
        self._device = torch.device(global_config.get("transfuser_policy_device", "cpu"))
        self._target_speed_km_h = float(global_config.get("transfuser_target_speed_km_h", 30.0))
        self._lookahead_index = int(global_config.get("transfuser_lookahead_index", 2))
        self._controller_type = str(global_config.get("transfuser_controller_type", "stabilized"))
        self._trajectory_nodes = []
        # Mirrors IDMPolicy.overtake_timer: incremented each step, reset to 0
        # after a lane change so the diffusion planner respects IDM's LANE_CHANGE_FREQ gate.
        self._lane_change_overtake_timer: int = 0

        self._model = V2TransfuserModel(self._model_config)
        self._load_checkpoint(self._checkpoint_path)
        self._model.eval()
        self._model.to(self._device)

    def reset(self):
        super().reset()
        self._lane_change_overtake_timer = 0
        self._clear_trajectory_visualization()

    def _build_mode_features(self) -> Dict[str, torch.Tensor]:
        """Build coarse_trajectories and mode_valid_mask from the live vehicle state."""
        from models.diffusion.mode_context import build_mode_context_from_vehicle
        from models.diffusion.mode_trajectory_generator import ModeTrajectoryGenerator
        from models.diffusion.mode_definitions import build_mode_slots, mode_slot_count
        mode_slots = build_mode_slots(
            keep_lane_count=self._model_config.mode_keep_lane_count,
            lane_change_left_count=self._model_config.mode_lane_change_left_count,
            lane_change_right_count=self._model_config.mode_lane_change_right_count,
            emergency_stop_count=self._model_config.mode_emergency_stop_count,
        )
        num_slots = mode_slot_count(
            self._model_config.mode_keep_lane_count,
            self._model_config.mode_lane_change_left_count,
            self._model_config.mode_lane_change_right_count,
            self._model_config.mode_emergency_stop_count,
        )
        try:
            current_map = getattr(getattr(self.control_object, "engine", None), "current_map", None)
            ctx = build_mode_context_from_vehicle(self.control_object, current_map=current_map)
            gen = ModeTrajectoryGenerator(
                keep_lane_high_speed_mps=self._model_config.mode_keep_high_speed_mps,
                keep_lane_medium_speed_mps=self._model_config.mode_keep_medium_speed_mps,
                keep_lane_low_speed_mps=self._model_config.mode_keep_low_speed_mps,
                emergency_decel_mps2=self._model_config.mode_emergency_decel_mps2,
                keep_lane_level_count=self._model_config.mode_keep_lane_count,
                lane_change_left_level_count=self._model_config.mode_lane_change_left_count,
                lane_change_right_level_count=self._model_config.mode_lane_change_right_count,
                emergency_stop_level_count=self._model_config.mode_emergency_stop_count,
                mode_slots=mode_slots,
            )
            out = gen.generate(ctx)
            return {
                "coarse_trajectories": torch.from_numpy(out.coarse_trajectories),
                "mode_valid_mask": torch.from_numpy(out.mode_valid_mask),
            }
        except Exception:
            return {
                "coarse_trajectories": torch.zeros((num_slots, 8, 2), dtype=torch.float32),
                "mode_valid_mask": torch.zeros((num_slots,), dtype=torch.bool),
            }

    def act(self, agent_id=None):
        # lane_decision is external conditioning — decide BEFORE model inference so
        # that target_point (which depends on lane_decision) is fed as model input.
        lane_decision = decide_lane_change_for_vehicle(
            self.control_object, overtake_timer=self._lane_change_overtake_timer
        )
        if lane_decision != LaneDecision.KEEP:
            self._lane_change_overtake_timer = 0
        else:
            self._lane_change_overtake_timer += 1

        observation_adapter = self.engine.agent_manager.observations[self.control_object.name]
        observation = getattr(observation_adapter, "current_observation", None)
        if observation is None:
            observation = observation_adapter.observe(self.control_object)
        features = observation_to_features(
            observation, self._model_config, vehicle=self.control_object, lane_decision=lane_decision
        )
        features["lane_decision"] = torch.tensor(int(lane_decision), dtype=torch.int8)
        features.update(self._build_mode_features())
        batched_features = {
            key: value.unsqueeze(0).to(self._device) if value.ndim > 0 else value.to(self._device)
            for key, value in features.items()
        }

        with torch.no_grad():
            predictions = self._model.infer_multimodal(batched_features)

        trajectory = predictions["trajectory"][0].detach().cpu().numpy()
        self._update_trajectory_visualization(trajectory)
        action, controller_debug = self._trajectory_to_action(trajectory)
        self.action_info["action"] = action
        self.action_info["predicted_trajectory"] = trajectory
        self.action_info["controller_debug"] = controller_debug
        self.action_info["lane_decision"] = int(lane_decision)
        for feature_key in (
            "camera_feature",
            "lidar_feature",
            "status_feature",
            "ego_state",
            "target_point",
            "preference_point",
            "target_line",
            "topology_polyline",
            "lane_decision",
        ):
            if feature_key in features:
                val = features[feature_key]
                self.action_info[feature_key] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val
        for mode_feature_key in ("coarse_trajectories", "mode_valid_mask"):
            if mode_feature_key in features:
                val = features[mode_feature_key]
                self.action_info[mode_feature_key] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val
        from models.diffusion.mode_definitions import build_mode_slots
        self.action_info["mode_slot_names"] = [
            slot.name
            for slot in build_mode_slots(
                keep_lane_count=self._model_config.mode_keep_lane_count,
                lane_change_left_count=self._model_config.mode_lane_change_left_count,
                lane_change_right_count=self._model_config.mode_lane_change_right_count,
                emergency_stop_count=self._model_config.mode_emergency_stop_count,
            )
        ]
        if "trajectory_candidates" in predictions:
            self.action_info["trajectory_candidates"] = (
                predictions["trajectory_candidates"][0].detach().cpu().numpy()
            )
        if "trajectory_mode_idx" in predictions:
            self.action_info["trajectory_mode_idx"] = int(predictions["trajectory_mode_idx"][0].item())
        if "trajectory_mode_logits" in predictions:
            self.action_info["trajectory_mode_logits"] = (
                predictions["trajectory_mode_logits"][0].detach().cpu().numpy()
            )
        return action

    def _trajectory_to_action(self, trajectory: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
        return compute_trajectory_control(
            trajectory=trajectory,
            lookahead_index=self._lookahead_index,
            current_speed_km_h=float(self.control_object.speed_km_h),
            target_speed_km_h=self._target_speed_km_h,
            controller_type=self._controller_type,
        )

    def _load_checkpoint(self, checkpoint_path: Optional[str]) -> None:
        if not checkpoint_path:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_transfuser_model."):
                cleaned_state_dict[key[len("_transfuser_model."):]] = value
            elif key.startswith("agent._transfuser_model."):
                cleaned_state_dict[key[len("agent._transfuser_model."):]] = value
            else:
                cleaned_state_dict[key] = value
        # Drop keys whose shape no longer matches the current model (e.g. plan_anchor
        # when ego_fut_mode changed from 7→10 in the hierarchical-anchor upgrade).
        # With use_dynamic_anchors=True, plan_anchor is only a fallback and can safely
        # retain its randomly-initialised value when the checkpoint shape differs.
        model_state = self._model.state_dict()
        mismatched_keys = [
            k for k, v in cleaned_state_dict.items()
            if k in model_state and v.shape != model_state[k].shape
        ]
        if mismatched_keys:
            import warnings
            warnings.warn(
                f"Dropping {len(mismatched_keys)} checkpoint key(s) due to shape mismatch "
                f"(will use model-init values): {mismatched_keys}",
                stacklevel=2,
            )
            for k in mismatched_keys:
                del cleaned_state_dict[k]
        try:
            self._model.load_state_dict(cleaned_state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load TransFuser checkpoint. "
                "The checkpoint is likely incompatible with the requested model_size/config. "
                f"checkpoint={checkpoint_path} model_size={self._model_config.model_size}"
            ) from exc

    def _clear_trajectory_visualization(self) -> None:
        for node in self._trajectory_nodes:
            node.detachNode()
            node.removeNode()
        self._trajectory_nodes = []

    def _update_trajectory_visualization(self, trajectory: np.ndarray) -> None:
        self._clear_trajectory_visualization()
        if not self.engine.global_config.get("use_render", False):
            return
        if trajectory is None or len(trajectory) == 0:
            return

        height = getattr(self.control_object, "HEIGHT", 2.0) * 0.5 + 0.3
        points = [np.asarray(self.control_object.position, dtype=np.float32)]
        for waypoint in trajectory:
            world_xy = self.control_object.convert_to_world_coordinates(
                [float(waypoint[0]), float(waypoint[1])], self.control_object.position
            )
            points.append(np.asarray(world_xy[:2], dtype=np.float32))

        for start, end in zip(points[:-1], points[1:]):
            node = self.engine._draw_line_3d(
                (float(start[0]), float(start[1]), height),
                (float(end[0]), float(end[1]), height),
                color=(0.95, 0.35, 0.1, 1.0),
                thickness=3.0,
            )
            node.reparentTo(self.engine.render)
            self._trajectory_nodes.append(node)

    def destroy(self):
        self._clear_trajectory_visualization()
        super().destroy()

    @classmethod
    def default_runtime_config(cls) -> Dict[str, Any]:
        config = TransfuserConfig()
        config_dict = asdict(config) if is_dataclass(config) else {}
        return {
            "transfuser_config": config_dict,
            "transfuser_checkpoint_path": None,
            "transfuser_policy_device": "cpu",
            "transfuser_target_speed_km_h": 30.0,
            "transfuser_lookahead_index": 2,
            "transfuser_controller_type": "stabilized",
        }
