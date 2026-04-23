from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.co_preference.geometry import build_topology_mask
from models.co_preference.model import CoPreferenceModel
from models.co_preference.teacher import make_teacher_label

DEFAULT_CONFIG_PATH = "configs/train/co_preference.yaml"
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/co_preference")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def build_model_from_config(config: Mapping[str, Any]) -> CoPreferenceModel:
    return CoPreferenceModel(
        status_dim=int(config.get("status_dim", 8)),
        relation_dim=int(config.get("relation_dim", 12)),
        hidden=tuple(config.get("hidden", (256, 256))),
    )


def imitation_loss(
    model: CoPreferenceModel,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(
        status_feature=batch["status_feature"],
        formation_relation_state=batch["formation_relation_state"],
        topology_mask=batch["topology_mask"].bool(),
    )
    topology_loss = F.cross_entropy(out["topology_logits"], batch["teacher_topology_choice"].long())
    s_loss = F.smooth_l1_loss(out["s"], batch["teacher_s"].float())
    total = topology_loss + s_loss
    return total, {
        "imitation_loss": float(total.detach().cpu()),
        "topology_loss": float(topology_loss.detach().cpu()),
        "s_loss": float(s_loss.detach().cpu()),
    }


def _sample_to_teacher(sample: Mapping[str, Any]) -> dict[str, Any]:
    polylines = {
        "current": sample.get("current_lane_polyline"),
        "left": sample.get("left_lane_polyline"),
        "right": sample.get("right_lane_polyline"),
        "branch": sample.get("branch_polyline", sample.get("right_branch_polyline", sample.get("left_branch_polyline"))),
    }
    target_point = np.asarray(sample.get("teacher_target_point", sample.get("target_point")), dtype=np.float32)
    label = make_teacher_label(target_point, polylines)
    return {
        "status_feature": np.asarray(sample["status_feature"], dtype=np.float32),
        "formation_relation_state": np.asarray(sample["formation_relation_state"], dtype=np.float32),
        "topology_mask": build_topology_mask(polylines).astype(np.float32),
        "teacher_topology_choice": np.asarray(label["teacher_topology_choice"], dtype=np.int64),
        "teacher_s": np.asarray(label["teacher_s"], dtype=np.float32),
    }


def load_imitation_npz(path: str | Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=True) as raw:
        size = int(raw["status_feature"].shape[0])
        rows = []
        for idx in range(size):
            sample = {key: raw[key][idx] for key in raw.files}
            rows.append(_sample_to_teacher(sample))
    return {
        key: torch.as_tensor(np.stack([row[key] for row in rows], axis=0))
        for key in rows[0].keys()
    }


def run_imitation(config: Mapping[str, Any], output_root: Path, max_steps: int) -> Path:
    dataset_path = config.get("imitation_dataset")
    if not dataset_path:
        raise ValueError("co-preference imitation requires imitation_dataset in config or CLI.")
    model = build_model_from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("imitation_lr", 1e-3)))
    batch = load_imitation_npz(dataset_path)
    steps = max(1, int(max_steps or config.get("imitation_steps", 1000)))
    last_metrics = {}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, last_metrics = imitation_loss(model, batch)
        loss.backward()
        optimizer.step()
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_root / "co_preference_imitation.pt"
    torch.save({"state_dict": model.state_dict(), "config": dict(config)}, ckpt_path)
    (output_root / "imitation_summary.json").write_text(json.dumps(last_metrics, indent=2), encoding="utf-8")
    return ckpt_path


def run_ppo(config: Mapping[str, Any], output_root: Path, total_env_steps: int) -> None:
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
        from ray.rllib.models import ModelCatalog
    except Exception as exc:  # pragma: no cover - depends on optional ray install
        raise RuntimeError("Install ray[rllib] to run co-preference PPO fine-tuning.") from exc

    from envs.co_preference_platoon_env import CoPreferencePlatoonEnv
    from models.co_preference.rllib_co_preference_model import CoPreferenceRLlibModel

    register_env("co_preference_platoon_env", lambda env_cfg: CoPreferencePlatoonEnv(env_cfg))
    ModelCatalog.register_custom_model("co_preference_model", CoPreferenceRLlibModel)
    def _planner_factory(_env_config):
        from metadrive.policy.diffusion_policy.transfuser_config import (
            build_transfuser_config,
            diffusion_model_config_to_overrides,
            load_diffusion_model_config,
        )
        from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
        from models.platoon.weight_migration import migrate_single_to_platoon

        model_cfg_path = str(config.get("model_config_path", "configs/diffusion/model.yaml"))
        model_cfg = load_diffusion_model_config(model_cfg_path)
        tf_config = build_transfuser_config(
            str(model_cfg.get("model_size", "small")),
            **diffusion_model_config_to_overrides(model_cfg),
        )
        planner = PlatoonDiffusionPlanner(tf_config, num_vehicles=int(config.get("num_agents", 3)))
        pretrained_ckpt = str(config.get("pretrained_ckpt", "") or "")
        if pretrained_ckpt:
            planner = migrate_single_to_platoon(pretrained_ckpt, planner)
        planner.freeze_for_co_preference()
        return planner

    env_cfg = dict(config.get("env_config", {}))
    env_cfg["planner_factory"] = _planner_factory
    env_cfg["reward_config"] = dict(config.get("reward_config", {}))
    env_cfg["preference_reward_config"] = dict(config.get("preference_reward_config", {}))
    train_batch_size = int(config.get("train_batch_size", 1024))
    rollout_fragment_length = int(config.get("rollout_fragment_length", 128))
    iterations = max(1, int(np.ceil(float(total_env_steps) / max(train_batch_size, 1))))
    algo_config = (
        PPOConfig()
        .environment("co_preference_platoon_env", env_config=env_cfg, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=int(config.get("num_rollout_workers", 0)),
            rollout_fragment_length=rollout_fragment_length,
        )
        .training(
            train_batch_size=train_batch_size,
            lr=float(config.get("lr", 3e-4)),
            gamma=float(config.get("gamma", 0.99)),
            model={
                "custom_model": "co_preference_model",
                "custom_model_config": dict(config.get("model_config", {})),
            },
        )
    )
    algo = algo_config.build()
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(iterations):
        result = algo.train()
        (output_root / "last_result.json").write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
    algo.save(str(output_root / "checkpoints"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train co-preference model for platoon target-point generation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stage", choices=("imitation", "ppo"), default="imitation")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=1000)
    parser.add_argument("--imitation-dataset", default="")
    parser.add_argument("--pretrained-ckpt", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.imitation_dataset:
        config["imitation_dataset"] = args.imitation_dataset
    if args.pretrained_ckpt:
        config["pretrained_ckpt"] = args.pretrained_ckpt
    output_root = Path(args.output_root)
    if args.stage == "imitation":
        ckpt_path = run_imitation(config, output_root, args.max_steps)
        print(f"[co_preference] imitation checkpoint: {ckpt_path}")
    else:
        run_ppo(config, output_root, args.total_env_steps)
        print(f"[co_preference] PPO outputs: {output_root}")


if __name__ == "__main__":
    main()
