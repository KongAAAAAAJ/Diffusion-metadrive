from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from evaluation.reward_terms import compute_trajectory_reward
from models.diffusion.diffusion_rl_scheduler import DiffusionRLScheduler


def _obs_to_tensor_batch(obs: Mapping[str, Mapping[str, object]], device: torch.device) -> Dict[str, Dict[str, Tensor]]:
    batch: Dict[str, Dict[str, Tensor]] = {}
    for agent_id, sample in obs.items():
        batch[agent_id] = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                tensor = value.to(device=device, dtype=torch.float32)
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            batch[agent_id][key] = tensor
    return batch


class MultiAgentGRPOTrainer:
    def __init__(self, model, ref_model, env, config: dict):
        self.model = model
        self.ref_model = ref_model
        self.env = env
        self.config = dict(config or {})
        self.device = next(model.parameters()).device
        self.scheduler = DiffusionRLScheduler(
            {
                "ddim_steps": self.config.get("ddim_steps", 4),
                "ddim_eta": self.config.get("ddim_eta", 0.5),
                "num_inference_steps": self.config.get("ddim_steps", 4),
                "eta": self.config.get("ddim_eta", 0.5),
                "prediction_type": self.config.get("prediction_type", "sample"),
            }
        )
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("Model has no trainable parameters after applying freeze config")
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(self.config.get("lr", 1e-3)))
        self.advantage_gamma = float(self.config.get("advantage_discount_gamma", 0.8))
        self.reward_config = dict(self.config.get("reward_config", {}))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 10.0))
        self.max_env_steps_per_rollout = int(self.config.get("max_env_steps_per_rollout", 0))
        self.global_step = 0
        self.env_step_counter = 0
        self.current_obs = None
        self._cached_current_preds: dict[str, list[Tensor]] = {}

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    def _evaluate_trajectory_group(self, agent_id: str, trajectories: Tensor) -> dict:
        if hasattr(self.env, "evaluate_trajectory_group"):
            if torch.is_tensor(trajectories):
                trajectories_np = trajectories.detach().cpu().numpy()
            else:
                trajectories_np = trajectories
            return self.env.evaluate_trajectory_group(agent_id, trajectories_np)

        step_infos = []
        crash_flags = []
        out_flags = []
        for trajectory in trajectories:
            progress = float(trajectory[-1, 0].item())
            info = {
                "progress": progress,
                "formation_error": float(torch.abs(trajectory[:, 1]).mean().item()),
                "min_gap": 10.0,
                "jerk": 0.0,
                "delta_steering": 0.0,
                "crash": False,
                "out_of_road": False,
            }
            step_infos.append([info for _ in range(int(trajectory.shape[0]))])
            crash_flags.append(False)
            out_flags.append(False)
        return {
            "step_infos": step_infos,
            "crash_flags": crash_flags,
            "out_of_road_flags": out_flags,
        }

    def collect_group_samples(self, group_size: int = 4, obs: dict | None = None) -> dict:
        """对每个智能体批量采样多组多 anchor 轨迹，并评估每个 anchor 的奖励、安全性等指标"""

        # 观测处理
        if obs is None:
            obs = self.current_obs if self.current_obs is not None else self.env.reset()
        self.current_obs = obs
        batch = _obs_to_tensor_batch(obs, self.device)
        with torch.no_grad():
            samples = self.scheduler.sample_with_log_prob(self.model, batch, num_groups=group_size)

        # 对每个 agent，遍历所有 anchor（num_groups × num_modes）
        rollouts: dict = {"agent_ids": list(batch.keys()), "batch": batch}
        for agent_id in batch.keys():
            sampled = samples[agent_id]
            trajectory_all = sampled["trajectory"]
            log_prob_all = sampled["log_prob"]
            num_groups, num_modes = trajectory_all.shape[:2]

            rewards_per_anchor = torch.zeros(num_groups, num_modes, dtype=torch.float32, device=self.device)
            crash_per_anchor = torch.zeros(num_groups, num_modes, dtype=torch.bool, device=self.device)
            out_per_anchor = torch.zeros(num_groups, num_modes, dtype=torch.bool, device=self.device)
            formation_per_anchor = torch.zeros(num_groups, num_modes, dtype=torch.float32, device=self.device)

            for mode_idx in range(num_modes):
                # 对一组轨迹（如[num_groups, step_num, action_dim]）进行环境相关的评估，输出每步的info字典（如progress、formation_error、crash、out_of_road等
                evaluation = self._evaluate_trajectory_group(agent_id, trajectory_all[:, mode_idx, :, :])
                
                # 计算单条轨迹的
                rewards_per_anchor[:, mode_idx] = torch.tensor(
                    [compute_trajectory_reward(step_infos, self.reward_config) for step_infos in evaluation["step_infos"]],
                    dtype=torch.float32,
                    device=self.device,
                )
                crash_per_anchor[:, mode_idx] = torch.as_tensor(
                    evaluation["crash_flags"], dtype=torch.bool, device=self.device
                )
                out_per_anchor[:, mode_idx] = torch.as_tensor(
                    evaluation["out_of_road_flags"], dtype=torch.bool, device=self.device
                )
                formation_per_anchor[:, mode_idx] = torch.tensor(
                    [
                        float(sum(float(step.get("formation_error", 0.0)) for step in step_infos) / max(len(step_infos), 1))
                        for step_infos in evaluation["step_infos"]
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

            # 选出最优的anchor
            flat_best_idx = int(torch.argmax(rewards_per_anchor.view(-1)).item())
            best_g = flat_best_idx // num_modes
            best_k = flat_best_idx % num_modes

            # 保存每个智能体的轨迹、奖励、安全性等评估结果，以及选出的最优anchor索引
            rollouts[agent_id] = {
                "trajectory": trajectory_all[:, best_k, :, :].detach(),
                "trajectory_all": trajectory_all.detach(),
                "log_prob": log_prob_all[:, best_k, :].detach(),
                "log_prob_all": log_prob_all.detach(),
                "reward": rewards_per_anchor[:, best_k],
                "reward_per_anchor": rewards_per_anchor,
                "crash_flag": crash_per_anchor[:, best_k],
                "crash_per_anchor": crash_per_anchor,
                "out_of_road_flag": out_per_anchor[:, best_k],
                "out_per_anchor": out_per_anchor,
                "formation_error": formation_per_anchor[:, best_k],
                "formation_per_anchor": formation_per_anchor,
                "diffusion_chain": sampled["diffusion_chain"],
                "best_g": best_g,
                "best_k": best_k,
                "mode_index": torch.full((num_groups,), best_k, dtype=torch.long, device=self.device),
            }
        return rollouts

    def step_env_with_best(self, rollouts: dict) -> dict:
        best_actions = {}
        for agent_id in rollouts["agent_ids"]:
            best_actions[agent_id] = (
                rollouts[agent_id]["trajectory_all"][rollouts[agent_id]["best_g"], rollouts[agent_id]["best_k"]]
                .detach()
                .cpu()
                .numpy()
            )

        obs, _, terminated, truncated, _ = self.env.step(best_actions)
        self.env_step_counter += 1
        if (
            terminated.get("__all__", False)
            or truncated.get("__all__", False)
            or (self.max_env_steps_per_rollout > 0 and self.env_step_counter >= self.max_env_steps_per_rollout)
        ):
            obs = self.env.reset()
            self.env_step_counter = 0
        self.current_obs = obs
        return obs

    def compute_advantages(self, rollouts: dict) -> dict:
        advantages: dict = {}
        discount = torch.tensor(
            [self.advantage_gamma ** (self.scheduler.step_num - i - 1) for i in range(self.scheduler.step_num)],
            dtype=torch.float32,
            device=self.device,
        )
        for agent_id in rollouts["agent_ids"]:
            reward = rollouts[agent_id]["reward_per_anchor"]
            crash_mask = rollouts[agent_id]["crash_per_anchor"] | rollouts[agent_id]["out_per_anchor"]
            mean_r = reward.mean(dim=0, keepdim=True)
            std_r = reward.std(dim=0, unbiased=False, keepdim=True)
            normalized = (reward - mean_r) / (std_r + 1e-4)
            positive_mask = reward > mean_r
            normalized = normalized.clamp(min=0.0) * positive_mask.float()

            for mode_idx in range(normalized.shape[1]):
                valid_vals = normalized[:, mode_idx][~crash_mask[:, mode_idx]]
                if valid_vals.numel() == 0:
                    continue
                scale = valid_vals.std(unbiased=False)
                if torch.isfinite(scale) and float(scale.item()) > 1e-6:
                    normalized[:, mode_idx] = normalized[:, mode_idx] / scale

            advantage = normalized.unsqueeze(-1) * discount.unsqueeze(0).unsqueeze(0)
            advantage = torch.where(crash_mask.unsqueeze(-1), torch.full_like(advantage, -1.0), advantage)
            advantages[agent_id] = advantage
        return advantages

    def compute_rl_loss(self, rollouts: dict, advantages: dict) -> Tensor:
        losses = []
        self._cached_current_preds = {}
        for agent_id in rollouts["agent_ids"]:
            replay_all, model_outputs = self.scheduler.replay_with_log_prob(
                self.model,
                rollouts["batch"],
                agent_id,
                rollouts[agent_id]["diffusion_chain"],
                return_model_outputs=True,
            )
            self._cached_current_preds[agent_id] = model_outputs
            advantage = advantages[agent_id]
            per_token_loss = -torch.exp(replay_all - replay_all.detach()) * advantage
            mask_nz = per_token_loss != 0
            rl_loss = (per_token_loss * mask_nz).sum() / mask_nz.sum().clamp(min=1)
            losses.append(rl_loss)
        return torch.stack(losses).mean()

    def _get_beta_reg(self) -> float:
        total = int(self.config.get("total_steps", 500))
        beta_max = float(self.config.get("beta_reg_max", 1.0))
        beta_min = float(self.config.get("beta_reg_min", 0.1))
        warmup_frac = float(self.config.get("beta_reg_warmup_frac", 0.3))
        decay_frac = float(self.config.get("beta_reg_decay_frac", 0.4))
        warmup_end = int(total * warmup_frac)
        decay_end = int(total * (warmup_frac + decay_frac))

        if self.global_step < warmup_end:
            return beta_max
        if self.global_step < decay_end:
            progress = (self.global_step - warmup_end) / max(decay_end - warmup_end, 1)
            return beta_max - progress * (beta_max - beta_min)
        return beta_min

    def compute_ref_reg_loss(self, rollouts: dict) -> Tensor:
        losses = []
        discount = torch.tensor(
            [self.advantage_gamma ** (self.scheduler.step_num - i - 1) for i in range(self.scheduler.step_num)],
            dtype=torch.float32,
            device=self.device,
        )
        for agent_id in rollouts["agent_ids"]:
            chain = rollouts[agent_id]["diffusion_chain"]
            current_preds = self._cached_current_preds.get(agent_id)
            if current_preds is None:
                _, current_preds = self.scheduler.replay_with_log_prob(
                    self.model,
                    rollouts["batch"],
                    agent_id,
                    chain,
                    return_model_outputs=True,
                )
            ref_preds = self.scheduler.compute_ref_predictions(
                self.ref_model,
                rollouts["batch"],
                agent_id,
                chain,
            )
            step_losses = []
            for step_idx, (curr, ref) in enumerate(zip(current_preds, ref_preds)):
                step_losses.append(discount[step_idx] * F.mse_loss(curr, ref.detach(), reduction="mean"))
            losses.append(torch.stack(step_losses).sum())
        return torch.stack(losses).mean()

    def compute_team_advantages(self, rollouts: dict, joint_groups, team_rewards) -> dict:
        agent_ids = rollouts["agent_ids"]
        num_groups = rollouts[agent_ids[0]]["reward_per_anchor"].shape[0]
        num_anchors = rollouts[agent_ids[0]]["reward_per_anchor"].shape[1]
        step_num = self.scheduler.step_num

        rewards = torch.tensor(team_rewards, dtype=torch.float32, device=self.device)
        if rewards.numel() > 1:
            normalized = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
            normalized = normalized.clamp(min=0.0)
        else:
            normalized = torch.where(rewards > 0, torch.ones_like(rewards), torch.zeros_like(rewards))

        for group_idx, group in enumerate(joint_groups):
            any_crash = False
            for agent_id, (g_idx, k_idx) in group.items():
                if bool(rollouts[agent_id]["crash_per_anchor"][g_idx, k_idx].item()):
                    any_crash = True
                    break
            if any_crash:
                normalized[group_idx] = -1.0

        discount = torch.tensor(
            [self.advantage_gamma ** (step_num - i - 1) for i in range(step_num)],
            dtype=torch.float32,
            device=self.device,
        )
        team_advantages = {}
        for agent_id in agent_ids:
            adv = torch.zeros(num_groups, num_anchors, step_num, dtype=torch.float32, device=self.device)
            for group_idx, group in enumerate(joint_groups):
                if agent_id not in group:
                    continue
                g_idx, k_idx = group[agent_id]
                if float(normalized[group_idx].item()) == -1.0:
                    adv[g_idx, k_idx, :] = -1.0
                else:
                    adv[g_idx, k_idx, :] = normalized[group_idx] * discount
            team_advantages[agent_id] = adv
        return team_advantages

    def compute_combined_advantages(self, local_advantages: dict, team_advantages: dict) -> dict:
        lambda_local = float(self.config.get("lambda_local", 0.7))
        lambda_team = float(self.config.get("lambda_team", 0.3))
        combined = {}
        for agent_id, local in local_advantages.items():
            team = team_advantages.get(agent_id, torch.zeros_like(local))
            combined[agent_id] = lambda_local * local + lambda_team * team
        return combined

    def update(self, rollouts: dict, joint_groups=None, team_rewards=None) -> dict:
        """MultiAgentGRPOTrainer参数更新"""

        # 计算优势函数
        local_advantages = self.compute_advantages(rollouts)
        if joint_groups is not None and team_rewards is not None:
            team_advantages = self.compute_team_advantages(rollouts, joint_groups, team_rewards)
            advantages = self.compute_combined_advantages(local_advantages, team_advantages)
        else:
            advantages = local_advantages

        self.optimizer.zero_grad(set_to_none=True)

        # 计算RL损失和参考模型正则化损失，并加权求和得到总损失
        rl_loss = self.compute_rl_loss(rollouts, advantages)
        ref_reg_loss = self.compute_ref_reg_loss(rollouts)
        beta_reg = self._get_beta_reg()
        loss = rl_loss + beta_reg * ref_reg_loss
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # 计算梯度范数，并执行优化器更新
        grad_norm_sq = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm_sq += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norm = grad_norm_sq ** 0.5
        self.optimizer.step()
        self.global_step += 1

        # 评估当前策略的KL散度、平均奖励、队形误差和碰撞率等指标
        with torch.no_grad():
            kl_terms = []
            mean_rewards = []
            formation_terms = []
            collision_terms = []
            for agent_id in rollouts["agent_ids"]:
                ref_replay = self.scheduler.replay_with_log_prob(
                    self.ref_model,
                    rollouts["batch"],
                    agent_id,
                    rollouts[agent_id]["diffusion_chain"],
                )
                new_replay = self.scheduler.replay_with_log_prob(
                    self.model,
                    rollouts["batch"],
                    agent_id,
                    rollouts[agent_id]["diffusion_chain"],
                )
                kl_terms.append(self.scheduler.compute_kl(new_replay, ref_replay))
                mean_rewards.append(rollouts[agent_id]["reward_per_anchor"].mean())
                formation_terms.append(rollouts[agent_id]["formation_per_anchor"].mean())
                collision_terms.append(
                    (rollouts[agent_id]["crash_per_anchor"] | rollouts[agent_id]["out_per_anchor"]).float().mean()
                )
            kl = torch.stack(kl_terms).mean()
            mean_reward = torch.stack(mean_rewards).mean()
            formation_error = torch.stack(formation_terms).mean()
            collision_rate = torch.stack(collision_terms).mean()

        team_reward_mean = float(sum(team_rewards) / max(len(team_rewards), 1)) if team_rewards is not None else 0.0
        return {
            "loss": float(loss.detach().item()),
            "rl_loss": float(rl_loss.detach().item()),
            "ref_reg_loss": float(ref_reg_loss.detach().item()),
            "kl": float(kl.detach().item()),
            "mean_reward": float(mean_reward.detach().item()),
            "formation_error": float(formation_error.detach().item()),
            "collision_rate": float(collision_rate.detach().item()),
            "beta_reg": float(beta_reg),
            "grad_norm": float(grad_norm),
            "team_reward_mean": float(team_reward_mean),
            "lambda_local": float(self.config.get("lambda_local", 0.7)),
            "lambda_team": float(self.config.get("lambda_team", 0.3)),
        }
