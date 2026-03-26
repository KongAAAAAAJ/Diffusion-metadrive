from __future__ import annotations

import math
from typing import Dict, List, Mapping

import numpy as np
import torch
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import Tensor


class DDIMSchedulerWithLogProb(DDIMScheduler):
    """DDIMScheduler extended to return per-step Gaussian log_prob.

    Mirrors ``DDIMScheduler_with_logprob`` from the DiffusionDriveV2
    reference (diffusiondrivev2_model_rl.py lines 540-676).

    Key design points
    -----------------
    * Multiplicative noise is only added when ``eta > 0``.  When eta == 0
      the step is fully deterministic (prev_sample == prev_sample_mean).
    * ``prev_sample.detach()`` ensures gradients flow only through
      ``prev_sample_mean`` (the model's prediction), not through the noise.
    * log_prob uses sigma = clip(std_dev_t, min=0.1) — same asymmetry as
      the reference (generation uses min=0.04, log_prob uses min=0.1).
    """

    def step(
        self,
        model_output: Tensor,
        timestep: int | Tensor,
        sample: Tensor,
        eta: float = 0.0,
        prev_sample: Tensor | None = None,
        use_clipped_model_output: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.num_inference_steps is None:
            raise ValueError("set_timesteps() must be called before step().")

        if torch.is_tensor(timestep):
            timestep_value = int(timestep.reshape(-1)[0].item())
        else:
            timestep_value = int(timestep)

        prev_timestep = timestep_value - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep_value].to(sample.device, sample.dtype)
        if prev_timestep >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device, sample.dtype)
        else:
            alpha_prod_t_prev = self.final_alpha_cumprod.to(sample.device, sample.dtype)
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
        else:
            raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        variance = self._get_variance(timestep_value, prev_timestep).to(sample.device, sample.dtype)
        std_dev_t = (eta * variance.sqrt()).clamp(min=1e-10)

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).clamp(min=0).sqrt() * pred_epsilon
        prev_sample_mean = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        if prev_sample is None:
            # FIX (Problem B): only apply multiplicative noise when eta > 0.
            # When eta == 0, std_dev_t_mul = 0 so variance_noise_mul collapses to
            # 1.0 and prev_sample = prev_sample_mean (fully deterministic).
            if eta > 0:
                std_dev_t_mul = std_dev_t.clamp(min=0.04)
            else:
                std_dev_t_mul = torch.zeros_like(std_dev_t)

            horizon_noise = randn_tensor(
                (model_output.shape[0], model_output.shape[1], 1, 1),
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            lateral_noise = randn_tensor(
                (model_output.shape[0], model_output.shape[1], 1, 1),
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance_noise_mul = torch.cat(
                [horizon_noise * std_dev_t_mul + 1.0, lateral_noise * std_dev_t_mul + 1.0], dim=-1
            ).repeat(1, 1, model_output.shape[2], 1)

            # Additive noise coefficient is always 0 (matches reference std_dev_t_add = 0).
            prev_sample = prev_sample_mean * variance_noise_mul

        # log_prob: Gaussian with sigma = clip(std_dev_t, min=0.1).
        # prev_sample.detach() → gradient flows only through prev_sample_mean.
        std_dev_for_log = std_dev_t.clamp(min=0.1)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_for_log ** 2))
            - torch.log(std_dev_for_log)
            - math.log(math.sqrt(2.0 * math.pi))
        ).sum(dim=(-2, -1))

        return prev_sample.type(sample.dtype), log_prob, prev_sample_mean.type(sample.dtype)


class DiffusionRLScheduler:
    """Orchestrates DDIM sampling with log_prob for RL training.

    Works with ``PlatoonDiffusionPlanner`` via its two RL helper methods:
      - ``planner.extract_rl_context(batch)``  → per-agent backbone context
      - ``planner.predict_denoised_traj(noisy_norm, timestep, context)``
        → predicted denoised trajectory (normalized xy, no grad needed)

    Two-phase forward (mirrors reference forward_train_rl / get_rlloss):
      1. ``sample_with_log_prob``  – no_grad, collects chain + log_probs
      2. ``replay_with_log_prob``  – with grad, replays chain → new log_probs
    """

    def __init__(self, ddim_config: dict):
        config = dict(ddim_config or {})
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.step_num = int(config.get("num_inference_steps", config.get("ddim_steps", 10)))
        self.eta = float(config.get("eta", config.get("ddim_eta", 0.5)))
        self.prediction_type = str(config.get("prediction_type", "sample"))
        # FIX (Problem A): truncated noise starting timestep (reference uses 8)
        self.trunc_timestep = int(config.get("trunc_timestep", 8))
        # FIX (Problem E): roll_step_range for manual timestep schedule (reference uses 20)
        self.roll_step_range = int(config.get("roll_step_range", 20))
        self.scheduler = DDIMSchedulerWithLogProb(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type=self.prediction_type,
            clip_sample=False,
        )

    def _roll_timesteps(self) -> np.ndarray:
        """Compute manual roll timesteps matching reference forward_train_rl.

        Reference (lines 801-807):
            step_ratio = 20 / step_num
            roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1]
        Result for step_num=10, roll_step_range=20: [18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
        """
        step_ratio = self.roll_step_range / self.step_num
        return (np.arange(0, self.step_num) * step_ratio).round()[::-1].copy().astype(np.int64)

    # ------------------------------------------------------------------
    # Phase 1 – sampling (no_grad)
    # ------------------------------------------------------------------

    def sample_with_log_prob(self, model, batch: dict, num_groups: int = 4) -> dict:
        """Sample ``num_groups`` trajectory candidates per agent with log_prob.

        Parameters
        ----------
        model : PlatoonDiffusionPlanner
        batch : {agent_id: {"camera", "lidar", "status", "formation_relation_state"}}
        num_groups : number of independent trajectory samples per agent

        Returns
        -------
        {agent_id: {
            "trajectory": Tensor [num_groups, ego_fut_mode, 8, 3],
            "log_prob":   Tensor [num_groups, ego_fut_mode, step_num],
            "diffusion_chain": list[Tensor],  # length = step_num + 1
        }}
        """
        # 1. Extract backbone context once (no_grad).
        with torch.no_grad():
            contexts, agent_ids = model.extract_rl_context(batch)

        device = next(iter(next(iter(batch.values())).values())).device
        # FIX (Problem E): set_timesteps(1000) for fine-grained alpha lookup,
        # then use manual roll_timesteps for the denoising loop.
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)
        roll_timesteps = self._roll_timesteps()

        th = model.model._trajectory_head
        ego_fut_mode = th.ego_fut_mode

        results: dict = {}
        for agent_id in agent_ids:
            ctx = contexts[agent_id]  # batch_size = 1

            # 2. FIX (Problem A): truncated noise initialisation from plan_anchor.
            # plan_anchor: [ego_fut_mode, 8, 3] → take xy, normalise, add small noise
            # at trunc_timestep=8 (same as reference forward_test, line 534-535).
            plan_xy = th.plan_anchor[..., :2].unsqueeze(0).repeat(num_groups, 1, 1, 1)  # [G, M, 8, 2]
            anchor_norm = th.norm_odo(
                torch.cat([plan_xy, torch.zeros_like(plan_xy[..., :1])], dim=-1)
            )[..., :2]  # [G, M, 8, 2] (normalised, xy only)

            noise = torch.randn_like(anchor_norm)
            trunc_t = torch.full((num_groups,), self.trunc_timestep, device=device, dtype=torch.long)
            sample = self.scheduler.add_noise(
                original_samples=anchor_norm, noise=noise, timesteps=trunc_t
            )  # [G, M, 8, 2]

            chain: List[Tensor] = [sample.detach().clone()]
            log_probs: List[Tensor] = []

            # 3. DDIM denoising loop.
            # Repeat context along batch dim to match num_groups.
            ctx_g = {
                k: (v.repeat(num_groups, *([1] * (v.ndim - 1))) if isinstance(v, Tensor) else v)
                for k, v in ctx.items()
            }

            with torch.no_grad():
                for ts in roll_timesteps:
                    ts_int = int(ts)
                    ts_batch = torch.full(
                        (num_groups,), ts_int, device=device, dtype=torch.long
                    )
                    model_out = model.predict_denoised_traj(sample, ts_batch, ctx_g)
                    prev_sample, log_prob, _ = self.scheduler.step(
                        model_output=model_out,
                        timestep=ts_int,
                        sample=sample,
                        eta=self.eta,
                    )
                    log_probs.append(log_prob)    # [G, M]
                    sample = prev_sample
                    chain.append(sample.detach().clone())

            # 4. Denormalise to physical coordinates + yaw via bezier.
            sample_phys = th.denorm_odo(
                torch.cat([sample, torch.zeros_like(sample[..., :1])], dim=-1)
            )[..., :2]  # [G, M, 8, 2]

            # Add bezier yaw if available (model has the method).
            if hasattr(th, "bezier_xyyaw"):
                trajectory = th.bezier_xyyaw(sample_phys)  # [G, M, 8, 3]
            else:
                trajectory = torch.cat(
                    [sample_phys, torch.zeros_like(sample_phys[..., :1])], dim=-1
                )  # fallback: zero yaw

            results[agent_id] = {
                "trajectory": trajectory,
                "log_prob": torch.stack(log_probs, dim=-1),  # [G, M, step_num]
                "diffusion_chain": chain,
            }
        return results

    # ------------------------------------------------------------------
    # Phase 2 – chain replay (with grad → log_prob.backward() works)
    # ------------------------------------------------------------------

    def replay_with_log_prob(
        self,
        model,
        batch: dict,
        agent_id: str,
        diffusion_chain: List[Tensor],
        return_model_outputs: bool = False,
    ):
        """Replay the stored diffusion chain with gradients.

        Re-extracts backbone context (or caller can cache it).  Passes the
        stored intermediate samples as ``prev_sample`` to the scheduler so
        the exact same samples are used; only log_prob is re-computed with
        the current model parameters (gradient flows through prev_sample_mean).

        Parameters
        ----------
        return_model_outputs : bool
            If True, also return a list of model predictions (one per step,
            each [G, M, 8, 2] normalized xy) alongside the log_prob tensor.
            Used by ``compute_il_loss`` for multi-step IL supervision.

        Returns
        -------
        log_prob : Tensor [G, ego_fut_mode, step_num]
        model_outputs (optional) : list[Tensor]  # length = step_num
        """
        if len(diffusion_chain) != self.step_num + 1:
            raise ValueError(
                f"diffusion_chain length {len(diffusion_chain)} != step_num+1={self.step_num + 1}"
            )

        contexts, _ = model.extract_rl_context(batch)
        ctx = contexts[agent_id]
        num_groups = diffusion_chain[0].shape[0]

        device = diffusion_chain[0].device
        # FIX (Problem E): same timestep schedule as sample_with_log_prob.
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)
        roll_timesteps = self._roll_timesteps()

        ctx_g = {
            k: (v.repeat(num_groups, *([1] * (v.ndim - 1))) if isinstance(v, Tensor) else v)
            for k, v in ctx.items()
        }

        replayed: List[Tensor] = []
        model_outputs: List[Tensor] = []
        for idx, ts in enumerate(roll_timesteps):
            ts_int = int(ts)
            sample = diffusion_chain[idx]
            next_sample = diffusion_chain[idx + 1]
            ts_batch = torch.full((num_groups,), ts_int, device=device, dtype=torch.long)
            model_out = model.predict_denoised_traj(sample, ts_batch, ctx_g)
            if return_model_outputs:
                model_outputs.append(model_out)
            _, log_prob, _ = self.scheduler.step(
                model_output=model_out,
                timestep=ts_int,
                sample=sample,
                eta=self.eta,
                prev_sample=next_sample,  # reuse stored sample; only log_prob uses grad
            )
            replayed.append(log_prob)
        log_prob_stack = torch.stack(replayed, dim=-1)  # [G, M, step_num]
        if return_model_outputs:
            return log_prob_stack, model_outputs
        return log_prob_stack

    def compute_ref_predictions(
        self,
        ref_model,
        batch: dict,
        agent_id: str,
        diffusion_chain: List[Tensor],
    ) -> List[Tensor]:
        """Replay the same diffusion chain through the frozen reference model."""
        if len(diffusion_chain) != self.step_num + 1:
            raise ValueError(
                f"diffusion_chain length {len(diffusion_chain)} != step_num+1={self.step_num + 1}"
            )

        with torch.no_grad():
            contexts, _ = ref_model.extract_rl_context(batch)
            ctx = contexts[agent_id]
            num_groups = diffusion_chain[0].shape[0]
            device = diffusion_chain[0].device
            self.scheduler.set_timesteps(self.num_train_timesteps, device=device)
            roll_timesteps = self._roll_timesteps()
            ctx_g = {
                key: (value.repeat(num_groups, *([1] * (value.ndim - 1))) if isinstance(value, Tensor) else value)
                for key, value in ctx.items()
            }

            ref_outputs: List[Tensor] = []
            for idx, ts in enumerate(roll_timesteps):
                del idx
                ts_int = int(ts)
                sample = diffusion_chain[len(ref_outputs)]
                ts_batch = torch.full((num_groups,), ts_int, device=device, dtype=torch.long)
                ref_outputs.append(ref_model.predict_denoised_traj(sample, ts_batch, ctx_g))
        return ref_outputs

    # ------------------------------------------------------------------
    # KL monitoring (not added to loss, used for early stopping / logging)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_kl(log_prob: Tensor, ref_log_prob: Tensor) -> Tensor:
        """First-order KL estimate: E_p[log p - log q] ≈ mean(log_prob - ref_log_prob).

        FIX (Problem D): replaced MSE(log_prob - ref_log_prob) with the
        correct importance-ratio-based KL approximation.  This is always
        finite; clamp to zero to keep the logged monitor non-negative.
        """
        return (log_prob - ref_log_prob).mean().clamp(min=0.0)
