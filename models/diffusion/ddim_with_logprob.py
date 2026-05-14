"""DDIM scheduler step with transition log-probability.

This mirrors the RL scheduler used by DiffusionDriveV2 while keeping the
implementation local to this project.  It returns the sampled previous state,
the log-probability of that transition, and the deterministic DDIM mean.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor


class DDIMSchedulerWithLogProb(DDIMScheduler):
    """DDIM scheduler that exposes log-prob for sampled transitions."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        variance_noise: Optional[torch.Tensor] = None,
        prev_sample: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del return_dict
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is None. Call set_timesteps() before step().")

        timestep_int = self._as_scalar_timestep(timestep)
        prev_timestep = timestep_int - self.config.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep_int].to(device=sample.device, dtype=sample.dtype)
        if prev_timestep >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(device=sample.device, dtype=sample.dtype)
        else:
            alpha_prod_t_prev = self.final_alpha_cumprod.to(device=sample.device, dtype=sample.dtype)
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
                -self.config.clip_sample_range,
                self.config.clip_sample_range,
            )

        variance = self._get_variance(timestep_int, prev_timestep).to(device=sample.device, dtype=sample.dtype)
        std_dev_t = (float(eta) * variance.sqrt()).clamp(min=1e-10)

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).clamp(min=0).sqrt() * pred_epsilon
        prev_sample_mean = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        if prev_sample is None:
            prev_sample = self._sample_prev_state(
                prev_sample_mean=prev_sample_mean,
                model_output=model_output,
                std_dev_t=std_dev_t,
                eta=float(eta),
                generator=generator,
                variance_noise=variance_noise,
            )

        log_std = std_dev_t.clamp(min=0.1)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (log_std**2))
            - torch.log(log_std)
            - torch.log(torch.sqrt(torch.as_tensor(2 * math.pi, device=sample.device, dtype=sample.dtype)))
        )
        log_prob = log_prob.sum(dim=(-2, -1))
        return prev_sample.to(dtype=sample.dtype), log_prob, prev_sample_mean.to(dtype=sample.dtype)

    @staticmethod
    def _as_scalar_timestep(timestep: Union[int, torch.Tensor]) -> int:
        if torch.is_tensor(timestep):
            if timestep.numel() != 1:
                flat = timestep.reshape(-1)
                if not torch.all(flat == flat[0]):
                    raise ValueError("DDIMSchedulerWithLogProb expects one shared timestep per step.")
                return int(flat[0].item())
            return int(timestep.item())
        return int(timestep)

    @staticmethod
    def _sample_prev_state(
        *,
        prev_sample_mean: torch.Tensor,
        model_output: torch.Tensor,
        std_dev_t: torch.Tensor,
        eta: float,
        generator: Optional[torch.Generator],
        variance_noise: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if variance_noise is not None and generator is not None:
            raise ValueError("Cannot pass both variance_noise and generator.")

        if eta > 0:
            std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
            std_dev_t_add = torch.tensor(0.0, device=std_dev_t.device, dtype=std_dev_t.dtype)
        else:
            std_dev_t_mul = torch.tensor(0.0, device=std_dev_t.device, dtype=std_dev_t.dtype)
            std_dev_t_add = torch.tensor(0.0, device=std_dev_t.device, dtype=std_dev_t.dtype)

        if variance_noise is None:
            variance_noise_horizon = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            ) * std_dev_t_mul + 1.0
            variance_noise_vert = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            ) * std_dev_t_mul + 1.0
            variance_noise_mul = torch.cat((variance_noise_horizon, variance_noise_vert), dim=-1)
        else:
            variance_noise_mul = variance_noise

        variance_noise_mul = variance_noise_mul.repeat(1, 1, model_output.shape[2], 1)

        variance_noise_x = randn_tensor(
            [model_output.shape[0], model_output.shape[1], 1, 1],
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        variance_noise_y = randn_tensor(
            [model_output.shape[0], model_output.shape[1], 1, 1],
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        variance_noise_add = torch.cat((variance_noise_x, variance_noise_y), dim=-1)
        variance_noise_add = variance_noise_add.repeat(1, 1, model_output.shape[2], 1)
        return prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add
