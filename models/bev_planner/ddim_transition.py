"""Shared explicit DDIM path and Gaussian transition semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from diffusers.schedulers import DDIMScheduler
from torch import Tensor


class DDIMTransitionError(RuntimeError):
    """Raised when the frozen DDIM path contract is violated."""


@dataclass(frozen=True)
class DDIMPathConfig:
    """The single runtime DDIM path used by inference and GRPO."""

    timesteps: tuple[int, ...] = (8, 5, 3, 0)
    previous_timesteps: tuple[int, ...] = (5, 3, 0, -1)
    eta: float = 1.0

    def __post_init__(self) -> None:
        if self.timesteps != (8, 5, 3, 0):
            raise DDIMTransitionError("DDIM timesteps are frozen to (8,5,3,0)")
        if self.previous_timesteps != (5, 3, 0, -1):
            raise DDIMTransitionError(
                "DDIM previous timesteps are frozen to (5,3,0,-1)"
            )
        if not math.isclose(float(self.eta), 1.0):
            raise DDIMTransitionError("DDIM eta is frozen to 1.0")

    @property
    def initial_timestep(self) -> int:
        return self.timesteps[0]

    @property
    def stochastic_transition_count(self) -> int:
        return len(self.timesteps) - 1

    def transitions(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.timesteps, self.previous_timesteps))

    def as_dict(self) -> dict[str, object]:
        return {
            "timesteps": list(self.timesteps),
            "previous_timesteps": list(self.previous_timesteps),
            "eta": float(self.eta),
            "decoder_calls": len(self.timesteps),
            "stochastic_log_probability_transitions": (
                self.stochastic_transition_count
            ),
        }


DEFAULT_DDIM_PATH: Final[DDIMPathConfig] = DDIMPathConfig()


@dataclass(frozen=True)
class DDIMNoiseBundle:
    """Initial and transition noise for one reproducible DDIM rollout."""

    initial_noise: Tensor
    transition_noises: tuple[Tensor, ...]

    @classmethod
    def sample(
        cls,
        shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        generator: torch.Generator,
        path: DDIMPathConfig = DEFAULT_DDIM_PATH,
    ) -> "DDIMNoiseBundle":
        if not isinstance(generator, torch.Generator):
            raise DDIMTransitionError("DDIM noise sampling requires a generator")
        initial = torch.randn(
            shape, device=device, dtype=torch.float32, generator=generator
        )
        transitions = tuple(
            torch.randn(
                shape, device=device, dtype=torch.float32, generator=generator
            )
            for _ in range(path.stochastic_transition_count)
        )
        return cls(initial_noise=initial, transition_noises=transitions)

    def validate(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        path: DDIMPathConfig = DEFAULT_DDIM_PATH,
    ) -> None:
        expected = tuple(shape)
        requested_device = torch.device(device)
        tensors = (self.initial_noise, *self.transition_noises)
        if len(self.transition_noises) != path.stochastic_transition_count:
            raise DDIMTransitionError(
                "DDIM noise bundle transition count does not match the path"
            )
        for value in tensors:
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.float32
                or tuple(value.shape) != expected
                or value.device.type != requested_device.type
                or (
                    requested_device.index is not None
                    and value.device.index != requested_device.index
                )
            ):
                raise DDIMTransitionError(
                    "DDIM noise bundle tensors must be colocated float32 with "
                    "the rollout shape"
                )
            if not bool(torch.isfinite(value).all()):
                raise DDIMTransitionError("DDIM noise bundle must be finite")


@dataclass(frozen=True)
class GaussianDDIMStep:
    prev_sample: Tensor
    mean: Tensor
    std: Tensor
    log_prob: Tensor | None


class StandardGaussianDDIM:
    """Explicit DDIM transition with exact Gaussian log-probability."""

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

    def add_noise(
        self, original: Tensor, noise: Tensor, timesteps: Tensor
    ) -> Tensor:
        return self.scheduler.add_noise(original, noise, timesteps)

    def step(
        self,
        *,
        model_output: Tensor,
        timestep: int,
        previous_timestep: int,
        sample: Tensor,
        eta: float,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        prev_sample: Tensor | None = None,
    ) -> GaussianDDIMStep:
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise DDIMTransitionError("DDIM timestep must be an integer")
        if timestep < 0 or timestep >= self.scheduler.config.num_train_timesteps:
            raise DDIMTransitionError("DDIM timestep is outside the scheduler")
        if (
            isinstance(previous_timestep, bool)
            or not isinstance(previous_timestep, int)
            or previous_timestep < -1
            or previous_timestep >= timestep
        ):
            raise DDIMTransitionError(
                "DDIM previous_timestep must be an integer in [-1,timestep)"
            )
        if timestep == 0 and previous_timestep != -1:
            raise DDIMTransitionError("the deterministic t=0 step must end at -1")
        if timestep > 0 and previous_timestep < 0:
            raise DDIMTransitionError("only t=0 may transition to -1")
        if sample.shape != model_output.shape:
            raise DDIMTransitionError("DDIM sample and model_output shapes must match")
        if sample.dtype != torch.float32 or model_output.dtype != torch.float32:
            raise DDIMTransitionError("DDIM probability tensors must use float32")
        if sample.device != model_output.device:
            raise DDIMTransitionError("DDIM inputs must use the same device")
        if not bool(torch.isfinite(sample).all()) or not bool(
            torch.isfinite(model_output).all()
        ):
            raise DDIMTransitionError("DDIM inputs must be finite")
        if not math.isfinite(float(eta)) or float(eta) < 0.0:
            raise DDIMTransitionError("DDIM eta must be non-negative and finite")
        if sum(value is not None for value in (generator, noise, prev_sample)) > 1:
            raise DDIMTransitionError(
                "provide at most one of generator, noise, or replayed prev_sample"
            )

        device = sample.device
        dtype = sample.dtype
        alpha_t = self.scheduler.alphas_cumprod[timestep].to(device, dtype)
        alpha_previous = (
            self.scheduler.alphas_cumprod[previous_timestep].to(device, dtype)
            if previous_timestep >= 0
            else self.scheduler.final_alpha_cumprod.to(device, dtype)
        )
        beta_t = 1.0 - alpha_t
        prediction = model_output.clamp(
            -float(self.scheduler.config.clip_sample_range),
            float(self.scheduler.config.clip_sample_range),
        )
        epsilon = (sample - alpha_t.sqrt() * prediction) / beta_t.sqrt().clamp_min(
            torch.finfo(dtype).eps
        )
        variance = (
            (1.0 - alpha_previous)
            / (1.0 - alpha_t).clamp_min(torch.finfo(dtype).eps)
            * (1.0 - alpha_t / alpha_previous)
        ).clamp_min(0.0)
        std = float(eta) * variance.sqrt()
        direction_scale = (1.0 - alpha_previous - std.square()).clamp_min(0.0)
        mean = alpha_previous.sqrt() * prediction + direction_scale.sqrt() * epsilon

        stochastic = bool(float(std.detach().cpu()) > 0.0)
        if noise is not None:
            if (
                noise.shape != sample.shape
                or noise.dtype != dtype
                or noise.device != device
                or not bool(torch.isfinite(noise).all())
            ):
                raise DDIMTransitionError(
                    "explicit DDIM noise must match the sample"
                )
            sampled = mean + std * noise if stochastic else mean
        elif prev_sample is not None:
            if (
                prev_sample.shape != sample.shape
                or prev_sample.dtype != dtype
                or prev_sample.device != device
                or not bool(torch.isfinite(prev_sample).all())
            ):
                raise DDIMTransitionError(
                    "replayed DDIM prev_sample must match the sample"
                )
            sampled = prev_sample
        elif stochastic:
            sampled = mean + std * torch.randn(
                sample.shape, dtype=dtype, device=device, generator=generator
            )
        else:
            sampled = mean

        log_prob = None
        if stochastic:
            elementwise = (
                -0.5 * ((sampled.detach() - mean) / std).square()
                - torch.log(std)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_prob = elementwise.sum(dim=(-2, -1))
        return GaussianDDIMStep(
            prev_sample=sampled,
            mean=mean,
            std=std,
            log_prob=log_prob,
        )


__all__ = [
    "DDIMNoiseBundle",
    "DDIMPathConfig",
    "DDIMTransitionError",
    "DEFAULT_DDIM_PATH",
    "GaussianDDIMStep",
    "StandardGaussianDDIM",
]
