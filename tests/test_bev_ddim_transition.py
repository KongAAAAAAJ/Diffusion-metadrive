from __future__ import annotations

import pytest
import torch

from models.bev_planner import (
    DDIMNoiseBundle,
    DDIMPathConfig,
    DDIMTransitionError,
    DEFAULT_DDIM_PATH,
    StandardGaussianDDIM,
)


def test_ddim_path_is_exactly_four_decoder_calls_and_three_pg_steps() -> None:
    assert DEFAULT_DDIM_PATH.timesteps == (8, 5, 3, 0)
    assert DEFAULT_DDIM_PATH.previous_timesteps == (5, 3, 0, -1)
    assert DEFAULT_DDIM_PATH.transitions() == (
        (8, 5),
        (5, 3),
        (3, 0),
        (0, -1),
    )
    assert DEFAULT_DDIM_PATH.stochastic_transition_count == 3
    with pytest.raises(DDIMTransitionError, match="frozen"):
        DDIMPathConfig(timesteps=(8, 0), previous_timesteps=(0, -1))


def test_explicit_transition_noise_and_replay_have_identical_log_probability() -> None:
    transition = StandardGaussianDDIM()
    shape = (2, 10, 8, 2)
    generator = torch.Generator().manual_seed(17)
    bundle = DDIMNoiseBundle.sample(
        shape,
        device=torch.device("cpu"),
        generator=generator,
    )
    sample = transition.add_noise(
        torch.zeros(shape),
        bundle.initial_noise,
        torch.full((2,), 8, dtype=torch.int64),
    )
    log_probs = []
    transition_index = 0
    for timestep, previous_timestep in DEFAULT_DDIM_PATH.transitions():
        noise = None
        if previous_timestep >= 0:
            noise = bundle.transition_noises[transition_index]
            transition_index += 1
        step = transition.step(
            model_output=torch.zeros_like(sample),
            timestep=timestep,
            previous_timestep=previous_timestep,
            sample=sample,
            eta=1.0,
            noise=noise,
        )
        replay = transition.step(
            model_output=torch.zeros_like(sample),
            timestep=timestep,
            previous_timestep=previous_timestep,
            sample=sample,
            eta=1.0,
            prev_sample=step.prev_sample,
        )
        if previous_timestep >= 0:
            assert step.log_prob is not None and replay.log_prob is not None
            assert step.log_prob.shape == (2, 10)
            assert torch.isfinite(step.log_prob).all()
            torch.testing.assert_close(step.log_prob, replay.log_prob)
            log_probs.append(step.log_prob)
        else:
            assert step.log_prob is None and replay.log_prob is None
        sample = step.prev_sample
    assert len(log_probs) == 3
