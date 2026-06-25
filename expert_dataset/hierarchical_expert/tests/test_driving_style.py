from __future__ import annotations

import numpy as np

from expert_dataset.hierarchical_expert.driving_style import (
    DrivingStyleProfile,
    STYLE_PRESETS,
    StyleSampler,
)


def test_default_profile_constructs_with_numeric_fields():
    profile = DrivingStyleProfile()

    assert isinstance(profile.aggression, float)
    assert isinstance(profile.time_headway, float)
    assert isinstance(profile.max_accel, float)
    assert isinstance(profile.lane_change_threshold, float)
    assert profile.lane_change_threshold == 0.2
    assert profile.lane_change_cooldown == 80
    assert profile.time_headway == 1.0
    assert profile.desired_lateral_speed == 0.7


def test_style_presets_cover_expected_profiles():
    assert set(STYLE_PRESETS.keys()) == {"conservative", "normal", "aggressive"}
    assert all(isinstance(profile, DrivingStyleProfile) for profile in STYLE_PRESETS.values())
    assert STYLE_PRESETS["conservative"].lane_change_threshold == 0.45
    assert STYLE_PRESETS["conservative"].lane_change_cooldown == 100
    assert STYLE_PRESETS["conservative"].desired_lateral_speed == 0.5
    assert STYLE_PRESETS["aggressive"].lane_change_threshold == 0.08
    assert STYLE_PRESETS["aggressive"].lane_change_cooldown == 50
    assert STYLE_PRESETS["aggressive"].desired_lateral_speed == 1.0


def test_style_sampler_is_stateful_but_reproducible_for_same_seed():
    sampler = StyleSampler(seed=42)
    first = sampler.sample()
    second = sampler.sample()
    fresh_first = StyleSampler(seed=42).sample()

    assert isinstance(first, DrivingStyleProfile)
    assert 0.0 <= first.aggression <= 1.0
    assert first != second
    assert fresh_first == first


def test_style_sampler_beta_distribution_is_conservative_and_positive():
    sampler = StyleSampler(seed=7)
    samples = [sampler.sample() for _ in range(1000)]
    aggressions = np.asarray([sample.aggression for sample in samples], dtype=np.float64)

    assert float(np.mean(aggressions)) < 0.4
    for sample in samples:
        assert sample.desired_speed_ratio > 0.0
        assert sample.time_headway > 0.0
        assert sample.max_accel > 0.0
        assert sample.comfortable_decel > 0.0
        assert sample.min_jam_distance > 0.0
        assert sample.velocity_exponent > 0.0
        assert sample.min_front_ttc > 0.0
        assert sample.min_rear_ttc > 0.0
        assert sample.reaction_time > 0.0
        assert sample.desired_lateral_speed > 0.0


def test_style_sampler_interpolates_lane_change_cooldown_with_new_range():
    sampler = StyleSampler(seed=11)
    samples = [sampler.sample() for _ in range(200)]
    cooldowns = np.asarray([sample.lane_change_cooldown for sample in samples], dtype=np.int64)

    assert np.min(cooldowns) >= STYLE_PRESETS["aggressive"].lane_change_cooldown
    assert np.max(cooldowns) <= STYLE_PRESETS["conservative"].lane_change_cooldown
    assert np.any(cooldowns > DrivingStyleProfile().lane_change_cooldown)
