from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=True)
class DrivingStyleProfile:
    aggression: float = 0.5
    desired_speed_ratio: float = 1.0
    time_headway: float = 1.0
    max_accel: float = 1.5
    comfortable_decel: float = 2.0
    min_jam_distance: float = 7.0
    velocity_exponent: float = 4.0
    politeness: float = 0.2
    lane_change_threshold: float = 0.2
    min_front_ttc: float = 3.0
    min_rear_ttc: float = 3.5
    reaction_time: float = 0.5
    desired_lateral_speed: float = 0.7
    route_urgency_weight: float = 2.0
    lane_change_cooldown: int = 80


STYLE_PRESETS = {
    "conservative": DrivingStyleProfile(
        aggression=0.2,
        desired_speed_ratio=0.9,
        time_headway=2.0,
        max_accel=1.0,
        comfortable_decel=1.5,
        min_jam_distance=10.0,
        velocity_exponent=6.0,
        politeness=0.5,
        lane_change_threshold=0.45,
        min_front_ttc=4.0,
        min_rear_ttc=5.0,
        reaction_time=0.7,
        desired_lateral_speed=0.5,
        lane_change_cooldown=100,
    ),
    "normal": DrivingStyleProfile(),
    "aggressive": DrivingStyleProfile(
        aggression=0.85,
        desired_speed_ratio=1.1,
        time_headway=0.8,
        max_accel=2.5,
        comfortable_decel=3.0,
        min_jam_distance=5.0,
        velocity_exponent=2.0,
        politeness=0.0,
        lane_change_threshold=0.08,
        min_front_ttc=2.0,
        min_rear_ttc=2.5,
        reaction_time=0.3,
        desired_lateral_speed=1.0,
        lane_change_cooldown=50,
    ),
}


class StyleSampler:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)

    def sample(self) -> DrivingStyleProfile:
        aggression = float(np.clip(self._rng.beta(2.0, 5.0), 0.0, 1.0))
        conservative = STYLE_PRESETS["conservative"]
        aggressive = STYLE_PRESETS["aggressive"]
        noisy = {}
        for field_name in DrivingStyleProfile.__dataclass_fields__:
            if field_name == "aggression":
                continue
            low = getattr(conservative, field_name)
            high = getattr(aggressive, field_name)
            base = low + (high - low) * aggression
            if field_name == "lane_change_cooldown":
                noisy[field_name] = base
                continue
            value = float(base * self._rng.normal(loc=1.0, scale=0.1))
            noisy[field_name] = max(value, 1e-3)

        cooldown = int(round(noisy.pop("lane_change_cooldown", DrivingStyleProfile().lane_change_cooldown)))
        return DrivingStyleProfile(
            aggression=aggression,
            lane_change_cooldown=max(cooldown, 1),
            **noisy,
        )
