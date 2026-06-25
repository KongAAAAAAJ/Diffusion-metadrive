from __future__ import annotations

from metadrive.manager.traffic_manager import PGTrafficManager


class CustomTrafficManager(PGTrafficManager):
    """Traffic manager that applies a configurable speed cap to background IDM policies."""

    CUSTOM_TARGET_SPEED = None

    def __init__(self):
        super().__init__()
        self._custom_target_speed = self.CUSTOM_TARGET_SPEED
        self._base_normal_speeds = {}

    def set_target_speed(self, speed_kmh: float | None) -> None:
        self._custom_target_speed = None if speed_kmh is None else float(speed_kmh)

    def before_step(self):
        for vehicle in self._traffic_vehicles:
            self._apply_speed_cap(vehicle)
        return super().before_step()

    def before_reset(self) -> None:
        super().before_reset()
        configured_target_speed = self.engine.global_config.get(
            "traffic_target_speed",
            self.CUSTOM_TARGET_SPEED,
        )
        if isinstance(configured_target_speed, (tuple, list)) and len(configured_target_speed) == 2:
            min_speed = float(configured_target_speed[0])
            max_speed = float(configured_target_speed[1])
            if max_speed < min_speed:
                min_speed, max_speed = max_speed, min_speed
            self._custom_target_speed = float(self.np_random.uniform(min_speed, max_speed))
        elif configured_target_speed is None:
            self._custom_target_speed = None
        else:
            self._custom_target_speed = float(configured_target_speed)
        self._base_normal_speeds = {}

    def _apply_speed_cap(self, vehicle) -> None:
        policy = self.engine.get_policy(vehicle.name)
        if policy is None:
            return

        policy_id = id(policy)
        if hasattr(policy, "NORMAL_SPEED") and policy_id not in self._base_normal_speeds:
            self._base_normal_speeds[policy_id] = float(policy.NORMAL_SPEED)

        if self._custom_target_speed is None:
            if policy_id in self._base_normal_speeds and hasattr(policy, "NORMAL_SPEED"):
                policy.NORMAL_SPEED = self._base_normal_speeds[policy_id]
            return

        if hasattr(policy, "NORMAL_SPEED"):
            base_normal_speed = self._base_normal_speeds.get(policy_id, float(policy.NORMAL_SPEED))
            policy.NORMAL_SPEED = min(base_normal_speed, self._custom_target_speed)

        current_target_speed = getattr(policy, "target_speed", None)
        if current_target_speed is not None and float(current_target_speed) > self._custom_target_speed:
            policy.target_speed = self._custom_target_speed
