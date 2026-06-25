from __future__ import annotations

import numpy as np


class RearEndGuardRegulator:
    BASE_MIN_GAP = 8.0
    HEADWAY_GUARD = 2.0
    TTC_SOFT = 4.0
    TTC_HARD = 2.0
    SOFT_BRAKE = -2.6
    HARD_BRAKE = -4.5
    MAX_BRAKE = -4.5
    ACC_RATE_LIMIT = 1.0

    def __init__(self):
        self._eps = 1e-6
        self._last_accel = None
        self.last_diagnostics = {
            "active": False,
            "acc": 0.0,
            "ttc": None,
            "gap": None,
        }

    def adjust_acceleration(self, ego, front_obj, front_dist, idm_acc) -> float:
        idm_acc = float(idm_acc)
        gap = float(front_dist)

        if front_obj is None:
            return self._finalize(idm_acc=idm_acc, accel_cmd=idm_acc, gap=gap, ttc=None)

        front_speed = self._front_speed(front_obj)
        if front_speed is None:
            return self._finalize(idm_acc=idm_acc, accel_cmd=idm_acc, gap=gap, ttc=None)

        ego_speed = max(float(getattr(ego, "speed", 0.0)), 0.0)
        closing_speed = ego_speed - front_speed
        if closing_speed <= 0.0:
            return self._finalize(idm_acc=idm_acc, accel_cmd=idm_acc, gap=gap, ttc=None)

        safe_gap = self.BASE_MIN_GAP + self.HEADWAY_GUARD * ego_speed
        ttc = gap / max(closing_speed, self._eps)

        if idm_acc < self.MAX_BRAKE:
            return self._finalize(idm_acc=idm_acc, accel_cmd=idm_acc, gap=gap, ttc=ttc, apply_rate_limit=False)
        elif ttc < self.TTC_HARD or gap < 0.5 * safe_gap:
            accel_cmd = max(min(idm_acc, self.HARD_BRAKE), self.MAX_BRAKE)
        elif gap < safe_gap or ttc < self.TTC_SOFT:
            accel_cmd = max(min(idm_acc, self.SOFT_BRAKE), self.MAX_BRAKE)
        else:
            accel_cmd = idm_acc

        return self._finalize(idm_acc=idm_acc, accel_cmd=accel_cmd, gap=gap, ttc=ttc)

    def _front_speed(self, front_obj):
        if hasattr(front_obj, "speed"):
            return max(float(front_obj.speed), 0.0)
        if hasattr(front_obj, "speed_km_h"):
            return max(float(front_obj.speed_km_h) / 3.6, 0.0)
        velocity = getattr(front_obj, "velocity_km_h", None)
        if velocity is not None:
            speed = np.linalg.norm(np.asarray(velocity, dtype=np.float64)) / 3.6
            return max(float(speed), 0.0)
        return None

    def _limit_accel_change(self, accel_cmd) -> float:
        accel_cmd = float(accel_cmd)
        if self._last_accel is None:
            self._last_accel = accel_cmd
            return accel_cmd
        limited = float(
            np.clip(
                accel_cmd,
                self._last_accel - self.ACC_RATE_LIMIT,
                self._last_accel + self.ACC_RATE_LIMIT,
            )
        )
        self._last_accel = limited
        return limited

    def _finalize(self, idm_acc, accel_cmd, gap, ttc, apply_rate_limit=True):
        if apply_rate_limit:
            final_acc = self._limit_accel_change(accel_cmd)
        else:
            final_acc = float(accel_cmd)
            self._last_accel = final_acc
        self.last_diagnostics = {
            "active": final_acc < float(idm_acc) - 1e-6,
            "acc": float(final_acc),
            "ttc": None if ttc is None else float(ttc),
            "gap": float(gap),
        }
        return float(final_acc)

    def reset(self) -> None:
        self._last_accel = None
