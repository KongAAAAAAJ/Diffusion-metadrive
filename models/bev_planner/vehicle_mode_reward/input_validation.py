"""Input-shape validation helpers for the counterfactual reward scorer."""

from __future__ import annotations

import numpy as np

from .config import JointRewardError
from .constants import NUM_MODES, NUM_ROLES, TRAJECTORY_SHAPE


class _CounterfactualInputMixin:
    @staticmethod
    def _array_without_optional_batch(
        value: object,
        *,
        unbatched_ndim: int,
        name: str,
    ) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == unbatched_ndim + 1:
            if array.shape[0] != 1:
                raise JointRewardError(f"{name} only supports an optional B=1 axis")
            array = array[0]
        if array.ndim != unbatched_ndim:
            raise JointRewardError(f"{name} has the wrong rank")
        return array

    def _candidate_values(
        self,
        candidates: object,
        *,
        trajectories_per_mode: int | None = None,
    ) -> np.ndarray:
        values = self._array_without_optional_batch(
            candidates, unbatched_ndim=5, name="candidates"
        )
        sample_count = (
            self.config.trajectories_per_mode
            if trajectories_per_mode is None
            else int(trajectories_per_mode)
        )
        expected = (
            NUM_ROLES,
            NUM_MODES,
            sample_count,
            *TRAJECTORY_SHAPE,
        )
        if (
            values.shape != expected
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise JointRewardError(
                "candidates must be finite floating-point [3,10,N,8,3]"
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def _frozen_all_values(self, trajectories: object) -> np.ndarray:
        values = self._array_without_optional_batch(
            trajectories,
            unbatched_ndim=4,
            name="frozen_all_mode_trajectories",
        )
        if (
            values.shape != (NUM_ROLES, NUM_MODES, *TRAJECTORY_SHAPE)
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise JointRewardError(
                "frozen_all_mode_trajectories must be finite floating-point "
                "[3,10,8,3]"
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def _frozen_argmax_values(self, trajectories: object) -> np.ndarray:
        values = self._array_without_optional_batch(
            trajectories,
            unbatched_ndim=3,
            name="frozen_argmax_joint_trajectories",
        )
        if (
            values.shape != (NUM_ROLES, *TRAJECTORY_SHAPE)
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise JointRewardError(
                "frozen_argmax_joint_trajectories must be finite floating-point "
                "[3,8,3]"
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def _valid_values(self, valid_mode_mask: object) -> np.ndarray:
        values = self._array_without_optional_batch(
            valid_mode_mask, unbatched_ndim=2, name="valid_mode_mask"
        )
        if values.shape != (NUM_ROLES, NUM_MODES) or values.dtype != np.bool_:
            raise JointRewardError("valid_mode_mask must be bool [3,10]")
        return np.ascontiguousarray(values, dtype=np.bool_)
