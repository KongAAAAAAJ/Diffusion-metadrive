"""Training-loop hook for Step-6 Risk-PACT diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor

from .config import RiskPACTConfig, RiskPACTVisualizationConfig

if TYPE_CHECKING:
    from .teacher import PACTTeacherResult


@dataclass(frozen=True)
class RiskPACTVisualizationEvent:
    step: int
    output_dir: Path
    event_index: int
    summary: dict[str, object]


class RiskPACTTrainingVisualizer:
    """Stateful gate around the Step-6 multi-source plotting utilities."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        risk_config: RiskPACTConfig | None = None,
        visualization_config: RiskPACTVisualizationConfig | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.risk_config = risk_config or RiskPACTConfig()
        self.config = visualization_config or RiskPACTVisualizationConfig()
        self._event_count = 0
        self._last_step: int | None = None

    @property
    def event_count(self) -> int:
        return int(self._event_count)

    def should_save(self, step: int) -> bool:
        cfg = self.config
        step = int(step)
        if not cfg.enabled or step < cfg.start_step:
            return False
        if (step - cfg.start_step) % cfg.interval_steps != 0:
            return False
        if cfg.max_events is not None and self._event_count >= cfg.max_events:
            return False
        if self._last_step == step:
            return False
        return True

    def maybe_save(
        self,
        *,
        step: int,
        background_actor_state: Tensor,
        background_actor_valid_mask: Tensor,
        platoon_actor_state: Tensor,
        platoon_actor_valid_mask: Tensor,
        road_sdf: Tensor | None,
        old_trajectory: Tensor,
        valid_executable_mode_mask: Tensor,
        teacher_result: "PACTTeacherResult",
    ) -> RiskPACTVisualizationEvent | None:
        step = int(step)
        if not self.should_save(step):
            return None
        from .visualize import save_risk_pact_debug_plots

        output_dir = self.run_dir / self.config.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = save_risk_pact_debug_plots(
            background_actor_state=background_actor_state,
            background_actor_valid_mask=background_actor_valid_mask,
            platoon_actor_state=platoon_actor_state,
            platoon_actor_valid_mask=platoon_actor_valid_mask,
            road_sdf=road_sdf,
            old_trajectory=old_trajectory,
            valid_executable_mode_mask=valid_executable_mode_mask,
            teacher_result=teacher_result,
            output_dir=output_dir,
            step=step,
            visualization_config=self.config,
            config=self.risk_config,
        )
        self._event_count += 1
        self._last_step = step
        return RiskPACTVisualizationEvent(
            step=step,
            output_dir=output_dir,
            event_index=self._event_count,
            summary=summary,
        )
