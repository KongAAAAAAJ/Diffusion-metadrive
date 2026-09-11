"""Training-loop hook for Risk-PACT visual diagnostics.

The key design goal is that visualization is *not* a separate executable mode.
The training loop creates one ``RiskPACTTrainingVisualizer`` and calls
``maybe_save`` after a teacher is built. When disabled, the hook is effectively
free and imports matplotlib only when an actual plot event is due.
"""
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


class RiskPACTTrainingVisualizer:
    """Small stateful gate around the Risk-PACT plotting utilities."""

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
        if not cfg.enabled:
            return False
        if step < cfg.start_step:
            return False
        if (step - cfg.start_step) % cfg.interval_steps != 0:
            return False
        if cfg.max_events is not None and self._event_count >= cfg.max_events:
            return False
        # Avoid duplicate plots if caller invokes maybe_save multiple times in one step.
        if self._last_step == step:
            return False
        return True

    def maybe_save(
        self,
        *,
        step: int,
        actor_state: Tensor,
        actor_valid_mask: Tensor,
        old_trajectory: Tensor,
        teacher_result: "PACTTeacherResult",
    ) -> RiskPACTVisualizationEvent | None:
        """Save debug plots if the configured step gate is active.

        Call this immediately after ``build_x0_pact_teacher``. Tensors are
        detached inside the plotting implementation; plotting therefore does not
        alter the optimization graph.
        """
        step = int(step)
        if not self.should_save(step):
            return None

        # Lazy import means formal training with enabled=False never imports pyplot.
        from .visualize import save_risk_pact_debug_plots

        output_dir = self.run_dir / self.config.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        save_risk_pact_debug_plots(
            actor_state=actor_state,
            actor_valid_mask=actor_valid_mask,
            old_trajectory=old_trajectory,
            teacher_trajectory=teacher_result.teacher_trajectory,
            output_dir=output_dir,
            step=step,
            batch_index=self.config.batch_index,
            role_index=self.config.role_index,
            mode_index=self.config.mode_index,
            config=self.risk_config,
        )
        self._event_count += 1
        self._last_step = step
        return RiskPACTVisualizationEvent(
            step=step,
            output_dir=output_dir,
            event_index=self._event_count,
        )
