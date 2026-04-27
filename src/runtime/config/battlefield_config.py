from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BattlefieldModelConfig:
    """CNN-only match-readiness: masked bottom-panel vs trained checkpoint."""

    score_threshold: float
    model_path: str
    model_layout_path: str
