from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BattlefieldDetectorConfig:
    method: str
    score_threshold: float
    sample_stride: int
    river_band_top_ratio: float
    river_band_bottom_ratio: float
    grass_band_top_ratio: float
    grass_band_bottom_ratio: float
    model_path: str | None
    model_input_size: int
    model_layout_path: str
