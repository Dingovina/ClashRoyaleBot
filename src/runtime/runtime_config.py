from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.runtime.viewport import GameViewport

BattlefieldTimeoutBehavior = Literal["idle", "exit_nonzero"]


@dataclass(frozen=True)
class RuntimeConfig:
    tick_interval_ms: int
    action_rate_limit_ms: int
    action_confidence_threshold: float
    no_op_confidence_threshold: float
    min_elixir_for_non_urgent_action: float
    max_ticks: int
    zones: dict[int, tuple[float, float]]
    spell_cards: set[str]
    capture_enabled: bool
    capture_debug_dir: str | None
    capture_every_n_ticks: int
    actuation_enabled: bool
    actuation_dry_run: bool
    actuation_select_to_click_delay_ms: int
    actuation_card_hotkeys: tuple[str, str, str, str]
    game_viewport: GameViewport
    match_readiness_enabled: bool
    battlefield_detector: str
    battlefield_score_threshold: float
    battlefield_sample_stride: int
    battlefield_river_band_top_ratio: float
    battlefield_river_band_bottom_ratio: float
    battlefield_grass_band_top_ratio: float
    battlefield_grass_band_bottom_ratio: float
    battlefield_wait_timeout_ms: int
    battlefield_timeout_behavior: BattlefieldTimeoutBehavior
    foreground_check_enabled: bool
    foreground_title_substrings: tuple[str, ...]
    battlefield_model_path: str | None
    battlefield_model_input_size: int
    battlefield_model_layout_path: str
