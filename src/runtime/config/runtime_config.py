from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.runtime.infra.viewport import GameViewport

BattlefieldTimeoutBehavior = Literal["idle", "exit_nonzero"]


@dataclass(frozen=True)
class RuntimeConfig:
    match_id: str
    tick_interval_ms: int
    action_rate_limit_ms: int
    action_confidence_threshold: float
    no_op_confidence_threshold: float
    min_elixir_for_non_urgent_action: float
    match_safety_max_ticks: int
    battlefield_end_score_threshold: float
    match_end_confirm_ticks: int
    match_end_check_every_n_ticks: int
    zones: dict[int, tuple[float, float]]
    spell_cards: set[str]
    capture_debug_save_enabled: bool
    capture_debug_dir: str | None
    capture_every_n_ticks: int
    actuation_select_to_click_delay_ms: int
    actuation_card_hotkeys: tuple[str, str, str, str]
    game_viewport: GameViewport
    match_readiness_enabled: bool
    battlefield_score_threshold: float
    battlefield_wait_timeout_ms: int
    battlefield_timeout_behavior: BattlefieldTimeoutBehavior
    foreground_check_enabled: bool
    foreground_title_substrings: tuple[str, ...]
    battlefield_model_path: str | None
    battlefield_model_layout_path: str
    elixir_model_path: str | None
    elixir_model_layout_path: str
    card_model_path: str | None
    card_model_layout_path: str
    hand_tick_log_enabled: bool
    hand_tick_log_path: str
    session_id: str
    card_name_aliases: dict[str, str]
    card_elixir_costs: dict[str, float]
