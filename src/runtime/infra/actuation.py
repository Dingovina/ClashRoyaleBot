from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.runtime.infra.keyboard_input import send_slot_hotkey
from src.runtime.domain.types import ActionDecision, ActionType
from src.runtime.infra.viewport import GameViewport
from src.runtime.domain.zones import ZoneMap


def resolve_card_hotkey(card_index: int, hotkeys: tuple[str, str, str, str]) -> str | None:
    if card_index not in (1, 2, 3, 4):
        return None
    return hotkeys[card_index - 1]


@dataclass(frozen=True)
class ActionExecutionResult:
    executed: bool
    reason: str
    click_x: int | None = None
    click_y: int | None = None


class InputActuator:
    def __init__(
        self,
        enabled: bool,
        dry_run: bool,
        logger: logging.Logger,
        game_viewport: GameViewport,
        select_to_click_delay_ms: int = 0,
        card_hotkeys: tuple[str, str, str, str] = ("1", "2", "3", "4"),
    ) -> None:
        self.enabled = enabled
        self.dry_run = dry_run
        self.logger = logger
        self.game_viewport = game_viewport
        self.select_to_click_delay_ms = max(0, select_to_click_delay_ms)
        self.card_hotkeys = card_hotkeys

    def execute(
        self,
        decision: ActionDecision,
        zone_map: ZoneMap,
        frame_width: int,
        frame_height: int,
    ) -> ActionExecutionResult:
        if decision.action_type != ActionType.DEPLOY:
            return ActionExecutionResult(executed=False, reason="decision_no_op")
        if not self.enabled:
            return ActionExecutionResult(executed=False, reason="actuation_disabled")
        if decision.card_index is None or decision.zone_id is None:
            return ActionExecutionResult(executed=False, reason="missing_action_fields")
        if frame_width <= 0 or frame_height <= 0:
            return ActionExecutionResult(executed=False, reason="invalid_frame_geometry")

        anchor = zone_map.anchors.get(decision.zone_id)
        if anchor is None:
            return ActionExecutionResult(executed=False, reason="unknown_zone_anchor")

        left, top, view_w, view_h = self.game_viewport.rect_for_frame(frame_width, frame_height)
        rect = self.game_viewport.anchor_rect
        norm_x = rect.left_ratio + anchor[0] * rect.width_ratio
        norm_y = rect.top_ratio + anchor[1] * rect.height_ratio
        click_x = left + int(norm_x * view_w)
        click_y = top + int(norm_y * view_h)

        if self.dry_run:
            return ActionExecutionResult(
                executed=True,
                reason="dry_run_executed",
                click_x=click_x,
                click_y=click_y,
            )

        hotkey = resolve_card_hotkey(decision.card_index, self.card_hotkeys)
        if hotkey is None:
            return ActionExecutionResult(executed=False, reason="illegal_card_index")

        send_slot_hotkey(hotkey, self.logger)
        if self.select_to_click_delay_ms > 0:
            time.sleep(self.select_to_click_delay_ms / 1000.0)

        try:
            import pyautogui
        except ModuleNotFoundError:
            self.logger.error("actuation_failed reason=pyautogui_not_installed")
            return ActionExecutionResult(executed=False, reason="pyautogui_not_installed")

        pyautogui.click(x=click_x, y=click_y)
        return ActionExecutionResult(
            executed=True,
            reason="input_sent",
            click_x=click_x,
            click_y=click_y,
        )
