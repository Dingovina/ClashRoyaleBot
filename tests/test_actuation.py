from __future__ import annotations

import unittest

from src.runtime.infra.actuation import InputActuator, resolve_card_hotkey
from src.runtime.infra.viewport import AnchorRect, GameViewport
from src.runtime.domain.types import ActionDecision
from src.runtime.domain.zones import build_default_zone_map


class ActuationTests(unittest.TestCase):
    def test_resolve_card_hotkey_maps_slots(self) -> None:
        hotkeys = ("q", "w", "e", "r")
        self.assertEqual(resolve_card_hotkey(1, hotkeys), "q")
        self.assertEqual(resolve_card_hotkey(4, hotkeys), "r")
        self.assertIsNone(resolve_card_hotkey(0, hotkeys))
        self.assertIsNone(resolve_card_hotkey(5, hotkeys))

    def test_dry_run_executes_deploy_action(self) -> None:
        zone_map = build_default_zone_map({7: (0.5, 0.75)})
        actuator = InputActuator(
            enabled=True,
            dry_run=True,
            logger=_NullLogger(),
            game_viewport=GameViewport(mode="full_frame"),
            select_to_click_delay_ms=0,
            card_hotkeys=("q", "w", "e", "r"),
        )
        decision = ActionDecision(action_type="deploy", reason="accepted", card_index=1, zone_id=7)

        result = actuator.execute(decision=decision, zone_map=zone_map, frame_width=1000, frame_height=800)

        self.assertTrue(result.executed)
        self.assertEqual(result.reason, "dry_run_executed")
        self.assertEqual(result.click_x, 500)
        self.assertEqual(result.click_y, 600)

    def test_disabled_actuation_skips_execution(self) -> None:
        zone_map = build_default_zone_map({7: (0.5, 0.75)})
        actuator = InputActuator(
            enabled=False,
            dry_run=True,
            logger=_NullLogger(),
            game_viewport=GameViewport(mode="full_frame"),
            select_to_click_delay_ms=0,
        )
        decision = ActionDecision(action_type="deploy", reason="accepted", card_index=1, zone_id=7)

        result = actuator.execute(decision=decision, zone_map=zone_map, frame_width=1000, frame_height=800)

        self.assertFalse(result.executed)
        self.assertEqual(result.reason, "actuation_disabled")

    def test_centered_strip_offsets_clicks_on_wide_frame(self) -> None:
        zone_map = build_default_zone_map({7: (0.5, 0.75)})
        viewport = GameViewport(mode="centered_strip", width=608, height=1080)
        actuator = InputActuator(
            enabled=True,
            dry_run=True,
            logger=_NullLogger(),
            game_viewport=viewport,
            select_to_click_delay_ms=0,
        )
        decision = ActionDecision(action_type="deploy", reason="accepted", card_index=1, zone_id=7)

        result = actuator.execute(decision=decision, zone_map=zone_map, frame_width=1920, frame_height=1080)

        self.assertTrue(result.executed)
        self.assertEqual(result.click_x, 960)
        self.assertEqual(result.click_y, 810)

    def test_anchor_rect_excludes_bottom_hand_strip(self) -> None:
        zone_map = build_default_zone_map({11: (0.5, 0.875)})
        viewport = GameViewport(
            mode="centered_strip",
            width=608,
            height=1080,
            anchor_rect=AnchorRect(left_ratio=0.0, top_ratio=0.1, width_ratio=1.0, height_ratio=0.65),
        )
        actuator = InputActuator(
            enabled=True,
            dry_run=True,
            logger=_NullLogger(),
            game_viewport=viewport,
            select_to_click_delay_ms=0,
        )
        decision = ActionDecision(action_type="deploy", reason="accepted", card_index=1, zone_id=11)

        result = actuator.execute(decision=decision, zone_map=zone_map, frame_width=1920, frame_height=1080)

        self.assertTrue(result.executed)
        self.assertEqual(result.click_x, 960)
        self.assertEqual(result.click_y, 722)

class _NullLogger:
    def error(self, _: str) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
