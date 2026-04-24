from __future__ import annotations

import logging
import unittest
from dataclasses import dataclass

from src.runtime.actuation import ActionExecutionResult
from src.runtime.capture import FrameObservation
from src.runtime.policy_gate import PolicyGate
from src.runtime.runtime_config import RuntimeConfig
from src.runtime.tick_orchestrator import TickOrchestrator
from src.runtime.types import ActionDecision, InferenceStatus, PerceptionSnapshot
from src.runtime.viewport import GameViewport
from src.runtime.zones import build_default_zone_map


def _config() -> RuntimeConfig:
    anchors = {i: (0.1, 0.1) for i in range(12)}
    return RuntimeConfig(
        tick_interval_ms=1,
        action_rate_limit_ms=1000,
        action_confidence_threshold=0.7,
        no_op_confidence_threshold=0.55,
        min_elixir_for_non_urgent_action=3.0,
        match_safety_max_ticks=100,
        battlefield_end_score_threshold=0.42,
        match_end_confirm_ticks=0,
        match_end_check_every_n_ticks=2,
        zones=anchors,
        spell_cards={"fireball"},
        capture_enabled=False,
        capture_debug_save_enabled=False,
        capture_debug_dir=None,
        capture_every_n_ticks=0,
        actuation_enabled=False,
        actuation_dry_run=True,
        actuation_select_to_click_delay_ms=0,
        actuation_card_hotkeys=("1", "2", "3", "4"),
        game_viewport=GameViewport(mode="full_frame"),
        match_readiness_enabled=False,
        battlefield_score_threshold=0.65,
        battlefield_wait_timeout_ms=0,
        battlefield_timeout_behavior="idle",
        foreground_check_enabled=False,
        foreground_title_substrings=("clash royale",),
        battlefield_model_path=None,
        battlefield_model_layout_path="configs/screen_layout_reference.yaml",
        elixir_model_enabled=False,
        elixir_model_path=None,
        elixir_model_layout_path="configs/screen_layout_reference.yaml",
        card_model_enabled=False,
        card_model_path=None,
        card_model_layout_path="configs/screen_layout_reference.yaml",
        hand_tick_log_enabled=False,
        hand_tick_log_path="logs/hand_cards_ticks.jsonl",
        session_id="test-session",
        card_name_aliases={"knight": "knight"},
        card_elixir_costs={"knight": 3.0},
    )


@dataclass
class _FakeFrameSource:
    def frame_for_tick(self, tick_id: int, include_pixels: bool):
        return FrameObservation(
            width=0,
            height=0,
            capture_latency_ms=0,
            source="disabled",
            pixels_bgra=None,
        )


@dataclass
class _FakePerception:
    def infer(self, frame: FrameObservation) -> PerceptionSnapshot:
        return PerceptionSnapshot(
            elixir=0.0,
            elixir_confidence=0.0,
            elixir_status=InferenceStatus.DISABLED,
            hand_cards=("unknown", "unknown", "unknown", "unknown"),
            hand_confidences=(0.0, 0.0, 0.0, 0.0),
            hand_status=InferenceStatus.DISABLED,
        )


class _FakeActuator:
    def __init__(self) -> None:
        self.last_decision: ActionDecision | None = None

    def execute(self, decision: ActionDecision, zone_map, frame_width: int, frame_height: int):
        self.last_decision = decision
        return ActionExecutionResult(executed=False, reason="test")


class _FakeSink:
    def __init__(self) -> None:
        self.rows = []

    def publish_tick(self, event) -> None:
        self.rows.append(event)


class TickOrchestratorTests(unittest.TestCase):
    def test_forces_no_op_when_match_not_ready(self) -> None:
        cfg = _config()
        zone_map = build_default_zone_map(cfg.zones)
        gate = PolicyGate(config=cfg, zone_map=zone_map)
        actuator = _FakeActuator()
        sink = _FakeSink()
        orchestrator = TickOrchestrator(
            config=cfg,
            logger=logging.getLogger("test"),
            zone_map=zone_map,
            frame_source=_FakeFrameSource(),
            perception_service=_FakePerception(),
            gate=gate,
            actuator=actuator,
            event_sink=sink,
        )
        ended = orchestrator.run_tick(
            tick_id=1,
            loop_start=0.0,
            match_ready=False,
            include_pixels=False,
            include_match_end_pixels=False,
            end_tracker=None,
        )
        self.assertFalse(ended)
        self.assertIsNotNone(actuator.last_decision)
        self.assertEqual(actuator.last_decision.action_type.value, "no_op")
        self.assertEqual(actuator.last_decision.reason.value, "match_readiness_not_ready")
        self.assertEqual(len(sink.rows), 1)


if __name__ == "__main__":
    unittest.main()
