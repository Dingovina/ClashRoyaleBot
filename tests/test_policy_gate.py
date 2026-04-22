from __future__ import annotations

import unittest

from src.runtime.policy_gate import PolicyGate
from src.runtime.runtime_config import RuntimeConfig
from src.runtime.types import CandidateAction, RuntimeState
from src.runtime.viewport import GameViewport
from src.runtime.zones import build_default_zone_map


def _config() -> RuntimeConfig:
    anchors = {i: (0.1, 0.1) for i in range(12)}
    return RuntimeConfig(
        tick_interval_ms=500,
        action_rate_limit_ms=1000,
        action_confidence_threshold=0.70,
        no_op_confidence_threshold=0.55,
        min_elixir_for_non_urgent_action=3.0,
        match_safety_max_ticks=500,
        battlefield_end_score_threshold=0.42,
        match_end_confirm_ticks=6,
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
        battlefield_wait_timeout_ms=120000,
        battlefield_timeout_behavior="idle",
        foreground_check_enabled=False,
        foreground_title_substrings=("clash royale", "google play games", "google play"),
        battlefield_model_path=None,
        battlefield_model_layout_path="configs/screen_layout_reference.yaml",
    )


def _state(timestamp_ms: int, elixir: float = 5.0) -> RuntimeState:
    return RuntimeState(tick_id=0, timestamp_ms=timestamp_ms, elixir=elixir)


def _candidate(confidence: float, zone_id: int = 7, urgent: bool = False) -> CandidateAction:
    return CandidateAction(
        card_index=1,
        card_name="knight",
        card_class="unit",
        zone_id=zone_id,
        confidence=confidence,
        urgent_defense=urgent,
    )


class PolicyGateTests(unittest.TestCase):
    def setUp(self) -> None:
        cfg = _config()
        self.gate = PolicyGate(config=cfg, zone_map=build_default_zone_map(cfg.zones))

    def test_no_op_when_confidence_below_no_op_threshold(self) -> None:
        decision = self.gate.decide(_state(0), _candidate(confidence=0.50))
        self.assertEqual(decision.action_type, "no_op")

    def test_deploy_when_confidence_and_legality_pass(self) -> None:
        decision = self.gate.decide(_state(0), _candidate(confidence=0.85, zone_id=7))
        self.assertEqual(decision.action_type, "deploy")
        self.assertEqual(decision.zone_id, 7)

    def test_rate_limit_blocks_second_immediate_action(self) -> None:
        first = self.gate.decide(_state(0), _candidate(confidence=0.85, zone_id=7))
        second = self.gate.decide(_state(500), _candidate(confidence=0.90, zone_id=7))
        self.assertEqual(first.action_type, "deploy")
        self.assertEqual(second.action_type, "no_op")
        self.assertEqual(second.reason, "rate_limited")

    def test_urgent_action_overrides_uncertainty_band(self) -> None:
        decision = self.gate.decide(_state(0, elixir=2.0), _candidate(confidence=0.60, urgent=True))
        self.assertEqual(decision.action_type, "deploy")


if __name__ == "__main__":
    unittest.main()
