from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.runtime.actuation import InputActuator
from src.runtime.adapters.jsonl_event_sink import JsonlTickEventSink
from src.runtime.adapters.perception_service import RuntimePerceptionService
from src.runtime.capture import FullscreenCapture
from src.runtime.match_exit import MatchExitTracker
from src.runtime.match_readiness import wait_for_match_readiness
from src.runtime.policy_gate import PolicyGate
from src.runtime.runtime_config import RuntimeConfig
from src.runtime.tick_orchestrator import TickOrchestrator
from src.runtime.zones import build_default_zone_map


@dataclass
class RuntimeService:
    config: RuntimeConfig
    logger: logging.Logger

    def run(self) -> int:
        zone_map = build_default_zone_map(self.config.zones)
        gate = PolicyGate(config=self.config, zone_map=zone_map)
        debug_dir = self.config.capture_debug_dir if self.config.capture_debug_save_enabled else None
        capture = FullscreenCapture(
            logger=self.logger,
            debug_dir=debug_dir,
            capture_every_n_ticks=self.config.capture_every_n_ticks,
        )
        actuator = InputActuator(
            enabled=True,
            dry_run=False,
            logger=self.logger,
            game_viewport=self.config.game_viewport,
            select_to_click_delay_ms=self.config.actuation_select_to_click_delay_ms,
            card_hotkeys=self.config.actuation_card_hotkeys,
        )
        orchestrator = TickOrchestrator(
            config=self.config,
            logger=self.logger,
            zone_map=zone_map,
            capture=capture,
            perception_service=RuntimePerceptionService(config=self.config, logger=self.logger),
            gate=gate,
            actuator=actuator,
            event_sink=JsonlTickEventSink(config=self.config, logger=self.logger),
        )

        exit_code, match_ready = wait_for_match_readiness(self.config, self.logger, capture)
        if exit_code != 0:
            return exit_code

        end_tracker = MatchExitTracker()
        end_tracking = (
            match_ready
            and self.config.match_end_confirm_ticks > 0
            and self.config.battlefield_model_path is not None
        )

        start = time.perf_counter()
        tick_id = 0
        while True:
            if self.config.match_safety_max_ticks > 0 and tick_id >= self.config.match_safety_max_ticks:
                self.logger.info("runtime_finished reason=match_safety_max_ticks ticks=%s", tick_id)
                return 0
            want_match_end_pixels = (
                end_tracking
                and (tick_id % self.config.match_end_check_every_n_ticks == 0)
            )
            want_elixir_pixels = True
            want_card_pixels = True
            match_ended = orchestrator.run_tick(
                tick_id=tick_id,
                loop_start=start,
                match_ready=match_ready,
                include_pixels=want_match_end_pixels or want_elixir_pixels or want_card_pixels,
                include_match_end_pixels=want_match_end_pixels,
                end_tracker=end_tracker if end_tracking else None,
            )
            if match_ended:
                self.logger.info(
                    "runtime_finished reason=battlefield_absent ticks=%s end_streak=%s",
                    tick_id + 1,
                    end_tracker.consecutive_below,
                )
                return 0
            tick_id += 1
