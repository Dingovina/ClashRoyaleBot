from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

from src.runtime.actuation import InputActuator
from src.runtime.battlefield_detector import BattlefieldDetectorConfig, evaluate_battlefield
from src.runtime.candidate_policy import propose_candidate_action
from src.runtime.capture import FrameObservation, FullscreenCapture
from src.runtime.config import RuntimeConfig
from src.runtime.foreground_win import foreground_matches, foreground_title_lower
from src.runtime.policy_gate import PolicyGate
from src.runtime.types import ActionDecision, RuntimeState
from src.runtime.zones import ZoneMap, build_default_zone_map


@dataclass
class RuntimeLoop:
    config: RuntimeConfig
    logger: logging.Logger

    def run(self) -> int:
        zone_map = build_default_zone_map(self.config.zones)
        gate = PolicyGate(config=self.config, zone_map=zone_map)
        capture = FullscreenCapture(
            logger=self.logger,
            debug_dir=self.config.capture_debug_dir,
            capture_every_n_ticks=self.config.capture_every_n_ticks,
        )
        actuator = InputActuator(
            enabled=self.config.actuation_enabled,
            dry_run=self.config.actuation_dry_run,
            logger=self.logger,
            game_viewport=self.config.game_viewport,
            select_to_click_delay_ms=self.config.actuation_select_to_click_delay_ms,
            card_hotkeys=self.config.actuation_card_hotkeys,
        )

        exit_code, match_ready = self._prepare_match_readiness(capture)
        if exit_code != 0:
            return exit_code

        self.logger.info(
            "runtime_started tick_interval_ms=%s action_rate_limit_ms=%s max_ticks=%s match_readiness=%s match_ready=%s",
            self.config.tick_interval_ms,
            self.config.action_rate_limit_ms,
            self.config.max_ticks,
            self.config.match_readiness_enabled,
            match_ready,
        )

        start = time.perf_counter()
        for tick_id in range(self.config.max_ticks):
            self._run_tick(
                tick_id=tick_id,
                loop_start=start,
                zone_map=zone_map,
                gate=gate,
                capture=capture,
                actuator=actuator,
                match_ready=match_ready,
            )

        self.logger.info("runtime_finished ticks=%s", self.config.max_ticks)
        return 0

    def _prepare_match_readiness(self, capture: FullscreenCapture) -> tuple[int, bool]:
        if not self.config.match_readiness_enabled:
            return (0, True)

        detector = BattlefieldDetectorConfig(
            method=self.config.battlefield_detector,
            score_threshold=self.config.battlefield_score_threshold,
            sample_stride=self.config.battlefield_sample_stride,
            river_band_top_ratio=self.config.battlefield_river_band_top_ratio,
            river_band_bottom_ratio=self.config.battlefield_river_band_bottom_ratio,
            grass_band_top_ratio=self.config.battlefield_grass_band_top_ratio,
            grass_band_bottom_ratio=self.config.battlefield_grass_band_bottom_ratio,
            model_path=self.config.battlefield_model_path,
            model_input_size=self.config.battlefield_model_input_size,
            model_layout_path=self.config.battlefield_model_layout_path,
        )

        if self.config.battlefield_wait_timeout_ms > 0:
            deadline = time.perf_counter() + self.config.battlefield_wait_timeout_ms / 1000.0
        else:
            deadline = None

        wait_tick = 0
        logged_foreground_skip = False
        while True:
            if deadline is not None and time.perf_counter() > deadline:
                self.logger.info(
                    "battlefield_wait_timeout behavior=%s timeout_ms=%s",
                    self.config.battlefield_timeout_behavior,
                    self.config.battlefield_wait_timeout_ms,
                )
                if self.config.battlefield_timeout_behavior == "exit_nonzero":
                    return (2, False)
                self.logger.info("battlefield_timeout_continue actuation_blocked=1")
                return (0, False)

            frame = capture.capture(wait_tick, include_pixels=True)
            wait_tick += 1

            if frame.width <= 0 or frame.source != "fullscreen" or not frame.pixels_bgra:
                self.logger.info("waiting_for_battlefield reason=capture_unavailable source=%s", frame.source)
                time.sleep(self.config.tick_interval_ms / 1000.0)
                continue

            if self.config.foreground_check_enabled:
                if sys.platform != "win32":
                    if not logged_foreground_skip:
                        self.logger.info("foreground_check_skipped reason=non_windows_platform")
                        logged_foreground_skip = True
                else:
                    title = foreground_title_lower()
                    if not foreground_matches(title or "", self.config.foreground_title_substrings):
                        self.logger.info(
                            "waiting_for_battlefield reason=foreground_mismatch title=%r",
                            title or "",
                        )
                        time.sleep(self.config.tick_interval_ms / 1000.0)
                        continue

            ready, score = evaluate_battlefield(
                frame_width=frame.width,
                frame_height=frame.height,
                pixels_bgra=frame.pixels_bgra,
                viewport=self.config.game_viewport,
                detector=detector,
                logger=self.logger,
            )
            if ready:
                self.logger.info(
                    "battlefield_detected score=%.3f threshold=%.3f",
                    score,
                    self.config.battlefield_score_threshold,
                )
                return (0, True)

            self.logger.info(
                "waiting_for_battlefield score=%.3f threshold=%.3f",
                score,
                self.config.battlefield_score_threshold,
            )
            time.sleep(self.config.tick_interval_ms / 1000.0)

    def _run_tick(
        self,
        tick_id: int,
        loop_start: float,
        zone_map: ZoneMap,
        gate: PolicyGate,
        capture: FullscreenCapture,
        actuator: InputActuator,
        *,
        match_ready: bool,
    ) -> None:
        tick_start = time.perf_counter()
        now_ms = int((tick_start - loop_start) * 1000)
        frame = _capture_or_stub(
            capture,
            self.config.capture_enabled,
            tick_id,
            include_pixels=False,
        )

        state = RuntimeState(
            tick_id=tick_id,
            timestamp_ms=now_ms,
            elixir=min(10.0, tick_id * 0.5),
        )
        candidate = propose_candidate_action(state=state, frame=frame)
        decision = gate.decide(state=state, candidate=candidate)
        if not match_ready:
            decision = ActionDecision(
                action_type="no_op",
                reason="match_readiness_not_ready",
                card_index=decision.card_index,
                zone_id=decision.zone_id,
            )
        execution = actuator.execute(
            decision=decision,
            zone_map=zone_map,
            frame_width=frame.width,
            frame_height=frame.height,
        )

        self.logger.info(
            "tick=%s ts_ms=%s elixir=%.1f frame=%sx%s capture_ms=%s candidate=%s confidence=%s decision=%s reason=%s card=%s zone=%s",
            state.tick_id,
            state.timestamp_ms,
            state.elixir,
            frame.width,
            frame.height,
            frame.capture_latency_ms,
            candidate.card_name if candidate else None,
            round(candidate.confidence, 3) if candidate else None,
            decision.action_type,
            decision.reason,
            decision.card_index,
            decision.zone_id,
        )
        self.logger.info(
            "action_attempt tick=%s decision=%s executed=%s exec_reason=%s click_x=%s click_y=%s",
            state.tick_id,
            decision.action_type,
            execution.executed,
            execution.reason,
            execution.click_x,
            execution.click_y,
        )

        elapsed_ms = int((time.perf_counter() - tick_start) * 1000)
        sleep_ms = max(0, self.config.tick_interval_ms - elapsed_ms)
        time.sleep(sleep_ms / 1000.0)


def _capture_or_stub(
    capture: FullscreenCapture, enabled: bool, tick_id: int, *, include_pixels: bool
) -> FrameObservation:
    if not enabled:
        return FrameObservation(
            width=0,
            height=0,
            capture_latency_ms=0,
            source="disabled",
            screenshot_path=None,
            pixels_bgra=None,
        )
    return capture.capture(tick_id, include_pixels=include_pixels)
