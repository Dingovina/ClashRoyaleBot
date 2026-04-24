from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.runtime.actuation import InputActuator
from src.runtime.battlefield_evaluate import infer_battlefield_probability
from src.runtime.card_evaluate import infer_hand_cards
from src.runtime.candidate_policy import propose_candidate_action
from src.runtime.capture import FullscreenCapture, frame_for_tick
from src.runtime.elixir_evaluate import infer_elixir_value
from src.runtime.match_exit import MatchExitTracker
from src.runtime.match_readiness import wait_for_match_readiness
from src.runtime.policy_gate import PolicyGate
from src.runtime.runtime_config import RuntimeConfig
from src.runtime.types import ActionDecision, RuntimeState
from src.runtime.zones import ZoneMap, build_default_zone_map


@dataclass
class RuntimeLoop:
    config: RuntimeConfig
    logger: logging.Logger

    def _write_hand_tick_log(self, tick_id: int, hand_cards: list[str], hand_confidences: list[float]) -> None:
        if not self.config.hand_tick_log_enabled:
            return
        path = Path(self.config.hand_tick_log_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "tick": tick_id,
                "cards": hand_cards,
                "confidences": [round(x, 4) for x in hand_confidences],
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        except Exception as exc:
            self.logger.warning("hand_tick_log_write_failed path=%s err=%s", path, exc)

    def run(self) -> int:
        zone_map = build_default_zone_map(self.config.zones)
        gate = PolicyGate(config=self.config, zone_map=zone_map)
        debug_dir = (
            self.config.capture_debug_dir if self.config.capture_debug_save_enabled else None
        )
        capture = FullscreenCapture(
            logger=self.logger,
            debug_dir=debug_dir,
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

        exit_code, match_ready = wait_for_match_readiness(self.config, self.logger, capture)
        if exit_code != 0:
            return exit_code

        end_tracker = MatchExitTracker()
        end_tracking = (
            match_ready
            and self.config.match_end_confirm_ticks > 0
            and self.config.battlefield_model_path is not None
        )

        self.logger.info(
            "runtime_started tick_interval_ms=%s action_rate_limit_ms=%s match_safety_max_ticks=%s "
            "match_end_confirm_ticks=%s match_end_check_every_n_ticks=%s match_readiness=%s match_ready=%s",
            self.config.tick_interval_ms,
            self.config.action_rate_limit_ms,
            self.config.match_safety_max_ticks,
            self.config.match_end_confirm_ticks,
            self.config.match_end_check_every_n_ticks,
            self.config.match_readiness_enabled,
            match_ready,
        )

        start = time.perf_counter()
        tick_id = 0
        while True:
            if self.config.match_safety_max_ticks > 0 and tick_id >= self.config.match_safety_max_ticks:
                self.logger.info("runtime_finished reason=match_safety_max_ticks ticks=%s", tick_id)
                return 0

            want_match_end_pixels = (
                end_tracking
                and self.config.capture_enabled
                and (tick_id % self.config.match_end_check_every_n_ticks == 0)
            )
            want_elixir_pixels = self.config.elixir_model_enabled and self.config.capture_enabled
            want_pixels = want_match_end_pixels or want_elixir_pixels

            match_ended = self._run_tick(
                tick_id=tick_id,
                loop_start=start,
                zone_map=zone_map,
                gate=gate,
                capture=capture,
                actuator=actuator,
                match_ready=match_ready,
                include_pixels=want_pixels,
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
        include_pixels: bool,
        include_match_end_pixels: bool,
        end_tracker: MatchExitTracker | None,
    ) -> bool:
        """
        Run one policy/actuation tick. Returns True when CNN match-end criteria are satisfied.
        """
        tick_start = time.perf_counter()
        now_ms = int((tick_start - loop_start) * 1000)
        frame = frame_for_tick(
            capture,
            self.config.capture_enabled,
            tick_id,
            include_pixels=include_pixels,
        )

        elixir_estimate = min(10.0, tick_id * 0.5)
        elixir_conf = 0.0
        elixir_source = "synthetic"
        if (
            self.config.elixir_model_enabled
            and frame.pixels_bgra
            and self.config.elixir_model_path
            and frame.width > 0
            and frame.height > 0
        ):
            inferred_elixir, inferred_conf = infer_elixir_value(
                frame_width=frame.width,
                frame_height=frame.height,
                pixels_bgra=frame.pixels_bgra,
                model_path=self.config.elixir_model_path,
                model_layout_path=self.config.elixir_model_layout_path,
                logger=self.logger,
            )
            if inferred_conf > 0.0:
                elixir_estimate = max(0.0, min(10.0, inferred_elixir))
                elixir_conf = inferred_conf
                elixir_source = "cnn"

        hand_cards: list[str] = ["unknown", "unknown", "unknown", "unknown"]
        hand_card_confidences: list[float] = [0.0, 0.0, 0.0, 0.0]
        if (
            self.config.card_model_enabled
            and frame.pixels_bgra
            and self.config.card_model_path
            and frame.width > 0
            and frame.height > 0
        ):
            inferred = infer_hand_cards(
                frame_width=frame.width,
                frame_height=frame.height,
                pixels_bgra=frame.pixels_bgra,
                model_path=self.config.card_model_path,
                model_layout_path=self.config.card_model_layout_path,
                logger=self.logger,
            )
            if len(inferred) == 4:
                hand_cards = [name for name, _ in inferred]
                hand_card_confidences = [conf for _, conf in inferred]
        self._write_hand_tick_log(tick_id, hand_cards, hand_card_confidences)

        state = RuntimeState(
            tick_id=tick_id,
            timestamp_ms=now_ms,
            elixir=elixir_estimate,
        )
        candidate = propose_candidate_action(
            state=state,
            frame=frame,
            hand_cards=hand_cards,
            hand_confidences=hand_card_confidences,
        )
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
            "tick=%s ts_ms=%s elixir=%.1f elixir_src=%s elixir_conf=%.3f frame=%sx%s capture_ms=%s "
            "candidate=%s confidence=%s hand=%s hand_conf=%s decision=%s reason=%s card=%s zone=%s",
            state.tick_id,
            state.timestamp_ms,
            state.elixir,
            elixir_source,
            elixir_conf,
            frame.width,
            frame.height,
            frame.capture_latency_ms,
            candidate.card_name if candidate else None,
            round(candidate.confidence, 3) if candidate else None,
            ",".join(hand_cards),
            ",".join(f"{x:.2f}" for x in hand_card_confidences),
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

        match_ended = False
        if (
            end_tracker is not None
            and include_match_end_pixels
            and frame.pixels_bgra
            and self.config.battlefield_model_path
            and frame.width > 0
            and frame.height > 0
        ):
            prob = infer_battlefield_probability(
                frame_width=frame.width,
                frame_height=frame.height,
                pixels_bgra=frame.pixels_bgra,
                viewport=self.config.game_viewport,
                model_path=self.config.battlefield_model_path,
                model_layout_path=self.config.battlefield_model_layout_path,
                logger=self.logger,
            )
            match_ended = end_tracker.observe_probability(
                prob=prob,
                end_threshold=self.config.battlefield_end_score_threshold,
                confirm_ticks=self.config.match_end_confirm_ticks,
                did_check=True,
            )
            self.logger.debug(
                "match_end_probe tick=%s prob=%.3f end_threshold=%s streak=%s",
                tick_id,
                prob,
                self.config.battlefield_end_score_threshold,
                end_tracker.consecutive_below,
            )

        elapsed_ms = int((time.perf_counter() - tick_start) * 1000)
        sleep_ms = max(0, self.config.tick_interval_ms - elapsed_ms)
        time.sleep(sleep_ms / 1000.0)
        return match_ended
