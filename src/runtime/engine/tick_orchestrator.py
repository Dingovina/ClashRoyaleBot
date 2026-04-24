from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.runtime.evaluation.battlefield_evaluate import infer_battlefield_probability
from src.runtime.domain.candidate_policy import propose_candidate_action
from src.runtime.infra.capture import FullscreenCapture, frame_for_tick
from src.runtime.domain.match_exit import MatchExitTracker
from src.runtime.domain.policy_gate import PolicyGate
from src.runtime.domain.ports import ActuatorPort, EventSink, PerceptionService, TickEvent
from src.runtime.config.runtime_config import RuntimeConfig
from src.runtime.domain.types import ActionDecision, ActionType, DecisionReason, RuntimeState
from src.runtime.domain.zones import ZoneMap


@dataclass
class TickOrchestrator:
    config: RuntimeConfig
    logger: logging.Logger
    zone_map: ZoneMap
    capture: FullscreenCapture
    perception_service: PerceptionService
    gate: PolicyGate
    actuator: ActuatorPort
    event_sink: EventSink

    def run_tick(
        self,
        tick_id: int,
        loop_start: float,
        *,
        match_ready: bool,
        include_pixels: bool,
        include_match_end_pixels: bool,
        end_tracker: MatchExitTracker | None,
    ) -> bool:
        tick_start = time.perf_counter()
        now_ms = int((tick_start - loop_start) * 1000)
        frame = frame_for_tick(
            self.capture,
            tick_id,
            include_pixels=include_pixels,
        )

        perception = self.perception_service.infer(frame)
        elixir_estimate = perception.elixir if perception.elixir_status == "ok" else min(10.0, tick_id * 0.5)

        state = RuntimeState(
            tick_id=tick_id,
            timestamp_ms=now_ms,
            elixir=elixir_estimate,
        )
        candidate = propose_candidate_action(state=state, perception=perception)
        decision = self.gate.decide(state=state, candidate=candidate)
        if not match_ready:
            decision = ActionDecision(
                action_type=ActionType.NO_OP,
                reason=DecisionReason.MATCH_READINESS_NOT_READY,
                card_index=decision.card_index,
                zone_id=decision.zone_id,
            )
        execution = self.actuator.execute(
            decision=decision,
            zone_map=self.zone_map,
            frame_width=frame.width,
            frame_height=frame.height,
        )
        self.event_sink.publish_tick(
            TickEvent(
                tick_id=tick_id,
                decision=decision,
                candidate_name=candidate.card_name if candidate else None,
                candidate_confidence=round(candidate.confidence, 3) if candidate else None,
                perception=perception,
            )
        )

        self.logger.info(
            "tick=%s ts_ms=%s elixir=%.1f elixir_status=%s elixir_conf=%.3f frame=%sx%s capture_ms=%s "
            "candidate=%s confidence=%s hand=%s hand_conf=%s decision=%s reason=%s card=%s zone=%s",
            state.tick_id,
            state.timestamp_ms,
            state.elixir,
            perception.elixir_status.value,
            perception.elixir_confidence,
            frame.width,
            frame.height,
            frame.capture_latency_ms,
            candidate.card_name if candidate else None,
            round(candidate.confidence, 3) if candidate else None,
            ",".join(perception.hand_cards),
            ",".join(f"{x:.2f}" for x in perception.hand_confidences),
            decision.action_type.value,
            decision.reason.value,
            decision.card_index,
            decision.zone_id,
        )
        self.logger.info(
            "action_attempt tick=%s decision=%s executed=%s exec_reason=%s click_x=%s click_y=%s",
            state.tick_id,
            decision.action_type.value,
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

        elapsed_ms = int((time.perf_counter() - tick_start) * 1000)
        sleep_ms = max(0, self.config.tick_interval_ms - elapsed_ms)
        time.sleep(sleep_ms / 1000.0)
        return match_ended
