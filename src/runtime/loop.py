from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.runtime.config import RuntimeConfig
from src.runtime.policy_gate import PolicyGate
from src.runtime.types import CandidateAction, RuntimeState
from src.runtime.zones import build_default_zone_map


@dataclass
class RuntimeLoop:
    config: RuntimeConfig
    logger: logging.Logger

    def run(self) -> None:
        zone_map = build_default_zone_map(self.config.zones)
        gate = PolicyGate(config=self.config, zone_map=zone_map)

        self.logger.info(
            "runtime_started tick_interval_ms=%s action_rate_limit_ms=%s max_ticks=%s",
            self.config.tick_interval_ms,
            self.config.action_rate_limit_ms,
            self.config.max_ticks,
        )

        start = time.perf_counter()
        for tick_id in range(self.config.max_ticks):
            tick_start = time.perf_counter()
            now_ms = int((tick_start - start) * 1000)

            state = RuntimeState(
                tick_id=tick_id,
                timestamp_ms=now_ms,
                elixir=min(10.0, tick_id * 0.5),
            )
            candidate = _mock_candidate_action(tick_id)
            decision = gate.decide(state=state, candidate=candidate)

            self.logger.info(
                "tick=%s ts_ms=%s elixir=%.1f candidate=%s confidence=%s decision=%s reason=%s card=%s zone=%s",
                state.tick_id,
                state.timestamp_ms,
                state.elixir,
                candidate.card_name if candidate else None,
                round(candidate.confidence, 3) if candidate else None,
                decision.action_type,
                decision.reason,
                decision.card_index,
                decision.zone_id,
            )

            elapsed_ms = int((time.perf_counter() - tick_start) * 1000)
            sleep_ms = max(0, self.config.tick_interval_ms - elapsed_ms)
            time.sleep(sleep_ms / 1000)

        self.logger.info("runtime_finished ticks=%s", self.config.max_ticks)


def _mock_candidate_action(tick_id: int) -> CandidateAction:
    card_index = (tick_id % 4) + 1
    card_name = ["knight", "archers", "fireball", "giant"][tick_id % 4]
    card_class = "spell" if card_name == "fireball" else "unit"
    zone_id = tick_id % 12
    confidence = 0.45 + ((tick_id % 7) * 0.08)
    urgent = tick_id % 9 == 0

    return CandidateAction(
        card_index=card_index,
        card_name=card_name,
        card_class=card_class,
        zone_id=zone_id,
        confidence=min(confidence, 0.98),
        urgent_defense=urgent,
    )
