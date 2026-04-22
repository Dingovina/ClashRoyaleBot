from __future__ import annotations

from dataclasses import dataclass

from src.runtime.config import RuntimeConfig
from src.runtime.types import ActionDecision, CandidateAction, RuntimeState
from src.runtime.zones import ZoneMap


@dataclass
class PolicyGate:
    config: RuntimeConfig
    zone_map: ZoneMap
    last_action_timestamp_ms: int | None = None

    def decide(self, state: RuntimeState, candidate: CandidateAction | None) -> ActionDecision:
        if candidate is None:
            return ActionDecision(action_type="no_op", reason="no_candidate")

        if self._is_rate_limited(state.timestamp_ms):
            return ActionDecision(action_type="no_op", reason="rate_limited")

        if candidate.confidence < self.config.no_op_confidence_threshold:
            return ActionDecision(action_type="no_op", reason="confidence_below_no_op_threshold")

        if candidate.confidence < self.config.action_confidence_threshold and not candidate.urgent_defense:
            return ActionDecision(action_type="no_op", reason="confidence_uncertainty_band")

        if state.elixir < self.config.min_elixir_for_non_urgent_action and not candidate.urgent_defense:
            return ActionDecision(action_type="no_op", reason="low_elixir_non_urgent")

        if not self.zone_map.is_zone_valid(candidate.zone_id, candidate.card_class):
            return ActionDecision(action_type="no_op", reason="illegal_zone_for_card_type")

        if candidate.card_index not in {1, 2, 3, 4}:
            return ActionDecision(action_type="no_op", reason="illegal_card_index")

        self.last_action_timestamp_ms = state.timestamp_ms
        return ActionDecision(
            action_type="deploy",
            reason="accepted",
            card_index=candidate.card_index,
            zone_id=candidate.zone_id,
        )

    def _is_rate_limited(self, now_ms: int) -> bool:
        if self.last_action_timestamp_ms is None:
            return False
        return (now_ms - self.last_action_timestamp_ms) < self.config.action_rate_limit_ms
