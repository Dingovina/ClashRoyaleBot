from __future__ import annotations

from dataclasses import dataclass

from src.runtime.config.runtime_config import RuntimeConfig
from src.runtime.domain.types import ActionDecision, ActionType, CandidateAction, DecisionReason, RuntimeState
from src.runtime.domain.zones import ZoneMap


@dataclass
class PolicyGate:
    config: RuntimeConfig
    zone_map: ZoneMap
    last_action_timestamp_ms: int | None = None

    def decide(self, state: RuntimeState, candidate: CandidateAction | None) -> ActionDecision:
        if candidate is None:
            return ActionDecision(action_type=ActionType.NO_OP, reason=DecisionReason.NO_CANDIDATE)

        if self._is_rate_limited(state.timestamp_ms):
            return ActionDecision(action_type=ActionType.NO_OP, reason=DecisionReason.RATE_LIMITED)

        if candidate.confidence < self.config.no_op_confidence_threshold:
            return ActionDecision(
                action_type=ActionType.NO_OP,
                reason=DecisionReason.CONFIDENCE_BELOW_NO_OP_THRESHOLD,
            )

        if candidate.confidence < self.config.action_confidence_threshold and not candidate.urgent_defense:
            return ActionDecision(
                action_type=ActionType.NO_OP,
                reason=DecisionReason.CONFIDENCE_UNCERTAINTY_BAND,
            )

        if state.elixir < self.config.min_elixir_for_non_urgent_action and not candidate.urgent_defense:
            return ActionDecision(
                action_type=ActionType.NO_OP,
                reason=DecisionReason.LOW_ELIXIR_NON_URGENT,
            )

        if not self.zone_map.is_zone_valid(candidate.zone_id, candidate.card_class):
            return ActionDecision(
                action_type=ActionType.NO_OP,
                reason=DecisionReason.ILLEGAL_ZONE_FOR_CARD_TYPE,
            )

        if candidate.card_index not in {1, 2, 3, 4}:
            return ActionDecision(action_type=ActionType.NO_OP, reason=DecisionReason.ILLEGAL_CARD_INDEX)

        raw_name = candidate.card_name.lower().strip()
        canonical_name = self.config.card_name_aliases.get(raw_name, raw_name)
        card_cost = self.config.card_elixir_costs.get(canonical_name)
        if card_cost is None:
            return ActionDecision(
                action_type=ActionType.NO_OP,
                reason=DecisionReason.UNKNOWN_CARD_ELIXIR_COST,
            )
        if state.elixir < card_cost:
            return ActionDecision(
                action_type=ActionType.NO_OP,
                reason=DecisionReason.INSUFFICIENT_ELIXIR_FOR_CARD,
            )

        self.last_action_timestamp_ms = state.timestamp_ms
        return ActionDecision(
            action_type=ActionType.DEPLOY,
            reason=DecisionReason.ACCEPTED,
            card_index=candidate.card_index,
            zone_id=candidate.zone_id,
        )

    def _is_rate_limited(self, now_ms: int) -> bool:
        if self.last_action_timestamp_ms is None:
            return False
        return (now_ms - self.last_action_timestamp_ms) < self.config.action_rate_limit_ms
