from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ActionType(StrEnum):
    DEPLOY = "deploy"
    NO_OP = "no_op"


class CardClass(StrEnum):
    UNIT = "unit"
    SPELL = "spell"


class DecisionReason(StrEnum):
    ACCEPTED = "accepted"
    NO_CANDIDATE = "no_candidate"
    RATE_LIMITED = "rate_limited"
    CONFIDENCE_BELOW_NO_OP_THRESHOLD = "confidence_below_no_op_threshold"
    CONFIDENCE_UNCERTAINTY_BAND = "confidence_uncertainty_band"
    LOW_ELIXIR_NON_URGENT = "low_elixir_non_urgent"
    ILLEGAL_ZONE_FOR_CARD_TYPE = "illegal_zone_for_card_type"
    ILLEGAL_CARD_INDEX = "illegal_card_index"
    UNKNOWN_CARD_ELIXIR_COST = "unknown_card_elixir_cost"
    INSUFFICIENT_ELIXIR_FOR_CARD = "insufficient_elixir_for_card"
    MATCH_READINESS_NOT_READY = "match_readiness_not_ready"


class InferenceStatus(StrEnum):
    OK = "ok"
    NO_PIXELS = "no_pixels"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass(frozen=True)
class CandidateAction:
    card_index: int
    card_name: str
    card_class: CardClass
    zone_id: int
    confidence: float
    urgent_defense: bool = False


@dataclass(frozen=True)
class ActionDecision:
    action_type: ActionType
    reason: DecisionReason
    card_index: int | None = None
    zone_id: int | None = None


@dataclass(frozen=True)
class RuntimeState:
    tick_id: int
    timestamp_ms: int
    elixir: float


@dataclass(frozen=True)
class PerceptionSnapshot:
    elixir: float
    elixir_confidence: float
    elixir_status: InferenceStatus
    hand_cards: tuple[str, str, str, str]
    hand_confidences: tuple[float, float, float, float]
    hand_status: InferenceStatus
