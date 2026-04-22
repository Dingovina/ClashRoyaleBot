from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ActionType = Literal["deploy", "no_op"]
CardClass = Literal["unit", "spell"]


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
    reason: str
    card_index: int | None = None
    zone_id: int | None = None


@dataclass(frozen=True)
class RuntimeState:
    tick_id: int
    timestamp_ms: int
    elixir: float
