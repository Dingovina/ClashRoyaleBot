from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.runtime.infra.capture import FrameObservation
from src.runtime.domain.types import ActionDecision, PerceptionSnapshot
from src.runtime.domain.zones import ZoneMap


class PerceptionService(Protocol):
    def infer(self, frame: FrameObservation) -> PerceptionSnapshot: ...


class ActuatorPort(Protocol):
    def execute(self, decision: ActionDecision, zone_map: ZoneMap, frame_width: int, frame_height: int): ...


@dataclass(frozen=True)
class TickEvent:
    tick_id: int
    decision: ActionDecision
    candidate_name: str | None
    candidate_confidence: float | None
    perception: PerceptionSnapshot


class EventSink(Protocol):
    def publish_tick(self, event: TickEvent) -> None: ...
