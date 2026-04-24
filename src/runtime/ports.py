from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.runtime.capture import FrameObservation
from src.runtime.types import ActionDecision, PerceptionSnapshot
from src.runtime.zones import ZoneMap


class FrameSource(Protocol):
    def frame_for_tick(self, tick_id: int, include_pixels: bool) -> FrameObservation: ...


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
