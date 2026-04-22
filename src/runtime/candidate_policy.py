from __future__ import annotations

from src.runtime.capture import FrameObservation
from src.runtime.types import CandidateAction, RuntimeState


def propose_candidate_action(state: RuntimeState, frame: FrameObservation) -> CandidateAction:
    # Sprint 1 keeps a deterministic scripted policy while plumbing real capture + actuation.
    card_index = (state.tick_id % 4) + 1
    card_name = ["knight", "archers", "fireball", "giant"][state.tick_id % 4]
    card_class = "spell" if card_name == "fireball" else "unit"
    zone_id = state.tick_id % 12
    confidence = 0.45 + ((state.tick_id % 7) * 0.08)
    urgent = state.tick_id % 9 == 0

    # Tiny capture-aware nudge to show frame signal can influence confidence.
    if frame.width > 0 and frame.height > 0:
        confidence += 0.01

    return CandidateAction(
        card_index=card_index,
        card_name=card_name,
        card_class=card_class,
        zone_id=zone_id,
        confidence=min(confidence, 0.98),
        urgent_defense=urgent,
    )
