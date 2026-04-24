from __future__ import annotations

from src.runtime.capture import FrameObservation
from src.runtime.types import CandidateAction, RuntimeState


def _card_class(card_name: str) -> str:
    return "spell" if card_name in {"fireball", "arrows", "zap"} else "unit"


def propose_candidate_action(
    state: RuntimeState,
    frame: FrameObservation,
    *,
    hand_cards: list[str] | None = None,
    hand_confidences: list[float] | None = None,
) -> CandidateAction:
    # Prefer detected hand cards when available; otherwise keep deterministic scripted fallback.
    card_index = (state.tick_id % 4) + 1
    card_name = ["knight", "archers", "fireball", "giant"][state.tick_id % 4]
    confidence = 0.45 + ((state.tick_id % 7) * 0.08)
    if hand_cards and hand_confidences and len(hand_cards) == 4 and len(hand_confidences) == 4:
        choices: list[tuple[int, str, float]] = []
        for idx, (name, conf) in enumerate(zip(hand_cards, hand_confidences), start=1):
            n = name.strip().lower()
            if n and n != "unknown" and n != "empty":
                choices.append((idx, n, conf))
        if choices:
            best_idx, best_name, best_conf = max(choices, key=lambda x: x[2])
            card_index = best_idx
            card_name = best_name
            confidence = max(confidence, min(0.99, best_conf))

    card_class = _card_class(card_name)
    zone_id = state.tick_id % 12
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
