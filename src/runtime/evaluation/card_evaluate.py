from __future__ import annotations

import logging
from pathlib import Path


def infer_hand_cards(
    *,
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes | None,
    model_path: str,
    model_layout_path: str,
    logger: logging.Logger,
) -> list[tuple[str, float]]:
    """
    Return per-slot predictions ``[(card_name, confidence), ...]`` for 4 hand slots.
    On failure or missing pixels returns an empty list.
    """
    if not pixels_bgra:
        return []
    try:
        from src.perception.infer.card_infer import get_card_runner

        runner = get_card_runner(Path(model_path), Path(model_layout_path), logger)
        return runner.infer_hand_cards(frame_width, frame_height, pixels_bgra)
    except Exception as exc:
        logger.warning("card_infer_failed err=%s", exc)
        return []
