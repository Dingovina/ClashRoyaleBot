from __future__ import annotations

import logging
from pathlib import Path


def infer_tower_hp_value(
    *,
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes | None,
    tower_type: str,
    model_path: str,
    model_layout_path: str,
    logger: logging.Logger,
) -> tuple[str, float]:
    """
    Return ``(hp_text, confidence)`` for one tower HP ROI.

    - ``hp_text`` is either digits (for example ``"2406"``) or ``"none"`` when no number is present.
    - On failure or missing pixels returns ``("none", 0.0)``.
    """
    if not pixels_bgra:
        return ("none", 0.0)
    try:
        from src.perception.infer.tower_hp_infer import get_tower_hp_runner

        runner = get_tower_hp_runner(Path(model_path), Path(model_layout_path), logger)
        return runner.infer_tower_hp(frame_width, frame_height, pixels_bgra, tower_type)
    except Exception as exc:
        logger.warning("tower_hp_infer_failed tower_type=%s err=%s", tower_type, exc)
        return ("none", 0.0)

