from __future__ import annotations

import logging
from pathlib import Path

from src.runtime.config.battlefield_config import BattlefieldModelConfig


def evaluate_battlefield(
    *,
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes | None,
    detector: BattlefieldModelConfig,
    logger: logging.Logger,
) -> tuple[bool, float]:
    """
    Returns (is_match_ready, probability). When pixels are missing, returns (False, 0.0).

    Uses the tiny CNN on the masked ``bottom_panel`` from ``model_layout_path`` (HUD slots zeroed).
    """
    if not pixels_bgra:
        return (False, 0.0)

    try:
        prob = _model_probability(frame_width, frame_height, pixels_bgra, detector, logger)
        logger.debug("battlefield_model_score prob=%.4f", prob)
        return (prob >= detector.score_threshold, prob)
    except Exception as exc:
        logger.warning("battlefield_model_failed err=%s", exc)
        return (False, 0.0)


def infer_battlefield_probability(
    *,
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes | None,
    model_path: str,
    model_layout_path: str,
    logger: logging.Logger,
) -> float:
    """
    Raw CNN probability of battlefield / in-match screen. On failure returns 1.0 so the runtime
    does not treat transient errors as match end.
    """
    if not pixels_bgra:
        return 0.0
    detector = BattlefieldModelConfig(
        score_threshold=1.0,
        model_path=model_path,
        model_layout_path=model_layout_path,
    )
    try:
        return _model_probability(frame_width, frame_height, pixels_bgra, detector, logger)
    except Exception as exc:
        logger.warning("battlefield_probability_infer_failed err=%s", exc)
        return 1.0


def _model_probability(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    detector: BattlefieldModelConfig,
    logger: logging.Logger,
) -> float:
    from src.perception.infer.battlefield_infer import get_battlefield_runner

    runner = get_battlefield_runner(
        Path(detector.model_path),
        Path(detector.model_layout_path),
        logger,
    )
    return runner.probability_battlefield(frame_width, frame_height, pixels_bgra)
