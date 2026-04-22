from __future__ import annotations

import logging
from pathlib import Path

from src.perception.battlefield_heuristic import score_battlefield_heuristic_bgra
from src.runtime.battlefield_config import BattlefieldDetectorConfig
from src.runtime.viewport import GameViewport, crop_playfield_bgra


def evaluate_battlefield(
    *,
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes | None,
    viewport: GameViewport,
    detector: BattlefieldDetectorConfig,
    logger: logging.Logger,
) -> tuple[bool, float]:
    """
    Returns (is_match_ready, score). When pixels are missing, returns (False, 0.0).

    ``method`` is ``heuristic`` (anchor ROI colors), ``model`` (CNN on masked ``bottom_panel``
    from ``model_layout_path``, with hand slots / next card / elixir zeroed), or ``blend``
    (average of heuristic score and model probability).
    """
    if not pixels_bgra:
        return (False, 0.0)

    method = detector.method.lower().strip()
    if method not in ("heuristic", "model", "blend"):
        logger.warning("battlefield_unknown_method method=%s using_heuristic", detector.method)
        method = "heuristic"

    try:
        if method == "heuristic":
            score = _heuristic_score_full_frame(
                frame_width, frame_height, pixels_bgra, viewport, detector, logger
            )
            return (score >= detector.score_threshold, score)

        if method == "model":
            prob = _model_probability(frame_width, frame_height, pixels_bgra, viewport, detector, logger)
            logger.debug("battlefield_model_score prob=%.4f", prob)
            return (prob >= detector.score_threshold, prob)

        h_score = _heuristic_score_full_frame(
            frame_width, frame_height, pixels_bgra, viewport, detector, logger
        )
        prob = _model_probability(frame_width, frame_height, pixels_bgra, viewport, detector, logger)
        score = 0.5 * h_score + 0.5 * prob
        logger.debug("battlefield_blend heuristic=%.3f model=%.3f combined=%.3f", h_score, prob, score)
        return (score >= detector.score_threshold, score)
    except Exception as exc:
        logger.warning("battlefield_detector_failed err=%s", exc)
        return (False, 0.0)


def _heuristic_score_full_frame(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    viewport: GameViewport,
    detector: BattlefieldDetectorConfig,
    logger: logging.Logger,
) -> float:
    rw, rh, roi = crop_playfield_bgra(frame_width, frame_height, pixels_bgra, viewport)
    if rw <= 0 or rh <= 0:
        return 0.0
    return score_battlefield_heuristic_bgra(
        rw,
        rh,
        roi,
        sample_stride=detector.sample_stride,
        river_top_ratio=detector.river_band_top_ratio,
        river_bottom_ratio=detector.river_band_bottom_ratio,
        grass_top_ratio=detector.grass_band_top_ratio,
        grass_bottom_ratio=detector.grass_band_bottom_ratio,
        logger=logger,
    )


def _model_probability(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    viewport: GameViewport,
    detector: BattlefieldDetectorConfig,
    logger: logging.Logger,
) -> float:
    from src.perception.battlefield_infer import get_battlefield_runner

    if not detector.model_path or not detector.model_layout_path:
        return 0.0
    _ = viewport  # CNN uses fullscreen layout YAML, not game_viewport geometry.
    runner = get_battlefield_runner(
        Path(detector.model_path),
        Path(detector.model_layout_path),
        logger,
    )
    return runner.probability_battlefield(frame_width, frame_height, pixels_bgra)
