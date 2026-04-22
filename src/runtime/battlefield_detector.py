from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.runtime.viewport import GameViewport


def _is_water_bgra(b: int, g: int, r: int) -> bool:
    """Heuristic: Clash river tiles are blue-dominant."""
    return b >= 95 and b >= r + 18 and b >= g + 12


def _is_grass_bgra(b: int, g: int, r: int) -> bool:
    """Heuristic: arena turf skews green vs red/blue."""
    return g >= 72 and g >= r + 10 and g >= b + 8


def crop_playfield_bgra(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    viewport: GameViewport,
) -> tuple[int, int, bytes]:
    """Crop the configured playfield (viewport × anchor_rect) from a fullscreen BGRA capture."""
    if frame_width <= 0 or frame_height <= 0:
        return (0, 0, b"")
    if len(pixels_bgra) != frame_width * frame_height * 4:
        raise ValueError("pixels_bgra size does not match frame dimensions")

    left, top, vw, vh = viewport.rect_for_frame(frame_width, frame_height)
    rect = viewport.anchor_rect
    x0 = left + int(rect.left_ratio * vw)
    y0 = top + int(rect.top_ratio * vh)
    roi_w = max(0, int(rect.width_ratio * vw))
    roi_h = max(0, int(rect.height_ratio * vh))

    x0 = max(0, min(x0, frame_width - 1))
    y0 = max(0, min(y0, frame_height - 1))
    roi_w = min(roi_w, frame_width - x0)
    roi_h = min(roi_h, frame_height - y0)
    if roi_w <= 0 or roi_h <= 0:
        return (0, 0, b"")

    row_stride = frame_width * 4
    out = bytearray(roi_w * roi_h * 4)
    out_row = 0
    for y in range(y0, y0 + roi_h):
        src = y * row_stride + x0 * 4
        out[out_row : out_row + roi_w * 4] = pixels_bgra[src : src + roi_w * 4]
        out_row += roi_w * 4
    return (roi_w, roi_h, bytes(out))


def score_battlefield_heuristic_bgra(
    roi_width: int,
    roi_height: int,
    roi_bgra: bytes,
    *,
    sample_stride: int,
    river_top_ratio: float,
    river_bottom_ratio: float,
    grass_top_ratio: float,
    grass_bottom_ratio: float,
    logger: logging.Logger | None = None,
) -> float:
    """
    Return a soft score in ~[0, 1]. Higher means more likely an in-match arena view.

    Uses two horizontal bands inside the ROI: a central band (river) and a lower band (turf).
    """
    if roi_width <= 0 or roi_height <= 0 or not roi_bgra:
        return 0.0
    if len(roi_bgra) != roi_width * roi_height * 4:
        raise ValueError("roi_bgra size does not match ROI dimensions")

    stride = max(1, sample_stride)
    ry0 = int(roi_height * river_top_ratio)
    ry1 = int(roi_height * river_bottom_ratio)
    gy0 = int(roi_height * grass_top_ratio)
    gy1 = int(roi_height * grass_bottom_ratio)
    ry0, ry1 = max(0, min(ry0, roi_height)), max(0, min(ry1, roi_height))
    gy0, gy1 = max(0, min(gy0, roi_height)), max(0, min(gy1, roi_height))
    if ry1 <= ry0 or gy1 <= gy0:
        return 0.0

    row = roi_width * 4

    def band_score(y0: int, y1: int, predicate: Callable[[int, int, int], bool]) -> float:
        hit = 0
        total = 0
        for y in range(y0, y1, stride):
            base = y * row
            for x in range(0, roi_width, stride):
                i = base + x * 4
                b, g, r, _a = roi_bgra[i], roi_bgra[i + 1], roi_bgra[i + 2], roi_bgra[i + 3]
                total += 1
                if predicate(b, g, r):
                    hit += 1
        return hit / total if total else 0.0

    river = band_score(ry0, ry1, _is_water_bgra)
    grass = band_score(gy0, gy1, _is_grass_bgra)
    score = min(1.0, river * 1.15) * 0.55 + min(1.0, grass * 1.15) * 0.45
    if logger:
        logger.debug(
            "battlefield_heuristic_detail river=%.3f grass=%.3f combined=%.3f",
            river,
            grass,
            score,
        )
    return float(score)


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

    if not detector.model_path:
        return 0.0
    path = Path(detector.model_path)
    runner = get_battlefield_runner(path, logger)
    return runner.probability_battlefield(frame_width, frame_height, pixels_bgra, viewport)


@dataclass(frozen=True)
class BattlefieldDetectorConfig:
    method: str
    score_threshold: float
    sample_stride: int
    river_band_top_ratio: float
    river_band_bottom_ratio: float
    grass_band_top_ratio: float
    grass_band_bottom_ratio: float
    model_path: str | None
    """Square side length used when training; checkpoint also stores this."""
    model_input_size: int


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

    ``method`` is ``heuristic`` (anchor ROI colors), ``model`` (CNN on viewport crop),
    or ``blend`` (average of heuristic score and model probability).
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

        # blend
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
