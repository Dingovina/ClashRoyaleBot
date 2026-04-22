from __future__ import annotations

import logging
from collections.abc import Callable


def _is_water_bgra(b: int, g: int, r: int) -> bool:
    """Heuristic: Clash river tiles are blue-dominant."""
    return b >= 95 and b >= r + 18 and b >= g + 12


def _is_grass_bgra(b: int, g: int, r: int) -> bool:
    """Heuristic: arena turf skews green vs red/blue."""
    return g >= 72 and g >= r + 10 and g >= b + 8


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
