from __future__ import annotations

import logging
from pathlib import Path


def infer_elixir_value(
    *,
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes | None,
    model_path: str,
    model_layout_path: str,
    logger: logging.Logger,
) -> tuple[float, float]:
    """
    Return `(elixir_value, confidence)` predicted from the `elixir_number` ROI.
    On failure or missing pixels returns `(0.0, 0.0)`.
    """
    if not pixels_bgra:
        return (0.0, 0.0)
    try:
        from src.perception.infer.elixir_infer import get_elixir_runner

        runner = get_elixir_runner(Path(model_path), Path(model_layout_path), logger)
        return runner.infer_elixir(frame_width, frame_height, pixels_bgra)
    except Exception as exc:
        logger.warning("elixir_infer_failed err=%s", exc)
        return (0.0, 0.0)
