from __future__ import annotations

import logging
import sys
import time

from src.runtime.config.battlefield_config import BattlefieldModelConfig
from src.runtime.evaluation.battlefield_evaluate import evaluate_battlefield
from src.runtime.infra.capture import FullscreenCapture
from src.runtime.infra.foreground_win import foreground_matches, foreground_title_lower
from src.runtime.config.runtime_config import RuntimeConfig


def wait_for_match_readiness(
    config: RuntimeConfig,
    logger: logging.Logger,
    capture: FullscreenCapture,
) -> tuple[int, bool]:
    """
    Optionally block until the battlefield CNN reports a match-ready frame.

    Returns ``(exit_code, match_ready)``. ``exit_code`` is non-zero when configured to exit on timeout.
    """
    if not config.match_readiness_enabled:
        return (0, True)

    if not config.battlefield_model_path:
        raise RuntimeError("match_readiness_enabled but battlefield_model_path is unset")

    detector = BattlefieldModelConfig(
        score_threshold=config.battlefield_score_threshold,
        model_path=config.battlefield_model_path,
        model_layout_path=config.battlefield_model_layout_path,
    )

    if config.battlefield_wait_timeout_ms > 0:
        deadline = time.perf_counter() + config.battlefield_wait_timeout_ms / 1000.0
    else:
        deadline = None

    wait_tick = 0
    logged_foreground_skip = False
    while True:
        if deadline is not None and time.perf_counter() > deadline:
            logger.info(
                "battlefield_wait_timeout behavior=%s timeout_ms=%s",
                config.battlefield_timeout_behavior,
                config.battlefield_wait_timeout_ms,
            )
            if config.battlefield_timeout_behavior == "exit_nonzero":
                return (2, False)
            logger.info("battlefield_timeout_continue actuation_blocked=1")
            return (0, False)

        frame = capture.capture(wait_tick, include_pixels=True)
        wait_tick += 1

        if frame.width <= 0 or frame.source != "fullscreen" or not frame.pixels_bgra:
            logger.info("waiting_for_battlefield reason=capture_unavailable source=%s", frame.source)
            time.sleep(config.tick_interval_ms / 1000.0)
            continue

        if config.foreground_check_enabled:
            if sys.platform != "win32":
                if not logged_foreground_skip:
                    logger.info("foreground_check_skipped reason=non_windows_platform")
                    logged_foreground_skip = True
            else:
                title = foreground_title_lower()
                if not foreground_matches(title or "", config.foreground_title_substrings):
                    logger.info(
                        "waiting_for_battlefield reason=foreground_mismatch title=%r",
                        title or "",
                    )
                    time.sleep(config.tick_interval_ms / 1000.0)
                    continue

        ready, score = evaluate_battlefield(
            frame_width=frame.width,
            frame_height=frame.height,
            pixels_bgra=frame.pixels_bgra,
            detector=detector,
            logger=logger,
        )
        if ready:
            logger.info(
                "battlefield_detected score=%.3f threshold=%.3f",
                score,
                config.battlefield_score_threshold,
            )
            return (0, True)

        logger.info(
            "waiting_for_battlefield score=%.3f threshold=%.3f",
            score,
            config.battlefield_score_threshold,
        )
        time.sleep(config.tick_interval_ms / 1000.0)
