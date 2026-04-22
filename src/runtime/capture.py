from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameObservation:
    width: int
    height: int
    capture_latency_ms: int
    source: str
    screenshot_path: str | None = None


class FullscreenCapture:
    def __init__(self, logger: logging.Logger, debug_dir: str | None, capture_every_n_ticks: int) -> None:
        self.logger = logger
        self.capture_every_n_ticks = max(0, capture_every_n_ticks)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, tick_id: int) -> FrameObservation:
        start = time.perf_counter()
        try:
            import mss
            import mss.tools
        except ModuleNotFoundError:
            latency = int((time.perf_counter() - start) * 1000)
            self.logger.warning("capture_unavailable reason=mss_not_installed")
            return FrameObservation(
                width=0,
                height=0,
                capture_latency_ms=latency,
                source="unavailable",
                screenshot_path=None,
            )

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            screenshot_path = self._maybe_save_debug_screenshot(
                tick_id=tick_id,
                rgb=shot.rgb,
                size=(shot.width, shot.height),
            )

        latency = int((time.perf_counter() - start) * 1000)
        return FrameObservation(
            width=shot.width,
            height=shot.height,
            capture_latency_ms=latency,
            source="fullscreen",
            screenshot_path=screenshot_path,
        )

    def _maybe_save_debug_screenshot(
        self,
        tick_id: int,
        rgb: bytes,
        size: tuple[int, int],
    ) -> str | None:
        if not self.debug_dir or self.capture_every_n_ticks <= 0:
            return None
        if tick_id % self.capture_every_n_ticks != 0:
            return None

        import mss.tools

        file_path = self.debug_dir / f"frame_{tick_id:05d}.png"
        mss.tools.to_png(rgb, size, output=str(file_path))
        return str(file_path)
