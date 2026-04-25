from __future__ import annotations

import logging
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameObservation:
    width: int
    height: int
    capture_latency_ms: int
    source: str
    screenshot_path: str | None = None
    # Raw BGRA from mss (`width * height * 4`) when `include_pixels=True` on capture.
    pixels_bgra: bytes | None = None


class FullscreenCapture:
    def __init__(
        self,
        logger: logging.Logger,
        debug_dir: str | None,
        capture_every_n_ticks: int,
        match_id: str,
    ) -> None:
        self.logger = logger
        self.capture_every_n_ticks = max(0, capture_every_n_ticks)
        self.match_id = _sanitize_name_token(match_id, fallback="local-match")
        self._capture_context = _CaptureNameContext()
        self.debug_dir = (Path(debug_dir) / self.match_id) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def update_capture_context(
        self,
        *,
        elixir_value: float | None,
        hand_cards: list[str] | tuple[str, ...] | None,
    ) -> None:
        self._capture_context = _CaptureNameContext.from_values(
            elixir_value=elixir_value,
            hand_cards=hand_cards,
        )

    def capture(self, tick_id: int, *, include_pixels: bool = False) -> FrameObservation:
        start = time.perf_counter()
        try:
            import mss
            import mss.tools as mss_tools
        except ModuleNotFoundError:
            latency = int((time.perf_counter() - start) * 1000)
            self.logger.warning("capture_unavailable reason=mss_not_installed")
            return FrameObservation(
                width=0,
                height=0,
                capture_latency_ms=latency,
                source="unavailable",
                screenshot_path=None,
                pixels_bgra=None,
            )

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            screenshot_path = self._maybe_save_debug_screenshot(
                tick_id=tick_id,
                rgb=shot.rgb,
                size=(shot.width, shot.height),
                mss_tools=mss_tools,
            )
            pixels_bgra = bytes(shot.bgra) if include_pixels else None

        latency = int((time.perf_counter() - start) * 1000)
        return FrameObservation(
            width=shot.width,
            height=shot.height,
            capture_latency_ms=latency,
            source="fullscreen",
            screenshot_path=screenshot_path,
            pixels_bgra=pixels_bgra,
        )

    def _maybe_save_debug_screenshot(
        self,
        tick_id: int,
        rgb: bytes,
        size: tuple[int, int],
        mss_tools: Any,
    ) -> str | None:
        if not self.debug_dir or self.capture_every_n_ticks <= 0:
            return None
        if tick_id % self.capture_every_n_ticks != 0:
            return None

        file_path = self.debug_dir / self._capture_context.filename_stem()
        mss_tools.to_png(rgb, size, output=str(file_path))
        return str(file_path)


def frame_for_tick(capture: FullscreenCapture, tick_id: int, *, include_pixels: bool) -> FrameObservation:
    """Return a real capture frame for the given tick."""
    return capture.capture(tick_id, include_pixels=include_pixels)


def _sanitize_name_token(value: str, *, fallback: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", normalized).strip("-_.")
    return sanitized or fallback


@dataclass(frozen=True)
class _CaptureNameContext:
    elixir: str = "unknown"
    card_1: str = "unknown"
    card_2: str = "unknown"
    card_3: str = "unknown"
    card_4: str = "unknown"

    @classmethod
    def from_values(
        cls,
        *,
        elixir_value: float | None,
        hand_cards: list[str] | tuple[str, ...] | None,
    ) -> "_CaptureNameContext":
        if elixir_value is None:
            elixir = "unknown"
        else:
            rounded = int(max(0, min(10, round(float(elixir_value)))))
            elixir = str(rounded)
        cards = list(hand_cards or [])
        while len(cards) < 4:
            cards.append("unknown")
        cards = cards[:4]
        safe_cards = [
            _sanitize_name_token(card, fallback="unknown")
            for card in cards
        ]
        return cls(elixir=elixir, card_1=safe_cards[0], card_2=safe_cards[1], card_3=safe_cards[2], card_4=safe_cards[3])

    def filename_stem(self) -> str:
        random_id = secrets.token_hex(5)
        return (
            f"CHECK_{self.elixir}_{self.card_1}_{self.card_2}_{self.card_3}_{self.card_4}_{random_id}.png"
        )
