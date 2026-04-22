from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

ViewportMode = Literal["full_frame", "explicit", "centered_strip"]


@dataclass(frozen=True)
class AnchorRect:
    """Fractions of the game viewport where board anchors (0..1) map (excludes hand UI, side chrome)."""

    left_ratio: float = 0.0
    top_ratio: float = 0.0
    width_ratio: float = 1.0
    height_ratio: float = 1.0


@dataclass(frozen=True)
class GameViewport:
    """Maps normalized board anchors (0..1) into pixel coordinates on the captured frame."""

    mode: ViewportMode
    left: int | None = None
    top: int | None = None
    width: int | None = None
    height: int | None = None
    anchor_rect: AnchorRect = field(default_factory=AnchorRect)

    def rect_for_frame(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        if self.mode == "full_frame":
            return (0, 0, frame_width, frame_height)
        if self.mode == "explicit":
            if self.left is None or self.top is None or self.width is None or self.height is None:
                raise ValueError("game_viewport mode explicit requires left, top, width, height")
            return (self.left, self.top, self.width, self.height)
        if self.mode == "centered_strip":
            if self.width is None or self.height is None:
                raise ValueError("game_viewport mode centered_strip requires width and height")
            left = max(0, (frame_width - self.width) // 2)
            top = max(0, (frame_height - self.height) // 2)
            return (left, top, self.width, self.height)
        raise ValueError(f"unknown game_viewport mode: {self.mode!r}")


def parse_game_viewport(runtime: dict[str, Any]) -> GameViewport:
    """Parse `runtime.game_viewport` from the top-level `runtime` mapping in `configs/runtime.yaml`."""
    raw = runtime.get("game_viewport")
    if not raw:
        return GameViewport(mode="full_frame")
    if not isinstance(raw, dict):
        raise ValueError("runtime.game_viewport must be a mapping")

    mode = str(raw.get("mode", "full_frame"))
    if mode not in ("full_frame", "explicit", "centered_strip"):
        raise ValueError(f"invalid runtime.game_viewport.mode: {mode!r}")

    return GameViewport(
        mode=cast(ViewportMode, mode),
        left=_optional_int(raw.get("left")),
        top=_optional_int(raw.get("top")),
        width=_optional_int(raw.get("width")),
        height=_optional_int(raw.get("height")),
        anchor_rect=_parse_anchor_rect(raw.get("anchor_rect")),
    )


def _parse_anchor_rect(raw: Any) -> AnchorRect:
    if raw is None:
        return AnchorRect()
    if not isinstance(raw, dict):
        raise ValueError("game_viewport.anchor_rect must be a mapping or null")

    rect = AnchorRect(
        left_ratio=float(raw.get("left_ratio", 0.0)),
        top_ratio=float(raw.get("top_ratio", 0.0)),
        width_ratio=float(raw.get("width_ratio", 1.0)),
        height_ratio=float(raw.get("height_ratio", 1.0)),
    )
    for name, value in (
        ("left_ratio", rect.left_ratio),
        ("top_ratio", rect.top_ratio),
        ("width_ratio", rect.width_ratio),
        ("height_ratio", rect.height_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"game_viewport.anchor_rect.{name} must be within [0, 1], got {value}")
    if rect.left_ratio + rect.width_ratio > 1.0 + 1e-6:
        raise ValueError("game_viewport.anchor_rect left_ratio + width_ratio must be <= 1")
    if rect.top_ratio + rect.height_ratio > 1.0 + 1e-6:
        raise ValueError("game_viewport.anchor_rect top_ratio + height_ratio must be <= 1")
    return rect


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
