from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PixelRect:
    """Inclusive pixel bounds in fullscreen capture coordinates (x rightward, y downward)."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


def intersect_pixel_rects(a: PixelRect, b: PixelRect) -> PixelRect | None:
    """Intersection of two inclusive pixel rectangles, or None if disjoint."""
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    if left > right or top > bottom:
        return None
    return PixelRect(left=left, top=top, right=right, bottom=bottom)


@dataclass(frozen=True)
class ScreenLayoutReference:
    schema_version: int
    id: str
    reference_frame_width: int
    reference_frame_height: int
    reference_notes: str | None
    bottom_panel: PixelRect
    hand_cards: tuple[PixelRect, PixelRect, PixelRect, PixelRect]
    next_card: PixelRect
    elixir_number: PixelRect

    def hud_subtract_rects(self) -> tuple[PixelRect, ...]:
        """Regions to zero out inside ``bottom_panel`` when training/inferring the battlefield CNN."""
        return (*self.hand_cards, self.next_card, self.elixir_number)


def _rect(raw: dict[str, Any]) -> PixelRect:
    r = PixelRect(
        left=int(raw["left"]),
        top=int(raw["top"]),
        right=int(raw["right"]),
        bottom=int(raw["bottom"]),
    )
    if r.right < r.left or r.bottom < r.top:
        raise ValueError(f"Invalid rect (need right>=left, bottom>=top): {raw}")
    return r


def load_screen_layout_reference(path: Path) -> ScreenLayoutReference:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")

    cards_raw = data.get("hand_cards")
    if not isinstance(cards_raw, list) or len(cards_raw) != 4:
        raise ValueError("hand_cards must be a list of exactly 4 slot entries")

    hand: list[PixelRect] = []
    for index, item in enumerate(cards_raw):
        if not isinstance(item, dict):
            raise ValueError(f"hand_cards[{index}] must be a mapping")
        hand.append(_rect(item))

    ref = data.get("reference_frame") or {}
    elixir_raw = data.get("elixir_number")
    if not isinstance(elixir_raw, dict):
        raise ValueError("elixir_number must be a mapping with left, top, right, bottom")

    return ScreenLayoutReference(
        schema_version=int(data.get("schema_version", 1)),
        id=str(data.get("id", "unknown")),
        reference_frame_width=int(ref.get("width", 0)) or 1920,
        reference_frame_height=int(ref.get("height", 0)) or 1080,
        reference_notes=str(ref["notes"]) if ref.get("notes") else None,
        bottom_panel=_rect(data["bottom_panel"]),
        hand_cards=(hand[0], hand[1], hand[2], hand[3]),
        next_card=_rect(data["next_card"]),
        elixir_number=_rect(elixir_raw),
    )
