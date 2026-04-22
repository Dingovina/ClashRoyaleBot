from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

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


@dataclass(frozen=True)
class RuntimeConfig:
    tick_interval_ms: int
    action_rate_limit_ms: int
    action_confidence_threshold: float
    no_op_confidence_threshold: float
    min_elixir_for_non_urgent_action: float
    max_ticks: int
    zones: dict[int, tuple[float, float]]
    spell_cards: set[str]
    capture_enabled: bool
    capture_debug_dir: str | None
    capture_every_n_ticks: int
    actuation_enabled: bool
    actuation_dry_run: bool
    actuation_select_to_click_delay_ms: int
    actuation_card_hotkeys: tuple[str, str, str, str]
    game_viewport: GameViewport

    @staticmethod
    def from_file(path: Path) -> "RuntimeConfig":
        data = _load_yaml(path)
        runtime = data["runtime"]
        board = data["board"]
        card_types = data.get("card_types", {})

        zones = {
            int(zone_id): (float(anchor[0]), float(anchor[1]))
            for zone_id, anchor in board["zones"].items()
        }

        return RuntimeConfig(
            tick_interval_ms=int(runtime["tick_interval_ms"]),
            action_rate_limit_ms=int(runtime["action_rate_limit_ms"]),
            action_confidence_threshold=float(runtime["action_confidence_threshold"]),
            no_op_confidence_threshold=float(runtime["no_op_confidence_threshold"]),
            min_elixir_for_non_urgent_action=float(runtime["min_elixir_for_non_urgent_action"]),
            max_ticks=int(runtime["max_ticks"]),
            zones=zones,
            spell_cards=set(card_types.get("spell_cards", [])),
            capture_enabled=bool(runtime.get("capture_enabled", True)),
            capture_debug_dir=runtime.get("capture_debug_dir"),
            capture_every_n_ticks=max(0, int(runtime.get("capture_every_n_ticks", 0))),
            actuation_enabled=bool(runtime.get("actuation_enabled", False)),
            actuation_dry_run=bool(runtime.get("actuation_dry_run", True)),
            actuation_select_to_click_delay_ms=max(
                0, int(runtime.get("actuation_select_to_click_delay_ms", 120))
            ),
            actuation_card_hotkeys=_parse_actuation_card_hotkeys(runtime),
            game_viewport=_parse_game_viewport(runtime),
        )


def _parse_actuation_card_hotkeys(runtime: dict[str, Any]) -> tuple[str, str, str, str]:
    raw = runtime.get("actuation_card_hotkeys")
    if raw is None:
        return ("1", "2", "3", "4")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("runtime.actuation_card_hotkeys must be a list of exactly 4 strings")

    keys: list[str] = []
    for index, item in enumerate(raw):
        key = str(item).strip().lower()
        if not key:
            raise ValueError(f"runtime.actuation_card_hotkeys[{index}] is empty")
        keys.append(key)

    return (keys[0], keys[1], keys[2], keys[3])


def _parse_game_viewport(runtime: dict[str, Any]) -> GameViewport:
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


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file)
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return parsed
