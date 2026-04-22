from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.runtime.viewport import GameViewport, parse_game_viewport


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
            game_viewport=parse_game_viewport(runtime),
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


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file)
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return parsed
