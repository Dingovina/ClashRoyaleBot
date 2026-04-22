from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file)
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return parsed
