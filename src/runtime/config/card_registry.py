from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CardRegistry:
    aliases: dict[str, str]
    elixir_costs: dict[str, float]
    spell_cards: set[str]

    def canonical_name(self, raw_name: str) -> str:
        key = raw_name.strip().lower()
        if not key:
            return key
        return self.aliases.get(key, key)


def load_card_registry(path: Path) -> CardRegistry:
    if not path.is_file():
        raise ValueError(f"card registry file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid card registry structure in {path}")

    cards_block = parsed.get("cards")
    if not isinstance(cards_block, dict) or not cards_block:
        raise ValueError(f"{path}: 'cards' must be a non-empty mapping")

    aliases: dict[str, str] = {}
    costs: dict[str, float] = {}
    spell_cards: set[str] = set()

    for canonical, raw in cards_block.items():
        canonical_name = str(canonical).strip().lower()
        if not canonical_name:
            raise ValueError(f"{path}: card name must not be empty")
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: card entry '{canonical_name}' must be a mapping")
        cost = float(raw.get("elixir_cost", 0))
        if cost <= 0:
            raise ValueError(f"{path}: card '{canonical_name}' elixir_cost must be > 0")
        costs[canonical_name] = cost

        card_class = str(raw.get("class", "unit")).strip().lower()
        if card_class == "spell":
            spell_cards.add(canonical_name)

        aliases[canonical_name] = canonical_name
        for alias in raw.get("aliases", []):
            alias_name = str(alias).strip().lower()
            if alias_name:
                aliases[alias_name] = canonical_name

    return CardRegistry(aliases=aliases, elixir_costs=costs, spell_cards=spell_cards)
