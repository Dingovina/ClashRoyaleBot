from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneMap:
    anchors: dict[int, tuple[float, float]]
    unit_zones: set[int]
    spell_zones: set[int]

    def is_zone_valid(self, zone_id: int, card_class: str) -> bool:
        if zone_id not in self.anchors:
            return False

        if card_class == "spell":
            return zone_id in self.spell_zones
        return zone_id in self.unit_zones


def build_default_zone_map(anchors: dict[int, tuple[float, float]]) -> ZoneMap:
    # In v1 all 12 zones are legal for spells.
    spell_zones = set(anchors.keys())

    # In v1 units are limited to player's half + bridge row.
    unit_zones = {6, 7, 8, 9, 10, 11, 3, 4, 5}

    return ZoneMap(
        anchors=anchors,
        unit_zones=unit_zones,
        spell_zones=spell_zones,
    )
