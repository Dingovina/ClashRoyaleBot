from __future__ import annotations

import unittest
from pathlib import Path

from src.runtime.config.card_registry import load_card_registry
from src.runtime.config.config_loader import load_runtime_config
from src.runtime.domain.types import ActionType, DecisionReason, InferenceStatus


class RuntimeContractsTests(unittest.TestCase):
    def test_runtime_decision_enums_are_stable(self) -> None:
        self.assertEqual(ActionType.DEPLOY.value, "deploy")
        self.assertEqual(ActionType.NO_OP.value, "no_op")
        self.assertEqual(DecisionReason.UNKNOWN_CARD_ELIXIR_COST.value, "unknown_card_elixir_cost")
        self.assertEqual(InferenceStatus.OK.value, "ok")

    def test_card_registry_contains_costs_for_all_aliases(self) -> None:
        reg = load_card_registry(Path("configs/card_registry.yaml"))
        for alias, canonical in reg.aliases.items():
            self.assertTrue(alias)
            self.assertIn(canonical, reg.elixir_costs)

    def test_runtime_uses_card_registry_spell_cards(self) -> None:
        cfg = load_runtime_config(Path("configs/runtime.yaml"))
        reg = load_card_registry(Path("configs/card_registry.yaml"))
        self.assertEqual(cfg.spell_cards, reg.spell_cards)
        self.assertEqual(cfg.card_elixir_costs, reg.elixir_costs)
        self.assertIn("musketeer", cfg.card_name_aliases)


if __name__ == "__main__":
    unittest.main()
