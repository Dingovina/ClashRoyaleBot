from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.runtime.config.config_loader import load_runtime_config


class ConfigBattlefieldWeightsTests(unittest.TestCase):
    def test_missing_checkpoint_under_artifacts_path_includes_training_hint(self) -> None:
        """Missing weights at .../artifacts/battlefield_cnn.pt should mention training (any cwd)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts").mkdir(parents=True)
            missing_ckpt = root / "artifacts" / "battlefield_cnn.pt"
            layout = Path(__file__).resolve().parents[1] / "configs" / "screen_layout_reference.yaml"
            yaml_path = root / "runtime.yaml"
            board = {
                "zones": {str(i): [0.5, 0.5] for i in range(12)},
            }
            runtime = {
                "match_id": "test-match",
                "tick_interval_ms": 500,
                "action_rate_limit_ms": 1000,
                "action_confidence_threshold": 0.7,
                "no_op_confidence_threshold": 0.55,
                "min_elixir_for_non_urgent_action": 3.0,
                "match_safety_max_ticks": 1,
                "battlefield_end_score_threshold": 0.42,
                "match_end_confirm_ticks": 6,
                "match_end_check_every_n_ticks": 2,
                "battlefield_score_threshold": 0.95,
                "match_readiness_enabled": True,
                "battlefield_model_path": str(missing_ckpt),
                "battlefield_model_layout_path": str(layout),
            }
            yaml_path.write_text(
                yaml.safe_dump({"runtime": runtime, "board": board}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_runtime_config(yaml_path)
            msg = str(ctx.exception).lower()
            self.assertIn("battlefield_cnn.pt", msg)
            self.assertIn("train", msg)
            self.assertIn("train_battlefield_classifier.py", msg)

    def test_invalid_match_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = root / "runtime.yaml"
            board = {
                "zones": {str(i): [0.5, 0.5] for i in range(12)},
            }
            runtime = {
                "match_id": "invalid match id!",
                "tick_interval_ms": 500,
                "action_rate_limit_ms": 1000,
                "action_confidence_threshold": 0.7,
                "no_op_confidence_threshold": 0.55,
                "min_elixir_for_non_urgent_action": 3.0,
                "match_safety_max_ticks": 1,
                "battlefield_end_score_threshold": 0.42,
                "match_end_confirm_ticks": 0,
                "match_end_check_every_n_ticks": 2,
                "battlefield_score_threshold": 0.95,
                "match_readiness_enabled": False,
                "battlefield_wait_timeout_ms": 0,
                "battlefield_timeout_behavior": "idle",
                "foreground_check_enabled": False,
                "capture_debug_save_enabled": False,
                "capture_every_n_ticks": 0,
                "actuation_select_to_click_delay_ms": 0,
                "actuation_card_hotkeys": ["1", "2", "3", "4"],
                "game_viewport": {"mode": "full_frame"},
                "elixir_model_path": str(root / "elixir.pt"),
                "elixir_model_layout_path": str(Path(__file__).resolve().parents[1] / "configs" / "screen_layout_reference.yaml"),
                "card_model_path": str(root / "card.pt"),
                "card_model_layout_path": str(Path(__file__).resolve().parents[1] / "configs" / "screen_layout_reference.yaml"),
                "card_registry_path": str(Path(__file__).resolve().parents[1] / "configs" / "card_registry.yaml"),
            }
            (root / "elixir.pt").touch()
            (root / "card.pt").touch()
            yaml_path.write_text(yaml.safe_dump({"runtime": runtime, "board": board}), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_runtime_config(yaml_path)
            self.assertIn("runtime.match_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
