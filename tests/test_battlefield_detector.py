from __future__ import annotations

import importlib.util
import logging
import tempfile
import unittest
from pathlib import Path

from src.perception.infer.battlefield_infer import clear_battlefield_runner_cache
from src.runtime.config.battlefield_config import BattlefieldModelConfig
from src.runtime.evaluation.battlefield_evaluate import evaluate_battlefield
from src.runtime.infra.viewport import AnchorRect, GameViewport, crop_playfield_bgra


class BattlefieldModelTests(unittest.TestCase):
    def test_crop_playfield_respects_viewport_and_anchor_rect(self) -> None:
        fw, fh = 200, 100
        full = bytearray([10, 20, 30, 255]) * (fw * fh)
        viewport = GameViewport(
            mode="centered_strip",
            width=100,
            height=100,
            anchor_rect=AnchorRect(left_ratio=0.0, top_ratio=0.0, width_ratio=1.0, height_ratio=0.5),
        )
        rw, rh, roi = crop_playfield_bgra(fw, fh, bytes(full), viewport)
        self.assertEqual(rw, 100)
        self.assertEqual(rh, 50)
        self.assertEqual(len(roi), rw * rh * 4)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_evaluate_battlefield_model_runs_with_checkpoint(self) -> None:
        import torch

        from src.perception.models.battlefield_net import BattlefieldScreenNet

        clear_battlefield_runner_cache()
        net = BattlefieldScreenNet()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            torch.save({"state_dict": net.state_dict(), "input_size": 64}, path)
            layout_yaml = Path(__file__).resolve().parents[1] / "configs" / "screen_layout_reference.yaml"
            det = BattlefieldModelConfig(
                score_threshold=0.01,
                model_path=str(path),
                model_layout_path=str(layout_yaml),
            )
            w, h = 80, 60
            buf = bytearray([40, 50, 60, 255]) * (w * h)
            ok, score = evaluate_battlefield(
                frame_width=w,
                frame_height=h,
                pixels_bgra=bytes(buf),
                detector=det,
                logger=logging.getLogger("test"),
            )
            self.assertIsInstance(ok, bool)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        finally:
            clear_battlefield_runner_cache()
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
