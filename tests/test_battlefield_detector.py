from __future__ import annotations

import importlib.util
import logging
import tempfile
import unittest
from pathlib import Path

from src.perception.battlefield_infer import clear_battlefield_runner_cache

from src.runtime.battlefield_detector import (
    BattlefieldDetectorConfig,
    crop_playfield_bgra,
    evaluate_battlefield,
    score_battlefield_heuristic_bgra,
)
from src.runtime.viewport import AnchorRect, GameViewport


class BattlefieldDetectorTests(unittest.TestCase):
    def test_heuristic_high_score_on_synthetic_arena_colors(self) -> None:
        w, h = 160, 120
        buf = bytearray([128, 128, 128, 255]) * (w * h)
        # River band rows 36..55 (~0.30..0.52): blue-dominant BGRA.
        for y in range(36, 56):
            for x in range(0, w, 2):
                i = (y * w + x) * 4
                buf[i : i + 4] = bytes([200, 90, 40, 255])
        # Grass band rows 70..105 (~0.55..0.90).
        for y in range(70, 106):
            for x in range(0, w, 2):
                i = (y * w + x) * 4
                buf[i : i + 4] = bytes([60, 190, 70, 255])
        roi = bytes(buf)
        score = score_battlefield_heuristic_bgra(
            w,
            h,
            roi,
            sample_stride=2,
            river_top_ratio=0.30,
            river_bottom_ratio=0.52,
            grass_top_ratio=0.55,
            grass_bottom_ratio=0.90,
            logger=None,
        )
        self.assertGreater(score, 0.2)

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

    def test_evaluate_battlefield_respects_threshold(self) -> None:
        w, h = 120, 100
        buf = bytearray([128, 128, 128, 255]) * (w * h)
        for y in range(30, 52):
            for x in range(0, w, 2):
                i = (y * w + x) * 4
                buf[i : i + 4] = bytes([200, 90, 40, 255])
        for y in range(56, 90):
            for x in range(0, w, 2):
                i = (y * w + x) * 4
                buf[i : i + 4] = bytes([60, 190, 70, 255])
        det = BattlefieldDetectorConfig(
            method="heuristic",
            score_threshold=0.50,
            sample_stride=2,
            river_band_top_ratio=0.30,
            river_band_bottom_ratio=0.52,
            grass_band_top_ratio=0.55,
            grass_band_bottom_ratio=0.90,
            model_path=None,
            model_input_size=128,
            model_layout_path="configs/screen_layout_reference.yaml",
        )
        ok, score = evaluate_battlefield(
            frame_width=w,
            frame_height=h,
            pixels_bgra=bytes(buf),
            viewport=GameViewport(mode="full_frame"),
            detector=det,
            logger=logging.getLogger("test"),
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(score, 0.50)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_model_evaluate_runs_with_checkpoint(self) -> None:
        import torch

        from src.perception.battlefield_net import BattlefieldScreenNet

        clear_battlefield_runner_cache()
        net = BattlefieldScreenNet()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            torch.save({"state_dict": net.state_dict(), "input_size": 64}, path)
            layout_yaml = Path(__file__).resolve().parents[1] / "configs" / "screen_layout_reference.yaml"
            det = BattlefieldDetectorConfig(
                method="model",
                score_threshold=0.01,
                sample_stride=4,
                river_band_top_ratio=0.30,
                river_band_bottom_ratio=0.52,
                grass_band_top_ratio=0.55,
                grass_band_bottom_ratio=0.90,
                model_path=str(path),
                model_input_size=64,
                model_layout_path=str(layout_yaml),
            )
            w, h = 80, 60
            buf = bytearray([40, 50, 60, 255]) * (w * h)
            ok, score = evaluate_battlefield(
                frame_width=w,
                frame_height=h,
                pixels_bgra=bytes(buf),
                viewport=GameViewport(mode="full_frame"),
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
