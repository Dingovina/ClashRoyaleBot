from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.perception.battlefield_roi import pil_rgb_masked_bottom_panel
from src.perception.elixir_roi import pil_rgb_elixir_number
from src.perception.screen_layout import load_screen_layout_reference


class TrainingRoiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "screen_layout_reference.yaml"
        cls.layout = load_screen_layout_reference(path)

    def test_elixir_crop_accepts_fullscreen_and_cropped_inputs(self) -> None:
        layout = self.layout
        full = np.zeros((1080, 1920, 3), dtype=np.uint8)
        full[layout.elixir_number.top : layout.elixir_number.bottom + 1, layout.elixir_number.left : layout.elixir_number.right + 1] = 255
        full_image = Image.fromarray(full)
        full_crop = pil_rgb_elixir_number(full_image, layout)
        self.assertEqual(full_crop.size, (layout.elixir_number.width, layout.elixir_number.height))

        already = Image.new("RGB", (layout.elixir_number.width, layout.elixir_number.height), color=(123, 0, 0))
        already_crop = pil_rgb_elixir_number(already, layout)
        self.assertEqual(already_crop.size, (layout.elixir_number.width, layout.elixir_number.height))

    def test_battlefield_crop_accepts_fullscreen_and_cropped_inputs(self) -> None:
        layout = self.layout
        full = Image.new("RGB", (1920, 1080), color=(10, 20, 30))
        full_crop = pil_rgb_masked_bottom_panel(full, layout)
        self.assertEqual(full_crop.size, (layout.bottom_panel.width, layout.bottom_panel.height))

        already = Image.new("RGB", (layout.bottom_panel.width, layout.bottom_panel.height), color=(10, 20, 30))
        already_crop = pil_rgb_masked_bottom_panel(already, layout)
        self.assertEqual(already_crop.size, (layout.bottom_panel.width, layout.bottom_panel.height))

        # Some captured datasets can be off by one pixel on height/width.
        almost = Image.new("RGB", (layout.bottom_panel.width, layout.bottom_panel.height - 1), color=(10, 20, 30))
        almost_crop = pil_rgb_masked_bottom_panel(almost, layout)
        self.assertEqual(almost_crop.size, (layout.bottom_panel.width, layout.bottom_panel.height - 1))


if __name__ == "__main__":
    unittest.main()
