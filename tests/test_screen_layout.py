from __future__ import annotations

import unittest
from pathlib import Path

from src.runtime.screen_layout import PixelRect, intersect_pixel_rects, load_screen_layout_reference


class ScreenLayoutTests(unittest.TestCase):
    def test_load_reference_yaml(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "screen_layout_reference.yaml"
        layout = load_screen_layout_reference(path)
        self.assertEqual(layout.schema_version, 1)
        self.assertEqual(layout.bottom_panel.left, 656)
        self.assertEqual(len(layout.hand_cards), 4)
        self.assertEqual(layout.hand_cards[0].left, 792)
        self.assertEqual(layout.elixir_bar.tick_width_px, 42)
        self.assertEqual(layout.elixir_bar.rect.width, 1240 - 818 + 1)

    def test_intersect_pixel_rects(self) -> None:
        a = PixelRect(0, 0, 10, 10)
        b = PixelRect(5, 5, 15, 15)
        inter = intersect_pixel_rects(a, b)
        assert inter is not None
        self.assertEqual(inter.left, 5)
        self.assertEqual(inter.top, 5)
        self.assertEqual(inter.right, 10)
        self.assertEqual(inter.bottom, 10)


if __name__ == "__main__":
    unittest.main()
