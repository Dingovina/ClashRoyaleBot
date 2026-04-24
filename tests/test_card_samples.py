from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.perception.datasets.card_samples import collect_card_labeled_pngs


class CardSamplesTests(unittest.TestCase):
    def test_collects_card_name_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mini-p.e.k.k.a_ab12cd.png").touch()
            (root / "empty_ff00ee.png").touch()
            samples = collect_card_labeled_pngs(root)
            labels = sorted([label for _, label in samples])
            self.assertEqual(labels, ["empty", "mini-p.e.k.k.a"])

    def test_raises_on_invalid_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "brokenname.png").touch()
            with self.assertRaises(ValueError):
                collect_card_labeled_pngs(root)


if __name__ == "__main__":
    unittest.main()
