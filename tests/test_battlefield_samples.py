from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.perception.datasets.battlefield_samples import collect_battlefield_labeled_pngs


class BattlefieldSamplesTests(unittest.TestCase):
    def test_raises_without_good_and_bad_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                collect_battlefield_labeled_pngs(root)

    def test_collects_good_and_bad_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "good").mkdir()
            (root / "bad").mkdir()
            (root / "good" / "a.png").touch()
            (root / "bad" / "b.png").touch()
            s = collect_battlefield_labeled_pngs(root)
            by_label = {p.name: y for p, y in s}
            self.assertEqual(by_label["a.png"], 1)
            self.assertEqual(by_label["b.png"], 0)


if __name__ == "__main__":
    unittest.main()
