from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class CropTrainingImagesTests(unittest.TestCase):
    def _script_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "scripts" / "data" / "crop_training_images.py"

    def _layout_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "configs" / "screen_layout_reference.yaml"

    def _write_fullscreen_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), color=(30, 40, 50)).save(path, format="PNG")

    def test_requires_exactly_one_split_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "m1"
            self._write_fullscreen_png(raw / "5_knight_archers_fireball_giant_a1.png")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self._script_path()),
                    "--raw-root",
                    str(root / "raw"),
                    "--processed-root",
                    str(root / "processed"),
                    "--layout-yaml",
                    str(self._layout_path()),
                    "--match-id",
                    "m1",
                    "--bf",
                    "--train",
                    "--val",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("exactly one split", (proc.stderr + proc.stdout).lower())

    def test_skips_check_files_and_deletes_processed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "m1"
            good_name = "5_knight_archers_fireball_giant_a1.png"
            check_name = "CHECK_5_knight_archers_fireball_giant_b2.png"
            self._write_fullscreen_png(raw / good_name)
            self._write_fullscreen_png(raw / check_name)
            subprocess.run(
                [
                    sys.executable,
                    str(self._script_path()),
                    "--raw-root",
                    str(root / "raw"),
                    "--processed-root",
                    str(root / "processed"),
                    "--layout-yaml",
                    str(self._layout_path()),
                    "--match-id",
                    "m1",
                    "--bf",
                    "--train",
                ],
                check=True,
            )
            out_file = root / "processed" / "train" / "battlefield_test" / "good" / good_name
            self.assertTrue(out_file.is_file())
            self.assertFalse((raw / good_name).exists())
            self.assertTrue((raw / check_name).exists())

    def test_cards_output_path_for_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "m1"
            self._write_fullscreen_png(raw / "5_knight_archers_fireball_giant_a1.png")
            subprocess.run(
                [
                    sys.executable,
                    str(self._script_path()),
                    "--raw-root",
                    str(root / "raw"),
                    "--processed-root",
                    str(root / "processed"),
                    "--layout-yaml",
                    str(self._layout_path()),
                    "--match-id",
                    "m1",
                    "--card",
                    "--train",
                ],
                check=True,
            )
            cards_dir = root / "processed" / "train" / "cards_train"
            card_files = list(cards_dir.glob("*.png"))
            self.assertGreaterEqual(len(card_files), 4)


if __name__ == "__main__":
    unittest.main()
