from __future__ import annotations

from pathlib import Path


def collect_battlefield_labeled_pngs(data_dir: Path) -> list[tuple[Path, int]]:
    """
    Collect ``(path, label)`` for PNG training/eval samples.

    Layout: ``data_dir/good/*.png`` → label ``1`` (in-match), ``data_dir/bad/*.png`` → label ``0``.
    Both subdirectories must exist.
    """
    good_dir = data_dir / "good"
    bad_dir = data_dir / "bad"
    if not good_dir.is_dir() or not bad_dir.is_dir():
        raise ValueError(
            f"Expected labeled PNG folders {good_dir} and {bad_dir} (each directory must exist)."
        )

    samples: list[tuple[Path, int]] = []
    for p in sorted(good_dir.glob("*.png")):
        samples.append((p, 1))
    for p in sorted(bad_dir.glob("*.png")):
        samples.append((p, 0))
    return samples
