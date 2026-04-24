#!/usr/bin/env python3
"""
Process labeled hand screenshots into per-card crops.

Input:
- ``data/raw/cards/*.png``
- each filename stem contains exactly 4 labels (left->right), split by ``_``
- label value can be ``empty`` to skip a slot

Output:
- crops saved to ``data/processed/cards/<card-name>_<random-id>.png``
- empty slots are also saved as ``empty_<random-id>.png``
- source screenshot is removed after successful processing
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.screen_layout import PixelRect, load_screen_layout_reference


def _parse_hand_labels(stem: str) -> tuple[str, str, str, str]:
    parts = [part.strip().lower() for part in stem.split("_")]
    if len(parts) < 4:
        raise ValueError(
            f"Expected at least 4 labels in filename stem, got {len(parts)} in '{stem}'"
        )
    # Allow optional trailing suffix like "_<random_id>" and ignore it.
    labels = tuple(parts[:4])
    if any(not label for label in labels):
        raise ValueError(f"Filename stem contains empty card label: '{stem}'")
    return labels  # type: ignore[return-value]


def _crop_rgb(arr: np.ndarray, rect: PixelRect) -> np.ndarray:
    h, w, _ = arr.shape
    l = max(0, min(rect.left, w - 1))
    t = max(0, min(rect.top, h - 1))
    r_in = min(w - 1, rect.right)
    b_in = min(h - 1, rect.bottom)
    if r_in < l or b_in < t:
        raise ValueError(f"Rect {rect} is outside image bounds {w}x{h}")
    return arr[t : b_in + 1, l : r_in + 1].copy()


def _sanitized_card_name(card_name: str) -> str:
    safe = card_name.strip().lower().replace(" ", "-")
    if not safe:
        raise ValueError("Card name cannot be empty")
    return safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Split labeled hand screenshots into per-card crops")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cards"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/cards"))
    parser.add_argument(
        "--layout-yaml",
        type=Path,
        default=Path("configs/screen_layout_reference.yaml"),
        help="Layout YAML with hand_cards rectangles",
    )
    parser.add_argument(
        "--id-bytes",
        type=int,
        default=5,
        help="Random suffix size in bytes (hex length = id-bytes*2)",
    )
    args = parser.parse_args()

    if args.id_bytes < 2:
        raise SystemExit("--id-bytes must be >= 2")
    if not args.raw_dir.is_dir():
        raise SystemExit(f"Raw cards directory does not exist: {args.raw_dir}")

    layout = load_screen_layout_reference(args.layout_yaml)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processed_files = 0
    saved_crops = 0
    skipped_bad_names = 0

    for src_path in sorted(args.raw_dir.glob("*.png")):
        try:
            labels = _parse_hand_labels(src_path.stem)
        except ValueError as exc:
            skipped_bad_names += 1
            print(f"skip_bad_name file={src_path.name} reason={exc}")
            continue

        with Image.open(src_path) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)

        for slot_index, card_name in enumerate(labels):
            crop = _crop_rgb(arr, layout.hand_cards[slot_index])
            out_name = f"{_sanitized_card_name(card_name)}_{secrets.token_hex(args.id_bytes)}.png"
            out_path = args.out_dir / out_name
            Image.fromarray(crop).save(out_path, format="PNG")
            saved_crops += 1

        src_path.unlink()
        processed_files += 1

    print(
        "done: files_processed={} crops_saved={} skipped_bad_names={} out_dir={}".format(
            processed_files,
            saved_crops,
            skipped_bad_names,
            args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
