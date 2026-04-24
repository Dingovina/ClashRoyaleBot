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
- source screenshot is removed only when ``--delete-source`` is enabled
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.roi.card_roi import pil_rgb_hand_card
from src.perception.roi.screen_layout import load_screen_layout_reference
from src.ml.manifest import write_dataset_manifest
from scripts.data.crop_result import CropResult


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


def _sanitized_card_name(card_name: str) -> str:
    safe = card_name.strip().lower().replace(" ", "-")
    if not safe:
        raise ValueError("Card name cannot be empty")
    return safe


def crop_card_images(
    *,
    raw_root: Path,
    processed_root: Path,
    layout_yaml: Path,
    id_bytes: int,
    delete_source: bool,
    dataset_id: str,
    write_manifest: bool = True,
) -> CropResult:
    raw_dir = raw_root / "cards"
    out_dir = processed_root / "cards"
    if id_bytes < 2:
        raise ValueError("--id-bytes must be >= 2")
    if not raw_dir.is_dir():
        raise ValueError(f"Raw cards directory does not exist: {raw_dir}")

    layout = load_screen_layout_reference(layout_yaml)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_files = 0
    skipped_bad_names = 0
    written_paths: list[Path] = []

    for src_path in sorted(raw_dir.glob("*.png")):
        try:
            labels = _parse_hand_labels(src_path.stem)
        except ValueError as exc:
            skipped_bad_names += 1
            print(f"skip_bad_name file={src_path.name} reason={exc}")
            continue

        with Image.open(src_path) as image:
            for slot_index, card_name in enumerate(labels):
                crop = pil_rgb_hand_card(image, layout, slot_index)
                out_name = f"{_sanitized_card_name(card_name)}_{secrets.token_hex(id_bytes)}.png"
                out_path = out_dir / out_name
                crop.save(out_path, format="PNG")
                written_paths.append(out_path)

        if delete_source:
            src_path.unlink()
        processed_files += 1

    if write_manifest:
        write_dataset_manifest(
            manifest_path=out_dir / "dataset_manifest.json",
            dataset_id=dataset_id,
            schema_version=1,
            source_root=raw_root,
            processed_root=processed_root,
            files=written_paths,
            extra={"script": "crop_card_images.py"},
        )

    return CropResult(processed=processed_files, skipped=skipped_bad_names, written_paths=written_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split labeled hand screenshots into per-card crops")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
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
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source screenshots after successful processing",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="cards-default",
        help="Dataset identifier written into dataset_manifest.json",
    )
    args = parser.parse_args()

    try:
        result = crop_card_images(
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            layout_yaml=args.layout_yaml,
            id_bytes=args.id_bytes,
            delete_source=args.delete_source,
            dataset_id=args.dataset_id,
            write_manifest=True,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "done: files_processed={} crops_saved={} skipped_bad_names={} processed_root={}".format(
            result.processed,
            result.crops_saved,
            result.skipped,
            args.processed_root,
        )
    )


if __name__ == "__main__":
    main()
