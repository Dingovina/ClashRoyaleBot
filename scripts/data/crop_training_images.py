#!/usr/bin/env python3
"""
Crop training screenshots to model-specific ROIs and save compact datasets.

Supported dataset layouts:
- battlefield: ``<raw-root>/battlefield_test/good/*.png`` and ``.../bad/*.png``
- elixir: ``<raw-root>/elixir_test/*.png`` (``<label>_<index>.png`` names are preserved)
- cards: ``<raw-root>/cards/*.png`` (``<card1>_<card2>_<card3>_<card4>[_suffix].png``)

Outputs are written under ``<processed-root>`` with the same relative structure.
Source PNG files are removed only when ``--delete-source`` is enabled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.manifest import write_dataset_manifest
from scripts.data.crop_battlefield_images import crop_battlefield_images
from scripts.data.crop_elixir_images import crop_elixir_images
from scripts.data.crop_card_images import crop_card_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop ClashRoyaleBot training screenshots to ROI-only PNGs")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Input root with full screenshots",
    )
    parser.add_argument(
        "--layout-yaml",
        type=Path,
        default=Path("configs/screen_layout_reference.yaml"),
        help="Screen layout YAML with bottom_panel and elixir_number rects",
    )
    parser.add_argument(
        "--target",
        choices=("battlefield", "elixir", "cards", "all"),
        default="all",
        help="Which dataset to crop",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Output root for cropped model-ready PNGs",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source screenshots from raw root after successful crop",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="roi-default",
        help="Dataset identifier for generated manifests",
    )
    parser.add_argument(
        "--cards-id-bytes",
        type=int,
        default=5,
        help="Random suffix size for generated card crop filenames",
    )
    args = parser.parse_args()

    total_processed = 0
    total_skipped = 0
    all_written_paths: list[Path] = []
    if args.target in {"battlefield", "all"}:
        result = crop_battlefield_images(
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            layout_yaml=args.layout_yaml,
            delete_source=args.delete_source,
        )
        total_processed += result.processed
        total_skipped += result.skipped
        all_written_paths.extend(result.written_paths)
        print(f"battlefield: processed={result.processed} skipped={result.skipped}")
    if args.target in {"elixir", "all"}:
        result = crop_elixir_images(
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            layout_yaml=args.layout_yaml,
            delete_source=args.delete_source,
        )
        total_processed += result.processed
        total_skipped += result.skipped
        all_written_paths.extend(result.written_paths)
        print(f"elixir: processed={result.processed} skipped={result.skipped}")
    if args.target in {"cards", "all"}:
        result = crop_card_images(
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            layout_yaml=args.layout_yaml,
            id_bytes=args.cards_id_bytes,
            delete_source=args.delete_source,
            dataset_id=f"{args.dataset_id}-cards",
            write_manifest=False,
        )
        total_processed += result.processed
        total_skipped += result.skipped
        all_written_paths.extend(result.written_paths)
        print(
            f"cards: processed={result.processed} skipped={result.skipped} "
            f"crops_saved={result.crops_saved}"
        )

    write_dataset_manifest(
        manifest_path=args.processed_root / "dataset_manifest.json",
        dataset_id=args.dataset_id,
        schema_version=1,
        source_root=args.raw_root,
        processed_root=args.processed_root,
        files=all_written_paths,
        extra={"script": "crop_training_images.py", "target": args.target},
    )

    print(
        "done: processed={} skipped={} raw_root={} processed_root={}".format(
            total_processed,
            total_skipped,
            args.raw_root,
            args.processed_root,
        )
    )


if __name__ == "__main__":
    main()
