#!/usr/bin/env python3
"""
Crop training screenshots to model-specific ROIs and save compact datasets.

Supported dataset layouts:
- battlefield: ``<raw-root>/battlefield_test/good/*.png`` and ``.../bad/*.png``
- elixir: ``<raw-root>/elixir_test/*.png`` (``<label>_<index>.png`` names are preserved)

Outputs are written under ``<processed-root>`` with the same relative structure.
Source PNG files are removed only when ``--delete-source`` is enabled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.battlefield_roi import pil_rgb_masked_bottom_panel
from src.perception.elixir_roi import pil_rgb_elixir_number
from src.perception.screen_layout import load_screen_layout_reference
from src.ml.manifest import write_dataset_manifest


def _collect_pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.png") if p.is_file())


def _save_cropped(src_path: Path, dst_path: Path, cropper, *, delete_source: bool) -> None:
    with Image.open(src_path) as image:
        cropped = cropper(image)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(dst_path, format="PNG")
    if delete_source:
        src_path.unlink()


def _crop_battlefield(
    raw_root: Path, processed_root: Path, layout, *, delete_source: bool
) -> tuple[int, int, list[Path]]:
    processed = 0
    skipped = 0
    written_paths: list[Path] = []
    for subset in ("good", "bad"):
        src_dir = raw_root / "battlefield_test" / subset
        dst_dir = processed_root / "battlefield_test" / subset
        files = _collect_pngs(src_dir)
        if not files:
            continue
        for src_path in files:
            dst_path = dst_dir / src_path.name
            try:
                _save_cropped(
                    src_path,
                    dst_path,
                    lambda im: pil_rgb_masked_bottom_panel(im, layout),
                    delete_source=delete_source,
                )
                processed += 1
                written_paths.append(dst_path)
            except ValueError:
                skipped += 1
    return processed, skipped, written_paths


def _crop_elixir(
    raw_root: Path, processed_root: Path, layout, *, delete_source: bool
) -> tuple[int, int, list[Path]]:
    processed = 0
    skipped = 0
    written_paths: list[Path] = []
    src_dir = raw_root / "elixir_test"
    dst_dir = processed_root / "elixir_test"
    for src_path in _collect_pngs(src_dir):
        dst_path = dst_dir / src_path.name
        try:
            _save_cropped(
                src_path,
                dst_path,
                lambda im: pil_rgb_elixir_number(im, layout),
                delete_source=delete_source,
            )
            processed += 1
            written_paths.append(dst_path)
        except ValueError:
            skipped += 1
    return processed, skipped, written_paths


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
        choices=("battlefield", "elixir", "all"),
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
    args = parser.parse_args()

    layout = load_screen_layout_reference(args.layout_yaml)
    total_processed = 0
    total_skipped = 0
    all_written_paths: list[Path] = []
    if args.target in {"battlefield", "all"}:
        processed, skipped, written = _crop_battlefield(
            args.raw_root,
            args.processed_root,
            layout,
            delete_source=args.delete_source,
        )
        total_processed += processed
        total_skipped += skipped
        all_written_paths.extend(written)
        print(f"battlefield: processed={processed} skipped={skipped}")
    if args.target in {"elixir", "all"}:
        processed, skipped, written = _crop_elixir(
            args.raw_root,
            args.processed_root,
            layout,
            delete_source=args.delete_source,
        )
        total_processed += processed
        total_skipped += skipped
        all_written_paths.extend(written)
        print(f"elixir: processed={processed} skipped={skipped}")

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
