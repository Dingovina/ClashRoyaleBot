#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.manifest import write_dataset_manifest
from src.perception.roi.screen_layout import PixelRect, load_screen_layout_reference
from scripts.data.crop_result import CropResult


def _collect_pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.png") if p.is_file())


def _crop_rect(image: Image.Image, rect: PixelRect) -> Image.Image:
    # PIL crop uses half-open bounds: right/bottom are exclusive.
    return image.convert("RGB").crop((rect.left, rect.top, rect.right + 1, rect.bottom + 1))


def crop_tower_hp_images(
    *,
    raw_root: Path | None,
    processed_root: Path | None,
    layout_yaml: Path,
    delete_source: bool,
    source_paths: list[Path] | None = None,
    output_dir: Path | None = None,
) -> CropResult:
    layout = load_screen_layout_reference(layout_yaml)
    regions = layout.tower_hp_regions
    if not regions:
        raise ValueError("No tower_hp_regions found in layout YAML")

    if source_paths is None:
        if raw_root is None:
            raise ValueError("raw_root is required when source_paths is not provided")
        source_paths = _collect_pngs(raw_root / "tower_hp_test")

    if output_dir is None:
        if processed_root is None:
            raise ValueError("processed_root is required when output_dir is not provided")
        output_dir = processed_root / "tower_hp_test"

    processed = 0
    skipped = 0
    written_paths: list[Path] = []
    processed_sources: list[Path] = []

    for src_path in sorted(source_paths):
        try:
            with Image.open(src_path) as image:
                for region_name, rect in regions.items():
                    dst_name = f"{src_path.stem}__{region_name}.png"
                    dst_path = output_dir / region_name / dst_name
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    cropped = _crop_rect(image, rect)
                    cropped.save(dst_path, format="PNG")
                    written_paths.append(dst_path)
            if delete_source:
                src_path.unlink()
            processed += 1
            processed_sources.append(src_path)
        except ValueError:
            skipped += 1

    return CropResult(
        processed=processed,
        skipped=skipped,
        written_paths=written_paths,
        processed_sources=processed_sources,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop screenshots to all tower HP ROIs")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--layout-yaml", type=Path, default=Path("configs/screen_layout_reference.yaml"))
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--dataset-id", type=str, default="tower-hp-default")
    args = parser.parse_args()

    result = crop_tower_hp_images(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        layout_yaml=args.layout_yaml,
        delete_source=args.delete_source,
    )
    write_dataset_manifest(
        manifest_path=args.processed_root / "tower_hp_test" / "dataset_manifest.json",
        dataset_id=args.dataset_id,
        schema_version=1,
        source_root=args.raw_root,
        processed_root=args.processed_root,
        files=result.written_paths,
        extra={"script": "crop_tower_hp_images.py"},
    )
    print(
        "tower_hp: processed={} skipped={} crops_saved={}".format(
            result.processed,
            result.skipped,
            result.crops_saved,
        )
    )


if __name__ == "__main__":
    main()
