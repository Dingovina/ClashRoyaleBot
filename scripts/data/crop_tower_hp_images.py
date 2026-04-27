#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.roi.screen_layout import PixelRect, load_screen_layout_reference
from scripts.data.crop_result import CropResult

_REGION_TO_HP_INDEX: dict[str, int] = {
    "friendly_left_princess": 0,
    "friendly_right_princess": 1,
    "friendly_king": 2,
    "enemy_left_princess": 3,
    "enemy_right_princess": 4,
    "enemy_king": 5,
}


def _collect_pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.png") if p.is_file())


def _crop_rect(image: Image.Image, rect: PixelRect) -> Image.Image:
    # PIL crop uses half-open bounds: right/bottom are exclusive.
    return image.convert("RGB").crop((rect.left, rect.top, rect.right + 1, rect.bottom + 1))


def _target_filename_for_region(src_path: Path, region_name: str) -> str:
    tokens = src_path.stem.split("_")
    if len(tokens) < 6:
        raise ValueError(f"Unexpected source filename format: {src_path.name}")
    hp_value = tokens[_REGION_TO_HP_INDEX[region_name]]
    random_id = "_".join(tokens[6:]).strip()
    if not random_id:
        # Keep filenames stable even for old files without explicit random-id in stem.
        random_id = hashlib.sha1(src_path.stem.encode("utf-8")).hexdigest()[:10]
    return f"{hp_value}_{random_id}.png"


def _hp_value_for_region(src_path: Path, region_name: str) -> str:
    tokens = src_path.stem.split("_")
    if len(tokens) < 6:
        raise ValueError(f"Unexpected source filename format: {src_path.name}")
    return tokens[_REGION_TO_HP_INDEX[region_name]]


def crop_tower_hp_images(
    *,
    raw_root: Path | None,
    processed_root: Path | None,
    layout_yaml: Path,
    delete_source: bool,
    skip_none: bool = True,
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
                    hp_value = _hp_value_for_region(src_path, region_name)
                    if skip_none and hp_value.lower() == "none":
                        continue
                    dst_name = _target_filename_for_region(src_path, region_name)
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
    parser.add_argument("--processed-root", type=Path, default=Path("data"))
    parser.add_argument("--layout-yaml", type=Path, default=Path("configs/screen_layout_reference.yaml"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional directory with source PNG screenshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for cropped tower HP regions",
    )
    parser.add_argument(
        "--skip-none",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip crops where target tower HP value is 'none' (default: enabled)",
    )
    parser.add_argument("--delete-source", action="store_true")
    args = parser.parse_args()

    source_paths = None
    if args.source_dir is not None:
        source_paths = _collect_pngs(args.source_dir)

    result = crop_tower_hp_images(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        layout_yaml=args.layout_yaml,
        delete_source=args.delete_source,
        skip_none=args.skip_none,
        source_paths=source_paths,
        output_dir=args.output_dir,
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
