#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.roi.battlefield_roi import pil_rgb_masked_bottom_panel
from src.perception.roi.screen_layout import load_screen_layout_reference
from src.ml.manifest import write_dataset_manifest
from scripts.data.crop_result import CropResult


def _collect_pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.png") if p.is_file())


def crop_battlefield_images(
    *,
    raw_root: Path,
    processed_root: Path,
    layout_yaml: Path,
    delete_source: bool,
) -> CropResult:
    layout = load_screen_layout_reference(layout_yaml)
    processed = 0
    skipped = 0
    written_paths: list[Path] = []
    for subset in ("good", "bad"):
        src_dir = raw_root / "battlefield_test" / subset
        dst_dir = processed_root / "battlefield_test" / subset
        for src_path in _collect_pngs(src_dir):
            dst_path = dst_dir / src_path.name
            try:
                with Image.open(src_path) as image:
                    cropped = pil_rgb_masked_bottom_panel(image, layout)
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    cropped.save(dst_path, format="PNG")
                if delete_source:
                    src_path.unlink()
                processed += 1
                written_paths.append(dst_path)
            except ValueError:
                skipped += 1
    return CropResult(processed=processed, skipped=skipped, written_paths=written_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop battlefield screenshots to bottom-panel ROI")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--layout-yaml", type=Path, default=Path("configs/screen_layout_reference.yaml"))
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--dataset-id", type=str, default="battlefield-default")
    args = parser.parse_args()

    result = crop_battlefield_images(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        layout_yaml=args.layout_yaml,
        delete_source=args.delete_source,
    )
    write_dataset_manifest(
        manifest_path=args.processed_root / "battlefield_test" / "dataset_manifest.json",
        dataset_id=args.dataset_id,
        schema_version=1,
        source_root=args.raw_root,
        processed_root=args.processed_root,
        files=result.written_paths,
        extra={"script": "crop_battlefield_images.py"},
    )
    print(f"battlefield: processed={result.processed} skipped={result.skipped}")


if __name__ == "__main__":
    main()
