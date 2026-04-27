#!/usr/bin/env python3
"""Crop manually validated raw match screenshots into train/val/test datasets."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.data.crop_battlefield_images import crop_battlefield_images
from scripts.data.crop_elixir_images import crop_elixir_images
from scripts.data.crop_card_images import crop_card_images


def _cards_dir_name(split: str) -> str:
    return f"cards_{split}"


def _battlefield_dir_name(split: str) -> str:
    return f"battlefield_{split}"


def _elixir_dir_name(split: str) -> str:
    return f"elixir_{split}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop validated match screenshots into split datasets")
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
        "--match-id",
        type=str,
        required=True,
        help="Match identifier: raw files are read from <raw-root>/<match-id>",
    )
    parser.add_argument("--card", action="store_true", help="Generate card crops")
    parser.add_argument("--bf", action="store_true", help="Generate battlefield crops (always to good)")
    parser.add_argument("--elixir", action="store_true", help="Generate elixir crops")
    parser.add_argument("--train", action="store_true", help="Write outputs into split 'train'")
    parser.add_argument("--val", action="store_true", help="Write outputs into split 'val'")
    parser.add_argument("--test", action="store_true", help="Write outputs into split 'test'")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Output root for split model-ready PNGs",
    )
    parser.add_argument(
        "--cards-id-bytes",
        type=int,
        default=5,
        help="Random suffix size for generated card crop filenames",
    )
    args = parser.parse_args()

    selected_targets = [name for name, on in (("card", args.card), ("bf", args.bf), ("elixir", args.elixir)) if on]
    if not selected_targets:
        raise SystemExit("Select at least one model target: --card, --bf, --elixir")
    selected_splits = [name for name, on in (("train", args.train), ("val", args.val), ("test", args.test)) if on]
    if len(selected_splits) != 1:
        raise SystemExit("Select exactly one split: one of --train, --val, --test")
    split = selected_splits[0]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", args.match_id):
        raise SystemExit("match-id must match ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

    raw_match_dir = args.raw_root / args.match_id
    if not raw_match_dir.is_dir():
        raise SystemExit(f"Raw match directory does not exist: {raw_match_dir}")

    source_files = sorted(p for p in raw_match_dir.glob("*.png") if p.is_file())
    to_process = [p for p in source_files if not p.name.startswith("CHECK_")]
    skipped_check = len(source_files) - len(to_process)
    if args.bf:
        print("warning: --bf always writes into battlefield_<split>/good")

    total_processed = 0
    total_skipped = skipped_check
    all_written_paths: list[Path] = []
    processed_by_target: dict[str, set[Path]] = {}

    split_root = args.processed_root / split

    if args.bf:
        result = crop_battlefield_images(
            raw_root=None,
            processed_root=None,
            layout_yaml=args.layout_yaml,
            delete_source=False,
            source_paths=to_process,
            output_good_dir=split_root / _battlefield_dir_name(split) / "good",
            include_bad_subset=False,
        )
        total_processed += result.processed
        total_skipped += result.skipped
        all_written_paths.extend(result.written_paths)
        processed_by_target["bf"] = set(result.processed_sources)
        print(f"battlefield: processed={result.processed} skipped={result.skipped}")
    if args.elixir:
        result = crop_elixir_images(
            raw_root=None,
            processed_root=None,
            layout_yaml=args.layout_yaml,
            delete_source=False,
            source_paths=to_process,
            output_dir=split_root / _elixir_dir_name(split),
        )
        total_processed += result.processed
        total_skipped += result.skipped
        all_written_paths.extend(result.written_paths)
        processed_by_target["elixir"] = set(result.processed_sources)
        print(f"elixir: processed={result.processed} skipped={result.skipped}")
    if args.card:
        result = crop_card_images(
            raw_root=None,
            processed_root=None,
            layout_yaml=args.layout_yaml,
            id_bytes=args.cards_id_bytes,
            delete_source=False,
            source_paths=to_process,
            output_dir=split_root / _cards_dir_name(split),
        )
        total_processed += result.processed
        total_skipped += result.skipped
        all_written_paths.extend(result.written_paths)
        processed_by_target["card"] = set(result.processed_sources)
        print(
            f"cards: processed={result.processed} skipped={result.skipped} "
            f"crops_saved={result.crops_saved}"
        )

    files_done = set(to_process)
    for target in selected_targets:
        done = processed_by_target.get(target, set())
        files_done &= done
    for src in sorted(files_done):
        src.unlink(missing_ok=True)

    print(
        "done: processed={} skipped={} deleted_raw={} raw_dir={} processed_root={}".format(
            total_processed,
            total_skipped,
            len(files_done),
            raw_match_dir,
            split_root,
        )
    )


if __name__ == "__main__":
    main()
