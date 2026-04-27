from __future__ import annotations

import re
from pathlib import Path

_NAME_RE = re.compile(r"^(?P<label>.+?)_(?P<id>[^_]+)\.png$", re.IGNORECASE)


def collect_card_labeled_pngs(data_dir: Path) -> list[tuple[Path, str]]:
    """
    Collect ``(path, label)`` card samples from per-card crops.

    Expected naming: ``<card-name>_<random-id>.png``.
    """
    if not data_dir.is_dir():
        raise ValueError(f"Expected existing directory with PNGs: {data_dir}")

    samples: list[tuple[Path, str]] = []
    invalid_names: list[str] = []

    for path in sorted(data_dir.glob("*.png")):
        match = _NAME_RE.match(path.name)
        if match is None:
            invalid_names.append(path.name)
            continue
        label = match.group("label").strip().lower()
        if not label:
            invalid_names.append(path.name)
            continue
        samples.append((path, label))

    if invalid_names:
        preview = ", ".join(invalid_names[:5])
        raise ValueError(
            f"Invalid card sample names in {data_dir}; expected <card-name>_<random-id>.png, got: {preview}"
        )
    if not samples:
        raise ValueError(f"No PNG samples found in {data_dir}")
    return samples
