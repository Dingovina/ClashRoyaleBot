from __future__ import annotations

import re
from pathlib import Path

_NAME_RE = re.compile(r"^(?P<label>\d+)_\d+\.png$", re.IGNORECASE)


def collect_elixir_labeled_pngs(data_dir: Path) -> list[tuple[Path, int]]:
    """
    Collect ``(path, label)`` samples for elixir digit training/eval.

    Expected file naming under ``data_dir``: ``<elixir>_<index>.png``,
    where ``<elixir>`` is in ``0..10`` (for example ``7_2.png``).
    """
    if not data_dir.is_dir():
        raise ValueError(f"Expected existing directory with PNGs: {data_dir}")

    samples: list[tuple[Path, int]] = []
    invalid_names: list[str] = []
    out_of_range: list[str] = []

    for path in sorted(data_dir.glob("*.png")):
        match = _NAME_RE.match(path.name)
        if match is None:
            invalid_names.append(path.name)
            continue
        label = int(match.group("label"))
        if label < 0 or label > 10:
            out_of_range.append(path.name)
            continue
        samples.append((path, label))

    if invalid_names:
        preview = ", ".join(invalid_names[:5])
        raise ValueError(
            f"Invalid elixir sample names in {data_dir}; expected <elixir>_<index>.png, got: {preview}"
        )
    if out_of_range:
        preview = ", ".join(out_of_range[:5])
        raise ValueError(f"Elixir label out of range [0..10] in: {preview}")
    if not samples:
        raise ValueError(f"No PNG samples found in {data_dir}")

    return samples
