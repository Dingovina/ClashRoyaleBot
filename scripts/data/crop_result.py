from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CropResult:
    processed: int
    skipped: int
    written_paths: list[Path]

    @property
    def crops_saved(self) -> int:
        return len(self.written_paths)
