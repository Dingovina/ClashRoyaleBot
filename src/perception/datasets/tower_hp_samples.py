from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_TOWER_TYPES = (
    "friendly_left_princess",
    "friendly_right_princess",
    "friendly_king",
    "enemy_left_princess",
    "enemy_right_princess",
    "enemy_king",
)

TOWER_GROUPS: dict[str, tuple[str, ...]] = {
    "all": _TOWER_TYPES,
    "king": ("friendly_king", "enemy_king"),
    "princess": (
        "friendly_left_princess",
        "friendly_right_princess",
        "enemy_left_princess",
        "enemy_right_princess",
    ),
}

EMPTY_TOKEN = "EMPTY"


@dataclass(frozen=True)
class TowerHpSample:
    path: Path
    tower_type: str
    hp_text: str
    is_empty: bool


def collect_tower_hp_samples(data_dir: Path) -> list[TowerHpSample]:
    """
    Collect tower HP samples from per-tower folders.

    Expected layout:
    - ``<data_dir>/<tower_type>/*.png`` where ``tower_type`` is one of the 6 known regions.
    - filename format: ``<hp>_<random-id>.png``.
      - ``<hp>`` is either a number (for example ``2406``) or ``none``.
    """
    if not data_dir.is_dir():
        raise ValueError(f"Expected existing directory with tower hp folders: {data_dir}")

    samples: list[TowerHpSample] = []
    invalid_names: list[str] = []
    missing_folders: list[str] = []

    for tower_type in _TOWER_TYPES:
        tower_dir = data_dir / tower_type
        if not tower_dir.is_dir():
            missing_folders.append(tower_type)
            continue

        for path in sorted(tower_dir.glob("*.png")):
            stem = path.stem.strip()
            if "_" not in stem:
                invalid_names.append(f"{tower_type}/{path.name}")
                continue
            hp_raw, random_id = stem.split("_", 1)
            hp = hp_raw.strip().lower()
            rid = random_id.strip()
            if not rid:
                invalid_names.append(f"{tower_type}/{path.name}")
                continue
            if hp == "none":
                samples.append(TowerHpSample(path=path, tower_type=tower_type, hp_text=EMPTY_TOKEN, is_empty=True))
                continue
            if not hp.isdigit():
                invalid_names.append(f"{tower_type}/{path.name}")
                continue
            samples.append(TowerHpSample(path=path, tower_type=tower_type, hp_text=hp, is_empty=False))

    if missing_folders:
        preview = ", ".join(missing_folders)
        raise ValueError(f"Missing tower hp subfolders in {data_dir}: {preview}")
    if invalid_names:
        preview = ", ".join(invalid_names[:5])
        raise ValueError(f"Invalid tower hp sample names in {data_dir}; expected <hp>_<random-id>.png, got: {preview}")
    if not samples:
        raise ValueError(f"No PNG samples found in {data_dir}")

    return samples


def filter_tower_hp_samples_by_group(samples: list[TowerHpSample], tower_group: str) -> list[TowerHpSample]:
    group = tower_group.strip().lower()
    allowed = TOWER_GROUPS.get(group)
    if allowed is None:
        raise ValueError(f"Unknown tower_group: {tower_group}. Expected one of: {sorted(TOWER_GROUPS)}")
    allowed_set = set(allowed)
    filtered = [sample for sample in samples if sample.tower_type in allowed_set]
    if not filtered:
        raise ValueError(f"No samples found for tower_group={group}")
    return filtered

