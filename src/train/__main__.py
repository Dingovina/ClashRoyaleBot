from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    config_path = Path("configs/train.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: expected mapping at top level")
    allowed_top = {"train"}
    unknown_top = sorted(k for k in config.keys() if k not in allowed_top)
    if unknown_top:
        raise ValueError(f"{config_path}: unknown top-level keys: {unknown_top}")
    train = config.get("train")
    if not isinstance(train, dict):
        raise ValueError(f"{config_path}: missing or invalid 'train' section")
    allowed_train = {"model_family"}
    unknown_train = sorted(k for k in train.keys() if k not in allowed_train)
    if unknown_train:
        raise ValueError(f"{config_path}: unknown train keys: {unknown_train}")

    model_family = str(train["model_family"]).strip()
    if not model_family:
        raise ValueError(f"{config_path}: train.model_family must not be empty")
    print(f"Train environment ready. model_family={model_family}")


if __name__ == "__main__":
    main()
