from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    config_path = Path("configs/train.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_family = config["train"]["model_family"]
    print(f"Train environment ready. model_family={model_family}")


if __name__ == "__main__":
    main()
