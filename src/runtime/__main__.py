from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.runtime.config.config_loader import load_runtime_config
from src.runtime.engine.runtime_service import RuntimeService


def main() -> None:
    parser = argparse.ArgumentParser(description="Clash Royale Bot runtime loop")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.yaml"),
        help="Path to runtime yaml config",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("runtime")

    config = load_runtime_config(args.config)
    exit_code = RuntimeService(config=config, logger=logger).run()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
