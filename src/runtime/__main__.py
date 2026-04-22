from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.runtime.config import RuntimeConfig
from src.runtime.loop import RuntimeLoop


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

    config = RuntimeConfig.from_file(args.config)
    RuntimeLoop(config=config, logger=logger).run()


if __name__ == "__main__":
    main()
