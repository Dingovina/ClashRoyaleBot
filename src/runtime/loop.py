from __future__ import annotations

import logging
from dataclasses import dataclass

from src.runtime.runtime_config import RuntimeConfig
from src.runtime.runtime_service import RuntimeService


@dataclass
class RuntimeLoop:
    config: RuntimeConfig
    logger: logging.Logger

    def run(self) -> int:
        return RuntimeService(config=self.config, logger=self.logger).run()
