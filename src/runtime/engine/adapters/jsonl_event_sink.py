from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.runtime.domain.ports import TickEvent
from src.runtime.config.runtime_config import RuntimeConfig


@dataclass
class JsonlTickEventSink:
    config: RuntimeConfig
    logger: logging.Logger

    def publish_tick(self, event: TickEvent) -> None:
        if not self.config.hand_tick_log_enabled:
            return
        path = Path(self.config.hand_tick_log_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "session_id": self.config.session_id,
                "tick": event.tick_id,
                "cards": list(event.perception.hand_cards),
                "confidences": [round(x, 4) for x in event.perception.hand_confidences],
                "candidate": event.candidate_name,
                "candidate_confidence": event.candidate_confidence,
                "decision": event.decision.action_type.value,
                "decision_reason": event.decision.reason.value,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        except Exception as exc:
            self.logger.warning("tick_event_write_failed path=%s err=%s", path, exc)
