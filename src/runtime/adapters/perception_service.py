from __future__ import annotations

import logging
from dataclasses import dataclass

from src.runtime.card_evaluate import infer_hand_cards
from src.runtime.capture import FrameObservation
from src.runtime.elixir_evaluate import infer_elixir_value
from src.runtime.runtime_config import RuntimeConfig
from src.runtime.types import InferenceStatus, PerceptionSnapshot


@dataclass
class RuntimePerceptionService:
    config: RuntimeConfig
    logger: logging.Logger

    def infer(self, frame: FrameObservation) -> PerceptionSnapshot:
        elixir_estimate = 0.0
        elixir_conf = 0.0
        elixir_status = InferenceStatus.DISABLED
        if self.config.elixir_model_enabled:
            elixir_estimate = 0.0
            elixir_status = InferenceStatus.NO_PIXELS
            if (
                frame.pixels_bgra
                and self.config.elixir_model_path
                and frame.width > 0
                and frame.height > 0
            ):
                inferred_elixir, inferred_conf = infer_elixir_value(
                    frame_width=frame.width,
                    frame_height=frame.height,
                    pixels_bgra=frame.pixels_bgra,
                    model_path=self.config.elixir_model_path,
                    model_layout_path=self.config.elixir_model_layout_path,
                    logger=self.logger,
                )
                if inferred_conf > 0.0:
                    elixir_estimate = max(0.0, min(10.0, inferred_elixir))
                    elixir_conf = inferred_conf
                    elixir_status = InferenceStatus.OK
                else:
                    elixir_status = InferenceStatus.FAILED

        hand_cards: tuple[str, str, str, str] = ("unknown", "unknown", "unknown", "unknown")
        hand_confidences: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        hand_status = InferenceStatus.DISABLED
        if self.config.card_model_enabled:
            hand_status = InferenceStatus.NO_PIXELS
            if (
                frame.pixels_bgra
                and self.config.card_model_path
                and frame.width > 0
                and frame.height > 0
            ):
                inferred = infer_hand_cards(
                    frame_width=frame.width,
                    frame_height=frame.height,
                    pixels_bgra=frame.pixels_bgra,
                    model_path=self.config.card_model_path,
                    model_layout_path=self.config.card_model_layout_path,
                    logger=self.logger,
                )
                if len(inferred) == 4:
                    hand_cards = tuple(name for name, _ in inferred)  # type: ignore[assignment]
                    hand_confidences = tuple(conf for _, conf in inferred)  # type: ignore[assignment]
                    hand_status = InferenceStatus.OK
                else:
                    hand_status = InferenceStatus.FAILED

        return PerceptionSnapshot(
            elixir=elixir_estimate,
            elixir_confidence=elixir_conf,
            elixir_status=elixir_status,
            hand_cards=hand_cards,
            hand_confidences=hand_confidences,
            hand_status=hand_status,
        )
