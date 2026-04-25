from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.perception.models.elixir_net import ElixirDigitNet
from src.perception.roi.elixir_roi import bgra_elixir_number_rgb_tensor
from src.perception.roi.screen_layout import ScreenLayoutReference, load_screen_layout_reference


def _torch_load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


_layout_cache: dict[str, ScreenLayoutReference] = {}


def get_screen_layout_reference(layout_path: Path) -> ScreenLayoutReference:
    key = str(layout_path.resolve())
    if key not in _layout_cache:
        _layout_cache[key] = load_screen_layout_reference(layout_path)
    return _layout_cache[key]


def clear_screen_layout_cache() -> None:
    _layout_cache.clear()


class ElixirModelRunner:
    """Loads elixir classifier weights once and runs CPU forward pass on ROI crop."""

    def __init__(self, checkpoint_path: Path, layout_path: Path, _logger: logging.Logger) -> None:
        ckpt = _torch_load_checkpoint(checkpoint_path)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise ValueError(f"Invalid checkpoint format (expected dict with state_dict): {checkpoint_path}")
        self.input_size = int(ckpt.get("input_size", 64))
        self.num_classes = int(ckpt.get("num_classes", 11))
        self._layout = get_screen_layout_reference(layout_path)
        self._net = ElixirDigitNet(num_classes=self.num_classes)
        self._net.load_state_dict(ckpt["state_dict"], strict=True)
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def infer_elixir(
        self,
        frame_width: int,
        frame_height: int,
        pixels_bgra: bytes,
    ) -> tuple[float, float]:
        device = next(self._net.parameters()).device
        x = bgra_elixir_number_rgb_tensor(
            frame_width,
            frame_height,
            pixels_bgra,
            self._layout,
            self.input_size,
            device=device,
        )
        logits = self._net(x).squeeze(0)
        probs = torch.softmax(logits, dim=0)
        conf, cls = torch.max(probs, dim=0)
        return (float(cls.item()), float(conf.item()))


_cached: dict[tuple[str, str], ElixirModelRunner] = {}


def get_elixir_runner(
    checkpoint_path: Path,
    layout_path: Path,
    logger: logging.Logger,
) -> ElixirModelRunner:
    ck = str(checkpoint_path.resolve())
    lk = str(layout_path.resolve())
    key = (ck, lk)
    if key not in _cached:
        _cached[key] = ElixirModelRunner(checkpoint_path, layout_path, logger)
        logger.info(
            "elixir_model_loaded path=%s layout=%s input_size=%s classes=%s",
            ck,
            lk,
            _cached[key].input_size,
            _cached[key].num_classes,
        )
    return _cached[key]


def clear_elixir_runner_cache() -> None:
    _cached.clear()
    clear_screen_layout_cache()
