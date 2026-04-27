from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.perception.models.battlefield_net import BattlefieldScreenNet
from src.perception.roi.battlefield_roi import bgra_masked_bottom_panel_rgb_tensor
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


class BattlefieldModelRunner:
    """Loads weights once and runs forward on CPU (masked bottom-panel ROI)."""

    def __init__(self, checkpoint_path: Path, layout_path: Path, _logger: logging.Logger) -> None:
        ckpt = _torch_load_checkpoint(checkpoint_path)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise ValueError(f"Invalid checkpoint format (expected dict with state_dict): {checkpoint_path}")
        self.input_size = int(ckpt.get("input_size", 128))
        self._layout = get_screen_layout_reference(layout_path)
        self._net = BattlefieldScreenNet()
        self._net.load_state_dict(ckpt["state_dict"], strict=True)
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def probability_battlefield(
        self,
        frame_width: int,
        frame_height: int,
        pixels_bgra: bytes,
    ) -> float:
        device = next(self._net.parameters()).device
        x = bgra_masked_bottom_panel_rgb_tensor(
            frame_width,
            frame_height,
            pixels_bgra,
            self._layout,
            self.input_size,
            device=device,
        )
        logit = self._net(x)
        return float(torch.sigmoid(logit).squeeze().cpu())


_cached: dict[tuple[str, str], BattlefieldModelRunner] = {}


def get_battlefield_runner(
    checkpoint_path: Path,
    layout_path: Path,
    logger: logging.Logger,
) -> BattlefieldModelRunner:
    ck = str(checkpoint_path.resolve())
    lk = str(layout_path.resolve())
    key = (ck, lk)
    if key not in _cached:
        _cached[key] = BattlefieldModelRunner(checkpoint_path, layout_path, logger)
        logger.info(
            "battlefield_model_loaded path=%s layout=%s input_size=%s",
            ck,
            lk,
            _cached[key].input_size,
        )
    return _cached[key]


def clear_battlefield_runner_cache() -> None:
    _cached.clear()
    clear_screen_layout_cache()
