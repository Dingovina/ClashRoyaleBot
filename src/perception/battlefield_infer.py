from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.perception.battlefield_net import BattlefieldScreenNet
from src.perception.battlefield_roi import bgra_masked_bottom_panel_rgb_tensor
from src.runtime.screen_layout import ScreenLayoutReference, load_screen_layout_reference
from src.runtime.viewport import GameViewport


def viewport_crop_box(frame_width: int, frame_height: int, viewport: GameViewport) -> tuple[int, int, int, int]:
    """Returns (left, top, right, bottom) pixel box for the game client strip."""
    left, top, vw, vh = viewport.rect_for_frame(frame_width, frame_height)
    vw = min(vw, frame_width - left)
    vh = min(vh, frame_height - top)
    left = max(0, min(left, frame_width - 1))
    top = max(0, min(top, frame_height - 1))
    vw = max(0, min(vw, frame_width - left))
    vh = max(0, min(vh, frame_height - top))
    return (left, top, left + vw, top + vh)


def bgra_viewport_to_rgb_tensor(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    viewport: GameViewport,
    out_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Crop viewport from BGRA frame, resize to out_size square, return float tensor [1,3,H,W] in [0,1] RGB."""
    l, t, r, b = viewport_crop_box(frame_width, frame_height, viewport)
    if r <= l + 1 or b <= t + 1:
        return torch.zeros(1, 3, out_size, out_size, device=device)

    full = torch.frombuffer(bytearray(pixels_bgra), dtype=torch.uint8).reshape(frame_height, frame_width, 4)
    patch_bgr = full[t:b, l:r, :3].float() / 255.0
    rgb = patch_bgr[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    return torch.nn.functional.interpolate(
        rgb, size=(out_size, out_size), mode="bilinear", align_corners=False
    )


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

    def __init__(self, checkpoint_path: Path, layout_path: Path, logger: logging.Logger) -> None:
        self._logger = logger
        self._path = checkpoint_path
        self._layout_path = layout_path
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
