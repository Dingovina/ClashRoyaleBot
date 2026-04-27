from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.perception.models.tower_hp_net import TowerHpNet
from src.perception.roi.screen_layout import PixelRect, ScreenLayoutReference, load_screen_layout_reference

_DIGIT_CHARSET = "0123456789"
_EMPTY_LABEL = "none"


def _torch_load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _bgra_tower_rgb_tensor(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    rect: PixelRect,
    *,
    out_width: int,
    out_height: int,
    device: torch.device,
) -> torch.Tensor:
    l = max(0, min(rect.left, frame_width - 1))
    t = max(0, min(rect.top, frame_height - 1))
    r_excl = min(frame_width, rect.right + 1)
    b_excl = min(frame_height, rect.bottom + 1)
    ch = b_excl - t
    cw = r_excl - l
    if ch <= 1 or cw <= 1:
        return torch.zeros(1, 3, out_height, out_width, device=device, dtype=torch.float32)

    full = torch.frombuffer(bytearray(pixels_bgra), dtype=torch.uint8).reshape(frame_height, frame_width, 4)
    patch_bgr = full[t:b_excl, l:r_excl, :3].float().to(device=device) / 255.0
    rgb = patch_bgr[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
    return torch.nn.functional.interpolate(
        rgb, size=(out_height, out_width), mode="bilinear", align_corners=False
    )


_layout_cache: dict[str, ScreenLayoutReference] = {}


def _get_layout(layout_path: Path) -> ScreenLayoutReference:
    key = str(layout_path.resolve())
    if key not in _layout_cache:
        _layout_cache[key] = load_screen_layout_reference(layout_path)
    return _layout_cache[key]


def clear_tower_hp_layout_cache() -> None:
    _layout_cache.clear()


class TowerHpModelRunner:
    def __init__(self, checkpoint_path: Path, layout_path: Path, _logger: logging.Logger) -> None:
        ckpt = _torch_load_checkpoint(checkpoint_path)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise ValueError(f"Invalid checkpoint format (expected dict with state_dict): {checkpoint_path}")
        self.input_width = int(ckpt.get("input_width", 128))
        self.input_height = int(ckpt.get("input_height", 32))
        self.blank_index = int(ckpt.get("blank_index", 10))
        self.presence_threshold = float(ckpt.get("presence_threshold", 0.5))
        self._layout = _get_layout(layout_path)
        digit_classes = self.blank_index + 1
        self._net = TowerHpNet(digit_classes=digit_classes)
        self._net.load_state_dict(ckpt["state_dict"], strict=True)
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad = False

    @staticmethod
    def _decode_digits(indices: list[int], *, blank_index: int) -> str:
        collapsed: list[int] = []
        prev = -1
        for idx in indices:
            if idx == prev:
                continue
            prev = idx
            if idx == blank_index:
                continue
            if idx < 0 or idx >= len(_DIGIT_CHARSET):
                continue
            collapsed.append(idx)
        if not collapsed:
            return ""
        return "".join(_DIGIT_CHARSET[idx] for idx in collapsed)

    @torch.inference_mode()
    def infer_tower_hp(
        self,
        frame_width: int,
        frame_height: int,
        pixels_bgra: bytes,
        tower_type: str,
    ) -> tuple[str, float]:
        rect = self._layout.tower_hp_regions.get(tower_type)
        if rect is None:
            raise ValueError(f"Unknown tower_type: {tower_type}")
        device = next(self._net.parameters()).device
        x = _bgra_tower_rgb_tensor(
            frame_width,
            frame_height,
            pixels_bgra,
            rect,
            out_width=self.input_width,
            out_height=self.input_height,
            device=device,
        )
        ctc_logits, presence_logits = self._net(x)
        presence_prob = float(torch.sigmoid(presence_logits.squeeze(0)).item())
        if presence_prob < self.presence_threshold:
            return (_EMPTY_LABEL, 1.0 - presence_prob)

        step_logits = ctc_logits.squeeze(0)
        step_probs = torch.softmax(step_logits, dim=1)
        step_conf, step_idx = torch.max(step_probs, dim=1)
        decoded = self._decode_digits(step_idx.tolist(), blank_index=self.blank_index)
        if not decoded:
            return (_EMPTY_LABEL, 1.0 - presence_prob)
        char_conf = float(step_conf.mean().item())
        return (decoded, min(presence_prob, char_conf))


_cached: dict[tuple[str, str], TowerHpModelRunner] = {}


def get_tower_hp_runner(checkpoint_path: Path, layout_path: Path, logger: logging.Logger) -> TowerHpModelRunner:
    ck = str(checkpoint_path.resolve())
    lk = str(layout_path.resolve())
    key = (ck, lk)
    if key not in _cached:
        _cached[key] = TowerHpModelRunner(checkpoint_path, layout_path, logger)
        logger.info(
            "tower_hp_model_loaded path=%s layout=%s input=%sx%s",
            ck,
            lk,
            _cached[key].input_width,
            _cached[key].input_height,
        )
    return _cached[key]


def clear_tower_hp_runner_cache() -> None:
    _cached.clear()
    clear_tower_hp_layout_cache()

