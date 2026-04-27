from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.perception.models.card_net import CardHandNet
from src.perception.roi.card_roi import bgra_hand_card_rgb_tensor
from src.perception.roi.screen_layout import ScreenLayoutReference, load_screen_layout_reference


def _torch_load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


_layout_cache: dict[str, ScreenLayoutReference] = {}


def _get_screen_layout_reference(layout_path: Path) -> ScreenLayoutReference:
    key = str(layout_path.resolve())
    if key not in _layout_cache:
        _layout_cache[key] = load_screen_layout_reference(layout_path)
    return _layout_cache[key]


class CardModelRunner:
    """Loads hand-card classifier weights once and runs CPU inference on all 4 hand slots."""

    def __init__(self, checkpoint_path: Path, layout_path: Path, _logger: logging.Logger) -> None:
        ckpt = _torch_load_checkpoint(checkpoint_path)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "idx_to_label" not in ckpt:
            raise ValueError(f"Invalid checkpoint format (expected state_dict + idx_to_label): {checkpoint_path}")
        self.input_size = int(ckpt.get("input_size", 96))
        self.idx_to_label = [str(x) for x in ckpt["idx_to_label"]]
        if not self.idx_to_label:
            raise ValueError(f"Checkpoint has empty class list: {checkpoint_path}")
        meta = ckpt.get("meta", {})
        self.grayscale_input = bool(meta.get("grayscale_input", True)) if isinstance(meta, dict) else True
        self._layout = _get_screen_layout_reference(layout_path)
        self._net = CardHandNet(num_classes=len(self.idx_to_label))
        self._net.load_state_dict(ckpt["state_dict"], strict=True)
        self._net.eval()
        for p in self._net.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def infer_hand_cards(
        self,
        frame_width: int,
        frame_height: int,
        pixels_bgra: bytes,
    ) -> list[tuple[str, float]]:
        device = next(self._net.parameters()).device
        out: list[tuple[str, float]] = []
        for rect in self._layout.hand_cards:
            x = bgra_hand_card_rgb_tensor(
                frame_width=frame_width,
                frame_height=frame_height,
                pixels_bgra=pixels_bgra,
                rect=rect,
                out_size=self.input_size,
                device=device,
            )
            if self.grayscale_input:
                luminance = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0.0, 1.0)
                x = torch.cat([luminance, luminance, luminance], dim=1)
            probs = torch.softmax(self._net(x).squeeze(0), dim=0)
            conf, pred_idx = torch.max(probs, dim=0)
            out.append((self.idx_to_label[int(pred_idx.item())], float(conf.item())))
        return out


_cached: dict[tuple[str, str], CardModelRunner] = {}


def get_card_runner(checkpoint_path: Path, layout_path: Path, logger: logging.Logger) -> CardModelRunner:
    ck = str(checkpoint_path.resolve())
    lk = str(layout_path.resolve())
    key = (ck, lk)
    if key not in _cached:
        _cached[key] = CardModelRunner(checkpoint_path, layout_path, logger)
        logger.info(
            "card_model_loaded path=%s layout=%s input_size=%s classes=%s grayscale_input=%s",
            ck,
            lk,
            _cached[key].input_size,
            len(_cached[key].idx_to_label),
            _cached[key].grayscale_input,
        )
    return _cached[key]


def clear_card_runner_cache() -> None:
    _cached.clear()
    _layout_cache.clear()
