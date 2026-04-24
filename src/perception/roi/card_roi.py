from __future__ import annotations

import numpy as np
import torch

from src.perception.roi.screen_layout import PixelRect, ScreenLayoutReference


def bgra_hand_card_rgb_tensor(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    rect: PixelRect,
    out_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Crop one hand-card slot from fullscreen BGRA and resize to a square RGB tensor."""
    l = max(0, min(rect.left, frame_width - 1))
    t = max(0, min(rect.top, frame_height - 1))
    r_excl = min(frame_width, rect.right + 1)
    b_excl = min(frame_height, rect.bottom + 1)
    ch = b_excl - t
    cw = r_excl - l
    if ch <= 1 or cw <= 1:
        return torch.zeros(1, 3, out_size, out_size, device=device, dtype=torch.float32)

    full = torch.frombuffer(bytearray(pixels_bgra), dtype=torch.uint8).reshape(frame_height, frame_width, 4)
    patch_bgr = full[t:b_excl, l:r_excl, :3].float().to(device=device) / 255.0
    rgb = patch_bgr[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
    return torch.nn.functional.interpolate(
        rgb, size=(out_size, out_size), mode="bilinear", align_corners=False
    )


def pil_rgb_hand_card(image: object, layout: ScreenLayoutReference, slot_index: int) -> object:
    """Return one hand-card slot crop from a fullscreen PIL RGB image."""
    if slot_index < 0 or slot_index >= len(layout.hand_cards):
        raise ValueError(f"slot_index out of range: {slot_index}")
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)  # type: ignore[union-attr]
    h, w, _ = arr.shape
    rect = layout.hand_cards[slot_index]
    if rect.right >= w or rect.bottom >= h:
        raise ValueError(
            f"Input image too small for fullscreen hand-card crop: got {w}x{h}, "
            f"need at least {(rect.right + 1)}x{(rect.bottom + 1)}"
        )
    crop = arr[rect.top : rect.bottom + 1, rect.left : rect.right + 1].copy()
    from PIL import Image

    return Image.fromarray(crop)
