from __future__ import annotations

import numpy as np
import torch

from src.perception.screen_layout import ScreenLayoutReference


def bgra_elixir_number_rgb_tensor(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    layout: ScreenLayoutReference,
    out_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Crop ``elixir_number`` from fullscreen BGRA, convert to RGB, resize to square tensor."""
    rect = layout.elixir_number
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


def pil_rgb_elixir_number(image: object, layout: ScreenLayoutReference) -> object:
    """Return ``elixir_number`` crop from a PIL RGB image.

    Accepts either:
    - a fullscreen screenshot that contains the absolute ``elixir_number`` rect, or
    - an already-cropped image whose size exactly matches ``elixir_number``.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)  # type: ignore[union-attr]
    h, w, _ = arr.shape
    rect = layout.elixir_number
    if w == rect.width and h == rect.height:
        from PIL import Image

        return Image.fromarray(arr.copy())
    if w <= rect.left or h <= rect.top:
        raise ValueError(
            f"Image size {w}x{h} is too small for elixir_number origin ({rect.left},{rect.top})"
        )
    l = max(0, min(rect.left, w - 1))
    t = max(0, min(rect.top, h - 1))
    r_in = min(w - 1, rect.right)
    b_in = min(h - 1, rect.bottom)
    crop = arr[t : b_in + 1, l : r_in + 1].copy()

    from PIL import Image

    return Image.fromarray(crop)
