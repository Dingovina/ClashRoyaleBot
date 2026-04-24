from __future__ import annotations

import numpy as np
import torch

from src.perception.screen_layout import ScreenLayoutReference, intersect_pixel_rects


def bgra_masked_bottom_panel_rgb_tensor(
    frame_width: int,
    frame_height: int,
    pixels_bgra: bytes,
    layout: ScreenLayoutReference,
    out_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Crop ``bottom_panel`` from a fullscreen BGRA frame, zero-out HUD rects (hand slots,
    next card, elixir number digit), convert to RGB, resize to a square for the battlefield CNN.
    """
    bp = layout.bottom_panel
    l = max(0, min(bp.left, frame_width - 1))
    t = max(0, min(bp.top, frame_height - 1))
    r_excl = min(frame_width, bp.right + 1)
    b_excl = min(frame_height, bp.bottom + 1)
    ch = b_excl - t
    cw = r_excl - l
    if ch <= 1 or cw <= 1:
        return torch.zeros(1, 3, out_size, out_size, device=device, dtype=torch.float32)

    full = torch.frombuffer(bytearray(pixels_bgra), dtype=torch.uint8).reshape(frame_height, frame_width, 4)
    patch_bgr = full[t:b_excl, l:r_excl, :3].float().to(device=device) / 255.0

    mask = torch.ones((ch, cw), device=device, dtype=torch.float32)
    for rect in layout.hud_subtract_rects():
        inter = intersect_pixel_rects(rect, bp)
        if inter is None:
            continue
        ly0 = inter.top - t
        ly1 = inter.bottom - t
        lx0 = inter.left - l
        lx1 = inter.right - l
        ly0c = int(max(0, ly0))
        ly1c = int(min(ch - 1, ly1))
        lx0c = int(max(0, lx0))
        lx1c = int(min(cw - 1, lx1))
        if ly0c <= ly1c and lx0c <= lx1c:
            mask[ly0c : ly1c + 1, lx0c : lx1c + 1] = 0.0

    masked_bgr = patch_bgr * mask.unsqueeze(-1)
    rgb = masked_bgr[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
    return torch.nn.functional.interpolate(
        rgb, size=(out_size, out_size), mode="bilinear", align_corners=False
    )


def pil_rgb_masked_bottom_panel(image: object, layout: ScreenLayoutReference) -> object:
    """Same masking as CNN runtime path for PIL RGB training images.

    Accepts either:
    - a fullscreen screenshot that contains ``bottom_panel`` in absolute coordinates, or
    - an already-cropped ``bottom_panel`` image with exact panel size.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)  # type: ignore[union-attr]
    h, w, _ = arr.shape
    bp = layout.bottom_panel
    if w == bp.width and h == bp.height:
        crop = arr.copy()
        origin_x = bp.left
        origin_y = bp.top
    else:
        if w <= bp.left or h <= bp.top:
            raise ValueError(
                f"Image size {w}x{h} is too small for bottom_panel origin ({bp.left},{bp.top})"
            )
        l = max(0, min(bp.left, w - 1))
        t = max(0, min(bp.top, h - 1))
        r_in = min(w - 1, bp.right)
        b_in = min(h - 1, bp.bottom)
        crop = arr[t : b_in + 1, l : r_in + 1].copy()
        origin_x = l
        origin_y = t
    ch, cw, _ = crop.shape

    for rect in layout.hud_subtract_rects():
        inter = intersect_pixel_rects(rect, bp)
        if inter is None:
            continue
        ly0 = int(max(0, inter.top - origin_y))
        ly1 = int(min(ch - 1, inter.bottom - origin_y))
        lx0 = int(max(0, inter.left - origin_x))
        lx1 = int(min(cw - 1, inter.right - origin_x))
        if ly0 <= ly1 and lx0 <= lx1:
            crop[ly0 : ly1 + 1, lx0 : lx1 + 1] = 0

    from PIL import Image

    return Image.fromarray(crop)
