#!/usr/bin/env python3
"""
Evaluate a trained battlefield CNN on labeled PNGs (same ROI as training).

Labels come from filenames: ``true_*.png`` → battlefield, ``false_*.png`` → not.

Requires: ``pip install -r requirements-ml.txt``

Example (from repository root):

  python scripts/eval_battlefield_classifier.py \\
    --checkpoint artifacts/battlefield_cnn.pt \\
    --data-dir data/battlefield_test \\
    --layout-yaml configs/screen_layout_reference.yaml \\
    --threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from PIL import Image

from src.perception.battlefield_net import BattlefieldScreenNet
from src.perception.battlefield_roi import pil_rgb_masked_bottom_panel
from src.perception.screen_layout import load_screen_layout_reference


def _collect_samples(data_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for p in sorted(data_dir.glob("*.png")):
        name = p.name.lower()
        if name.startswith("true_"):
            samples.append((p, 1))
        elif name.startswith("false_"):
            samples.append((p, 0))
    return samples


def _load_tensor(path: Path, layout, size: int) -> torch.Tensor:
    im = Image.open(path)
    crop = pil_rgb_masked_bottom_panel(im, layout)
    crop = crop.resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score labeled PNGs with the battlefield CNN")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/battlefield_test"))
    parser.add_argument(
        "--layout-yaml",
        type=Path,
        default=Path("configs/screen_layout_reference.yaml"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Predict battlefield if sigmoid(logit) >= threshold",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    ckpt = _torch_load(args.checkpoint)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise SystemExit(f"Invalid checkpoint (expected state_dict): {args.checkpoint}")

    input_size = int(ckpt.get("input_size", 128))
    layout = load_screen_layout_reference(args.layout_yaml)
    samples = _collect_samples(args.data_dir)
    if not samples:
        raise SystemExit(f"No true_*.png / false_*.png under {args.data_dir}")

    device = torch.device("cpu")
    net = BattlefieldScreenNet().to(device)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()

    tp = fp = tn = fn = 0
    rows: list[tuple[str, int, float, int, bool]] = []

    with torch.inference_mode():
        for path, y_true in samples:
            x = _load_tensor(path, layout, input_size).unsqueeze(0).to(device)
            prob = float(torch.sigmoid(net(x)).squeeze().cpu())
            y_pred = 1 if prob >= args.threshold else 0
            ok = y_pred == y_true
            rows.append((path.name, y_true, prob, y_pred, ok))
            if y_true == 1 and y_pred == 1:
                tp += 1
            elif y_true == 0 and y_pred == 1:
                fp += 1
            elif y_true == 0 and y_pred == 0:
                tn += 1
            else:
                fn += 1

    correct = sum(1 for *_, ok in rows if ok)
    n = len(rows)
    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"threshold={args.threshold}  input_size={input_size}  n={n}")
    print(f"correct={correct}/{n}  accuracy={correct / n:.1%}")
    print(f"confusion  tp={tp} fp={fp} tn={tn} fn={fn}  (rows: true pos / false pos / true neg / false neg)")
    print()
    print(f"{'file':<28} {'label':>5} {'prob':>7} {'pred':>4} {'ok':>3}")
    for name, y_true, prob, y_pred, ok in rows:
        lab = "bf" if y_true == 1 else "no"
        prd = "bf" if y_pred == 1 else "no"
        print(f"{name:<28} {lab:>5} {prob:>7.3f} {prd:>4} {'yes' if ok else 'no':>3}")


if __name__ == "__main__":
    main()
