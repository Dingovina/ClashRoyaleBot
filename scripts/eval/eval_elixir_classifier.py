#!/usr/bin/env python3
"""
Evaluate elixir classifier checkpoint on labeled cropped elixir ROI PNGs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.datasets.elixir_samples import collect_elixir_labeled_pngs
from src.perception.models.elixir_net import ElixirDigitNet


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate elixir classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/train/elixir_test"))
    args = parser.parse_args()

    ckpt = _torch_load(args.checkpoint)
    if "state_dict" not in ckpt:
        raise SystemExit(f"Invalid checkpoint format: {args.checkpoint}")

    num_classes = int(ckpt.get("num_classes", 11))
    input_size = int(ckpt.get("input_size", 64))
    labels = list(range(num_classes))
    net = ElixirDigitNet(num_classes=num_classes)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()

    try:
        samples = collect_elixir_labeled_pngs(args.data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rows: list[tuple[str, int, int, float, bool]] = []
    confusion: list[list[int]] = [[0 for _ in labels] for _ in labels]
    correct = 0
    with torch.inference_mode():
        for path, y_true in samples:
            x = _load_tensor(path, input_size).unsqueeze(0)
            probs = torch.softmax(net(x).squeeze(0), dim=0)
            conf, pred_idx = torch.max(probs, dim=0)
            y_pred = int(pred_idx.item())
            ok = y_pred == y_true
            if 0 <= y_true < num_classes and 0 <= y_pred < num_classes:
                confusion[y_true][y_pred] += 1
            rows.append((path.name, y_true, y_pred, float(conf.item()), ok))
            if ok:
                correct += 1

    n = len(rows)
    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"n={n} correct={correct} accuracy={correct / max(1, n):.1%}")
    print()
    print("per_class_recall:")
    for y in labels:
        tp = confusion[y][y]
        total_true = sum(confusion[y])
        recall = tp / total_true if total_true else 0.0
        print(f"  {y:>2} recall={recall:>6.1%}  support={total_true}")
    print()
    print("confusion_matrix (rows=true, cols=pred):")
    print("  true\\pred " + " ".join(f"{y:>4d}" for y in labels))
    for y_true in labels:
        row = " ".join(f"{confusion[y_true][y_pred]:>4d}" for y_pred in labels)
        print(f"  {y_true:>9d} {row}")
    print()
    print(f"{'file':<36} {'true':>4} {'pred':>4} {'conf':>7} {'ok':>3}")
    for file_name, y_true, y_pred, conf, ok in rows:
        print(f"{file_name:<36} {y_true:>4d} {y_pred:>4d} {conf:>7.3f} {'yes' if ok else 'no':>3}")


if __name__ == "__main__":
    main()
