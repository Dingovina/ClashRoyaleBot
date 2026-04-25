#!/usr/bin/env python3
"""
Evaluate hand-card classifier checkpoint on labeled card crops.
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

from src.perception.models.card_net import CardHandNet
from src.perception.datasets.card_samples import collect_card_labeled_pngs


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_tensor(path: Path, size: int, *, grayscale_input: bool) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if grayscale_input:
        image = image.convert("L").convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate card classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/val/cards_val"))
    parser.add_argument(
        "--rgb-input",
        action="store_true",
        help="Force RGB input; otherwise uses checkpoint grayscale setting or grayscale by default",
    )
    args = parser.parse_args()

    ckpt = _torch_load(args.checkpoint)
    if "state_dict" not in ckpt or "idx_to_label" not in ckpt:
        raise SystemExit(f"Invalid checkpoint format: {args.checkpoint}")
    idx_to_label: list[str] = list(ckpt["idx_to_label"])
    label_to_idx: dict[str, int] = {name: i for i, name in enumerate(idx_to_label)}
    input_size = int(ckpt.get("input_size", 96))
    grayscale_input = bool(ckpt.get("meta", {}).get("grayscale_input", True)) and not args.rgb_input

    net = CardHandNet(num_classes=len(idx_to_label))
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()

    try:
        samples = collect_card_labeled_pngs(args.data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rows: list[tuple[str, str, str, float, bool]] = []
    confusion: dict[str, dict[str, int]] = {label: {p: 0 for p in idx_to_label} for label in idx_to_label}
    unknown_true_labels: set[str] = set()
    correct = 0
    with torch.inference_mode():
        for path, true_name in samples:
            if true_name not in label_to_idx:
                rows.append((path.name, true_name, "<unknown-class>", 0.0, False))
                unknown_true_labels.add(true_name)
                continue
            x = _load_tensor(path, input_size, grayscale_input=grayscale_input).unsqueeze(0)
            probs = torch.softmax(net(x).squeeze(0), dim=0)
            conf, pred_idx = torch.max(probs, dim=0)
            pred_name = idx_to_label[int(pred_idx.item())]
            ok = pred_name == true_name
            confusion[true_name][pred_name] += 1
            rows.append((path.name, true_name, pred_name, float(conf.item()), ok))
            if ok:
                correct += 1

    n = len(rows)
    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"grayscale_input={grayscale_input}")
    print(f"n={n} correct={correct} accuracy={correct / max(1, n):.1%}")
    if unknown_true_labels:
        print(f"unknown_true_labels={sorted(unknown_true_labels)}")
    print()
    print("per_class_recall:")
    for label in idx_to_label:
        tp = confusion[label][label]
        total_true = sum(confusion[label].values())
        recall = tp / total_true if total_true else 0.0
        print(f"  {label:<18} recall={recall:>6.1%}  support={total_true}")
    print()
    print("confusion_matrix (rows=true, cols=pred):")
    print("  true\\pred " + " ".join(f"{label:>8}" for label in idx_to_label))
    for true_label in idx_to_label:
        row = " ".join(f"{confusion[true_label][pred_label]:>8d}" for pred_label in idx_to_label)
        print(f"  {true_label:>9} {row}")
    print()
    print(f"{'file':<44} {'true':<18} {'pred':<18} {'conf':>7} {'ok':>3}")
    for file_name, y_true, y_pred, conf, ok in rows:
        print(f"{file_name:<44} {y_true:<18} {y_pred:<18} {conf:>7.3f} {'yes' if ok else 'no':>3}")


if __name__ == "__main__":
    main()
