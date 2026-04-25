#!/usr/bin/env python3
"""
Evaluate a trained battlefield CNN on labeled cropped PNGs.

Labels: ``good/*.png`` → battlefield, ``bad/*.png`` → not (under ``--data-dir``).

Requires: ``pip install -r requirements-ml.txt``

Example (from repository root, one line):

  python scripts/eval/eval_battlefield_classifier.py --checkpoint artifacts/battlefield_cnn.pt --data-dir data/processed/train/battlefield_train --threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from PIL import Image

from src.perception.models.battlefield_net import BattlefieldScreenNet
from src.perception.datasets.battlefield_samples import collect_battlefield_labeled_pngs


def _collect_samples(data_dir: Path) -> list[tuple[Path, int]]:
    try:
        return collect_battlefield_labeled_pngs(data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _load_tensor(path: Path, size: int) -> torch.Tensor:
    im = Image.open(path)
    im = im.convert("RGB").resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score labeled PNGs with the battlefield CNN")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/val/battlefield_val"))
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
    samples = _collect_samples(args.data_dir)
    if not samples:
        raise SystemExit(f"No labeled PNGs under {args.data_dir}/good and {args.data_dir}/bad")

    device = torch.device("cpu")
    net = BattlefieldScreenNet().to(device)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()

    tp = fp = tn = fn = 0
    rows: list[tuple[str, int, float, int, bool]] = []

    with torch.inference_mode():
        for path, y_true in samples:
            x = _load_tensor(path, input_size).unsqueeze(0).to(device)
            prob = float(torch.sigmoid(net(x)).squeeze().cpu())
            y_pred = 1 if prob >= args.threshold else 0
            ok = y_pred == y_true
            root = args.data_dir.resolve()
            try:
                rel = str(path.resolve().relative_to(root))
            except ValueError:
                rel = path.name
            rows.append((rel, y_true, prob, y_pred, ok))
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
    recall_bf = tp / (tp + fn) if (tp + fn) else 0.0
    recall_no = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"threshold={args.threshold}  input_size={input_size}  n={n}")
    print(f"correct={correct}/{n}  accuracy={correct / n:.1%}")
    print("per_class_recall:")
    print(f"  battlefield recall={recall_bf:.1%}  support={tp + fn}")
    print(f"  non_battlefield recall={recall_no:.1%}  support={tn + fp}")
    print("confusion_matrix (rows=true, cols=pred):")
    print("  true\\pred        bf       no")
    print(f"         bf {tp:8d} {fn:8d}")
    print(f"         no {fp:8d} {tn:8d}")
    print()
    print(f"{'file':<40} {'label':>5} {'prob':>7} {'pred':>4} {'ok':>3}")
    for name, y_true, prob, y_pred, ok in rows:
        lab = "bf" if y_true == 1 else "no"
        prd = "bf" if y_pred == 1 else "no"
        print(f"{name:<40} {lab:>5} {prob:>7.3f} {prd:>4} {'yes' if ok else 'no':>3}")


if __name__ == "__main__":
    main()
