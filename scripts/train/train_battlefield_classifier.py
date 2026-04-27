#!/usr/bin/env python3
"""
Train the tiny battlefield screen classifier on cropped battlefield ROI PNGs.

Expects ``--train-data-dir/{good,bad}/*.png`` for training and ``--val-data-dir/{good,bad}/*.png`` for validation.

Requires: ``pip install -r requirements-ml.txt``

Example (run from the repository root so ``src`` resolves). In **PowerShell** use a single line: the
caret (^) is **not** a line-continuation character (unlike cmd.exe), so it is passed to Python and breaks argparse.

  python scripts/train/train_battlefield_classifier.py --train-data-dir data/processed/train/battlefield_train --val-data-dir data/processed/val/battlefield_val --out artifacts/battlefield_cnn.pt
"""
from __future__ import annotations

import argparse
import random
import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from PIL import Image

from src.perception.models.battlefield_net import BattlefieldScreenNet
from src.perception.datasets.battlefield_samples import collect_battlefield_labeled_pngs


def _collect_samples(data_dir: Path, *, min_count: int) -> list[tuple[Path, int]]:
    try:
        samples = collect_battlefield_labeled_pngs(data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(samples) < min_count:
        raise SystemExit(
            f"Need at least {min_count} labeled PNGs under {data_dir}/good and {data_dir}/bad, found {len(samples)}"
        )
    return samples


def _load_tensor(path: Path, size: int) -> torch.Tensor:
    im = Image.open(path)
    im = im.convert("RGB").resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-dir", type=Path, default=Path("data/processed/train/battlefield_train"))
    parser.add_argument("--val-data-dir", type=Path, default=Path("data/processed/val/battlefield_val"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/battlefield_cnn.pt"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", type=str, default="battlefield-default")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_items = _collect_samples(args.train_data_dir, min_count=4)
    val_items = _collect_samples(args.val_data_dir, min_count=2)

    device = torch.device("cpu")
    net = BattlefieldScreenNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def batch_tensors(items: list[tuple[Path, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [_load_tensor(p, args.input_size) for p, _ in items]
        ys = torch.tensor([y for _, y in items], dtype=torch.float32, device=device)
        return torch.stack(xs, dim=0), ys

    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        net.train()
        rng = random.Random(args.seed + epoch)
        order = list(range(len(train_items)))
        rng.shuffle(order)
        shuffled = [train_items[i] for i in order]
        total_loss = 0.0
        for start in range(0, len(shuffled), 4):
            chunk = shuffled[start : start + 4]
            if len(chunk) < 2:
                continue
            xb, yb = batch_tensors(chunk)
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(chunk)

        net.eval()
        with torch.inference_mode():
            vx, vy = batch_tensors(val_items)
            vloss = float(loss_fn(net(vx), vy))
        avg_train = total_loss / max(1, len(shuffled))
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch={epoch+1:3d} train_loss={avg_train:.4f} val_loss={vloss:.4f} best_val={best_val:.4f}")

    if best_state is None:
        best_state = net.state_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "input_size": args.input_size,
            "meta": {
                "schema_version": 1,
                "dataset_id": args.dataset_id,
                "train_samples": len(train_items),
                "val_samples": len(val_items),
                "train_data_dir": str(args.train_data_dir),
                "val_data_dir": str(args.val_data_dir),
                "roi": "masked_bottom_panel",
            },
        },
        args.out,
    )
    print(f"Wrote {args.out.resolve()} (best val loss {best_val:.4f}, meta={json.dumps({'dataset_id': args.dataset_id})})")


if __name__ == "__main__":
    main()
