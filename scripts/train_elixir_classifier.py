#!/usr/bin/env python3
"""
Train a tiny CNN that predicts current elixir value (0..10) from the elixir-number ROI.

Expects PNG files named ``<elixir>_<index>.png`` under ``--data-dir``.
Example: ``7_2.png`` means label 7.

The model sees only ``elixir_number`` from ``--layout-yaml``.

Requires: ``pip install -r requirements-ml.txt``

Example (run from repository root, one line):
  python scripts/train_elixir_classifier.py --data-dir data/processed/elixir_test --layout-yaml configs/screen_layout_reference.yaml --out artifacts/elixir_cnn.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from PIL import Image

from src.perception.elixir_net import ElixirDigitNet
from src.perception.elixir_roi import pil_rgb_elixir_number
from src.perception.elixir_samples import collect_elixir_labeled_pngs
from src.perception.screen_layout import load_screen_layout_reference


def _collect_samples(data_dir: Path) -> list[tuple[Path, int]]:
    try:
        samples = collect_elixir_labeled_pngs(data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(samples) < 8:
        raise SystemExit(f"Need at least 8 labeled PNGs under {data_dir}, found {len(samples)}")
    return samples


def _stratified_split(
    samples: list[tuple[Path, int]], *, val_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    by_cls: dict[int, list[Path]] = defaultdict(list)
    for path, y in samples:
        by_cls[y].append(path)

    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for cls in sorted(by_cls):
        paths = by_cls[cls].copy()
        rng.shuffle(paths)
        if len(paths) == 1:
            train.append((paths[0], cls))
            continue
        n_val = max(1, int(round(len(paths) * val_fraction)))
        n_val = min(n_val, len(paths) - 1)
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]
        train.extend((p, cls) for p in train_paths)
        val.extend((p, cls) for p in val_paths)
    return train, val


def _load_tensor(path: Path, layout, size: int) -> torch.Tensor:
    im = Image.open(path)
    crop = pil_rgb_elixir_number(im, layout)
    crop = crop.resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train elixir digit CNN (0..10)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/elixir_test"))
    parser.add_argument(
        "--layout-yaml",
        type=Path,
        default=Path("configs/screen_layout_reference.yaml"),
        help="Screen layout with elixir_number rect",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/elixir_cnn.pt"))
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    layout = load_screen_layout_reference(args.layout_yaml)
    samples = _collect_samples(args.data_dir)
    train_items, val_items = _stratified_split(samples, val_fraction=args.val_fraction, seed=args.seed)

    device = torch.device("cpu")
    net = ElixirDigitNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    def batch_tensors(items: list[tuple[Path, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [_load_tensor(p, layout, args.input_size) for p, _ in items]
        ys = torch.tensor([y for _, y in items], dtype=torch.long, device=device)
        return torch.stack(xs, dim=0), ys

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        net.train()
        rng = random.Random(args.seed + epoch)
        order = list(range(len(train_items)))
        rng.shuffle(order)
        shuffled = [train_items[i] for i in order]

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for start in range(0, len(shuffled), args.batch_size):
            chunk = shuffled[start : start + args.batch_size]
            xb, yb = batch_tensors(chunk)
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach()) * len(chunk)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == yb).sum().item())
            total_seen += len(chunk)

        net.eval()
        with torch.inference_mode():
            if val_items:
                vx, vy = batch_tensors(val_items)
                vlogits = net(vx)
                vloss = float(loss_fn(vlogits, vy))
                vacc = float((vlogits.argmax(dim=1) == vy).float().mean().item())
            else:
                vloss = float("nan")
                vacc = float("nan")

        avg_train_loss = total_loss / max(1, total_seen)
        train_acc = total_correct / max(1, total_seen)
        improved = False
        if val_items:
            if vloss < best_val_loss:
                improved = True
        else:
            if avg_train_loss < best_val_loss:
                improved = True
        if improved:
            best_val_loss = vloss if val_items else avg_train_loss
            best_val_acc = vacc if val_items else train_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_items:
                print(
                    "epoch={:3d} train_loss={:.4f} train_acc={:.1%} val_loss={:.4f} "
                    "val_acc={:.1%} best_val_loss={:.4f} best_val_acc={:.1%}".format(
                        epoch + 1, avg_train_loss, train_acc, vloss, vacc, best_val_loss, best_val_acc
                    )
                )
            else:
                print(
                    "epoch={:3d} train_loss={:.4f} train_acc={:.1%} best_loss={:.4f}".format(
                        epoch + 1, avg_train_loss, train_acc, best_val_loss
                    )
                )

    if best_state is None:
        best_state = net.state_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "input_size": args.input_size,
            "num_classes": 11,
            "meta": {
                "train_samples": len(train_items),
                "val_samples": len(val_items),
                "layout_yaml": str(args.layout_yaml),
                "roi": "elixir_number",
                "labels": "filename_prefix_0_to_10",
            },
        },
        args.out,
    )
    print(f"Wrote {args.out.resolve()} (best_loss={best_val_loss:.4f}, best_acc={best_val_acc:.1%})")


if __name__ == "__main__":
    main()
