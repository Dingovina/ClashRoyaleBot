#!/usr/bin/env python3
"""
Train the tiny battlefield screen classifier on labeled PNGs.

Expects files named ``true_*.png`` (in-match) and ``false_*.png`` (not in-match), e.g. under
``data/battlefield_test/``.

Requires: ``pip install -r requirements-ml.txt``

Example:
  python scripts/train_battlefield_classifier.py --data-dir data/battlefield_test \\
    --runtime-yaml configs/runtime.yaml --out artifacts/battlefield_cnn.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import yaml

from src.perception.battlefield_net import BattlefieldScreenNet
from src.runtime.viewport import parse_game_viewport


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def _collect_samples(data_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for p in sorted(data_dir.glob("*.png")):
        name = p.name.lower()
        if name.startswith("true_"):
            samples.append((p, 1))
        elif name.startswith("false_"):
            samples.append((p, 0))
    if len(samples) < 4:
        raise SystemExit(f"Need at least 4 labeled PNGs in {data_dir}, found {len(samples)}")
    return samples


def _stratified_split(
    samples: list[tuple[Path, int]], *, val_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    by_cls: dict[int, list[Path]] = {0: [], 1: []}
    for path, y in samples:
        by_cls[y].append(path)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for cls, paths in by_cls.items():
        paths = paths.copy()
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_fraction)))
        if len(paths) <= n_val:
            n_val = max(1, len(paths) - 1) if len(paths) > 1 else 0
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]
        if not train_paths:
            train_paths = val_paths[:-1]
            val_paths = val_paths[-1:]
        train.extend((p, cls) for p in train_paths)
        val.extend((p, cls) for p in val_paths)
    return train, val


def _viewport_from_yaml(runtime_yaml: Path):
    data = _load_yaml(runtime_yaml)
    return parse_game_viewport(data["runtime"])


def _crop_viewport_rgb(image: Image.Image, viewport) -> Image.Image:
    w, h = image.size
    left, top, vw, vh = viewport.rect_for_frame(w, h)
    vw = min(vw, w - left)
    vh = min(vh, h - top)
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    box = (left, top, left + vw, top + vh)
    return image.crop(box).convert("RGB")


def _load_tensor(path: Path, viewport, size: int) -> torch.Tensor:
    im = Image.open(path)
    crop = _crop_viewport_rgb(im, viewport)
    crop = crop.resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/battlefield_test"))
    parser.add_argument("--runtime-yaml", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/battlefield_cnn.pt"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    viewport = _viewport_from_yaml(args.runtime_yaml)
    samples = _collect_samples(args.data_dir)
    train_items, val_items = _stratified_split(samples, val_fraction=args.val_fraction, seed=args.seed)

    device = torch.device("cpu")
    net = BattlefieldScreenNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def batch_tensors(items: list[tuple[Path, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [_load_tensor(p, viewport, args.input_size) for p, _ in items]
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
                "train_samples": len(train_items),
                "val_samples": len(val_items),
                "runtime_yaml": str(args.runtime_yaml),
            },
        },
        args.out,
    )
    print(f"Wrote {args.out.resolve()} (best val loss {best_val:.4f})")


if __name__ == "__main__":
    main()
