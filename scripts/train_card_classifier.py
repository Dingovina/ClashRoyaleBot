#!/usr/bin/env python3
"""
Train a CNN that predicts card identity from hand-slot crops.

Expects files named ``<card-name>_<random-id>.png`` under ``--data-dir``.
Example: ``mini-p.e.k.k.a_a1b2c3.png``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.card_net import CardHandNet
from src.perception.card_samples import collect_card_labeled_pngs
from src.ml.manifest import write_artifact_manifest


def _collect_samples(data_dir: Path) -> list[tuple[Path, str]]:
    try:
        samples = collect_card_labeled_pngs(data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(samples) < 20:
        raise SystemExit(f"Need at least 20 labeled card PNGs under {data_dir}, found {len(samples)}")
    return samples


def _label_index(samples: list[tuple[Path, str]]) -> tuple[dict[str, int], list[str]]:
    labels = sorted({name for _, name in samples})
    if len(labels) < 2:
        raise SystemExit("Need at least 2 distinct card labels for training")
    return {name: i for i, name in enumerate(labels)}, labels


def _stratified_split(
    samples: list[tuple[Path, str]], val_fraction: float, seed: int
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    rng = random.Random(seed)
    by_cls: dict[str, list[Path]] = defaultdict(list)
    for path, name in samples:
        by_cls[name].append(path)

    train: list[tuple[Path, str]] = []
    val: list[tuple[Path, str]] = []
    for cls in sorted(by_cls):
        paths = by_cls[cls].copy()
        rng.shuffle(paths)
        if len(paths) == 1:
            train.append((paths[0], cls))
            continue
        n_val = max(1, int(round(len(paths) * val_fraction)))
        n_val = min(n_val, len(paths) - 1)
        val.extend((p, cls) for p in paths[:n_val])
        train.extend((p, cls) for p in paths[n_val:])
    return train, val


def _apply_card_state_augmentation(
    image: Image.Image,
    rng: random.Random,
    *,
    aug_enabled: bool,
    rotate_prob: float,
    rotate_max_degrees: float,
    translate_prob: float,
    translate_max_px: int,
    brightness_min: float,
    brightness_max: float,
    contrast_min: float,
    contrast_max: float,
    color_min: float,
    color_max: float,
    blur_prob: float,
    blur_min_radius: float,
    blur_max_radius: float,
) -> Image.Image:
    im = image.convert("RGB")
    if not aug_enabled:
        return im
    if rng.random() < rotate_prob:
        angle = rng.uniform(-rotate_max_degrees, rotate_max_degrees)
        im = im.rotate(angle, resample=Image.BICUBIC, expand=False)
    if rng.random() < translate_prob and translate_max_px > 0:
        dx = rng.randint(-translate_max_px, translate_max_px)
        dy = rng.randint(-translate_max_px, translate_max_px)
        im = im.transform(
            im.size,
            Image.Transform.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BICUBIC,
        )
    # Use only generic photometric noise. No synthetic "insufficient elixir" masks:
    # those states should be learned from real labeled samples.
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(brightness_min, brightness_max))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(contrast_min, contrast_max))
    im = ImageEnhance.Color(im).enhance(rng.uniform(color_min, color_max))
    if rng.random() < blur_prob:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(blur_min_radius, blur_max_radius)))
    return im


def _to_tensor(
    path: Path,
    size: int,
    *,
    train: bool,
    rng: random.Random,
    aug_enabled: bool,
    rotate_prob: float,
    rotate_max_degrees: float,
    translate_prob: float,
    translate_max_px: int,
    brightness_min: float,
    brightness_max: float,
    contrast_min: float,
    contrast_max: float,
    color_min: float,
    color_max: float,
    blur_prob: float,
    blur_min_radius: float,
    blur_max_radius: float,
    grayscale_input: bool,
) -> torch.Tensor:
    image = Image.open(path)
    if train:
        image = _apply_card_state_augmentation(
            image,
            rng,
            aug_enabled=aug_enabled,
            rotate_prob=rotate_prob,
            rotate_max_degrees=rotate_max_degrees,
            translate_prob=translate_prob,
            translate_max_px=translate_max_px,
            brightness_min=brightness_min,
            brightness_max=brightness_max,
            contrast_min=contrast_min,
            contrast_max=contrast_max,
            color_min=color_min,
            color_max=color_max,
            blur_prob=blur_prob,
            blur_min_radius=blur_min_radius,
            blur_max_radius=blur_max_radius,
        )
    else:
        image = image.convert("RGB")
    if grayscale_input:
        image = image.convert("L").convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hand-card classifier from cropped slot PNGs")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cards"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/card_cnn.pt"))
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", type=str, default="cards-default")
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=None,
        help="Optional path for artifact manifest JSON (default: <out>.manifest.json)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable all training augmentations",
    )
    parser.add_argument("--rotate-prob", type=float, default=0.45)
    parser.add_argument("--rotate-max-degrees", type=float, default=3.0)
    parser.add_argument("--translate-prob", type=float, default=0.2)
    parser.add_argument("--translate-max-px", type=int, default=2)
    parser.add_argument("--brightness-min", type=float, default=0.8)
    parser.add_argument("--brightness-max", type=float, default=1.2)
    parser.add_argument("--contrast-min", type=float, default=0.8)
    parser.add_argument("--contrast-max", type=float, default=1.25)
    parser.add_argument("--color-min", type=float, default=0.75)
    parser.add_argument("--color-max", type=float, default=1.25)
    parser.add_argument("--blur-prob", type=float, default=0.2)
    parser.add_argument("--blur-min-radius", type=float, default=0.2)
    parser.add_argument("--blur-max-radius", type=float, default=0.8)
    parser.add_argument(
        "--rgb-input",
        action="store_true",
        help="Keep RGB input (default uses grayscale conversion for robustness)",
    )
    args = parser.parse_args()
    if args.brightness_min > args.brightness_max:
        raise SystemExit("brightness-min must be <= brightness-max")
    if args.contrast_min > args.contrast_max:
        raise SystemExit("contrast-min must be <= contrast-max")
    if args.color_min > args.color_max:
        raise SystemExit("color-min must be <= color-max")
    if args.blur_min_radius > args.blur_max_radius:
        raise SystemExit("blur-min-radius must be <= blur-max-radius")
    if not 0.0 <= args.rotate_prob <= 1.0:
        raise SystemExit("rotate-prob must be in [0, 1]")
    if args.rotate_max_degrees < 0.0:
        raise SystemExit("rotate-max-degrees must be >= 0")
    if not 0.0 <= args.translate_prob <= 1.0:
        raise SystemExit("translate-prob must be in [0, 1]")
    if args.translate_max_px < 0:
        raise SystemExit("translate-max-px must be >= 0")
    if not 0.0 <= args.blur_prob <= 1.0:
        raise SystemExit("blur-prob must be in [0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = _collect_samples(args.data_dir)
    label_to_idx, idx_to_label = _label_index(samples)
    train_items, val_items = _stratified_split(samples, val_fraction=args.val_fraction, seed=args.seed)

    device = torch.device("cpu")
    net = CardHandNet(num_classes=len(idx_to_label)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    def batch_tensors(
        items: list[tuple[Path, str]], *, train: bool, seed_offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(args.seed + seed_offset)
        xs = [
            _to_tensor(
                p,
                args.input_size,
                train=train,
                rng=rng,
                aug_enabled=not args.no_augment,
                rotate_prob=args.rotate_prob,
                rotate_max_degrees=args.rotate_max_degrees,
                translate_prob=args.translate_prob,
                translate_max_px=args.translate_max_px,
                brightness_min=args.brightness_min,
                brightness_max=args.brightness_max,
                contrast_min=args.contrast_min,
                contrast_max=args.contrast_max,
                color_min=args.color_min,
                color_max=args.color_max,
                blur_prob=args.blur_prob,
                blur_min_radius=args.blur_min_radius,
                blur_max_radius=args.blur_max_radius,
                grayscale_input=not args.rgb_input,
            )
            for p, _ in items
        ]
        ys = torch.tensor([label_to_idx[name] for _, name in items], dtype=torch.long, device=device)
        return torch.stack(xs, dim=0), ys

    best_loss = float("inf")
    best_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        net.train()
        order_rng = random.Random(args.seed + epoch)
        order = list(range(len(train_items)))
        order_rng.shuffle(order)
        shuffled = [train_items[i] for i in order]

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for start in range(0, len(shuffled), args.batch_size):
            chunk = shuffled[start : start + args.batch_size]
            xb, yb = batch_tensors(chunk, train=True, seed_offset=epoch * 1000 + start)
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(chunk)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_seen += len(chunk)

        net.eval()
        with torch.inference_mode():
            if val_items:
                vx, vy = batch_tensors(val_items, train=False, seed_offset=epoch + 99999)
                vlogits = net(vx)
                vloss = float(loss_fn(vlogits, vy))
                vacc = float((vlogits.argmax(dim=1) == vy).float().mean().item())
            else:
                vloss = float("nan")
                vacc = float("nan")

        train_loss = total_loss / max(1, total_seen)
        train_acc = total_correct / max(1, total_seen)
        improved = vloss < best_loss if val_items else train_loss < best_loss
        if improved:
            best_loss = vloss if val_items else train_loss
            best_acc = vacc if val_items else train_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                "epoch={:3d} train_loss={:.4f} train_acc={:.1%} val_loss={:.4f} val_acc={:.1%} "
                "best_loss={:.4f} best_acc={:.1%}".format(
                    epoch + 1, train_loss, train_acc, vloss, vacc, best_loss, best_acc
                )
            )

    if best_state is None:
        best_state = net.state_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "input_size": args.input_size,
            "num_classes": len(idx_to_label),
            "idx_to_label": idx_to_label,
            "label_to_idx": label_to_idx,
            "meta": {
                "schema_version": 1,
                "dataset_id": args.dataset_id,
                "train_samples": len(train_items),
                "val_samples": len(val_items),
                "data_dir": str(args.data_dir),
                "task": "hand_card_classification",
                "grayscale_input": not args.rgb_input,
            },
        },
        args.out,
    )
    manifest_path = args.artifact_manifest if args.artifact_manifest else args.out.with_suffix(".manifest.json")
    write_artifact_manifest(
        manifest_path=manifest_path,
        model_id="card-hand-net",
        task="hand_card_classification",
        dataset_id=args.dataset_id,
        checkpoint_path=args.out,
        metrics={"best_loss": best_loss, "best_acc": best_acc},
        train_args={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    )
    print(
        "Wrote {} (best_loss={:.4f}, best_acc={:.1%}, classes={})".format(
            args.out.resolve(), best_loss, best_acc, json.dumps(idx_to_label)
        )
    )


if __name__ == "__main__":
    main()
