#!/usr/bin/env python3
"""
Train the tiny battlefield screen classifier on labeled PNGs.

Expects ``--data-dir/good/*.png`` (in-match, label 1) and ``--data-dir/bad/*.png`` (not in-match, label 0).

The CNN sees only the ``bottom_panel`` region from ``--layout-yaml``, with hand slots,
next-card peek, and elixir count digit pixels zeroed (same preprocessing as runtime inference).

Requires: ``pip install -r requirements-ml.txt``

Example (run from the repository root so ``src`` resolves). In **PowerShell** use a single line: the
caret (^) is **not** a line-continuation character (unlike cmd.exe), so it is passed to Python and breaks argparse.

  python scripts/train/train_battlefield_classifier.py --data-dir data/processed/battlefield_test --layout-yaml configs/screen_layout_reference.yaml --out artifacts/battlefield_cnn.pt
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
from src.perception.roi.battlefield_roi import pil_rgb_masked_bottom_panel
from src.perception.datasets.battlefield_samples import collect_battlefield_labeled_pngs
from src.perception.roi.screen_layout import load_screen_layout_reference
from src.ml.manifest import write_artifact_manifest


def _collect_samples(data_dir: Path) -> list[tuple[Path, int]]:
    try:
        samples = collect_battlefield_labeled_pngs(data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(samples) < 4:
        raise SystemExit(
            f"Need at least 4 labeled PNGs under {data_dir}/good and {data_dir}/bad, found {len(samples)}"
        )
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


def _load_tensor(path: Path, layout, size: int) -> torch.Tensor:
    im = Image.open(path)
    crop = pil_rgb_masked_bottom_panel(im, layout)
    crop = crop.resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/battlefield_test"))
    parser.add_argument(
        "--layout-yaml",
        type=Path,
        default=Path("configs/screen_layout_reference.yaml"),
        help="Screen layout with bottom_panel and HUD rects to mask",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/battlefield_cnn.pt"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", type=str, default="battlefield-default")
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    layout = load_screen_layout_reference(args.layout_yaml)
    samples = _collect_samples(args.data_dir)
    train_items, val_items = _stratified_split(samples, val_fraction=args.val_fraction, seed=args.seed)

    device = torch.device("cpu")
    net = BattlefieldScreenNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def batch_tensors(items: list[tuple[Path, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = [_load_tensor(p, layout, args.input_size) for p, _ in items]
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
                "layout_yaml": str(args.layout_yaml),
                "roi": "masked_bottom_panel",
            },
        },
        args.out,
    )
    manifest_path = args.artifact_manifest if args.artifact_manifest else args.out.with_suffix(".manifest.json")
    write_artifact_manifest(
        manifest_path=manifest_path,
        model_id="battlefield-screen-net",
        task="battlefield_binary_classification",
        dataset_id=args.dataset_id,
        checkpoint_path=args.out,
        metrics={"best_val_loss": best_val},
        train_args={
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
        },
    )
    print(f"Wrote {args.out.resolve()} (best val loss {best_val:.4f}, meta={json.dumps({'dataset_id': args.dataset_id})})")


if __name__ == "__main__":
    main()
