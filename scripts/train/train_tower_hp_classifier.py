#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.perception.datasets.tower_hp_samples import (
    EMPTY_TOKEN,
    TOWER_GROUPS,
    TowerHpSample,
    collect_tower_hp_samples,
    filter_tower_hp_samples_by_group,
)
from src.perception.models.tower_hp_net import TowerHpNet

_DIGIT_CHARSET = "0123456789"
_BLANK_INDEX = len(_DIGIT_CHARSET)


def _encode_digits(text: str) -> list[int]:
    return [int(ch) for ch in text]


def _decode_digits(indices: list[int], *, blank_index: int) -> str:
    collapsed: list[int] = []
    prev = -1
    for idx in indices:
        if idx == prev:
            continue
        prev = idx
        if idx == blank_index:
            continue
        collapsed.append(idx)
    if not collapsed:
        return ""
    return "".join(_DIGIT_CHARSET[idx] for idx in collapsed)


def _load_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    t = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def _evaluate_exact_match(
    net: TowerHpNet,
    samples: list[TowerHpSample],
    *,
    batch_size: int,
    input_width: int,
    input_height: int,
    device: torch.device,
    presence_threshold: float,
) -> float:
    if not samples:
        return 0.0
    net.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            xb = torch.stack([_load_tensor(s.path, input_width, input_height) for s in batch], dim=0).to(device)
            ctc_logits, presence_logits = net(xb)
            presence_probs = torch.sigmoid(presence_logits)
            for i, sample in enumerate(batch):
                is_has_hp = float(presence_probs[i].item()) >= presence_threshold
                if not is_has_hp:
                    pred = EMPTY_TOKEN
                else:
                    step_logits = ctc_logits[i]
                    step_indices = step_logits.argmax(dim=1).tolist()
                    decoded = _decode_digits(step_indices, blank_index=_BLANK_INDEX)
                    pred = decoded if decoded else EMPTY_TOKEN
                if pred == sample.hp_text:
                    correct += 1
    return correct / max(1, len(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tower HP sequence model (presence + CTC digits)")
    parser.add_argument("--train-data-dir", type=Path, default=Path("data/processed/train/tower_hp_train"))
    parser.add_argument("--val-data-dir", type=Path, default=Path("data/processed/val/tower_hp_val"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/tower_hp_crnn.pt"))
    parser.add_argument("--input-width", type=int, default=128)
    parser.add_argument("--input-height", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", type=str, default="tower-hp-default")
    parser.add_argument(
        "--tower-group",
        type=str,
        choices=sorted(TOWER_GROUPS.keys()),
        default="all",
        help="Tower group to train on: all, king, princess",
    )
    parser.add_argument("--lambda-ctc", type=float, default=1.0)
    parser.add_argument("--presence-pos-weight", type=float, default=1.0)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    try:
        train_samples = filter_tower_hp_samples_by_group(
            collect_tower_hp_samples(args.train_data_dir), args.tower_group
        )
        val_samples = filter_tower_hp_samples_by_group(collect_tower_hp_samples(args.val_data_dir), args.tower_group)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(train_samples) < 8:
        raise SystemExit(f"Need at least 8 train samples under {args.train_data_dir}, found {len(train_samples)}")
    if len(val_samples) < 1:
        raise SystemExit(f"Need at least 1 val sample under {args.val_data_dir}, found {len(val_samples)}")

    device = torch.device("cpu")
    net = TowerHpNet(digit_classes=_BLANK_INDEX + 1).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ctc_loss_fn = nn.CTCLoss(blank=_BLANK_INDEX, zero_infinity=True)
    presence_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.presence_pos_weight], dtype=torch.float32, device=device)
    )

    best_exact = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        net.train()
        order = list(range(len(train_samples)))
        random.Random(args.seed + epoch).shuffle(order)
        shuffled = [train_samples[i] for i in order]
        total_loss = 0.0
        total_presence_loss = 0.0
        total_ctc_loss = 0.0
        seen = 0

        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            xb = torch.stack([_load_tensor(s.path, args.input_width, args.input_height) for s in batch], dim=0).to(device)
            y_presence = torch.tensor([0.0 if s.is_empty else 1.0 for s in batch], dtype=torch.float32, device=device)

            ctc_logits, presence_logits = net(xb)
            presence_loss = presence_loss_fn(presence_logits, y_presence)

            has_mask = y_presence > 0.5
            if int(has_mask.sum().item()) > 0:
                ctc_has = ctc_logits[has_mask]
                log_probs = torch.log_softmax(ctc_has, dim=2).permute(1, 0, 2)
                has_samples = [sample for sample in batch if not sample.is_empty]
                encoded = [_encode_digits(sample.hp_text) for sample in has_samples]
                flat_targets = torch.tensor(
                    [item for seq in encoded for item in seq],
                    dtype=torch.long,
                    device=device,
                )
                target_lengths = torch.tensor([len(seq) for seq in encoded], dtype=torch.long, device=device)
                input_lengths = torch.full(
                    (len(has_samples),),
                    fill_value=ctc_has.shape[1],
                    dtype=torch.long,
                    device=device,
                )
                ctc_loss = ctc_loss_fn(log_probs, flat_targets, input_lengths, target_lengths)
            else:
                ctc_loss = torch.zeros((), dtype=torch.float32, device=device)

            loss = presence_loss + args.lambda_ctc * ctc_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach()) * len(batch)
            total_presence_loss += float(presence_loss.detach()) * len(batch)
            total_ctc_loss += float(ctc_loss.detach()) * len(batch)
            seen += len(batch)

        val_exact = _evaluate_exact_match(
            net,
            val_samples,
            batch_size=args.batch_size,
            input_width=args.input_width,
            input_height=args.input_height,
            device=device,
            presence_threshold=args.presence_threshold,
        )
        train_loss = total_loss / max(1, seen)
        train_presence_loss = total_presence_loss / max(1, seen)
        train_ctc_loss = total_ctc_loss / max(1, seen)

        if val_exact > best_exact:
            best_exact = val_exact
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                "epoch={:3d} train_loss={:.4f} presence_loss={:.4f} ctc_loss={:.4f} "
                "val_exact={:.1%} best_exact={:.1%}".format(
                    epoch + 1, train_loss, train_presence_loss, train_ctc_loss, val_exact, best_exact
                )
            )

    if best_state is None:
        best_state = net.state_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "input_width": args.input_width,
            "input_height": args.input_height,
            "digit_charset": list(_DIGIT_CHARSET),
            "blank_index": _BLANK_INDEX,
            "presence_threshold": args.presence_threshold,
            "decode_mode": "greedy_ctc_with_presence_head",
            "meta": {
                "schema_version": 1,
                "dataset_id": args.dataset_id,
                "task": "tower_hp_text_with_empty_presence",
                "tower_group": args.tower_group,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "train_data_dir": str(args.train_data_dir),
                "val_data_dir": str(args.val_data_dir),
            },
        },
        args.out,
    )
    print(f"Wrote {args.out.resolve()} (best_exact={best_exact:.1%})")


if __name__ == "__main__":
    main()

