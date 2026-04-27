#!/usr/bin/env python3
"""
Evaluate tower HP sequence checkpoint on labeled per-tower PNG crops.
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

from src.perception.datasets.tower_hp_samples import (
    TOWER_GROUPS,
    TowerHpSample,
    collect_tower_hp_samples,
    filter_tower_hp_samples_by_group,
)
from src.perception.models.tower_hp_net import TowerHpNet

_DIGIT_CHARSET = "0123456789"


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    t = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, 3)
    return (t.float() / 255.0).permute(2, 0, 1)


def _decode_digits(indices: list[int], *, blank_index: int) -> str:
    collapsed: list[int] = []
    prev = -1
    for idx in indices:
        if idx == prev:
            continue
        prev = idx
        if idx == blank_index:
            continue
        if idx < 0 or idx >= len(_DIGIT_CHARSET):
            continue
        collapsed.append(idx)
    if not collapsed:
        return ""
    return "".join(_DIGIT_CHARSET[idx] for idx in collapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tower HP sequence classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/val/tower_hp_val"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--tower-group",
        type=str,
        choices=sorted(TOWER_GROUPS.keys()),
        default=None,
        help="Tower group for evaluation; default uses checkpoint meta tower_group or 'all'",
    )
    args = parser.parse_args()

    ckpt = _torch_load(args.checkpoint)
    if "state_dict" not in ckpt:
        raise SystemExit(f"Invalid checkpoint format: {args.checkpoint}")
    input_width = int(ckpt.get("input_width", 128))
    input_height = int(ckpt.get("input_height", 32))
    blank_index = int(ckpt.get("blank_index", 10))
    presence_threshold = float(ckpt.get("presence_threshold", 0.5))
    ckpt_group = str(ckpt.get("meta", {}).get("tower_group", "all")).lower() if isinstance(ckpt.get("meta"), dict) else "all"
    tower_group = args.tower_group if args.tower_group is not None else ckpt_group
    if tower_group not in TOWER_GROUPS:
        raise SystemExit(f"Invalid tower_group: {tower_group}")

    net = TowerHpNet(digit_classes=blank_index + 1)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()

    try:
        samples = filter_tower_hp_samples_by_group(collect_tower_hp_samples(args.data_dir), tower_group)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rows: list[tuple[str, str, str, str, float, bool]] = []
    per_tower_total: dict[str, int] = {}
    per_tower_correct: dict[str, int] = {}
    char_correct = 0
    char_total = 0
    empty_tp = 0
    empty_fp = 0
    empty_fn = 0
    has_tp = 0
    has_fp = 0
    has_fn = 0
    correct = 0

    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            xb = torch.stack([_load_tensor(s.path, input_width, input_height) for s in batch], dim=0)
            ctc_logits, presence_logits = net(xb)
            presence_probs = torch.sigmoid(presence_logits)

            for i, sample in enumerate(batch):
                presence_prob = float(presence_probs[i].item())
                pred_empty = presence_prob < presence_threshold
                if pred_empty:
                    pred = "none"
                    conf = 1.0 - presence_prob
                else:
                    step_logits = ctc_logits[i]
                    step_probs = torch.softmax(step_logits, dim=1)
                    step_conf, step_idx = torch.max(step_probs, dim=1)
                    decoded = _decode_digits(step_idx.tolist(), blank_index=blank_index)
                    pred = decoded if decoded else "none"
                    conf = min(presence_prob, float(step_conf.mean().item()))

                y_true = "none" if sample.is_empty else sample.hp_text
                ok = pred == y_true
                if ok:
                    correct += 1
                per_tower_total[sample.tower_type] = per_tower_total.get(sample.tower_type, 0) + 1
                if ok:
                    per_tower_correct[sample.tower_type] = per_tower_correct.get(sample.tower_type, 0) + 1

                true_empty = sample.is_empty
                if true_empty and pred == "none":
                    empty_tp += 1
                elif not true_empty and pred == "none":
                    empty_fp += 1
                elif true_empty and pred != "none":
                    empty_fn += 1

                true_has = not true_empty
                pred_has = pred != "none"
                if true_has and pred_has:
                    has_tp += 1
                elif not true_has and pred_has:
                    has_fp += 1
                elif true_has and not pred_has:
                    has_fn += 1

                if true_has and pred_has:
                    m = min(len(sample.hp_text), len(pred))
                    char_correct += sum(1 for j in range(m) if sample.hp_text[j] == pred[j])
                    char_total += len(sample.hp_text)

                rows.append((sample.path.name, sample.tower_type, y_true, pred, conf, ok))

    n = len(rows)
    acc = correct / max(1, n)
    empty_precision = empty_tp / max(1, (empty_tp + empty_fp))
    empty_recall = empty_tp / max(1, (empty_tp + empty_fn))
    has_precision = has_tp / max(1, (has_tp + has_fp))
    has_recall = has_tp / max(1, (has_tp + has_fn))
    char_acc = char_correct / max(1, char_total)

    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"tower_group={tower_group}")
    print(f"n={n} correct={correct} accuracy={acc:.1%}")
    print(f"presence_threshold={presence_threshold:.2f}")
    print()
    print("per_tower_accuracy:")
    for tower_type in sorted(per_tower_total.keys()):
        tower_n = per_tower_total[tower_type]
        tower_ok = per_tower_correct.get(tower_type, 0)
        print(f"  {tower_type:<24} accuracy={tower_ok / max(1, tower_n):>6.1%}  support={tower_n}")
    print()
    print("presence_metrics:")
    print(f"  EMPTY precision={empty_precision:.1%} recall={empty_recall:.1%}")
    print(f"  HAS_HP precision={has_precision:.1%} recall={has_recall:.1%}")
    print(f"  character_accuracy={char_acc:.1%}")
    print()
    print(f"{'file':<34} {'tower_type':<24} {'true':>8} {'pred':>8} {'conf':>7} {'ok':>3}")
    for file_name, tower_type, y_true, y_pred, conf, ok in rows:
        print(
            f"{file_name:<34} {tower_type:<24} {y_true:>8} {y_pred:>8} "
            f"{conf:>7.3f} {'yes' if ok else 'no':>3}"
        )


if __name__ == "__main__":
    main()

