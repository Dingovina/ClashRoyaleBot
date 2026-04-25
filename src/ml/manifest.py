from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_dataset_manifest(
    *,
    manifest_path: Path,
    dataset_id: str,
    schema_version: int,
    source_root: Path,
    processed_root: Path,
    files: list[Path],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "dataset_id": dataset_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_root": str(source_root),
        "processed_root": str(processed_root),
        "files": [_file_record(path) for path in sorted(files)],
    }
    if extra:
        payload["extra"] = extra
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_artifact_manifest(
    *,
    manifest_path: Path,
    model_id: str,
    task: str,
    dataset_id: str,
    checkpoint_path: Path,
    metrics: dict[str, Any],
    train_args: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "model_id": model_id,
        "task": task,
        "dataset_id": dataset_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit_or_unknown(),
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "train_args": train_args,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": str(path),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _git_commit_or_unknown() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip() or "unknown"
    except Exception:
        return "unknown"
