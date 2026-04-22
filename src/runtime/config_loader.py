from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.runtime.runtime_config import BattlefieldTimeoutBehavior, RuntimeConfig
from src.runtime.viewport import parse_game_viewport


def load_runtime_config(path: Path) -> RuntimeConfig:
    data = _load_yaml(path)
    if "runtime" not in data:
        found = list(data.keys())
        raise ValueError(
            f"{path}: missing top-level key 'runtime'. Found keys: {found}. "
            "Fix the YAML so the block starts with exactly `runtime:` (no stray characters)."
        )
    if "board" not in data:
        raise ValueError(f"{path}: missing top-level key 'board'. Found keys: {list(data.keys())}")
    runtime = data["runtime"]
    board = data["board"]
    card_types = data.get("card_types", {})

    zones = {
        int(zone_id): (float(anchor[0]), float(anchor[1]))
        for zone_id, anchor in board["zones"].items()
    }

    cfg = RuntimeConfig(
        tick_interval_ms=int(runtime["tick_interval_ms"]),
        action_rate_limit_ms=int(runtime["action_rate_limit_ms"]),
        action_confidence_threshold=float(runtime["action_confidence_threshold"]),
        no_op_confidence_threshold=float(runtime["no_op_confidence_threshold"]),
        min_elixir_for_non_urgent_action=float(runtime["min_elixir_for_non_urgent_action"]),
        max_ticks=int(runtime["max_ticks"]),
        zones=zones,
        spell_cards=set(card_types.get("spell_cards", [])),
        capture_enabled=bool(runtime.get("capture_enabled", True)),
        capture_debug_dir=runtime.get("capture_debug_dir"),
        capture_every_n_ticks=max(0, int(runtime.get("capture_every_n_ticks", 0))),
        actuation_enabled=bool(runtime.get("actuation_enabled", False)),
        actuation_dry_run=bool(runtime.get("actuation_dry_run", True)),
        actuation_select_to_click_delay_ms=max(
            0, int(runtime.get("actuation_select_to_click_delay_ms", 120))
        ),
        actuation_card_hotkeys=_parse_actuation_card_hotkeys(runtime),
        game_viewport=parse_game_viewport(runtime),
        match_readiness_enabled=bool(runtime.get("match_readiness_enabled", True)),
        battlefield_detector=_parse_battlefield_detector(runtime),
        battlefield_score_threshold=float(runtime.get("battlefield_score_threshold", 0.14)),
        battlefield_sample_stride=max(1, int(runtime.get("battlefield_sample_stride", 10))),
        battlefield_river_band_top_ratio=float(runtime.get("battlefield_river_band_top_ratio", 0.30)),
        battlefield_river_band_bottom_ratio=float(
            runtime.get("battlefield_river_band_bottom_ratio", 0.52)
        ),
        battlefield_grass_band_top_ratio=float(runtime.get("battlefield_grass_band_top_ratio", 0.55)),
        battlefield_grass_band_bottom_ratio=float(
            runtime.get("battlefield_grass_band_bottom_ratio", 0.90)
        ),
        battlefield_wait_timeout_ms=max(0, int(runtime.get("battlefield_wait_timeout_ms", 120000))),
        battlefield_timeout_behavior=_parse_battlefield_timeout_behavior(
            runtime.get("battlefield_timeout_behavior", "idle")
        ),
        foreground_check_enabled=bool(runtime.get("foreground_check_enabled", False)),
        foreground_title_substrings=_parse_foreground_title_substrings(runtime),
        battlefield_model_path=_parse_optional_path(runtime.get("battlefield_model_path")),
        battlefield_model_input_size=max(32, int(runtime.get("battlefield_model_input_size", 128))),
        battlefield_model_layout_path=_parse_battlefield_model_layout_path(runtime),
    )
    _validate_runtime_config(cfg)
    return cfg


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_optional_path(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _parse_battlefield_model_layout_path(runtime: dict[str, Any]) -> str:
    raw = runtime.get("battlefield_model_layout_path")
    explicit = _parse_optional_path(raw)
    if explicit:
        return explicit
    return "configs/screen_layout_reference.yaml"


def _validate_runtime_config(cfg: RuntimeConfig) -> None:
    if cfg.match_readiness_enabled and not cfg.capture_enabled:
        raise ValueError("match_readiness_enabled requires capture_enabled to be true")
    if cfg.battlefield_river_band_top_ratio >= cfg.battlefield_river_band_bottom_ratio:
        raise ValueError("battlefield river band top ratio must be less than bottom ratio")
    if cfg.battlefield_grass_band_top_ratio >= cfg.battlefield_grass_band_bottom_ratio:
        raise ValueError("battlefield grass band top ratio must be less than bottom ratio")
    if cfg.battlefield_detector in ("model", "blend"):
        if not cfg.battlefield_model_path:
            raise ValueError("battlefield_model_path is required when battlefield_detector is model or blend")
        mp = Path(cfg.battlefield_model_path)
        if not mp.is_file():
            raise ValueError(f"battlefield_model_path does not exist or is not a file: {mp}")
        lp = Path(cfg.battlefield_model_layout_path)
        if not lp.is_file():
            raise ValueError(f"battlefield_model_layout_path does not exist or is not a file: {lp}")
        if not _torch_available():
            raise ValueError(
                "PyTorch is required for battlefield_detector model/blend; install with: pip install -r requirements-ml.txt"
            )


def _parse_battlefield_detector(runtime: dict[str, Any]) -> str:
    raw = str(runtime.get("battlefield_detector", "heuristic")).lower().strip()
    if raw not in ("heuristic", "model", "blend"):
        raise ValueError(f"runtime.battlefield_detector must be heuristic, model, or blend, got {raw!r}")
    return raw


def _parse_battlefield_timeout_behavior(raw: Any) -> BattlefieldTimeoutBehavior:
    value = str(raw or "idle").lower().strip()
    if value == "idle":
        return "idle"
    if value == "exit_nonzero":
        return "exit_nonzero"
    raise ValueError("runtime.battlefield_timeout_behavior must be 'idle' or 'exit_nonzero'")


def _parse_foreground_title_substrings(runtime: dict[str, Any]) -> tuple[str, ...]:
    raw = runtime.get("foreground_title_substrings")
    if raw is None:
        return ("clash royale", "google play games", "google play")
    if not isinstance(raw, list) or not raw:
        raise ValueError("runtime.foreground_title_substrings must be a non-empty list of strings")
    out: list[str] = []
    for index, item in enumerate(raw):
        s = str(item).strip().lower()
        if not s:
            raise ValueError(f"runtime.foreground_title_substrings[{index}] is empty")
        out.append(s)
    return tuple(out)


def _parse_actuation_card_hotkeys(runtime: dict[str, Any]) -> tuple[str, str, str, str]:
    raw = runtime.get("actuation_card_hotkeys")
    if raw is None:
        return ("1", "2", "3", "4")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("runtime.actuation_card_hotkeys must be a list of exactly 4 strings")

    keys: list[str] = []
    for index, item in enumerate(raw):
        key = str(item).strip().lower()
        if not key:
            raise ValueError(f"runtime.actuation_card_hotkeys[{index}] is empty")
        keys.append(key)

    return (keys[0], keys[1], keys[2], keys[3])


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file)
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return parsed
