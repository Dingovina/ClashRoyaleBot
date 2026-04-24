from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.runtime.card_registry import load_card_registry
from src.runtime.runtime_config import BattlefieldTimeoutBehavior, RuntimeConfig
from src.runtime.viewport import parse_game_viewport

DEFAULT_BATTLEFIELD_CHECKPOINT = "artifacts/battlefield_cnn.pt"
DEFAULT_ELIXIR_CHECKPOINT = "artifacts/elixir_cnn.pt"
DEFAULT_CARD_CHECKPOINT = "artifacts/card_cnn.pt"
DEFAULT_ELIXIR_LAYOUT_PATH = "configs/screen_layout_reference.yaml"
DEFAULT_CARD_LAYOUT_PATH = "configs/screen_layout_reference.yaml"
DEFAULT_CARD_ELIXIR_COSTS_PATH = "configs/card_elixir_costs.yaml"
DEFAULT_CARD_REGISTRY_PATH = "configs/card_registry.yaml"

TRAIN_BATTLEFIELD_CLASSIFIER_HELP = (
    "Install ML dependencies: pip install -r requirements-ml.txt\n"
    "Train from labeled PNGs under good/ (in-match) and bad/ (not), run from repo root (one line; "
    "do not use cmd's ^ in PowerShell—it is passed to Python as an argument):\n"
    "  python scripts/train_battlefield_classifier.py --data-dir data/processed/battlefield_test "
    "--layout-yaml configs/screen_layout_reference.yaml --out artifacts/battlefield_cnn.pt\n"
    "Optionally set --input-size to match your checkpoint (default 128). "
    "Then set runtime.battlefield_model_path in configs/runtime.yaml if the file is not at the default path."
)

TRAIN_ELIXIR_CLASSIFIER_HELP = (
    "Install ML dependencies: pip install -r requirements-ml.txt\n"
    "Train from labeled PNGs named <elixir>_<index>.png under data/processed/elixir_test, run from repo root:\n"
    "  python scripts/train_elixir_classifier.py --data-dir data/processed/elixir_test "
    "--layout-yaml configs/screen_layout_reference.yaml --out artifacts/elixir_cnn.pt\n"
    "Then set runtime.elixir_model_path in configs/runtime.yaml if the file is not at the default path."
)

TRAIN_CARD_CLASSIFIER_HELP = (
    "Install ML dependencies: pip install -r requirements-ml.txt\n"
    "Train from hand-slot crops named <card-name>_<random-id>.png under data/processed/cards:\n"
    "  python scripts/train_card_classifier.py --data-dir data/processed/cards --out artifacts/card_cnn.pt\n"
    "Then set runtime.card_model_path in configs/runtime.yaml if the file is not at the default path."
)


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
    registry = _parse_card_registry(runtime, card_types)

    zones = {
        int(zone_id): (float(anchor[0]), float(anchor[1]))
        for zone_id, anchor in board["zones"].items()
    }

    match_readiness_enabled = bool(runtime.get("match_readiness_enabled", True))
    model_path = _parse_optional_path(runtime.get("battlefield_model_path"))
    if match_readiness_enabled and not model_path:
        model_path = DEFAULT_BATTLEFIELD_CHECKPOINT
    elixir_model_enabled = bool(runtime.get("elixir_model_enabled", False))
    elixir_model_path = _parse_optional_path(runtime.get("elixir_model_path"))
    if elixir_model_enabled and not elixir_model_path:
        elixir_model_path = DEFAULT_ELIXIR_CHECKPOINT
    card_model_enabled = bool(runtime.get("card_model_enabled", False))
    card_model_path = _parse_optional_path(runtime.get("card_model_path"))
    if card_model_enabled and not card_model_path:
        card_model_path = DEFAULT_CARD_CHECKPOINT

    cfg = RuntimeConfig(
        tick_interval_ms=int(runtime["tick_interval_ms"]),
        action_rate_limit_ms=int(runtime["action_rate_limit_ms"]),
        action_confidence_threshold=float(runtime["action_confidence_threshold"]),
        no_op_confidence_threshold=float(runtime["no_op_confidence_threshold"]),
        min_elixir_for_non_urgent_action=float(runtime["min_elixir_for_non_urgent_action"]),
        match_safety_max_ticks=_parse_match_safety_max_ticks(runtime),
        battlefield_end_score_threshold=float(runtime.get("battlefield_end_score_threshold", 0.42)),
        match_end_confirm_ticks=max(0, int(runtime.get("match_end_confirm_ticks", 6))),
        match_end_check_every_n_ticks=max(1, int(runtime.get("match_end_check_every_n_ticks", 2))),
        zones=zones,
        spell_cards=registry.spell_cards,
        capture_enabled=bool(runtime.get("capture_enabled", True)),
        capture_debug_save_enabled=bool(runtime.get("capture_debug_save_enabled", True)),
        capture_debug_dir=runtime.get("capture_debug_dir"),
        capture_every_n_ticks=max(0, int(runtime.get("capture_every_n_ticks", 0))),
        actuation_enabled=bool(runtime.get("actuation_enabled", False)),
        actuation_dry_run=bool(runtime.get("actuation_dry_run", True)),
        actuation_select_to_click_delay_ms=max(
            0, int(runtime.get("actuation_select_to_click_delay_ms", 120))
        ),
        actuation_card_hotkeys=_parse_actuation_card_hotkeys(runtime),
        game_viewport=parse_game_viewport(runtime),
        match_readiness_enabled=match_readiness_enabled,
        battlefield_score_threshold=float(runtime.get("battlefield_score_threshold", 0.65)),
        battlefield_wait_timeout_ms=max(0, int(runtime.get("battlefield_wait_timeout_ms", 120000))),
        battlefield_timeout_behavior=_parse_battlefield_timeout_behavior(
            runtime.get("battlefield_timeout_behavior", "idle")
        ),
        foreground_check_enabled=bool(runtime.get("foreground_check_enabled", False)),
        foreground_title_substrings=_parse_foreground_title_substrings(runtime),
        battlefield_model_path=model_path,
        battlefield_model_layout_path=_parse_battlefield_model_layout_path(runtime),
        elixir_model_enabled=elixir_model_enabled,
        elixir_model_path=elixir_model_path,
        elixir_model_layout_path=_parse_elixir_model_layout_path(runtime),
        card_model_enabled=card_model_enabled,
        card_model_path=card_model_path,
        card_model_layout_path=_parse_card_model_layout_path(runtime),
        hand_tick_log_enabled=bool(runtime.get("hand_tick_log_enabled", True)),
        hand_tick_log_path=_parse_hand_tick_log_path(runtime),
        session_id=_parse_session_id(runtime),
        card_name_aliases=registry.aliases,
        card_elixir_costs=registry.elixir_costs,
    )
    _validate_runtime_config(cfg)
    return cfg


def _parse_match_safety_max_ticks(runtime: dict[str, Any]) -> int:
    """Hard cap on main-loop ticks (0 = no cap when CNN match-end is enabled)."""
    return max(0, int(runtime.get("match_safety_max_ticks", 7200)))


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


def _parse_elixir_model_layout_path(runtime: dict[str, Any]) -> str:
    raw = runtime.get("elixir_model_layout_path")
    explicit = _parse_optional_path(raw)
    if explicit:
        return explicit
    return DEFAULT_ELIXIR_LAYOUT_PATH


def _parse_card_elixir_costs() -> dict[str, float]:
    path = Path(DEFAULT_CARD_ELIXIR_COSTS_PATH)
    if not path.is_file():
        raise ValueError(f"card elixir costs file does not exist: {path}")
    raw = _load_yaml(path)
    out: dict[str, float] = {}
    for name, value in raw.items():
        key = str(name).strip().lower()
        if not key:
            raise ValueError(f"{path}: contains an empty card name")
        cost = float(value)
        if cost <= 0:
            raise ValueError(f"{path}: cost for '{key}' must be > 0")
        out[key] = cost
    if not out:
        raise ValueError(f"{path}: must not be empty")
    return out


@dataclass(frozen=True)
class _RegistryParseResult:
    aliases: dict[str, str]
    elixir_costs: dict[str, float]
    spell_cards: set[str]


def _parse_card_registry(runtime: dict[str, Any], card_types: dict[str, Any]) -> _RegistryParseResult:
    path_raw = runtime.get("card_registry_path", DEFAULT_CARD_REGISTRY_PATH)
    path = Path(str(path_raw).strip())
    if path.is_file():
        reg = load_card_registry(path)
        return _RegistryParseResult(
            aliases=reg.aliases,
            elixir_costs=reg.elixir_costs,
            spell_cards=set(reg.spell_cards),
        )
    # Backward-compatible fallback while configs migrate to card_registry.yaml
    fallback_costs = _parse_card_elixir_costs()
    fallback_aliases = {k: k for k in fallback_costs}
    fallback_spells = {str(x).strip().lower() for x in card_types.get("spell_cards", []) if str(x).strip()}
    return _RegistryParseResult(
        aliases=fallback_aliases,
        elixir_costs=fallback_costs,
        spell_cards=fallback_spells,
    )


def _parse_card_model_layout_path(runtime: dict[str, Any]) -> str:
    raw = runtime.get("card_model_layout_path")
    explicit = _parse_optional_path(raw)
    if explicit:
        return explicit
    return DEFAULT_CARD_LAYOUT_PATH


def _parse_hand_tick_log_path(runtime: dict[str, Any]) -> str:
    raw = runtime.get("hand_tick_log_path")
    explicit = _parse_optional_path(raw)
    if explicit:
        return explicit
    return "logs/hand_cards_ticks.jsonl"


def _parse_session_id(runtime: dict[str, Any]) -> str:
    raw = _parse_optional_path(runtime.get("session_id"))
    if raw:
        return raw
    return "local-runtime"


def _is_default_battlefield_checkpoint(path: Path) -> bool:
    norm = path.as_posix().replace("\\", "/")
    return norm == DEFAULT_BATTLEFIELD_CHECKPOINT or norm.endswith("/artifacts/battlefield_cnn.pt")


def _validate_runtime_config(cfg: RuntimeConfig) -> None:
    if cfg.match_readiness_enabled and not cfg.capture_enabled:
        raise ValueError("match_readiness_enabled requires capture_enabled to be true")

    if cfg.match_readiness_enabled:
        if not cfg.battlefield_model_path:
            raise ValueError(
                "match_readiness_enabled requires battlefield_model_path (or rely on the default path)"
            )

        mp = Path(cfg.battlefield_model_path)
        if not mp.is_file():
            if _is_default_battlefield_checkpoint(mp):
                lead = (
                    f"Missing default battlefield classifier weights: {DEFAULT_BATTLEFIELD_CHECKPOINT} "
                    f"(resolved as {mp.resolve()}). Create this file by training the model."
                )
            else:
                lead = f"battlefield_model_path does not exist or is not a file: {mp}"
            raise ValueError(f"{lead}\n\n{TRAIN_BATTLEFIELD_CLASSIFIER_HELP}")

        lp = Path(cfg.battlefield_model_layout_path)
        if not lp.is_file():
            raise ValueError(f"battlefield_model_layout_path does not exist or is not a file: {lp}")

        if not _torch_available():
            raise ValueError(
                "PyTorch is required for match readiness (battlefield CNN). "
                "Install with: pip install -r requirements-ml.txt"
            )

    if cfg.elixir_model_enabled:
        if not cfg.capture_enabled:
            raise ValueError("elixir_model_enabled requires capture_enabled to be true")
        if not cfg.elixir_model_path:
            raise ValueError("elixir_model_enabled requires elixir_model_path (or default path)")
        ep = Path(cfg.elixir_model_path)
        if not ep.is_file():
            if ep.as_posix().replace("\\", "/").endswith("/artifacts/elixir_cnn.pt"):
                lead = (
                    f"Missing default elixir classifier weights: {DEFAULT_ELIXIR_CHECKPOINT} "
                    f"(resolved as {ep.resolve()}). Create this file by training the model."
                )
            else:
                lead = f"elixir_model_path does not exist or is not a file: {ep}"
            raise ValueError(f"{lead}\n\n{TRAIN_ELIXIR_CLASSIFIER_HELP}")
        lp = Path(cfg.elixir_model_layout_path)
        if not lp.is_file():
            raise ValueError(f"elixir_model_layout_path does not exist or is not a file: {lp}")
        if not _torch_available():
            raise ValueError(
                "PyTorch is required for elixir CNN inference. Install with: pip install -r requirements-ml.txt"
            )

    if cfg.card_model_enabled:
        if not cfg.capture_enabled:
            raise ValueError("card_model_enabled requires capture_enabled to be true")
        if not cfg.card_model_path:
            raise ValueError("card_model_enabled requires card_model_path (or default path)")
        cp = Path(cfg.card_model_path)
        if not cp.is_file():
            if cp.as_posix().replace("\\", "/").endswith("/artifacts/card_cnn.pt"):
                lead = (
                    f"Missing default card classifier weights: {DEFAULT_CARD_CHECKPOINT} "
                    f"(resolved as {cp.resolve()}). Create this file by training the model."
                )
            else:
                lead = f"card_model_path does not exist or is not a file: {cp}"
            raise ValueError(f"{lead}\n\n{TRAIN_CARD_CLASSIFIER_HELP}")
        lp = Path(cfg.card_model_layout_path)
        if not lp.is_file():
            raise ValueError(f"card_model_layout_path does not exist or is not a file: {lp}")
        if not _torch_available():
            raise ValueError(
                "PyTorch is required for card CNN inference. Install with: pip install -r requirements-ml.txt"
            )

    _validate_match_exit(cfg)


def _validate_match_exit(cfg: RuntimeConfig) -> None:
    if cfg.match_end_confirm_ticks == 0:
        if cfg.match_safety_max_ticks <= 0:
            raise ValueError(
                "match_end_confirm_ticks=0 disables CNN-based match end; "
                "set match_safety_max_ticks > 0 so the runtime cannot run unbounded."
            )
        return

    if not cfg.capture_enabled:
        raise ValueError("match_end_confirm_ticks > 0 requires capture_enabled")
    if not cfg.battlefield_model_path:
        raise ValueError(
            "match_end_confirm_ticks > 0 requires battlefield_model_path (same CNN as match readiness)"
        )
    mp = Path(cfg.battlefield_model_path)
    if not mp.is_file():
        raise ValueError(f"battlefield_model_path does not exist or is not a file: {mp}")
    lp = Path(cfg.battlefield_model_layout_path)
    if not lp.is_file():
        raise ValueError(f"battlefield_model_layout_path does not exist or is not a file: {lp}")
    if not _torch_available():
        raise ValueError(
            "PyTorch is required for CNN match-end detection. Install with: pip install -r requirements-ml.txt"
        )
    if cfg.match_end_check_every_n_ticks < 1:
        raise ValueError("match_end_check_every_n_ticks must be >= 1")
    et = cfg.battlefield_end_score_threshold
    if not 0.0 < et < 1.0:
        raise ValueError("battlefield_end_score_threshold must be in (0, 1)")
    if et >= cfg.battlefield_score_threshold:
        raise ValueError(
            "battlefield_end_score_threshold must be less than battlefield_score_threshold "
            "(start gate vs end-of-match hysteresis)."
        )


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
