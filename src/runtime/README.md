# Runtime Module

This package contains real-time match execution code.

## Sprint 0 scope
- Config-driven tick loop (`500 ms` by default).
- Global action rate limit (`1000 ms` by default).
- Policy gate with confidence and legality checks.
- 12-zone board map with fixed anchor points.

## Sprint 1 additions
- Fullscreen capture plumbing with optional debug frame dumps.
- Keyboard + mouse actuation layer with safe `dry-run` mode.
- Configurable delay between card hotkey and placement click (`actuation_select_to_click_delay_ms`).
- Configurable per-slot hotkeys (`actuation_card_hotkeys`, e.g. `q`/`w`/`e`/`r` to match in-game bindings).
- Structured `action_attempt` logs for each decision execution.
- `game_viewport` maps normalized board anchors into the real on-screen playfield (for letterboxed / centered portrait windows).
- `game_viewport.anchor_rect` maps anchors into a sub-rectangle of the client window (for example to exclude the bottom hand / elixir bar).

## Current modules
- `config.py` loads runtime settings from `configs/runtime.yaml`.
- `zones.py` stores 4x3 zone geometry and legality masks.
- `policy_gate.py` applies no-op/confidence/rate-limit rules.
- `capture.py` grabs fullscreen frames from the active display.
- `actuation.py` sends keyboard card selection (`pynput` first, `pyautogui` fallback) and mouse click placement (`pyautogui`).
- `candidate_policy.py` provides scripted candidate actions for reliability tests.
- `loop.py` runs capture -> policy -> gate -> actuation and logs decisions.
- `__main__.py` starts runtime via `python -m src.runtime`.
