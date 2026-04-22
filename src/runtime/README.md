# Runtime Module

This package contains real-time match execution code.

## Sprint 0 scope
- Config-driven tick loop (`500 ms` by default).
- Global action rate limit (`1000 ms` by default).
- Policy gate with confidence and legality checks.
- 12-zone board map with fixed anchor points.

## Sprint 2 additions
- **Match readiness gate:** before the tick loop, the runtime optionally waits on fullscreen capture until a battlefield signal crosses `battlefield_score_threshold`, with logs `waiting_for_battlefield` / `battlefield_detected` / `battlefield_wait_timeout`.
- **Battlefield signal:** `battlefield_detector` can be **`heuristic`** (river/turf color bands on `game_viewport` × `anchor_rect`), **`model`** (tiny CNN on the **`bottom_panel`** crop from `battlefield_model_layout_path` with hand slots, next-card peek, and elixir bar masked to black; needs `pip install -r requirements-ml.txt`, `battlefield_model_path`, and layout YAML), or **`blend`** (average of heuristic score and model probability). For `model` / `blend`, tune `battlefield_score_threshold` in a typical probability band (for example `0.55`–`0.75`) after validation.
- **No actuation until ready:** while the gate has not passed (or after a timeout with `battlefield_timeout_behavior: idle`), policy decisions are forced to `NO_OP` with reason `match_readiness_not_ready`, so **no card hotkeys and no deploy clicks** are emitted.
- **Timeouts:** `battlefield_wait_timeout_ms` (use `0` for no wall-clock limit) plus `battlefield_timeout_behavior` `idle` (keep looping with actuation blocked) or `exit_nonzero` (process exits with code **2**).
- **Foreground (optional, Windows):** `foreground_check_enabled` compares the foreground window title (lowercased) against `foreground_title_substrings` before accepting a battlefield; non-Windows hosts log `foreground_check_skipped` once and do not enforce the check.
- **Config validation:** `match_readiness_enabled: true` requires `capture_enabled: true` (YAML load fails otherwise).

## Sprint 1 additions
- Fullscreen capture plumbing with optional debug frame dumps.
- Keyboard + mouse actuation layer with safe `dry-run` mode.
- Configurable delay between card hotkey and placement click (`actuation_select_to_click_delay_ms`).
- Configurable per-slot hotkeys (`actuation_card_hotkeys`, e.g. `q`/`w`/`e`/`r` to match in-game bindings).
- Structured `action_attempt` logs for each decision execution.
- `game_viewport` maps normalized board anchors into the real on-screen playfield (for letterboxed / centered portrait windows).
- `game_viewport.anchor_rect` maps anchors into a sub-rectangle of the client window (for example to exclude the bottom hand / elixir bar).

## Screen layout reference (Sprint 3+ perception)
- `configs/screen_layout_reference.yaml` stores **pixel rectangles** in fullscreen capture space for the bottom hand panel, four hand slots, next-card peek, and elixir bar (plus `tick_width_px` per elixir pip). Tuned for a centered ~608×1080 client on a 1920×1080-class capture; duplicate and edit if your geometry differs.
- `screen_layout.py` exposes `load_screen_layout_reference(Path)` and frozen `PixelRect` / `ScreenLayoutReference` for typed access in future perception code.

## Current modules
- `screen_layout.py` loads annotated HUD regions from YAML for future state extraction.
- `battlefield_detector.py` implements heuristic and/or CNN-based battlefield scoring for the readiness wait loop.
- `src/perception/battlefield_net.py`, `battlefield_roi.py`, and `battlefield_infer.py` define the classifier and masked bottom-panel tensor path (optional dependency: see `requirements-ml.txt`).
- `foreground_win.py` reads the Windows foreground window title for optional focus gating.
- `viewport.py` defines `GameViewport` / `AnchorRect` and parses `runtime.game_viewport` from YAML.
- `config.py` loads runtime settings from `configs/runtime.yaml`.
- `zones.py` stores 4x3 zone geometry and legality masks.
- `policy_gate.py` applies no-op/confidence/rate-limit rules.
- `capture.py` grabs fullscreen frames from the active display.
- `actuation.py` sends keyboard card selection (`pynput` first, `pyautogui` fallback) and mouse click placement (`pyautogui`).
- `candidate_policy.py` provides scripted candidate actions for reliability tests.
- `loop.py` optionally waits for match readiness, then runs capture -> policy -> gate -> actuation and logs decisions.
- `__main__.py` starts runtime via `python -m src.runtime`.
