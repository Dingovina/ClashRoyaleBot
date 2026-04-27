# Runtime Module

This package contains real-time match execution code.

## Sprint 0 scope

- Config-driven tick loop (`500 ms` by default).
- Global action rate limit (`1000 ms` by default).
- Policy gate with confidence and legality checks.
- 12-zone board map with fixed anchor points.

## Sprint 2 additions

- **Match readiness gate:** before the tick loop, the runtime optionally waits on fullscreen capture until the **battlefield CNN** probability crosses `battlefield_score_threshold`, with logs `waiting_for_battlefield` / `battlefield_detected` / `battlefield_wait_timeout`.
- **Match end (main loop):** after readiness, ticks run until the same CNN reports probability **below** `battlefield_end_score_threshold` for `match_end_confirm_ticks` **consecutive** probes (probes run every `match_end_check_every_n_ticks` ticks when pixels are captured). Then the loop stops with `runtime_finished reason=battlefield_absent`. `battlefield_end_score_threshold` must be **lower** than `battlefield_score_threshold` (hysteresis). **`match_safety_max_ticks`** caps total ticks as a failsafe (`0` = no cap when CNN match-end is active).
- **CNN-only:** there is no heuristic or blend mode. Weights come from `battlefield_model_path` (default `artifacts/battlefield_cnn.pt`). Input tensor size is read from the checkpoint. Layout rectangles come from `battlefield_model_layout_path` (default `configs/screen_layout_reference.yaml`). PyTorch is required when `match_readiness_enabled` is true (`pip install -r requirements-ml.txt`).
- **No actuation until ready:** while the gate has not passed (or after a timeout with `battlefield_timeout_behavior: idle`), policy decisions are forced to `NO_OP` with reason `match_readiness_not_ready`, so **no card hotkeys and no deploy clicks** are emitted.
- **Timeouts:** `battlefield_wait_timeout_ms` (use `0` for no wall-clock limit) plus `battlefield_timeout_behavior` `idle` (keep looping with actuation blocked) or `exit_nonzero` (process exits with code **2**).
- **Foreground (optional, Windows):** `foreground_check_enabled` compares the foreground window title (lowercased) against `foreground_title_substrings` before accepting a frame; non-Windows hosts log `foreground_check_skipped` once and do not enforce the check.
- **Config validation:** runtime requires existing CNN checkpoints/layout files for battlefield + elixir + cards, plus PyTorch. If a default checkpoint path is missing, the error text includes **how to train** the model (see the root `README.md`).

## Sprint 1 additions

- Fullscreen capture plumbing with optional debug frame dumps.
- When debug capture is enabled, PNGs are written under `data/raw/<match_id>/` with `CHECK_...` filenames for manual label verification.
- Keyboard + mouse actuation layer with safe `dry-run` mode.
- Configurable delay between card hotkey and placement click (`actuation_select_to_click_delay_ms`).
- Configurable per-slot hotkeys (`actuation_card_hotkeys`, e.g. `q`/`w`/`e`/`r` to match in-game bindings).
- Structured `action_attempt` logs for each decision execution.
- `game_viewport` maps normalized board anchors into the real on-screen playfield (for letterboxed / centered portrait windows).
- `game_viewport.anchor_rect` maps anchors into a sub-rectangle of the client window (for example to exclude the bottom hand / elixir bar).

## Screen layout reference (perception)

- `configs/screen_layout_reference.yaml` stores **pixel rectangles** in fullscreen capture space for the bottom hand panel, four hand slots, next-card peek, and the **elixir count digit** (not the full bar). Tuned for a centered ~608×1080 client on a 1920×1080-class capture; duplicate and edit if your geometry differs.
- `src/perception/screen_layout.py` exposes `load_screen_layout_reference(Path)` and frozen `PixelRect` / `ScreenLayoutReference`.

## Current modules

- `runtime_config.py` — frozen `RuntimeConfig` dataclass.
- `config_loader.py` — loads `configs/runtime.yaml` via `load_runtime_config(Path)`.
- `match_readiness.py` — pre-loop wait until the battlefield CNN accepts the capture.
- `match_exit.py` — streak counter for CNN-based match end.
- `battlefield_evaluate.py` / `battlefield_config.py` — CNN scoring for readiness and raw probability for match end.
- `src/perception/battlefield_net.py`, `battlefield_roi.py`, `battlefield_infer.py` — classifier and masked bottom-panel tensor path.
- `foreground_win.py` reads the Windows foreground window title for optional focus gating.
- `viewport.py` defines `GameViewport` / `AnchorRect`, `crop_playfield_bgra`, and parses `runtime.game_viewport` from YAML.
- `zones.py` stores 4x3 zone geometry and legality masks.
- `policy_gate.py` applies no-op/confidence/rate-limit rules.
- `capture.py` grabs fullscreen frames from the active display; optional PNG dumps under `capture_debug_dir` when `capture_debug_save_enabled` is true.
- `keyboard_input.py` sends slot hotkeys (`pynput` first, `pyautogui` fallback).
- `actuation.py` orchestrates hotkey + placement click (`pyautogui`).
- `candidate_policy.py` provides scripted candidate actions for reliability tests.
- `runtime_service.py` runs runtime lifecycle; `tick_orchestrator.py` executes capture → perception → policy/gate → actuation and logs decisions.
- `__main__.py` starts runtime via `python -m src.runtime` or `python scripts/runtime/run_runtime.py` from the repo (see `scripts/run_runtime.*`).
