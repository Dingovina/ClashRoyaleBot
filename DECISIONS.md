# Architecture Decisions Log

Use this file to record high-impact technical decisions.

## Status Legend
- Proposed
- Accepted
- Deprecated

## Decision Template

### DEC-XXXX: Title
- Status: Proposed
- Date: YYYY-MM-DD
- Context:
  - What problem is being solved?
  - What constraints matter?
- Decision:
  - Chosen option and short rationale.
- Consequences:
  - Benefits.
  - Trade-offs and risks.
- Alternatives considered:
  - Option A
  - Option B

---

### DEC-0001: Use Python + PyTorch as default ML stack
- Status: Accepted
- Date: 2026-04-22
- Context:
  - Project needs fast iteration for perception and imitation learning.
  - Target hardware is a laptop-class Windows machine.
- Decision:
  - Use Python and PyTorch as the initial default stack.
- Consequences:
  - Good ecosystem and community support.
  - Need to validate runtime performance on target hardware.
- Alternatives considered:
  - TensorFlow
  - JAX

---

### DEC-0002: v1 decision timing and action throttling
- Status: Accepted
- Date: 2026-04-22
- Context:
  - Runtime must be stable on laptop hardware.
  - User requires real-time behavior with safety-first action validity.
- Decision:
  - Use a fixed 500 ms decision loop and a global 1000 ms action rate limit.
  - Do not add a separate per-card cooldown in v1.
- Consequences:
  - Simpler and safer control loop with lower invalid action risk.
  - May reduce tactical responsiveness in fast situations.
- Alternatives considered:
  - 100-250 ms decision loop
  - Per-card cooldown policies

---

### DEC-0003: v1 board zoning and placement strategy
- Status: Accepted
- Date: 2026-04-22
- Context:
  - v1 should avoid fine-grained placement complexity.
  - User approved coarse geometry and fixed points.
- Decision:
  - Split board into 12 zones (4 rows x 3 columns) with fixed anchor points.
  - Use separate legality/priority logic for spell cards and unit cards.
- Consequences:
  - Easier training target space and more deterministic actuation.
  - Lower tactical precision than continuous coordinate placement.
- Alternatives considered:
  - 6 or 8 zones
  - Continuous coordinate output from policy

---

### DEC-0004: v1 NO_OP and action confidence thresholds
- Status: Accepted
- Date: 2026-04-22
- Context:
  - User requested explicit default thresholds selected by the assistant.
  - v1 prioritizes legal and stable behavior.
- Decision:
  - Execute non-urgent action only if confidence is at least 0.70.
  - Choose `NO_OP` if confidence is below 0.55.
  - In the 0.55-0.70 band, choose `NO_OP` unless urgent defense trigger is active.
  - Prefer `NO_OP` when elixir is below 3 and no urgent threat exists.
- Consequences:
  - Reduces reckless low-confidence actions.
  - Can miss borderline good opportunities until thresholds are tuned.
- Alternatives considered:
  - Single threshold only
  - Aggressive low-threshold action policy

---

### DEC-0005: Milestone — Sprint 0 and Sprint 1 runtime shell
- Status: Accepted
- Date: 2026-04-23
- Context:
  - Need a reproducible train/runtime split and a safe real-time loop before perception and learning land.
- Decision:
  - Treat Sprint 0–1 as **closed** from a planning-doc perspective: repository layout, Docker train/runtime images and compose, YAML-driven `RuntimeConfig`, fixed **500 ms** tick loop with mock/scripted candidate policy, **12-zone** anchors with unit/spell legality masks, **policy gate** (confidence bands, elixir guard, global rate limit), **mss** fullscreen capture with optional frame dumps, **game viewport** (centered strip + `anchor_rect` for hand bar), **actuation** (slot hotkeys via **pynput** with **pyautogui** fallback, click placement, configurable delay after hotkey), structured **`action_attempt`** logging, and **unit tests** for gate and actuation mapping.
- Consequences:
  - `ROADMAP.md` holds **forward plans only**; this entry is the milestone record for shipped Sprint 0–1 scope.
  - Operators must match in-game hotkeys and viewport numbers in `configs/runtime.yaml` (including English keyboard layout where required).
- Alternatives considered:
  - Keep sprint checklists inside `ROADMAP.md` indefinitely (rejected: splits “plan” vs “history”).

---

### DEC-0007: Sprint 2 — match readiness gate (battlefield CNN)
- Status: Accepted (amended 2026-04-23)
- Date: 2026-04-22
- Context:
  - The runtime must not send card hotkeys or placement clicks while the operator is still on the desktop, launcher, or main menu.
  - A small binary classifier on a fixed HUD crop is acceptable for this gate; tuning stays YAML-driven with clear operator logs.
- Decision:
  - Add a **pre-loop wait** on live fullscreen capture when `match_readiness_enabled` is true (requires `capture_enabled`).
  - Use a **tiny CNN** on the **masked `bottom_panel`** from `battlefield_model_layout_path` (hand slots, next-card peek, and elixir count digit region zeroed), with weights from `battlefield_model_path` (default `artifacts/battlefield_cnn.pt`). Probability is compared to `battlefield_score_threshold` (see `src/runtime/battlefield_evaluate.py` and `src/perception/battlefield_infer.py`). **Heuristic and blend detectors are removed**; PyTorch is required when the gate is enabled.
  - Emit structured logs: `waiting_for_battlefield` (includes `reason=` for capture/foreground cases), `battlefield_detected`, `battlefield_wait_timeout`, and `battlefield_timeout_continue` when falling back to idle actuation lockout.
  - If the wait exceeds `battlefield_wait_timeout_ms` (`0` disables the deadline), honor `battlefield_timeout_behavior`: **`idle`** keeps the main loop but forces `NO_OP` actuation with `match_readiness_not_ready`; **`exit_nonzero`** terminates the process with **exit code 2**.
  - Optional **`foreground_check_enabled`** on **Windows** compares the foreground window title to `foreground_title_substrings`; other platforms log a one-time skip and do not block on title.
  - If the default weights file is missing, **config load** fails with an error that includes **how to train** the classifier (root `README.md` documents the full flow).
- Consequences:
  - Operators must install **`requirements-ml.txt`**, ship a trained `.pt`, and tune `battlefield_score_threshold` using `scripts/eval_battlefield_classifier.py` or live logs.
  - Full HD BGRA copies (~8 MB) occur during the wait loop while frames are classified.
- Alternatives considered:
  - **Heuristic river/turf bands** on the playfield ROI (removed in favor of the CNN for more reliable deck/menu discrimination).
  - **Template matching** on a reference patch (rejected to avoid brittle binary assets and extra OpenCV coupling).

---

### DEC-0008: Main loop exit when battlefield CNN drops (match end)
- Status: Accepted
- Date: 2026-04-23
- Context:
  - The runtime previously stopped after a fixed ``max_ticks`` tick budget, which does not track real match length.
  - The same battlefield classifier used for match readiness can detect return to menus / post-battle UI when the in-match signal disappears.
- Decision:
  - Replace fixed ``max_ticks`` as the primary stop condition with **CNN-based match end**: periodically (``match_end_check_every_n_ticks``) capture BGRA and run the same model; when probability stays **below** ``battlefield_end_score_threshold`` for ``match_end_confirm_ticks`` consecutive probes, exit the main loop with log ``runtime_finished reason=battlefield_absent``.
  - Require **hysteresis**: ``battlefield_end_score_threshold`` must be **less than** ``battlefield_score_threshold`` (start gate vs end gate).
  - Keep ``match_safety_max_ticks`` as a **hard ceiling** on loop iterations (``0`` = no cap when CNN match-end is enabled with ``match_end_confirm_ticks > 0``).
  - If ``match_end_confirm_ticks`` is ``0``, CNN match-end is disabled and ``match_safety_max_ticks`` must be ``> 0`` so the process cannot run forever without another stop condition.
- Consequences:
  - Extra fullscreen captures with pixels on a subset of ticks during the match; operators tune sampling and thresholds for latency vs reliability.
- Alternatives considered:
  - **Fixed max_ticks only** (rejected as primary control per product request).

---

### DEC-0006: Documentation split — roadmap vs README vs decisions
- Status: Accepted
- Date: 2026-04-23
- Context:
  - `ROADMAP.md` mixed locked context, architecture narrative, and completed sprint checklists, which obscured upcoming work.
- Decision:
  - **`ROADMAP.md`:** upcoming tasks and sprint exit criteria **only** (unchecked items).
  - **`README.md`:** goals, constraints, scope summary, architecture direction, and an **implementation status** pointer to ADRs and `src/runtime/README.md`.
  - **`DECISIONS.md`:** ADRs and milestone records (e.g. DEC-0005).
- Consequences:
  - Less duplication; clearer onboarding for contributors.
  - Requires discipline: on ship, update `DECISIONS.md` and README status, not the roadmap checklist.
- Alternatives considered:
  - Single living `ROADMAP.md` with `[x]` sections (rejected per documentation split).
