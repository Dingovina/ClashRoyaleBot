# Clash Royale AI - Action Roadmap

This file is the living project plan and should be updated after each important decision.

## 1) Locked Context

### Runtime environment
- Platform: Google Play Games on Windows 11.
- Mode: fullscreen game window (vertical game area with side bands).
- Match start and restart: manual.
- In-match pause: not available.
- No need to support minimized or covered game window.

### Allowed agent actions
- Select card by keyboard keys `1`, `2`, `3`, `4`.
- Select deployment position by mouse click.
- No human-like delay simulation required in v1.

### Hardware and performance targets
- CPU: Intel i5-1155G7 (4C/8T), RAM: 16 GB.
- Real-time inference required.
- Capture target: 60 FPS (adaptive fallback allowed).
- Model size budget: up to 30 GB (can be increased later).

### Product priorities (v1)
- Correct legal actions are more important than tactical strength.
- Decision loop interval: 500 ms.
- Global action rate limit: at most one action per 1000 ms.
- Separate per-card cooldown is not required.
- "Do nothing" must be a valid action.
- First playable version uses one fixed user-selected deck.

---

## 2) Action Policy Defaults (v1)

These thresholds are defaults and can be tuned after first evaluation runs.

- `ACTION_CONFIDENCE_THRESHOLD = 0.70`
  - Execute a model/rule action only when confidence is at least 0.70 and legality checks pass.
- `NO_OP_CONFIDENCE_THRESHOLD = 0.55`
  - If best action confidence is below 0.55, choose `NO_OP`.
- `UNCERTAINTY_BAND = [0.55, 0.70)`
  - In this band, choose `NO_OP` unless an urgent defensive trigger is active.
- `MIN_ELIXIR_FOR_NON_URGENT_ACTION = 3`
  - If elixir estimate is below 3 and no urgent threat is detected, prefer `NO_OP`.

---

## 3) Board Geometry and Placement (v1)

- Board is split into 12 zones: 4 rows x 3 columns.
- Each zone has a fixed anchor point for placement clicks.
- Action output format:
  - `card_index` in `[1..4]` or `NO_OP`,
  - `zone_id` in `[0..11]` for deployable actions.

### Placement logic split
- Unit cards:
  - Use unit-specific legal zone masks.
  - Prefer safer anchors under uncertainty.
- Spell cards:
  - Use separate spell zone masks and targeting priorities.
  - Allow more aggressive zones for damage/value spells.

---

## 4) Primary Goal

- Build an AI agent that plays Clash Royale at average-player level or better, using only normal player-visible information.

### Long-term success target
- Win rate against real players: >75% (project target, not v1 target).

### v1 success target (14-day demo)
- End-to-end system runs full matches in real time.
- Invalid action rate is close to zero.
- Basic tactical behavior exists (not random).
- Evaluation on at least 20 matches.

---

## 5) Scope Boundaries

### In scope now
- Rule-based baseline.
- Imitation learning from available match data.
- Coarse battlefield geometry and approximate enemy unit recognition.
- Separate train/runtime environments with Docker support.

### Out of scope now (recorded for later)
- Opponent hand-cycle tracking.
- Full detailed unit/card recognition from day one.
- Universal all-decks policy from first demo.
- Safe mode and tournament-specific policies.

---

## 6) Architecture Tracks

## Track A - Perception (Vision -> State)
- Detect and track:
  - own hand cards (4 slots),
  - elixir estimate,
  - tower HP (friendly/enemy),
  - match timer,
  - enemy units (coarse classes first).
- Convert raw frame stream to compact state vector.
- Add confidence values for each detected component.

## Track B - Control (State -> Action)
- Produce action tuple:
  - `card_index` in `[1..4]` or `NO_OP`,
  - `zone_id` for coarse board region.
- Enforce legality gates:
  - selected card exists and is deployable,
  - zone is valid for the card type and current side,
  - global rate limit is respected.

## Track C - Learning
- Baseline rules to validate loop reliability.
- Imitation model trained on collected datasets.
- RL is postponed until post-demo milestone.

## Track D - Evaluation
- Core metrics:
  - win rate (20-match sets),
  - tower damage delta,
  - invalid action count,
  - action latency,
  - runtime stability.

---

## 7) 14-Day Sprint Plan

## Sprint 0 (Day 1-2) - Environment and Skeleton
- [x] Finalize repository structure.
- [x] Create train/runtime Docker environments.
- [x] Add a minimal runtime app skeleton (capture loop + action interface placeholders).
- [x] Implement fixed 500 ms tick contract.

### Exit criteria
- [ ] `docker compose` starts both environments.
- [x] Runtime skeleton can run without crashing.

## Sprint 1 (Day 3-5) - Observe + Act Reliability
- Implement fullscreen capture for Google Play Games.
- Implement keyboard card selection (`1-4`) and mouse placement click.
- Add strict legality and global 1000 ms action rate-limit guards.
- Add structured logging for every action attempt.

### Exit criteria
- Agent can execute scripted legal action sequences in live match context.
- Invalid action attempts are detected and logged.

## Sprint 2 (Day 6-9) - Coarse Perception + Rule Baseline
- Implement coarse state extraction:
  - hand presence,
  - timer,
  - elixir estimate,
  - tower HP proxy,
  - enemy unit coarse grouping.
- Implement first rule policy:
  - defense-first logic,
  - no-op under low confidence.

### Exit criteria
- Full match can run with rule policy end-to-end.
- Logs contain state snapshots and chosen actions.

## Sprint 3 (Day 10-12) - Imitation Pipeline v1
- Build dataset export from recorded sessions.
- Train first imitation model (single fixed deck regime).
- Connect inference model into runtime loop.

### Exit criteria
- Model inference works in real time on target laptop.
- Behavior is measurably better than random baseline.

## Sprint 4 (Day 13-14) - Evaluation and Demo Hardening
- Run at least 20 evaluation matches.
- Report:
  - win rate,
  - damage delta,
  - invalid actions,
  - latency and stability.
- Fix highest-impact runtime failures only.

### Exit criteria
- Demo package and report are ready.

---

## 8) Backlog (Post-Demo)

1. Opponent deck memory and anticipation.
2. Card/stat database with level scaling.
3. Deck recommendation module from user collection.
4. RL fine-tuning on top of imitation policy.
5. Generalization to multiple deck archetypes.

---

## 9) Open Decisions

These decisions are still required before implementation starts:
- Initial strategy for experiment tracking tool.
- Data access pipeline for esports match ingestion.

---

## 10) Plan Update Rules

- Keep exactly one "current sprint" marked as active.
- Update exit criteria status after every work session.
- Move deferred ideas to backlog without deleting them.
- Record architecture-level decisions in `DECISIONS.md`.
