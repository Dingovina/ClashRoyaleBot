# Roadmap

This file lists **upcoming work only**. Product constraints, accepted defaults, and shipped milestones are documented in `README.md`, `DECISIONS.md`, and `src/runtime/README.md`.

## Current sprint

**Sprint 2 — Match readiness gate**

## Sprint 2 (Day 6–7) — Match readiness gate

- [ ] On startup, runtime stays in an **idle / waiting** state until a **battlefield-present** signal is inferred from captured frames (no card hotkeys, no deploy clicks).
- [ ] Implement a **battlefield detector** (minimal viable approach: template match, lightweight heuristic, or small classifier — pick one and document trade-offs) with configurable thresholds and clear logs (`waiting_for_battlefield`, `battlefield_detected`, etc.).
- [ ] Add **timeouts / safe fallback** if the battlefield never appears (e.g. remain idle or exit with a clear status code).
- [ ] *(Optional)* **Foreground / game-window check:** before leaving idle, verify the expected **Clash Royale** (or **Google Play Games** host) window is focused; if not, log and stay idle.

### Sprint 2 exit criteria

- [ ] Cold start on main menu, desktop, or wrong focused window produces **zero** actuation events until a valid battlefield is visible (and optional focus check passes).
- [ ] One short note in repo (e.g. `DECISIONS.md` or `src/runtime/README.md`) documents baseline false negative / false positive behavior for the chosen detector.

## Sprint 3 (Day 8–11) — Coarse perception + rule baseline

- [ ] Coarse state extraction from frames:
  - [ ] hand presence (four slots),
  - [ ] match timer,
  - [ ] elixir estimate,
  - [ ] tower HP proxy,
  - [ ] enemy unit coarse grouping.
- [ ] First rule policy:
  - [ ] defense-first logic,
  - [ ] no-op under low confidence.

### Sprint 3 exit criteria

- [ ] Full match can run with rule policy end-to-end.
- [ ] Logs contain state snapshots and chosen actions.

## Sprint 4 (Day 12–14) — Imitation pipeline v1

- [ ] Dataset export from recorded sessions.
- [ ] Train first imitation model (single fixed deck regime).
- [ ] Connect inference model into runtime loop.

### Sprint 4 exit criteria

- [ ] Model inference runs in real time on target laptop.
- [ ] Behavior is measurably better than random baseline.

## Sprint 5 (Day 15–16) — Evaluation and demo hardening

- [ ] Run at least 20 evaluation matches.
- [ ] Report win rate, damage delta, invalid actions, latency, stability.
- [ ] Fix only highest-impact runtime failures.

### Sprint 5 exit criteria

- [ ] Demo package and report are ready.

## Backlog (post-demo)

1. Opponent deck memory and anticipation.
2. Card/stat database with level scaling.
3. Deck recommendation module from user collection.
4. RL fine-tuning on top of imitation policy.
5. Generalization to multiple deck archetypes.

## Open decisions

- [ ] Initial strategy for experiment tracking tool.
- [ ] Data access pipeline for esports match ingestion.

## How to update this roadmap

- Add or edit **future** tasks and unchecked exit criteria here.
- When work ships, add or update an entry in `DECISIONS.md` and refresh the **Implementation status** section in `README.md`.
- Do not use this file as a changelog; keep history in git and ADRs.
