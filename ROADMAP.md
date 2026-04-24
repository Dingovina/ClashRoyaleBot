# Roadmap

This file lists **upcoming work only**. Product constraints, accepted defaults, and shipped milestones are documented in `README.md`, `DECISIONS.md`, and `src/runtime/README.md`.

## Current sprint

**Sprint 3 — Elixir digit perception + pre-play elixir gate**

## Sprint 3 (Day 8–9) — Elixir digit perception + pre-play elixir gate

- [ ] Elixir digit perception from `elixir_number` region:
  - [ ] build labeled `data/processed/elixir_test` subset and split for validation,
  - [ ] implement robust digit inference for values `0..10`,
  - [ ] emit per-tick `elixir_estimate` and confidence into logs.
- [ ] Runtime pre-play elixir gate:
  - [ ] map current hand card to configured elixir cost,
  - [ ] before actuation, block card play when `elixir_estimate < card_cost`,
  - [ ] force `NO_OP` with explicit reason (for example `insufficient_elixir`).
- [ ] Verification:
  - [ ] unit tests for gate logic (`enough` vs `insufficient` elixir),
  - [ ] integration check that no hotkey/click is sent when elixir is insufficient.

### Sprint 3 exit criteria

- [ ] Runtime reliably reads elixir digit on the reference layout.
- [ ] Card play is never attempted when estimated elixir is below card cost.
- [ ] Logs clearly show elixir estimate, card cost, and gate reason.

## Sprint 4 (Day 10–13) — Coarse perception + rule baseline

- [ ] Coarse state extraction from frames:
  - [ ] hand presence (four slots),
  - [ ] match timer,
  - [ ] elixir estimate,
  - [ ] tower HP proxy,
  - [ ] enemy unit coarse grouping.
- [ ] First rule policy:
  - [ ] defense-first logic,
  - [ ] no-op under low confidence.

### Sprint 4 exit criteria

- [ ] Full match can run with rule policy end-to-end.
- [ ] Logs contain state snapshots and chosen actions.

## Sprint 5 (Day 14–16) — Imitation pipeline v1

- [ ] Dataset export from recorded sessions.
- [ ] Train first imitation model (single fixed deck regime).
- [ ] Connect inference model into runtime loop.

### Sprint 5 exit criteria

- [ ] Model inference runs in real time on target laptop.
- [ ] Behavior is measurably better than random baseline.

## Sprint 6 (Day 17–18) — Evaluation and demo hardening

- [ ] Run at least 20 evaluation matches.
- [ ] Report win rate, damage delta, invalid actions, latency, stability.
- [ ] Fix only highest-impact runtime failures.

### Sprint 6 exit criteria

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
