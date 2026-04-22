# ClashRoyaleBot

Pet project exploring applied AI for **Clash Royale**.

The goal is to build and train a model that reads the same on-screen information a human player sees and makes decisions at roughly average-player skill, without extra hidden state.

The repository is early-stage. Over time it will grow architecture notes, training steps, tooling, and experiment results.

**Fair play:** the project is not meant to create an unfair advantage. The agent is intended to operate under the same constraints as a normal player.

---

## Where to look

| Topic | Location |
|--------|-----------|
| **Upcoming sprints and tasks** | `ROADMAP.md` |
| **Architecture decisions (ADRs)** | `DECISIONS.md` |
| **Runtime package (modules, shipped behavior)** | `src/runtime/README.md` |
| **Runtime configuration** | `configs/runtime.yaml` |
| **Docker usage** | `docs/docker-basics.md` |

---

## Product goals

### Long-term

- Win rate against real players **> 75%** (aspirational; not a v1 target).

### v1 (14-day demo)

- End-to-end real-time matches.
- Invalid action rate near zero.
- Basic non-random tactics.
- Evaluation on at least **20** matches.

---

## Operating constraints (v1)

- **Platform:** Google Play Games on **Windows 11**, fullscreen client (portrait playfield with side bands). Manual match start; no in-match pause; no minimized-window support.
- **Actions:** configurable keyboard hotkeys for four hand slots (see `configs/runtime.yaml`) plus mouse placement. No simulated human reaction delays in v1.
- **Hardware target:** laptop-class CPU (e.g. Intel i5-1155G7, 16 GB RAM), real-time inference, capture toward **60 FPS** (adaptive fallback allowed).
- **Priorities:** legal actions over raw strength; **500 ms** tick; global **1000 ms** action rate limit; `NO_OP` allowed; single fixed user deck for first playable build.

Default confidence and elixir rules are **accepted ADRs** (`DECISIONS.md`, e.g. DEC-0002–0004) and mirrored in `configs/runtime.yaml`.

---

## Scope

**In scope (near term):** rule baseline, imitation learning, coarse geometry and unit classes, Docker-backed train/runtime split.

**Out of scope (for now):** opponent hand-cycle tracking, full card/unit recognition on day one, universal all-decks policy, tournament-only “safe mode”.

---

## Architecture direction

Work is sequenced in `ROADMAP.md` by sprint. High-level tracks:

- **Perception:** frames → compact state + per-component confidence.
- **Control:** state → `(card slot, zone)` with legality and rate limits.
- **Learning:** rules first, then imitation; RL after demo milestone.
- **Evaluation:** win rate, damage delta, invalid actions, latency, stability.

---

## Implementation status

**Sprints 0–1 (environment + observe/act shell):** delivered as described in **`DECISIONS.md` (DEC-0005)** and in **`src/runtime/README.md`** (tick loop, policy gate, zones, capture, viewport/anchor mapping, actuation, tests, Dockerfiles).

**Next:** Sprint 2 in `ROADMAP.md`.
