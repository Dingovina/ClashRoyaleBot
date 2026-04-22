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
| **Run runtime locally** | `scripts/run_runtime.py` (or `scripts/run_runtime.bat` / `run_runtime.ps1` on Windows) |
| **Battlefield CNN (match readiness, train, eval)** | `scripts/train_battlefield_classifier.py`, `scripts/eval_battlefield_classifier.py`, `artifacts/battlefield_cnn.pt`, `configs/screen_layout_reference.yaml`, `requirements-ml.txt` |
| **HUD pixel layout (hand / elixir reference)** | `configs/screen_layout_reference.yaml`, `src/perception/screen_layout.py` |
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

## Training the battlefield classifier (match readiness)

When `match_readiness_enabled` is true, the runtime loads **`artifacts/battlefield_cnn.pt`** by default (override with `battlefield_model_path` in `configs/runtime.yaml`). If that file is missing, configuration loading fails with instructions to train the model.

1. Install ML dependencies from the repository root: `pip install -r requirements-ml.txt`.
2. Add labeled fullscreen captures under a directory such as `data/battlefield_test/`: filenames **`true_*.png`** for in-match / arena views and **`false_*.png`** for menus, loading, deck, etc. (at least four samples; see `scripts/train_battlefield_classifier.py` for naming rules).
3. Train from the **repository root** (so `src` resolves):

   ```powershell
   python scripts/train_battlefield_classifier.py `
     --data-dir data/battlefield_test `
     --layout-yaml configs/screen_layout_reference.yaml `
     --out artifacts/battlefield_cnn.pt
   ```

   **cmd.exe** may use `^` at end-of-line for continuation. Optional flags: `--input-size`, `--epochs`, `--lr` (see `python scripts/train_battlefield_classifier.py --help`).
4. Tune `battlefield_score_threshold` in `configs/runtime.yaml` after checking probabilities, for example with `python scripts/eval_battlefield_classifier.py --checkpoint artifacts/battlefield_cnn.pt --data-dir data/battlefield_test`.
5. If your HUD geometry differs from the reference 1920×1080 layout, copy and edit `configs/screen_layout_reference.yaml`, then point `battlefield_model_layout_path` at your file for both training and runtime.

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

**Sprint 2 (match readiness gate):** delivered as described in **`DECISIONS.md` (DEC-0007)** and **`src/runtime/README.md`** (pre-loop wait on live capture, tiny CNN on the masked bottom panel, structured wait/detect logs, timeout behaviors, optional Windows foreground title check).

**Next:** current sprint in `ROADMAP.md` (Sprint 3 — coarse perception + rule baseline).
