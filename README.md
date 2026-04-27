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
| **Card registry (costs, aliases, classes)** | `configs/card_registry.yaml` |
| **Run runtime locally** | `scripts/runtime/run_runtime.py` (or `scripts/run_runtime.bat` / `run_runtime.ps1` on Windows) |
| **Battlefield CNN (match readiness, train, eval)** | `scripts/train/train_battlefield_classifier.py`, `scripts/eval/eval_battlefield_classifier.py`, `artifacts/battlefield_cnn.pt`, `data/processed/train/battlefield_train`, `data/processed/val/battlefield_val`, `requirements-ml.txt` |
| **Hand-card classifier (train, eval)** | `scripts/train/train_card_classifier.py`, `scripts/eval/eval_card_classifier.py`, `data/processed/train/cards_train`, `data/processed/val/cards_val`, `artifacts/card_cnn.pt` |
| **Elixir classifier (train, eval)** | `scripts/train/train_elixir_classifier.py`, `scripts/eval/eval_elixir_classifier.py`, `data/processed/train/elixir_train`, `data/processed/val/elixir_val`, `artifacts/elixir_cnn.pt` |
| **Dataset ROI crop utility** | `scripts/data/crop_training_images.py` |
| **HUD pixel layout (hand / elixir digit reference)** | `configs/screen_layout_reference.yaml`, `src/perception/screen_layout.py` |
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
2. Put raw labeled captures under `data/raw/battlefield_test/good/` (in-match) and `data/raw/battlefield_test/bad/` (not in-match), each as `*.png` (at least four images total; see `scripts/train/train_battlefield_classifier.py`).
3. Crop manually validated raw screenshots from `data/raw/<match_id>/` into split datasets under `data/processed/<split>/`:

   ```powershell
   python scripts/data/crop_training_images.py --match-id local-match --card --bf --elixir --train
   ```

   Runtime capture writes files as `CHECK_<elixir>_<card1>_<card2>_<card3>_<card4>_<random-id>.png`.
   Remove the `CHECK_` prefix only after manual validation. The crop script skips `CHECK_*` and deletes successfully
   processed source files.
4. Train from the **repository root** (so `src` resolves):

   ```powershell
   python scripts/train/train_battlefield_classifier.py `
     --train-data-dir data/processed/train/battlefield_train `
     --val-data-dir data/processed/val/battlefield_val `
     --out artifacts/battlefield_cnn.pt
   ```

   **cmd.exe** may use `^` at end-of-line for continuation. Optional flags: `--input-size`, `--epochs`, `--lr` (see `python scripts/train/train_battlefield_classifier.py --help`).
5. Tune `battlefield_score_threshold` in `configs/runtime.yaml` after checking probabilities, for example with `python scripts/eval/eval_battlefield_classifier.py --checkpoint artifacts/battlefield_cnn.pt --data-dir data/processed/val/battlefield_val`.
6. If your HUD geometry differs from the reference 1920×1080 layout, copy and edit `configs/screen_layout_reference.yaml`, then point `battlefield_model_layout_path` at your file for both training and runtime.

## Migration notes (legacy cleanup)

- Runtime config now accepts only top-level keys: `runtime`, `board`.
- Legacy `card_types` and `configs/card_elixir_costs.yaml` fallback are removed; use `configs/card_registry.yaml`.
- `board.rows`/`board.cols` are removed; keep only `board.zones`.
- ROI training helpers now expect fullscreen screenshots for crop functions.

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

**Sprint 2 (match readiness + match end):** delivered as described in **`DECISIONS.md` (DEC-0007, DEC-0008)** and **`src/runtime/README.md`** (pre-loop wait on live capture, tiny CNN on the masked bottom panel, **automatic loop exit** when the CNN stays below `battlefield_end_score_threshold` for several probes, `match_safety_max_ticks` failsafe, structured logs, optional Windows foreground title check).

**Next:** current sprint in `ROADMAP.md` (Sprint 3 — coarse perception + rule baseline).
