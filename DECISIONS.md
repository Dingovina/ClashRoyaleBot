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
