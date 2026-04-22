# Runtime Module

This package contains real-time match execution code.

## Sprint 0 scope
- Config-driven tick loop (`500 ms` by default).
- Global action rate limit (`1000 ms` by default).
- Policy gate with confidence and legality checks.
- 12-zone board map with fixed anchor points.

## Current modules
- `config.py` loads runtime settings from `configs/runtime.yaml`.
- `zones.py` stores 4x3 zone geometry and legality masks.
- `policy_gate.py` applies no-op/confidence/rate-limit rules.
- `loop.py` runs the tick loop and logs decisions.
- `__main__.py` starts runtime via `python -m src.runtime`.
