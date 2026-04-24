#!/usr/bin/env python3
"""
Run the real-time runtime loop (capture → policy → actuation).

Equivalent to ``python -m src.runtime`` from the repository root, but works when invoked
from any working directory: the repo root is added to ``sys.path`` and set as cwd so
default ``configs/runtime.yaml`` resolves correctly.

Examples::

  python scripts/runtime/run_runtime.py
  python scripts/runtime/run_runtime.py --config configs/runtime.yaml
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_REPO_ROOT)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runtime.__main__ import main

if __name__ == "__main__":
    main()
