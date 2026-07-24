#!/usr/bin/env python
"""Thin wrapper so the CLI also runs as a plain script.

Prefer ``python -m sts.run ...`` from the repo root. This wrapper adds the repo
root to sys.path so ``python scripts/run_text_sts.py ...`` works too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sts.run import main

if __name__ == "__main__":
    main()
