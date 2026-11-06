#!/usr/bin/env python3
"""Run proposed/main; see experiments/README.md for its design and protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments._shared.entrypoints import launch


if __name__ == "__main__":
    launch('proposed/main')
