#!/usr/bin/env python3
"""G029 runner: one runtime, one experiment engine, adversary as a stage.

Writes receipts/headless/RUNTIME_EXPERIMENT_ADVERSARY.json and
receipts/headless/RUNTIME_GENOME.json. Does not re-measure MLX. Does not
open the deleted Q5_K GGUF.
"""
from __future__ import annotations

from experiment_engine import main

if __name__ == "__main__":
    raise SystemExit(main())
