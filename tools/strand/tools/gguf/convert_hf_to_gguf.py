#!/usr/bin/env python3
"""Upstream llama.cpp HF→GGUF converter (relocated to vendor/gguf/)."""
from __future__ import annotations
import runpy
from pathlib import Path
_TARGET = Path(__file__).resolve().parents[4] / "vendor" / "gguf" / "convert_hf_to_gguf.py"
if not _TARGET.is_file():
    raise SystemExit(f"vendored converter missing: {_TARGET}")
runpy.run_path(str(_TARGET), run_name="__main__")
