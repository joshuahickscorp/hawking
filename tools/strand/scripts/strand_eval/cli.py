"""CLI re-export — implementation lives in tools/strand/tools/strand_eval."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path
_TARGET = Path(__file__).resolve().parents[2] / "tools" / "strand_eval" / "cli.py"
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
