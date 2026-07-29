"""Re-export of tools/strand/tools/strand_eval (single authority copy)."""
from __future__ import annotations
import sys
from pathlib import Path
_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from strand_eval import *  # noqa: F401,F403
from strand_eval import HARNESS_VERSION, locate_repo_root  # noqa: F401
