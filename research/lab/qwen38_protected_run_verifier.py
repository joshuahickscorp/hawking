"""Internal compatibility view of Hawking's protected-run verifier."""
from __future__ import annotations

import sys

from hawking import qwen38_protected_run_verifier as _canonical

# Preserve sealed research imports while running precisely one verifier body.
sys.modules[__name__] = _canonical
