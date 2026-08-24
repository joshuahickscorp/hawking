from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class Workspace:
    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.git_root: Optional[str] = None
        self._detect_git()

    def _detect_git(self):
        try:
            import subprocess
            proc = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.root, capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                self.git_root = os.path.realpath(proc.stdout.strip())
        except Exception:
            pass

    def resolve(self, ref: str) -> Optional[str]:
        full = os.path.join(self.root, ref)
        if os.path.isfile(full):
            return ref
        return None
