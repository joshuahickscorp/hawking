import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent
_REENTRY_ENV = "HAWKING_PYTEST_VALIDATION_ACTIVE"
_VALIDATED = False


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail before collection unless every pinned validation source is exact."""
    del session
    global _VALIDATED
    if _VALIDATED:
        return
    if os.environ.get(_REENTRY_ENV) == "1":
        raise pytest.UsageError("recursive Hawking validation-pack check refused")
    environment = os.environ.copy()
    environment[_REENTRY_ENV] = "1"
    command = [sys.executable, str(ROOT / "tools/hawking_packs.py"), "validation"]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise pytest.UsageError(f"cannot validate the Hawking test pack: {exc}") from exc
    if result.returncode:
        details = "\n".join(
            value for value in (result.stderr.strip(), result.stdout.strip()) if value
        )
        raise pytest.UsageError(
            f"Hawking validation-pack check failed with exit {result.returncode}"
            + (f":\n{details}" if details else "")
        )
    _VALIDATED = True
