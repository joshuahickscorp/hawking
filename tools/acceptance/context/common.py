"""Receipt helpers for the HCLI context-family acceptance lane."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[3]
RECEIPT_DIR = REPO / "receipts" / "acceptance"
ROADMAP = Path("/Users/scammermike/Downloads/H-ROADMAP.md")
SCHEMA = "hawking.acceptance.gate.v1"


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def quote_roadmap(start: int, end: int) -> str:
    if not ROADMAP.is_file():
        return f"(roadmap missing at {ROADMAP}; span {start}-{end})"
    lines = ROADMAP.read_text(encoding="utf-8").splitlines()
    chunk = lines[start - 1 : end]
    return "\n".join(chunk)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "__dict__") and not callable(value):
        try:
            return jsonable(vars(value))
        except TypeError:
            return repr(value)
    return repr(value)


def write_receipt(gate: str, payload: Mapping[str, Any]) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = RECEIPT_DIR / f"{gate}.json"
    body = jsonable(dict(payload))
    body.setdefault("schema", SCHEMA)
    body.setdefault("gate", gate)
    body.setdefault("generated_at", now_utc())
    body.setdefault("git_head", git_head())
    body.setdefault("criterion_altered", False)
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def rewrite(path: Path, text: str) -> None:
    """Write `text` and guarantee mtime_ns moves so invalidation can see it."""
    old = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    if old is not None and st.st_mtime_ns == old:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
